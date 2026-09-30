"""Crash-safe append-only manifest: know what is already done, and trust that answer.

The manifest is a JSON Lines file. One record is appended -- and flushed to disk
-- after every finished item, so a run that dies at item 340 of 400 can restart
and skip straight to item 341.

Three properties make that trustworthy:

1. **Append-only.** Nothing is rewritten in place, so there is no window in
   which the whole file is half-new and half-old.
2. **One record per write.** A record is written with a single ``os.write`` on a
   file opened ``O_APPEND``, then ``fsync``-ed. A crash can therefore only ever
   damage the *last* line, never a line in the middle.
3. **Recovery on load.** A torn trailing line is dropped (or salvaged, if it
   happens to be complete JSON), and unforgivingly damaged regions are moved to
   a ``.corrupt-<timestamp>`` backup instead of being silently discarded.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "ItemStatus",
    "ManifestError",
    "ManifestRecord",
    "RecoveryReport",
    "ResumeManifest",
    "coerce_status",
    "latest_by_item",
    "read_records",
    "scan_file",
    "status_value",
]

#: On-disk manifest version, written in the header line.
MANIFEST_VERSION = 1


class ManifestError(RuntimeError):
    """A manifest write could not be completed durably."""


class ItemStatus(str, Enum):
    """Outcome of one item.

    ``OK`` is the only status that counts as done. ``CORRUPTED`` and
    ``PERMANENT_FAILURE`` are the exception list: the item produced output that
    must not be trusted, so nothing was saved for it.
    """

    OK = "ok"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_FAILURE = "permanent_failure"
    CORRUPTED = "corrupted"
    SKIPPED = "skipped"

    @property
    def is_failure(self) -> bool:
        """True for anything that is not ``OK`` or ``SKIPPED``."""
        return self not in (ItemStatus.OK, ItemStatus.SKIPPED)

    @property
    def is_exception(self) -> bool:
        """True when the item needs a human: corruption or a permanent failure."""
        return self in (ItemStatus.CORRUPTED, ItemStatus.PERMANENT_FAILURE)


#: Statuses that satisfy :meth:`ResumeManifest.is_done`.
DEFAULT_DONE_STATUSES: frozenset[str] = frozenset({ItemStatus.OK.value})

_RECORD_KEYS = frozenset(
    {
        "kind",
        "v",
        "seq",
        "id",
        "status",
        "ts",
        "epoch",
        "checksum",
        "size",
        "path",
        "attempts",
        "reason",
        "url",
        "meta",
    }
)


@dataclass(frozen=True)
class ManifestRecord:
    """One item's outcome.

    ``checksum`` and ``size`` describe the artefact that was actually written,
    which is what makes a later "is this file still the one we verified?"
    question answerable.
    """

    item_id: str
    status: str
    timestamp: str
    epoch: float
    checksum: Optional[str] = None
    size: Optional[int] = None
    path: Optional[str] = None
    attempts: int = 0
    reason: Optional[str] = None
    url: Optional[str] = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    seq: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        item_id: Union[str, int],
        status: Union[str, ItemStatus],
        *,
        checksum: Optional[str] = None,
        size: Optional[int] = None,
        path: Optional[Union[str, os.PathLike[str]]] = None,
        attempts: int = 0,
        reason: Optional[str] = None,
        url: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
        seq: int = 0,
        clock: Optional[Any] = None,
    ) -> "ManifestRecord":
        """Build a record stamped with the current UTC time."""
        now = time.time() if clock is None else float(clock())
        return cls(
            item_id=str(item_id),
            status=coerce_status(status),
            timestamp=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            epoch=now,
            checksum=checksum,
            size=None if size is None else int(size),
            path=None if path is None else str(path),
            attempts=int(attempts),
            reason=reason,
            url=url,
            meta=dict(meta or {}),
            seq=int(seq),
        )

    def to_mapping(self) -> dict[str, Any]:
        """JSON-serialisable payload, including any unknown keys we preserved."""
        payload: dict[str, Any] = dict(self.extra)
        payload.update(
            {
                "kind": "item",
                "v": MANIFEST_VERSION,
                "seq": self.seq,
                "id": self.item_id,
                "status": self.status,
                "ts": self.timestamp,
                "epoch": self.epoch,
                "checksum": self.checksum,
                "size": self.size,
                "path": self.path,
                "attempts": self.attempts,
                "reason": self.reason,
                "url": self.url,
                "meta": dict(self.meta),
            }
        )
        return payload

    def to_json_line(self) -> str:
        """Serialise to exactly one line, newline included."""
        return json.dumps(self.to_mapping(), ensure_ascii=False, sort_keys=True) + "\n"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ManifestRecord":
        """Parse a mapping, tolerating missing fields and preserving unknown ones.

        Unknown keys are kept in :attr:`extra` and re-emitted on write, so a
        manifest written by a newer version still round-trips.
        """
        item_id = data.get("id")
        status = data.get("status")
        if item_id is None or status is None:
            raise ValueError("manifest record needs both 'id' and 'status'")
        epoch = data.get("epoch")
        timestamp = data.get("ts")
        if epoch is None:
            epoch = time.time()
        if timestamp is None:
            timestamp = datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat(
                timespec="milliseconds"
            )
        return cls(
            item_id=str(item_id),
            status=str(status),
            timestamp=str(timestamp),
            epoch=float(epoch),
            checksum=data.get("checksum"),
            size=None if data.get("size") is None else int(data["size"]),
            path=data.get("path"),
            attempts=int(data.get("attempts") or 0),
            reason=data.get("reason"),
            url=data.get("url"),
            meta=dict(data.get("meta") or {}),
            seq=int(data.get("seq") or 0),
            extra={k: v for k, v in data.items() if k not in _RECORD_KEYS},
        )

    @property
    def status_enum(self) -> Optional[ItemStatus]:
        """The status as an :class:`ItemStatus`, or ``None`` if unrecognised."""
        try:
            return ItemStatus(self.status)
        except ValueError:
            return None

    def __repr__(self) -> str:
        return (
            f"<ManifestRecord {self.item_id!r} {self.status} attempts={self.attempts}"
            f"{' corrupted' if self.status == ItemStatus.CORRUPTED.value else ''}>"
        )


@dataclass(frozen=True)
class RecoveryReport:
    """What happened while reading the manifest back.

    ``clean`` is the question the caller usually asks; everything else exists so
    a damaged manifest can be explained instead of merely survived.
    """

    lines_read: int = 0
    records_loaded: int = 0
    blank_lines: int = 0
    skipped_lines: int = 0
    dropped_tail_bytes: int = 0
    salvaged_tail: bool = False
    interior_corruption: bool = False
    interior_corruption_line: Optional[int] = None
    interior_corruption_offset: Optional[int] = None
    quarantined_bytes: int = 0
    backup_path: Optional[str] = None
    repaired: bool = False
    bytes_on_disk: int = 0

    @property
    def clean(self) -> bool:
        """True when every byte on disk was a well-formed record."""
        return not (
            self.dropped_tail_bytes
            or self.interior_corruption
            or self.skipped_lines
        )

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, handy for logging or a run report."""
        return {
            "lines_read": self.lines_read,
            "records_loaded": self.records_loaded,
            "blank_lines": self.blank_lines,
            "skipped_lines": self.skipped_lines,
            "dropped_tail_bytes": self.dropped_tail_bytes,
            "salvaged_tail": self.salvaged_tail,
            "interior_corruption": self.interior_corruption,
            "interior_corruption_line": self.interior_corruption_line,
            "quarantined_bytes": self.quarantined_bytes,
            "backup_path": self.backup_path,
            "repaired": self.repaired,
            "bytes_on_disk": self.bytes_on_disk,
            "clean": self.clean,
        }

    def summary(self) -> str:
        """One-line human summary of recovery."""
        if self.clean:
            return f"{self.records_loaded} records, clean"
        parts = [f"{self.records_loaded} records"]
        if self.salvaged_tail:
            parts.append("salvaged torn tail record")
        elif self.dropped_tail_bytes:
            parts.append(f"dropped {self.dropped_tail_bytes}B torn tail")
        if self.interior_corruption:
            parts.append(
                f"quarantined region after line {self.interior_corruption_line} "
                f"({self.quarantined_bytes}B -> {self.backup_path})"
            )
        if self.skipped_lines:
            parts.append(f"skipped {self.skipped_lines} unreadable line(s)")
        return ", ".join(parts)


class ResumeManifest:
    """Append-only record of what a run has already finished.

    Args:
        path: manifest file (``*.jsonl``). Created if absent.
        done_statuses: statuses that satisfy :meth:`is_done` (default: ``ok``).
        fsync: flush each record to disk before returning from :meth:`append`.
            Keeping this on is what makes "completed" mean "survived a crash".
        repair: rewrite the file on load when damage was found, so the valid
            prefix stays contiguous (see :class:`RecoveryReport`).
        clock: injectable time source, for tests.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        done_statuses: Iterable[Union[str, ItemStatus]] = DEFAULT_DONE_STATUSES,
        fsync: bool = True,
        repair: bool = True,
        create: bool = True,
        clock: Optional[Any] = None,
    ) -> None:
        self.path = Path(path)
        self.fsync = bool(fsync)
        self.repair = bool(repair)
        self.clock = clock or time.time
        self.done_statuses = frozenset(status_value(s) for s in done_statuses)
        self._lock = threading.RLock()
        self._fd: Optional[int] = None
        self._latest: dict[str, ManifestRecord] = {}
        self._order: list[str] = []
        self._seq = 0
        self.recovery = RecoveryReport()
        self._offloaded_records: list[ManifestRecord] = []

        if self.path.exists():
            self.reload()
        elif create:
            self._create()

    # ------------------------------------------------------------- inspection
    def __len__(self) -> int:
        return len(self._latest)

    def __contains__(self, item_id: object) -> bool:
        return self.is_done(item_id)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[ManifestRecord]:
        return iter(self.records())

    @property
    def sequence(self) -> int:
        """Highest sequence number written so far."""
        return self._seq

    def is_done(self, item_id: Union[str, int]) -> bool:
        """True when *item_id* already has a record with a done status."""
        record = self._latest.get(str(item_id))
        return record is not None and record.status in self.done_statuses

    def get(self, item_id: Union[str, int]) -> Optional[ManifestRecord]:
        """Latest record for *item_id*, or ``None``."""
        return self._latest.get(str(item_id))

    def status_of(self, item_id: Union[str, int]) -> Optional[str]:
        """Latest status for *item_id*, or ``None``."""
        record = self._latest.get(str(item_id))
        return None if record is None else record.status

    def records(self) -> list[ManifestRecord]:
        """One record per item id -- the latest -- in first-seen order."""
        return [self._latest[i] for i in self._order if i in self._latest]

    def completed_ids(self) -> list[str]:
        """Item ids that satisfy :meth:`is_done`, in manifest order."""
        return [r.item_id for r in self.records() if r.status in self.done_statuses]

    def pending(self, item_ids: Iterable[Union[str, int]]) -> list[str]:
        """The subset of *item_ids* still to do, order preserved."""
        return [str(i) for i in item_ids if not self.is_done(str(i))]

    def failures(self) -> list[ManifestRecord]:
        """Every non-``ok`` record (what a run has to explain)."""
        return [r for r in self.records() if r.status not in self.done_statuses]

    def exceptions(self) -> list[ManifestRecord]:
        """The exception list: corrupted output and permanent failures.

        These items produced something that must not be mistaken for a good
        result, so the run reports them instead of hiding them.
        """
        return [
            r
            for r in self.records()
            if (r.status_enum is not None and r.status_enum.is_exception)
        ]

    def status_counts(self) -> dict[str, int]:
        """Count of records per status."""
        counts: dict[str, int] = {}
        for record in self.records():
            counts[record.status] = counts.get(record.status, 0) + 1
        return dict(sorted(counts.items()))

    def total_bytes(self) -> int:
        """Sum of recorded artefact sizes (0 for records without a size)."""
        return sum(r.size or 0 for r in self.records())

    # ---------------------------------------------------------------- writing
    def append(self, record: ManifestRecord) -> ManifestRecord:
        """Durably append *record* and return it (with its sequence number set).

        Raises:
            ManifestError: the record could not be written in full.
        """
        with self._lock:
            stamped = replace(record, seq=self._seq + 1)
            line = stamped.to_json_line().encode("utf-8")
            fd = self._ensure_fd()
            try:
                written = os.write(fd, line)
            except OSError as exc:  # pragma: no cover - disk full / permissions
                raise ManifestError(f"could not append to {self.path}: {exc}") from exc
            if written != len(line):
                self._rollback(fd, len(line) - written)
                raise ManifestError(
                    f"short write to {self.path}: wrote {written} of {len(line)} bytes"
                )
            if self.fsync:
                os.fsync(fd)
            self._apply(stamped)
            return stamped

    def mark(
        self,
        item_id: Union[str, int],
        status: Union[str, ItemStatus],
        *,
        checksum: Optional[str] = None,
        size: Optional[int] = None,
        path: Optional[Union[str, os.PathLike[str]]] = None,
        attempts: int = 0,
        reason: Optional[str] = None,
        url: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> ManifestRecord:
        """Build a stamped record and append it in one call."""
        record = ManifestRecord.create(
            str(item_id),
            status,
            checksum=checksum,
            size=size,
            path=path,
            attempts=attempts,
            reason=reason,
            url=url,
            meta=meta,
            clock=self.clock,
        )
        return self.append(record)

    def _apply(self, record: ManifestRecord) -> None:
        item = record.item_id
        if item not in self._latest:
            self._order.append(item)
        self._latest[item] = record
        self._seq = max(self._seq, record.seq)

    def _rollback(self, fd: int, extra_bytes: int) -> None:
        """Remove bytes from a short write so the file stays parseable."""
        try:
            size = os.fstat(fd).st_size
            os.ftruncate(fd, max(0, size - extra_bytes))
            os.fsync(fd)
        except OSError:  # pragma: no cover - best effort
            pass

    def _ensure_fd(self) -> int:
        if self._fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
            )
        return self._fd

    def close(self) -> None:
        """Close the append handle. The manifest stays usable; it reopens on demand."""
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None

    def __enter__(self) -> "ResumeManifest":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ load
    def _create(self) -> None:
        header = {
            "kind": "manifest",
            "v": MANIFEST_VERSION,
            "created_at": datetime.fromtimestamp(
                float(self.clock()), tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "created_epoch": float(self.clock()),
        }
        fd = self._ensure_fd()
        line = (json.dumps(header, sort_keys=True) + "\n").encode("utf-8")
        os.write(fd, line)
        if self.fsync:
            os.fsync(fd)
        self.recovery = RecoveryReport(lines_read=1, bytes_on_disk=len(line))

    def reload(self) -> RecoveryReport:
        """Re-read the manifest from disk, repairing damage if configured.

        Safe to call after another process appended; also what makes the crash
        tests possible.
        """
        with self._lock:
            self.close()
            report, records, trusted_bytes = self._scan()
            self._latest = {}
            self._order = []
            self._seq = 0
            for record in records:
                self._apply(record)
            if len(trusted_bytes) != report.bytes_on_disk and self.repair:
                report = self._rewrite(trusted_bytes, report)
            elif len(trusted_bytes) != report.bytes_on_disk:
                # Not repairing: new appends would land after the damage and be
                # invisible to the next load, so refuse to silently accept that.
                raise ManifestError(
                    f"{self.path} has damage at line "
                    f"{report.interior_corruption_line} and repair is disabled; "
                    "the manifest cannot be appended to safely"
                )
            self.recovery = report
            return report

    def _scan(self) -> tuple[RecoveryReport, list[ManifestRecord], bytes]:
        """Read and validate the file; returns (report, records, trusted bytes)."""
        return scan_file(self.path)


    def _rewrite(self, trusted: bytes, report: RecoveryReport) -> RecoveryReport:
        """Replace the file with the trusted prefix, backing up anything removed."""
        backup: Optional[Path] = None
        if report.interior_corruption:
            backup = self.path.with_name(
                f"{self.path.name}.corrupt-{int(time.time())}"
            )
            try:
                os.replace(self.path, backup)
            except OSError:  # pragma: no cover - best effort
                backup = None
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            if trusted:
                write_all(fd, trusted)
            if self.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self.path)
        return replace(
            report,
            backup_path=None if backup is None else str(backup),
            repaired=True,
        )

    # ------------------------------------------------------------- compaction
    def compact(
        self,
        *,
        drop_statuses: Iterable[Union[str, ItemStatus]] = (),
    ) -> int:
        """Rewrite the file with one record per item (latest wins).

        Returns the number of records kept. This is a deliberate rewrite, so it
        goes through a temp file and ``os.replace`` -- either the old manifest or
        the new one is on disk, never a blend of the two.
        """
        with self._lock:
            drop = {status_value(s) for s in drop_statuses}
            self.close()
            kept = [r for r in self.records() if r.status not in drop]
            header = {
                "kind": "manifest",
                "v": MANIFEST_VERSION,
                "created_at": datetime.fromtimestamp(
                    float(self.clock()), tz=timezone.utc
                ).isoformat(timespec="milliseconds"),
                "created_epoch": float(self.clock()),
                "compacted": True,
            }
            blob = (json.dumps(header, sort_keys=True) + "\n").encode("utf-8")
            blob += "".join(r.to_json_line() for r in kept).encode("utf-8")
            tmp = self.path.with_name(self.path.name + ".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                write_all(fd, blob)
                if self.fsync:
                    os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, self.path)
            self._latest = {}
            self._order = []
            self._seq = 0
            for record in kept:
                self._apply(record)
            self.recovery = RecoveryReport(
                lines_read=len(kept) + 1,
                records_loaded=len(kept),
                bytes_on_disk=len(blob),
                repaired=True,
            )
            return len(kept)

    # ---------------------------------------------------------------- reports
    def stats(self) -> dict[str, Any]:
        """Everything a run report needs, in one dict."""
        return {
            "manifest_path": str(self.path),
            "items_logged": len(self),
            "sequence": self._seq,
            "status_counts": self.status_counts(),
            "completed": len(self.completed_ids()),
            "failures": len(self.failures()),
            "exceptions": len(self.exceptions()),
            "bytes_recorded": self.total_bytes(),
            "recovery": self.recovery.as_dict(),
        }

    def write_exception_report(self, path: Union[str, os.PathLike[str]]) -> Path:
        """Write the exception list to a JSON file and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": str(self.path),
            "generated_at": datetime.fromtimestamp(
                float(self.clock()), tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "exceptions": [r.to_mapping() for r in self.exceptions()],
        }
        blob = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            write_all(fd, blob)
            if self.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        return target

    def __repr__(self) -> str:
        return (
            f"<ResumeManifest {str(self.path)!r} items={len(self)} "
            f"done={len(self.completed_ids())} exceptions={len(self.exceptions())}>"
        )


def status_value(status: Union[str, ItemStatus]) -> str:
    """Normalise a status to its string form without validating it."""
    if isinstance(status, ItemStatus):
        return status.value
    return str(status)


def coerce_status(status: Union[str, ItemStatus]) -> str:
    """Validate *status* against :class:`ItemStatus` and return its string form.

    Raises:
        ValueError: the status is not a known item status. A typo in a status
            would otherwise be written to disk and silently never match the
            resume check.
    """
    if isinstance(status, ItemStatus):
        return status.value
    return ItemStatus(str(status)).value


def read_records(path: Union[str, os.PathLike[str]]) -> tuple[list[ManifestRecord], RecoveryReport]:
    """Read the records from a manifest file without opening it for appends.

    Returns a ``(records, report)`` pair. Nothing is written.
    """
    _report, records, _trusted = scan_file(path)
    return records, _report


def latest_by_item(records: Sequence[ManifestRecord]) -> dict[str, ManifestRecord]:
    """Collapse a record sequence to the latest record per item id."""
    latest: dict[str, ManifestRecord] = {}
    for record in records:
        latest[record.item_id] = record
    return latest


def write_all(fd: int, blob: bytes) -> int:
    """Write every byte of *blob* to *fd*, looping over short writes.

    Used for deliberate whole-file rewrites (compaction, reports), where a
    partial write is harmless because the file is replaced atomically.
    """
    written = 0
    while written < len(blob):
        chunk = os.write(fd, blob[written:])
        if chunk <= 0:  # pragma: no cover - defensive
            raise ManifestError(f"write returned {chunk}")
        written += chunk
    return written


def scan_file(
    path: Union[str, os.PathLike[str]]
) -> tuple[RecoveryReport, list[ManifestRecord], bytes]:
    """Parse a manifest file without opening it for appends.

    Returns a ``(report, records, trusted_bytes)`` triple: the recovery report,
    the records that could be trusted, and the byte prefix of the file that is
    safe to keep. Nothing is written -- use :class:`ResumeManifest` if you want
    the file repaired. See :class:`RecoveryReport` for the damage vocabulary.
    """
    path = Path(path)
    if not path.exists():
        return RecoveryReport(), [], b""

    data = path.read_bytes()
    total = len(data)
    terminated = data.endswith(b"\n")
    pieces = data.split(b"\n")
    if terminated:
        pieces = pieces[:-1]

    records: list[ManifestRecord] = []
    trusted_end = 0
    offset = 0
    blank = 0
    skipped = 0
    dropped_tail = 0
    salvaged = False
    interior = False
    interior_line: Optional[int] = None
    interior_offset: Optional[int] = None

    for index, raw in enumerate(pieces):
        line_start = offset
        offset += len(raw) + 1  # +1 for the newline that split() removed
        is_last = index == len(pieces) - 1
        is_torn = is_last and not terminated
        if not raw.strip():
            blank += 1
            trusted_end = min(offset, total)
            continue
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("line is not a JSON object")
        except (ValueError, UnicodeDecodeError):
            if is_torn:
                # A crash landed a partial line at the end of the file. The bytes
                # cannot be trusted, so drop them; the item is simply re-fetched.
                dropped_tail = len(raw) + 1
                break
            # Damage in the middle: everything after it is suspect, so stop and
            # let the caller quarantine the region instead of guessing.
            interior = True
            interior_line = index + 1
            interior_offset = line_start
            break
        if is_torn:
            # No trailing newline, yet the line parses: the write landed in full
            # and only the newline was lost. Keep the record and re-terminate.
            salvaged = True
        if parsed.get("kind") == "manifest":
            trusted_end = min(offset, total)
            continue
        try:
            record = ManifestRecord.from_mapping(parsed)
        except (ValueError, TypeError):
            skipped += 1
            trusted_end = min(offset, total)
            continue
        records.append(record)
        trusted_end = min(offset, total)

    report = RecoveryReport(
        lines_read=len(pieces),
        records_loaded=len(records),
        blank_lines=blank,
        skipped_lines=skipped,
        dropped_tail_bytes=dropped_tail,
        salvaged_tail=salvaged,
        interior_corruption=interior,
        interior_corruption_line=interior_line,
        interior_corruption_offset=interior_offset,
        quarantined_bytes=(total - trusted_end) if interior else 0,
        bytes_on_disk=total,
    )
    trusted = data[:trusted_end]
    if salvaged and not trusted.endswith(b"\n"):
        # Force a repair pass so the next append does not glue onto this line.
        trusted += b"\n"
    return report, records, trusted
