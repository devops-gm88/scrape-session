"""The manifest's crash-safety claims, tested by actually breaking it.

A resume log that quietly loses the whole run when a write is interrupted is
worse than no log at all, because it makes a partial run look complete. So the
tests below do not check the happy path and stop: they kill a write mid-flight,
kill a whole process mid-write, and corrupt the middle of the file, then assert
what the next run sees.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scrape_session import (
    ItemStatus,
    ManifestError,
    ManifestRecord,
    ResumeManifest,
    latest_by_item,
    read_records,
    scan_file,
)

HALF_RECORD_CRASH = "simulated crash: the process died part way through a write"


class ManifestTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "manifest.jsonl"

    def manifest(self, **kwargs) -> ResumeManifest:
        manifest = ResumeManifest(self.path, fsync=False, **kwargs)
        self.addCleanup(manifest.close)
        return manifest

    def lines(self) -> list[str]:
        return [
            line
            for line in self.path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]


class BasicsTests(ManifestTestCase):
    def test_new_manifest_starts_with_a_header_line(self) -> None:
        self.manifest()
        lines = self.lines()
        self.assertEqual(len(lines), 1)
        header = json.loads(lines[0])
        self.assertEqual(header["kind"], "manifest")
        self.assertEqual(header["v"], 1)

    def test_mark_appends_one_line_per_item_with_evidence(self) -> None:
        manifest = self.manifest()
        record = manifest.mark(
            "doc-1",
            ItemStatus.OK,
            checksum="sha256:abc",
            size=1234,
            path=self.dir / "out" / "doc-1.pdf",
            attempts=2,
            url="http://example.com/doc/1",
        )
        self.assertEqual(record.seq, 1)
        self.assertTrue(record.timestamp.endswith("+00:00"))
        self.assertEqual(record.path, str(self.dir / "out" / "doc-1.pdf"))
        stored = json.loads(self.lines()[-1])
        self.assertEqual(stored["id"], "doc-1")
        self.assertEqual(stored["status"], "ok")
        self.assertEqual(stored["checksum"], "sha256:abc")
        self.assertEqual(stored["size"], 1234)
        self.assertEqual(stored["attempts"], 2)

    def test_resume_skips_completed_items(self) -> None:
        manifest = self.manifest()
        manifest.mark("a", ItemStatus.OK, size=10)
        manifest.mark("b", ItemStatus.CORRUPTED, reason="truncated_body")
        manifest.mark("c", ItemStatus.OK, size=30)
        self.assertTrue(manifest.is_done("a"))
        self.assertFalse(manifest.is_done("b"), "corruption is not done")
        self.assertEqual(manifest.completed_ids(), ["a", "c"])
        self.assertEqual(manifest.pending(["a", "b", "c", "d"]), ["b", "d"])
        self.assertIn("a", manifest)
        self.assertNotIn("b", manifest)

    def test_reload_from_disk_restores_the_state(self) -> None:
        manifest = self.manifest()
        manifest.mark("a", ItemStatus.OK, size=10)
        manifest.mark("b", ItemStatus.OK, size=20)
        manifest.close()
        reopened = self.manifest()
        self.assertEqual(reopened.completed_ids(), ["a", "b"])
        self.assertEqual(reopened.sequence, 2)
        self.assertTrue(reopened.recovery.clean)

    def test_latest_record_wins_for_a_repeated_item(self) -> None:
        manifest = self.manifest()
        manifest.mark("a", ItemStatus.RETRYABLE_FAILURE, reason="http_429")
        self.assertFalse(manifest.is_done("a"))
        manifest.mark("a", ItemStatus.OK, size=10)
        self.assertTrue(manifest.is_done("a"))
        self.assertEqual(len(manifest.records()), 1)
        self.assertEqual(manifest.get("a").status, "ok")
        self.assertEqual(manifest.status_counts(), {"ok": 1})

    def test_status_can_move_backwards_and_is_reported_as_the_latest(self) -> None:
        manifest = self.manifest()
        manifest.mark("a", ItemStatus.OK)
        manifest.mark("a", ItemStatus.PERMANENT_FAILURE, reason="http_403")
        self.assertFalse(manifest.is_done("a"))
        self.assertEqual(manifest.status_of("a"), "permanent_failure")

    def test_unknown_status_string_is_rejected_at_write_time(self) -> None:
        manifest = self.manifest()
        with self.assertRaises(ValueError):
            manifest.mark("a", "sucess")  # a typo must not reach the file
        self.assertEqual(len(manifest), 0)

    def test_unknown_status_on_disk_is_tolerated_when_read(self) -> None:
        self.path.write_text(
            json.dumps({"kind": "item", "id": "a", "status": "quarantined"}) + "\n",
            encoding="utf-8",
        )
        manifest = self.manifest()
        self.assertFalse(manifest.is_done("a"))
        self.assertEqual(manifest.status_of("a"), "quarantined")
        self.assertIsNone(manifest.get("a").status_enum)

    def test_unknown_fields_survive_a_round_trip(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "kind": "item",
                    "id": "a",
                    "status": "ok",
                    "future_field": {"nested": [1, 2]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = self.manifest()
        record = manifest.get("a")
        self.assertEqual(record.extra["future_field"], {"nested": [1, 2]})
        manifest.append(record)
        self.assertEqual(
            json.loads(self.lines()[-1])["future_field"], {"nested": [1, 2]}
        )

    def test_exception_list_holds_corruption_and_permanent_failures(self) -> None:
        manifest = self.manifest()
        manifest.mark("ok", ItemStatus.OK)
        manifest.mark("bad", ItemStatus.CORRUPTED, reason="truncated_body")
        manifest.mark("denied", ItemStatus.PERMANENT_FAILURE, reason="http_403")
        manifest.mark("flaky", ItemStatus.RETRYABLE_FAILURE, reason="http_429")
        self.assertEqual(
            [r.item_id for r in manifest.exceptions()], ["bad", "denied"]
        )
        self.assertEqual(
            [r.item_id for r in manifest.failures()], ["bad", "denied", "flaky"]
        )
        self.assertEqual(len(manifest), 4, "len() counts distinct items logged")
        self.assertEqual(
            len(manifest.completed_ids()), 1, "only the ok item counts as done"
        )

    def test_stats_and_repr_describe_the_run(self) -> None:
        manifest = self.manifest()
        manifest.mark("a", ItemStatus.OK, size=100)
        manifest.mark("b", ItemStatus.CORRUPTED, size=10, reason="empty_body")
        stats = manifest.stats()
        self.assertEqual(stats["items_logged"], 2)
        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["exceptions"], 1)
        self.assertEqual(stats["bytes_recorded"], 110)
        self.assertEqual(stats["status_counts"], {"corrupted": 1, "ok": 1})
        self.assertIn("exceptions=1", repr(manifest))

    def test_write_exception_report_lists_the_exceptions(self) -> None:
        manifest = self.manifest()
        manifest.mark("bad", ItemStatus.CORRUPTED, reason="truncated_body")
        target = manifest.write_exception_report(self.dir / "exceptions.json")
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["exceptions"]), 1)
        self.assertEqual(payload["exceptions"][0]["id"], "bad")
        self.assertEqual(payload["exceptions"][0]["reason"], "truncated_body")

    def test_short_write_rolls_back_and_raises(self) -> None:
        manifest = self.manifest()
        manifest.mark("a", ItemStatus.OK)
        real_write = os.write

        def half_write(fd: int, data: bytes) -> int:
            return real_write(fd, data[: max(1, len(data) // 2)])

        with mock.patch("scrape_session.manifest.os.write", side_effect=half_write):
            with self.assertRaises(ManifestError):
                manifest.mark("b", ItemStatus.OK)
        manifest.close()
        reopened = self.manifest()
        self.assertEqual(reopened.completed_ids(), ["a"])
        self.assertTrue(reopened.recovery.clean, "the torn bytes were rolled back")


class CrashRecoveryTests(ManifestTestCase):
    """A crash can only ever reach the last line, because that is all we write."""

    def test_partial_line_at_the_end_is_dropped_and_the_run_survives(self) -> None:
        manifest = self.manifest()
        for index in range(5):
            manifest.mark(f"item-{index}", ItemStatus.OK, size=index * 100)
        manifest.close()
        whole = self.path.read_bytes()
        self.path.write_bytes(whole + b'{"kind":"item","id":"item-5","stat')

        manifest = self.manifest()
        self.assertEqual(manifest.completed_ids(), [f"item-{i}" for i in range(5)])
        self.assertFalse(manifest.is_done("item-5"))
        self.assertTrue(manifest.recovery.dropped_tail_bytes > 0)
        self.assertFalse(manifest.recovery.interior_corruption)
        self.assertFalse(manifest.recovery.clean)
        self.assertEqual(manifest.recovery.records_loaded, 5)
        self.assertIn("torn tail", manifest.recovery.summary())

    def test_the_file_is_repaired_so_the_next_append_is_visible(self) -> None:
        manifest = self.manifest()
        manifest.mark("item-0", ItemStatus.OK)
        manifest.close()
        self.path.write_bytes(self.path.read_bytes() + b'{"kind":"item","id":"i')

        manifest = self.manifest()
        self.assertFalse(manifest.recovery.clean)
        self.assertTrue(manifest.recovery.repaired, "the damage must be cleared")
        manifest.mark("item-1", ItemStatus.OK)
        manifest.close()

        reopened = self.manifest()
        self.assertEqual(reopened.completed_ids(), ["item-0", "item-1"])
        self.assertTrue(reopened.recovery.clean)
        for line in self.lines():
            json.loads(line)  # every line is still whole JSON

    def test_a_write_that_died_mid_record_leaves_the_earlier_records_intact(self) -> None:
        manifest = self.manifest()
        for index in range(4):
            manifest.mark(f"item-{index}", ItemStatus.OK, size=index)
        real_write = os.write
        calls = {"n": 0}

        def dying_write(fd: int, data: bytes) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                # The process dies here: half a record reached the disk, no
                # rollback ran, no fsync completed.
                real_write(fd, data[: len(data) // 2])
                raise RuntimeError(HALF_RECORD_CRASH)
            return real_write(fd, data)

        with mock.patch("scrape_session.manifest.os.write", side_effect=dying_write):
            with self.assertRaises(RuntimeError):
                manifest.mark("item-4", ItemStatus.OK, size=4)
        manifest.close()

        recovered = self.manifest()
        self.assertEqual(
            recovered.completed_ids(), [f"item-{i}" for i in range(4)]
        )
        self.assertEqual(recovered.sequence, 4)
        self.assertTrue(recovered.recovery.dropped_tail_bytes > 0)
        self.assertEqual(len(recovered.recovery.summary()) > 0, True)

    def test_a_process_killed_mid_write_loses_only_the_in_flight_record(self) -> None:
        manifest = self.manifest()
        for index in range(5):
            manifest.mark(f"item-{index}", ItemStatus.OK, size=index)
        manifest.close()
        size_before = self.path.stat().st_size

        # A real second process appends to the same manifest and dies part way
        # through the write (os._exit: no unwinding, no flush, no cleanup).
        source_root = Path(__file__).resolve().parents[1] / "src"
        script = f"""
import os, sys
sys.path.insert(0, {str(source_root)!r})
from scrape_session import ItemStatus, ResumeManifest

manifest = ResumeManifest(sys.argv[1], fsync=False)
real_write = os.write

def dying_write(fd, data):
    real_write(fd, data[: max(1, len(data) // 3)])
    os._exit(9)

os.write = dying_write
manifest.mark("item-5", ItemStatus.OK, size=5)
os._exit(0)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, str(self.path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            9,
            f"the child was supposed to die mid-write; stderr={completed.stderr}",
        )
        self.assertGreater(
            self.path.stat().st_size,
            size_before,
            "the child's partial write should be on disk",
        )

        recovered = self.manifest()
        self.assertEqual(recovered.completed_ids(), [f"item-{i}" for i in range(5)])
        self.assertFalse(recovered.is_done("item-5"))
        self.assertGreater(recovered.recovery.dropped_tail_bytes, 0)
        self.assertFalse(recovered.recovery.interior_corruption)

        recovered.mark("item-5", ItemStatus.OK, size=5)
        recovered.close()
        self.assertEqual(
            self.manifest().completed_ids(), [f"item-{i}" for i in range(6)]
        )

    def test_a_complete_record_missing_only_its_newline_is_salvaged(self) -> None:
        manifest = self.manifest()
        manifest.mark("item-0", ItemStatus.OK, size=1)
        manifest.close()
        blob = self.path.read_bytes()
        self.assertTrue(blob.endswith(b"\n"))
        # The write landed in full; only the trailing newline was lost.
        self.path.write_bytes(blob[:-1])

        manifest = self.manifest()
        self.assertTrue(manifest.recovery.salvaged_tail)
        self.assertEqual(manifest.completed_ids(), ["item-0"])
        manifest.mark("item-1", ItemStatus.OK, size=2)
        manifest.close()
        reopened = self.manifest()
        self.assertEqual(reopened.completed_ids(), ["item-0", "item-1"])
        self.assertTrue(reopened.recovery.clean)

    def test_damage_in_the_middle_quarantines_the_rest_and_keeps_the_prefix(self) -> None:
        manifest = self.manifest()
        for index in range(6):
            manifest.mark(f"item-{index}", ItemStatus.OK, size=index)
        manifest.close()
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        original = self.path.read_bytes()
        lines.insert(3, "this is not json at all\n")
        self.path.write_text("".join(lines), encoding="utf-8")
        corrupted = self.path.read_bytes()

        manifest = self.manifest()
        self.assertTrue(manifest.recovery.interior_corruption)
        self.assertEqual(manifest.recovery.interior_corruption_line, 4)
        self.assertEqual(manifest.completed_ids(), ["item-0", "item-1"])
        self.assertFalse(manifest.is_done("item-2"))
        backup = Path(manifest.recovery.backup_path or "")
        self.assertTrue(backup.exists(), "nothing may be discarded silently")
        self.assertEqual(
            backup.read_bytes(), corrupted, "the backup is the untouched original"
        )
        self.assertNotEqual(backup.read_bytes(), original)

        manifest.mark("item-2", ItemStatus.OK, size=2)
        manifest.close()
        reopened = self.manifest()
        self.assertTrue(reopened.recovery.clean)
        self.assertEqual(reopened.completed_ids(), ["item-0", "item-1", "item-2"])

    def test_repair_disabled_refuses_to_append_to_a_damaged_manifest(self) -> None:
        self.path.write_text(
            '{"kind":"item","id":"a","status":"ok"}\nnot json\n'
            '{"kind":"item","id":"b","status":"ok"}\n',
            encoding="utf-8",
        )
        before = self.path.read_bytes()
        with self.assertRaises(ManifestError):
            ResumeManifest(self.path, fsync=False, repair=False)
        self.assertEqual(self.path.read_bytes(), before, "the file is left alone")

    def test_a_record_that_parses_but_has_no_id_is_skipped_not_fatal(self) -> None:
        self.path.write_text(
            '{"kind":"item","id":"a","status":"ok"}\n'
            '{"kind":"item","status":"ok"}\n'
            '{"kind":"item","id":"b","status":"ok"}\n',
            encoding="utf-8",
        )
        manifest = self.manifest()
        self.assertEqual(manifest.completed_ids(), ["a", "b"])
        self.assertEqual(manifest.recovery.skipped_lines, 1)
        self.assertFalse(manifest.recovery.clean)
        self.assertFalse(manifest.recovery.interior_corruption)

    def test_blank_lines_are_tolerated(self) -> None:
        self.path.write_text(
            '{"kind":"item","id":"a","status":"ok"}\n\n\n', encoding="utf-8"
        )
        manifest = self.manifest()
        self.assertEqual(manifest.completed_ids(), ["a"])
        self.assertEqual(manifest.recovery.blank_lines, 2)

    def test_an_empty_file_is_a_valid_starting_point(self) -> None:
        self.path.write_bytes(b"")
        manifest = self.manifest()
        self.assertEqual(len(manifest), 0)
        self.assertTrue(manifest.recovery.clean)
        manifest.mark("a", ItemStatus.OK)
        self.assertTrue(self.manifest().is_done("a"))

    def test_scan_file_reports_without_repairing(self) -> None:
        self.path.write_text(
            '{"kind":"item","id":"a","status":"ok"}\n{"partial":', encoding="utf-8"
        )
        before = self.path.read_bytes()
        records, report = read_records(self.path)
        self.assertEqual([r.item_id for r in records], ["a"])
        self.assertGreater(report.dropped_tail_bytes, 0)
        self.assertFalse(report.repaired)
        self.assertEqual(self.path.read_bytes(), before, "scanning must not write")

    def test_scan_file_on_a_missing_file_is_empty_and_clean(self) -> None:
        report, records, trusted = scan_file(self.dir / "nope.jsonl")
        self.assertEqual(records, [])
        self.assertEqual(trusted, b"")
        self.assertTrue(report.clean)
        self.assertEqual(report.bytes_on_disk, 0)


class ConcurrencyAndCompactionTests(ManifestTestCase):
    def test_concurrent_appends_do_not_interleave(self) -> None:
        manifest = self.manifest()
        threads = 6
        per_thread = 25
        errors: list[BaseException] = []

        def worker(worker_id: int) -> None:
            try:
                for index in range(per_thread):
                    manifest.mark(f"w{worker_id}-{index}", ItemStatus.OK, size=index)
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()
        self.assertEqual(errors, [])
        manifest.close()

        reopened = self.manifest()
        self.assertEqual(len(reopened), threads * per_thread)
        self.assertTrue(reopened.recovery.clean, "no line may be interleaved")
        self.assertEqual(reopened.sequence, threads * per_thread)

    def test_compact_keeps_one_record_per_item_and_is_readable(self) -> None:
        manifest = self.manifest()
        for round_number in range(3):
            for item in ("a", "b", "c"):
                manifest.mark(
                    item,
                    ItemStatus.OK if round_number else ItemStatus.RETRYABLE_FAILURE,
                    size=round_number,
                )
        self.assertEqual(len(manifest.records()), 3)
        self.assertEqual(manifest.sequence, 9)

        kept = manifest.compact()
        self.assertEqual(kept, 3)
        reopened = self.manifest()
        self.assertEqual(reopened.completed_ids(), ["a", "b", "c"])
        self.assertTrue(reopened.recovery.clean)
        self.assertEqual(len(reopened.records()), 3)
        self.assertEqual(
            json.loads(self.lines()[0])["compacted"], True, "header notes the rewrite"
        )

    def test_compact_can_drop_records_by_status(self) -> None:
        manifest = self.manifest()
        manifest.mark("a", ItemStatus.OK)
        manifest.mark("b", ItemStatus.SKIPPED)
        self.assertEqual(manifest.compact(drop_statuses=[ItemStatus.SKIPPED]), 1)
        self.assertEqual([r.item_id for r in self.manifest().records()], ["a"])

    def test_latest_by_item_helper(self) -> None:
        records = [
            ManifestRecord.create("a", "retryable_failure"),
            ManifestRecord.create("a", "ok"),
            ManifestRecord.create("b", "ok"),
        ]
        latest = latest_by_item(records)
        self.assertEqual(latest["a"].status, "ok")
        self.assertEqual(sorted(latest), ["a", "b"])

    def test_manifest_works_as_a_context_manager(self) -> None:
        with ResumeManifest(self.path, fsync=False) as manifest:
            manifest.mark("a", ItemStatus.OK)
            self.assertTrue(manifest.is_done("a"))
        self.assertEqual(self.manifest().completed_ids(), ["a"])

    def test_fsync_is_used_by_default(self) -> None:
        with mock.patch("scrape_session.manifest.os.fsync") as fsync:
            manifest = ResumeManifest(self.path, fsync=True)
            manifest.mark("a", ItemStatus.OK)
            manifest.close()
        self.assertGreaterEqual(fsync.call_count, 2, "header and record are flushed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
