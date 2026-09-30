"""Decide whether fetched bytes are trustworthy.

The expensive mistake in extraction work is not a failed request -- it is a
*successful* one. A ``200`` that returns a login page, a zero-byte file, or half
a PDF will be saved as if it worked, and nobody notices until the client opens
the folder. By then the run's own logs say everything succeeded.

:class:`IntegrityCheck` turns a :class:`~scrape_session.transport.Response` into
an explicit verdict, so the caller can act instead of hoping:

===== ====================== =============================================
ok     save it                status is 2xx, body complete, type and size
                              match, every validator passed
retry  try again later        ``429``/retryable ``5xx`` -- the server could
                              not serve this request *now*
fail   report it, do not retry``403``/``404`` and friends -- retrying cannot
                              change the answer
bad    exceptions list        2xx with truncated, empty, wrong-typed or
                              unexpected content: the artefact is corrupt and
                              must never be saved as a success
===== ====================== =============================================
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from .transport import Response, is_retryable_status

__all__ = [
    "Assessment",
    "IntegrityCheck",
    "Verdict",
    "html_payload",
    "json_validator",
    "magic_validator",
    "sha256_hex",
]

#: A validator inspects ``(body, response)`` and returns a reason string when the
#: artefact is unacceptable, or ``None`` when it is fine.
Validator = Callable[[bytes, Response], Optional[str]]

#: How much of a body to sniff when guessing at HTML error pages.
_SNIFF_BYTES = 512


class Verdict(str, Enum):
    """What an integrity check concluded about one fetched artefact."""

    OK = "ok"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_FAILURE = "permanent_failure"
    CORRUPTED = "corrupted"

    @property
    def is_ok(self) -> bool:
        """True only for :attr:`OK`."""
        return self is Verdict.OK

    @property
    def retryable(self) -> bool:
        """True when another attempt could plausibly succeed."""
        return self is Verdict.RETRYABLE_FAILURE

    @property
    def is_failure(self) -> bool:
        """True for anything that is not :attr:`OK`."""
        return self is not Verdict.OK

    @property
    def needs_attention(self) -> bool:
        """True when a human should look at this item."""
        return self in (Verdict.PERMANENT_FAILURE, Verdict.CORRUPTED)

    @property
    def manifest_status(self) -> str:
        """The :class:`~scrape_session.manifest.ItemStatus` value this maps to."""
        return self.value


@dataclass(frozen=True)
class Assessment:
    """The verdict plus the evidence behind it."""

    verdict: Verdict
    reason: str
    status: int = 0
    size: int = 0
    checksum: Optional[str] = None
    content_type: str = ""
    detail: str = ""
    checks: Mapping[str, bool] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the artefact may be saved."""
        return self.verdict is Verdict.OK

    @property
    def retryable(self) -> bool:
        """True when the caller should back off and retry."""
        return self.verdict.retryable

    @property
    def corrupted(self) -> bool:
        """True when the bytes arrived but cannot be trusted."""
        return self.verdict is Verdict.CORRUPTED

    @property
    def description(self) -> str:
        """One-line explanation suitable for a run log."""
        bits = [self.verdict.value, self.reason]
        if self.status:
            bits.append(f"HTTP {self.status}")
        bits.append(f"{self.size}B")
        if self.detail:
            bits.append(self.detail)
        return " | ".join(bits)

    def as_fields(self) -> dict[str, Any]:
        """Manifest-ready fields describing this assessment."""
        return {
            "status": self.verdict.manifest_status,
            "checksum": self.checksum,
            "size": self.size,
            "reason": self.reason,
        }

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, handy for logging or a JSON report."""
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "status": self.status,
            "size": self.size,
            "checksum": self.checksum,
            "content_type": self.content_type,
            "detail": self.detail,
            "checks": dict(self.checks),
        }


def sha256_hex(data: bytes) -> str:
    """Return the ``sha256:<hex>`` digest used throughout this package."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def html_payload(body: bytes) -> bool:
    """True when the body looks like an HTML document.

    A session-authenticated app that silently redirects (or just answers) an
    expired request with its login page returns ``200 text/html``. That page is
    the single most common way a failed fetch gets saved as a PDF.
    """
    head = body[:_SNIFF_BYTES].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or (
        b"<head" in head and b"<body" in head
    )


def magic_validator(*prefixes: bytes) -> Validator:
    """Build a validator requiring the body to start with one of *prefixes*.

    Example: ``magic_validator(b"%PDF-")``.
    """
    if not prefixes:
        raise ValueError("magic_validator needs at least one prefix")
    wanted = tuple(prefixes)

    def validate(body: bytes, response: Response) -> Optional[str]:
        if not body.startswith(wanted):
            prefix_list = ", ".join(repr(p) for p in wanted)
            return f"body does not start with any of {prefix_list}"
        return None

    validate.__name__ = "magic_validator"  # type: ignore[attr-defined]
    return validate


def json_validator(
    *, mode: Union[str, type] = "any", max_bytes: Optional[int] = None
) -> Validator:
    """Build a validator requiring the body to be valid JSON.

    ``mode`` may be ``"any"``, ``"object"``, ``"array"`` or a Python type such as
    ``dict`` / ``list``.
    """

    def validate(body: bytes, response: Response) -> Optional[str]:
        if max_bytes is not None and len(body) > max_bytes:
            return f"body exceeds max_bytes={max_bytes}"
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            return f"body is not valid JSON: {exc}"
        if mode == "any":
            return None
        expected: Any = mode
        if mode == "object":
            expected = dict
        elif mode == "array":
            expected = list
        if not isinstance(parsed, expected):
            return f"JSON root is {type(parsed).__name__}, expected {expected!r}"
        return None

    validate.__name__ = "json_validator"  # type: ignore[attr-defined]
    return validate


class IntegrityCheck:
    """Classify a response as ok, retryable failure, permanent failure, or corruption.

    Args:
        expect_content_type: accepted media types, e.g. ``"application/pdf"`` or
            ``("application/json", "text/json")``. A wildcard family such as
            ``"image/*"`` is also accepted. Matching is on the media type only,
            so ``application/pdf; charset=binary`` still matches.
        min_bytes: smallest acceptable body; ``0`` means an empty body is fine.
            The default of ``1`` treats a zero-byte ``200`` as corruption.
        validators: extra callables that can reject an otherwise plausible body.
        detect_html: when an expected content type is set and it is not HTML,
            treat an HTML body as corruption regardless of the declared type.
        require_success: when ``False``, non-2xx statuses are not classified as
            failures (useful when you deliberately fetch error pages).
    """

    def __init__(
        self,
        *,
        expect_content_type: Optional[Union[str, Sequence[str]]] = None,
        min_bytes: int = 1,
        max_bytes: Optional[int] = None,
        validators: Iterable[Validator] = (),
        detect_html: bool = True,
        require_success: bool = True,
    ) -> None:
        self.expect_content_type = _normalise_types(expect_content_type)
        self.min_bytes = int(min_bytes)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.validators = tuple(validators)
        self.detect_html = bool(detect_html)
        self.require_success = bool(require_success)

    # ------------------------------------------------------------------ public
    def assess(
        self,
        response: Response,
        *,
        expected_size: Optional[int] = None,
        expected_content_type: Optional[Union[str, Sequence[str]]] = None,
        min_bytes: Optional[int] = None,
        validators: Iterable[Validator] = (),
    ) -> Assessment:
        """Inspect *response* and return an :class:`Assessment`.

        Per-call arguments override the instance defaults, so one check object
        can serve a mixed workload (PDFs with a size known from a previous run,
        JSON listings with none).
        """
        body = response.body
        checksum = sha256_hex(body) if body else None
        status = int(response.status)
        content_type = response.content_type
        checks: dict[str, bool] = {}

        def verdict(
            value: Verdict, reason: str, detail: str = ""
        ) -> Assessment:
            return Assessment(
                verdict=value,
                reason=reason,
                status=status,
                size=len(body),
                checksum=checksum,
                content_type=content_type,
                detail=detail,
                checks=checks,
            )

        # 1. A transport that knows the body was cut short outranks everything.
        if response.truncated:
            checks["complete_body"] = False
            return verdict(
                Verdict.CORRUPTED,
                "truncated_body",
                response.truncated_reason or "transport reported an incomplete body",
            )
        checks["complete_body"] = True

        # 2. HTTP status decides the failure family before content is judged.
        if not (200 <= status < 300):
            if not self.require_success:
                checks["status_acceptable"] = True
            elif is_retryable_status(status):
                return verdict(Verdict.RETRYABLE_FAILURE, f"http_{status}")
            else:
                return verdict(Verdict.PERMANENT_FAILURE, f"http_{status}")
        else:
            checks["status_acceptable"] = True

        # 3. Declared length vs. actual length: a silent truncation signal that
        #    needs no cooperation from the transport.
        declared = response.declared_length
        if declared is not None and declared != len(body):
            checks["length_matches"] = False
            return verdict(
                Verdict.CORRUPTED,
                "content_length_mismatch",
                f"server declared {declared}B, received {len(body)}B",
            )
        checks["length_matches"] = True

        # 4. Wrong content type: the classic expired-session login page.
        wanted = (
            _normalise_types(expected_content_type)
            if expected_content_type is not None
            else self.expect_content_type
        )
        if wanted:
            if not _type_matches(content_type, wanted):
                checks["content_type"] = False
                return verdict(
                    Verdict.CORRUPTED,
                    "content_type_mismatch",
                    f"got {content_type or '(none)'!r}, expected one of {sorted(wanted)}",
                )
            checks["content_type"] = True
            if self.detect_html and html_payload(body) and "text/html" not in wanted:
                checks["html_payload"] = False
                return verdict(
                    Verdict.CORRUPTED,
                    "html_payload",
                    f"body looks like an HTML page while {sorted(wanted)} was expected",
                )
        checks["html_payload"] = not html_payload(body)

        # 5. Size floors: an empty or absurdly small body is corruption.
        floor = self.min_bytes if min_bytes is None else int(min_bytes)
        if len(body) < floor:
            checks["min_bytes"] = False
            return verdict(
                Verdict.CORRUPTED,
                "empty_body" if not body else "body_below_min_bytes",
                f"received {len(body)}B, minimum {floor}B",
            )
        checks["min_bytes"] = True

        if self.max_bytes is not None and len(body) > self.max_bytes:
            checks["max_bytes"] = False
            return verdict(
                Verdict.CORRUPTED,
                "body_above_max_bytes",
                f"received {len(body)}B, maximum {self.max_bytes}B",
            )
        checks["max_bytes"] = True

        # 6. Size known from a previous run (or a listing) must still match.
        if expected_size is not None and int(expected_size) != len(body):
            checks["expected_size"] = False
            return verdict(
                Verdict.CORRUPTED,
                "size_mismatch",
                f"expected {int(expected_size)}B, received {len(body)}B",
            )
        checks["expected_size"] = True

        # 7. Caller-supplied validators run last, on content that passed everything
        #    structural.
        for validator in tuple(validators) + self.validators:
            failure = validator(body, response)
            if failure:
                name = getattr(validator, "__name__", type(validator).__name__)
                checks[f"validator:{name}"] = False
                return verdict(Verdict.CORRUPTED, f"validator_failed:{name}", failure)
        checks["validators"] = True

        return verdict(Verdict.OK, "ok")

    def is_ok(self, response: Response, **kwargs: Any) -> bool:
        """Convenience predicate: ``self.assess(response, **kwargs).ok``."""
        return self.assess(response, **kwargs).ok

    # --------------------------------------------------------------- factories
    @classmethod
    def for_pdf(cls, *, min_bytes: int = 1024, **kwargs: Any) -> "IntegrityCheck":
        """A check for PDF artefacts: PDF type, minimum size, ``%PDF-`` magic.

        ``min_bytes`` defaults to 1024 because a real PDF is never a few bytes;
        raise or lower it to suit the documents you actually receive.
        """
        kwargs.setdefault("expect_content_type", "application/pdf")
        validators = tuple(kwargs.pop("validators", ()))
        return cls(
            min_bytes=min_bytes,
            validators=(magic_validator(b"%PDF-"),) + validators,
            **kwargs,
        )

    @classmethod
    def for_json(cls, *, root: Union[str, type] = "any", **kwargs: Any) -> "IntegrityCheck":
        """A check for JSON endpoints: JSON type plus a JSON parse."""
        kwargs.setdefault("expect_content_type", ("application/json", "text/json"))
        validators = tuple(kwargs.pop("validators", ()))
        return cls(
            validators=(json_validator(mode=root),) + validators,
            **kwargs,
        )

    @classmethod
    def for_html(cls, *, min_bytes: int = 128, **kwargs: Any) -> "IntegrityCheck":
        """A check for HTML pages (detail pages, listings)."""
        kwargs.setdefault("expect_content_type", "text/html")
        return cls(min_bytes=min_bytes, detect_html=False, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<IntegrityCheck expect={sorted(self.expect_content_type) or 'any'} "
            f"min_bytes={self.min_bytes} validators={len(self.validators)}>"
        )


def _normalise_types(
    value: Optional[Union[str, Sequence[str]]]
) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value.split(";", 1)[0].strip().lower()})
    return frozenset(
        str(item).split(";", 1)[0].strip().lower() for item in value if str(item).strip()
    )


def _type_matches(content_type: str, wanted: frozenset[str]) -> bool:
    actual = (content_type or "").strip().lower()
    if not actual:
        return False
    for candidate in wanted:
        if candidate == actual:
            return True
        if candidate.endswith("/*") and actual.startswith(candidate[:-1]):
            return True
        # A suffix match lets "application/pdf" accept "application/x-pdf".
        if candidate.split("/")[-1] == actual.split("/")[-1]:
            return True
    return False
