"""scrape-session: the boring parts of long extraction jobs, done properly.

Four primitives, each usable on its own, all standard library only:

* :class:`~scrape_session.session.SessionKeeper` -- sign in once, persist the
  session, reload it next run, re-authenticate only when it is actually gone.
* :class:`~scrape_session.manifest.ResumeManifest` -- append a durable record
  after every finished item so a crashed run can skip straight to what is left.
* :class:`~scrape_session.ratelimit.RateLimiter` -- a ceiling you enforce, plus
  bounded exponential backoff for genuinely transient failures.
* :class:`~scrape_session.integrity.IntegrityCheck` -- separate "the request
  failed" from "the bytes are wrong", so a truncated PDF is never saved as a win.

Everything is transport-agnostic: the primitives accept any
``Callable[[Request], Response]``, so the same code drives stdlib HTTP or a
browser you supply::

    from scrape_session import IntegrityCheck, RateLimiter, Request, ResumeManifest
    from scrape_session import SessionKeeper, UrllibTransport

    session = SessionKeeper("state/session.json", authenticate=sign_in)
    manifest = ResumeManifest("state/manifest.jsonl")
    limiter = RateLimiter(max_requests_per_second=2, policy=RetryPolicy())
    check = IntegrityCheck.for_pdf()

    fetch = session.bind(UrllibTransport())
    todo = manifest.pending(item_ids)
    for item_id in todo:
        response = limiter.run(fetch, Request(f"https://example.com/item/{item_id}"))
        assessment = check.assess(response)
        ...
"""

from __future__ import annotations

from .integrity import (
    Assessment,
    IntegrityCheck,
    Verdict,
    html_payload,
    json_validator,
    magic_validator,
    sha256_hex,
)
from .manifest import (
    DEFAULT_DONE_STATUSES,
    MANIFEST_VERSION,
    ItemStatus,
    ManifestError,
    ManifestRecord,
    RecoveryReport,
    ResumeManifest,
    coerce_status,
    latest_by_item,
    read_records,
    scan_file,
)
from .ratelimit import (
    RETRYABLE_STATUSES,
    RateLimiter,
    RateLimiterStats,
    RetryEvent,
    RetryExhausted,
    RetryPolicy,
    parse_retry_after,
)
from .session import (
    SESSION_FILE_VERSION,
    Cookie,
    SessionExpired,
    SessionKeeper,
    SessionStats,
)
from .transport import (
    ConnectionFailure,
    Fetch,
    Headers,
    Request,
    Response,
    TimeoutFailure,
    Transport,
    TransportError,
    UrllibTransport,
    as_fetch,
    is_retryable_status,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DONE_STATUSES",
    "MANIFEST_VERSION",
    "RETRYABLE_STATUSES",
    "SESSION_FILE_VERSION",
    "Assessment",
    "ConnectionFailure",
    "Cookie",
    "Fetch",
    "Headers",
    "IntegrityCheck",
    "ItemStatus",
    "ManifestError",
    "ManifestRecord",
    "RateLimiter",
    "RateLimiterStats",
    "RecoveryReport",
    "Request",
    "Response",
    "ResumeManifest",
    "RetryEvent",
    "RetryExhausted",
    "RetryPolicy",
    "SessionExpired",
    "SessionKeeper",
    "SessionStats",
    "TimeoutFailure",
    "Transport",
    "TransportError",
    "UrllibTransport",
    "Verdict",
    "__version__",
    "as_fetch",
    "coerce_status",
    "html_payload",
    "is_retryable_status",
    "json_validator",
    "latest_by_item",
    "magic_validator",
    "parse_retry_after",
    "read_records",
    "scan_file",
    "sha256_hex",
]
