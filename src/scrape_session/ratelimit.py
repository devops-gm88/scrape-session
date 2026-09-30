"""Pacing and bounded retries: stay under the ceiling, and know when to stop.

Two jobs, kept deliberately separate:

* :class:`RateLimiter` never lets requests start faster than the configured
  ceiling. The ceiling is yours to set; this class only enforces it.
* :class:`RetryPolicy` decides whether a *specific* failure is worth another
  attempt. ``429`` and the retryable ``5xx`` codes are; ``403`` and ``404`` are
  not, because asking again cannot change the answer.

The clock and the sleep function are injectable, so the backoff schedule can be
asserted exactly in a test instead of being inferred from wall-clock timing.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterable
from datetime import timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union

from .transport import (
    RETRYABLE_STATUSES,
    ConnectionFailure,
    Fetch,
    Request,
    Response,
    TimeoutFailure,
    TransportError,
    as_fetch,
    is_retryable_status,
)

__all__ = [
    "RETRYABLE_STATUSES",
    "RateLimiter",
    "RateLimiterStats",
    "RetryEvent",
    "RetryExhausted",
    "RetryPolicy",
    "is_retryable_status",
]


class RetryExhausted(TransportError):
    """Every allowed attempt failed at the transport level.

    Carries the last :class:`~scrape_session.transport.TransportError` as
    ``__cause__`` and records how many attempts were made. HTTP-level failures
    (a real response with a bad status) never raise this -- they are returned so
    the caller can classify them.
    """

    def __init__(self, message: str, *, attempts: int, last_error: BaseException) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


@dataclass(frozen=True)
class RetryEvent:
    """One decision to wait and try again -- useful for logs and run reports."""

    url: str
    attempt: int
    max_attempts: int
    delay: float
    status: Optional[int] = None
    error: Optional[str] = None
    reason: str = ""

    @property
    def next_attempt(self) -> int:
        """The attempt number the delay is buying."""
        return self.attempt + 1

    def __str__(self) -> str:
        what = f"HTTP {self.status}" if self.status is not None else (self.error or "error")
        return (
            f"attempt {self.attempt}/{self.max_attempts} failed ({what}); "
            f"sleeping {self.delay:.3f}s"
        )


@dataclass
class RetryPolicy:
    """How many times to retry, and how long to wait between attempts.

    Args:
        max_attempts: total attempts including the first (``1`` disables retry).
        base_delay: delay before the second attempt, in seconds.
        multiplier: growth factor applied per additional attempt.
        max_delay: ceiling for a single computed delay.
        jitter: fraction of the computed delay added at random, to avoid every
            worker retrying in lockstep. ``0.0`` makes the schedule exact.
        retry_statuses: statuses treated as transient.
        respect_retry_after: honour a ``Retry-After`` header when the server
            sends one (it is never shortened below the computed backoff).
        max_retry_after: clamp on ``Retry-After``, so a hostile or buggy header
            cannot park the run for an hour.
        retry_transport_errors: whether connection/timeout failures are retried.
    """

    max_attempts: int = 4
    base_delay: float = 0.5
    multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.25
    retry_statuses: frozenset[int] = RETRYABLE_STATUSES
    respect_retry_after: bool = True
    max_retry_after: float = 60.0
    retry_transport_errors: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")

    def is_retryable(self, status: Optional[int]) -> bool:
        """True when *status* is in this policy's retryable set."""
        if status is None:
            return False
        return int(status) in self.retry_statuses

    def is_retryable_error(self, error: BaseException) -> bool:
        """True when *error* is a transport failure worth retrying."""
        if isinstance(error, TransportError):
            return self.retry_transport_errors and error.retryable
        if isinstance(error, (ConnectionFailure, TimeoutFailure)):  # pragma: no cover
            return self.retry_transport_errors
        return False

    def backoff(self, attempt: int, *, rng: Optional[random.Random] = None) -> float:
        """Delay in seconds after *attempt* (1-based) failed.

        The schedule is ``base_delay * multiplier ** (attempt - 1)``, clamped to
        ``max_delay``, plus up to ``jitter`` of that value. Jitter is added
        before the final clamp, so ``max_delay`` stays a real ceiling.
        """
        raw = self.base_delay * (self.multiplier ** max(0, int(attempt) - 1))
        delay = min(raw, self.max_delay)
        if self.jitter > 0:
            source = rng if rng is not None else random
            delay += delay * self.jitter * source.random()
            delay = min(delay, self.max_delay)
        return round(delay, 6)

    def delay_for(
        self,
        attempt: int,
        *,
        retry_after: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> float:
        """The delay to actually use, honouring ``Retry-After`` when it is longer."""
        delay = self.backoff(attempt, rng=rng)
        if retry_after is not None and self.respect_retry_after:
            delay = max(delay, min(float(retry_after), self.max_retry_after))
        return round(delay, 6)

    def next_attempts(self) -> int:
        """Attempts remaining after the first failure (``max_attempts - 1``)."""
        return max(0, self.max_attempts - 1)


def parse_retry_after(value: Optional[str], *, now: Optional[float] = None) -> Optional[float]:
    """Parse a ``Retry-After`` header into seconds.

    Accepts the delta-seconds form (``"120"``) and the HTTP-date form
    (``"Wed, 21 Oct 2015 07:28:00 GMT"``). Returns ``None`` when unparseable.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, parsed.timestamp() - (time.time() if now is None else now))


@dataclass(frozen=True)
class RateLimiterStats:
    """What the limiter actually did, for a run report.

    ``attempts`` counts every HTTP attempt, including attempts that failed at the
    transport level. ``responses`` counts only the attempts that produced an
    actual HTTP response, so the two differ exactly by the connection failures.
    """

    responses: int = 0
    attempts: int = 0
    retries: int = 0
    throttle_sleep: float = 0.0
    backoff_sleep: float = 0.0

    @property
    def sleep_total(self) -> float:
        """Total seconds spent waiting, pacing plus backoff."""
        return round(self.throttle_sleep + self.backoff_sleep, 6)

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, handy for logging."""
        return {
            "responses": self.responses,
            "attempts": self.attempts,
            "retries": self.retries,
            "throttle_sleep": round(self.throttle_sleep, 6),
            "backoff_sleep": round(self.backoff_sleep, 6),
            "sleep_total": self.sleep_total,
        }


class RateLimiter:
    """Enforce a request ceiling and retry transient failures with backoff.

    Args:
        max_requests_per_second: ceiling on request *starts*. ``None`` disables
            pacing (backoff still applies).
        min_interval: floor on the gap between request starts, if you prefer to
            express the ceiling as a delay.
        policy: retry/backoff rules.
        clock: monotonic time source (injectable for tests).
        sleeper: sleep function (injectable for tests).
        rng: random source used for jitter (injectable for repeatable tests).
        on_retry: optional callback receiving a :class:`RetryEvent` each time a
            retry is scheduled.
    """

    def __init__(
        self,
        *,
        max_requests_per_second: Optional[float] = None,
        min_interval: float = 0.0,
        policy: Optional[RetryPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
        on_retry: Optional[Callable[[RetryEvent], None]] = None,
    ) -> None:
        if max_requests_per_second is not None and max_requests_per_second <= 0:
            raise ValueError("max_requests_per_second must be > 0")
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = float(min_interval)
        self.policy = policy if policy is not None else RetryPolicy()
        self.clock = clock
        self.sleeper = sleeper
        self.rng = rng if rng is not None else random.Random()
        self.on_retry = on_retry

        if max_requests_per_second is not None:
            self.interval = max(self.min_interval, 1.0 / float(max_requests_per_second))
        else:
            self.interval = self.min_interval

        self._next_allowed = 0.0
        self._lock = threading.Lock()
        self._requests = 0
        self._attempts = 0
        self._retries = 0
        self._throttle_sleep = 0.0
        self._backoff_sleep = 0.0

    # ------------------------------------------------------------------ pacing
    def acquire(self) -> float:
        """Block until the next request is allowed; return seconds waited."""
        waited = 0.0
        with self._lock:
            if self.interval <= 0:
                return 0.0
            now = self.clock()
            target = max(now, self._next_allowed)
            wait = target - now
            self._next_allowed = target + self.interval
        if wait > 0:
            self.sleeper(wait)
            waited = wait
            self._throttle_sleep += waited
        return waited

    # ------------------------------------------------------------------- retry
    def should_retry(self, response: Response) -> bool:
        """True when *response*'s status is worth another attempt."""
        return self.policy.is_retryable(response.status)

    def run(self, fetch: Union[Fetch, Any], request: Request) -> Response:
        """Send *request* with pacing and bounded retries.

        Returns the last :class:`Response`, including a failure status once the
        attempts are used up (so the caller can classify it). Raises
        :class:`RetryExhausted` only when every attempt failed at the transport
        level and no response was ever produced.
        """
        send = as_fetch(fetch)
        attempt = 0
        last_error: Optional[BaseException] = None
        while attempt < self.policy.max_attempts:
            attempt += 1
            self.acquire()
            self._attempts += 1
            started = self.clock()
            try:
                response = send(request)
                last_error = None
            except TransportError as exc:
                last_error = exc
                if not self.policy.is_retryable_error(exc) or attempt >= self.policy.max_attempts:
                    raise RetryExhausted(
                        f"{request.method} {request.url} failed after {attempt} attempt(s): {exc}",
                        attempts=attempt,
                        last_error=exc,
                    ) from exc
                self._sleep_before_retry(
                    attempt,
                    RetryEvent(
                        url=request.url,
                        attempt=attempt,
                        max_attempts=self.policy.max_attempts,
                        delay=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                        reason="transport_error",
                    ),
                )
                continue

            self._requests += 1
            response = replace(
                response,
                attempts=attempt,
                elapsed=round(max(0.0, self.clock() - started), 6),
            )
            if not self.should_retry(response) or attempt >= self.policy.max_attempts:
                return response
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            self._sleep_before_retry(
                attempt,
                RetryEvent(
                    url=request.url,
                    attempt=attempt,
                    max_attempts=self.policy.max_attempts,
                    delay=0.0,
                    status=response.status,
                    reason=f"HTTP {response.status}",
                ),
                retry_after=retry_after,
            )

        if last_error is not None:  # pragma: no cover - loop always returns or raises
            raise RetryExhausted(
                f"{request.method} {request.url} failed after {attempt} attempt(s): {last_error}",
                attempts=attempt,
                last_error=last_error,
            ) from last_error
        raise RetryExhausted(  # pragma: no cover - defensive
            f"{request.method} {request.url} produced no response",
            attempts=attempt,
            last_error=TransportError("no response"),
        )

    def _sleep_before_retry(
        self,
        attempt: int,
        event: RetryEvent,
        *,
        retry_after: Optional[float] = None,
    ) -> None:
        delay = self.policy.delay_for(attempt, retry_after=retry_after, rng=self.rng)
        self._retries += 1
        final = replace(event, delay=delay)
        if self.on_retry is not None:
            self.on_retry(final)
        if delay > 0:
            self.sleeper(delay)
            self._backoff_sleep += delay

    # ------------------------------------------------------------------ report
    def stats(self) -> RateLimiterStats:
        """Counters for requests, attempts, retries and time spent waiting."""
        return RateLimiterStats(
            responses=self._requests,
            attempts=self._attempts,
            retries=self._retries,
            throttle_sleep=self._throttle_sleep,
            backoff_sleep=self._backoff_sleep,
        )

    def reset_stats(self) -> None:
        """Zero the counters without touching pacing state."""
        self._requests = 0
        self._attempts = 0
        self._retries = 0
        self._throttle_sleep = 0.0
        self._backoff_sleep = 0.0

    def run_all(
        self, fetch: Union[Fetch, Any], requests: Iterable[Request]
    ) -> list[Response]:
        """Convenience: :meth:`run` over several requests, in order."""
        return [self.run(fetch, request) for request in requests]

    def __enter__(self) -> "RateLimiter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def __repr__(self) -> str:
        ceiling = (
            "unlimited"
            if self.max_requests_per_second is None
            else f"{self.max_requests_per_second}/s"
        )
        return (
            f"<RateLimiter ceiling={ceiling} interval={self.interval:.3f}s "
            f"max_attempts={self.policy.max_attempts}>"
        )
