"""Retry and pacing behaviour, tested against the local fixture server.

The fixture is what makes these tests meaningful: ``429`` twice then ``200`` is
a real HTTP exchange with real socket-level success and failure, and ``403`` not
being retried is asserted by counting the requests that actually arrived.
"""

from __future__ import annotations

import random
import unittest

from scrape_session import (
    ConnectionFailure,
    RateLimiter,
    Request,
    Response,
    RetryExhausted,
    RetryPolicy,
    UrllibTransport,
    is_retryable_status,
)
from scrape_session.ratelimit import parse_retry_after

from .testserver import FixtureServer, Hit, Route, run_server

OK_BODY = b"payload"


class FakeClock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds

    @property
    def total(self) -> float:
        return round(sum(self.sleeps), 6)


class RetryPolicyTests(unittest.TestCase):
    def test_429_and_5xx_are_retryable_and_4xx_is_not(self) -> None:
        policy = RetryPolicy()
        for status in (408, 425, 429, 500, 502, 503, 504):
            self.assertTrue(policy.is_retryable(status), f"{status} should retry")
        for status in (400, 401, 403, 404, 409, 410, 422, 451):
            self.assertFalse(policy.is_retryable(status), f"{status} must not retry")

    def test_retryable_status_set_matches_helper(self) -> None:
        for status in range(100, 600):
            self.assertEqual(
                is_retryable_status(status), RetryPolicy().is_retryable(status)
            )

    def test_backoff_is_exponential_and_capped(self) -> None:
        policy = RetryPolicy(base_delay=0.1, multiplier=2.0, max_delay=1.0, jitter=0.0)
        self.assertEqual([policy.backoff(n) for n in (1, 2, 3, 4, 5)], [0.1, 0.2, 0.4, 0.8, 1.0])

    def test_jitter_stays_within_bounds_and_under_the_ceiling(self) -> None:
        policy = RetryPolicy(base_delay=0.5, multiplier=2.0, max_delay=2.0, jitter=0.5)
        rng = random.Random(1234)
        for attempt in range(1, 12):
            delay = policy.backoff(attempt, rng=rng)
            self.assertLessEqual(delay, 2.0)
            base = min(0.5 * 2 ** (attempt - 1), 2.0)
            self.assertGreaterEqual(delay, base)
            self.assertLessEqual(delay, base * 1.5 + 1e-9)

    def test_retry_after_is_respected_but_never_shortens_backoff(self) -> None:
        policy = RetryPolicy(base_delay=1.0, multiplier=2.0, jitter=0.0, max_retry_after=10.0)
        self.assertEqual(policy.delay_for(1, retry_after=0.1), 1.0)
        self.assertEqual(policy.delay_for(1, retry_after=5.0), 5.0)
        self.assertEqual(policy.delay_for(1, retry_after=600.0), 10.0)

    def test_invalid_policy_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=0)
        with self.assertRaises(ValueError):
            RetryPolicy(multiplier=0.5)
        with self.assertRaises(ValueError):
            RetryPolicy(base_delay=-1)

    def test_parse_retry_after_accepts_seconds_and_http_date(self) -> None:
        self.assertEqual(parse_retry_after("120"), 120.0)
        self.assertEqual(parse_retry_after(" 7 "), 7.0)
        self.assertIsNone(parse_retry_after("soon"))
        self.assertIsNone(parse_retry_after(""))
        self.assertIsNone(parse_retry_after(None))
        # The HTTP-date form is converted into a delay relative to "now".
        parsed = parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=1445411280.0)
        self.assertEqual(parsed, 1200.0)


class RateLimiterUnitTests(unittest.TestCase):
    def test_backoff_schedule_uses_injected_clock_exactly(self) -> None:
        clock = FakeClock()
        policy = RetryPolicy(max_attempts=4, base_delay=0.25, multiplier=2.0, jitter=0.0)
        limiter = RateLimiter(policy=policy, clock=clock, sleeper=clock.sleep)
        queue = [
            Response(429, {}, b"a"),
            Response(503, {}, b"b"),
            Response(200, {}, b"c"),
        ]

        def fetch(request: Request) -> Response:
            return queue.pop(0)

        response = limiter.run(fetch, Request("http://example.com/x"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.attempts, 3)
        self.assertEqual(clock.sleeps, [0.25, 0.5])
        self.assertEqual(
            limiter.stats().as_dict(),
            {
                "responses": 3,
                "attempts": 3,
                "retries": 2,
                "throttle_sleep": 0.0,
                "backoff_sleep": 0.75,
                "sleep_total": 0.75,
            },
        )

    def test_pacing_enforces_the_configured_ceiling(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(
            max_requests_per_second=4, policy=RetryPolicy(jitter=0.0), clock=clock, sleeper=clock.sleep
        )
        waited = [limiter.acquire() for _ in range(5)]
        # Four requests per second means 0.25s between starts; the first is free.
        self.assertEqual(waited, [0.0, 0.25, 0.25, 0.25, 0.25])
        self.assertEqual(clock.sleeps, [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(clock.total, 1.0)

    def test_min_interval_and_rate_cannot_be_configured_inconsistently(self) -> None:
        limiter = RateLimiter(max_requests_per_second=10, min_interval=0.5)
        self.assertEqual(limiter.interval, 0.5)  # the stricter of the two wins
        with self.assertRaises(ValueError):
            RateLimiter(max_requests_per_second=0)
        with self.assertRaises(ValueError):
            RateLimiter(min_interval=-1)

    def test_response_attempts_are_reported(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(
            policy=RetryPolicy(max_attempts=3, base_delay=1.0, jitter=0.0),
            clock=clock,
            sleeper=clock.sleep,
        )
        attempts = 0

        def fetch(request: Request) -> Response:
            nonlocal attempts
            attempts += 1
            return Response(429, {}, b"nope")

        response = limiter.run(fetch, Request("http://example.com/x"))
        self.assertEqual(response.status, 429)
        self.assertEqual(response.attempts, 3)
        self.assertEqual(attempts, 3)

    def test_transport_errors_are_retried_then_raised_as_retry_exhausted(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(
            policy=RetryPolicy(max_attempts=2, base_delay=0.1, jitter=0.0),
            clock=clock,
            sleeper=clock.sleep,
        )

        def fetch(request: Request) -> Response:
            raise ConnectionFailure("refused")

        with self.assertRaises(RetryExhausted) as caught:
            limiter.run(fetch, Request("http://example.com/x"))
        self.assertEqual(caught.exception.attempts, 2)
        self.assertIsInstance(caught.exception.last_error, ConnectionFailure)
        self.assertEqual(clock.sleeps, [0.1])

    def test_transport_errors_are_not_retried_when_disabled(self) -> None:
        calls = 0

        def fetch(request: Request) -> Response:
            nonlocal calls
            calls += 1
            raise ConnectionFailure("refused")

        limiter = RateLimiter(
            policy=RetryPolicy(max_attempts=3, retry_transport_errors=False, jitter=0.0),
            sleeper=lambda _: None,
        )
        with self.assertRaises(RetryExhausted):
            limiter.run(fetch, Request("http://example.com/x"))
        self.assertEqual(calls, 1)

    def test_on_retry_callback_reports_the_schedule(self) -> None:
        events = []
        limiter = RateLimiter(
            policy=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=0.0),
            sleeper=lambda _: None,
            on_retry=events.append,
        )

        def fetch(request: Request) -> Response:
            return Response(503, {}, b"")

        limiter.run(fetch, Request("http://example.com/x"))
        self.assertEqual([e.attempt for e in events], [1, 2])
        self.assertEqual([e.delay for e in events], [0.1, 0.2])
        self.assertEqual([e.status for e in events], [503, 503])
        self.assertIn("HTTP 503", str(events[0]))

    def test_acquire_is_skipped_when_no_ceiling_is_set(self) -> None:
        limiter = RateLimiter(sleeper=lambda _: None)
        self.assertEqual(limiter.acquire(), 0.0)
        self.assertEqual(limiter.stats().sleep_total, 0.0)

    def test_run_all_processes_requests_in_order(self) -> None:
        seen: list[str] = []

        def fetch(request: Request) -> Response:
            seen.append(request.url)
            return Response(200, {}, b"x")

        limiter = RateLimiter(policy=RetryPolicy(jitter=0.0), sleeper=lambda _: None)
        responses = limiter.run_all(
            fetch,
            [Request(f"http://example.com/{i}") for i in range(3)],
        )
        self.assertEqual(len(responses), 3)
        self.assertEqual(
            seen, ["http://example.com/0", "http://example.com/1", "http://example.com/2"]
        )


class RateLimiterAgainstServerTests(unittest.TestCase):
    """The same behaviour, but over real sockets on a local port."""

    def setUp(self) -> None:
        self.transport = UrllibTransport(timeout=5.0)

    def limiter(self, **kwargs) -> RateLimiter:
        kwargs.setdefault("policy", RetryPolicy(max_attempts=4, base_delay=0.01, jitter=0.0))
        kwargs.setdefault("sleeper", lambda _: None)
        return RateLimiter(**kwargs)

    def test_backoff_on_429_then_success(self) -> None:
        routes = {
            "/flaky": Route.sequence_of(
                Route(429, b"too many requests", "text/plain"),
                Route(429, b"too many requests", "text/plain"),
                Route(200, OK_BODY, "text/plain"),
            )
        }
        with run_server(routes) as server:
            limiter = self.limiter()
            response = limiter.run(self.transport, Request(server.url_for("/flaky")))
            self.assertEqual(response.status, 200)
            self.assertEqual(response.body, OK_BODY)
            self.assertEqual(response.attempts, 3)
            self.assertEqual(server.hits_for("/flaky"), 3)
            self.assertEqual(limiter.stats().retries, 2)

    def test_no_retry_on_403(self) -> None:
        with run_server({"/forbidden": Route(403, b"forbidden", "text/plain")}) as server:
            limiter = self.limiter()
            response = limiter.run(self.transport, Request(server.url_for("/forbidden")))
            self.assertEqual(response.status, 403)
            self.assertEqual(response.attempts, 1)
            self.assertEqual(server.hits_for("/forbidden"), 1, "403 must be sent once")
            self.assertEqual(limiter.stats().retries, 0)

    def test_no_retry_on_404(self) -> None:
        with run_server() as server:
            limiter = self.limiter()
            response = limiter.run(self.transport, Request(server.url_for("/missing")))
            self.assertEqual(response.status, 404)
            self.assertEqual(server.hits_for("/missing"), 1)

    def test_503_is_retried_until_attempts_are_exhausted(self) -> None:
        with run_server({"/down": Route(503, b"unavailable", "text/plain")}) as server:
            limiter = self.limiter()
            response = limiter.run(self.transport, Request(server.url_for("/down")))
            self.assertEqual(response.status, 503)
            self.assertEqual(response.attempts, 4)
            self.assertEqual(server.hits_for("/down"), 4)

    def test_connection_drop_before_headers_is_retried(self) -> None:
        with run_server({"/reset": Route(close_before_headers=True)}) as server:
            limiter = self.limiter()
            with self.assertRaises(RetryExhausted):
                limiter.run(self.transport, Request(server.url_for("/reset")))
            self.assertEqual(server.hits_for("/reset"), 4)

    def test_retry_after_header_paces_the_retry(self) -> None:
        clock = FakeClock()
        sleeps: list[float] = []
        limiter = RateLimiter(
            policy=RetryPolicy(max_attempts=2, base_delay=0.05, jitter=0.0),
            clock=clock,
            sleeper=lambda seconds: (sleeps.append(round(seconds, 6)), clock.sleep(seconds)),
        )
        with run_server(
            {
                "/slow-down": Route.sequence_of(
                    Route(429, b"nope", "text/plain", headers={"Retry-After": "2"}),
                    Route(200, OK_BODY, "text/plain"),
                )
            }
        ) as server:
            response = limiter.run(self.transport, Request(server.url_for("/slow-down")))
        self.assertEqual(response.status, 200)
        self.assertEqual(sleeps, [2.0], "the server's Retry-After must be honoured")

    def test_pacing_spaces_requests_apart_in_real_time(self) -> None:
        with run_server({"/tick": Route(200, OK_BODY, "text/plain")}) as server:
            limiter = RateLimiter(
                max_requests_per_second=20, policy=RetryPolicy(jitter=0.0)
            )
            for _ in range(3):
                limiter.run(self.transport, Request(server.url_for("/tick")))
            gaps = [hit.at for hit in server.requests_for("/tick")]
        self.assertEqual(len(gaps), 3)
        # 20 requests/second means a 0.05s floor between starts. Allow slack for
        # a loaded machine, but the ceiling must actually be enforced.
        self.assertGreaterEqual(gaps[1] - gaps[0], 0.04)
        self.assertGreaterEqual(gaps[2] - gaps[1], 0.04)

    def test_retry_after_header_from_the_server_is_parsed(self) -> None:
        with run_server(
            {"/h": Route(429, b"", "text/plain", headers={"Retry-After": "3"})}
        ) as server:
            response = self.transport.send(Request(server.url_for("/h")))
        self.assertEqual(response.headers.get("retry-after"), "3")
        self.assertEqual(parse_retry_after(response.headers.get("retry-after")), 3.0)

    def test_custom_handler_route_can_change_behaviour_mid_run(self) -> None:
        allowed = {"count": 0}

        def gate(hit: Hit) -> Route:
            allowed["count"] += 1
            if allowed["count"] < 2:
                return Route(429, b"wait", "text/plain")
            return Route(200, OK_BODY, "text/plain")

        with run_server({"/gate": gate}) as server:
            limiter = self.limiter()
            response = limiter.run(self.transport, Request(server.url_for("/gate")))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.attempts, 2)
        self.assertEqual(allowed["count"], 2)


class LimiterFixtureSanityTests(unittest.TestCase):
    """Guards on the fixture itself: a broken fixture would fake these results."""

    def test_server_serves_on_an_ephemeral_loopback_port(self) -> None:
        with run_server({"/x": Route(200, b"hello", "text/plain")}) as server:
            self.assertGreater(server.port, 0)
            self.assertTrue(server.url.startswith("http://127.0.0.1:"))
            response = UrllibTransport(timeout=5).send(Request(server.url_for("/x")))
            self.assertEqual(response.body, b"hello")

    def test_hit_log_records_method_headers_and_query(self) -> None:
        with run_server({"/log": Route(200, b"ok", "text/plain")}) as server:
            UrllibTransport(timeout=5).send(
                Request(server.url_for("/log", page=2), headers={"X-Test": "yes"})
            )
            hits = server.requests_for("/log")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].method, "GET")
        self.assertEqual(hits[0].header("x-test"), "yes")
        self.assertEqual(hits[0].param("page"), "2")

    def test_unknown_path_is_a_404_without_touching_the_network(self) -> None:
        with run_server() as server:
            response = UrllibTransport(timeout=5).send(Request(server.url_for("/nope")))
            self.assertEqual(response.status, 404)
            self.assertEqual(response.body, b"not found")

    def test_fixture_reports_itself_when_stopped(self) -> None:
        server = FixtureServer()
        self.assertIn("stopped", repr(server))
        with self.assertRaises(RuntimeError):
            _ = server.port


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
