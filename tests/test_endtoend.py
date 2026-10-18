"""All four primitives driving one job, against the local fixture server.

Nothing here talks to a real site. The fixture plays a session-authenticated app
that hands out PDFs by id, throttles, denies one item outright, serves a login
page where a document should be, and truncates one body on the way out.

The point of these tests is the interaction: a resumed run must not re-fetch what
is done, a corrupt artefact must not be saved, and the same job must run
unchanged over a transport that is not urllib.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from scrape_session import (
    IntegrityCheck,
    ItemStatus,
    RateLimiter,
    Request,
    Response,
    ResumeManifest,
    RetryExhausted,
    RetryPolicy,
    SessionKeeper,
    UrllibTransport,
    as_fetch,
)

from .testserver import Hit, Route, make_pdf, run_server

LOGIN_PAGE = (
    b"<!DOCTYPE html><html><head><title>Sign in</title></head><body>"
    b"Your session has expired. Please sign in again to download this document."
    b"</body></html>"
)


@dataclass
class JobResult:
    """What one pass of the job produced."""

    processed: list[str]
    saved: list[str]
    exceptions: list[str]
    failed_with: str | None = None


class PdfJob:
    """The composition the README describes, in about forty lines.

    Order matters: the rate limiter wraps the session-bound fetch, so the retry
    budget is spent inside the session that is kept alive, and the integrity
    check runs on whatever comes back -- before anything is written.
    """

    def __init__(
        self,
        *,
        transport,
        session: SessionKeeper,
        manifest: ResumeManifest,
        limiter: RateLimiter,
        check: IntegrityCheck,
        out_dir: Path,
    ) -> None:
        self.session = session
        self.manifest = manifest
        self.limiter = limiter
        self.check = check
        self.out_dir = Path(out_dir)
        self.fetch = session.bind(transport)

    def item_url(self, item_id: str) -> str:
        """Where the document for *item_id* lives (overridden by the fixture base)."""
        return f"/item/{item_id}.pdf"

    def run(self, item_ids: list[str], *, stop_after: int | None = None) -> JobResult:
        """Process the items that are not already done.

        ``stop_after`` simulates the run dying: the manifest already holds every
        completed item, so the next pass picks up where this one stopped.
        """
        base = self.base_url
        todo = self.manifest.pending(item_ids)
        result = JobResult(processed=[], saved=[], exceptions=[])
        for index, item_id in enumerate(todo):
            if stop_after is not None and index >= stop_after:
                result.failed_with = f"simulated crash after {index} item(s)"
                return result
            record = self.fetch_item(item_id, f"{base}{self.item_url(item_id)}")
            result.processed.append(item_id)
            if record.status == ItemStatus.OK.value:
                result.saved.append(item_id)
            elif record.status in (ItemStatus.CORRUPTED.value, ItemStatus.PERMANENT_FAILURE.value):
                result.exceptions.append(item_id)
        return result

    base_url = ""

    def fetch_item(self, item_id: str, url: str):
        """Fetch, judge, and only then write; returns the manifest record."""
        import os as _os

        try:
            response = self.limiter.run(self.fetch, Request(url))
        except RetryExhausted as exc:
            return self.manifest.mark(
                item_id,
                ItemStatus.RETRYABLE_FAILURE,
                attempts=exc.attempts,
                reason=f"transport_error:{type(exc.last_error).__name__}",
                url=url,
            )

        assessment = self.check.assess(response)
        if not assessment.ok:
            # Nothing is written for a failed or corrupt artefact. The record
            # carries the evidence instead, and the item lands in the
            # exception list rather than looking like a success.
            return self.manifest.mark(
                item_id,
                assessment.verdict.manifest_status,
                checksum=assessment.checksum,
                size=assessment.size,
                attempts=response.attempts,
                reason=assessment.reason,
                url=url,
            )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        target = self.out_dir / f"{item_id}.pdf"
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(response.body)
        # Verify what actually reached the disk before calling it done.
        on_disk = tmp.read_bytes()
        digest = "sha256:" + hashlib.sha256(on_disk).hexdigest()
        if digest != assessment.checksum or len(on_disk) != assessment.size:
            tmp.unlink(missing_ok=True)
            return self.manifest.mark(
                item_id,
                ItemStatus.CORRUPTED,
                attempts=response.attempts,
                reason="write_verification_failed",
                url=url,
            )
        _os.replace(tmp, target)
        return self.manifest.mark(
            item_id,
            ItemStatus.OK,
            checksum=digest,
            size=len(on_disk),
            path=target,
            attempts=response.attempts,
            url=url,
        )


class JobHarness:
    """Builds a :class:`PdfJob` wired to a fixture server, with a temp workdir."""

    def __init__(self, test: unittest.TestCase, routes: dict, items: list[str]) -> None:
        self.items = items
        self._tmp = TemporaryDirectory()
        test.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        self.state = {"logins": 0, "valid": set(), "hits": {}}

        def login(hit: Hit) -> Route:
            self.state["logins"] += 1
            token = f"tok-{self.state['logins']}"
            self.state["valid"].add(token)
            return Route(
                200,
                b"signed in",
                "text/plain",
                headers={"Set-Cookie": f"sid={token}; Path=/; HttpOnly"},
            )

        self._server_cm = run_server({"/login": login, **routes})
        self.server = self._server_cm.__enter__()
        # Every document route sits behind the session, like the real app.
        for path, original in routes.items():
            self.server.set_route(path, gated(original, self.state["valid"]))

    def close(self) -> None:
        self._server_cm.__exit__(None, None, None)

    def transport(self) -> UrllibTransport:
        return UrllibTransport(timeout=5.0)

    def session(self, transport, **kwargs) -> SessionKeeper:
        def sign_in(keeper: SessionKeeper) -> None:
            response = transport.send(
                Request(f"{self.server.url}/login", method="POST", body=b"user=demo")
            )
            if response.status != 200:
                raise RuntimeError(f"login failed: HTTP {response.status}")
            keeper.capture(response)

        return SessionKeeper(
            self.workdir / "session.json",
            authenticate=sign_in,
            transport=transport,
            **kwargs,
        )

    def job(
        self,
        transport=None,
        *,
        check: IntegrityCheck | None = None,
        limiter: RateLimiter | None = None,
        **session_kwargs,
    ) -> PdfJob:
        transport = transport or self.transport()
        session = self.session(transport, **session_kwargs)
        manifest = ResumeManifest(self.workdir / "manifest.jsonl", fsync=False)
        limiter = limiter or RateLimiter(
            policy=RetryPolicy(max_attempts=4, base_delay=0.01, jitter=0.0),
            sleeper=lambda _: None,
        )
        job = PdfJob(
            transport=transport,
            session=session,
            manifest=manifest,
            limiter=limiter,
            check=check or IntegrityCheck.for_pdf(min_bytes=512),
            out_dir=self.workdir / "out",
        )
        job.base_url = self.server.url
        self.session_keeper = session
        self.manifest = manifest
        return job


def gated(route, valid_tokens) -> "callable":
    """Wrap *route* so it only answers requests carrying a live session cookie.

    ``route`` may be a :class:`Route`, a handler callable, or a sequence; the
    wrapper resolves it rather than returning it, so a sequence route still
    replays 429-then-200 instead of collapsing to its default.
    """

    def handler(hit: Hit) -> Route:
        if hit.cookie("sid") not in valid_tokens:
            return Route(401, b"unauthorized", "text/plain")
        if callable(route):
            return route(hit)
        return route.resolve(hit)

    return handler


def document_routes(
    items: list[str],
    *,
    truncate: set[str] = (),
    login_page: set[str] = (),
    denied: set[str] = (),
    flaky: set[str] = (),
) -> dict:
    """Build one route per item id plus the interesting failure modes."""
    routes: dict = {}
    for item in items:
        body = make_pdf(2048, marker=f"%PDF-1.7 {item}".encode())
        if item in denied:
            routes[f"/item/{item}.pdf"] = Route(403, b"forbidden", "text/plain")
        elif item in login_page:
            routes[f"/item/{item}.pdf"] = Route(200, LOGIN_PAGE, "text/html; charset=utf-8")
        elif item in truncate:
            routes[f"/item/{item}.pdf"] = Route(
                200, body, "application/pdf", declared_length=len(body), truncate_to=800
            )
        elif item in flaky:
            routes[f"/item/{item}.pdf"] = Route.sequence_of(
                Route(429, b"slow down", "text/plain"),
                Route(200, body, "application/pdf"),
            )
        else:
            routes[f"/item/{item}.pdf"] = Route(200, body, "application/pdf")
    return routes


class FullRunTests(unittest.TestCase):
    def test_a_clean_run_saves_every_item_with_a_verified_checksum(self) -> None:
        items = [f"d{i:02d}" for i in range(6)]
        harness = JobHarness(self, document_routes(items), items)
        self.addCleanup(harness.close)
        job = harness.job()
        result = job.run(items)

        self.assertEqual(result.processed, items)
        self.assertEqual(result.saved, items)
        self.assertEqual(result.exceptions, [])
        self.assertEqual(harness.server.hits_for("/login"), 1)
        for item in items:
            target = harness.workdir / "out" / f"{item}.pdf"
            self.assertTrue(target.exists(), item)
            body = target.read_bytes()
            self.assertEqual(job.manifest.get(item).size, len(body))
            self.assertEqual(
                job.manifest.get(item).checksum,
                "sha256:" + hashlib.sha256(body).hexdigest(),
            )
        self.assertEqual(job.manifest.status_counts(), {"ok": len(items)})
        self.assertEqual(job.manifest.total_bytes(), 6 * 2048)
        self.assertTrue(job.manifest.recovery.clean)

    def test_corrupt_and_denied_items_go_to_the_exception_list_and_are_not_saved(self) -> None:
        items = [f"d{i:02d}" for i in range(5)]
        harness = JobHarness(
            self,
            document_routes(items, truncate={"d01"}, login_page={"d02"}, denied={"d03"}),
            items,
        )
        self.addCleanup(harness.close)
        job = harness.job()
        result = job.run(items)

        self.assertEqual(result.saved, ["d00", "d04"])
        self.assertEqual(result.exceptions, ["d01", "d02", "d03"])
        for item in ("d01", "d02", "d03"):
            self.assertFalse(
                (harness.workdir / "out" / f"{item}.pdf").exists(),
                f"{item} produced untrustworthy bytes and must not be on disk",
            )
        self.assertEqual(
            job.manifest.status_of("d01"), ItemStatus.CORRUPTED.value
        )
        self.assertEqual(job.manifest.get("d01").reason, "truncated_body")
        self.assertEqual(
            job.manifest.status_of("d02"), ItemStatus.CORRUPTED.value
        )
        self.assertEqual(job.manifest.status_of("d03"), ItemStatus.PERMANENT_FAILURE.value)
        self.assertEqual(
            [r.item_id for r in job.manifest.exceptions()], ["d01", "d02", "d03"]
        )
        report = job.manifest.write_exception_report(harness.workdir / "exceptions.json")
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["exceptions"]), 3)

    def test_a_429_is_retried_and_the_item_still_saves(self) -> None:
        items = ["d00", "d01", "d02"]
        harness = JobHarness(self, document_routes(items, flaky={"d01"}), items)
        self.addCleanup(harness.close)
        job = harness.job()
        result = job.run(items)
        self.assertEqual(result.saved, items)
        self.assertEqual(harness.server.hits_for("/item/d01.pdf"), 2)
        self.assertEqual(job.manifest.get("d01").attempts, 2)
        self.assertEqual(job.manifest.get("d00").attempts, 1)
        self.assertEqual(job.limiter.stats().retries, 1)

    def test_a_403_costs_exactly_one_request(self) -> None:
        items = ["d00", "d01"]
        harness = JobHarness(self, document_routes(items, denied={"d01"}), items)
        self.addCleanup(harness.close)
        job = harness.job()
        job.run(items)
        self.assertEqual(harness.server.hits_for("/item/d01.pdf"), 1)
        self.assertEqual(job.manifest.get("d01").attempts, 1)
        self.assertEqual(job.limiter.stats().retries, 0)


class ResumeTests(unittest.TestCase):
    def test_a_second_run_skips_completed_items(self) -> None:
        items = [f"d{i:02d}" for i in range(10)]
        harness = JobHarness(self, document_routes(items), items)
        self.addCleanup(harness.close)

        first = harness.job()
        first_run = first.run(items, stop_after=4)
        self.assertEqual(first_run.saved, items[:4])
        self.assertIsNotNone(first_run.failed_with)
        self.assertEqual(len(first.manifest.completed_ids()), 4)

        second = harness.job(check=first.check)
        second_run = second.run(items)
        self.assertEqual(second_run.processed, items[4:])
        self.assertEqual(second_run.saved, items[4:])
        for item in items:
            self.assertEqual(
                harness.server.hits_for(f"/item/{item}.pdf"),
                1,
                f"{item} must be fetched exactly once across both runs",
            )
        self.assertEqual(second.manifest.completed_ids(), items)
        self.assertEqual(harness.server.hits_for("/login"), 1)

    def test_a_crashed_run_resumes_and_never_refetches_finished_items(self) -> None:
        items = [f"d{i:02d}" for i in range(8)]
        harness = JobHarness(self, document_routes(items), items)
        self.addCleanup(harness.close)

        first = harness.job()
        with self.assertRaises(KeyboardInterrupt):
            # A hard stop in the middle of the loop, exactly like a ctrl-C or a
            # killed process: whatever completed is already on disk.
            try:
                for index, item in enumerate(items):
                    if index == 3:
                        raise KeyboardInterrupt
                    first.fetch_item(item, f"{harness.server.url}/item/{item}.pdf")
            finally:
                first.manifest.close()

        resumed = harness.job()
        result = resumed.run(items)
        self.assertEqual(result.processed, items[3:])
        self.assertEqual(resumed.manifest.completed_ids(), items)
        for item in items:
            self.assertEqual(harness.server.hits_for(f"/item/{item}.pdf"), 1)

    def test_corrupt_items_are_retried_on_the_next_run_not_marked_done(self) -> None:
        items = ["d00", "d01"]
        routing = document_routes(items, truncate={"d01"})
        harness = JobHarness(self, routing, items)
        self.addCleanup(harness.close)

        first = harness.job()
        self.assertEqual(first.run(items).exceptions, ["d01"])
        self.assertEqual(harness.server.hits_for("/item/d01.pdf"), 1)

        # The fixture is fixed between runs, as it would be in real life.
        body = make_pdf(2048, marker=b"%PDF-1.7 d01")
        harness.server.set_route("/item/d01.pdf", Route(200, body, "application/pdf"))

        second = harness.job()
        result = second.run(items)
        self.assertEqual(result.saved, ["d01"])
        self.assertEqual(result.exceptions, [])
        self.assertEqual(harness.server.hits_for("/item/d00.pdf"), 1, "d00 stayed done")
        self.assertEqual(second.manifest.status_counts(), {"ok": 2})


class TransportAgnosticTests(unittest.TestCase):
    """The primitives never import a transport, so any callable will do."""

    def canned(self, payloads: dict[str, bytes]) -> object:
        class FakeBrowser:
            """Stands in for a browser driver: returns Responses, opens no sockets."""

            def __init__(self) -> None:
                self.calls: list[str] = []

            def __call__(self, request: Request) -> Response:
                self.calls.append(request.url)
                if request.url not in payloads:
                    return Response(404, {"content-type": "text/plain"}, b"missing")
                body = payloads[request.url]
                return Response(
                    200,
                    {"content-type": "application/pdf", "content-length": str(len(body))},
                    body,
                )

        return FakeBrowser()

    def test_a_plain_callable_transport_drives_the_whole_job(self) -> None:
        items = ["a", "b", "c"]
        payloads = {f"https://example.com/item/{i}.pdf": make_pdf(2048) for i in items}
        transport = self.canned(payloads)
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            manifest = ResumeManifest(workdir / "m.jsonl", fsync=False)
            limiter = RateLimiter(policy=RetryPolicy(jitter=0.0), sleeper=lambda _: None)
            session = SessionKeeper(
                workdir / "s.json",
                # A browser driver may own the real session; here the login
                # "callback" just records a header.
                authenticate=lambda keeper: keeper.set_header("X-Driver", "fake-browser"),
            )
            job = PdfJob(
                transport=transport,
                session=session,
                manifest=manifest,
                limiter=limiter,
                check=IntegrityCheck.for_pdf(min_bytes=512),
                out_dir=workdir / "out",
            )
            job.base_url = "https://example.com"
            result = job.run(items)
            self.assertEqual(result.saved, items)
            self.assertEqual(len(transport.calls), 3)
            self.assertEqual(manifest.completed_ids(), items)

    def test_as_fetch_accepts_a_callable_a_send_method_and_duck_types(self) -> None:
        def callable_transport(request: Request) -> Response:
            return Response(200, {}, b"callable")

        class SendOnly:
            def send(self, request: Request) -> Response:
                return Response(200, {}, b"send")

        class CallableObject:
            def __call__(self, request: Request) -> Response:
                return Response(200, {}, b"object")

        for transport, expected in (
            (callable_transport, b"callable"),
            (SendOnly(), b"send"),
            (CallableObject(), b"object"),
        ):
            response = as_fetch(transport)(Request("https://example.com/"))
            self.assertEqual(response.body, expected)
        with self.assertRaises(TypeError):
            as_fetch(object())

    def test_a_browser_style_transport_works_with_the_rate_limiter(self) -> None:
        responses = [
            Response(429, {"content-type": "text/plain"}, b"slow down"),
            Response(200, {"content-type": "application/pdf"}, make_pdf(2048)),
        ]
        calls: list[int] = []

        def browser(request: Request) -> Response:
            calls.append(1)
            return responses[min(len(calls) - 1, 1)]

        limiter = RateLimiter(
            policy=RetryPolicy(max_attempts=3, base_delay=0.01, jitter=0.0),
            sleeper=lambda _: None,
        )
        response = limiter.run(browser, Request("https://example.com/item/1.pdf"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.attempts, 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(IntegrityCheck.for_pdf(min_bytes=512).is_ok(response))


class PaginatedDirectoryTests(unittest.TestCase):
    """The second shape of job: a paginated listing with click-through detail pages.

    The listing pages are HTML with ids in them; the "parser" here is a regex,
    because this package does not parse anything -- it fetches reliably and
    remembers what it already has.
    """

    def test_listing_then_detail_pages_resume_correctly(self) -> None:
        import re

        total_pages = 3
        per_page = 2
        all_ids = [f"e{i:02d}" for i in range(total_pages * per_page)]

        def listing(page: int) -> bytes:
            start = page * per_page
            chunk = all_ids[start : start + per_page]
            rows = "".join(
                f'<a class="row" href="/detail/{i}">{i}</a>' for i in chunk
            )
            return f"<html><body><h1>Page {page + 1}</h1>{rows}</body></html>".encode()

        def list_route(hit: Hit) -> Route:
            page = int(hit.param("page", "0") or 0)
            return Route(200, listing(page), "text/html; charset=utf-8")

        routes: dict = {"/list": list_route}
        for i in all_ids:
            routes[f"/detail/{i}"] = Route(
                200, make_pdf(1024, marker=f"%PDF-1.7 {i}".encode()), "application/pdf"
            )

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        workdir = Path(tmp.name)
        transport = UrllibTransport(timeout=5.0)
        state = {"logins": 0, "valid": set()}

        def login(hit: Hit) -> Route:
            state["logins"] += 1
            token = f"t{state['logins']}"
            state["valid"].add(token)
            return Route(200, b"ok", "text/plain", headers={"Set-Cookie": f"sid={token}; Path=/"})

        with run_server({"/login": login, **routes}) as server:
            for path, original in routes.items():
                server.set_route(path, gated(original, state["valid"]))

            def sign_in(keeper: SessionKeeper) -> None:
                response = transport.send(
                    Request(f"{server.url}/login", method="POST", body=b"u=d")
                )
                keeper.capture(response)

            session = SessionKeeper(workdir / "session.json", authenticate=sign_in)
            manifest = ResumeManifest(workdir / "manifest.jsonl", fsync=False)
            limiter = RateLimiter(
                max_requests_per_second=1000,
                policy=RetryPolicy(max_attempts=3, base_delay=0.01, jitter=0.0),
                sleeper=lambda _: None,
            )
            check_html = IntegrityCheck.for_html(min_bytes=16)
            check_pdf = IntegrityCheck.for_pdf(min_bytes=512)
            fetch = session.bind(transport)

            # Pass 1: read the listing pages, skip anything already done.
            discovered: list[str] = []
            for page in range(total_pages):
                response = limiter.run(fetch, Request(server.url_for("/list", page=page)))
                assessment = check_html.assess(response)
                self.assertTrue(assessment.ok, assessment.description)
                discovered.extend(
                    re.findall(r'href="/detail/([^"]+)"', response.text)
                )
            self.assertEqual(discovered, all_ids)

            for item in manifest.pending(discovered):
                response = limiter.run(fetch, Request(f"{server.url}/detail/{item}"))
                assessment = check_pdf.assess(response)
                if not assessment.ok:
                    manifest.mark(item, assessment.verdict.manifest_status, reason=assessment.reason)
                    continue
                target = workdir / "out" / f"{item}.pdf"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.body)
                manifest.mark(
                    item,
                    ItemStatus.OK,
                    checksum=assessment.checksum,
                    size=assessment.size,
                    path=target,
                )

            self.assertEqual(manifest.completed_ids(), all_ids)
            self.assertEqual(manifest.status_counts(), {"ok": len(all_ids)})
            self.assertEqual(state["logins"], 1, "one sign-in for the whole crawl")
            self.assertEqual(server.hits_for("/login"), 1)

            # Pass 2: nothing left to do, and the server sees no new requests.
            before = server.total_hits()
            self.assertEqual(manifest.pending(discovered), [])
            self.assertEqual(
                server.total_hits(), before, "a resumed run must not re-fetch"
            )
            manifest.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
