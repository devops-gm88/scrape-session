#!/usr/bin/env python3
"""Run all four primitives against a local server. No network, no install.

    python examples/demo.py

The script starts its own HTTP server on an ephemeral loopback port and plays a
session-authenticated document app: it hands out PDFs by id, rate limits, denies
one item, truncates one body, and serves a login page where a document should be.
Then it runs a 12-item job through ``SessionKeeper``, ``ResumeManifest``,
``RateLimiter`` and ``IntegrityCheck`` -- including a first pass that "dies"
part way through, so the second pass has to resume.

Everything printed is measured during the run; nothing is estimated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional
from urllib.parse import urlparse

# Runnable straight from a clone, before any install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scrape_session import (  # noqa: E402
    IntegrityCheck,
    ItemStatus,
    RateLimiter,
    Request,
    Response,
    ResumeManifest,
    RetryEvent,
    RetryExhausted,
    RetryPolicy,
    SessionKeeper,
    UrllibTransport,
)

# --------------------------------------------------------------------------- #
# A tiny local site to fetch from
# --------------------------------------------------------------------------- #


class Reply(NamedTuple):
    """What the demo server should send."""

    status: int = 200
    body: bytes = b""
    content_type: str = "application/octet-stream"
    headers: dict | None = None
    truncate_to: Optional[int] = None


def pdf_bytes(item: str, size: int = 2048) -> bytes:
    """Plausible PDF bytes of exactly *size*, with the item id inside."""
    head = f"%PDF-1.7 demo document {item}".encode()
    tail = b"\n%%EOF\n"
    return head + b"x" * (size - len(head) - len(tail)) + tail


LOGIN_PAGE = (
    b"<!DOCTYPE html><html><head><title>Sign in</title></head><body>"
    b"Your session has expired. Please sign in again.</body></html>"
)


class DemoSite:
    """The behaviour of each item, plus the counters that prove what happened."""

    def __init__(self, items: list[str], behaviour: dict[str, str]) -> None:
        self.behaviour = behaviour
        self.items = items
        self.logins = 0
        self.hits: dict[str, int] = {}
        self.valid_tokens: set[str] = set()
        self._lock = threading.Lock()

    def record_hit(self, path: str) -> int:
        """Count a request and return its 0-based index for that path."""
        with self._lock:
            self.hits[path] = self.hits.get(path, 0) + 1
            return self.hits[path]

    def login(self) -> Reply:
        with self._lock:
            self.logins += 1
            token = f"session-{self.logins}"
            self.valid_tokens.add(token)
        return Reply(
            200,
            b"signed in",
            "text/plain",
            {"Set-Cookie": f"sid={token}; Path=/; HttpOnly", "X-Login-Number": str(self.logins)},
        )

    def document(self, item: str, hit_index: int, cookie: Optional[str]) -> Reply:
        if cookie not in self.valid_tokens:
            return Reply(401, b"unauthorized", "text/plain")
        kind = self.behaviour.get(item, "ok")
        if kind == "denied":
            return Reply(403, b"forbidden: not entitled to this document", "text/plain")
        if kind == "login_page":
            return Reply(200, LOGIN_PAGE, "text/html; charset=utf-8")
        if kind == "truncated":
            body = pdf_bytes(item)
            return Reply(
                200,
                body,
                "application/pdf",
                headers={"Content-Length": str(len(body))},
                truncate_to=900,
            )
        if kind == "flaky" and hit_index == 0:
            return Reply(429, b"slow down", "text/plain", {"Retry-After": "1"})
        if kind == "flaky" and hit_index == 1:
            return Reply(429, b"slow down", "text/plain", {"Retry-After": "1"})
        return Reply(200, pdf_bytes(item), "application/pdf")


class DemoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "scrape-session-demo/0.1"

    def log_message(self, *args: Any) -> None:
        return None

    def do_GET(self) -> None:  # noqa: N802 - required by the base class
        self._respond(self._dispatch("GET"))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self._respond(self._dispatch("POST"))

    def _dispatch(self, method: str) -> Reply:
        site: DemoSite = self.server.site  # type: ignore[attr-defined]
        parsed = urlparse(self.path)
        path = parsed.path
        cookie = None
        for chunk in (self.headers.get("Cookie") or "").split(";"):
            key, _, value = chunk.strip().partition("=")
            if key == "sid":
                cookie = value
        index = site.record_hit(path) - 1

        if path == "/login" and method == "POST":
            return site.login()
        if path.startswith("/item/") and path.endswith(".pdf"):
            item = path[len("/item/") : -len(".pdf")]
            return site.document(item, index, cookie)
        return Reply(404, b"not found", "text/plain")

    def _respond(self, reply: Reply) -> None:
        body = reply.body
        declared = len(body)
        headers = dict(reply.headers or {})
        if "Content-Length" in headers:
            declared = int(headers.pop("Content-Length"))
        self.send_response(reply.status)
        self.send_header("Content-Type", reply.content_type)
        for name, value in headers.items():
            self.send_header(name, value)
        if reply.status not in (204, 304):
            self.send_header("Content-Length", str(declared))
        self.end_headers()
        if reply.truncate_to is not None:
            self.wfile.write(body[: reply.truncate_to])
            self.wfile.flush()
            self.close_connection = True
            try:
                self.connection.shutdown(2)
            except OSError:
                pass
            self.connection.close()
            return
        if body:
            self.wfile.write(body)
        self.wfile.flush()


@contextmanager
def local_site(site: DemoSite) -> Iterator[str]:
    """Serve *site* on an ephemeral loopback port; yields the base URL."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    httpd.daemon_threads = True
    httpd.site = site  # type: ignore[attr-defined]
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# Demo plumbing
# --------------------------------------------------------------------------- #


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def note(message: str) -> None:
    print(f"    {message}")


def compose(base_url: str, workdir: Path, *, pacing: float, on_retry: Callable[[RetryEvent], None]):
    """Wire the four primitives together for a PDF job.

    This is the whole composition, and it is the same shape for a paginated
    listing: the limiter wraps the session-bound fetch, and the integrity check
    runs before anything is written.
    """
    transport = UrllibTransport(timeout=10.0)
    session = SessionKeeper(
        workdir / "session.json",
        transport=transport,
        ttl_seconds=None,
    )

    def sign_in(keeper: SessionKeeper) -> None:
        """The caller owns the login: sign in, then hand the result to the keeper."""
        response = transport.send(
            Request(
                f"{base_url}/login",
                method="POST",
                body=b"username=demo&password=demo",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        )
        if response.status != 200:
            raise RuntimeError(f"login failed with HTTP {response.status}")
        keeper.capture(response)

    session.authenticate = sign_in
    manifest = ResumeManifest(workdir / "manifest.jsonl")
    limiter = RateLimiter(
        max_requests_per_second=pacing,
        policy=RetryPolicy(max_attempts=4, base_delay=0.2, multiplier=2.0, jitter=0.0),
        on_retry=on_retry,
    )
    check = IntegrityCheck.for_pdf(min_bytes=1024, max_bytes=8 * 1024 * 1024)
    return transport, session, manifest, limiter, check


def fetch_one(
    *,
    item: str,
    url: str,
    fetch,
    limiter: RateLimiter,
    check: IntegrityCheck,
    manifest: ResumeManifest,
    out_dir: Path,
    verbose: bool,
):
    """Fetch one item, judge it, and only then write it.

    Returns the manifest record. A corrupted artefact is never written; it is
    recorded as an exception instead.
    """
    try:
        response = limiter.run(fetch, Request(url))
    except RetryExhausted as exc:
        reason = f"transport_error:{type(exc.last_error).__name__}"
        note(f"{item}: gave up after {exc.attempts} attempt(s) ({reason})")
        return manifest.mark(item, ItemStatus.RETRYABLE_FAILURE, attempts=exc.attempts, reason=reason, url=url)

    assessment = check.assess(response)
    if not assessment.ok:
        marker = "CORRUPTED" if assessment.corrupted else "FAILED"
        note(
            f"{item}: {marker} after {response.attempts} attempt(s) -> "
            f"{assessment.reason} ({assessment.detail or 'no detail'})"
        )
        return manifest.mark(
            item,
            assessment.verdict.manifest_status,
            checksum=assessment.checksum,
            size=assessment.size,
            attempts=response.attempts,
            reason=assessment.reason,
            url=url,
            meta={"detail": assessment.detail} if assessment.detail else None,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{item}.pdf"
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(response.body)
    on_disk = tmp.read_bytes()
    digest = "sha256:" + hashlib.sha256(on_disk).hexdigest()
    if digest != assessment.checksum or len(on_disk) != assessment.size:
        tmp.unlink(missing_ok=True)
        note(f"{item}: write verification failed, nothing was saved")
        return manifest.mark(item, ItemStatus.CORRUPTED, reason="write_verification_failed", url=url)
    os.replace(tmp, target)
    if verbose:
        note(f"{item}: ok {len(on_disk)}B {digest[:19]}... -> {target.name}")
    return manifest.mark(
        item,
        ItemStatus.OK,
        checksum=digest,
        size=len(on_disk),
        path=target,
        attempts=response.attempts,
        url=url,
    )


def process(
    items: list[str],
    *,
    base_url: str,
    fetch,
    limiter: RateLimiter,
    check: IntegrityCheck,
    manifest: ResumeManifest,
    out_dir: Path,
    stop_after: Optional[int],
    verbose: bool,
) -> int:
    """Run the pending items; ``stop_after`` simulates the process dying."""
    todo = manifest.pending(items)
    for index, item in enumerate(todo):
        if stop_after is not None and index >= stop_after:
            note(f"!! simulated crash: stopping after {index} item(s) in this pass")
            return index
        fetch_one(
            item=item,
            url=f"{base_url}/item/{item}.pdf",
            fetch=fetch,
            limiter=limiter,
            check=check,
            manifest=manifest,
            out_dir=out_dir,
            verbose=verbose,
        )
    return len(todo)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="where to write session/manifest/output (default: a temp dir)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the work directory instead of deleting it on exit",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args(argv)
    verbose = not args.quiet

    items = [f"doc-{i:02d}" for i in range(1, 13)]
    behaviour = {
        "doc-03": "flaky",  # 429, 429, then 200
        "doc-08": "denied",  # 403: not retried
        "doc-09": "truncated",  # 200 with half the declared bytes
        "doc-10": "login_page",  # 200 text/html where a PDF belongs
    }
    site = DemoSite(items, behaviour)

    workdir = args.workdir
    temporary = workdir is None
    if temporary:
        workdir = Path(tempfile.mkdtemp(prefix="scrape-session-demo-"))
    assert workdir is not None
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "out"

    retries: list[RetryEvent] = []

    def on_retry(event: RetryEvent) -> None:
        retries.append(event)
        if verbose:
            note(f"        retry: {event}")

    rule("scrape-session demo: four primitives, one local server, no network")
    print(f"work directory : {workdir}")
    print(f"items          : {len(items)} ({', '.join(items)})")

    with local_site(site) as base_url:
        print(f"local site     : {base_url}  (ephemeral port, loopback only)")

        transport, session, manifest, limiter, check = compose(
            base_url, workdir, pacing=20.0, on_retry=on_retry
        )

        # ---------------------------------------------------------------- pass 1
        rule("1. SessionKeeper + RateLimiter: pass 1 signs in, then dies part way")
        started = time.monotonic()
        session.ensure_fresh()
        note(
            f"session: signed in (login #{site.logins}), cookie "
            f"{session.cookie_header!r}, stored at {session.store_path.name}"
        )
        fetch = session.bind(transport)
        process(
            items,
            base_url=base_url,
            fetch=fetch,
            limiter=limiter,
            check=check,
            manifest=manifest,
            out_dir=out_dir,
            stop_after=7,
            verbose=verbose,
        )
        note(f"pass 1 finished in {time.monotonic() - started:.2f}s")
        note(f"manifest: {manifest.stats()['status_counts']}  ({len(manifest)} item(s) logged)")

        # ---------------------------------------------------------------- pass 2
        rule("2. ResumeManifest: pass 2 restores the session and skips what is done")
        # A new process would build the keeper from scratch; do exactly that.
        transport2, session2, _, limiter2, check2 = compose(
            base_url, workdir, pacing=20.0, on_retry=on_retry
        )
        manifest2 = ResumeManifest(workdir / "manifest.jsonl")
        note(
            f"session file restored: {session2.restored}"
            f"  (sign-ins on record: {session2.stats().signins}, logins served: {site.logins})"
        )
        fetch2 = session2.bind(transport2)
        done_before = list(manifest2.completed_ids())
        note(f"already done, will not be re-fetched: {', '.join(done_before)}")
        pending = manifest2.pending(items)
        note(f"still to do: {', '.join(pending)}")
        process(
            items,
            base_url=base_url,
            fetch=fetch2,
            limiter=limiter2,
            check=check2,
            manifest=manifest2,
            out_dir=out_dir,
            stop_after=None,
            verbose=verbose,
        )

        # ------------------------------------------------------------- evidence
        rule("3. What the server actually saw")
        for item in items:
            hits = site.hits.get(f"/item/{item}.pdf", 0)
            status = manifest2.status_of(item) or "missing"
            print(f"    {item}  requests={hits}  status={status}")
        print()
        print(f"    /login requests: {site.hits.get('/login', 0)} for {sum(site.hits.values())} total requests")
        print(f"    retries scheduled: {len(retries)}")
        for event in retries:
            print(f"        {event}")
        print(f"    limiter pass 1 : {limiter.stats().as_dict()}")
        print(f"    limiter pass 2 : {limiter2.stats().as_dict()}")

        rule("4. IntegrityCheck: what was saved, and what was refused")
        for record in manifest2.records():
            if record.status == ItemStatus.OK.value:
                print(
                    f"    ok         {record.item_id}  {record.size}B  attempts={record.attempts}  "
                    f"{record.checksum[:24]}...  {Path(record.path or '').name}"
                )
            else:
                print(
                    f"    {record.status:<10} {record.item_id}  attempts={record.attempts}  "
                    f"reason={record.reason}"
                )
        exceptions = manifest2.exceptions()
        print()
        print(f"    exceptions list ({len(exceptions)} item(s)) -- nothing written to disk for these:")
        for record in exceptions:
            target = out_dir / f"{record.item_id}.pdf"
            print(
                f"        {record.item_id}: {record.reason} "
                f"(output exists on disk: {target.exists()})"
            )
        report = manifest2.write_exception_report(workdir / "exceptions.json")
        print(f"    exception report written: {report.name}")

        rule("Summary")
        saved = sorted(p.name for p in out_dir.glob("*.pdf"))
        total = sum(p.stat().st_size for p in out_dir.glob("*.pdf"))
        print(f"    items planned           : {len(items)}")
        print(f"    saved                   : {len(saved)} ({total}B total)")
        print(f"    exceptions              : {len(exceptions)} "
              f"({', '.join(r.item_id for r in exceptions)})")
        print(f"    sign-ins for the job    : {site.hits.get('/login', 0)}")
        print(f"    manifest status counts  : {manifest2.status_counts()}")
        print(f"    manifest read at startup: {manifest2.recovery.summary()}")
        print(
            "    still pending           : "
            f"{manifest2.pending(items) or 'none'}"
            "   (failed/corrupt items stay pending on purpose:"
            " a re-run retries them once the app or the session is fixed)"
        )
        print()
        print(f"    files written under     : {out_dir}")
        print(f"    manifest                : {workdir / 'manifest.jsonl'}")
        print(f"    session (0600)          : {workdir / 'session.json'}")
        if verbose:
            print(f"    first manifest line     : {json.loads((workdir / 'manifest.jsonl').read_text().splitlines()[0])}")

    if temporary and not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"\nwork directory removed (pass --keep to inspect it): {workdir}")
    elif not temporary:
        print(f"\nwork directory kept: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
