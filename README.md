# scrape-session

[![CI](https://github.com/devops-gm88/scrape-session/actions/workflows/ci.yml/badge.svg)](https://github.com/devops-gm88/scrape-session/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)

Four small primitives for long-running extraction jobs, where the hard part is
never the selector.

Standard library only. Python 3.10+. No dependencies, no network access needed to
run the tests or the demo.

* **`SessionKeeper`** — sign in once, persist the session, reuse it next run,
  re-authenticate only when the session is genuinely gone.
* **`ResumeManifest`** — record every finished item durably, so a run that dies
  at item 340 of 400 resumes at 341 instead of starting over.
* **`RateLimiter`** — a request ceiling you enforce, plus bounded exponential
  backoff for genuinely transient failures. `429`/`5xx` retry; `403`/`404` do not.
* **`IntegrityCheck`** — decide whether the bytes are trustworthy, so a truncated
  PDF is never saved as if it worked.

Every primitive works over any `Callable[[Request], Response]`, so the same code
drives stdlib HTTP or a browser you supply. There is no bundled scraper, no
selector engine, and no site-specific code: this package is the plumbing around
whatever fetches and parses.

---

## The problem

Two failure modes cost real money on fixed-price extraction work.

**A run that dies and starts over is unusable.** A job that fetches 3,500
documents will eventually lose its network, its session, or its own process. If
the run keeps no durable record of which items finished, the only options are to
re-run everything (paying again for what was already fetched, and inviting a
rate limit) or to hand-check the output folder. Neither is a deliverable.

**A truncated file saved as a success is worse than a failure.** A session-
authenticated app that hands back its login page with `200 OK`, a proxy that
closes the connection mid-body, a `206` with no `Content-Range` — all of these
produce a file on disk. The log says "ok", the count says 3,500, and the client
finds empty or HTML files weeks later. A failure you can see is cheap; a silent
one is not.

This package exists so those two things are handled once, tested, instead of
re-derived under deadline in every rescue job.

---

## Install

```bash
pip install -e .            # or just add src/ to PYTHONPATH; nothing to install
```

The test suite and the demo run straight from a clone with no install step.

---

## 1. `SessionKeeper`

```python
from scrape_session import Request, SessionKeeper, UrllibTransport

transport = UrllibTransport(timeout=30)

def sign_in(keeper: SessionKeeper) -> None:
    """Your login, not ours: scrape-session never assumes a login form."""
    response = transport.send(
        Request(
            "https://example.com/login",
            method="POST",
            body=b"username=...&password=...",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    )
    keeper.capture(response)                       # store every Set-Cookie
    keeper.set_header("X-CSRF", response.headers.get("x-csrf-token", ""))

session = SessionKeeper(
    "state/session.json",                          # written 0600: treat as a credential
    authenticate=sign_in,
    transport=transport,
    ttl_seconds=8 * 3600,                          # optional assumed lifetime
)

session.ensure_fresh()       # signs in only if nothing usable was on disk
fetch = session.bind(transport)
response = fetch(Request("https://example.com/doc/1"))

print(session.stats().as_dict())
# {'signins': 1, 'requests': 1, 'expired': False, 'cookies': 1, ...}
```

`bind()` is the useful part: it ensures a session, and when the server answers
`401` it signs in again and replays the request **exactly once**. A corrupt
session file is quarantined to `session.json.corrupt-<timestamp>` and the run
signs in again rather than dying.

---

## 2. `ResumeManifest`

```python
from scrape_session import ItemStatus, ResumeManifest

manifest = ResumeManifest("state/manifest.jsonl")

for item_id in manifest.pending(item_ids):        # skips what is already done
    body, digest = fetch_document(item_id)
    manifest.mark(                                   # one line, fsynced, then return
        item_id,
        ItemStatus.OK,
        checksum=digest,
        size=len(body),
        path=out_path,
        attempts=attempts,
    )

print(manifest.completed_ids())
print(manifest.exceptions())        # corrupted + permanent failures, with reasons
print(manifest.recovery.summary())  # what the last load found on disk
```

A record looks like this:

```json
{"kind": "item", "v": 1, "seq": 7, "id": "doc-07", "status": "ok",
 "ts": "2026-01-01T12:00:00.000+00:00", "epoch": 1767268800.0,
 "checksum": "sha256:883593fd...", "size": 2048, "path": "/out/doc-07.pdf",
 "attempts": 1, "reason": null, "url": "https://example.com/doc/7", "meta": {}}
```

Crash safety is the whole point, so it is enforced structurally:

* **append-only** — nothing is rewritten in place, so there is no window where
  the file is half-old and half-new;
* **one `os.write` per record** on a file opened `O_APPEND`, then `fsync`, so an
  interrupted write can only ever damage the *last* line, never a line in the
  middle;
* **recovery on load** — a torn trailing line is dropped (or kept, if it happens
  to be complete JSON that merely lost its newline), and a damaged interior
  region is moved to `manifest.jsonl.corrupt-<timestamp>` before the file is
  repaired, so nothing is ever discarded in silence.

Only `ok` counts as done. A `corrupted` item stays pending, so the next run
retries it — which is what you want after the session is fixed.

---

## 3. `RateLimiter`

```python
from scrape_session import RateLimiter, Request, RetryPolicy

limiter = RateLimiter(
    max_requests_per_second=2,                     # ceiling you enforce, per process
    policy=RetryPolicy(max_attempts=4, base_delay=1.0, multiplier=2.0, jitter=0.25),
)

response = limiter.run(fetch, Request("https://example.com/doc/1"))
print(response.attempts, limiter.stats().as_dict())
```

* `429` and the retryable `5xx` codes are retried with exponential backoff;
  `403`, `404` and the rest of the `4xx` family are returned immediately, because
  asking again cannot change the answer.
* A `Retry-After` header is honoured, but never *shortens* the computed backoff,
  and is clamped by `max_retry_after` so a bad header cannot park the run.
* Connection-level failures (refused, reset, timed out) are retried; if every
  attempt fails at that level, `RetryExhausted` is raised with the last error
  attached. Transport errors and truncation are deliberately different things —
  see below.
* The clock and the sleep function are injectable, so the backoff schedule is
  asserted exactly in the tests instead of being inferred from wall-clock timing.

---

## 4. `IntegrityCheck`

```python
from scrape_session import IntegrityCheck, ItemStatus

check = IntegrityCheck.for_pdf(min_bytes=1024)     # type + size + %PDF- magic

assessment = check.assess(response)
if assessment.ok:
    target.write_bytes(response.body)   # verify against assessment.checksum if you like
else:
    manifest.mark(
        item_id,
        assessment.verdict.manifest_status,
        reason=assessment.reason,
        size=assessment.size,
        checksum=assessment.checksum,
    )
```

`assessment.reason` is the specific finding — `truncated_body`,
`content_length_mismatch`, `empty_body`, `content_type_mismatch`,
`html_payload`, `size_mismatch`, `validator_failed:<name>` — not just "failed".

### The decision table

| verdict | when | what to do |
| --- | --- | --- |
| `ok` | `2xx`, complete body, length matches, type and size match, validators passed | save it, record checksum + size |
| `retryable_failure` | `408`, `425`, `429`, retryable `5xx` | back off and retry |
| `permanent_failure` | other `4xx`, e.g. `403`, `404` | report it; retrying cannot help |
| `corrupted` | `2xx` whose body is truncated, empty, the wrong type, the wrong size, an HTML page where a document belongs, or fails a validator | **exceptions list — never saved as a success** |

Rules are applied in that order. The familiar three-way split is
ok / retryable / corrupted; `permanent_failure` is kept separate because "not
retryable" and "corrupted" are different problems with different fixes, and
filing a `403` under either of the other two would be a lie.

Two corruption signals deserve a note:

* **Truncation is not a transport error.** When a connection closes mid-body,
  `UrllibTransport` recovers the partial bytes and returns
  `Response(truncated=True, truncated_reason=...)` rather than raising. The
  server answered; the answer is incomplete. That is corruption, and it belongs
  in the exceptions list.
* **An HTML body with the wrong declared type is corruption.** A `200
  text/html` login page where `application/pdf` was expected is caught even if
  the server lies about the content type, by sniffing the first bytes of the body.

---

## The transport contract

Everything above accepts any `Callable[[Request], Response]`. That is the entire
interface:

```python
from scrape_session import Request, Response

def my_transport(request: Request) -> Response:
    ...
```

`UrllibTransport` is the stdlib implementation. To drive a browser instead, write
the adapter — scrape-session deliberately does not depend on one:

```python
def playwright_transport(request: Request) -> Response:
    """Illustrative only: this repo does not import playwright."""
    page = browser.new_page()
    result = page.request.fetch(
        request.url,
        method=request.method,
        headers=dict(request.headers),
        data=request.body,
    )
    return Response(
        status=result.status,
        headers=result.headers,
        body=result.body(),
        url=request.url,
    )

session.bind(playwright_transport)   # same primitives, browser fetch underneath
```

A browser usually owns its own cookies, so a driver-backed fetch can ignore the
`Cookie` header `SessionKeeper` adds and still use the keeper for expiry
detection, re-authentication and the manifest.

---

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

Last run in this repository:

```
Ran 127 tests in 1.7s

OK
```

No network access is required or used: every test runs against a local
`http.server` bound to an ephemeral loopback port. The fixture exists so the
awkward cases can be produced on demand and exactly — a `429` twice then a `200`,
a `403` that must not be retried, a body that promises 400 bytes and delivers
100, a connection dropped before headers, a cookie that expires mid-run.

What the suite pins down, in the words of the test names:

* `test_backoff_on_429_then_success` — two `429`s then a `200`; asserts the server
  saw exactly 3 requests and the limiter took 3 attempts.
* `test_no_retry_on_403` / `test_no_retry_on_404` — asserts the server saw
  **exactly one** request, which is the only way to prove a retry did not happen.
* `test_a_truncated_pdf_is_corruption_not_a_success` — a real truncated response
  is classified `corrupted`, the artefact is not written, and `output exists on
  disk: False`.
* `test_a_second_run_skips_completed_items` — two passes over 10 items; asserts
  each item was requested **exactly once across both runs**.
* `test_a_process_killed_mid_write_loses_only_the_in_flight_record` — a real
  second process appends to the manifest and dies mid-write.
* Plus: pacing assertions on an injected clock, session reuse across three runs
  (`/login` requested exactly once), expiry by TTL and by validator probe, `401`
  re-auth and replay, cookie deletion by `Max-Age=0`, a corrupt session file,
  forward-compatible manifests, concurrent appends, compaction, and both job
  shapes (document-per-item and paginated listing with detail pages).

### How the crash-safe manifest property is proven

Four tests, from mildest to most realistic:

1. **`test_partial_line_at_the_end_is_dropped_and_the_run_survives`** — a partial
   JSON line is appended to the file by hand (the shape a crash leaves behind).
   Reopening recovers all previous records, reports `dropped_tail_bytes > 0`, and
   does not mark the interrupted item done.
2. **`test_the_file_is_repaired_so_the_next_append_is_visible`** — after recovery
   the file is re-terminated, so the *next* append produces a separate, parseable
   line instead of gluing onto the garbage. This is the bug that makes naive
   "just skip bad lines" resume logs lose records silently.
3. **`test_a_write_that_died_mid_record_leaves_the_earlier_records_intact`** —
   `os.write` is patched to write half the buffer and then raise, which is exactly
   what a power loss during the write looks like: no rollback, no `fsync`.
4. **`test_a_process_killed_mid_write_loses_only_the_in_flight_record`** — a
   separate Python process appends to the same manifest, writes a third of the
   record, and calls `os._exit(9)`. The test asserts the child really died
   (`returncode == 9`) and that its partial write is on disk, then reopens the
   manifest: all five earlier records are intact, the interrupted item is not
   marked done, and appending it again works.

Also covered: **`test_damage_in_the_middle_quarantines_the_rest_and_keeps_the_prefix`**
— an inserted garbage line mid-file keeps the valid prefix, moves the damaged
file to `manifest.jsonl.corrupt-<timestamp>`, and the test asserts the backup is
byte-identical to the damaged original.

---

## Demo

```bash
python examples/demo.py
```

It starts its own HTTP server on a loopback port, plays a session-authenticated
document app (12 items, one that rate limits twice, one that denies, one that
truncates, one that returns a login page), and runs the whole thing twice — the
first pass deliberately "dies" after 7 items. Real output from this repository,
with the paths elided:

```
1. SessionKeeper + RateLimiter: pass 1 signs in, then dies part way
    session: signed in (login #1), cookie 'sid=session-1', stored at session.json
    doc-01: ok 2048B sha256:468bbc753bac... -> doc-01.pdf
    doc-02: ok 2048B sha256:0e58464c3deb... -> doc-02.pdf
            retry: attempt 1/4 failed (HTTP 429); sleeping 1.000s
            retry: attempt 2/4 failed (HTTP 429); sleeping 1.000s
    doc-03: ok 2048B sha256:883593fd246c... -> doc-03.pdf
    ...
    !! simulated crash: stopping after 7 item(s) in this pass

2. ResumeManifest: pass 2 restores the session and skips what is done
    session file restored: True  (sign-ins on record: 1, logins served: 1)
    already done, will not be re-fetched: doc-01, doc-02, doc-03, doc-04, doc-05, doc-06, doc-07
    still to do: doc-08, doc-09, doc-10, doc-11, doc-12
    doc-08: FAILED after 1 attempt(s) -> http_403 (no detail)
    doc-09: CORRUPTED after 1 attempt(s) -> truncated_body
    doc-10: CORRUPTED after 1 attempt(s) -> content_type_mismatch (got 'text/html', expected one of ['application/pdf'])
    doc-11: ok 2048B sha256:119192ba75d2... -> doc-11.pdf
    doc-12: ok 2048B sha256:d7a6edd97632... -> doc-12.pdf

3. What the server actually saw
    doc-01  requests=1  status=ok
    doc-02  requests=1  status=ok
    doc-03  requests=3  status=ok
    ...
    doc-08  requests=1  status=permanent_failure
    doc-09  requests=1  status=corrupted
    doc-10  requests=1  status=corrupted
    ...
    /login requests: 1 for 15 total requests
    retries scheduled: 2
    limiter pass 1 : {'responses': 9, 'attempts': 9, 'retries': 2, 'throttle_sleep': 0.27, 'backoff_sleep': 2.0, 'sleep_total': 2.27}

4. IntegrityCheck: what was saved, and what was refused
    exceptions list (3 item(s)) -- nothing written to disk for these:
        doc-08: http_403 (output exists on disk: False)
        doc-09: truncated_body (output exists on disk: False)
        doc-10: content_type_mismatch (output exists on disk: False)

Summary
    items planned           : 12
    saved                   : 9 (18432B total)
    exceptions              : 3 (doc-08, doc-09, doc-10)
    sign-ins for the job    : 1
    manifest status counts  : {'corrupted': 2, 'ok': 9, 'permanent_failure': 1}
```

Ports, timings, paths and digests vary between runs; the request counts,
statuses and totals do not. Use `--keep` (or `--workdir DIR`) to inspect the
manifest, session file and downloaded documents afterwards.

Every number quoted in this README comes from the runs shown above or from an
assertion in the test suite. There are no benchmarks or speed-up claims here,
because none were measured.

---

## Design notes

* **Why not `requests`, `playwright` or `tenacity`?** A rescue job often has to
  run on a locked-down machine or an air-gapped VM with nothing but the standard
  library, and a forty-line dependency list is its own outage. Browser support is
  an interface here, not a dependency.
* **Why JSONL and not SQLite?** One append per record with no schema migration is
  the smallest thing that survives `kill -9`, and the manifest stays readable
  with `grep` on a client's machine at 2am.
* **Why `fsync` by default?** "Completed" has to mean "survived losing the
  machine", otherwise the resume logic lies. `fsync=False` is available when a
  run's re-fetch cost is trivially low.
* **Why is truncation corruption rather than a retry?** A retry tells you nothing
  about the bytes you already received and hides a server-side problem. The
  exception list is where a human sees it.
* **Bounded retries only.** Retries are capped by attempt count, never by total
  elapsed time, and there is no circuit breaker — see below.

---

## What this does not do

Being explicit, because a package like this is easy to oversell:

* **No HTML/PDF parsing and no selectors.** Nothing here extracts fields, follows
  links, or understands a page. Bring your own parser.
* **No browser bundled.** There is a transport interface and a worked adapter
  sketch; you supply the driver and its dependency.
* **No proxy rotation, user-agent pooling or CAPTCHA handling.** Pacing is a
  client-side ceiling you choose, not a server quota and not an evasion tool.
* **No `robots.txt`, terms-of-service or legal review.** Your job, not a
  library's.
* **No multi-process coordination.** The manifest assumes a single writer
  process. Appends within one process are thread-safe; two processes appending
  to one manifest are not coordinated (the append itself is atomic, the counters
  and compaction are not).
* **No distributed or resumable-across-hosts state.** Everything is local files.
* **No scheduling, queueing, dashboards, metrics endpoint or alerting.** You get
  counters; wiring them somewhere is your call.
* **No total-time budget, circuit breaker, or adaptive rate control.** The
  limiter paces at the rate you configure, even when the server is unhappy.
* **No async or parallel fetching.** Deliberate: pacing and resume are easier to
  trust when one process does one thing at a time. `RateLimiter.acquire()` is
  lock-guarded so pacing survives threads, but the job loop shown here is
  single-threaded and its counters are not atomic.
* **No retry of corrupted artefacts, ever.** By design — see the decision table.
* **No partial-file resume.** An interrupted *download* restarts; only completed
  items are skipped.
* **Cookie handling is a simple jar.** No domain/path scoping, no SameSite or
  Secure enforcement, no subdomain rules — fine for one app, not a browser.

---

## Security note

`session.json` holds live credentials (cookies, tokens). It is written with mode
`0600` and the `save()` path is atomic (temp file, `fsync`, `os.replace`), but it
is **not encrypted**. `.gitignore` excludes `state/`, `*.session.json`, `*.jsonl`
and `*.corrupt-*`; check your own paths before committing anything.

## License

MIT — see [LICENSE](LICENSE).
