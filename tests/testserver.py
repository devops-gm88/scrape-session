"""A local ``http.server`` used as a fixture. It never touches the network.

The fixture exists so the awkward cases can be produced on demand and exactly:
a ``429`` twice in a row, a ``403`` that must not be retried, a body that
promises 400 bytes and delivers 100, a connection dropped before headers, a
session cookie that expires mid-run. None of that can be relied on from a real
site, and a test that needs it cannot be trusted if it has to guess.
"""

from __future__ import annotations

import json
import socket
import threading
import traceback
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Union
from urllib.parse import parse_qs, urlparse

__all__ = [
    "FixtureServer",
    "Hit",
    "Route",
    "make_pdf",
    "run_server",
]


@dataclass(frozen=True)
class Hit:
    """One request as the server saw it."""

    method: str
    path: str
    query: Mapping[str, list[str]]
    headers: Mapping[str, str]
    body: bytes
    at: float
    index: int = 0

    def header(self, name: str, default: str = "") -> str:
        """Case-insensitive header lookup."""
        return self.headers.get(name.lower(), default)

    def cookie(self, name: str) -> Optional[str]:
        """Value of one cookie from the ``Cookie`` header, or ``None``."""
        raw = self.header("cookie")
        for chunk in raw.split(";"):
            key, _, value = chunk.strip().partition("=")
            if key == name:
                return value
        return None

    def param(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """First value of a query-string parameter."""
        values = self.query.get(name)
        return values[0] if values else default


@dataclass
class Route:
    """What the server should do for a matching request.

    Args:
        status: HTTP status to send.
        body: response body.
        content_type: value for ``Content-Type``.
        headers: extra headers.
        delay: seconds to sleep before replying (used to make pacing observable).
        declared_length: override the ``Content-Length`` header, which is how a
            truncated response is faked without lying to the client library.
        truncate_to: send only this many body bytes, then drop the connection.
        close_before_headers: drop the connection without replying at all.
        sequence: per-hit routes; the last entry repeats forever. This is how
            "429, 429, then 200" is expressed.
        handler: callable receiving the :class:`Hit` and returning a
            :class:`Route`; use it for anything stateful.
    """

    status: int = 200
    body: bytes = b""
    content_type: str = "application/octet-stream"
    headers: Mapping[str, str] = field(default_factory=dict)
    delay: float = 0.0
    declared_length: Optional[int] = None
    truncate_to: Optional[int] = None
    close_before_headers: bool = False
    sequence: tuple["Route", ...] = ()
    handler: Optional[Callable[[Hit], "Route"]] = None

    def resolve(self, hit: Hit) -> "Route":
        """Resolve this route for a specific request.

        A sequence is indexed by how many times the path has been hit; a
        handler gets the request itself; a plain route is returned unchanged.
        """
        if self.sequence:
            index = min(hit.index, len(self.sequence) - 1)
            return self.sequence[index].resolve(hit)
        if self.handler is not None:
            return self.handler(hit)
        return self

    @classmethod
    def sequence_of(cls, *routes: "Route") -> "Route":
        """Build a route that replays *routes* and then repeats the last one."""
        return cls(sequence=tuple(routes))

    @classmethod
    def json(cls, payload: Any, status: int = 200, **kwargs: Any) -> "Route":
        """Build a JSON route."""
        body = json.dumps(payload).encode("utf-8")
        kwargs.setdefault("content_type", "application/json")
        return cls(status=status, body=body, **kwargs)

    @classmethod
    def html(cls, text: str, status: int = 200, **kwargs: Any) -> "Route":
        """Build an HTML route."""
        kwargs.setdefault("content_type", "text/html; charset=utf-8")
        return cls(status=status, body=text.encode("utf-8"), **kwargs)


class FixtureServer:
    """A threaded HTTP server on an ephemeral loopback port.

    The port is chosen by the OS (``bind`` to port 0), so tests never collide
    with each other or with anything already running on the machine.
    """

    def __init__(self, routes: Optional[Mapping[str, Union[Route, Callable[[Hit], Route]]]] = None) -> None:
        self._routes: dict[str, Route] = {}
        self.hits: dict[str, int] = {}
        self.log: list[Hit] = []
        self.lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        for path, route in (routes or {}).items():
            self.set_route(path, route)

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> "FixtureServer":
        """Bind, listen, and serve in a background thread."""
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        httpd.daemon_threads = True
        httpd.fixture = self  # type: ignore[attr-defined]
        self._httpd = httpd
        # poll_interval keeps shutdown() responsive: the default 0.5s would add
        # half a second to every test that tears a server down.
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.02},
            name="fixture-server",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Shut the server down and join its thread."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def port(self) -> int:
        """The loopback port the OS assigned."""
        if self._httpd is None:
            raise RuntimeError("server is not running")
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        """Base URL, e.g. ``http://127.0.0.1:52143``."""
        return f"http://127.0.0.1:{self.port}"

    def url_for(self, path: str, **params: Any) -> str:
        """Absolute URL for *path* with optional query parameters."""
        if not path.startswith("/"):
            path = "/" + path
        if not params:
            return self.url + path
        query = "&".join(f"{key}={value}" for key, value in params.items())
        return f"{self.url}{path}?{query}"

    def __enter__(self) -> "FixtureServer":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------ routing
    def set_route(
        self, path: str, route: Union[Route, Callable[[Hit], Route]]
    ) -> None:
        """Install or replace the route for *path*."""
        if callable(route) and not isinstance(route, Route):
            route = Route(handler=route)
        with self.lock:
            self._routes[path] = route  # type: ignore[assignment]

    def remove_route(self, path: str) -> None:
        """Remove a route, restoring the default 404 behaviour."""
        with self.lock:
            self._routes.pop(path, None)

    def route_for(self, path: str, hit: Optional[Hit] = None) -> Route:
        """Resolve the route for *path* for the given request."""
        with self.lock:
            route = self._routes.get(path)
        if route is None:
            return Route(status=404, body=b"not found", content_type="text/plain")
        if hit is None:
            hit = Hit(
                method="GET", path=path, query={}, headers={}, body=b"", at=time.monotonic()
            )
        return route.resolve(hit)

    # ---------------------------------------------------------------- counters
    def hits_for(self, path: str) -> int:
        """How many requests have been made to *path*."""
        with self.lock:
            return self.hits.get(path, 0)

    def total_hits(self) -> int:
        """How many requests the server has received in total."""
        with self.lock:
            return sum(self.hits.values())

    def requests_for(self, path: str) -> list[Hit]:
        """Every logged request for *path*, in arrival order."""
        with self.lock:
            return [hit for hit in self.log if hit.path == path]

    def reset_counters(self) -> None:
        """Zero the hit counters and clear the request log (routes are kept)."""
        with self.lock:
            self.hits.clear()
            self.log.clear()

    def __repr__(self) -> str:
        return f"<FixtureServer {self.url if self._httpd else 'stopped'}>"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "scrape-session-fixture/0.1"

    # Keep the test output clean: the fixture's request log is the record.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        fixture: FixtureServer = self.server.fixture  # type: ignore[attr-defined]
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""

        headers = {str(k).lower(): str(v) for k, v in self.headers.items()}
        with fixture.lock:
            index = fixture.hits.get(path, 0)
            fixture.hits[path] = index + 1
            hit = Hit(
                method=method,
                path=path,
                query=parse_qs(parsed.query),
                headers=headers,
                body=body,
                at=time.monotonic(),
                index=index,
            )
            fixture.log.append(hit)

        try:
            route = fixture.route_for(path, hit)
            if route.delay > 0:
                time.sleep(route.delay)
        except Exception:  # a broken fixture must be visible, not a dropped socket
            traceback.print_exc()
            self._respond(Route(500, b"fixture route raised", "text/plain"))
            return
        self._respond(route)

    def _respond(self, route: Route) -> None:
        if route.close_before_headers:
            # Drop the connection with no reply: the client sees a reset rather
            # than a status code, which is the retryable transport failure.
            self.close_connection = True
            self._hard_close()
            return

        payload = route.body
        declared = route.declared_length
        if declared is None:
            declared = len(payload)

        self.send_response(route.status)
        self.send_header("Content-Type", route.content_type)
        for name, value in route.headers.items():
            self.send_header(name, value)
        if route.status not in (204, 304):
            self.send_header("Content-Length", str(declared))
        self.end_headers()

        if route.truncate_to is not None:
            self.wfile.write(payload[: route.truncate_to])
            self.wfile.flush()
            self.close_connection = True
            self._hard_close()
            return

        if payload:
            self.wfile.write(payload)
        self.wfile.flush()

    def _hard_close(self) -> None:
        try:
            self.wfile.flush()
        except Exception:
            pass
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass


@contextmanager
def run_server(
    routes: Optional[Mapping[str, Union[Route, Callable[[Hit], Route]]]] = None,
) -> Iterator[FixtureServer]:
    """Context manager yielding a started :class:`FixtureServer`."""
    server = FixtureServer(routes)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def make_pdf(size: int = 2048, *, marker: bytes = b"%PDF-1.7 fixture") -> bytes:
    """Build plausible PDF bytes of exactly *size*.

    Starts with ``%PDF-`` magic and ends with ``%%EOF``, padded in between, so
    integrity checks that inspect the head and tail of a document have something
    real to look at.
    """
    if size < len(marker) + 8:
        raise ValueError("size too small for a PDF-ish body")
    tail = b"\n%%EOF\n"
    filler = b"x" * (size - len(marker) - len(tail))
    return marker + filler + tail
