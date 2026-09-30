"""Transport-agnostic request/response types and a standard-library HTTP transport.

Everything in this package talks to the network through one tiny contract::

    Fetch = Callable[[Request], Response]

Bring your own implementation -- :class:`UrllibTransport` is the stdlib one, but a
browser driver (or a test double) only has to return a :class:`Response`.
"""

from __future__ import annotations

import http.client
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Protocol, Union, runtime_checkable

__all__ = [
    "RETRYABLE_STATUSES",
    "ConnectionFailure",
    "Fetch",
    "Headers",
    "Request",
    "Response",
    "TimeoutFailure",
    "Transport",
    "TransportError",
    "UrllibTransport",
    "as_fetch",
    "is_retryable_status",
]

#: HTTP status codes that are worth trying again.
#:
#: The list is deliberately explicit and small. ``429`` (rate limited) and the
#: retryable ``5xx`` family mean "the server could not serve you right now".
#: ``403`` and ``404`` are *not* in here: the same request will keep returning
#: the same answer, so retrying only wastes the rate-limit budget.
RETRYABLE_STATUSES: frozenset[int] = frozenset(
    {
        408,  # Request Timeout
        425,  # Too Early
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
        507,  # Insufficient Storage
        509,  # Bandwidth Limit Exceeded
    }
)


def is_retryable_status(status: int) -> bool:
    """Return ``True`` when *status* is a transient HTTP failure code."""
    return int(status) in RETRYABLE_STATUSES


class TransportError(Exception):
    """No usable HTTP response was produced.

    Raised for connection refusals, DNS failures, timeouts and connections
    dropped before any response headers arrived. It is *not* raised for a
    truncated body: that is a real response with incomplete content, and it is
    reported as ``Response.truncated`` so integrity checking can classify it as
    corruption rather than a transient failure.
    """

    retryable: bool = True


class ConnectionFailure(TransportError):
    """Connection refused, reset, or an otherwise unusable socket."""


class TimeoutFailure(TransportError):
    """The request timed out."""


class Headers(Mapping[str, str]):
    """A case-insensitive header bag that keeps duplicate names.

    ``headers["content-type"]`` returns the first value for that name (the
    convenient behaviour), while :meth:`get_all` returns every value, which is
    what you need for ``Set-Cookie``.
    """

    __slots__ = ("_store",)

    def __init__(
        self,
        items: Optional[
            Union[
                "Headers",
                Mapping[str, Union[str, Iterable[str]]],
                Iterable[tuple[str, str]],
            ]
        ] = None,
    ) -> None:
        self._store: dict[str, list[str]] = {}
        if items is None:
            return
        raw: Iterable[tuple[str, Any]]
        if isinstance(items, Headers):
            raw = items.raw_items()
        elif isinstance(items, Mapping):
            raw = items.items()
        else:
            raw = items
        for name, value in raw:
            if isinstance(value, (list, tuple)):
                for one in value:
                    self.add(name, one)
            else:
                self.add(name, value)

    # -- construction helpers -------------------------------------------------
    @classmethod
    def of(cls, *pairs: tuple[str, str]) -> "Headers":
        """Build headers from ``("Name", "value")`` pairs."""
        return cls(pairs)

    def add(self, name: str, value: Any) -> None:
        """Append a value without removing existing values for *name*."""
        self._store.setdefault(str(name).strip().lower(), []).append(str(value))

    def set(self, name: str, value: Any) -> None:  # noqa: A003 - mirrors dict API
        """Replace every existing value for *name*."""
        self._store[str(name).strip().lower()] = [str(value)]

    def update(self, other: "Headers | Mapping[str, str] | Iterable[tuple[str, str]]") -> None:
        """Merge *other* into this bag, overwriting on name collisions."""
        for name, value in Headers(other).raw_items():
            self.set(name, value)

    def merged(
        self, other: "Headers | Mapping[str, str] | Iterable[tuple[str, str]]"
    ) -> "Headers":
        """Return a new bag with *other* layered on top of this one."""
        combined = self.copy()
        combined.update(other)
        return combined

    def copy(self) -> "Headers":
        """Return an independent copy."""
        clone = Headers()
        for name, value in self.raw_items():
            clone.add(name, value)
        return clone

    # -- Mapping protocol -----------------------------------------------------
    def __getitem__(self, name: str) -> str:
        key = name.strip().lower()
        try:
            return self._store[key][0]
        except KeyError:
            raise KeyError(name) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name.strip().lower() in self._store

    def get(self, name: str, default: Any = None) -> Any:  # type: ignore[override]
        """First value for *name*, or *default* when absent."""
        values = self._store.get(name.strip().lower())
        if not values:
            return default
        return values[0]

    def get_all(self, name: str) -> tuple[str, ...]:
        """Every value recorded for *name*, in arrival order."""
        return tuple(self._store.get(name.strip().lower(), ()))

    def raw_items(self) -> tuple[tuple[str, str], ...]:
        """All ``(name, value)`` pairs, including duplicates."""
        return tuple(
            (name, value) for name, values in self._store.items() for value in values
        )

    def __repr__(self) -> str:
        inner = ", ".join(f"{name}={value!r}" for name, value in self.items())
        return f"Headers({inner})"


@dataclass(frozen=True)
class Request:
    """A transport-neutral request.

    A browser-driver transport should honour ``method``/``url`` at minimum and
    may ignore ``headers``/``body`` when the driver handles those itself.
    """

    url: str
    method: str = "GET"
    headers: Headers = field(default_factory=Headers)
    body: Optional[bytes] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.headers, Headers):
            object.__setattr__(self, "headers", Headers(self.headers))

    def with_headers(
        self, extra: "Headers | Mapping[str, str] | Iterable[tuple[str, str]]"
    ) -> "Request":
        """Return a copy carrying *extra* headers (existing names are replaced)."""
        return replace(self, headers=self.headers.merged(extra))

    def with_meta(self, **extra: Any) -> "Request":
        """Return a copy with extra caller metadata attached."""
        merged = dict(self.meta)
        merged.update(extra)
        return replace(self, meta=merged)

    def __repr__(self) -> str:
        return f"Request({self.method} {self.url!r})"


@dataclass(frozen=True)
class Response:
    """A transport-neutral response.

    ``truncated`` is the important field: it means the transport received fewer
    body bytes than the server promised. That is *not* the same as a failed
    request -- the server answered, the answer is just incomplete -- so it is
    surfaced here instead of being raised as a transport error.
    """

    status: int
    headers: Headers = field(default_factory=Headers)
    body: bytes = b""
    url: str = ""
    truncated: bool = False
    truncated_reason: Optional[str] = None
    attempts: int = 1
    elapsed: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.headers, Headers):
            object.__setattr__(self, "headers", Headers(self.headers))
        if not isinstance(self.body, bytes):
            object.__setattr__(self, "body", bytes(self.body))

    @property
    def reason(self) -> str:
        """Short human-readable outcome, e.g. ``"HTTP 200"`` or ``"connection reset"``."""
        if self.status:
            return f"HTTP {self.status}"
        return self.truncated_reason or "no response"

    @property
    def is_success(self) -> bool:
        """True for a 2xx status with a complete body."""
        return 200 <= self.status < 300 and not self.truncated

    @property
    def content_type(self) -> str:
        """Lowercased media type with parameters stripped (``""`` when absent)."""
        raw = self.headers.get("content-type", "")
        return str(raw).split(";", 1)[0].strip().lower()

    @property
    def content_type_raw(self) -> str:
        """The ``Content-Type`` header exactly as received."""
        return str(self.headers.get("content-type", ""))

    @property
    def declared_length(self) -> Optional[int]:
        """The ``Content-Length`` the server promised, when present and numeric."""
        raw = self.headers.get("content-length")
        if raw is None:
            return None
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return None

    @property
    def size(self) -> int:
        """Number of body bytes actually received."""
        return len(self.body)

    @property
    def text(self) -> str:
        """Body decoded as UTF-8 with replacement for undecodable bytes."""
        return self.body.decode("utf-8", errors="replace")

    @property
    def retryable_status(self) -> bool:
        """True when the status code is in :data:`RETRYABLE_STATUSES`."""
        return is_retryable_status(self.status)

    def __repr__(self) -> str:
        flag = " truncated" if self.truncated else ""
        return f"<Response {self.status} {self.size}B{flag}>"


@runtime_checkable
class Transport(Protocol):
    """Anything that can turn a :class:`Request` into a :class:`Response`.

    The whole package only ever calls this, so a browser driver, an in-process
    cache, or a test double is a drop-in replacement for HTTP.
    """

    def send(self, request: Request) -> Response:  # pragma: no cover - protocol
        ...


#: The callable form of :class:`Transport`. Downstream primitives accept this,
#: so a plain function (``lambda req: Response(...)``) is a valid transport.
Fetch = Callable[[Request], Response]


def as_fetch(transport: Union[Transport, Fetch, Any]) -> Fetch:
    """Normalise *transport* into a ``Callable[[Request], Response]``.

    Accepts a bare callable, a :class:`Transport` (anything with ``.send``), or
    an object implementing ``__call__`` that returns a :class:`Response`.
    """
    if callable(transport) and not hasattr(transport, "send"):
        return transport  # type: ignore[return-value]
    send = getattr(transport, "send", None)
    if callable(send):
        return send  # type: ignore[return-value]
    if callable(transport):
        return transport  # type: ignore[return-value]
    raise TypeError(
        "transport must be a callable or expose a callable .send(request); "
        f"got {type(transport).__name__}"
    )


class UrllibTransport:
    """Standard-library HTTP transport built on :mod:`urllib.request`.

    Body truncation is detected rather than raised: when the socket closes
    before ``Content-Length`` bytes arrive, :mod:`http.client` raises
    ``IncompleteRead`` -- the partial bytes are recovered and the response is
    returned with ``truncated=True``. Connection-level failures raise
    :class:`TransportError` subclasses instead.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        *,
        opener: Optional[urllib.request.OpenerDirector] = None,
    ) -> None:
        self.timeout = float(timeout)
        self._opener = opener if opener is not None else urllib.request.build_opener()

    # Allow ``transport(request)`` as well as ``transport.send(request)``.
    def __call__(self, request: Request) -> Response:
        return self.send(request)

    def send(self, request: Request) -> Response:
        """Perform *request* and return a :class:`Response`.

        Raises:
            TimeoutFailure: the request timed out.
            ConnectionFailure: the connection could not be established or was
                dropped before response headers arrived.
        """
        urllib_request = urllib.request.Request(
            request.url, data=request.body, method=request.method
        )
        for name, value in request.headers.raw_items():
            urllib_request.add_header(name, value)

        response: Optional[http.client.HTTPResponse] = None
        try:
            response = self._opener.open(urllib_request, timeout=self.timeout)
            status = int(response.getcode() or 0)
            headers = Headers(response.headers.items())
            try:
                body = response.read()
            except http.client.IncompleteRead as exc:
                return Response(
                    status=status,
                    headers=headers,
                    body=bytes(exc.partial or b""),
                    url=request.url,
                    truncated=True,
                    truncated_reason=(
                        f"connection closed mid-body: {len(exc.partial or b'')}B "
                        f"received, {exc.expected}B of the declared body never arrived"
                    ),
                )
            return Response(status=status, headers=headers, body=body, url=request.url)
        except urllib.error.HTTPError as exc:
            headers = Headers(exc.headers.items()) if exc.headers is not None else Headers()
            try:
                try:
                    body = exc.read() or b""
                except http.client.IncompleteRead as inner:
                    return Response(
                        status=int(exc.code),
                        headers=headers,
                        body=bytes(inner.partial or b""),
                        url=request.url,
                        truncated=True,
                        truncated_reason=(
                            f"connection closed mid-body: {len(inner.partial or b'')}B "
                            f"received, {inner.expected}B of the declared body never arrived"
                        ),
                    )
                return Response(
                    status=int(exc.code), headers=headers, body=body, url=request.url
                )
            finally:
                # HTTPError is a file-like object; not closing it leaks the
                # socket and makes the interpreter complain at shutdown.
                try:
                    exc.close()
                except Exception:  # pragma: no cover - close must never mask a result
                    pass
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise TimeoutFailure(f"timeout after {self.timeout}s: {request.url}") from exc
            raise ConnectionFailure(f"{request.url}: {reason}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutFailure(f"timeout after {self.timeout}s: {request.url}") from exc
        except (ConnectionError, http.client.HTTPException, OSError) as exc:
            raise ConnectionFailure(f"{request.url}: {exc!r}") from exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:  # pragma: no cover - close must never mask a result
                    pass

    def close(self) -> None:
        """Release pooled connections. Safe to call repeatedly."""
        opener = getattr(self._opener, "close", None)
        if callable(opener):
            opener()

    def __enter__(self) -> "UrllibTransport":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
