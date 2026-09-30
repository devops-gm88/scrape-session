"""Persist and reuse an authenticated session across runs.

A job that signs in once per item is unusable at scale: it is slow, it looks
like credential stuffing, and it is the first thing to break. :class:`SessionKeeper`
signs in once, writes the resulting cookies and headers to disk, reloads them on
the next run, and calls your login callback again only when the session is
genuinely gone.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Optional, Union

from .transport import Fetch, Headers, Request, Response, as_fetch

__all__ = ["Cookie", "SessionExpired", "SessionKeeper", "SessionStats"]

#: Bumped whenever the on-disk session layout changes incompatibly.
SESSION_FILE_VERSION = 1

DEFAULT_USER_AGENT = "scrape-session/0.1 (+https://example.com/scrape-session)"


class SessionExpired(RuntimeError):
    """The session is gone and no ``authenticate`` callback was supplied."""


@dataclass
class Cookie:
    """One cookie in the session jar."""

    name: str
    value: str
    domain: str = ""
    path: str = "/"
    expires_at: Optional[float] = None
    secure: bool = False
    created_at: float = field(default_factory=time.time)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """True when the cookie carries an expiry that has already passed."""
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at

    def to_mapping(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "name": self.name,
            "value": self.value,
            "domain": self.domain,
            "path": self.path,
            "expires_at": self.expires_at,
            "secure": self.secure,
            "created_at": self.created_at,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Cookie":
        """Rebuild a cookie from :meth:`to_mapping` output (tolerant of extras)."""
        expires = data.get("expires_at")
        return cls(
            name=str(data["name"]),
            value=str(data.get("value", "")),
            domain=str(data.get("domain", "")),
            path=str(data.get("path", "/")),
            expires_at=float(expires) if expires is not None else None,
            secure=bool(data.get("secure", False)),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass(frozen=True)
class SessionStats:
    """Counters that describe what the session actually did."""

    signins: int
    requests: int
    restored: bool
    expired: bool
    session_age_seconds: float
    cookies: int

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, handy for logging."""
        return {
            "signins": self.signins,
            "requests": self.requests,
            "restored": self.restored,
            "expired": self.expired,
            "session_age_seconds": round(self.session_age_seconds, 3),
            "cookies": self.cookies,
        }


class SessionKeeper:
    """Owns a session's cookies/headers and its on-disk representation.

    Args:
        store_path: where the session is persisted (created with mode ``0o600``;
            treat it as a credential file).
        authenticate: caller-supplied login callback, called as
            ``authenticate(keeper)``. It may call :meth:`capture` with a login
            response, :meth:`add_cookie`, or :meth:`set_header`; it may also
            simply *return* a mapping of headers or a :class:`Response` whose
            cookies should be captured.
        transport: default transport used by :meth:`fetch` when none is passed.
        headers: static headers sent with every request (e.g. ``Accept``).
        ttl_seconds: optional assumed session lifetime; after it elapses the
            session is treated as expired even if nothing said so.
        validator: optional ``validator(keeper) -> bool`` probe used to decide
            whether a restored session still works.
        expired_statuses: statuses that mean "session is gone" (default ``401``).
    """

    def __init__(
        self,
        store_path: Union[str, os.PathLike[str]],
        *,
        authenticate: Optional[Callable[["SessionKeeper"], Any]] = None,
        transport: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        ttl_seconds: Optional[float] = None,
        validator: Optional[Callable[["SessionKeeper"], bool]] = None,
        expired_statuses: Iterable[int] = (401,),
        user_agent: str = DEFAULT_USER_AGENT,
        clock: Callable[[], float] = time.time,
        fsync: bool = True,
        autoload: bool = True,
    ) -> None:
        self.store_path = Path(store_path)
        self.authenticate = authenticate
        self.transport = transport
        self.ttl_seconds = ttl_seconds
        self.validator = validator
        self.expired_statuses = frozenset(int(s) for s in expired_statuses)
        self.clock = clock
        self.fsync = bool(fsync)

        self._headers = Headers()
        self._default_headers = Headers(headers or {})
        self._default_headers.set("user-agent", user_agent)
        self._cookies: dict[str, Cookie] = {}
        self._lock = threading.RLock()

        self.created_at: float = self.clock()
        self.updated_at: float = self.created_at
        self.expires_at: Optional[float] = None
        self.signins: int = 0
        self.requests: int = 0
        self.restored: bool = False
        self.last_invalid_reason: Optional[str] = None
        self._loaded: bool = False
        self._loaded_state: Optional[dict[str, Any]] = None

        if autoload:
            self.restore()

    # ------------------------------------------------------------------ state
    @property
    def headers(self) -> Headers:
        """Session-scoped headers (returned live; mutate via :meth:`set_header`)."""
        return self._headers

    @property
    def cookies(self) -> dict[str, Cookie]:
        """Snapshot of the live cookie jar, expired entries removed."""
        self._drop_expired_cookies()
        return dict(self._cookies)

    @property
    def cookie_header(self) -> str:
        """The ``Cookie:`` header value implied by the jar (``""`` when empty)."""
        self._drop_expired_cookies()
        return "; ".join(f"{c.name}={c.value}" for c in self._cookies.values())

    @property
    def has_credentials(self) -> bool:
        """True when anything reusable is held (cookies or session headers)."""
        self._drop_expired_cookies()
        return bool(self._cookies) or len(self._headers) > 0

    @property
    def expired(self) -> bool:
        """True when no credentials are held or the assumed lifetime has lapsed."""
        if not self.has_credentials:
            return True
        if self.expires_at is not None and self.clock() >= self.expires_at:
            return True
        return False

    @property
    def age_seconds(self) -> float:
        """Seconds since the current session was established."""
        return max(0.0, self.clock() - self.created_at)

    # ------------------------------------------------------------ credentials
    def set_header(self, name: str, value: str) -> None:
        """Attach a session header (e.g. a bearer token)."""
        with self._lock:
            self._headers.set(name, value)
            self._touch()

    def add_cookie(self, name: str, value: str, **attributes: Any) -> Cookie:
        """Store a cookie directly, bypassing ``Set-Cookie`` parsing."""
        with self._lock:
            cookie = Cookie(name=name, value=value, **attributes)
            self._cookies[name] = cookie
            self._touch()
            return cookie

    def capture(self, response: Response) -> tuple[str, ...]:
        """Record every ``Set-Cookie`` in *response*.

        Returns the names of cookies that changed. A cookie sent with
        ``Max-Age=0`` or a past ``Expires`` deletes the entry.
        """
        changed: list[str] = []
        with self._lock:
            for raw in response.headers.get_all("set-cookie"):
                changed.extend(self._parse_set_cookie(raw))
            if changed:
                self._touch()
        return tuple(changed)

    def _parse_set_cookie(self, raw: str) -> list[str]:
        changed: list[str] = []
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except Exception:  # pragma: no cover - malformed cookie must not kill a run
            return []
        for name, morsel in jar.items():
            expires_at = self._cookie_expiry(morsel)
            if expires_at is not None and expires_at <= self.clock():
                if name in self._cookies:
                    del self._cookies[name]
                    changed.append(name)
                continue
            self._cookies[name] = Cookie(
                name=name,
                value=morsel.value,
                domain=morsel["domain"] or "",
                path=morsel["path"] or "/",
                expires_at=expires_at,
                secure=bool(morsel["secure"]),
                created_at=self.clock(),
            )
            changed.append(name)
        return changed

    def _cookie_expiry(self, morsel: Any) -> Optional[float]:
        if morsel["max-age"]:
            try:
                return self.clock() + float(morsel["max-age"])
            except (TypeError, ValueError):
                pass
        if morsel["expires"]:
            try:
                parsed = parsedate_to_datetime(morsel["expires"])
            except (TypeError, ValueError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        return None

    def _drop_expired_cookies(self) -> None:
        now = self.clock()
        for name in [n for n, c in self._cookies.items() if c.is_expired(now)]:
            del self._cookies[name]

    def _touch(self) -> None:
        self.updated_at = self.clock()

    # --------------------------------------------------------------- lifecycle
    def invalidate(self, reason: str = "invalidated by caller") -> None:
        """Mark the session unusable and clear reusable credentials."""
        with self._lock:
            self.last_invalid_reason = reason
            self.expires_at = 0.0

    def clear(self) -> None:
        """Forget all credentials (does not delete the file until :meth:`save`)."""
        with self._lock:
            self._cookies.clear()
            self._headers = Headers()
            self.expires_at = None
            self.last_invalid_reason = None

    def signin(self) -> None:
        """Run the caller-supplied login callback, then persist the result.

        Raises:
            SessionExpired: when no ``authenticate`` callback was supplied.
        """
        with self._lock:
            if self.authenticate is None:
                raise SessionExpired(
                    "session needs re-authentication but no authenticate callback "
                    f"was supplied (store: {self.store_path})"
                )
            self.clear()
            result = self.authenticate(self)
            self._apply_auth_result(result)
            self.signins += 1
            self.created_at = self.clock()
            if self.ttl_seconds is not None:
                self.expires_at = self.clock() + float(self.ttl_seconds)
            self._touch()
            self.save()
            return None

    def _apply_auth_result(self, result: Any) -> None:
        if result is None:
            return
        if isinstance(result, Response):
            self.capture(result)
            return
        if isinstance(result, Mapping):
            for name, value in result.items():
                self._headers.set(str(name), str(value))
            return
        raise TypeError(
            "authenticate() must return None, a Response, or a mapping of headers; "
            f"got {type(result).__name__}"
        )

    def ensure_fresh(self, *, force: bool = False) -> bool:
        """Guarantee a usable session, signing in when needed.

        Returns ``True`` when a new sign-in happened, ``False`` when the
        existing session was reused. This is the method that makes "sign in
        once" true across processes.
        """
        if not self._loaded:
            self.restore()
        with self._lock:
            if force or self.expired:
                self.signin()
                return True
            if self.validator is not None and not self.validator(self):
                self.invalidate("validator reported the session is unusable")
                self.signin()
                return True
            return False

    # ------------------------------------------------------------------- I/O
    def restore(self) -> bool:
        """Load persisted cookies/headers from disk.

        Returns ``True`` when a usable session was recovered. A malformed file is
        quarantined to ``<store>.corrupt-<timestamp>`` and reported as ``False``
        rather than raising: a damaged session file must never kill a run.
        """
        self._loaded = True
        path = self.store_path
        if not path.exists():
            return False
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("session file is not a JSON object")
        except (OSError, ValueError) as exc:
            self._quarantine(reason=f"unreadable session file: {exc}")
            return False

        self._headers = Headers()
        self._cookies = {}
        for name, value in dict(data.get("headers", {})).items():
            self._headers.set(str(name), str(value))
        for entry in data.get("cookies", []):
            if not isinstance(entry, dict):
                continue
            try:
                cookie = Cookie.from_mapping(entry)
            except (TypeError, KeyError, ValueError, AttributeError):
                continue
            self._cookies[cookie.name] = cookie

        self.created_at = float(data.get("created_at", self.clock()))
        self.updated_at = float(data.get("updated_at", self.created_at))
        expires = data.get("expires_at")
        self.expires_at = float(expires) if expires is not None else None
        self.signins = int(data.get("signins", 0))
        self.restored = self.has_credentials and not self.expired
        self._loaded_state = data
        return self.restored

    def save(self) -> Path:
        """Write the session to disk atomically (temp file, ``fsync``, ``os.replace``)."""
        with self._lock:
            self._drop_expired_cookies()
            self._touch()
            payload = {
                "v": SESSION_FILE_VERSION,
                "kind": "scrape-session/session",
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "expires_at": self.expires_at,
                "signins": self.signins,
                "last_invalid_reason": self.last_invalid_reason,
                "headers": {name: value for name, value in self._headers.items()},
                "cookies": [c.to_mapping() for c in self._cookies.values()],
            }
            blob = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            path = self.store_path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, blob)
                if self.fsync:
                    os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
            if self.fsync:
                self._fsync_dir(path.parent)
            return path

    def _quarantine(self, reason: str) -> Optional[Path]:
        backup = self.store_path.with_name(
            f"{self.store_path.name}.corrupt-{int(self.clock())}"
        )
        try:
            os.replace(self.store_path, backup)
        except OSError:  # pragma: no cover - best effort
            return None
        self.last_invalid_reason = reason
        self._cookies.clear()
        self._headers = Headers()
        self.expires_at = None
        return backup

    @staticmethod
    def _fsync_dir(directory: Path) -> None:
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:  # pragma: no cover - not supported everywhere
            return
        try:
            os.fsync(fd)
        except OSError:  # pragma: no cover
            pass
        finally:
            os.close(fd)

    # ---------------------------------------------------------------- requests
    def apply(self, request: Request) -> Request:
        """Return *request* with session headers and a ``Cookie`` header attached."""
        extra = Headers(self._default_headers)
        extra.update(self._headers)
        cookie = self.cookie_header
        if cookie and "cookie" not in request.headers:
            extra.set("cookie", cookie)
        return request.with_headers(extra)

    def fetch(self, fetch: Any, request: Request) -> Response:
        """Send *request* through *fetch*, then harvest any cookies it returned.

        A response whose status is in ``expired_statuses`` (``401`` by default)
        invalidates the session so the next :meth:`ensure_fresh` signs in again.
        """
        prepared = self.apply(request)
        response = as_fetch(fetch)(prepared)
        self.capture(response)
        self.requests += 1
        if response.status in self.expired_statuses:
            self.invalidate(f"server returned HTTP {response.status}")
        return response

    def bind(self, fetch: Any, *, refresh_on_expiry: bool = True) -> Fetch:
        """Return a :data:`Fetch` that keeps the session alive automatically.

        The returned callable ensures a session exists before the request, and
        -- when the server says the session is gone -- signs in again and
        replays the request exactly once. The replay is what converts "run died
        at item 900 with a 401" into "run kept going".
        """
        inner = as_fetch(fetch)

        def bound(request: Request) -> Response:
            self.ensure_fresh()
            response = self.fetch(inner, request)
            if (
                refresh_on_expiry
                and response.status in self.expired_statuses
                and self.authenticate is not None
            ):
                self.signin()
                response = self.fetch(inner, request)
            return response

        return bound

    # ------------------------------------------------------------------ misc
    def state(self) -> dict[str, Any]:
        """Live view of the session as it would be persisted."""
        self._drop_expired_cookies()
        return {
            "store_path": str(self.store_path),
            "restored_from_disk": self.restored,
            "cookies": sorted(self._cookies),
            "headers": sorted(self._headers),
            "expires_at": self.expires_at,
            "age_seconds": round(self.age_seconds, 3),
            "expired": self.expired,
            "last_invalid_reason": self.last_invalid_reason,
        }

    def stats(self) -> SessionStats:
        """Counters describing sign-ins, requests and session age."""
        return SessionStats(
            signins=self.signins,
            requests=self.requests,
            restored=self.restored,
            expired=self.expired,
            session_age_seconds=self.age_seconds,
            cookies=len(self.cookies),
        )

    def __repr__(self) -> str:
        return (
            f"<SessionKeeper store={str(self.store_path)!r} cookies={len(self._cookies)} "
            f"signins={self.signins} restored={self.restored}>"
        )
