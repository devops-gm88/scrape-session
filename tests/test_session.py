"""Session reuse, expiry detection and re-authentication.

The claim being tested is "sign in once, not once per item". The evidence is the
fixture server's own request counter for ``/login``: the tests assert the exact
number of sign-ins across one, two and three runs, and that the protected
endpoint kept answering with the persisted cookie.
"""

from __future__ import annotations

import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scrape_session import (
    Request,
    Response,
    SessionExpired,
    SessionKeeper,
    UrllibTransport,
)

from .testserver import Hit, Route, make_pdf, run_server


class SessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.store = self.dir / "session.json"
        self.transport = UrllibTransport(timeout=5.0)

    @staticmethod
    def site() -> dict:
        """Server-side session state shared by the routes below."""
        return {"logins": 0, "valid": set(), "last_token": None}

    def routes(self, state: dict) -> dict:
        def login(hit: Hit) -> Route:
            state["logins"] += 1
            token = f"tok-{state['logins']}"
            state["valid"].add(token)
            state["last_token"] = token
            return Route(
                200,
                b"signed in",
                "text/plain",
                headers={
                    "Set-Cookie": f"sid={token}; Path=/; HttpOnly",
                    "X-Login-Number": str(state["logins"]),
                },
            )

        def protected(hit: Hit) -> Route:
            if hit.cookie("sid") in state["valid"]:
                return Route(200, make_pdf(1024), "application/pdf")
            return Route(401, b"unauthorized", "text/plain")

        def expire(hit: Hit) -> Route:
            state["valid"].clear()
            return Route(200, b"expired", "text/plain")

        return {"/login": login, "/protected": protected, "/expire": expire}

    def authenticator(self, base_url: str, keeper_box: dict | None = None):
        """A caller-supplied login callback: POST /login, capture the cookie."""

        def sign_in(keeper: SessionKeeper) -> None:
            response = self.transport.send(
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
            if keeper_box is not None:
                keeper_box["logins"] = keeper_box.get("logins", 0) + 1

        return sign_in


class SessionReuseTests(SessionTestCase):
    def test_signs_in_once_and_reuses_the_session_across_runs(self) -> None:
        state = self.site()
        with run_server(self.routes(state)) as server:
            first = SessionKeeper(
                self.store, authenticate=self.authenticator(server.url), transport=self.transport
            )
            self.assertFalse(first.restored, "nothing on disk yet")
            self.assertTrue(first.ensure_fresh(), "the first run must sign in")
            for _ in range(3):
                response = first.fetch(self.transport, Request(server.url_for("/protected")))
                self.assertEqual(response.status, 200)
            self.assertTrue(self.store.exists())

            second = SessionKeeper(
                self.store, authenticate=self.authenticator(server.url), transport=self.transport
            )
            self.assertTrue(second.restored, "the second run restores from disk")
            self.assertFalse(second.ensure_fresh(), "no sign-in was needed")
            for _ in range(3):
                response = second.fetch(self.transport, Request(server.url_for("/protected")))
                self.assertEqual(response.status, 200)

            # Three runs, one sign-in.
            third = SessionKeeper(
                self.store, authenticate=self.authenticator(server.url), transport=self.transport
            )
            bound = third.bind(self.transport)
            for _ in range(3):
                self.assertEqual(bound(Request(server.url_for("/protected"))).status, 200)

            self.assertEqual(server.hits_for("/login"), 1)
            self.assertEqual(server.hits_for("/protected"), 9)
        self.assertEqual(state["logins"], 1)
        self.assertEqual(third.stats().signins, 1, "sign-ins persist across runs")

    def test_the_cookie_is_persisted_and_resent(self) -> None:
        state = self.site()
        with run_server(self.routes(state)) as server:
            keeper = SessionKeeper(
                self.store, authenticate=self.authenticator(server.url), transport=self.transport
            )
            keeper.ensure_fresh()
            keeper.fetch(self.transport, Request(server.url_for("/protected")))
            payload = json.loads(self.store.read_text(encoding="utf-8"))
            token = state["last_token"]
            self.assertEqual(payload["cookies"][0]["name"], "sid")
            self.assertEqual(payload["cookies"][0]["value"], token)
            self.assertEqual(payload["signins"], 1)

            hits = server.requests_for("/protected")
            self.assertEqual(hits[0].cookie("sid"), token)

    def test_the_session_file_is_owner_only_and_valid_json(self) -> None:
        state = self.site()
        with run_server(self.routes(state)) as server:
            keeper = SessionKeeper(
                self.store, authenticate=self.authenticator(server.url), transport=self.transport
            )
            keeper.ensure_fresh()
        json.loads(self.store.read_text(encoding="utf-8"))
        if os.name == "posix":
            mode = stat.S_IMODE(self.store.stat().st_mode)
            self.assertEqual(mode, 0o600, f"expected 0600, got {oct(mode)}")

    def test_no_temporary_files_are_left_behind(self) -> None:
        state = self.site()
        with run_server(self.routes(state)) as server:
            keeper = SessionKeeper(
                self.store, authenticate=self.authenticator(server.url), transport=self.transport
            )
            for _ in range(3):
                keeper.ensure_fresh(force=True)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["session.json"])

    def test_restore_returns_false_when_nothing_is_stored(self) -> None:
        keeper = SessionKeeper(self.store, autoload=False)
        self.assertFalse(keeper.restore())
        self.assertTrue(keeper.expired, "no credentials means expired")
        self.assertIsNone(keeper.cookie_header or None)

    def test_session_headers_are_persisted_and_applied(self) -> None:
        state = self.site()
        with run_server(self.routes(state)) as server:
            keeper = SessionKeeper(self.store, transport=self.transport)
            keeper.set_header("X-Api-Key", "secret-token")
            keeper.save()

            reopened = SessionKeeper(self.store, transport=self.transport)
            self.assertTrue(reopened.restored)
            sent = reopened.apply(
                Request(server.url_for("/protected"), headers={"Accept": "application/pdf"})
            )
            self.assertEqual(sent.headers.get("x-api-key"), "secret-token")
            self.assertEqual(sent.headers.get("accept"), "application/pdf")
            self.assertIn("scrape-session", sent.headers.get("user-agent"))

    def test_apply_does_not_overwrite_a_cookie_set_by_the_caller(self) -> None:
        keeper = SessionKeeper(self.store, autoload=False)
        keeper.add_cookie("sid", "from-jar")
        prepared = keeper.apply(Request("http://example.com/x", headers={"Cookie": "sid=explicit"}))
        self.assertEqual(prepared.headers.get("cookie"), "sid=explicit")


class ExpiryTests(SessionTestCase):
    def test_401_triggers_re_authentication_and_one_replay(self) -> None:
        state = self.site()
        with run_server(self.routes(state)) as server:
            logins: dict = {}
            keeper = SessionKeeper(
                self.store,
                authenticate=self.authenticator(server.url, logins),
                transport=self.transport,
            )
            bound = keeper.bind(self.transport)
            self.assertEqual(bound(Request(server.url_for("/protected"))).status, 200)
            self.assertEqual(logins["logins"], 1)

            # The server-side session is dropped behind our back.
            self.transport.send(Request(server.url_for("/expire")))
            response = bound(Request(server.url_for("/protected")))
            self.assertEqual(response.status, 200, "the replay after re-auth must work")
            self.assertEqual(server.hits_for("/protected"), 3)
            self.assertEqual(server.hits_for("/login"), 2, "exactly one extra sign-in")
            self.assertEqual(logins["logins"], 2)

    def test_401_without_a_replay_leaves_the_session_invalidated(self) -> None:
        state = self.site()
        with run_server(self.routes(state)) as server:
            keeper = SessionKeeper(
                self.store, authenticate=self.authenticator(server.url), transport=self.transport
            )
            keeper.ensure_fresh()
            self.transport.send(Request(server.url_for("/expire")))
            response = keeper.fetch(self.transport, Request(server.url_for("/protected")))
            self.assertEqual(response.status, 401)
            self.assertTrue(keeper.expired, "a 401 must invalidate the session")
            self.assertIn("401", keeper.last_invalid_reason or "")
            self.assertTrue(keeper.ensure_fresh(), "and the next call signs in again")
            self.assertEqual(server.hits_for("/login"), 2)

    def test_ttl_expiry_forces_a_new_sign_in(self) -> None:
        state = self.site()
        now = {"t": 1000.0}
        with run_server(self.routes(state)) as server:
            keeper = SessionKeeper(
                self.store,
                authenticate=self.authenticator(server.url),
                transport=self.transport,
                ttl_seconds=60,
                clock=lambda: now["t"],
            )
            self.assertTrue(keeper.ensure_fresh())
            self.assertFalse(keeper.ensure_fresh(), "still inside the ttl")
            now["t"] += 61
            self.assertTrue(keeper.expired)
            self.assertTrue(keeper.ensure_fresh(), "past the ttl, sign in again")
            self.assertEqual(server.hits_for("/login"), 2)
            self.assertEqual(server.hits_for("/protected"), 0)

    def test_validator_probe_can_force_re_authentication(self) -> None:
        state = self.site()
        with run_server(self.routes(state)) as server:
            probe_calls = {"n": 0}

            def validator(keeper: SessionKeeper) -> bool:
                probe_calls["n"] += 1
                return probe_calls["n"] == 1

            keeper = SessionKeeper(
                self.store,
                authenticate=self.authenticator(server.url),
                transport=self.transport,
                validator=validator,
            )
            self.assertTrue(keeper.ensure_fresh(), "first call signs in")
            self.assertFalse(keeper.ensure_fresh(), "the probe says the session is fine")
            self.assertTrue(keeper.ensure_fresh(), "the probe says it is not")
            self.assertEqual(server.hits_for("/login"), 2)
            self.assertEqual(probe_calls["n"], 2)

    def test_cookie_expiry_in_the_past_is_not_reused(self) -> None:
        state = self.site()
        token = "tok-1"
        state["valid"].add(token)
        state["logins"] = 1
        self.store.write_text(
            json.dumps(
                {
                    "v": 1,
                    "kind": "scrape-session/session",
                    "created_at": 0.0,
                    "updated_at": 0.0,
                    "expires_at": None,
                    "signins": 1,
                    "headers": {},
                    "cookies": [
                        {
                            "name": "sid",
                            "value": token,
                            "domain": "",
                            "path": "/",
                            "expires_at": 1.0,
                            "secure": False,
                            "created_at": 0.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        keeper = SessionKeeper(self.store)
        self.assertFalse(keeper.restored, "an expired cookie is not a session")
        self.assertEqual(keeper.cookies, {})

    def test_max_age_zero_deletes_a_cookie(self) -> None:
        keeper = SessionKeeper(self.store, autoload=False)
        keeper.add_cookie("sid", "value")
        self.assertIn("sid", keeper.cookies)
        response = Response(200, {"Set-Cookie": "sid=; Max-Age=0; Path=/"}, b"")
        changed = keeper.capture(response)
        self.assertEqual(changed, ("sid",))
        self.assertNotIn("sid", keeper.cookies)

    def test_invalidate_and_clear(self) -> None:
        keeper = SessionKeeper(self.store, autoload=False)
        keeper.add_cookie("sid", "value")
        keeper.set_header("X-Token", "t")
        self.assertFalse(keeper.expired)
        keeper.invalidate("test")
        self.assertTrue(keeper.expired)
        self.assertEqual(keeper.last_invalid_reason, "test")
        keeper.clear()
        self.assertFalse(keeper.has_credentials)
        self.assertIsNone(keeper.last_invalid_reason)


class AuthenticationCallbackTests(SessionTestCase):
    def test_missing_authenticate_callback_raises_session_expired(self) -> None:
        keeper = SessionKeeper(self.store, autoload=False)
        with self.assertRaises(SessionExpired) as caught:
            keeper.ensure_fresh()
        self.assertIn("authenticate", str(caught.exception))
        self.assertIn("session.json", str(caught.exception))
        self.assertFalse(self.store.exists(), "a failed sign-in writes nothing")

    def test_callback_may_return_a_response_or_a_header_mapping(self) -> None:
        def returns_response(keeper: SessionKeeper) -> Response:
            return Response(200, {"Set-Cookie": "sid=abc; Path=/"}, b"")

        def returns_headers(keeper: SessionKeeper) -> dict:
            return {"Authorization": "Bearer token-123"}

        first = SessionKeeper(self.store / "a.json", authenticate=returns_response)
        first.ensure_fresh()
        self.assertEqual(first.cookie_header, "sid=abc")
        second = SessionKeeper(self.store / "b.json", authenticate=returns_headers)
        second.ensure_fresh()
        self.assertEqual(second.headers.get("authorization"), "Bearer token-123")

    def test_callback_may_set_cookies_directly(self) -> None:
        def direct(keeper: SessionKeeper) -> None:
            keeper.add_cookie("sid", "made-by-callback")
            keeper.set_header("X-Mode", "direct")

        keeper = SessionKeeper(self.store, authenticate=direct)
        self.assertTrue(keeper.ensure_fresh())
        self.assertEqual(keeper.cookie_header, "sid=made-by-callback")
        self.assertEqual(keeper.headers.get("x-mode"), "direct")

    def test_callback_returning_junk_is_rejected_loudly(self) -> None:
        keeper = SessionKeeper(self.store, authenticate=lambda k: 42)
        with self.assertRaises(TypeError):
            keeper.ensure_fresh()

    def test_a_failing_login_propagates_and_is_not_persisted(self) -> None:
        state = self.site()

        def broken(keeper: SessionKeeper) -> None:
            raise RuntimeError("bad credentials")

        with run_server(self.routes(state)) as server:
            keeper = SessionKeeper(self.store, authenticate=broken, transport=self.transport)
            with self.assertRaises(RuntimeError):
                keeper.ensure_fresh()
            self.assertEqual(keeper.signins, 0)
            self.assertFalse(self.store.exists())
            self.assertEqual(server.hits_for("/login"), 0)


class RecoveryTests(SessionTestCase):
    def test_a_corrupt_session_file_is_quarantined_not_fatal(self) -> None:
        state = self.site()
        self.store.write_text("{not json at all", encoding="utf-8")
        with run_server(self.routes(state)) as server:
            keeper = SessionKeeper(
                self.store, authenticate=self.authenticator(server.url), transport=self.transport
            )
            self.assertFalse(keeper.restored)
            self.assertIn("unreadable session file", keeper.last_invalid_reason or "")
            quarantined = [
                p for p in self.dir.iterdir() if ".corrupt-" in p.name
            ]
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].read_text(encoding="utf-8"), "{not json at all")
            self.assertTrue(keeper.ensure_fresh(), "it simply signs in again")
            self.assertEqual(server.hits_for("/login"), 1)
            self.assertTrue(keeper.restored is False)

    def test_a_session_file_with_unusable_entries_is_salvaged(self) -> None:
        self.store.write_text(
            json.dumps(
                {
                    "v": 1,
                    "headers": {"X-Good": "yes"},
                    "cookies": [{"value": "no-name"}, "not-a-mapping", {"name": "sid", "value": "v"}],
                }
            ),
            encoding="utf-8",
        )
        keeper = SessionKeeper(self.store)
        self.assertEqual(list(keeper.cookies), ["sid"])
        self.assertEqual(keeper.headers.get("x-good"), "yes")
        self.assertTrue(keeper.restored)

    def test_state_and_stats_describe_the_session(self) -> None:
        keeper = SessionKeeper(self.store, autoload=False)
        keeper.add_cookie("sid", "v")
        state = keeper.state()
        self.assertEqual(state["cookies"], ["sid"])
        self.assertFalse(state["expired"])
        stats = keeper.stats()
        self.assertEqual(stats.cookies, 1)
        self.assertEqual(stats.requests, 0)
        self.assertIn("signins", stats.as_dict())
        self.assertIn("SessionKeeper", repr(keeper))

    def test_save_creates_missing_parent_directories(self) -> None:
        nested = self.dir / "deep" / "deeper" / "session.json"
        keeper = SessionKeeper(nested, autoload=False)
        keeper.add_cookie("sid", "v")
        keeper.save()
        self.assertTrue(nested.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
