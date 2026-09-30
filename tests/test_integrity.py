"""Corruption must be distinguished from failure.

The interesting assertions here are the ones that say *no*: a ``200`` with half
a PDF is not a success, an HTML login page is not a PDF, and a ``403`` is not
worth retrying. Each is produced by the local fixture server as a real HTTP
response.
"""

from __future__ import annotations

import hashlib
import unittest

from scrape_session import (
    IntegrityCheck,
    ItemStatus,
    Request,
    Response,
    UrllibTransport,
    Verdict,
    json_validator,
    magic_validator,
    sha256_hex,
)

from .testserver import Route, make_pdf, run_server

LOGIN_PAGE = b"<!DOCTYPE html><html><head><title>Sign in</title></head><body>Sign in</body></html>"


class VerdictTests(unittest.TestCase):
    def test_verdict_properties(self) -> None:
        self.assertTrue(Verdict.OK.is_ok)
        self.assertFalse(Verdict.OK.is_failure)
        self.assertTrue(Verdict.RETRYABLE_FAILURE.retryable)
        self.assertFalse(Verdict.PERMANENT_FAILURE.retryable)
        self.assertTrue(Verdict.CORRUPTED.needs_attention)
        self.assertTrue(Verdict.PERMANENT_FAILURE.needs_attention)
        self.assertFalse(Verdict.RETRYABLE_FAILURE.needs_attention)

    def test_verdict_values_map_onto_manifest_statuses(self) -> None:
        for verdict in Verdict:
            self.assertIsNotNone(ItemStatus(verdict.manifest_status))


class ShaAndHelperTests(unittest.TestCase):
    def test_checksum_matches_hashlib(self) -> None:
        body = b"the quick brown fox"
        self.assertEqual(
            sha256_hex(body), "sha256:" + hashlib.sha256(body).hexdigest()
        )

    def test_magic_validator_requires_a_prefix(self) -> None:
        with self.assertRaises(ValueError):
            magic_validator()

    def test_json_validator_rejects_bad_json_and_wrong_roots(self) -> None:
        validator = json_validator(mode="object")
        self.assertIsNone(validator(b'{"a": 1}', Response(200)))
        self.assertIn("not valid JSON", validator(b"{oops", Response(200)) or "")
        self.assertIn("expected", validator(b"[1, 2]", Response(200)) or "")

    def test_json_validator_max_bytes(self) -> None:
        validator = json_validator(max_bytes=4)
        self.assertIsNotNone(validator(b'{"a": 1}', Response(200)))


class IntegrityCheckUnitTests(unittest.TestCase):
    def make(self, status: int = 200, body: bytes = b"x", **headers) -> Response:
        return Response(status, headers, body)

    def test_2xx_with_no_declared_length_and_content_is_ok(self) -> None:
        assessment = IntegrityCheck().assess(self.make(200, b"hello"))
        self.assertTrue(assessment.ok)
        self.assertEqual(assessment.reason, "ok")
        self.assertEqual(assessment.size, 5)
        self.assertEqual(assessment.checksum, sha256_hex(b"hello"))

    def test_empty_body_on_200_is_corruption_not_success(self) -> None:
        assessment = IntegrityCheck().assess(self.make(200, b""))
        self.assertIs(assessment.verdict, Verdict.CORRUPTED)
        self.assertEqual(assessment.reason, "empty_body")
        self.assertIsNone(assessment.checksum)

    def test_min_bytes_and_max_bytes_bounds(self) -> None:
        check = IntegrityCheck(min_bytes=10, max_bytes=20)
        self.assertEqual(check.assess(self.make(200, b"x" * 5)).reason, "body_below_min_bytes")
        self.assertEqual(check.assess(self.make(200, b"x" * 25)).reason, "body_above_max_bytes")
        self.assertTrue(check.assess(self.make(200, b"x" * 15)).ok)

    def test_declared_length_mismatch_is_corruption(self) -> None:
        response = self.make(200, b"short", **{"Content-Length": "500"})
        assessment = IntegrityCheck().assess(response)
        self.assertIs(assessment.verdict, Verdict.CORRUPTED)
        self.assertEqual(assessment.reason, "content_length_mismatch")
        self.assertIn("declared 500B", assessment.detail)

    def test_truncated_flag_outranks_everything_else(self) -> None:
        response = Response(
            200,
            {"content-type": "application/pdf"},
            b"%PDF-1.7 partial",
            truncated=True,
            truncated_reason="connection closed",
        )
        assessment = IntegrityCheck.for_pdf(min_bytes=1).assess(response)
        self.assertIs(assessment.verdict, Verdict.CORRUPTED)
        self.assertEqual(assessment.reason, "truncated_body")
        self.assertFalse(assessment.retryable)

    def test_size_mismatch_against_a_previous_run_is_corruption(self) -> None:
        assessment = IntegrityCheck().assess(self.make(200, b"1234567890"), expected_size=999)
        self.assertEqual(assessment.reason, "size_mismatch")

    def test_wrong_content_type_is_corruption(self) -> None:
        response = self.make(200, LOGIN_PAGE, **{"Content-Type": "text/html"})
        assessment = IntegrityCheck.for_pdf(min_bytes=1).assess(response)
        self.assertIs(assessment.verdict, Verdict.CORRUPTED)
        self.assertIn(assessment.reason, {"content_type_mismatch", "html_payload"})
        self.assertFalse(assessment.retryable)

    def test_html_body_declared_as_pdf_is_caught_by_sniffing(self) -> None:
        response = self.make(200, LOGIN_PAGE, **{"Content-Type": "application/pdf"})
        assessment = IntegrityCheck(min_bytes=1, expect_content_type="application/pdf").assess(
            response
        )
        self.assertIs(assessment.verdict, Verdict.CORRUPTED)
        self.assertEqual(assessment.reason, "html_payload")
        self.assertIn("HTML page", assessment.detail)

    def test_content_type_parameters_and_wildcards_are_tolerated(self) -> None:
        check = IntegrityCheck(min_bytes=1, expect_content_type="application/pdf")
        response = self.make(200, b"%PDF-1.7", **{"Content-Type": "application/pdf; charset=binary"})
        self.assertTrue(check.assess(response).ok)
        wildcard = IntegrityCheck(min_bytes=1, expect_content_type="image/*")
        png = self.make(200, b"\x89PNG\r\n\x1a\n", **{"Content-Type": "image/png"})
        self.assertTrue(wildcard.assess(png).ok)

    def test_missing_content_type_is_corruption_when_one_is_expected(self) -> None:
        assessment = IntegrityCheck(min_bytes=1, expect_content_type="application/json").assess(
            self.make(200, b'{"a":1}')
        )
        self.assertEqual(assessment.reason, "content_type_mismatch")

    def test_retryable_statuses_are_retryable_failures(self) -> None:
        for status in (429, 500, 502, 503, 504):
            assessment = IntegrityCheck(expect_content_type="application/pdf").assess(
                Response(status, {}, b"")
            )
            self.assertIs(assessment.verdict, Verdict.RETRYABLE_FAILURE, status)
            self.assertTrue(assessment.retryable, status)

    def test_4xx_is_a_permanent_failure_and_not_a_retry(self) -> None:
        for status in (400, 401, 403, 404, 410, 422):
            assessment = IntegrityCheck(expect_content_type="application/pdf").assess(
                Response(status, {}, b"denied")
            )
            self.assertIs(assessment.verdict, Verdict.PERMANENT_FAILURE, status)
            self.assertFalse(assessment.retryable, status)
            self.assertFalse(assessment.corrupted, status)

    def test_require_success_false_lets_error_pages_through(self) -> None:
        check = IntegrityCheck(require_success=False, min_bytes=1)
        assessment = check.assess(Response(404, {"content-type": "text/html"}, b"<html>nope</html>"))
        self.assertTrue(assessment.ok)

    def test_validators_run_last_and_are_named_in_the_reason(self) -> None:
        def must_be_signed(body: bytes, response: Response) -> str | None:
            return None if body.endswith(b"%%EOF") else "missing %%EOF trailer"

        check = IntegrityCheck(min_bytes=1, validators=(must_be_signed,))
        ok = check.assess(self.make(200, b"body %%EOF"))
        self.assertTrue(ok.ok)
        bad = check.assess(self.make(200, b"body without trailer"))
        self.assertEqual(bad.verdict, Verdict.CORRUPTED)
        self.assertTrue(bad.reason.startswith("validator_failed:must_be_signed"))
        self.assertIn("missing %%EOF", bad.detail)

    def test_per_call_overrides_do_not_leak_into_the_instance(self) -> None:
        check = IntegrityCheck(min_bytes=1)
        self.assertTrue(check.assess(self.make(200, b"ok"), min_bytes=1).ok)
        self.assertFalse(check.assess(self.make(200, b"ok"), min_bytes=10).ok)
        self.assertTrue(check.assess(self.make(200, b"ok")).ok)

    def test_assessment_fields_feed_the_manifest(self) -> None:
        assessment = IntegrityCheck(min_bytes=1).assess(self.make(200, b"hello"))
        fields = assessment.as_fields()
        self.assertEqual(fields["status"], ItemStatus.OK.value)
        self.assertEqual(fields["size"], 5)
        self.assertEqual(fields["checksum"], sha256_hex(b"hello"))
        self.assertIn("checks", assessment.as_dict())
        self.assertIn("HTTP 200", assessment.description)

    def test_is_ok_convenience(self) -> None:
        check = IntegrityCheck(min_bytes=1)
        self.assertTrue(check.is_ok(self.make(200, b"hello")))
        self.assertFalse(check.is_ok(self.make(500, b"")))

    def test_factories_configure_sensible_defaults(self) -> None:
        self.assertEqual(IntegrityCheck.for_pdf().min_bytes, 1024)
        self.assertIn("application/pdf", IntegrityCheck.for_pdf().expect_content_type)
        self.assertIn("application/json", IntegrityCheck.for_json().expect_content_type)
        self.assertFalse(IntegrityCheck.for_html().detect_html)
        self.assertIn("IntegrityCheck", repr(IntegrityCheck.for_pdf()))


class IntegrityAgainstServerTests(unittest.TestCase):
    """The same decisions, driven by responses that really crossed a socket."""

    def setUp(self) -> None:
        self.transport = UrllibTransport(timeout=5.0)
        self.check = IntegrityCheck.for_pdf(min_bytes=512)

    def test_a_truncated_pdf_is_corruption_not_a_success(self) -> None:
        routes = {
            "/doc.pdf": Route(
                200,
                make_pdf(8192),
                "application/pdf",
                declared_length=8192,
                truncate_to=1000,
            )
        }
        with run_server(routes) as server:
            response = self.transport.send(Request(server.url_for("/doc.pdf")))
        self.assertTrue(response.truncated, "the transport must notice the short body")
        assessment = self.check.assess(response)
        self.assertIs(assessment.verdict, Verdict.CORRUPTED)
        self.assertEqual(assessment.reason, "truncated_body")
        self.assertFalse(assessment.retryable)
        self.assertEqual(assessment.size, 1000)
        self.assertEqual(response.declared_length, 8192)

    def test_a_complete_pdf_is_ok_and_its_checksum_matches_the_bytes(self) -> None:
        body = make_pdf(4096)
        with run_server({"/doc.pdf": Route(200, body, "application/pdf")}) as server:
            response = self.transport.send(Request(server.url_for("/doc.pdf")))
        assessment = self.check.assess(response)
        self.assertTrue(assessment.ok, assessment.description)
        self.assertEqual(assessment.checksum, "sha256:" + hashlib.sha256(body).hexdigest())
        self.assertEqual(assessment.size, len(body))

    def test_expired_session_login_page_served_as_200_is_corruption(self) -> None:
        with run_server(
            {"/doc.pdf": Route(200, LOGIN_PAGE, "text/html; charset=utf-8")}
        ) as server:
            response = self.transport.send(Request(server.url_for("/doc.pdf")))
        assessment = self.check.assess(response)
        self.assertIs(assessment.verdict, Verdict.CORRUPTED)
        self.assertFalse(assessment.retryable)
        self.assertIn(assessment.reason, {"content_type_mismatch", "html_payload"})

    def test_empty_200_is_corruption(self) -> None:
        with run_server({"/doc.pdf": Route(200, b"", "application/pdf")}) as server:
            response = self.transport.send(Request(server.url_for("/doc.pdf")))
        self.assertEqual(self.check.assess(response).reason, "empty_body")

    def test_429_over_http_is_retryable(self) -> None:
        with run_server({"/doc.pdf": Route(429, b"slow down", "text/plain")}) as server:
            response = self.transport.send(Request(server.url_for("/doc.pdf")))
        assessment = self.check.assess(response)
        self.assertIs(assessment.verdict, Verdict.RETRYABLE_FAILURE)
        self.assertTrue(assessment.retryable)

    def test_403_over_http_is_a_permanent_failure(self) -> None:
        with run_server({"/doc.pdf": Route(403, b"forbidden", "text/plain")}) as server:
            response = self.transport.send(Request(server.url_for("/doc.pdf")))
        assessment = self.check.assess(response)
        self.assertIs(assessment.verdict, Verdict.PERMANENT_FAILURE)
        self.assertFalse(assessment.retryable)
        self.assertNotEqual(assessment.verdict, Verdict.CORRUPTED)

    def test_404_over_http_is_a_permanent_failure(self) -> None:
        with run_server() as server:
            response = self.transport.send(Request(server.url_for("/gone.pdf")))
        self.assertIs(self.check.assess(response).verdict, Verdict.PERMANENT_FAILURE)

    def test_json_endpoint_validator_passes_and_fails(self) -> None:
        routes = {
            "/list.json": Route.json({"items": [1, 2, 3]}),
            "/broken.json": Route(200, b'{"items": [1, 2', "application/json"),
        }
        check = IntegrityCheck.for_json(root="object", min_bytes=2)
        with run_server(routes) as server:
            good = self.transport.send(Request(server.url_for("/list.json")))
            bad = self.transport.send(Request(server.url_for("/broken.json")))
        self.assertTrue(check.assess(good).ok)
        assessment = check.assess(bad)
        self.assertEqual(assessment.verdict, Verdict.CORRUPTED)
        self.assertTrue(assessment.reason.startswith("validator_failed"))

    def test_sniffing_catches_an_html_body_labelled_json(self) -> None:
        with run_server(
            {"/list.json": Route(200, LOGIN_PAGE, "application/json")}
        ) as server:
            response = self.transport.send(Request(server.url_for("/list.json")))
        assessment = IntegrityCheck.for_json(min_bytes=2).assess(response)
        self.assertEqual(assessment.verdict, Verdict.CORRUPTED)
        self.assertEqual(assessment.reason, "html_payload")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
