"""Tests for safe phone-provider diagnostics."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.phone.penipumy_client import PenipuApiError
from src.phone.providers.models import ProviderDiagnosticResult, diagnostic_rows, normalize_error_code
from src.phone.providers.penipumy_provider import test_penipumy_connection as run_penipumy_connection_test
from src.phone.providers.veriphone_provider import test_veriphone_connection as run_veriphone_connection_test


class PhoneProviderDiagnosticTests(unittest.TestCase):
    def test_error_code_normalization_covers_api_key_and_provider_failures(self) -> None:
        cases = [
            (None, "API key is not configured", "missing_key"),
            (402, "Payment Required: insufficient credits", "insufficient_credits"),
            (401, "Invalid key rejected", "authentication_failed"),
            (403, "Forbidden", "authentication_failed"),
            (429, "Rate limit reached", "rate_limited"),
            (None, "Lookup timed out", "timeout"),
            (None, "Network connection failed", "connection_failed"),
            (None, "Phone number is invalid", "invalid_number"),
            (200, "Malformed JSON response format", "invalid_response"),
            (503, "", "provider_error"),
        ]

        for status_code, error_message, expected in cases:
            with self.subTest(error_message=error_message):
                self.assertEqual(normalize_error_code(status_code, error_message), expected)

    def test_diagnostic_rows_show_key_variable_but_hide_key_value(self) -> None:
        diagnostic = ProviderDiagnosticResult(
            provider_id="veriphone",
            provider_name="Veriphone.io Carrier Lookup",
            configured=True,
            key_detected=True,
            key_source="Environment variable: VERIPHONE_API_KEY",
            key_variable="VERIPHONE_API_KEY",
            reachable=True,
            authentication_status="Not rejected",
            request_accepted="Yes",
            http_status=200,
            provider_success=True,
            response_time_ms=125.0,
            fields_returned=5,
            fields_populated=4,
            rate_limit="remaining=99",
            fallback_used=False,
            request_id="req-123",
            error_code="none",
            error_message="",
        )

        rows = diagnostic_rows(diagnostic)
        key_row = next(row for row in rows if row["Check"] == "Key variable")
        combined_text = " ".join(str(value) for row in rows for value in row.values())

        self.assertEqual(key_row["Result"], "VERIPHONE_API_KEY")
        self.assertEqual(key_row["Detail"], "Value hidden")
        self.assertNotIn("secret", combined_text.lower())

    def test_missing_key_diagnostics_do_not_attempt_live_lookup(self) -> None:
        veriphone = run_veriphone_connection_test(
            "+60162404384",
            "",
            key_source="Not configured",
            key_variable="-",
        )
        penipumy = run_penipumy_connection_test(
            "+60162404384",
            "",
            key_source="Not configured",
            key_variable="-",
        )

        for diagnostic in (veriphone, penipumy):
            with self.subTest(provider=diagnostic.provider_id):
                self.assertFalse(diagnostic.configured)
                self.assertFalse(diagnostic.key_detected)
                self.assertEqual(diagnostic.authentication_status, "No")
                self.assertEqual(diagnostic.error_code, "missing_key")
                self.assertEqual(diagnostic.key_variable, "-")

    def test_penipumy_authentication_failure_is_reported_without_key_leakage(self) -> None:
        with patch(
            "src.phone.providers.penipumy_provider.fetch_phone_reputation",
            side_effect=PenipuApiError(403, "PenipuMY API returned 403: Invalid API key"),
        ):
            diagnostic = run_penipumy_connection_test(
                "+60162404384",
                "secret-penipu-key",
                key_source="Session input",
                key_variable="session",
            )

        self.assertTrue(diagnostic.configured)
        self.assertEqual(diagnostic.http_status, 403)
        self.assertEqual(diagnostic.error_code, "authentication_failed")
        self.assertEqual(diagnostic.authentication_status, "Rejected")
        self.assertNotIn("secret-penipu-key", diagnostic.error_message)


if __name__ == "__main__":
    unittest.main()
