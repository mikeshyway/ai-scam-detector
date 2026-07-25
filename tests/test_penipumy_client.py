"""Tests for the low-level PenipuMY API client."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock, patch

from src.phone.penipumy_client import (
    PenipuApiError,
    PenipuConfigurationError,
    fetch_phone_reputation,
)


class PenipuMYClientTests(unittest.TestCase):
    def test_fetch_uses_api_key_header_and_normalized_query(self) -> None:
        class FakeResponse:
            status_code = 200
            reason = "OK"
            headers = {
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "99",
            }

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "phone": "60162404384",
                    "police_report_count": 0,
                    "verified_report_count": 0,
                    "spam": False,
                    "fraud": False,
                }

        fake_get = Mock(return_value=FakeResponse())
        fake_requests = types.SimpleNamespace(
            get=fake_get,
            Timeout=TimeoutError,
            ConnectionError=ConnectionError,
            RequestException=Exception,
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = fetch_phone_reputation("016-240 4384", "secret-penipu-key", timeout_seconds=7)

        self.assertEqual(result.query, "60162404384")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.rate_limit, {"limit": "100", "remaining": "99"})
        fake_get.assert_called_once()
        endpoint, = fake_get.call_args.args
        self.assertEqual(endpoint, "https://penipu.my/api/v1/phone")
        self.assertEqual(fake_get.call_args.kwargs["headers"], {"X-API-Key": "secret-penipu-key"})
        self.assertEqual(fake_get.call_args.kwargs["params"], {"q": "60162404384"})
        self.assertEqual(fake_get.call_args.kwargs["timeout"], 7)

    def test_missing_api_key_fails_before_network_request(self) -> None:
        with self.assertRaises(PenipuConfigurationError):
            fetch_phone_reputation("+60162404384", "")

    def test_api_error_preserves_status_without_exposing_key(self) -> None:
        class FakeResponse:
            status_code = 403
            reason = "Forbidden"
            headers: dict[str, str] = {}

            @staticmethod
            def json() -> dict[str, object]:
                return {"error": "Invalid API key"}

        fake_get = Mock(return_value=FakeResponse())
        fake_requests = types.SimpleNamespace(
            get=fake_get,
            Timeout=TimeoutError,
            ConnectionError=ConnectionError,
            RequestException=Exception,
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaises(PenipuApiError) as context:
                fetch_phone_reputation("+60162404384", "secret-penipu-key")

        self.assertEqual(context.exception.status_code, 403)
        self.assertIn("PenipuMY API returned 403", str(context.exception))
        self.assertNotIn("secret-penipu-key", str(context.exception))


if __name__ == "__main__":
    unittest.main()
