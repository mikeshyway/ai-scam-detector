"""Tests for the Veriphone.io phone metadata client."""

from __future__ import annotations

import sys
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    spec = spec_from_file_location(module_name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {relative_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


veriphone_client = _load_module("veriphone_client_under_test", "src/phone/veriphone_client.py")


class VeriphoneClientTests(unittest.TestCase):
    def test_lookup_uses_bearer_header_and_static_mode(self) -> None:
        class FakeResponse:
            status_code = 200
            reason = "OK"
            headers: dict[str, str] = {}

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "status": "success",
                    "phone": "+14169670000",
                    "phone_valid": True,
                    "phone_type": "fixed_line",
                    "carrier": "Bell",
                    "country_code": "CA",
                    "e164": "+14169670000",
                    "local_number": "(416) 967-0000",
                    "mode": "static",
                }

        fake_get = Mock(return_value=FakeResponse())
        fake_requests = types.SimpleNamespace(
            get=fake_get,
            Timeout=TimeoutError,
            ConnectionError=ConnectionError,
            RequestException=Exception,
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = veriphone_client.lookup_veriphone_phone("+14169670000", "secret-veriphone-key")

        self.assertTrue(result["ok"])
        self.assertEqual(result["status_code"], 200)
        fake_get.assert_called_once()
        _, kwargs = fake_get.call_args
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer secret-veriphone-key"})
        self.assertEqual(kwargs["params"], {"phone": "+14169670000", "mode": "static"})

    def test_payment_required_preserves_message_without_key(self) -> None:
        class FakeResponse:
            status_code = 402
            reason = "Payment Required"
            headers: dict[str, str] = {}

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "status": "error",
                    "code": 402,
                    "type": "PaymentRequired",
                    "message": "Insufficient credits",
                }

        fake_requests = types.SimpleNamespace(
            get=Mock(return_value=FakeResponse()),
            Timeout=TimeoutError,
            ConnectionError=ConnectionError,
            RequestException=Exception,
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            result = veriphone_client.lookup_veriphone_phone("+14169670000", "secret-veriphone-key")

        self.assertFalse(result["ok"])
        self.assertEqual(result["status_code"], 402)
        self.assertIn("Insufficient credits", str(result["error"]))
        self.assertNotIn("secret-veriphone-key", str(result["error"]))


if __name__ == "__main__":
    unittest.main()
