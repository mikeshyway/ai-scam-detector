"""Tests for Phone Number tab API-key source resolution."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app import phone_tab


class PhoneProviderKeyResolutionTests(unittest.TestCase):
    def test_session_key_overrides_environment_key(self) -> None:
        with (
            patch.dict(os.environ, {"VERIPHONE_API_KEY": "env-veriphone-key"}, clear=True),
            patch.object(phone_tab, "_secret_value", return_value=""),
            patch.object(phone_tab, "_section_secret", return_value=""),
        ):
            result = phone_tab._resolve_phone_provider_key("veriphone", "session-veriphone-key")

        self.assertTrue(result["configured"])
        self.assertEqual(result["key"], "session-veriphone-key")
        self.assertEqual(result["source"], "Session input")
        self.assertEqual(result["variable"], "session")
        self.assertNotIn("session-veriphone-key", result["masked_key"])
        self.assertTrue(str(result["masked_key"]).endswith("key"))

    def test_veriphone_environment_key_is_supported(self) -> None:
        with (
            patch.dict(os.environ, {"VERIPHONE_API_KEY": "env-veriphone-key"}, clear=True),
            patch.object(phone_tab, "_secret_value", return_value=""),
            patch.object(phone_tab, "_section_secret", return_value=""),
        ):
            result = phone_tab._resolve_phone_provider_key("veriphone")

        self.assertTrue(result["configured"])
        self.assertEqual(result["key"], "env-veriphone-key")
        self.assertEqual(result["source"], "Environment variable: VERIPHONE_API_KEY")
        self.assertEqual(result["variable"], "VERIPHONE_API_KEY")

    def test_penipumy_current_and_legacy_environment_names_are_supported(self) -> None:
        for env_name in ("PENIPUMY_API_KEY", "PENIPU_API_KEY"):
            with self.subTest(env_name=env_name):
                with (
                    patch.dict(os.environ, {env_name: "env-penipu-key"}, clear=True),
                    patch.object(phone_tab, "_secret_value", return_value=""),
                    patch.object(phone_tab, "_section_secret", return_value=""),
                ):
                    result = phone_tab._resolve_phone_provider_key("penipumy")

                self.assertTrue(result["configured"])
                self.assertEqual(result["key"], "env-penipu-key")
                self.assertEqual(result["source"], f"Environment variable: {env_name}")
                self.assertEqual(result["variable"], env_name)

    def test_streamlit_secret_section_is_supported(self) -> None:
        def fake_section_secret(section_name: str, field_name: str) -> str:
            if (section_name, field_name) == ("penipumy", "api_key"):
                return "section-penipu-key"
            return ""

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(phone_tab, "_secret_value", return_value=""),
            patch.object(phone_tab, "_section_secret", side_effect=fake_section_secret),
        ):
            result = phone_tab._resolve_phone_provider_key("penipumy")

        self.assertTrue(result["configured"])
        self.assertEqual(result["key"], "section-penipu-key")
        self.assertEqual(result["source"], "Streamlit secret: [penipumy].api_key")
        self.assertEqual(result["variable"], "api_key")

    def test_missing_key_returns_safe_not_configured_metadata(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(phone_tab, "_secret_value", return_value=""),
            patch.object(phone_tab, "_section_secret", return_value=""),
        ):
            result = phone_tab._resolve_phone_provider_key("veriphone")

        self.assertFalse(result["configured"])
        self.assertEqual(result["key"], "")
        self.assertEqual(result["source"], "Not configured")
        self.assertEqual(result["variable"], "-")
        self.assertEqual(result["masked_key"], "Not available")


if __name__ == "__main__":
    unittest.main()
