"""Tests for shared explainability evidence helpers."""

from __future__ import annotations

import unittest

from src.text.explainability import (
    analyse_domain_indicators,
    find_legitimate_indicators,
    find_suspicious_phrases,
)


class ExplainabilityTests(unittest.TestCase):
    def test_domain_indicators_detect_brand_and_sender_mismatch(self) -> None:
        findings = analyse_domain_indicators(
            "Reset your account at https://paypal-security-login.xyz/login "
            "or contact support-maybank@gmail.com."
        )
        labels = {item["label"] for item in findings}

        self.assertIn("Brand Imitation Domain", labels)
        self.assertIn("Free Email Brand Sender", labels)

    def test_domain_indicators_feed_suspicious_phrase_pipeline(self) -> None:
        findings = find_suspicious_phrases("Open http://45.22.10.9/login now.")
        categories = {item["category"] for item in findings}
        phrases = {item["phrase"] for item in findings}

        self.assertIn("IP Address Link", categories)
        self.assertIn("45.22.10.9", phrases)

    def test_defanged_credential_url_gets_specific_domain_category(self) -> None:
        findings = analyse_domain_indicators(
            "Open hxxp://corporate-security-auth-portal-login[.]com/verify/index.html"
        )
        categories = {item["category"] for item in findings}
        phrases = {item["phrase"] for item in findings}

        self.assertIn("Credential Landing Page", categories)
        self.assertIn("corporate-security-auth-portal-login.com", phrases)

    def test_legitimate_indicators_detect_context_without_scoring(self) -> None:
        findings = find_legitimate_indicators(
            "Microsoft Teams meeting agenda from IT support. "
            "Do not enter your password outside official company systems."
        )
        types = {item["Type"] for item in findings}

        self.assertIn("Meeting Agenda", types)
        self.assertIn("Official Work Platform", types)
        self.assertIn("Department Workflow", types)
        self.assertIn("Safe Credential Warning", types)


if __name__ == "__main__":
    unittest.main()
