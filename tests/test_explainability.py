"""Tests for shared explainability evidence helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.text.explainability import (
    analyse_domain_indicators,
    find_legitimate_indicators,
    find_suspicious_phrases,
    top_model_terms,
)


class _FakeMatrix:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def toarray(self) -> np.ndarray:
        return self._values


class _FakeVectorizer:
    def transform(self, _texts: list[str]) -> _FakeMatrix:
        return _FakeMatrix(np.array([[1.0, 0.5, 0.0]]))

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(["alpha", "beta", "gamma"])


class _FakeEstimator:
    def __init__(self, coefficients: list[float]) -> None:
        self.coef_ = np.array([coefficients])


class _FakeCalibratedClassifier:
    def __init__(self, estimator: _FakeEstimator) -> None:
        self.estimator = estimator


class _FakeCalibratedModel:
    calibrated_classifiers_ = [
        _FakeCalibratedClassifier(_FakeEstimator([0.2, -0.4, 0.0])),
        _FakeCalibratedClassifier(_FakeEstimator([0.4, -0.2, 0.0])),
    ]


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

    def test_top_model_terms_unwraps_calibrated_linear_models(self) -> None:
        terms = top_model_terms(
            "alpha beta",
            _FakeVectorizer(),
            _FakeCalibratedModel(),
            top_n=2,
        )

        self.assertEqual([term["term"] for term in terms], ["alpha", "beta"])
        self.assertEqual(terms[0]["method"], "calibrated_linear_coefficient")


if __name__ == "__main__":
    unittest.main()
