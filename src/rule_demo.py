"""Educational rule-based demo predictions for non-email fallback flows."""

from __future__ import annotations

from src.explainability import find_suspicious_phrases


def rule_based_text_prediction(text: str) -> dict[str, object]:
    findings = find_suspicious_phrases(text)
    score = min(0.95, 0.15 + 0.12 * len(findings))
    label = 1 if score >= 0.5 else 0
    confidence = score if label == 1 else 1.0 - score
    return {
        "label": label,
        "label_name": "Suspicious" if label == 1 else "Legitimate",
        "confidence": confidence,
        "probabilities": {"Legitimate": 1.0 - score, "Suspicious": score},
        "model_name": "Educational rule demo",
        "findings": findings,
    }
