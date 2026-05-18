"""Explainability helpers for student-facing scam awareness feedback."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SuspiciousPattern:
    phrase: str
    category: str
    reason: str


PATTERNS: tuple[SuspiciousPattern, ...] = (
    SuspiciousPattern("urgent", "Urgency", "Scams often pressure victims to act quickly."),
    SuspiciousPattern("immediately", "Urgency", "High-pressure language can reduce careful checking."),
    SuspiciousPattern("within 24 hours", "Urgency", "Short deadlines are common in phishing messages."),
    SuspiciousPattern("final warning", "Threat", "Threatening consequences is a common manipulation tactic."),
    SuspiciousPattern("account suspended", "Threat", "Fake account problems are often used to steal logins."),
    SuspiciousPattern("verify your account", "Credential request", "Credential verification links can be phishing."),
    SuspiciousPattern("password", "Credential request", "Legitimate staff should not ask for passwords."),
    SuspiciousPattern("otp", "Credential request", "One-time passwords should never be shared."),
    SuspiciousPattern("one time password", "Credential request", "One-time passwords should never be shared."),
    SuspiciousPattern("login", "Credential request", "Login requests should be checked against the official site."),
    SuspiciousPattern("bank account", "Payment", "Financial details are high-risk information."),
    SuspiciousPattern("wire transfer", "Payment", "Unusual transfer requests are a common scam sign."),
    SuspiciousPattern("gift card", "Payment", "Gift cards are a common irreversible payment method in scams."),
    SuspiciousPattern("cryptocurrency", "Payment", "Crypto payment requests are difficult to reverse."),
    SuspiciousPattern("tuition payment", "Payment", "Student payment scams often impersonate university finance teams."),
    SuspiciousPattern("scholarship", "Impersonation", "Scholarship offers can be used as bait for personal data."),
    SuspiciousPattern("internship offer", "Impersonation", "Fake job offers often target students."),
    SuspiciousPattern("dean", "Authority", "Authority names can be used to make fake requests feel official."),
    SuspiciousPattern("professor", "Authority", "Academic impersonation is relevant to students."),
    SuspiciousPattern("do not tell anyone", "Secrecy", "Requests for secrecy are a major social engineering sign."),
    SuspiciousPattern("confidential", "Secrecy", "Secrecy pressure can isolate the victim."),
    SuspiciousPattern("click here", "Link", "Generic link prompts should be checked carefully."),
    SuspiciousPattern("urltoken", "Link", "External URLs can lead to credential harvesting pages."),
    SuspiciousPattern("emailtoken", "Identity", "Email addresses should be checked against official domains."),
    SuspiciousPattern("moneytoken", "Payment", "Money amounts can indicate payment pressure or financial bait."),
    SuspiciousPattern("phonetoken", "Contact switch", "Moving channels can bypass normal verification."),
    SuspiciousPattern("kindly", "Tone", "Some scam templates use unusually formal phrasing."),
    SuspiciousPattern("winner", "Reward", "Prize language is often used as bait."),
    SuspiciousPattern("selected", "Reward", "Unexpected selection or rewards can be suspicious."),
)


def find_suspicious_phrases(text: str) -> list[dict[str, object]]:
    """Find suspicious words and phrases with category-specific explanations."""

    if not text:
        return []

    lowered = text.lower()
    findings: list[dict[str, object]] = []
    for pattern in PATTERNS:
        phrase = pattern.phrase.lower()
        regex = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
        for match in regex.finditer(lowered):
            findings.append(
                {
                    "phrase": text[match.start() : match.end()],
                    "category": pattern.category,
                    "reason": pattern.reason,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    findings.sort(key=lambda item: (int(item["start"]), -int(item["end"])))
    deduped: list[dict[str, object]] = []
    seen: set[tuple[int, int, str]] = set()
    for item in findings:
        key = (int(item["start"]), int(item["end"]), str(item["category"]))
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def highlighted_html(text: str, findings: list[dict[str, object]]) -> str:
    """Return escaped HTML with suspicious spans highlighted."""

    if not text:
        return ""
    if not findings:
        return html.escape(text).replace("\n", "<br>")

    spans = sorted(
        [(int(item["start"]), int(item["end"]), str(item["category"])) for item in findings],
        key=lambda item: item[0],
    )
    merged: list[tuple[int, int, str]] = []
    cursor_end = -1
    for start, end, category in spans:
        if start < cursor_end:
            continue
        merged.append((start, end, category))
        cursor_end = end

    pieces: list[str] = []
    cursor = 0
    for start, end, category in merged:
        pieces.append(html.escape(text[cursor:start]))
        title = html.escape(category)
        phrase = html.escape(text[start:end])
        pieces.append(f'<mark title="{title}">{phrase}</mark>')
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces).replace("\n", "<br>")


def educational_summary(
    label: str,
    confidence: float | None,
    findings: list[dict[str, object]],
) -> str:
    """Create a short student-friendly explanation for a prediction."""

    count = len(findings)
    if confidence is None:
        if count:
            return f"Demo mode found {count} suspicious pattern(s). Train the ML model for a real prediction."
        return "No strong rule-based scam signals were found. Train the ML model for a real prediction."

    percent = round(confidence * 100, 1)
    if "Suspicious" in label or "AI-generated" in label:
        return (
            f"The model classified this as suspicious with {percent}% confidence. "
            f"It also found {count} explainable warning pattern(s)."
        )
    return (
        f"The model classified this as lower risk with {percent}% confidence. "
        f"Still verify sender identity and links before acting."
    )


def rule_based_text_prediction(text: str) -> dict[str, object]:
    """Fallback educational demo when trained text models are not available."""

    findings = find_suspicious_phrases(text)
    score = min(0.95, 0.15 + 0.12 * len(findings))
    label = 1 if score >= 0.5 else 0
    confidence = score if label == 1 else 1.0 - score
    return {
        "label": label,
        "label_name": "Suspicious" if label == 1 else "Legitimate",
        "confidence": confidence,
        "probabilities": {
            "Legitimate": 1.0 - score,
            "Suspicious": score,
        },
        "model_name": "Educational rule demo",
        "findings": findings,
    }


def top_model_terms(
    text: str,
    vectorizer: Any,
    model: Any,
    *,
    top_n: int = 10,
) -> list[dict[str, object]]:
    """Return influential terms from the trained model when the estimator exposes weights."""

    try:
        X = vectorizer.transform([text])
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        active = X.toarray()[0]
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            return []

        weights = None
        if hasattr(model, "feature_log_prob_") and model.feature_log_prob_.shape[0] >= 2:
            weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        elif hasattr(model, "feature_importances_"):
            weights = model.feature_importances_

        if weights is None:
            return []

        scores = active[active_indices] * weights[active_indices]
        order = np.argsort(np.abs(scores))[::-1][:top_n]
        return [
            {
                "term": str(feature_names[active_indices[index]]),
                "score": float(scores[index]),
            }
            for index in order
        ]
    except Exception:
        return []

