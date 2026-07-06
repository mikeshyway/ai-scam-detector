"""Manual phone-number reputation demo page."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_components import (
    apply_chart_theme,
    get_demo_data,
    render_analysis_ready,
    render_content_card_close,
    render_content_card_open,
    render_detection_tool_intro,
    render_info_banner,
    render_result_card,
    render_section_header,
)


def _normalize_phone(value: str) -> str:
    return re.sub(r"\D+", "", value)


def _risk_from_number(value: str, demo_phones: pd.DataFrame) -> dict[str, object]:
    normalized = _normalize_phone(value)
    if not normalized:
        return {"risk_score": 0, "tag": "no number entered", "reports": 0, "source": "manual input"}

    demo_phones = demo_phones.copy()
    demo_phones["normalized"] = demo_phones["phone_number"].map(_normalize_phone)
    match = demo_phones[demo_phones["normalized"].str.endswith(normalized[-6:])]
    if not match.empty:
        row = match.iloc[0].to_dict()
        return row

    unusual_length = len(normalized) < 9 or len(normalized) > 15
    repeated_digits = bool(re.search(r"(\d)\1{5,}", normalized))
    score = 20 + (25 if unusual_length else 0) + (25 if repeated_digits else 0)
    return {
        "risk_score": min(score, 95),
        "tag": "unknown number heuristic",
        "reports": 0,
        "source": "manual heuristic demo",
    }


def _phone_indicator_rows(value: str, result: dict[str, object]) -> pd.DataFrame:
    normalized = _normalize_phone(value)
    if not normalized:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    reports = int(result.get("reports", 0) or 0)
    risk_score = int(result.get("risk_score", 0) or 0)
    tag = str(result.get("tag", "unknown")).strip()
    source = str(result.get("source", "manual heuristic demo")).strip()

    if reports > 0:
        rows.append(
            {
                "Indicator": "Reported number match",
                "Category": "Reputation",
                "Specific Tactic": tag,
                "Direction": "Raises caller risk",
                "Evidence": f"{reports} synthetic report(s)",
                "Explanation": "Numbers with repeated scam reports should be verified through an official channel before responding.",
            }
        )

    if risk_score >= 70:
        rows.append(
            {
                "Indicator": "High synthetic risk score",
                "Category": "Reputation",
                "Specific Tactic": "Complaint-weighted risk",
                "Direction": "Raises caller risk",
                "Evidence": f"{risk_score}%",
                "Explanation": "The demo reputation record places this caller in the high-risk range.",
            }
        )

    if len(normalized) < 9 or len(normalized) > 15:
        rows.append(
            {
                "Indicator": "Unusual number length",
                "Category": "Number pattern",
                "Specific Tactic": "Malformed or incomplete caller ID",
                "Direction": "Raises caller risk",
                "Evidence": f"{len(normalized)} digits",
                "Explanation": "Very short or very long numbers may indicate malformed, spoofed, or non-standard caller ID data.",
            }
        )

    if re.search(r"(\d)\1{5,}", normalized):
        rows.append(
            {
                "Indicator": "Repeated digit pattern",
                "Category": "Number pattern",
                "Specific Tactic": "Artificial-looking caller ID",
                "Direction": "Raises caller risk",
                "Evidence": "6+ repeated digits",
                "Explanation": "Long repeated digit sequences can indicate test numbers, spoofed numbers, or low-quality caller ID data.",
            }
        )

    if not (normalized.startswith("60") or normalized.startswith("0")):
        rows.append(
            {
                "Indicator": "Region mismatch check",
                "Category": "Caller identity",
                "Specific Tactic": "Non-local prefix for Malaysia-focused demo",
                "Direction": "Needs verification",
                "Evidence": f"+{normalized[:2]} prefix",
                "Explanation": "A non-local prefix is not automatically suspicious, but it should match the identity claimed by the caller.",
            }
        )

    if not rows:
        rows.append(
            {
                "Indicator": "No strong pattern",
                "Category": "Caller identity",
                "Specific Tactic": "Manual review",
                "Direction": "Lower caller risk",
                "Evidence": source,
                "Explanation": "No strong pattern was found in the demo checker. Still verify sensitive requests through official channels.",
            }
        )

    return pd.DataFrame(rows)


def render_phone_risk_page(root: Path, history: list[dict[str, object]]) -> None:
    render_detection_tool_intro(
        title="Phone Number",
        description=(
            "Assess the potential risk of a caller using phone number characteristics "
            "and user-provided context to highlight suspicious patterns that may require "
            "additional verification."
        ),
        icon="solar:phone-calling-rounded-bold-duotone",
        accent="orange",
    )

    render_section_header(
        "Review a caller number",
        "A lightweight reputation and pattern check designed for educational demonstration.",
        "Caller identity",
    )
    render_info_banner(
        "This is a manual educational checker using synthetic reputation data. "
        "It does not connect to telecom systems.",
        kind="warning",
        code="SCOPE",
    )

    demo_phones = get_demo_data()["phones"]
    render_content_card_open("violet")
    number = st.text_input("Enter a phone number to check", placeholder="+60 12 345 6789")
    if st.button("Check number", type="primary"):
        result = _risk_from_number(number, demo_phones)
        score = int(result["risk_score"])
        render_analysis_ready("Phone risk check complete - results ready below")
        render_result_card(
            "Caller ID risk result",
            score,
            f"Tag: {result['tag']}. Reports in demo data: {int(result['reports'])}.",
        )
        st.dataframe(
            _phone_indicator_rows(number, result),
            hide_index=True,
            use_container_width=True,
        )
        history.insert(
            0,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "Phone",
                "prediction": "High risk" if score >= 70 else "Medium risk" if score >= 40 else "Lower risk",
                "confidence": score,
                "model": "Manual reputation demo",
                "preview": number,
            },
        )
    render_content_card_close()

    render_section_header("Synthetic reputation table", eyebrow="Reference data")
    render_content_card_open("green")
    st.dataframe(demo_phones, hide_index=True, use_container_width=True)
    fig = px.histogram(demo_phones, x="risk_score", nbins=8, title="Demo phone risk distribution")
    st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
    render_content_card_close()

    render_section_header("Realistic integration alternatives", eyebrow="Future scope")
    render_info_banner(
        "To build a true caller-ID style system, you would need a separate Android app, VoIP/Twilio call-flow "
        "integration, or a campus-reported scam-number database. Those require explicit consent, platform "
        "permissions, and legal review.",
        kind="info",
        code="NEXT",
    )
