"""Phone-number reputation lookup page."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.ui_components import (
    render_analysis_ready,
    render_content_card_close,
    render_content_card_open,
    render_detection_tool_intro,
    render_info_banner,
    render_result_card,
    render_section_header,
)
from src.phone.phone_lookup import lookup_phone, normalise_phone_query, validate_phone_query


def _configured_api_key() -> str:
    """Read a PenipuMY API key from environment variables or Streamlit secrets."""

    env_key = os.environ.get("PENIPU_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        direct_secret = str(st.secrets.get("PENIPU_API_KEY", "")).strip()
        if direct_secret:
            return direct_secret
    except Exception:
        pass

    try:
        section = st.secrets.get("penipu", {})
        section_key = str(section.get("api_key", "")).strip()
        if section_key:
            return section_key
    except Exception:
        pass

    return ""


def _source_label(source: str) -> str:
    labels = {
        "penipumy_api": "PenipuMY API",
        "local_fallback": "Local fallback dataset",
        "unknown_fallback": "Unknown fallback",
    }
    return labels.get(source, source.replace("_", " ").title())


def _render_phone_step(index: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="email-step-header">
            <span>{index}</span>
            <div>
                <h3>{title}</h3>
                <p>{body}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_api_setup() -> str:
    configured_key = _configured_api_key()

    with st.expander("PenipuMY API setup", expanded=not bool(configured_key)):
        if configured_key:
            st.success("Using a PenipuMY API key from environment variables or Streamlit secrets.")
        else:
            render_info_banner(
                "No PenipuMY API key was detected. The lookup will automatically use the local fallback "
                "dataset when you check a number.",
                kind="warning",
                code="FALLBACK",
            )

        manual_key = st.text_input(
            "Session API key",
            type="password",
            placeholder="Paste PenipuMY API key for this session",
            help="The key is sent only to PenipuMY through the X-API-Key request header.",
        )
        st.caption(
            "Local setup: `$env:PENIPU_API_KEY='your_key_here'` before launching Streamlit. "
            "Do not commit API keys to GitHub."
        )

    return manual_key.strip() or configured_key


def _business_rows(record: dict[str, Any]) -> pd.DataFrame:
    business = record.get("business") if isinstance(record.get("business"), dict) else None
    if business:
        fields = [
            ("Display Name", business.get("display_name")),
            ("Tier", business.get("tier")),
            ("Brand", business.get("brand_name")),
            ("Branch", business.get("branch_name")),
            ("Address", business.get("address")),
            ("Website", business.get("website")),
            ("Maps Place", business.get("place_url")),
            ("Rating", business.get("rating")),
            ("Review Count", business.get("review_count")),
            ("Opening Hours", business.get("opening_hours_status")),
            ("Scam Advisory", business.get("scam_alert_banner")),
            ("Spoofing Reports", business.get("spoofing_report_count")),
        ]
    else:
        fields = [
            ("Business Name", record.get("business_name")),
            ("Business Tier", record.get("business_tier")),
            ("Source", record.get("source")),
        ]

    return pd.DataFrame(
        [{"Field": field, "Value": value} for field, value in fields if value not in (None, "")]
    )


def _record_phone_result(
    history: list[dict[str, object]],
    phone_number: str,
    result: dict[str, Any],
) -> None:
    risk = dict(result.get("risk", {}))
    explanation = dict(result.get("explanation", {}))
    indicators = explanation.get("indicators", [])
    flags = ", ".join(
        str(row.get("Indicator", ""))
        for row in indicators
        if isinstance(row, dict) and row.get("Indicator")
    )

    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Phone",
            "prediction": risk.get("risk_level", "Unknown"),
            "confidence": float(risk.get("risk_score", 0)),
            "model": "PenipuMY API + fallback reputation rules",
            "source": _source_label(str(result.get("source", ""))),
            "preview": phone_number,
            "flags": flags,
            "explanation": explanation.get("summary", ""),
        },
    )


def _render_result(root: Path, lookup_result: dict[str, Any], claimed_identity: str, history: list[dict[str, object]]) -> None:
    record = dict(lookup_result.get("record", {}))
    risk = dict(lookup_result.get("risk", {}))
    explanation = dict(lookup_result.get("explanation", {}))
    metrics = dict(explanation.get("metrics", {}))
    source = str(lookup_result.get("source", "unknown_fallback"))
    fallback_reason = str(lookup_result.get("fallback_reason") or "")
    phone = str(record.get("phone") or "")
    score = float(risk.get("risk_score", 0))
    risk_level = str(risk.get("risk_level", "Unknown"))

    render_analysis_ready("Phone reputation check complete - results ready below")

    if risk_level == "Unknown":
        render_info_banner(
            "This number was not found in the live API or local dataset. This does not guarantee the number is safe.",
            kind="warning",
            code="UNKNOWN",
        )
    else:
        render_result_card(
            "Caller reputation result",
            score,
            str(explanation.get("summary", "No explanation returned.")),
        )

    _render_phone_step("03", "Review Evidence", "Inspect source, fallback status, reports, and recommended action.")

    render_content_card_open("violet")
    cols = st.columns(3)
    cols[0].metric("Risk Level", risk_level)
    cols[1].metric("Source Used", _source_label(source))
    cols[2].metric("Reports Found", int(record.get("police_report_count", 0) or 0) + int(record.get("verified_report_count", 0) or 0))

    cols = st.columns(3)
    cols[0].metric("Business Status", metrics.get("Business Status", "No match"))
    cols[1].metric("Fallback Status", "Fallback used" if source != "penipumy_api" else "Live API")
    cols[2].metric("Review Required", "Yes" if risk.get("review_required", True) else "No")

    if fallback_reason:
        render_info_banner(
            f"Live API unavailable or not configured. Reason: {fallback_reason}. "
            f"Using {_source_label(source).lower()} instead.",
            kind="warning",
            code="FALLBACK",
        )

    if claimed_identity.strip():
        st.caption(f"Claimed caller identity: {claimed_identity.strip()}")

    st.markdown("**Explanation**")
    st.write(explanation.get("summary", "No explanation returned."))
    st.markdown("**Recommended Action**")
    st.write(explanation.get("recommended_action", risk.get("recommended_action", "")))

    indicators = explanation.get("indicators", [])
    if indicators:
        st.dataframe(pd.DataFrame(indicators), hide_index=True, use_container_width=True)
    else:
        st.info("No caller reputation indicators were returned.")

    rate_limit = dict(lookup_result.get("rate_limit", {}))
    if rate_limit.get("limit") or rate_limit.get("remaining"):
        st.caption(
            f"PenipuMY rate limit: {rate_limit.get('remaining') or '?'} request(s) remaining "
            f"out of {rate_limit.get('limit') or '?'} today."
        )

    render_content_card_close()

    business_df = _business_rows(record)
    if not business_df.empty:
        render_section_header(
            "Caller identity details",
            "Business or fallback identity fields associated with the queried number.",
            "Business status",
        )
        render_content_card_open("green")
        st.dataframe(business_df, hide_index=True, use_container_width=True)
        render_content_card_close()

    with st.expander("Normalized lookup object", expanded=False):
        st.code(json.dumps(lookup_result, indent=2, ensure_ascii=False, default=str), language="json")

    _record_phone_result(history, phone, lookup_result)


def render_phone_risk_page(root: Path, history: list[dict[str, object]]) -> None:
    render_detection_tool_intro(
        title="Phone Number",
        description=(
            "Check caller reputation using PenipuMY when available, then fall back to a local processed "
            "dataset and transparent rules so the system remains usable during API limits or network failures."
        ),
        icon="solar:phone-calling-rounded-bold-duotone",
        accent="orange",
    )

    render_section_header(
        "Phone number reputation lookup",
        "Phone numbers are reputation lookups, not ML text classifiers. The system explains database evidence and fallback status.",
        "Caller reputation",
    )

    api_key = _render_api_setup()

    _render_phone_step("01", "Enter Phone Number", "Use +60, 60, or 0 prefixes for Malaysia-focused phone lookups.")
    render_content_card_open("violet")
    number = st.text_input(
        "Phone number",
        placeholder="+60 12 345 6789",
        help="PenipuMY accepts phone numbers with 8 to 15 digits.",
    )
    claimed_identity = st.text_input(
        "Claimed caller identity (optional)",
        placeholder="Example: bank officer, courier, university support, unknown caller",
    )

    ok, validation_message = validate_phone_query(number)
    normalized = normalise_phone_query(number)
    if number.strip() and not ok:
        st.warning(validation_message)
    elif normalized:
        st.caption(f"Normalized lookup query: `{normalized}`")
    render_content_card_close()

    _render_phone_step("02", "Check Reputation", "Try PenipuMY first, then local fallback, then unknown safe-demo fallback.")
    check_button = st.button("Check Phone Reputation", type="primary", use_container_width=True)

    render_info_banner(
        "This module does not intercept calls or verify telecom ownership. It checks reputation evidence and teaches safe response.",
        kind="info",
        code="SCOPE",
    )

    if not check_button:
        return

    if not ok:
        st.warning(validation_message)
        return

    with st.spinner("Checking caller reputation..."):
        try:
            lookup_result = lookup_phone(normalized, root, api_key=api_key)
        except ValueError as exc:
            st.warning(str(exc))
            return

    _render_result(root, lookup_result, claimed_identity, history)
