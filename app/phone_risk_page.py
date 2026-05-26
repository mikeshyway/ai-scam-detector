"""Manual phone-number reputation demo page."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_components import get_demo_data, render_demo_notice


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


def render_phone_risk_page(root: Path, history: list[dict[str, object]]) -> None:
    render_demo_notice(root)
    st.subheader("Phone Number Risk Demo")
    st.warning(
        "Streamlit cannot automatically intercept phone calls, record conversations, or access caller ID. "
        "This page is a safe manual demo using synthetic reputation data."
    )
    st.markdown(
        """
        **Phone demo integration options**

        - Same WiFi: open `http://<laptop-ip>:8501` on the phone.
        - Public demo: expose Streamlit with a temporary HTTPS tunnel such as ngrok.
        - Real caller ID integration: requires a separate Android/iOS app or telecom/VoIP API.
        """
    )

    demo_phones = get_demo_data()["phones"]
    number = st.text_input("Enter a phone number to check", placeholder="+60 12 345 6789")
    if st.button("Check number", type="primary"):
        result = _risk_from_number(number, demo_phones)
        score = int(result["risk_score"])
        if score >= 70:
            st.error(f"High risk score: {score}/100")
        elif score >= 40:
            st.warning(f"Medium risk score: {score}/100")
        else:
            st.success(f"Lower risk score: {score}/100")
        st.write(f"Tag: **{result['tag']}**")
        st.write(f"Reports in demo data: **{int(result['reports'])}**")
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

    st.subheader("Synthetic Reputation Table")
    st.dataframe(demo_phones, hide_index=True, use_container_width=True)
    fig = px.histogram(demo_phones, x="risk_score", nbins=8, title="Demo phone risk distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Realistic Alternatives")
    st.write(
        "To build a true caller-ID style system, you would need a separate Android app, VoIP/Twilio call-flow "
        "integration, or a campus-reported scam-number database. Those require explicit consent, platform "
        "permissions, and legal review."
    )
