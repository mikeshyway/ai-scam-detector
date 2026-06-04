"""Unified Detection Center page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_detection_center_page(root: Path, history: list[dict[str, object]]) -> None:
    st.subheader("Detection Center")
    st.write(
        "Use these lightweight educational checkers after or alongside the Simulation Lab. "
        "Each tool focuses on explanation, not enterprise-grade security enforcement."
    )

    email_tab, transcript_tab, voice_tab, phone_tab = st.tabs(
        [
            "A. Email Phishing Checker",
            "B. Transcript Scam Checker",
            "C. AI Voice / Deepfake Checker",
            "D. Phone Number Risk Checker",
        ]
    )

    with email_tab:
        from app.email_tab import render_email_tab

        render_email_tab(root, history)
    with transcript_tab:
        from app.transcript_tab import render_transcript_tab

        render_transcript_tab(root, history)
    with voice_tab:
        from app.audio_tab import render_audio_tab

        render_audio_tab(root, history)
    with phone_tab:
        from app.phone_risk_page import render_phone_risk_page

        render_phone_risk_page(root, history)
