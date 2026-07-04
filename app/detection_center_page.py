"""Unified Detection Center page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui_components import render_feature_card, render_section_header


def render_detection_center_page(root: Path, history: list[dict[str, object]]) -> None:
    render_section_header(
        "Detection center",
        "Choose the evidence channel you want to inspect. Each checker uses channel-specific indicators and explanations.",
        "Evidence analysis",
    )

    email_tab, transcript_tab, voice_tab, phone_tab = st.tabs(
        [
            "Email and messages",
            "Voice transcripts",
            "AI-generated speech",
            "Phone number risk",
        ]
    )

    with email_tab:
        render_feature_card(
            "Email phishing and message checker",
            "TF-IDF classification and explainable phrase analysis for pasted or uploaded text.",
            "01",
        )
        from app.email_tab import render_email_tab

        render_email_tab(root, history)
    with transcript_tab:
        render_feature_card(
            "Call and meeting transcript checker",
            "Detects urgency, authority pressure, secrecy, credential requests, and payment language.",
            "02",
        )
        from app.transcript_tab import render_transcript_tab

        render_transcript_tab(root, history)
    with voice_tab:
        render_feature_card(
            "AI-generated speech checker",
            "Uses MFCC features and an SVM model to inspect uploaded WAV or FLAC speech recordings.",
            "03",
        )
        from app.audio_deepseek_tab import render_audio_deepseek_tab

        render_audio_deepseek_tab(root, history)
    with phone_tab:
        render_feature_card(
            "Phone number caller-risk checker",
            "Compares a manually entered number with synthetic reputation data and lightweight risk heuristics.",
            "04",
        )
        from app.phone_risk_page import render_phone_risk_page

        render_phone_risk_page(root, history)
