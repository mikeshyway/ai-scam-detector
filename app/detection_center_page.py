"""Unified Detection Center page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui_components import render_detection_center_header, render_detection_tab_selector


def render_detection_center_page(root: Path, history: list[dict[str, object]]) -> None:
    if "active_detection_tab" not in st.session_state:
        st.session_state.active_detection_tab = "email"
    if st.session_state.active_detection_tab not in {"email", "transcript", "phone"}:
        st.session_state.active_detection_tab = "email"

    render_detection_center_header()

    active_tab = render_detection_tab_selector(st.session_state.active_detection_tab)

    if active_tab == "email":
        from app.email_tab import render_email_tab

        render_email_tab(root, history)
    elif active_tab == "transcript":
        from app.transcript_tab import render_transcript_tab

        render_transcript_tab(root, history)
    elif active_tab == "phone":
        from app.phone_tab import render_phone_risk_page

        render_phone_risk_page(root, history)
