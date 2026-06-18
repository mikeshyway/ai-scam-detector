"""Streamlit entry point for the capstone scam detection platform."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui_components import (
    APP_TITLE,
    clear_all_caches,
    inject_css,
    render_global_header,
    render_kpi_row,
    render_sidebar_brand,
    render_sidebar_status,
)


def _simulation(root: Path, history: list[dict[str, object]]) -> None:
    from app.simulation_lab_page import render_simulation_lab_page

    render_simulation_lab_page(root, history)


def _email(root: Path, history: list[dict[str, object]]) -> None:
    from app.detection_center_page import render_detection_center_page

    render_detection_center_page(root, history)


def _live_audio(root: Path, history: list[dict[str, object]]) -> None:
    from app.live_audio_page import render_live_audio_page

    render_live_audio_page(root, history)


def _report(root: Path, history: list[dict[str, object]]) -> None:
    from app.report_page import render_report_page

    render_report_page(root, history)


PAGES = {
    "Scam Simulation Lab": _simulation,
    "Live Audio Detection": _live_audio,
    "Detection Center": _email,
    "AI Report Generator": _report,
}

PAGE_ICONS = {
    "Scam Simulation Lab": "SIM",
    "Live Audio Detection": "LIVE",
    "Detection Center": "DET",
    "AI Report Generator": "REP",
}


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_page" not in st.session_state:
        st.session_state.active_page = next(iter(PAGES))
    if st.session_state.active_page not in PAGES:
        st.session_state.active_page = next(iter(PAGES))


def _select_page(page_name: str) -> None:
    st.session_state.active_page = page_name


def main() -> None:
    st.set_page_config(
        page_title="AI-FDS - Fraud Detection",
        page_icon=":shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    inject_css()

    with st.sidebar:
        render_sidebar_brand()
        for page_name in PAGES:
            is_active = st.session_state.active_page == page_name
            if st.button(
                f"{PAGE_ICONS.get(page_name, '')} {page_name}",
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                _select_page(page_name)
                st.rerun()
        st.divider()
        render_sidebar_status(ROOT)
        if st.button("Clear cached data/resources", use_container_width=True):
            clear_all_caches()
            st.rerun()
        st.caption("Refresh cached resources after replacing datasets or trained artifacts.")

    selected_page = st.session_state.active_page
    render_global_header(ROOT, selected_page)
    render_kpi_row(st.session_state.history)
    PAGES[selected_page](ROOT, st.session_state.history)


if __name__ == "__main__":
    main()
