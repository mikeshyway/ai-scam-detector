"""Streamlit entry point for the capstone scam detection platform."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from app.ui_components import (
    APP_TITLE,
    inject_css,
    render_global_header,
    render_kpi_row,
    render_sidebar_brand,
    render_sidebar_navigation,
    render_sidebar_status,
)


def _email(root: Path, history: list[dict[str, object]]) -> None:
    from app.detection_center_page import render_detection_center_page

    render_detection_center_page(root, history)


def _report(root: Path, history: list[dict[str, object]]) -> None:
    from app.report_page import render_report_page

    render_report_page(root, history)


PAGES = {
    "Detection Center": _email,
    "AI Report Generator": _report,
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
        render_sidebar_navigation(
            active_page=st.session_state.active_page,
            on_select=_select_page,
        )
        with st.container(key="sidebar_status_dock"):
            render_sidebar_status(ROOT)

    selected_page = st.session_state.active_page
    render_global_header(ROOT, selected_page)
    render_kpi_row(st.session_state.history)
    PAGES[selected_page](ROOT, st.session_state.history)


if __name__ == "__main__":
    main()
