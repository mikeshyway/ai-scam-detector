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
    render_sidebar_status,
)


def _home(root: Path, history: list[dict[str, object]]) -> None:
    from app.home_page import render_home_page

    render_home_page(root, history)


def _dashboard(root: Path, history: list[dict[str, object]]) -> None:
    from app.dashboard_page import render_dashboard_page

    render_dashboard_page(root, history)


def _simulation(root: Path, history: list[dict[str, object]]) -> None:
    from app.simulation_lab_page import render_simulation_lab_page

    render_simulation_lab_page(root, history)


def _email(root: Path, history: list[dict[str, object]]) -> None:
    from app.detection_center_page import render_detection_center_page

    render_detection_center_page(root, history)


def _report(root: Path, history: list[dict[str, object]]) -> None:
    from app.report_page import render_report_page

    render_report_page(root, history)


def _explainability(root: Path, history: list[dict[str, object]]) -> None:
    from app.explainability_page import render_explainability_page

    render_explainability_page(root, history)


def _history(root: Path, history: list[dict[str, object]]) -> None:
    from app.history_tab import render_history_tab

    render_history_tab(root, history)


PAGES = {
    "Home": _home,
    "Dashboard": _dashboard,
    "Scam Simulation Lab": _simulation,
    "Detection Center": _email,
    "AI Report Generator": _report,
    "Transparency Hub": _explainability,
    "Session History": _history,
}


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"
    if st.session_state.active_page not in PAGES:
        st.session_state.active_page = "Home"


def _select_page(page_name: str) -> None:
    st.session_state.active_page = page_name


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    inject_css()

    with st.sidebar:
        st.title("AI Scam Defense Lab")
        st.caption("Explainable uploaded-evidence analysis")
        st.divider()
        for page_name in PAGES:
            is_active = st.session_state.active_page == page_name
            if st.button(
                page_name,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                _select_page(page_name)
                st.rerun()
        st.divider()
        render_sidebar_status(ROOT)
        st.divider()
        if st.button("Clear cached data/resources", use_container_width=True):
            clear_all_caches()
            st.rerun()
        st.caption("Use this after inserting official datasets or replacing trained model artifacts.")

    selected_page = st.session_state.active_page
    render_global_header(ROOT, selected_page)
    PAGES[selected_page](ROOT, st.session_state.history)


if __name__ == "__main__":
    main()
