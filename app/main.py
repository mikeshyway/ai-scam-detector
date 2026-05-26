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


def _quiz(root: Path, history: list[dict[str, object]]) -> None:
    from app.quiz_page import render_quiz_page

    render_quiz_page(root, history)


def _history(root: Path, history: list[dict[str, object]]) -> None:
    from app.history_tab import render_history_tab

    render_history_tab(root, history)


PAGES = {
    "🏠 Home": _home,
    "🎮 Scam Simulation Lab": _simulation,
    "📊 Dashboard": _dashboard,
    "🛡️ Detection Center": _email,
    "📄 AI Report Generator": _report,
    "🎓 Scenario-Based Quiz": _quiz,
    "🔎 Transparency Hub": _explainability,
    "🕘 Session History": _history,
}


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


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
        st.title("AI-based Scam System")
        st.caption("Student-focused fraud awareness prototype")
        st.divider()
        selected_page = st.radio("Pages", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        render_sidebar_status(ROOT)
        st.divider()
        if st.button("Clear cached data/resources", use_container_width=True):
            clear_all_caches()
            st.rerun()
        st.caption("Use this after inserting official datasets or replacing trained model artifacts.")

    render_global_header(ROOT, selected_page)
    PAGES[selected_page](ROOT, st.session_state.history)


if __name__ == "__main__":
    main()
