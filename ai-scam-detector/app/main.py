"""Streamlit entry point for the AI Scam Detector dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio_tab import render_audio_tab
from app.email_tab import render_email_tab
from app.history_tab import render_history_tab
from app.transcript_tab import render_transcript_tab


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def _model_status() -> dict[str, bool]:
    expected = {
        "Email TF-IDF": ROOT / "models" / "email_vectorizer.pkl",
        "Email Naive Bayes": ROOT / "models" / "email_nb.pkl",
        "Email Decision Tree": ROOT / "models" / "email_dt.pkl",
        "Transcript TF-IDF": ROOT / "models" / "transcript_vectorizer.pkl",
        "Transcript Naive Bayes": ROOT / "models" / "transcript_nb.pkl",
        "Audio SVM": ROOT / "models" / "audio_svm.pkl",
    }
    return {name: path.exists() for name, path in expected.items()}


def main() -> None:
    st.set_page_config(
        page_title="AI Scam Detector",
        page_icon=":shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        mark {
            background: #ffe08a;
            color: #2b2112;
            padding: 0.05rem 0.2rem;
            border-radius: 0.2rem;
        }
        .small-muted { color: #666; font-size: 0.88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("AI Scam Detector")
        st.caption("Educational multi-modal scam awareness demo")
        st.divider()
        st.subheader("Model artifacts")
        for name, exists in _model_status().items():
            st.write(("[OK] " if exists else "[MISSING] ") + name)
        st.divider()
        st.caption(
            "Text tabs use an educational rule demo if trained models are missing. "
            "Audio prediction needs the trained SVM artifact."
        )

    email_tab, transcript_tab, audio_tab, history_tab = st.tabs(
        ["Email", "Transcript", "Audio", "Session history"]
    )

    with email_tab:
        render_email_tab(ROOT, st.session_state.history)
    with transcript_tab:
        render_transcript_tab(ROOT, st.session_state.history)
    with audio_tab:
        render_audio_tab(ROOT, st.session_state.history)
    with history_tab:
        render_history_tab(st.session_state.history)


if __name__ == "__main__":
    main()
