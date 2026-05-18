"""Shared Streamlit UI helpers and cached app data."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st


APP_TITLE = "AI-based Spam and Caller Fraud Detection System"
APP_SUBTITLE = (
    "Educational platform for phishing emails, scam transcripts, synthetic speech, "
    "and student scam-awareness practice."
)


MODEL_ARTIFACTS = {
    "Email TF-IDF": "models/email_vectorizer.pkl",
    "Email Naive Bayes": "models/email_nb.pkl",
    "Email Decision Tree": "models/email_dt.pkl",
    "Transcript TF-IDF": "models/transcript_vectorizer.pkl",
    "Transcript Naive Bayes": "models/transcript_nb.pkl",
    "Audio SVM": "models/audio_svm.pkl",
}


@st.cache_data(show_spinner=False)
def get_demo_data() -> dict[str, object]:
    from src.demo_data import build_demo_bundle

    return build_demo_bundle()


@st.cache_data(show_spinner=False)
def get_model_status(root: str) -> dict[str, bool]:
    base = Path(root)
    return {name: (base / relative).exists() for name, relative in MODEL_ARTIFACTS.items()}


@st.cache_data(show_spinner=False)
def get_dataset_status(root: str) -> dict[str, bool]:
    base = Path(root)
    checks = {
        "SpamAssassin spam": base / "data/raw/spamassassin/spam",
        "SpamAssassin ham": base / "data/raw/spamassassin/ham",
        "Transcript CSV": base / "data/raw/transcripts/scam_nonscam_calls.csv",
        "YouTube demo CSV": base / "data/raw/transcripts/youtube_scam_transcripts.csv",
        "ASVspoof labels": base / "data/raw/asvspoof_subset/labels.csv",
    }
    status: dict[str, bool] = {}
    for name, path in checks.items():
        if path.is_dir():
            status[name] = any(item.is_file() and item.name != ".gitkeep" for item in path.iterdir())
        else:
            status[name] = path.exists()
    return status


def official_data_present(root: str) -> bool:
    return any(get_dataset_status(root).values())


def source_code_url(root: str) -> str:
    return os.environ.get("PROJECT_REPO_URL", "").strip() or f"file:///{Path(root).as_posix()}"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(148, 163, 184, 0.2); }
        .app-header {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            background:
              linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(20, 184, 166, 0.08)),
              rgba(15, 23, 42, 0.05);
        }
        .app-header h1 { margin: 0.15rem 0 0.35rem 0; font-size: 2rem; line-height: 1.15; }
        .app-header p { margin: 0; color: #94a3b8; max-width: 980px; }
        .eyebrow { color: #38bdf8; font-weight: 700; font-size: 0.78rem; letter-spacing: 0; }
        .status-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.85rem; }
        .status-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 999px;
            padding: 0.24rem 0.62rem;
            font-size: 0.82rem;
            background: rgba(15, 23, 42, 0.08);
        }
        .demo-warning {
            border-left: 4px solid #f59e0b;
            padding: 0.7rem 0.9rem;
            border-radius: 8px;
            background: rgba(245, 158, 11, 0.12);
            margin-bottom: 1rem;
        }
        .feature-card {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 10px;
            padding: 1rem;
            min-height: 138px;
            background: rgba(15, 23, 42, 0.04);
        }
        .feature-card h3 { margin-top: 0; font-size: 1.05rem; }
        mark {
            background: #ffe08a;
            color: #2b2112;
            padding: 0.05rem 0.2rem;
            border-radius: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_header(root: Path, active_page: str) -> None:
    model_status = get_model_status(str(root))
    loaded_count = sum(model_status.values())
    dataset_mode = "Official data detected" if official_data_present(str(root)) else "Synthetic demo data active"
    source = source_code_url(str(root))
    st.markdown(
        f"""
        <div class="app-header">
          <div class="eyebrow">Capstone educational prototype</div>
          <h1>{APP_TITLE}</h1>
          <p>{APP_SUBTITLE}</p>
          <div class="status-row">
            <span class="status-chip">📍 Page: {active_page}</span>
            <span class="status-chip">🧪 {dataset_mode}</span>
            <span class="status-chip">🧠 Models loaded: {loaded_count}/{len(model_status)}</span>
            <span class="status-chip">🔗 Source: {source}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_notice(root: Path) -> None:
    if official_data_present(str(root)):
        return
    notice = get_demo_data()["notice"]
    st.markdown(
        f"""
        <div class="demo-warning">
          <strong>Temporary demo dataset active.</strong>
          The app is using self-forged synthetic examples because official datasets are not inserted yet.
          Remove this demo mode after training with official datasets.
          <br><code>{notice}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(root: Path) -> None:
    st.caption("Model artifacts")
    for name, exists in get_model_status(str(root)).items():
        st.write(("✅ " if exists else "⚠️ ") + name)

    st.divider()
    st.caption("Dataset readiness")
    for name, exists in get_dataset_status(str(root)).items():
        st.write(("✅ " if exists else "🧪 ") + name)


def clear_all_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
