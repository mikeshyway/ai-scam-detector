"""Shared Streamlit UI helpers and cached app data."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st


APP_TITLE = "AI-based Spam and Caller Fraud Detection System"
APP_SUBTITLE = (
    "A capstone learning platform for uploaded scam evidence, explainable AI detection, "
    "and student decision practice."
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
        :root {
            --surface: rgba(15, 23, 42, 0.055);
            --surface-strong: rgba(15, 23, 42, 0.095);
            --line: rgba(148, 163, 184, 0.24);
            --muted: #64748b;
            --accent: #2563eb;
            --accent-2: #0f766e;
            --danger: #dc2626;
            --radius: 16px;
        }
        .block-container {
            padding-top: 5rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.08rem;
            line-height: 1.25;
            letter-spacing: 0;
            margin-bottom: 0.1rem;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }
        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            border-radius: 12px;
            min-height: 2.55rem;
            border: 1px solid transparent;
            box-shadow: none;
            font-weight: 650;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1d4ed8, #0f766e);
            color: white;
            border-color: rgba(255,255,255,0.14);
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: transparent;
            color: inherit;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: var(--surface);
            border-color: var(--line);
        }
        .app-shell {
            animation: fade-up 220ms ease-out;
        }
        .app-header {
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1.15rem;
            background:
              linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(15, 118, 110, 0.08)),
              var(--surface);
        }
        .app-header h1 {
            margin: 0.2rem 0 0.45rem 0;
            font-size: clamp(1.55rem, 3vw, 2.25rem);
            line-height: 1.12;
            letter-spacing: 0;
        }
        .app-header p {
            margin: 0;
            color: var(--muted);
            max-width: 880px;
            font-size: 0.98rem;
        }
        .eyebrow {
            color: var(--accent);
            font-weight: 800;
            font-size: 0.76rem;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.26rem 0.7rem;
            font-size: 0.82rem;
            background: rgba(255,255,255,0.04);
            color: inherit;
        }
        .demo-warning {
            border: 1px solid rgba(245, 158, 11, 0.32);
            border-left: 4px solid #f59e0b;
            padding: 0.8rem 0.95rem;
            border-radius: 12px;
            background: rgba(245, 158, 11, 0.10);
            margin-bottom: 1rem;
        }
        .feature-card, .scenario-panel {
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 1rem;
            background: var(--surface);
            box-shadow: 0 12px 34px rgba(2, 6, 23, 0.06);
        }
        .feature-card {
            min-height: 132px;
        }
        .feature-card h3, .scenario-panel h2 {
            margin-top: 0;
            letter-spacing: 0;
        }
        .feature-card p {
            color: var(--muted);
            margin-bottom: 0;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.55fr) minmax(250px, 0.45fr);
            gap: 1rem;
            align-items: center;
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.4rem;
            margin-bottom: 1rem;
            background:
              radial-gradient(circle at 80% 10%, rgba(37, 99, 235, 0.20), transparent 28%),
              radial-gradient(circle at 12% 20%, rgba(20, 184, 166, 0.14), transparent 25%),
              var(--surface);
        }
        .hero-grid h2 {
            font-size: clamp(1.65rem, 3.4vw, 2.75rem);
            line-height: 1.05;
            margin: 0 0 0.7rem 0;
            letter-spacing: 0;
        }
        .hero-grid p {
            color: var(--muted);
            margin: 0;
            max-width: 740px;
        }
        .cyber-avatar {
            position: relative;
            min-height: 190px;
            display: grid;
            place-items: center;
        }
        .avatar-orbit {
            position: absolute;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: 1px solid rgba(37, 99, 235, 0.28);
            animation: slow-spin 18s linear infinite;
        }
        .avatar-shield {
            width: 106px;
            height: 126px;
            border-radius: 30px 30px 42px 42px;
            display: grid;
            place-items: center;
            font-weight: 900;
            font-size: 1.85rem;
            color: white;
            background: linear-gradient(160deg, #1d4ed8, #0f766e);
            box-shadow: 0 18px 50px rgba(37, 99, 235, 0.28);
        }
        .avatar-caption {
            position: absolute;
            bottom: 0;
            color: var(--muted);
            font-size: 0.9rem;
        }
        .scenario-panel {
            margin: 0.75rem 0 1rem 0;
        }
        .scenario-panel pre {
            white-space: pre-wrap;
            font-family: inherit;
            margin-bottom: 0;
            line-height: 1.55;
        }
        .scenario-meta {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        mark {
            background: #fde68a;
            color: #2b2112;
            padding: 0.05rem 0.2rem;
            border-radius: 0.2rem;
        }
        @keyframes fade-up {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slow-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @media (max-width: 900px) {
            .block-container { padding-top: 4.5rem; }
            .hero-grid { grid-template-columns: 1fr; }
            .cyber-avatar { min-height: 150px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_header(root: Path, active_page: str) -> None:
    model_status = get_model_status(str(root))
    loaded_count = sum(model_status.values())
    dataset_mode = "Official data detected" if official_data_present(str(root)) else "Synthetic demo data active"
    st.markdown(
        f"""
        <div class="app-shell">
          <div class="app-header">
            <div class="eyebrow">Capstone educational prototype</div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
            <div class="status-row">
              <span class="status-chip">Page: {active_page}</span>
              <span class="status-chip">Dataset: {dataset_mode}</span>
              <span class="status-chip">Models loaded: {loaded_count}/{len(model_status)}</span>
            </div>
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
          The app is using synthetic examples because official datasets are not inserted yet.
          Remove this demo mode after training with official datasets.
          <br><code>{notice}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(root: Path) -> None:
    with st.expander("System status", expanded=False):
        st.caption("Model artifacts")
        for name, exists in get_model_status(str(root)).items():
            st.write(("[OK] " if exists else "[Missing] ") + name)

        st.caption("Dataset readiness")
        for name, exists in get_dataset_status(str(root)).items():
            st.write(("[Ready] " if exists else "[Demo] ") + name)


def clear_all_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
