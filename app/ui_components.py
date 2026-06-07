"""Shared GuardAI-style Streamlit UI helpers and cached app data."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Literal

import streamlit as st


APP_TITLE = "AI-based Spam and Caller Fraud Detection System"
APP_SUBTITLE = (
    "Uploaded-evidence scam detection with explainable AI feedback for students."
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
            status[name] = any(
                item.is_file() and item.name != ".gitkeep" for item in path.iterdir()
            )
        else:
            status[name] = path.exists()
    return status


def official_data_present(root: str) -> bool:
    return any(get_dataset_status(root).values())


def source_code_url(root: str) -> str:
    return os.environ.get("PROJECT_REPO_URL", "").strip() or f"file:///{Path(root).as_posix()}"


def inject_css() -> None:
    """Apply the shared GuardAI visual system to native Streamlit elements."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

        :root {
            --guard-bg: #07111f;
            --guard-sidebar: #0b1628;
            --guard-panel: #101e33;
            --guard-panel-2: #152942;
            --guard-line: rgba(148, 163, 184, 0.17);
            --guard-line-strong: rgba(148, 163, 184, 0.32);
            --guard-text: #edf2f7;
            --guard-muted: #94a3b8;
            --guard-faint: #64748b;
            --guard-blue: #3d8ee8;
            --guard-blue-dark: #2d7dd2;
            --guard-teal: #00c9a7;
            --guard-amber: #f0a500;
            --guard-red: #ef5b5b;
            --guard-green: #22c55e;
            --guard-radius: 8px;
        }

        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: "DM Sans", sans-serif;
        }

        .stApp {
            background:
                linear-gradient(rgba(61, 142, 232, 0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(61, 142, 232, 0.025) 1px, transparent 1px),
                var(--guard-bg);
            background-size: 32px 32px;
            color: var(--guard-text);
        }

        [data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.9);
            border-bottom: 1px solid var(--guard-line);
            backdrop-filter: blur(12px);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 4.75rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
            letter-spacing: 0;
        }

        h1, h2, h3, h4 {
            color: var(--guard-text);
        }

        p, [data-testid="stCaptionContainer"] {
            color: var(--guard-muted);
        }

        [data-testid="stSidebar"] {
            background: var(--guard-sidebar);
            border-right: 1px solid var(--guard-line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        .sidebar-brand {
            padding: 0.15rem 0.15rem 1.05rem;
            border-bottom: 1px solid var(--guard-line);
            margin-bottom: 0.8rem;
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }

        .brand-mark {
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            border: 1px solid rgba(61, 142, 232, 0.55);
            border-radius: 8px;
            color: #ffffff;
            background: #17365d;
            font-family: "Space Mono", monospace;
            font-size: 0.7rem;
            font-weight: 700;
        }

        .brand-name {
            color: var(--guard-text);
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.15;
        }

        .brand-caption {
            margin-top: 0.5rem;
            color: var(--guard-faint);
            font-size: 0.75rem;
            line-height: 1.45;
        }

        [data-testid="stSidebar"] .stButton > button {
            min-height: 2.55rem;
            justify-content: flex-start;
            padding: 0.5rem 0.75rem;
            border: 1px solid transparent;
            border-radius: 7px;
            box-shadow: none;
            font-weight: 600;
            transition: background 140ms ease, border-color 140ms ease;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            color: #ffffff;
            background: #17365d;
            border-color: rgba(61, 142, 232, 0.5);
            box-shadow: inset 3px 0 0 var(--guard-teal);
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            color: var(--guard-muted);
            background: transparent;
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            color: var(--guard-text);
            background: rgba(255, 255, 255, 0.045);
            border-color: var(--guard-line);
        }

        .app-header {
            position: relative;
            overflow: hidden;
            padding: 1.35rem 1.45rem 1.25rem;
            margin-bottom: 1.35rem;
            border: 1px solid var(--guard-line);
            border-radius: var(--guard-radius);
            background: var(--guard-panel);
            box-shadow: inset 4px 0 0 var(--guard-blue);
        }

        .app-header::after {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 34%;
            height: 100%;
            background: repeating-linear-gradient(
                135deg,
                transparent 0,
                transparent 12px,
                rgba(61, 142, 232, 0.045) 12px,
                rgba(61, 142, 232, 0.045) 13px
            );
            pointer-events: none;
        }

        .app-header h1 {
            position: relative;
            z-index: 1;
            max-width: 860px;
            margin: 0.25rem 0 0.45rem;
            font-size: clamp(1.65rem, 3vw, 2.35rem);
            line-height: 1.12;
        }

        .app-header p {
            position: relative;
            z-index: 1;
            max-width: 800px;
            margin: 0;
            font-size: 0.94rem;
        }

        .eyebrow, .section-kicker {
            color: var(--guard-teal);
            font-family: "Space Mono", monospace;
            font-size: 0.69rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .status-row {
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.6rem;
            border: 1px solid var(--guard-line);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.035);
            color: var(--guard-muted);
            font-family: "Space Mono", monospace;
            font-size: 0.67rem;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--guard-teal);
            box-shadow: 0 0 8px rgba(0, 201, 167, 0.7);
        }

        .section-header {
            margin: 1.35rem 0 0.8rem;
        }

        .section-header h2 {
            margin: 0.25rem 0 0.2rem;
            font-size: 1.35rem;
        }

        .section-header p {
            max-width: 790px;
            margin: 0;
            font-size: 0.9rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(220px, 0.55fr);
            gap: 1rem;
            min-height: 310px;
            align-items: stretch;
            margin-bottom: 1.35rem;
        }

        .hero-copy, .hero-console {
            border: 1px solid var(--guard-line);
            border-radius: var(--guard-radius);
            background: var(--guard-panel);
        }

        .hero-copy {
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 2rem;
            box-shadow: inset 4px 0 0 var(--guard-teal);
        }

        .hero-copy h2 {
            max-width: 760px;
            margin: 0.5rem 0 0.9rem;
            font-size: clamp(1.8rem, 4vw, 3.15rem);
            line-height: 1.04;
        }

        .hero-copy p {
            max-width: 680px;
            margin: 0;
            line-height: 1.7;
        }

        .hero-console {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 1.25rem;
            background: #0d1a2c;
        }

        .console-label {
            color: var(--guard-faint);
            font-family: "Space Mono", monospace;
            font-size: 0.68rem;
            text-transform: uppercase;
        }

        .radar {
            position: relative;
            width: 150px;
            height: 150px;
            margin: 0.7rem auto;
            border: 1px solid rgba(61, 142, 232, 0.4);
            border-radius: 50%;
            background:
                linear-gradient(90deg, transparent 49.5%, rgba(61,142,232,.22) 50%, transparent 50.5%),
                linear-gradient(transparent 49.5%, rgba(61,142,232,.22) 50%, transparent 50.5%),
                radial-gradient(circle, rgba(0,201,167,.16) 0 3%, transparent 4% 36%, rgba(61,142,232,.13) 37% 38%, transparent 39% 67%, rgba(61,142,232,.13) 68% 69%, transparent 70%);
        }

        .radar::after {
            content: "";
            position: absolute;
            inset: 10px;
            border-radius: 50%;
            background: conic-gradient(from 0deg, rgba(0,201,167,.45), transparent 24%);
            animation: radar-sweep 4s linear infinite;
        }

        .console-status {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-top: 0.75rem;
            border-top: 1px solid var(--guard-line);
            color: var(--guard-muted);
            font-family: "Space Mono", monospace;
            font-size: 0.68rem;
        }

        .feature-card, .scenario-panel, [data-testid="stMetric"], [data-testid="stDataFrame"],
        [data-testid="stFileUploaderDropzone"], [data-testid="stExpander"] {
            border: 1px solid var(--guard-line);
            border-radius: var(--guard-radius);
            background: var(--guard-panel);
        }

        .feature-card {
            min-height: 145px;
            padding: 1rem;
            transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
        }

        .feature-card:hover {
            transform: translateY(-2px);
            border-color: var(--guard-line-strong);
            background: var(--guard-panel-2);
        }

        .feature-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            margin-bottom: 0.85rem;
            border: 1px solid rgba(61, 142, 232, 0.45);
            border-radius: 6px;
            color: var(--guard-blue);
            background: rgba(61, 142, 232, 0.08);
            font-family: "Space Mono", monospace;
            font-size: 0.68rem;
            font-weight: 700;
        }

        .feature-card h3 {
            margin: 0 0 0.45rem;
            font-size: 0.98rem;
        }

        .feature-card p {
            margin: 0;
            font-size: 0.85rem;
            line-height: 1.55;
        }

        .scenario-panel {
            padding: 1.1rem;
            margin: 0.75rem 0 1rem;
        }

        .scenario-panel pre {
            white-space: pre-wrap;
            font-family: "DM Sans", sans-serif;
            line-height: 1.6;
        }

        .scenario-meta {
            color: var(--guard-blue);
            font-family: "Space Mono", monospace;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(var(--metric-count), minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.75rem 0 1.25rem;
        }

        .metric-tile {
            min-height: 100px;
            padding: 0.9rem 1rem;
            border: 1px solid var(--guard-line);
            border-radius: var(--guard-radius);
            background: var(--guard-panel);
        }

        .metric-value {
            color: var(--guard-text);
            font-family: "Space Mono", monospace;
            font-size: 1.45rem;
            font-weight: 700;
        }

        .metric-label {
            margin-top: 0.3rem;
            color: var(--guard-muted);
            font-size: 0.78rem;
        }

        .info-banner {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            padding: 0.85rem 1rem;
            margin: 0.75rem 0 1rem;
            border: 1px solid rgba(61, 142, 232, 0.3);
            border-left: 3px solid var(--guard-blue);
            border-radius: var(--guard-radius);
            background: rgba(61, 142, 232, 0.08);
            color: var(--guard-muted);
            font-size: 0.87rem;
            line-height: 1.5;
        }

        .info-banner.warning {
            border-color: rgba(240, 165, 0, 0.3);
            border-left-color: var(--guard-amber);
            background: rgba(240, 165, 0, 0.08);
        }

        .info-banner.danger {
            border-color: rgba(239, 91, 91, 0.3);
            border-left-color: var(--guard-red);
            background: rgba(239, 91, 91, 0.08);
        }

        .info-banner.success {
            border-color: rgba(34, 197, 94, 0.3);
            border-left-color: var(--guard-green);
            background: rgba(34, 197, 94, 0.08);
        }

        .banner-code {
            min-width: 31px;
            height: 24px;
            display: inline-grid;
            place-items: center;
            border: 1px solid currentColor;
            border-radius: 5px;
            font-family: "Space Mono", monospace;
            font-size: 0.62rem;
            font-weight: 700;
        }

        .demo-warning {
            padding: 0.8rem 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(240, 165, 0, 0.28);
            border-left: 3px solid var(--guard-amber);
            border-radius: var(--guard-radius);
            background: rgba(240, 165, 0, 0.075);
            color: var(--guard-muted);
            font-size: 0.83rem;
        }

        [data-testid="stMetric"] {
            padding: 0.85rem 0.95rem;
        }

        [data-testid="stMetricValue"] {
            font-family: "Space Mono", monospace;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            padding: 0.25rem;
            border: 1px solid var(--guard-line);
            border-radius: var(--guard-radius);
            background: var(--guard-sidebar);
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.55rem;
            padding: 0 0.8rem;
            border-radius: 6px;
            color: var(--guard-muted);
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: #17365d !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        .stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a {
            min-height: 2.55rem;
            border: 1px solid var(--guard-line-strong);
            border-radius: 7px;
            box-shadow: none;
            font-weight: 650;
        }

        .stButton > button[kind="primary"] {
            border-color: var(--guard-blue-dark);
            color: #ffffff;
            background: var(--guard-blue-dark);
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--guard-blue);
            color: #ffffff;
        }

        .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
            border-color: var(--guard-line-strong) !important;
            border-radius: 7px !important;
            background: #0c192b !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            padding: 1.1rem;
            background: #0c192b;
        }

        [data-testid="stAlert"] {
            border-radius: var(--guard-radius);
            border-width: 1px;
        }

        [data-testid="stDataFrame"] {
            overflow: hidden;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
        }

        hr {
            border-color: var(--guard-line) !important;
        }

        code {
            color: #8bd8cb;
            background: rgba(0, 201, 167, 0.08);
        }

        mark {
            padding: 0.08rem 0.22rem;
            border-radius: 3px;
            color: #fff4d6;
            background: rgba(240, 165, 0, 0.42);
        }

        @keyframes radar-sweep {
            to { transform: rotate(360deg); }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 4.25rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .hero-console {
                min-height: 230px;
            }
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 560px) {
            .app-header {
                padding: 1rem;
            }
            .hero-copy {
                padding: 1.25rem;
            }
            .metric-strip {
                grid-template-columns: 1fr;
            }
            .status-chip {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="brand-lockup">
            <div class="brand-mark">AI</div>
            <div class="brand-name">GuardAI Learning Lab</div>
          </div>
          <div class="brand-caption">
            Explainable scam detection and defensive decision practice.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_header(root: Path, active_page: str) -> None:
    model_status = get_model_status(str(root))
    loaded_count = sum(model_status.values())
    dataset_mode = "Official dataset" if official_data_present(str(root)) else "Synthetic demo"
    safe_page = html.escape(active_page)
    st.markdown(
        f"""
        <div class="app-header">
          <div class="eyebrow">GuardAI / Capstone educational prototype</div>
          <h1>{APP_TITLE}</h1>
          <p>{APP_SUBTITLE}</p>
          <div class="status-row">
            <span class="status-chip"><span class="status-dot"></span>{safe_page}</span>
            <span class="status-chip">DATA / {dataset_mode}</span>
            <span class="status-chip">MODELS / {loaded_count} of {len(model_status)} ready</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = (
        f'<div class="section-kicker">{html.escape(eyebrow)}</div>' if eyebrow else ""
    )
    subtitle_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
          {eyebrow_html}
          <h2>{html.escape(title)}</h2>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(title: str, body: str, index: str = "01") -> None:
    st.markdown(
        f"""
        <div class="feature-card">
          <div class="feature-index">{html.escape(index)}</div>
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list[dict[str, object]]) -> None:
    tiles = []
    for metric in metrics:
        color = html.escape(str(metric.get("color", "var(--guard-text)")))
        tiles.append(
            f"""
            <div class="metric-tile">
              <div class="metric-value" style="color:{color}">
                {html.escape(str(metric.get("value", "-")))}
              </div>
              <div class="metric-label">{html.escape(str(metric.get("label", "")))}</div>
            </div>
            """
        )
    st.markdown(
        f'<div class="metric-strip" style="--metric-count:{max(1, len(metrics))}">'
        + "".join(tiles)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_info_banner(
    body: str,
    kind: Literal["info", "warning", "danger", "success"] = "info",
    code: str = "INFO",
) -> None:
    st.markdown(
        f"""
        <div class="info-banner {kind}">
          <span class="banner-code">{html.escape(code)}</span>
          <div>{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_notice(root: Path) -> None:
    if official_data_present(str(root)):
        return
    st.markdown(
        """
        <div class="demo-warning">
          <strong>Temporary demonstration data is active.</strong>
          Replace synthetic examples with the official datasets before reporting final model results.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(root: Path) -> None:
    with st.expander("System readiness", expanded=False):
        st.caption("Model artifacts")
        for name, exists in get_model_status(str(root)).items():
            st.write(("[READY] " if exists else "[MISSING] ") + name)

        st.caption("Dataset sources")
        for name, exists in get_dataset_status(str(root)).items():
            st.write(("[READY] " if exists else "[DEMO] ") + name)


def clear_all_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
