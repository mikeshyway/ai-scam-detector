"""Shared AI-FDS Streamlit UI helpers and cached app data."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Literal

import streamlit as st
import streamlit.components.v1 as components


APP_TITLE = "AI-based Spam and Caller Fraud Detection System"
APP_SUBTITLE = "AI-FDS · Explainable fraud detection & student decision practice"

MODEL_ARTIFACTS = {
    "Email TF-IDF": "models/email_vectorizer.pkl",
    "Email Naive Bayes": "models/email_nb.pkl",
    "Email Decision Tree": "models/email_dt.pkl",
    "Transcript TF-IDF": "models/transcript_vectorizer.pkl",
    "Transcript Naive Bayes": "models/transcript_nb.pkl",
    "Audio SVM": "models/audio_svm.pkl",
}

CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#94A3B8"},
    "xaxis": {
        "gridcolor": "rgba(148,163,184,0.08)",
        "linecolor": "rgba(148,163,184,0.15)",
        "zerolinecolor": "rgba(148,163,184,0.10)",
    },
    "yaxis": {
        "gridcolor": "rgba(148,163,184,0.08)",
        "linecolor": "rgba(148,163,184,0.15)",
        "zerolinecolor": "rgba(148,163,184,0.10)",
    },
    "colorway": ["#6366F1", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"],
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
    """Apply the shared AI-FDS visual system to native Streamlit elements."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-base: #0D1117;
            --bg-surface: #161B27;
            --bg-raised: #1E2638;
            --bg-inset: #0A0E18;
            --grad-primary: linear-gradient(135deg, #6366F1 0%, #8B5CF6 40%, #06B6D4 100%);
            --grad-danger: linear-gradient(135deg, #EF4444 0%, #F97316 100%);
            --grad-success: linear-gradient(135deg, #10B981 0%, #06B6D4 100%);
            --grad-warning: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
            --accent-violet: #6366F1;
            --accent-purple: #8B5CF6;
            --accent-cyan: #06B6D4;
            --accent-emerald: #10B981;
            --accent-amber: #F59E0B;
            --accent-red: #EF4444;
            --text-primary: #F0F6FF;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --border-subtle: rgba(148, 163, 184, 0.10);
            --border-medium: rgba(148, 163, 184, 0.20);
            --border-accent: rgba(99, 102, 241, 0.40);
            --glow-violet: 0 0 20px rgba(99, 102, 241, 0.25);
            --glow-cyan: 0 0 20px rgba(6, 182, 212, 0.20);
            --glow-red: 0 0 20px rgba(239, 68, 68, 0.25);
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
            --radius-xl: 28px;
            --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
            --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
        }

        body, .stApp, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif;
            background: var(--bg-base);
            color: var(--text-primary);
        }

        code, pre, .mono, [data-testid="stCode"] {
            font-family: 'JetBrains Mono', ui-monospace, monospace;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(99,102,241,0.14), transparent 28rem),
                radial-gradient(circle at 94% 4%, rgba(6,182,212,0.10), transparent 24rem),
                linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px),
                var(--bg-base);
            background-size: auto, auto, 34px 34px, 34px 34px, auto;
        }

        [data-testid="stHeader"] {
            background: rgba(13,17,23,0.82);
            border-bottom: 1px solid var(--border-subtle);
            backdrop-filter: blur(14px);
        }

        .block-container {
            max-width: 1260px;
            padding: 4.9rem 1.5rem 4rem;
        }

        h1 {
            font-size: clamp(1.8rem, 3.5vw, 2.6rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--text-primary);
        }

        h2 {
            font-size: clamp(1.3rem, 2.5vw, 1.75rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }

        h3 {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        p, label, [data-testid="stCaptionContainer"] {
            color: var(--text-secondary);
        }

        [data-testid="stSidebar"] {
            background: var(--bg-surface) !important;
            border-right: 1px solid var(--border-subtle) !important;
            min-width: 72px !important;
            transition: min-width 280ms var(--ease-out), max-width 280ms var(--ease-out) !important;
        }

        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 240px !important;
            max-width: 240px !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.8rem;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 1rem 0.85rem 0.75rem;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 0.5rem;
        }

        .brand-mark {
            width: 38px;
            height: 38px;
            flex-shrink: 0;
            background: var(--grad-primary);
            clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 0.85rem;
            color: #fff;
            box-shadow: var(--glow-violet);
        }

        .brand-name {
            font-weight: 800;
            font-size: 0.92rem;
            line-height: 1.1;
            color: var(--text-primary);
        }

        .brand-caption {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 1px;
        }

        [data-testid="stSidebar"] .stButton > button {
            display: flex;
            align-items: center;
            gap: 10px;
            min-height: 2.55rem;
            padding: 0.6rem 0.85rem;
            border-radius: var(--radius-md);
            margin: 2px 0;
            cursor: pointer;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-secondary);
            border: 1px solid transparent;
            background: transparent;
            box-shadow: none;
            transition: all 180ms var(--ease-out);
            justify-content: flex-start;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--bg-raised);
            color: var(--text-primary);
            border-color: var(--border-subtle);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(6,182,212,0.10) 100%);
            color: #A5B4FC;
            border-color: var(--border-accent);
            box-shadow: var(--glow-violet);
        }

        .system-status {
            margin: 1rem 8px 0;
            padding: 0.75rem;
            background: var(--bg-inset);
            border-radius: var(--radius-md);
            border: 1px solid var(--border-subtle);
        }

        .system-status-title {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .system-status-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            font-size: 0.78rem;
            padding: 3px 0;
            color: var(--text-secondary);
        }

        .system-status-state.ready { color: var(--accent-emerald); }
        .system-status-state.missing { color: var(--accent-red); }
        .system-status-state.demo { color: var(--accent-amber); }

        .app-header {
            background:
                radial-gradient(circle at 15% 50%, rgba(99,102,241,0.18) 0%, transparent 45%),
                radial-gradient(circle at 85% 20%, rgba(6,182,212,0.14) 0%, transparent 40%),
                radial-gradient(circle at 50% 100%, rgba(139,92,246,0.10) 0%, transparent 50%),
                var(--bg-surface);
            border: 1px solid var(--border-accent);
            border-radius: var(--radius-xl);
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: var(--glow-violet), 0 4px 24px rgba(0,0,0,0.25);
        }

        .app-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--grad-primary);
            opacity: 0.8;
        }

        .app-header .eyebrow,
        .section-header .eyebrow {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            background: var(--grad-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .app-header .eyebrow {
            margin-bottom: 0.5rem;
        }

        .app-header h1 {
            margin: 0 0 0.55rem;
            max-width: 900px;
            position: relative;
            z-index: 1;
        }

        .app-header p {
            margin: 0;
            max-width: 780px;
            color: var(--text-secondary);
            position: relative;
            z-index: 1;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
            position: relative;
            z-index: 1;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.79rem;
            font-weight: 600;
            border: 1px solid;
            backdrop-filter: blur(4px);
        }

        .status-chip .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            animation: pulse-dot 2s infinite;
        }

        .status-chip.page {
            background: rgba(99,102,241,0.15);
            border-color: rgba(99,102,241,0.40);
            color: #A5B4FC;
        }

        .status-chip.data-demo {
            background: rgba(245,158,11,0.12);
            border-color: rgba(245,158,11,0.34);
            color: #FCD34D;
        }

        .status-chip.data-official,
        .status-chip.models-ready {
            background: rgba(16,185,129,0.12);
            border-color: rgba(16,185,129,0.32);
            color: #6EE7B7;
        }

        .status-chip.models-missing {
            background: rgba(239,68,68,0.12);
            border-color: rgba(239,68,68,0.32);
            color: #FCA5A5;
        }

        .status-chip.models-partial {
            background: rgba(245,158,11,0.12);
            border-color: rgba(245,158,11,0.32);
            color: #FCD34D;
        }

        .status-chip.page .dot { background: var(--accent-violet); }
        .status-chip.data-demo .dot,
        .status-chip.models-partial .dot { background: var(--accent-amber); }
        .status-chip.data-official .dot,
        .status-chip.models-ready .dot { background: var(--accent-emerald); }
        .status-chip.models-missing .dot { background: var(--accent-red); }

        .section-header {
            margin: 1.75rem 0 1rem;
        }

        .section-header .eyebrow {
            margin-bottom: 0.3rem;
        }

        .section-header h2 {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            margin: 0 0 0.35rem;
        }

        .section-header p {
            color: var(--text-secondary);
            font-size: 0.92rem;
            line-height: 1.55;
            margin: 0;
            max-width: 680px;
        }

        .banner, .demo-warning {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.85rem 1rem;
            border-radius: var(--radius-md);
            font-size: 0.88rem;
            line-height: 1.55;
            margin: 0.75rem 0;
            border: 1px solid;
            animation: slide-in 300ms var(--ease-out);
            color: var(--text-secondary);
        }

        .banner.info { background: rgba(99,102,241,0.08); border-color: rgba(99,102,241,0.30); }
        .banner.warning, .demo-warning {
            background: rgba(245,158,11,0.08);
            border-color: rgba(245,158,11,0.32);
            border-left: 3px solid var(--accent-amber);
        }
        .banner.danger { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.30); border-left: 3px solid var(--accent-red); }
        .banner.success { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.28); border-left: 3px solid var(--accent-emerald); }

        .banner-tag {
            flex-shrink: 0;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            padding: 0.18rem 0.55rem;
            border-radius: 4px;
            margin-top: 0.08rem;
        }

        .banner.info .banner-tag { background: rgba(99,102,241,0.20); color: #818CF8; }
        .banner.warning .banner-tag, .demo-warning .banner-tag { background: rgba(245,158,11,0.20); color: #FCD34D; }
        .banner.danger .banner-tag { background: rgba(239,68,68,0.18); color: #FCA5A5; }
        .banner.success .banner-tag { background: rgba(16,185,129,0.18); color: #6EE7B7; }

        .content-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.25rem 1.35rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
            transition: border-color 200ms, box-shadow 200ms;
            position: relative;
            overflow: hidden;
        }

        .content-card:hover {
            border-color: var(--border-medium);
            box-shadow: 0 4px 24px rgba(0,0,0,0.20);
        }

        .content-card.accent-violet::before,
        .content-card.accent-red::before,
        .content-card.accent-green::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
        }

        .content-card.accent-violet::before { background: var(--grad-primary); }
        .content-card.accent-red::before { background: var(--grad-danger); }
        .content-card.accent-green::before { background: var(--grad-success); }

        .feature-card, .scenario-panel {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1rem;
            min-height: 145px;
            transition: transform 180ms var(--ease-spring), border-color 180ms, box-shadow 180ms;
        }

        .feature-card:hover, .scenario-panel:hover {
            transform: translateY(-2px);
            border-color: var(--border-medium);
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }

        .feature-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            margin-bottom: 0.85rem;
            border-radius: 9px;
            background: rgba(99,102,241,0.16);
            color: #A5B4FC;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 700;
            border: 1px solid rgba(99,102,241,0.32);
        }

        .feature-card p {
            color: var(--text-secondary);
            font-size: 0.88rem;
            line-height: 1.55;
            margin-bottom: 0;
        }

        .scenario-panel pre {
            white-space: pre-wrap;
            font-family: 'Plus Jakarta Sans', sans-serif;
            line-height: 1.6;
            color: var(--text-secondary);
        }

        .scenario-meta {
            font-family: 'JetBrains Mono', monospace;
            color: #A5B4FC;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .kpi-strip, .metric-strip {
            display: grid;
            grid-template-columns: repeat(var(--metric-count), minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0 0 1.25rem;
        }

        .kpi-tile, .metric-tile {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1rem 1.1rem;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
            transition: transform 180ms var(--ease-spring), box-shadow 180ms;
        }

        .kpi-tile:hover, .metric-tile:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }

        .kpi-icon {
            font-size: 1.3rem;
            margin-bottom: 0.1rem;
        }

        .kpi-value, .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1;
            font-family: 'JetBrains Mono', monospace;
        }

        .kpi-label, .metric-label {
            font-size: 0.78rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .scroll-notify {
            display: flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(90deg, rgba(99,102,241,0.15) 0%, rgba(6,182,212,0.10) 100%);
            border: 1px solid rgba(99,102,241,0.35);
            border-radius: 999px;
            padding: 0.4rem 1rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: #A5B4FC;
            width: fit-content;
            margin: 0.5rem 0 1rem;
            animation: pulse-banner 2s ease-in-out infinite;
        }

        .scroll-notify-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-violet);
            animation: pulse-dot 1s infinite;
        }

        .result-card {
            border-radius: var(--radius-lg);
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1rem;
            border: 1px solid;
            animation: scale-in 220ms var(--ease-out);
        }

        .result-card.high {
            background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(249,115,22,0.05) 100%), var(--bg-surface);
            border-color: rgba(239,68,68,0.40);
            border-left: 3px solid var(--accent-red);
        }

        .result-card.medium {
            background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(234,179,8,0.04) 100%), var(--bg-surface);
            border-color: rgba(245,158,11,0.38);
            border-left: 3px solid var(--accent-amber);
        }

        .result-card.safe {
            background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(6,182,212,0.04) 100%), var(--bg-surface);
            border-color: rgba(16,185,129,0.32);
            border-left: 3px solid var(--accent-emerald);
        }

        .risk-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 1px solid;
            margin-bottom: 0.65rem;
        }

        .risk-badge.high { background: rgba(239,68,68,0.15); color: #FCA5A5; border-color: rgba(239,68,68,0.35); }
        .risk-badge.medium { background: rgba(245,158,11,0.15); color: #FCD34D; border-color: rgba(245,158,11,0.32); }
        .risk-badge.safe { background: rgba(16,185,129,0.12); color: #6EE7B7; border-color: rgba(16,185,129,0.28); }

        .result-card h3 {
            margin: 0 0 0.35rem;
        }

        .result-card p {
            margin: 0;
            color: var(--text-secondary);
            line-height: 1.55;
        }

        .chart-placeholder {
            background: var(--bg-inset);
            border: 1.5px dashed var(--border-medium);
            border-radius: var(--radius-lg);
            min-height: 280px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            color: var(--text-muted);
            font-size: 0.88rem;
        }

        .chart-spinner {
            width: 36px;
            height: 36px;
            border: 3px solid var(--border-subtle);
            border-top-color: var(--accent-violet);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        [data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-inset) !important;
            border: 1.5px dashed var(--border-medium) !important;
            border-radius: var(--radius-md) !important;
            padding: 1rem !important;
            transition: border-color 200ms, background 200ms !important;
        }

        [data-testid="stFileUploader"]:hover,
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-violet) !important;
            background: rgba(99,102,241,0.04) !important;
        }

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button {
            background: var(--grad-primary) !important;
            color: #fff !important;
            border: none !important;
            border-radius: var(--radius-sm) !important;
            font-weight: 700 !important;
            font-size: 0.85rem !important;
            padding: 0.45rem 1rem !important;
            box-shadow: var(--glow-violet) !important;
        }

        [data-testid="stFileUploaderFileName"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.82rem !important;
        }

        [data-testid="stSlider"] > div > div > div {
            background: var(--bg-inset) !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--grad-primary) !important;
            border: 2px solid #fff !important;
            box-shadow: var(--glow-violet) !important;
            width: 18px !important;
            height: 18px !important;
        }

        [data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
            background: var(--grad-primary) !important;
        }

        .app-shell {
            animation: fade-up 250ms var(--ease-out) both;
        }

        @keyframes fade-up {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slide-in {
            from { opacity: 0; transform: translateX(12px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes scale-in {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.35; }
        }

        @keyframes pulse-banner {
            0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.25); }
            50% { box-shadow: 0 0 0 6px rgba(99,102,241,0); }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background: var(--bg-inset) !important;
            border: 1px solid var(--border-medium) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.88rem !important;
            transition: border-color 180ms !important;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--accent-violet) !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
            outline: none !important;
        }

        [data-testid="stSelectbox"] > div > div,
        [data-baseweb="select"] > div {
            background: var(--bg-inset) !important;
            border: 1px solid var(--border-medium) !important;
            border-radius: var(--radius-sm) !important;
        }

        [data-testid="stSelectbox"] > div > div:focus-within {
            border-color: var(--accent-violet) !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
        }

        [data-testid="stButton"] > button[kind="primary"],
        .stButton > button[kind="primary"] {
            background: var(--grad-primary) !important;
            color: #fff !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            padding: 0.55rem 1.4rem !important;
            box-shadow: var(--glow-violet) !important;
            transition: transform 150ms var(--ease-spring), box-shadow 150ms !important;
        }

        [data-testid="stButton"] > button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(99,102,241,0.40) !important;
        }

        [data-testid="stButton"] > button[kind="secondary"],
        .stButton > button[kind="secondary"],
        .stDownloadButton > button {
            background: var(--bg-raised) !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border-medium) !important;
            border-radius: var(--radius-md) !important;
            font-weight: 600 !important;
            transition: background 150ms, border-color 150ms !important;
        }

        [data-testid="stButton"] > button[kind="secondary"]:hover,
        .stButton > button[kind="secondary"]:hover,
        .stDownloadButton > button:hover {
            background: rgba(99,102,241,0.10) !important;
            border-color: var(--border-accent) !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stExpander"] {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden !important;
        }

        [data-testid="stExpander"] summary {
            padding: 0.7rem 1rem !important;
            font-weight: 600 !important;
            font-size: 0.90rem !important;
        }

        [data-testid="stExpander"] summary:hover {
            background: var(--bg-raised) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            padding: 0.25rem;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            background: var(--bg-inset);
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.65rem;
            padding: 0 0.85rem;
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            color: #fff !important;
            background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(6,182,212,0.12)) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        [data-testid="stDivider"] hr,
        hr {
            border: none !important;
            height: 1px !important;
            background: var(--border-subtle) !important;
            margin: 1rem 0 !important;
        }

        [data-testid="stAlert"] {
            border-radius: var(--radius-md) !important;
            border: 1px solid !important;
        }

        [data-testid="stMetric"],
        [data-testid="stDataFrame"] {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }

        [data-testid="stMetric"] {
            padding: 0.85rem 1rem;
        }

        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
        }

        code {
            color: #A5B4FC;
            background: rgba(99,102,241,0.12);
            border-radius: 6px;
        }

        mark {
            padding: 0.08rem 0.22rem;
            border-radius: 4px;
            color: #FCD34D;
            background: rgba(245,158,11,0.18);
        }

        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-inset); }
        ::-webkit-scrollbar-thumb {
            background: var(--border-medium);
            border-radius: 999px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-violet);
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
            .kpi-strip, .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 560px) {
            .app-header {
                padding: 1.2rem;
                border-radius: var(--radius-lg);
            }
            .status-chip {
                width: 100%;
            }
            .kpi-strip, .metric-strip {
                grid-template-columns: 1fr;
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
          <div class="brand-mark">AI</div>
          <div>
            <div class="brand-name">AI-FDS</div>
            <div class="brand-caption">AI Fraud Detection System</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_header(root: Path, active_page: str) -> None:
    model_status = get_model_status(str(root))
    loaded_count = sum(model_status.values())
    total = len(model_status)
    data_ready = official_data_present(str(root))
    model_class = (
        "models-ready" if loaded_count == total else "models-missing" if loaded_count == 0 else "models-partial"
    )
    data_class = "data-official" if data_ready else "data-demo"
    data_label = "Official dataset" if data_ready else "Synthetic demo"
    safe_page = html.escape(active_page)
    st.markdown(
        f"""
        <div class="app-header app-shell">
          <div class="eyebrow">AI-FDS / Capstone educational prototype</div>
          <h1>{APP_TITLE}</h1>
          <p>{APP_SUBTITLE}</p>
          <div class="status-row">
            <span class="status-chip page"><span class="dot"></span>{safe_page}</span>
            <span class="status-chip {data_class}"><span class="dot"></span>Data / {data_label}</span>
            <span class="status-chip {model_class}"><span class="dot"></span>Models / {loaded_count} of {total} ready</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(history: list[dict[str, object]]) -> None:
    confidence_values = [
        float(item.get("confidence", 0))
        for item in history
        if str(item.get("confidence", "")).replace(".", "", 1).isdigit()
    ]
    risky_terms = ("suspicious", "high risk", "ai-generated", "chunk analysis")
    threats = sum(
        1
        for item in history
        if any(term in str(item.get("prediction", "")).lower() for term in risky_terms)
    )
    chunks = sum(int(item.get("chunks", 0) or 0) for item in history)
    avg_risk = sum(confidence_values) / len(confidence_values) if confidence_values else 0
    metrics = [
        {"label": "Files Analysed", "value": len(history), "icon": "📁", "color": "#6366F1"},
        {"label": "Threats Detected", "value": threats, "icon": "⚠️", "color": "#EF4444"},
        {"label": "Chunks Processed", "value": chunks, "icon": "🧩", "color": "#06B6D4"},
        {"label": "Avg Risk Score", "value": f"{avg_risk:.0f}%", "icon": "📊", "color": "#F59E0B"},
    ]
    tiles = []
    for item in metrics:
        tiles.append(
            f"""
            <div class="kpi-tile app-shell">
              <div class="kpi-icon">{html.escape(str(item["icon"]))}</div>
              <div class="kpi-value" style="color:{html.escape(str(item["color"]))}">
                {html.escape(str(item["value"]))}
              </div>
              <div class="kpi-label">{html.escape(str(item["label"]))}</div>
            </div>
            """
        )
    st.markdown(
        f'<div class="kpi-strip" style="--metric-count:{len(metrics)}">'
        + "".join(tiles)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header app-shell">
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
        <div class="feature-card app-shell">
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
        color = html.escape(str(metric.get("color", "var(--text-primary)")))
        tiles.append(
            f"""
            <div class="metric-tile app-shell">
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
        <div class="banner {kind} app-shell">
          <span class="banner-tag">{html.escape(code)}</span>
          <div>{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_content_card_open(accent: Literal["violet", "red", "green"] = "violet") -> None:
    st.markdown(
        f'<div class="content-card accent-{html.escape(accent)} app-shell">',
        unsafe_allow_html=True,
    )


def render_content_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_ready(message: str = "New analysis ready - scroll to results") -> None:
    try:
        st.toast(message, icon="✅")
    except Exception:
        pass
    st.markdown(
        f"""
        <div class="scroll-notify app-shell">
          <span class="scroll-notify-dot"></span>
          {html.escape(message)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        f"""
        <div id="toast-container" style="
          position:fixed; top:1.25rem; right:1.25rem; z-index:9999;
          display:flex; flex-direction:column; gap:0.5rem;
          pointer-events:none;">
        </div>
        <script>
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.style.cssText = `
          background:rgba(16,185,129,0.12);
          border:1px solid rgba(16,185,129,0.40);
          border-radius:12px;
          padding:0.7rem 1rem;
          font-family:'Plus Jakarta Sans',sans-serif;
          font-size:0.85rem;
          font-weight:600;
          color:#6EE7B7;
          min-width:260px;
          max-width:340px;
          display:flex;
          align-items:center;
          gap:0.6rem;
          backdrop-filter:blur(12px);
          pointer-events:all;
          animation:toast-in 300ms cubic-bezier(0.34,1.56,0.64,1) forwards;
        `;
        toast.innerHTML = `<span>✓</span><span>{html.escape(message)}</span>`;
        container.appendChild(toast);
        setTimeout(() => {{
          toast.style.animation = 'toast-out 250ms ease-in forwards';
          setTimeout(() => toast.remove(), 250);
        }}, 3500);
        </script>
        <style>
        @keyframes toast-in {{
          from {{ opacity:0; transform:translateX(30px) scale(0.92); }}
          to {{ opacity:1; transform:translateX(0) scale(1); }}
        }}
        @keyframes toast-out {{
          from {{ opacity:1; transform:translateX(0) scale(1); }}
          to {{ opacity:0; transform:translateX(30px) scale(0.90); }}
        }}
        </style>
        """,
        height=0,
    )


def render_result_card(title: str, risk_score: float, summary: str) -> None:
    if risk_score >= 70:
        level = "high"
        label = "High Risk"
    elif risk_score >= 40:
        level = "medium"
        label = "Suspicious"
    else:
        level = "safe"
        label = "Lower Risk"
    st.markdown(
        f"""
        <div class="result-card {level}">
          <span class="risk-badge {level}">{label} · {risk_score:.1f}%</span>
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(summary)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_placeholder(label: str = "Chart will appear here") -> None:
    st.markdown(
        f"""
        <div class="chart-placeholder">
          <div class="chart-spinner"></div>
          <span>{html.escape(label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig):
    fig.update_layout(**CHART_THEME)
    return fig


def render_demo_notice(root: Path) -> None:
    if official_data_present(str(root)):
        return
    st.markdown(
        """
        <div class="demo-warning app-shell">
          <span class="banner-tag">Demo</span>
          <div>
            <strong>Temporary demonstration data is active.</strong>
            Replace synthetic examples with the official datasets before reporting final model results.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(root: Path) -> None:
    model_rows = []
    for name, exists in get_model_status(str(root)).items():
        state = "ready" if exists else "missing"
        label = "Ready" if exists else "Missing"
        model_rows.append(
            f"""
            <div class="system-status-row">
              <span>{html.escape(name)}</span>
              <span class="system-status-state {state}">● {label}</span>
            </div>
            """
        )

    dataset_rows = []
    for name, exists in get_dataset_status(str(root)).items():
        state = "ready" if exists else "demo"
        label = "Ready" if exists else "Demo"
        dataset_rows.append(
            f"""
            <div class="system-status-row">
              <span>{html.escape(name)}</span>
              <span class="system-status-state {state}">● {label}</span>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="system-status">
          <div class="system-status-title">System Status</div>
          {''.join(model_rows)}
          <div style="height:1px;background:var(--border-subtle);margin:0.55rem 0;"></div>
          {''.join(dataset_rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_all_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
