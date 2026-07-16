"""Shared AI-FDS Streamlit UI helpers and cached app data."""

from __future__ import annotations

import html
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import streamlit as st


APP_TITLE = "AI-based Spam and Caller Fraud Detection System"
APP_SUBTITLE = "AI-FDS - Explainable fraud detection and student decision practice"

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
    "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#9CA3AF"},
    "xaxis": {
        "gridcolor": "rgba(148,163,184,0.09)",
        "linecolor": "rgba(148,163,184,0.16)",
        "zerolinecolor": "rgba(148,163,184,0.10)",
    },
    "yaxis": {
        "gridcolor": "rgba(148,163,184,0.09)",
        "linecolor": "rgba(148,163,184,0.16)",
        "zerolinecolor": "rgba(148,163,184,0.10)",
    },
    "colorway": ["#2563EB", "#0891B2", "#059669", "#D97706", "#DC2626", "#475569"],
}


@st.cache_data(show_spinner=False)
def get_demo_data() -> dict[str, object]:
    from src.data.demo_data import build_demo_bundle

    return build_demo_bundle()


@st.cache_data(show_spinner=False)
def get_model_status(root: str) -> dict[str, bool]:
    base = Path(root)
    return {name: (base / relative).exists() for name, relative in MODEL_ARTIFACTS.items()}


@st.cache_data(show_spinner=False)
def get_dataset_status(root: str) -> dict[str, bool]:
    base = Path(root)
    checks = {
        "Email datasets": base / "data/raw/email",
        "Transcript datasets": base / "data/raw/voice_transcript",
        "ASVspoof source": base / "data/raw/asvspoof_2019_dataset_subset",
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
        <script src="https://code.iconify.design/iconify-icon/2.1.0/iconify-icon.min.js"></script>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        :root {
            --bg-base:#0B1220;
            --bg-surface:#111827;
            --bg-raised:#172033;
            --bg-inset:#070E1B;
            --grad-primary:linear-gradient(135deg,#1D4ED8 0%,#0E7490 58%,#059669 100%);
            --grad-danger:linear-gradient(135deg,#B91C1C 0%,#D97706 100%);
            --grad-success:linear-gradient(135deg,#047857 0%,#0891B2 100%);
            --accent-violet:#2563EB;
            --accent-purple:#0E7490;
            --accent-cyan:#0891B2;
            --accent-emerald:#059669;
            --accent-amber:#D97706;
            --accent-red:#DC2626;
            --text-primary:#F8FAFC;
            --text-secondary:#A7B0BF;
            --text-muted:#6B7280;
            --border-subtle:rgba(148,163,184,.12);
            --border-medium:rgba(148,163,184,.24);
            --border-accent:rgba(14,116,144,.42);
            --glow-violet:0 0 18px rgba(37,99,235,.18);
            --glow-cyan:0 0 18px rgba(8,145,178,.16);
            --glow-red:0 0 18px rgba(220,38,38,.20);
            --radius-sm:8px;
            --radius-md:12px;
            --radius-lg:18px;
            --radius-xl:24px;
            --ease-spring:cubic-bezier(.34,1.56,.64,1);
            --ease-out:cubic-bezier(.16,1,.3,1);
        }
        body,.stApp,[data-testid="stAppViewContainer"]{font-family:'Plus Jakarta Sans',ui-sans-serif,system-ui,sans-serif;background:var(--bg-base);color:var(--text-primary);}
        code,pre,.mono,[data-testid="stCode"]{font-family:'JetBrains Mono',ui-monospace,monospace;}
        .stApp{background:radial-gradient(circle at 18% 6%,rgba(37,99,235,.10),transparent 28rem),radial-gradient(circle at 88% 12%,rgba(8,145,178,.08),transparent 26rem),var(--bg-base);}
        [data-testid="stHeader"]{background:rgba(11,18,32,.86);border-bottom:1px solid var(--border-subtle);backdrop-filter:blur(14px);}
        .block-container{max-width:1260px;padding:4.8rem 1.5rem 4rem;}
        h1{font-size:clamp(1.8rem,3.5vw,2.55rem);font-weight:800;letter-spacing:-.04em;color:var(--text-primary);}
        h2{font-size:clamp(1.3rem,2.5vw,1.7rem);font-weight:750;letter-spacing:-.025em;color:var(--text-primary);}
        h3{font-size:1.05rem;font-weight:650;color:var(--text-primary);}
        p,label,[data-testid="stCaptionContainer"]{color:var(--text-secondary);}
        .app-header{background:radial-gradient(circle at 18% 45%,rgba(37,99,235,.14),transparent 46%),radial-gradient(circle at 88% 20%,rgba(8,145,178,.12),transparent 42%),linear-gradient(135deg,rgba(17,24,39,.98),rgba(12,35,49,.92));border:1px solid rgba(8,145,178,.36);border-radius:var(--radius-xl);padding:1.7rem 2rem;margin-bottom:1.25rem;position:relative;overflow:hidden;box-shadow:0 12px 32px rgba(0,0,0,.24),var(--glow-cyan);}
        .app-header:before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--grad-primary);opacity:.75;}
        .app-header .eyebrow,.section-header .eyebrow{font-size:.72rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;background:var(--grad-primary);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
        .app-header h1{margin:0 0 .55rem;max-width:900px;position:relative;z-index:1;}
        .app-header p{margin:0;max-width:780px;color:var(--text-secondary);position:relative;z-index:1;}
        .status-row{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem;position:relative;z-index:1;}
        .status-chip{display:inline-flex;align-items:center;gap:6px;padding:.28rem .75rem;border-radius:999px;font-size:.78rem;font-weight:650;border:1px solid;backdrop-filter:blur(4px);}
        .status-chip .dot{width:6px;height:6px;border-radius:50%;animation:pulse-dot 2s infinite;}
        .status-chip.page{background:rgba(37,99,235,.14);border-color:rgba(37,99,235,.36);color:#BFDBFE;}
        .status-chip.data-demo,.status-chip.models-partial{background:rgba(217,119,6,.13);border-color:rgba(217,119,6,.34);color:#FCD34D;}
        .status-chip.data-official,.status-chip.models-ready{background:rgba(5,150,105,.13);border-color:rgba(5,150,105,.34);color:#86EFAC;}
        .status-chip.models-missing{background:rgba(220,38,38,.12);border-color:rgba(220,38,38,.33);color:#FCA5A5;}
        .status-chip.page .dot{background:var(--accent-violet)}.status-chip.data-demo .dot,.status-chip.models-partial .dot{background:var(--accent-amber)}.status-chip.data-official .dot,.status-chip.models-ready .dot{background:var(--accent-emerald)}.status-chip.models-missing .dot{background:var(--accent-red)}
        .section-header{margin:1.65rem 0 .9rem;}
        .section-header h2{font-size:1.42rem;font-weight:800;letter-spacing:-.025em;margin:0 0 .35rem;}
        .section-header p{color:var(--text-secondary);font-size:.92rem;line-height:1.55;margin:0;max-width:720px;}
        .banner,.demo-warning{display:flex;align-items:flex-start;gap:.75rem;padding:.85rem 1rem;border-radius:var(--radius-md);font-size:.88rem;line-height:1.55;margin:.75rem 0;border:1px solid;animation:slide-in 300ms var(--ease-out);color:var(--text-secondary);}
        .banner.info{background:rgba(37,99,235,.08);border-color:rgba(37,99,235,.28)}.banner.warning,.demo-warning{background:rgba(217,119,6,.08);border-color:rgba(217,119,6,.32);border-left:3px solid var(--accent-amber)}.banner.danger{background:rgba(220,38,38,.08);border-color:rgba(220,38,38,.30);border-left:3px solid var(--accent-red)}.banner.success{background:rgba(5,150,105,.08);border-color:rgba(5,150,105,.28);border-left:3px solid var(--accent-emerald)}
        .banner-tag{flex-shrink:0;font-size:.68rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;padding:.18rem .55rem;border-radius:4px;margin-top:.08rem;}
        .banner.info .banner-tag{background:rgba(37,99,235,.18);color:#BFDBFE}.banner.warning .banner-tag,.demo-warning .banner-tag{background:rgba(217,119,6,.20);color:#FCD34D}.banner.danger .banner-tag{background:rgba(220,38,38,.18);color:#FCA5A5}.banner.success .banner-tag{background:rgba(5,150,105,.18);color:#86EFAC}
        .content-card,.feature-card,.scenario-panel{background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);padding:1.2rem 1.3rem;margin-bottom:1.05rem;box-shadow:0 2px 12px rgba(0,0,0,.15);transition:border-color 200ms,box-shadow 200ms,transform 180ms var(--ease-spring);position:relative;overflow:hidden;}
        .content-card:hover,.feature-card:hover,.scenario-panel:hover{border-color:var(--border-medium);box-shadow:0 6px 22px rgba(0,0,0,.20);}
        .content-card.accent-violet:before,.content-card.accent-red:before,.content-card.accent-green:before{content:'';position:absolute;top:0;left:0;right:0;height:2px}.content-card.accent-violet:before{background:var(--grad-primary)}.content-card.accent-red:before{background:var(--grad-danger)}.content-card.accent-green:before{background:var(--grad-success)}
        .section-divider{display:flex;align-items:center;gap:14px;margin:1rem 0 1.25rem}.section-divider:before,.section-divider:after{content:'';flex:1;height:1px;background:rgba(148,163,184,.12)}.section-divider span{font-size:.74rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--text-muted)}
        .soft-panel{background:rgba(17,24,39,.45);border:1px solid rgba(148,163,184,.08);border-radius:14px;padding:1rem 1.1rem;margin:.5rem 0 1rem;color:var(--text-secondary);font-size:.9rem;line-height:1.6}
        .soft-panel p{margin:0;color:var(--text-secondary)}.soft-panel strong{color:var(--text-primary)}
        .accent-strip{border-left:4px solid var(--accent-cyan);padding-left:1rem;margin:1rem 0;color:var(--text-secondary)}.accent-strip h3{margin:0 0 .25rem}.accent-strip p{margin:0}
        .feature-card{min-height:145px}.feature-card:hover{transform:translateY(-2px)}
        .feature-index{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;margin-bottom:.85rem;border-radius:9px;background:rgba(37,99,235,.15);color:#BFDBFE;font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:750;border:1px solid rgba(37,99,235,.28);}
        .feature-card p{color:var(--text-secondary);font-size:.88rem;line-height:1.55;margin-bottom:0;}
        .kpi-strip,.metric-strip{display:grid;grid-template-columns:repeat(var(--metric-count),minmax(0,1fr));gap:.8rem;margin:0 0 1.25rem;}
        .kpi-tile,.metric-tile{background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:var(--radius-lg);padding:1rem 1.1rem;display:flex;flex-direction:column;gap:.3rem;box-shadow:0 2px 10px rgba(0,0,0,.12);transition:transform 180ms var(--ease-spring),box-shadow 180ms;}
        .kpi-tile:hover,.metric-tile:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.18)}
        .kpi-icon{font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:800;letter-spacing:.08em;color:var(--text-muted);text-transform:uppercase}.kpi-value,.metric-value{font-size:1.85rem;font-weight:800;letter-spacing:-.04em;line-height:1;font-family:'JetBrains Mono',monospace}.kpi-label,.metric-label{font-size:.76rem;font-weight:550;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em}
        .scroll-notify{display:flex;align-items:center;gap:8px;background:linear-gradient(90deg,rgba(37,99,235,.14),rgba(8,145,178,.10));border:1px solid rgba(8,145,178,.34);border-radius:999px;padding:.4rem 1rem;font-size:.82rem;font-weight:650;color:#BAE6FD;width:fit-content;margin:.5rem 0 1rem;animation:pulse-banner 2s ease-in-out infinite}.scroll-notify-dot{width:8px;height:8px;border-radius:50%;background:var(--accent-cyan);animation:pulse-dot 1s infinite}
        .result-card{border-radius:var(--radius-lg);padding:1rem 1.1rem;margin:.75rem 0 1rem;border:1px solid;animation:scale-in 220ms var(--ease-out)}.result-card.high{background:linear-gradient(135deg,rgba(220,38,38,.08),rgba(217,119,6,.05)),var(--bg-surface);border-color:rgba(220,38,38,.38);border-left:3px solid var(--accent-red)}.result-card.medium{background:linear-gradient(135deg,rgba(217,119,6,.08),rgba(202,138,4,.04)),var(--bg-surface);border-color:rgba(217,119,6,.36);border-left:3px solid var(--accent-amber)}.result-card.safe{background:linear-gradient(135deg,rgba(5,150,105,.08),rgba(8,145,178,.04)),var(--bg-surface);border-color:rgba(5,150,105,.30);border-left:3px solid var(--accent-emerald)}
        .risk-badge{display:inline-flex;align-items:center;gap:5px;padding:.22rem .65rem;border-radius:999px;font-size:.76rem;font-weight:750;text-transform:uppercase;letter-spacing:.05em;border:1px solid;margin-bottom:.65rem}.risk-badge.high{background:rgba(220,38,38,.15);color:#FCA5A5;border-color:rgba(220,38,38,.35)}.risk-badge.medium{background:rgba(217,119,6,.15);color:#FCD34D;border-color:rgba(217,119,6,.32)}.risk-badge.safe{background:rgba(5,150,105,.12);color:#86EFAC;border-color:rgba(5,150,105,.28)}
        [data-testid="stFileUploaderDropzone"]{background:var(--bg-inset)!important;border:1.5px dashed var(--border-medium)!important;border-radius:var(--radius-md)!important;padding:1rem!important;transition:border-color 200ms,background 200ms!important}[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent-cyan)!important;background:rgba(8,145,178,.04)!important}[data-testid="stFileUploaderDropzone"] button{background:var(--grad-primary)!important;color:#fff!important;border:none!important;border-radius:var(--radius-sm)!important;font-weight:750!important;font-size:.85rem!important;padding:.45rem 1rem!important;box-shadow:var(--glow-cyan)!important}
        [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{background:var(--bg-inset)!important;border:1px solid var(--border-medium)!important;border-radius:var(--radius-sm)!important;color:var(--text-primary)!important;font-family:'JetBrains Mono',monospace!important;font-size:.88rem!important;transition:border-color 180ms!important}[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-color:var(--accent-cyan)!important;box-shadow:0 0 0 3px rgba(8,145,178,.14)!important;outline:none!important}
        [data-testid="stSelectbox"]>div>div,[data-baseweb="select"]>div{background:var(--bg-inset)!important;border:1px solid var(--border-medium)!important;border-radius:var(--radius-sm)!important}
        [data-testid="stButton"]>button[kind="primary"],.stButton>button[kind="primary"]{background:var(--grad-primary)!important;color:#fff!important;border:none!important;border-radius:var(--radius-md)!important;font-weight:750!important;font-size:.9rem!important;padding:.55rem 1.4rem!important;box-shadow:var(--glow-cyan)!important;transition:transform 150ms var(--ease-spring),box-shadow 150ms!important}[data-testid="stButton"]>button[kind="primary"]:hover,.stButton>button[kind="primary"]:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(8,145,178,.30)!important}
        [data-testid="stButton"]>button[kind="secondary"],.stButton>button[kind="secondary"],.stDownloadButton>button{background:var(--bg-raised)!important;color:var(--text-secondary)!important;border:1px solid var(--border-medium)!important;border-radius:var(--radius-md)!important;font-weight:650!important;transition:background 150ms,border-color 150ms!important}[data-testid="stButton"]>button[kind="secondary"]:hover,.stButton>button[kind="secondary"]:hover,.stDownloadButton>button:hover{background:rgba(8,145,178,.10)!important;border-color:var(--border-accent)!important;color:var(--text-primary)!important}
        [data-testid="stExpander"],[data-testid="stMetric"],[data-testid="stDataFrame"]{background:var(--bg-surface)!important;border:1px solid var(--border-subtle)!important;border-radius:var(--radius-md)!important;overflow:hidden!important}[data-testid="stMetric"]{padding:.85rem 1rem}[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace}
        .stTabs [data-baseweb="tab-list"]{gap:.25rem;padding:.25rem;border:1px solid var(--border-subtle);border-radius:var(--radius-lg);background:var(--bg-inset)}.stTabs [data-baseweb="tab"]{height:2.65rem;padding:0 .85rem;border-radius:var(--radius-md);color:var(--text-secondary);font-weight:750}.stTabs [aria-selected="true"]{color:#fff!important;background:linear-gradient(135deg,rgba(37,99,235,.20),rgba(8,145,178,.12))!important}.stTabs [data-baseweb="tab-highlight"]{display:none}
        [data-testid="stDivider"] hr,hr{border:none!important;height:1px!important;background:var(--border-subtle)!important;margin:1rem 0!important}mark{padding:.08rem .22rem;border-radius:4px;color:#FCD34D;background:rgba(217,119,6,.18)}

        /* ---------- COMPACT KPI ROW ---------- */
        .compact-kpi-row {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:18px;
            margin:18px 0 28px;
        }
        .compact-kpi-card {
            height:92px;
            display:flex;
            align-items:center;
            gap:24px;
            padding:18px 24px;
            border-radius:22px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)),
                #101827;
            border:1px solid rgba(148,163,184,0.20);
            box-shadow:
                0 14px 34px rgba(15,23,42,0.28),
                inset 0 1px 0 rgba(255,255,255,0.06);
            overflow:hidden;
            position:relative;
        }
        .kpi-icon-wrapper {
            width:58px;
            height:58px;
            flex:0 0 58px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:18px;
            background:rgba(15,23,42,0.55);
            position:relative;
        }
        .kpi-icon-wrapper .kpi-orbit {
            position:absolute;
            inset:-8px;
            border-radius:999px;
            border:1px dashed currentColor;
            opacity:.45;
            animation:orbitSpin 9s linear infinite;
        }
        .kpi-icon-wrapper iconify-icon {
            position:relative;
            z-index:1;
            font-size:32px;
        }
        .kpi-icon-wrapper .kpi-mask-icon {
            position:relative;
            z-index:1;
            display:inline-block;
            width:32px;
            height:32px;
            background:currentColor;
            -webkit-mask:var(--kpi-icon) center / contain no-repeat;
            mask:var(--kpi-icon) center / contain no-repeat;
        }
        .kpi-icon-wrapper.blue,
        .kpi-value.blue {color:#3B82F6;}
        .kpi-icon-wrapper.red,
        .kpi-value.red {color:#F43F5E;}
        .kpi-icon-wrapper.orange,
        .kpi-value.orange {color:#F97316;}
        .compact-kpi-card .kpi-title {
            color:#CBD5E1;
            font-size:1rem;
            font-weight:750;
            white-space:nowrap;
        }
        .compact-kpi-card .kpi-divider {
            width:1px;
            height:38px;
            background:rgba(203,213,225,0.22);
            margin-left:auto;
        }
        .compact-kpi-card .kpi-value {
            min-width:72px;
            text-align:right;
            font-size:2.1rem;
            line-height:1;
            font-weight:900;
            letter-spacing:-0.04em;
            font-family:'JetBrains Mono',monospace;
        }
        @keyframes orbitSpin {
            from {transform:rotate(0deg);}
            to {transform:rotate(360deg);}
        }
        @media(max-width:900px) {
            .compact-kpi-row {grid-template-columns:1fr;}
            .compact-kpi-card {height:86px;}
        }

        /* ---------- DETECTION TOOL INTRO CARD ---------- */
        .detection-tool-intro {
            display:flex;
            align-items:center;
            gap:28px;
            width:100%;
            padding:28px 32px;
            margin:18px 0 22px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015)),
                #101827;
            border:1px solid rgba(148,163,184,0.20);
            border-radius:22px;
            box-shadow:
                0 14px 34px rgba(15,23,42,0.22),
                inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .detection-tool-icon {
            width:82px;
            height:82px;
            flex:0 0 82px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:26px;
            background:rgba(37,99,235,0.12);
            color:#3B82F6;
        }
        .detection-tool-icon iconify-icon {
            font-size:44px;
        }
        .detection-tool-mask-icon {
            display:inline-block;
            width:44px;
            height:44px;
            background:currentColor;
            -webkit-mask:var(--tool-icon) center / contain no-repeat;
            mask:var(--tool-icon) center / contain no-repeat;
        }
        .detection-tool-copy {
            flex:1;
            min-width:0;
        }
        .detection-tool-copy h2 {
            margin:0 0 10px;
            color:#F8FAFC!important;
            font-size:1.65rem;
            font-weight:850;
            letter-spacing:-0.035em;
        }
        .detection-tool-copy p {
            margin:0;
            width:100%;
            max-width:none;
            color:#B6C2D6!important;
            font-size:1rem;
            line-height:1.65;
            font-weight:500;
        }
        .detection-tool-intro.purple .detection-tool-icon {
            background:rgba(139,92,246,0.14);
            color:#A78BFA;
        }
        .detection-tool-intro.orange .detection-tool-icon {
            background:rgba(249,115,22,0.14);
            color:#F97316;
        }
        @media(max-width:700px) {
            .detection-tool-intro {
                align-items:flex-start;
                gap:18px;
                padding:22px;
            }
            .detection-tool-icon {
                width:64px;
                height:64px;
                flex-basis:64px;
                border-radius:20px;
            }
            .detection-tool-icon iconify-icon {
                font-size:34px;
            }
            .detection-tool-mask-icon {
                width:34px;
                height:34px;
            }
            .detection-tool-copy h2 {
                font-size:1.35rem;
            }
        }

        /* ---------- EMAIL EVIDENCE WORKFLOW ---------- */
        .email-step-header {
            display:flex;
            align-items:flex-start;
            gap:14px;
            margin:20px 0 10px;
        }
        .email-step-header>span {
            width:34px;
            height:34px;
            flex:0 0 34px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:12px;
            background:linear-gradient(135deg, rgba(37,99,235,.28), rgba(8,145,178,.16));
            border:1px solid rgba(96,165,250,.28);
            color:#BFDBFE;
            font-family:'JetBrains Mono',monospace;
            font-size:.78rem;
            font-weight:850;
            box-shadow:0 10px 24px rgba(37,99,235,.12);
        }
        .email-step-header h3 {
            margin:0 0 4px;
            color:#F8FAFC!important;
            font-size:1.02rem;
            line-height:1.2;
            font-weight:800;
            letter-spacing:-.018em;
        }
        .email-step-header p {
            margin:0;
            color:#9CA3AF!important;
            font-size:.86rem;
            line-height:1.55;
        }
        .email-source-card {
            min-height:102px;
            display:flex;
            align-items:flex-start;
            gap:14px;
            padding:18px;
            margin-bottom:14px;
            border-radius:18px;
            background:linear-gradient(135deg, rgba(15,23,42,.78), rgba(8,16,32,.62));
            border:1px solid rgba(148,163,184,.16);
            box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
        }
        .email-source-icon {
            width:46px;
            height:46px;
            flex:0 0 46px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:15px;
            background:rgba(37,99,235,.16);
            border:1px solid rgba(96,165,250,.24);
            color:#93C5FD;
            font-family:'JetBrains Mono',monospace;
            font-size:.65rem;
            font-weight:850;
            letter-spacing:.04em;
        }
        .email-source-card h4 {
            margin:0 0 6px;
            color:#F8FAFC!important;
            font-size:1rem;
            font-weight:800;
            letter-spacing:-.015em;
        }
        .email-source-card p {
            margin:0;
            color:#A7B0BF!important;
            font-size:.84rem;
            line-height:1.55;
        }
        .email-format-row {
            display:flex;
            flex-wrap:wrap;
            align-items:center;
            gap:8px;
            margin-top:10px;
            color:#9CA3AF;
            font-size:.82rem;
        }
        .email-format-row strong {
            color:#CBD5E1;
            margin-right:2px;
        }
        .email-format-row span {
            display:inline-flex;
            align-items:center;
            height:26px;
            padding:0 10px;
            border-radius:999px;
            background:rgba(37,99,235,.12);
            border:1px solid rgba(96,165,250,.20);
            color:#BFDBFE;
            font-family:'JetBrains Mono',monospace;
            font-size:.72rem;
            font-weight:750;
        }
        .email-panel-title {
            margin:0 0 4px;
            color:#F8FAFC;
            font-size:1rem;
            font-weight:820;
            letter-spacing:-.015em;
        }
        .email-engine-card {
            height:100%;
            min-height:190px;
            padding:18px;
            border-radius:18px;
            background:
                radial-gradient(circle at 12% 18%, rgba(37,99,235,.16), transparent 9rem),
                linear-gradient(135deg, rgba(15,23,42,.78), rgba(8,16,32,.72));
            border:1px solid rgba(148,163,184,.16);
        }
        .email-engine-title {
            color:#CBD5E1;
            font-size:.78rem;
            font-weight:850;
            letter-spacing:.1em;
            text-transform:uppercase;
            margin-bottom:12px;
        }
        .email-engine-state {
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:6px 11px;
            border-radius:999px;
            font-size:.82rem;
            font-weight:850;
            margin-bottom:14px;
        }
        .email-engine-state span {
            width:8px;
            height:8px;
            border-radius:999px;
            animation:pulse-dot 1.8s ease-in-out infinite;
        }
        .email-engine-state.ready {
            color:#86EFAC;
            background:rgba(5,150,105,.12);
            border:1px solid rgba(5,150,105,.28);
        }
        .email-engine-state.ready span {
            background:#34D399;
        }
        .email-engine-state.missing {
            color:#FCA5A5;
            background:rgba(220,38,38,.12);
            border:1px solid rgba(220,38,38,.28);
        }
        .email-engine-state.missing span {
            background:#F87171;
        }
        .email-engine-count {
            color:#F8FAFC;
            font-size:1.28rem;
            font-weight:850;
            letter-spacing:-.035em;
            margin-bottom:12px;
        }
        .email-engine-models {
            display:flex;
            flex-wrap:wrap;
            gap:7px;
        }
        .email-engine-models span {
            padding:5px 9px;
            border-radius:999px;
            font-size:.72rem;
            font-weight:750;
            border:1px solid;
        }
        .email-engine-models span.ready {
            color:#BFDBFE;
            background:rgba(37,99,235,.12);
            border-color:rgba(96,165,250,.22);
        }
        .email-engine-models span.missing {
            color:#9CA3AF;
            background:rgba(148,163,184,.06);
            border-color:rgba(148,163,184,.14);
        }
        .email-analyze-card {
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:18px;
            margin:20px 0 12px;
            padding:20px 22px;
            border-radius:20px;
            background:
                radial-gradient(circle at 12% 0%, rgba(37,99,235,.14), transparent 14rem),
                linear-gradient(135deg, rgba(17,24,39,.92), rgba(10,18,32,.92));
            border:1px solid rgba(148,163,184,.18);
            box-shadow:0 14px 30px rgba(0,0,0,.16);
        }
        .email-analyze-eyebrow {
            color:#60A5FA;
            font-size:.7rem;
            font-weight:850;
            letter-spacing:.12em;
            text-transform:uppercase;
            margin-bottom:8px;
        }
        .email-analyze-card h3 {
            margin:0 0 4px;
            color:#F8FAFC!important;
            font-size:1rem;
            font-weight:800;
        }
        .email-analyze-card p {
            margin:0;
            color:#A7B0BF!important;
            font-size:.92rem;
        }
        @media(max-width:760px) {
            .email-analyze-card {
                grid-template-columns:1fr;
            }
        }

        /* ---------- DETECTION CENTER HEADER + RHOMBUS NAV ---------- */
        .detection-center-hero {
            display:flex;
            align-items:center;
            gap:22px;
            margin:10px 0 18px;
        }
        .detection-center-hero-icon {
            width:70px;
            height:70px;
            flex:0 0 70px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:22px;
            color:#7C6CFF;
            background:
                radial-gradient(circle at 32% 24%, rgba(96,165,250,.38), transparent 28px),
                radial-gradient(circle at 72% 82%, rgba(124,108,255,.28), transparent 36px),
                linear-gradient(135deg, rgba(124,108,255,.28), rgba(15,23,42,.55));
            box-shadow:0 16px 34px rgba(37,99,235,.18);
        }
        .detection-center-hero-icon .hero-svg-icon {
            width:42px;
            height:42px;
        }
        .detection-center-hero-copy h1 {
            margin:0 0 8px;
            color:#F8FAFC!important;
            font-size:clamp(1.8rem,3vw,2.45rem);
            line-height:1.08;
            font-weight:850;
            letter-spacing:-.05em;
        }
        .detection-center-hero-copy p {
            margin:0;
            max-width:980px;
            color:#B6C2D6!important;
            font-size:.98rem;
            line-height:1.65;
            font-weight:500;
        }
        .st-key-detection_tab_wrapper {
            margin:8px 0 0;
        }
        .st-key-detection_tab_wrapper [data-testid="stHorizontalBlock"] {
            gap:0!important;
        }
        .st-key-detection_tab_wrapper [data-testid="column"] {
            padding:0!important;
            position:relative;
            overflow:visible!important;
        }

        [class*="st-key-detection_tab_card_"] {
            position:relative;
            height:64px;
            min-height:64px;
            overflow:visible!important;
        }

        [class*="st-key-detection_tab_card_"][data-testid="stVerticalBlock"] {
            gap:0!important;
        }

        [class*="st-key-detection_tab_card_"] [data-testid="stElementContainer"]:has([data-testid="stButton"]),
        [class*="st-key-detection_tab_card_"] [data-testid="stElementContainer"][class*="st-key-detection_tab_button_"] {
            position:absolute!important;
            inset:0!important;
            z-index:5!important;
            height:64px!important;
            margin:0!important;
        }

        [class*="st-key-detection_tab_card_"] .stButton,
        [class*="st-key-detection_tab_card_"] [data-testid="stButton"] {
            width:100%!important;
            height:64px!important;
            margin:0!important;
        }

        [class*="st-key-detection_tab_card_"] .stButton>button,
        [class*="st-key-detection_tab_card_"] [data-testid="stButton"]>button {
            width:100%!important;
            height:64px!important;
            min-height:64px!important;
            margin:0!important;
            padding:0!important;
            opacity:0!important;
            cursor:pointer!important;
            border-radius:18px!important;
            transform:none!important;
            box-shadow:none!important;
        }

        [class*="st-key-detection_tab_card_"] .stButton>button:hover,
        [class*="st-key-detection_tab_card_"] [data-testid="stButton"]>button:hover {
            transform:none!important;
            box-shadow:none!important;
        }
        .detection-tab-visual {
            position:relative;
            z-index:2;
            height:64px;
            margin:0;
            padding:0 24px;
            display:flex;
            align-items:center;
            justify-content:center;
            gap:13px;
            background:linear-gradient(135deg, rgba(15,23,42,.96), rgba(8,16,32,.96));
            border:1px solid rgba(148,163,184,.24);
            color:#AEB7CB;
            box-shadow:inset 0 1px 0 rgba(255,255,255,.04), 0 10px 24px rgba(0,0,0,.16);
            clip-path:polygon(0 0, calc(100% - 22px) 0, 100% 50%, calc(100% - 22px) 100%, 0 100%, 14px 50%);
            transition:all 180ms ease;
            pointer-events:none;
        }
        .st-key-detection_tab_wrapper [data-testid="column"]:first-of-type .detection-tab-visual {
            border-radius:16px 0 0 16px;
            clip-path:polygon(0 0, calc(100% - 22px) 0, 100% 50%, calc(100% - 22px) 100%, 0 100%);
        }
        .st-key-detection_tab_wrapper [data-testid="column"]:last-of-type .detection-tab-visual {
            border-radius:0 16px 16px 0;
            clip-path:polygon(14px 0, 100% 0, 100% 100%, 14px 100%, 0 50%);
        }
        .detection-tab-visual .detection-tab-icon {
            width:24px;
            height:24px;
            background:currentColor;
            -webkit-mask:var(--tab-icon) center / contain no-repeat;
            mask:var(--tab-icon) center / contain no-repeat;
        }
        .detection-tab-visual span {
            color:inherit!important;
            font-size:.92rem;
            font-weight:750;
            letter-spacing:-.01em;
        }
        .detection-tab-visual.active {
            z-index:2;
            color:#F8FAFC;
            background:radial-gradient(circle at 20% 50%, rgba(37,99,235,.35), transparent 18rem),
                linear-gradient(135deg, rgba(30,64,175,.95), rgba(15,23,42,.96));
            border:1.5px solid #2563EB;
            box-shadow:0 0 0 1px rgba(37,99,235,.25), 0 14px 30px rgba(37,99,235,.20);
        }
        .detection-tab-visual.active .detection-tab-icon {
            color:#60A5FA;
        }
        [class*="st-key-detection_tab_card_"]:hover .detection-tab-visual {
            color:#F8FAFC;
            border-color:rgba(96,165,250,.55);
        }
        .detection-tool-intro {
            margin-top:18px;
        }
        .detection-tool-copy h2 {
            font-size:1.35rem!important;
        }
        .detection-tool-copy p {
            font-size:.9rem!important;
            line-height:1.6!important;
        }

        /* ---------- HEADER BANNER ---------- */
        .hero-banner {
            position:relative;
            overflow:hidden;
            min-height:310px;
            padding:42px 48px;
            background:
                radial-gradient(circle at 86% 42%, rgba(99,102,241,0.28), transparent 22rem),
                radial-gradient(circle at 94% 8%, rgba(14,165,233,0.18), transparent 18rem),
                linear-gradient(135deg, #111827 0%, #0B1220 48%, #161A35 100%);
            border:1px solid rgba(148,163,184,.18);
            border-radius:28px;
            box-shadow:
                0 22px 55px rgba(0,0,0,0.28),
                inset 0 1px 0 rgba(255,255,255,0.08);
            color:#F8FAFC!important;
        }
        .hero-banner::before {
            content:"";
            position:absolute;
            inset:0;
            background:
                linear-gradient(125deg, transparent 0%, transparent 55%, rgba(99,102,241,0.10) 55%, rgba(14,165,233,0.14) 100%);
            pointer-events:none;
        }
        .hero-content {
            position:relative;
            z-index:2;
            max-width:720px;
        }
        .hero-banner h1 {
            margin:0;
            margin-bottom:0;
            max-width:760px;
            color:#F8FAFC!important;
            font-size: clamp(1.7rem, 3vw, 3rem);
            line-height:1.16;
            font-weight:900;
            letter-spacing:-0.055em;
        }
        .hero-banner p {
            margin-top:28px;
            margin-bottom:0;
            max-width:740px;
            color:#B6C2D6!important;
            font-size: clamp(0.95rem, 1.05vw, 1.05rem);
            line-height:1.65;
            font-weight:500;
        }
        .hero-visual {
            position:absolute;
            right:70px;
            top:50%;
            transform:translateY(-50%);
            width:320px;
            height:320px;
            z-index:1;
        }
        .hero-orbit {
            position:absolute;
            inset:40px;
            border:2px dashed rgba(124,108,255,.18);
            border-radius:50%;
            box-shadow:0 0 42px rgba(124,108,255,.12);
        }
        .hero-orbit.orbit-two {
            inset:72px;
            border-style:solid;
            opacity:.42;
        }
        .shield-main {
            position:absolute;
            inset:0;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#7C6CFF;
            filter:drop-shadow(0 18px 42px rgba(124,108,255,.22));
        }
        .shield-main iconify-icon {
            font-size:130px;
        }
        .shield-main .hero-svg-icon {
            width:130px;
            height:130px;
        }
        .floating-icon {
            position:absolute;
            width:54px;
            height:54px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:18px;
            background:rgba(255,255,255,.12);
            border:1px solid rgba(199,210,254,.24);
            color:#A7B4FF;
            box-shadow:0 8px 22px rgba(0,0,0,.12);
            backdrop-filter:blur(14px);
            animation:hero-float 4.8s ease-in-out infinite;
        }
        .floating-icon iconify-icon {
            font-size:28px;
        }
        .hero-svg-icon {
            display:inline-block;
            width:28px;
            height:28px;
            background:currentColor;
            -webkit-mask:var(--hero-icon) center / contain no-repeat;
            mask:var(--hero-icon) center / contain no-repeat;
        }
        .icon-chat {left:50%;top:20px;transform:translateX(-50%);animation-delay:-.4s;}
        .icon-mic {left:22px;top:50%;transform:translateY(-50%);animation-delay:-1.4s;}
        .icon-phone {right:22px;top:50%;transform:translateY(-50%);animation-delay:-2.1s;}
        .icon-audio {left:50%;bottom:20px;transform:translateX(-50%);animation-delay:-3s;}
        @keyframes hero-float {
            0%,100% {translate:0 0;}
            50% {translate:0 -8px;}
        }
        @media(max-width:1050px) {
            .hero-visual {opacity:.38;right:20px;}
            .hero-content {max-width:760px;}
        }
        @media(max-width:760px) {
            .hero-banner {min-height:auto;padding:30px 26px;border-radius:22px;}
            .hero-banner h1 {font-size:clamp(2rem,9vw,3rem);}
            .hero-banner p {margin-top:20px;}
            .hero-visual {display:none;}
        }
        /* =========================================================
           AI-FDS SIDEBAR REDESIGN
           ========================================================= */

        [data-testid="stSidebar"] {
            width:310px!important;
            min-width:300px!important;
            max-width:320px!important;
            height:100vh!important;
            background:
                radial-gradient(circle at 12% 4%, rgba(37,99,235,.10), transparent 15rem),
                linear-gradient(180deg, #0E1829 0%, #0B1424 100%)!important;
            border-right:1px solid rgba(148,163,184,.14)!important;
        }

        [data-testid="stSidebar"] > div:first-child {
            width:100%!important;
            height:100vh!important;
            padding:.9rem .95rem 1rem!important;
            box-sizing:border-box!important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            overflow-x:hidden!important;
            overflow-y:hidden!important;
            height:100%!important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            min-height:calc(100vh - 2rem)!important;
            display:flex!important;
            flex-direction:column!important;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap:.35rem;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebar"] button[title*="sidebar" i],
        button[title*="sidebar" i],
        button[aria-label*="sidebar" i] {
            display:none!important;
            visibility:hidden!important;
            pointer-events:none!important;
        }

        .aifds-sidebar-brand {
            display:flex;
            align-items:center;
            gap:.82rem;
            padding:.45rem .35rem .9rem;
            margin-bottom:.75rem;
            border-bottom:1px solid rgba(148,163,184,.13);
        }

        .aifds-sidebar-logo {
            position:relative;
            flex:0 0 48px;
            width:48px;
            height:54px;
            display:flex;
            align-items:center;
            justify-content:center;
            clip-path:polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
            background:linear-gradient(145deg, #2563EB 0%, #0891B2 55%, #10B981 100%);
            box-shadow:0 12px 28px rgba(37,99,235,.24);
        }

        .aifds-sidebar-logo .aifds-sidebar-icon-mask {
            position:relative;
            width:32px;
            height:32px;
            color:#FFFFFF;
            opacity:.96;
            filter:drop-shadow(0 4px 10px rgba(255,255,255,.22));
        }

        .aifds-sidebar-brand-copy {
            min-width:0;
        }

        .aifds-sidebar-brand-title {
            color:#F8FAFC;
            font-size:1.02rem;
            font-weight:850;
            line-height:1.15;
            letter-spacing:0;
        }

        .aifds-sidebar-brand-subtitle {
            margin-top:.22rem;
            color:#77849A;
            font-size:.66rem;
            font-weight:550;
            line-height:1.3;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }

        .st-key-sidebar_navigation {
            margin-bottom:.65rem;
            padding:.12rem .08rem .22rem;
            flex:0 0 auto;
            overflow:visible!important;
        }

        [class*="st-key-sidebar_nav_card_"] {
            position:relative;
            height:86px;
            min-height:86px;
            padding:4px 2px;
            margin-bottom:.42rem!important;
            overflow:visible!important;
            box-sizing:border-box;
        }

        [class*="st-key-sidebar_nav_card_"][data-testid="stVerticalBlock"] {
            gap:0!important;
            overflow:visible!important;
        }

        .aifds-sidebar-nav-visual {
            position:relative;
            z-index:2;
            pointer-events:none;
            width:100%;
            height:78px;
            min-height:78px;
            display:grid;
            grid-template-columns:42px minmax(0,1fr) 18px;
            align-items:center;
            gap:.68rem;
            box-sizing:border-box;
            padding:.6rem .72rem;
            border:1px solid rgba(148,163,184,.18);
            border-radius:15px;
            background:linear-gradient(145deg, rgba(22,32,51,.96), rgba(14,24,41,.96));
            box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
            transform:translateY(-1px);
            transition:border-color 180ms ease, box-shadow 180ms ease;
        }

        .aifds-sidebar-nav-visual.active {
            border-color:#2563EB;
            background:
                radial-gradient(circle at 18% 50%, rgba(37,99,235,.24), transparent 9rem),
                linear-gradient(145deg, rgba(20,38,72,.98), rgba(14,26,48,.98));
            box-shadow:
                0 0 0 1px rgba(37,99,235,.18),
                0 12px 26px rgba(37,99,235,.16),
                inset 0 1px 0 rgba(255,255,255,.04);
        }

        .aifds-sidebar-nav-icon {
            width:40px;
            height:40px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:12px;
            color:#FFFFFF;
        }

        .aifds-sidebar-nav-visual.blue .aifds-sidebar-nav-icon {
            background:linear-gradient(145deg, #2563EB, #1D4ED8);
            box-shadow:0 9px 22px rgba(37,99,235,.24);
        }

        .aifds-sidebar-nav-visual.green .aifds-sidebar-nav-icon {
            background:linear-gradient(145deg, #10B981, #059669);
            box-shadow:0 9px 22px rgba(16,185,129,.20);
        }

        .aifds-sidebar-icon-mask {
            display:inline-block;
            width:1em;
            height:1em;
            flex:0 0 auto;
            background:currentColor;
            -webkit-mask:var(--sidebar-icon) center / contain no-repeat;
            mask:var(--sidebar-icon) center / contain no-repeat;
        }

        .aifds-sidebar-nav-icon .aifds-sidebar-icon-mask {
            width:23px;
            height:23px;
        }

        .aifds-sidebar-nav-copy {
            min-width:0;
        }

        .aifds-sidebar-nav-title {
            color:#F8FAFC;
            font-size:.8rem;
            font-weight:800;
            line-height:1.25;
            letter-spacing:0;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }

        .aifds-sidebar-nav-description {
            margin-top:.23rem;
            color:#94A3B8;
            font-size:.59rem;
            line-height:1.3;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }

        .aifds-sidebar-nav-arrow {
            color:#94A3B8;
            width:19px;
            height:19px;
        }

        [class*="st-key-sidebar_nav_card_"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
            position:absolute!important;
            inset:4px 2px!important;
            z-index:5!important;
            margin:0!important;
            height:78px!important;
        }

        [class*="st-key-sidebar_nav_card_"] .stButton,
        [class*="st-key-sidebar_nav_card_"] [data-testid="stButton"] {
            width:100%!important;
            height:100%!important;
            margin:0!important;
        }

        [class*="st-key-sidebar_nav_card_"] .stButton > button,
        [class*="st-key-sidebar_nav_card_"] [data-testid="stButton"] > button {
            width:100%!important;
            height:78px!important;
            min-height:78px!important;
            margin:0!important;
            padding:0!important;
            opacity:0!important;
            cursor:pointer!important;
        }

        [class*="st-key-sidebar_nav_card_"]:hover .aifds-sidebar-nav-visual {
            border-color:rgba(96,165,250,.52);
            box-shadow:
                0 12px 28px rgba(0,0,0,.20),
                inset 0 1px 0 rgba(255,255,255,.04);
        }

        .st-key-sidebar_status_dock {
            margin-top:auto!important;
            padding-top:.5rem;
            flex:0 0 auto;
        }

        .aifds-sidebar-status-panel {
            padding:.72rem .72rem .66rem;
            border:1px solid rgba(148,163,184,.14);
            border-radius:14px;
            background:linear-gradient(160deg, rgba(10,19,34,.96), rgba(8,16,29,.96));
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,.025),
                0 12px 28px rgba(0,0,0,.16);
        }

        .aifds-sidebar-status-heading {
            display:flex;
            align-items:center;
            gap:.48rem;
            padding:.05rem .1rem .52rem;
            color:#94A3B8;
            font-size:.64rem;
            font-weight:850;
            letter-spacing:0;
        }

        .aifds-sidebar-status-live-dot {
            width:7px;
            height:7px;
            border-radius:999px;
            background:#10B981;
            box-shadow:0 0 12px rgba(16,185,129,.55);
        }

        .aifds-sidebar-status-divider {
            height:1px;
            margin:0 0 .3rem;
            background:rgba(148,163,184,.12);
        }

        .aifds-sidebar-model-list {
            display:flex;
            flex-direction:column;
            gap:.18rem;
        }

        .aifds-sidebar-model-group {
            border-bottom:1px solid rgba(148,163,184,.08);
        }

        .aifds-sidebar-model-group:last-child {
            border-bottom:none;
        }

        .aifds-sidebar-model-summary {
            list-style:none;
            display:grid;
            grid-template-columns:24px minmax(0,1fr) auto 12px;
            align-items:center;
            gap:.42rem;
            min-height:34px;
            padding:.26rem .04rem;
            cursor:pointer;
        }

        .aifds-sidebar-model-summary::-webkit-details-marker {
            display:none;
        }

        .aifds-sidebar-model-summary::after {
            content:"";
            width:9px;
            height:9px;
            border-right:1.5px solid #94A3B8;
            border-bottom:1.5px solid #94A3B8;
            transform:rotate(45deg) translateY(-2px);
            transition:transform 160ms ease;
        }

        .aifds-sidebar-model-group[open] .aifds-sidebar-model-summary::after {
            transform:rotate(225deg) translateY(-2px);
        }

        .aifds-sidebar-model-icon {
            width:24px;
            height:24px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:7px;
            color:#DCEBFF;
            background:rgba(37,99,235,.15);
            border:1px solid rgba(148,163,184,.10);
        }

        .aifds-sidebar-model-icon .aifds-sidebar-icon-mask {
            width:15px;
            height:15px;
        }

        .aifds-sidebar-model-icon.blue {
            color:#60A5FA;
            background:rgba(37,99,235,.17);
        }

        .aifds-sidebar-model-icon.purple {
            color:#A78BFA;
            background:rgba(139,92,246,.17);
        }

        .aifds-sidebar-model-icon.pink {
            color:#EC4899;
            background:rgba(236,72,153,.15);
        }

        .aifds-sidebar-model-icon.orange {
            color:#F97316;
            background:rgba(249,115,22,.15);
        }

        .aifds-sidebar-model-detail-list {
            display:flex;
            flex-direction:column;
            padding:0 0 .24rem 1.82rem;
        }

        .aifds-sidebar-detail-row {
            display:grid;
            grid-template-columns:minmax(0,1fr) auto;
            align-items:center;
            gap:.45rem;
            min-height:28px;
            padding:.22rem 0;
            border-top:1px solid rgba(148,163,184,.07);
        }

        .aifds-sidebar-detail-row:last-child {
            border-bottom:none;
        }

        .aifds-sidebar-model-copy,
        .aifds-sidebar-detail-copy {
            min-width:0;
        }

        .aifds-sidebar-model-name,
        .aifds-sidebar-detail-name {
            color:#E5EEF9;
            font-size:.64rem;
            font-weight:760;
            line-height:1.2;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }

        .aifds-sidebar-detail-date {
            margin-top:.12rem;
            color:#76849B;
            font-size:.55rem;
            font-weight:580;
            line-height:1.15;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }

        .aifds-sidebar-status-pill {
            flex:0 0 auto;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            gap:.3rem;
            min-width:58px;
            padding:.22rem .46rem;
            border-radius:999px;
            font-size:.56rem;
            font-weight:800;
            line-height:1;
            text-transform:lowercase;
        }

        .aifds-sidebar-status-pill > span {
            width:5px;
            height:5px;
            border-radius:999px;
        }

        .aifds-sidebar-status-pill.active {
            color:#34D399;
            background:rgba(5,150,105,.14);
        }

        .aifds-sidebar-status-pill.active > span {
            background:#34D399;
        }

        .aifds-sidebar-status-pill.pending {
            color:#FBBF24;
            background:rgba(217,119,6,.14);
        }

        .aifds-sidebar-status-pill.pending > span {
            background:#F59E0B;
        }

        .aifds-sidebar-status-pill.inactive {
            color:#F87171;
            background:rgba(220,38,38,.13);
        }

        .aifds-sidebar-status-pill.inactive > span {
            background:#F87171;
        }

        @media(max-width:900px) {
            [data-testid="stSidebar"] {
                min-width:280px!important;
            }

            .aifds-sidebar-nav-description,
            .aifds-sidebar-brand-subtitle {
                white-space:normal;
            }
        }

        @keyframes fade-up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}@keyframes slide-in{from{opacity:0;transform:translateX(12px)}to{opacity:1;transform:translateX(0)}}@keyframes scale-in{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.35}}@keyframes pulse-banner{0%,100%{box-shadow:0 0 0 0 rgba(8,145,178,.24)}50%{box-shadow:0 0 0 6px rgba(8,145,178,0)}}.app-shell{animation:fade-up 250ms var(--ease-out) both}
        ::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:var(--bg-inset)}::-webkit-scrollbar-thumb{background:var(--border-medium);border-radius:999px}::-webkit-scrollbar-thumb:hover{background:var(--accent-cyan)}
        @media(max-width:900px){.block-container{padding-top:4.25rem;padding-left:1rem;padding-right:1rem}.kpi-strip,.metric-strip{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){.app-header{padding:1.2rem;border-radius:var(--radius-lg)}.status-chip{width:100%}.kpi-strip,.metric-strip{grid-template-columns:1fr}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clean(value: object) -> str:
    return html.escape(str(value), quote=True)


def _iconify_url(icon: str) -> str:
    if ":" not in icon:
        return ""
    prefix, name = icon.split(":", 1)
    return f"https://api.iconify.design/{quote(prefix)}/{quote(name)}.svg"


def _sidebar_icon(icon: str, class_name: str = "aifds-sidebar-icon-mask") -> str:
    icon_url = _iconify_url(icon)
    if not icon_url:
        return ""
    return f'<span class="{_clean(class_name)}" style="--sidebar-icon:url({_clean(icon_url)})"></span>'


def render_sidebar_brand() -> None:
    st.html(
        '<div class="aifds-sidebar-brand">'
        '<div class="aifds-sidebar-logo">'
        f'{_sidebar_icon("solar:shield-star-bold-duotone")}'
        '</div>'
        '<div class="aifds-sidebar-brand-copy">'
        '<div class="aifds-sidebar-brand-title">AI-FDS</div>'
        '<div class="aifds-sidebar-brand-subtitle">AI Fraud Detection System</div>'
        '</div>'
        '</div>'
    )


def render_sidebar_navigation(
    *,
    active_page: str,
    on_select: Callable[[str], None],
) -> None:
    navigation_items = [
        {
            "page": "Detection Center",
            "title": "Detection Center",
            "description": "Analyze emails, voice & phone",
            "icon": "solar:shield-bold-duotone",
            "tone": "blue",
        },
        {
            "page": "AI Report Generator",
            "title": "AI Report Generator",
            "description": "Generate professional reports",
            "icon": "solar:document-text-bold-duotone",
            "tone": "green",
        },
    ]

    with st.container(key="sidebar_navigation"):
        for item in navigation_items:
            page_name = str(item["page"])
            is_active = active_page == page_name

            with st.container(
                key=f"sidebar_nav_card_{page_name.lower().replace(' ', '_')}",
            ):
                st.html(
                    f'<div class="aifds-sidebar-nav-visual {"active" if is_active else ""} {_clean(item["tone"])}">'
                    '<div class="aifds-sidebar-nav-icon">'
                    f'{_sidebar_icon(str(item["icon"]))}'
                    '</div>'
                    '<div class="aifds-sidebar-nav-copy">'
                    f'<div class="aifds-sidebar-nav-title">{_clean(item["title"])}</div>'
                    f'<div class="aifds-sidebar-nav-description">{_clean(item["description"])}</div>'
                    '</div>'
                    f'{_sidebar_icon("solar:alt-arrow-right-linear", "aifds-sidebar-icon-mask aifds-sidebar-nav-arrow")}'
                    '</div>'
                )

                if st.button(
                    f"Open {page_name}",
                    key=f"sidebar_nav_button_{page_name}",
                    use_container_width=True,
                    type="secondary",
                ):
                    on_select(page_name)
                    st.rerun()


def render_detection_tool_intro(
    title: str,
    description: str,
    icon: str,
    accent: str = "blue",
) -> None:
    icon_url = _iconify_url(icon)
    icon_html = (
        f'<span class="detection-tool-mask-icon" style="--tool-icon:url({_clean(icon_url)})"></span>'
        if icon_url
        else ""
    )
    st.html(
        f'<div class="detection-tool-intro {_clean(accent)}">'
        f'<div class="detection-tool-icon">{icon_html}</div>'
        '<div class="detection-tool-copy">'
        f'<h2>{_clean(title)}</h2>'
        f'<p>{_clean(description)}</p>'
        '</div>'
        '</div>'
    )


def render_detection_center_header() -> None:
    icon_url = _iconify_url("ph:binoculars-duotone")
    st.html(
        '<div class="detection-center-hero">'
        '<div class="detection-center-hero-icon">'
        f'<span class="hero-svg-icon" style="--hero-icon:url({_clean(icon_url)})"></span>'
        '</div>'
        '<div class="detection-center-hero-copy">'
        '<h1>Detection Center</h1>'
        '<p>Choose a detection tool below to analyze suspicious content across emails, voice transcripts, '
        'and phone numbers. Our AI models will highlight risks and key evidence to help you stay safe.</p>'
        '</div>'
        '</div>'
    )


def render_detection_tab_selector(active_tab: str) -> str:
    tabs = [
        {
            "key": "email",
            "title": "Emails and Messages",
            "icon": "solar:letter-bold-duotone",
        },
        {
            "key": "transcript",
            "title": "Voice Transcript",
            "icon": "solar:microphone-3-bold-duotone",
        },
        {
            "key": "phone",
            "title": "Phone Number",
            "icon": "solar:phone-calling-rounded-bold-duotone",
        },
    ]

    if active_tab not in {str(tab["key"]) for tab in tabs}:
        active_tab = "email"
        st.session_state.active_detection_tab = active_tab

    with st.container(key="detection_tab_wrapper"):
        cols = st.columns(3, gap="small")

        for col, tab in zip(cols, tabs):
            tab_key = str(tab["key"])
            tab_title = str(tab["title"])
            icon_url = _iconify_url(str(tab["icon"]))
            is_active = active_tab == tab_key

            with col:
                with st.container(key=f"detection_tab_card_{tab_key}"):
                    st.html(
                        f'<div class="detection-tab-visual {"active" if is_active else ""}">'
                        f'<span class="detection-tab-icon" style="--tab-icon:url({_clean(icon_url)})"></span>'
                        f'<span>{_clean(tab_title)}</span>'
                        '</div>'
                    )

                    if st.button(
                        tab_title,
                        key=f"detection_tab_button_{tab_key}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        st.session_state.active_detection_tab = tab_key
                        st.rerun()

    return str(st.session_state.active_detection_tab)


def render_global_header(root: Path, active_page: str) -> None:
    st.html(
        '<div class="hero-banner">'
        '<div class="hero-content">'
        '<h1>AI-based Spam and Caller<br>Fraud Detection System</h1>'
        '<p>Investigate suspicious messages, phone calls, and caller identities using '
        'multiple AI models with explainable results and evidence-based reporting.</p>'
        "</div>"
        '<div class="hero-visual" aria-hidden="true">'
        '<div class="hero-orbit orbit-one"></div>'
        '<div class="hero-orbit orbit-two"></div>'
        '<div class="floating-icon icon-chat"><span class="hero-svg-icon" style="--hero-icon:url(https://api.iconify.design/solar/chat-round-dots-bold-duotone.svg)"></span></div>'
        '<div class="floating-icon icon-mic"><span class="hero-svg-icon" style="--hero-icon:url(https://api.iconify.design/solar/microphone-3-bold-duotone.svg)"></span></div>'
        '<div class="floating-icon icon-phone"><span class="hero-svg-icon" style="--hero-icon:url(https://api.iconify.design/solar/phone-calling-rounded-bold-duotone.svg)"></span></div>'
        '<div class="floating-icon icon-audio"><span class="hero-svg-icon" style="--hero-icon:url(https://api.iconify.design/solar/soundwave-bold-duotone.svg)"></span></div>'
        '<div class="shield-main"><span class="hero-svg-icon" style="--hero-icon:url(https://api.iconify.design/solar/shield-check-bold-duotone.svg)"></span></div>'
        "</div>"
        "</div>"
    )


def render_kpi_row(history: list[dict[str, object]]) -> None:
    """
    Compact dashboard KPI cards.
    """

    real_history = [item for item in history if not bool(item.get("is_demo"))]
    total_files = len(real_history)

    threat_keywords = (
        "suspicious",
        "scam",
        "fraud",
        "phishing",
        "high risk",
    )

    threats_detected = sum(
        1
        for item in real_history
        if any(
            keyword in str(item.get("prediction", "")).lower()
            for keyword in threat_keywords
        )
    )

    confidence_values = [
        float(item.get("confidence", 0))
        for item in real_history
        if isinstance(item.get("confidence"), (int, float))
    ]

    average_risk = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0
    )

    st.html(
        '<div class="compact-kpi-row">'
        '<div class="compact-kpi-card blue">'
        '<div class="kpi-icon-wrapper blue"><div class="kpi-orbit"></div>'
        '<span class="kpi-mask-icon" style="--kpi-icon:url(https://api.iconify.design/solar/document-text-bold-duotone.svg)"></span></div>'
        '<div class="kpi-title">Files Analysed</div>'
        '<div class="kpi-divider"></div>'
        f'<div class="kpi-value blue">{total_files}</div>'
        '</div>'
        '<div class="compact-kpi-card red">'
        '<div class="kpi-icon-wrapper red"><div class="kpi-orbit"></div>'
        '<span class="kpi-mask-icon" style="--kpi-icon:url(https://api.iconify.design/solar/danger-triangle-bold-duotone.svg)"></span></div>'
        '<div class="kpi-title">Threats Detected</div>'
        '<div class="kpi-divider"></div>'
        f'<div class="kpi-value red">{threats_detected}</div>'
        '</div>'
        '<div class="compact-kpi-card orange">'
        '<div class="kpi-icon-wrapper orange"><div class="kpi-orbit"></div>'
        '<span class="kpi-mask-icon" style="--kpi-icon:url(https://api.iconify.design/solar/shield-warning-bold-duotone.svg)"></span></div>'
        '<div class="kpi-title">Average Risk</div>'
        '<div class="kpi-divider"></div>'
        f'<div class="kpi-value orange">{average_risk:.0f}%</div>'
        '</div>'
        '</div>'
    )


def render_section_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = f'<div class="eyebrow">{_clean(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f"<p>{_clean(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="section-header app-shell">{eyebrow_html}<h2>{_clean(title)}</h2>{subtitle_html}</div>', unsafe_allow_html=True)


def render_feature_card(title: str, body: str, index: str = "01") -> None:
    st.markdown(f'<div class="feature-card app-shell"><div class="feature-index">{_clean(index)}</div><h3>{_clean(title)}</h3><p>{_clean(body)}</p></div>', unsafe_allow_html=True)


def render_metric_row(metrics: list[dict[str, object]]) -> None:
    tiles = []
    for metric in metrics:
        color = _clean(metric.get("color", "var(--text-primary)"))
        tiles.append(f'<div class="metric-tile app-shell"><div class="metric-value" style="color:{color}">{_clean(metric.get("value", "-"))}</div><div class="metric-label">{_clean(metric.get("label", ""))}</div></div>')
    st.markdown(f'<div class="metric-strip" style="--metric-count:{max(1, len(metrics))}">{"".join(tiles)}</div>', unsafe_allow_html=True)


def render_info_banner(body: str, kind: Literal["info", "warning", "danger", "success"] = "info", code: str = "INFO") -> None:
    st.markdown(f'<div class="banner {kind} app-shell"><span class="banner-tag">{_clean(code)}</span><div>{_clean(body)}</div></div>', unsafe_allow_html=True)


def render_content_card_open(accent: Literal["violet", "red", "green"] = "violet") -> None:
    st.markdown(f'<div class="content-card accent-{_clean(accent)}">', unsafe_allow_html=True)


def render_content_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_section_divider(title: str) -> None:
    st.markdown(
        f'<div class="section-divider"><span>{_clean(title)}</span></div>',
        unsafe_allow_html=True,
    )


def render_soft_panel(body: str) -> None:
    st.markdown(
        f'<div class="soft-panel"><p>{_clean(body)}</p></div>',
        unsafe_allow_html=True,
    )


def render_soft_panel_html(body_html: str) -> None:
    st.markdown(
        f'<div class="soft-panel">{body_html}</div>',
        unsafe_allow_html=True,
    )


def render_accent_strip(title: str, body: str = "") -> None:
    body_html = f"<p>{_clean(body)}</p>" if body else ""
    st.markdown(
        f'<div class="accent-strip"><h3>{_clean(title)}</h3>{body_html}</div>',
        unsafe_allow_html=True,
    )


def render_analysis_ready(message: str = "New analysis ready - scroll to results") -> None:
    try:
        st.toast(message)
    except Exception:
        pass
    st.markdown(f'<div class="scroll-notify app-shell"><span class="scroll-notify-dot"></span>{_clean(message)}</div>', unsafe_allow_html=True)


def render_result_card(title: str, risk_score: float, summary: str) -> None:
    if risk_score >= 70:
        level, label = "high", "High Risk"
    elif risk_score >= 40:
        level, label = "medium", "Suspicious"
    else:
        level, label = "safe", "Lower Risk"
    st.markdown(f'<div class="result-card {level}"><span class="risk-badge {level}">{label} - {risk_score:.1f}%</span><h3>{_clean(title)}</h3><p>{_clean(summary)}</p></div>', unsafe_allow_html=True)


def render_chart_placeholder(label: str = "Chart will appear here") -> None:
    st.markdown(f'<div class="chart-placeholder"><div class="chart-spinner"></div><span>{_clean(label)}</span></div>', unsafe_allow_html=True)


def apply_chart_theme(fig):
    fig.update_layout(**CHART_THEME)
    return fig


def render_demo_notice(root: Path) -> None:
    if official_data_present(str(root)):
        return
    st.markdown('<div class="demo-warning app-shell"><span class="banner-tag">Demo</span><div><strong>Temporary demonstration data is active.</strong> Replace synthetic examples with the official datasets before reporting final model results.</div></div>', unsafe_allow_html=True)


def render_sidebar_status(root: Path) -> None:
    groups = _sidebar_model_status_items(root)
    rows = [_sidebar_status_group_html(group) for group in groups]

    st.html(
        '<div class="aifds-sidebar-status-panel">'
        '<div class="aifds-sidebar-status-heading">'
        '<span class="aifds-sidebar-status-live-dot"></span>'
        '<span>AI MODEL STATUS</span>'
        '</div>'
        '<div class="aifds-sidebar-status-divider"></div>'
        f'<div class="aifds-sidebar-model-list">{"".join(rows)}</div>'
        '</div>'
    )


def _sidebar_model_status_items(
    root: Path,
) -> list[dict[str, object]]:
    models_dir = root / "models"
    email_models = [
        ("Email Naive Bayes", models_dir / "email_nb.pkl", [models_dir / "email_vectorizer.pkl"]),
        ("Email Decision Tree", models_dir / "email_dt.pkl", [models_dir / "email_vectorizer.pkl"]),
        ("Email SVM", models_dir / "email_svm.pkl", [models_dir / "email_vectorizer.pkl"]),
        ("Email Random Forest", models_dir / "email_rf.pkl", [models_dir / "email_vectorizer.pkl"]),
        ("Email XGBoost", models_dir / "email_xgb.pkl", [models_dir / "email_vectorizer.pkl"]),
    ]
    transcript_models = [
        ("Transcript Naive Bayes", models_dir / "transcript_nb.pkl", [models_dir / "transcript_vectorizer.pkl"]),
        ("Transcript Decision Tree", models_dir / "transcript_dt.pkl", [models_dir / "transcript_vectorizer.pkl"]),
        ("Transcript SVM", models_dir / "transcript_svm.pkl", [models_dir / "transcript_vectorizer.pkl"]),
        ("Transcript Random Forest", models_dir / "transcript_rf.pkl", [models_dir / "transcript_vectorizer.pkl"]),
        ("Transcript XGBoost", models_dir / "transcript_xgb.pkl", [models_dir / "transcript_vectorizer.pkl"]),
    ]
    audio_models = [
        ("Audio SVM", models_dir / "audio_svm.pkl", []),
        ("Audio Behavioral RF", models_dir / "audio_behavior_rf.pkl", []),
    ]

    email_details = [_artifact_status_item(*item) for item in email_models]
    transcript_details = [_artifact_status_item(*item) for item in transcript_models]
    audio_details = [_artifact_status_item(*item) for item in audio_models]
    api_details = [
        _phone_api_status_item("Omkar Carrier Lookup API", "omkar"),
        _phone_api_status_item("PenipuMY", "penipumy"),
    ]

    return [
        _group_status_item(
            name="EMAIL",
            icon="solar:letter-bold-duotone",
            tone="blue",
            details=email_details,
        ),
        _group_status_item(
            name="VOICE TRANSCRIPT",
            icon="solar:document-text-bold-duotone",
            tone="purple",
            details=transcript_details,
        ),
        _group_status_item(
            name="VOICE AUDIO",
            icon="solar:soundwave-bold-duotone",
            tone="pink",
            details=audio_details,
        ),
        _group_status_item(
            name="PHONE LOOKUP API",
            icon="solar:phone-calling-rounded-bold-duotone",
            tone="orange",
            details=api_details,
        ),
    ]


def _artifact_status_item(
    name: str,
    artifact_path: Path,
    dependency_paths: list[Path],
) -> dict[str, str]:
    artifact_exists = artifact_path.exists()
    dependencies_ready = all(path.exists() for path in dependency_paths)
    if artifact_exists and dependencies_ready:
        status = "active"
    elif artifact_exists or any(path.exists() for path in dependency_paths):
        status = "pending"
    else:
        status = "inactive"

    return {
        "name": name,
        "status": status,
        "date_label": "Last trained",
        "date": _latest_path_date([artifact_path]),
    }


def _group_status_item(
    name: str,
    icon: str,
    tone: str,
    details: list[dict[str, str]],
) -> dict[str, object]:
    statuses = [item["status"] for item in details]
    if statuses and all(status == "active" for status in statuses):
        status = "active"
    elif any(status == "active" for status in statuses) or any(status == "pending" for status in statuses):
        status = "pending"
    else:
        status = "inactive"

    return {
        "name": name,
        "status": status,
        "icon": icon,
        "tone": tone,
        "details": details,
    }


def _phone_api_status_item(name: str, provider_id: str) -> dict[str, str]:
    enabled_key = f"phone_{provider_id}_enabled"
    enabled_widget_key = f"_{enabled_key}_toggle"
    enabled = bool(st.session_state.get(enabled_key, st.session_state.get(enabled_widget_key, True)))
    configured = bool(_configured_phone_provider_key(provider_id))
    if not enabled:
        status = "inactive"
    elif configured:
        status = "active"
    else:
        status = "pending"

    return {
        "name": name,
        "status": status,
        "date_label": "Last tested",
        "date": "Not tested",
    }


def _configured_phone_provider_key(provider_id: str) -> str:
    if provider_id == "penipumy":
        session_key = "phone_penipumy_api_key"
        env_keys = ("PENIPUMY_API_KEY", "PENIPU_API_KEY")
        secret_sections = ("penipumy", "penipu")
    else:
        session_key = "phone_omkar_api_key"
        env_keys = ("OMKAR_API_KEY", "OMKAR_CARRIER_API_KEY")
        secret_sections = ("omkar", "carrier_lookup")

    session_value = str(st.session_state.get(session_key, "") or "").strip()
    if not session_value:
        session_value = str(st.session_state.get(f"_{session_key}_input", "") or "").strip()
    if session_value:
        return session_value

    for key in env_keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
        secret_value = _streamlit_secret(key)
        if secret_value:
            return secret_value

    for section_name in secret_sections:
        secret_value = _streamlit_secret(section_name, "api_key")
        if secret_value:
            return secret_value

    return ""


def _streamlit_secret(*keys: str) -> str:
    try:
        value: object = st.secrets
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else value.get(key, {})
        return str(value or "").strip() if not isinstance(value, dict) else ""
    except Exception:
        return ""


def _latest_path_date(paths: list[Path]) -> str:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return "Not trained"
    latest = max(path.stat().st_mtime for path in existing_paths)
    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d")


def _sidebar_status_group_html(group: dict[str, object]) -> str:
    status = str(group["status"])
    details = list(group.get("details", []))
    return (
        '<details class="aifds-sidebar-model-group" name="aifds-sidebar-model-status">'
        '<summary class="aifds-sidebar-model-summary">'
        f'<div class="aifds-sidebar-model-icon {_clean(group["tone"])}">'
        f'{_sidebar_icon(str(group["icon"]))}'
        '</div>'
        f'<div class="aifds-sidebar-model-name">{_clean(group["name"])}</div>'
        f'{_sidebar_status_pill_html(status)}'
        '</summary>'
        '<div class="aifds-sidebar-model-detail-list">'
        + "".join(_sidebar_detail_row_html(item) for item in details)
        + "</div>"
        "</details>"
    )


def _sidebar_detail_row_html(item: dict[str, str]) -> str:
    status = item["status"]
    return (
        '<div class="aifds-sidebar-detail-row">'
        '<div class="aifds-sidebar-detail-copy">'
        f'<div class="aifds-sidebar-detail-name">{_clean(item["name"])}</div>'
        f'<div class="aifds-sidebar-detail-date">{_clean(item["date_label"])}: {_clean(item["date"])}</div>'
        '</div>'
        f'{_sidebar_status_pill_html(status)}'
        '</div>'
    )


def _sidebar_status_pill_html(status: str) -> str:
    return (
        f'<div class="aifds-sidebar-status-pill {_clean(status)}">'
        '<span></span>'
        f'{_clean(status)}'
        '</div>'
    )


def clear_all_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
