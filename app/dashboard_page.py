"""Dashboard page that summarizes demo data, model artifacts, and session activity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_components import get_demo_data, get_model_status, official_data_present, render_demo_notice


@st.cache_data(show_spinner=False)
def _demo_summary() -> pd.DataFrame:
    demo = get_demo_data()
    emails = demo["emails"].copy()
    transcripts = demo["transcripts"].copy()
    phones = demo["phones"].copy()
    return pd.DataFrame(
        [
            {"Dataset": "Synthetic emails", "Rows": len(emails), "Suspicious examples": int((emails["label"] == "Suspicious").sum())},
            {"Dataset": "Synthetic transcripts", "Rows": len(transcripts), "Suspicious examples": int((transcripts["label"] == "Scam").sum())},
            {"Dataset": "Synthetic phone reputation", "Rows": len(phones), "Suspicious examples": int((phones["risk_score"] >= 60).sum())},
        ]
    )


def render_dashboard_page(root: Path, history: list[dict[str, object]]) -> None:
    render_demo_notice(root)
    st.subheader("System Dashboard")

    status = get_model_status(str(root))
    summary = _demo_summary()
    cols = st.columns(4)
    if history:
        hist = pd.DataFrame(history)
        scam_count = int(hist["prediction"].astype(str).str.contains("Suspicious|High risk|AI-generated|Completed|Rolling", case=False, regex=True).sum())
        legit_count = max(0, len(hist) - scam_count)
        avg_conf = float(pd.to_numeric(hist["confidence"], errors="coerce").fillna(0).mean())
    else:
        scam_count = 0
        legit_count = 0
        avg_conf = 0.0
    cols[0].metric("Total scans performed", len(history))
    cols[1].metric("Scam detections", scam_count)
    cols[2].metric("Legitimate detections", legit_count)
    cols[3].metric("Avg confidence", f"{avg_conf:.1f}%")

    chart_col, table_col = st.columns([0.56, 0.44])
    with chart_col:
        fig = px.bar(
            summary,
            x="Dataset",
            y=["Rows", "Suspicious examples"],
            barmode="group",
            title="Temporary demo dataset coverage",
        )
        st.plotly_chart(fig, use_container_width=True)
    with table_col:
        st.dataframe(summary, hide_index=True, use_container_width=True)

    model_df = pd.DataFrame(
        [{"Artifact": name, "Status": "Loaded" if exists else "Missing"} for name, exists in status.items()]
    )
    fig = px.pie(model_df, names="Status", title="Model artifact readiness", hole=0.45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Session Activity")
    if history:
        hist = pd.DataFrame(history)
        st.dataframe(hist, hide_index=True, use_container_width=True)
        fig = px.histogram(hist, x="confidence", color="type", title="Session confidence distribution")
        st.plotly_chart(fig, use_container_width=True)
        trend = hist.iloc[::-1].reset_index(drop=True)
        trend["scan_number"] = trend.index + 1
        fig = px.line(trend, x="scan_number", y="confidence", color="type", markers=True, title="Detection trend analytics")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detections have been run in this browser session yet.")
