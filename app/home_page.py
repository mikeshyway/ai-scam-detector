"""Home page for the capstone app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui_components import get_demo_data, render_demo_notice


def render_home_page(root: Path, history: list[dict[str, object]]) -> None:
    render_demo_notice(root)
    st.subheader("Project Focus")
    st.write(
        "This system teaches students how AI-assisted scams can be detected across text, "
        "meeting transcripts, and uploaded voice recordings. The main goal is awareness "
        "and explainability, not replacing commercial caller-ID or cybersecurity products."
    )

    cols = st.columns(3)
    cards = [
        ("📧 Email phishing", "Paste or upload messages and inspect scam language, confidence scores, and highlighted red flags."),
        ("📞 Caller fraud transcripts", "Analyze call, Zoom, Teams, or Google Meet transcripts for urgency, payment, secrecy, and identity cues."),
        ("🎙️ AI speech detection", "Upload .wav or .flac voice recordings to view waveform, spectrogram, MFCC features, and SVM output."),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="feature-card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)

    st.subheader("Current Demo Dataset")
    demo = get_demo_data()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Synthetic emails", len(demo["emails"]))
    metric_cols[1].metric("Synthetic transcripts", len(demo["transcripts"]))
    metric_cols[2].metric("Synthetic phone records", len(demo["phones"]))
    metric_cols[3].metric("Quiz questions", len(demo["quiz"]))

    st.info(
        "When the official datasets are added and models are trained, keep the app pages but remove "
        "the temporary synthetic demo-data dependency from demonstrations and screenshots."
    )

    if history:
        st.subheader("Latest Session Result")
        st.dataframe(history[:3], hide_index=True, use_container_width=True)
