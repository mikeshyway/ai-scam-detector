"""Home page for the capstone app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui_components import get_demo_data, render_demo_notice


def _stat_card(value: str, label: str, source: str) -> str:
    return (
        '<div class="feature-card">'
        f'<h3>{value}</h3><p>{label}</p>'
        f'<span class="scenario-meta">{source}</span>'
        '</div>'
    )


def render_home_page(root: Path, history: list[dict[str, object]]) -> None:
    render_demo_notice(root)
    st.markdown(
        """
        <div class="hero-grid">
          <div>
            <h2>Train students to recognise, investigate, and resist scam pressure.</h2>
            <p>
              This capstone combines simulation, explainable AI detection, and student quizzes
              so the tool feels like a learning environment rather than a plain file checker.
            </p>
          </div>
          <div class="cyber-avatar">
            <div class="avatar-orbit"></div>
            <div class="avatar-shield">AI</div>
            <div class="avatar-caption">Scam Defense Coach</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Global Scam Context")
    stat_cols = st.columns(3)
    stat_cards = [
        ("$12.5B+", "Reported US consumer fraud losses in 2024.", "FTC 2024"),
        ("859,532", "Internet crime complaints received in 2024.", "FBI IC3 2024"),
        ("$16.6B+", "Reported losses linked to IC3 complaints in 2024.", "FBI IC3 2024"),
    ]
    for col, (value, label, source) in zip(stat_cols, stat_cards):
        with col:
            st.markdown(_stat_card(value, label, source), unsafe_allow_html=True)

    st.subheader("Project Focus")
    st.write(
        "This system teaches students how AI-assisted scams work by combining a turn-based "
        "simulation lab with explainable detection for text, transcripts, phone-risk examples, "
        "and uploaded voice recordings."
    )

    cols = st.columns(4)
    cards = [
        ("🎮 Simulation lab", "Experience scam pressure, make a decision before the timer ends, then investigate and defend."),
        ("📧 Email phishing", "Paste or upload messages and inspect scam language, confidence scores, and highlighted red flags."),
        ("📞 Caller fraud transcripts", "Analyze call, Zoom, Teams, or Google Meet transcripts for urgency, payment, secrecy, and identity cues."),
        ("🎙️ AI speech detection", "Upload .wav or .flac voice recordings to view waveform, spectrogram, MFCC features, and SVM output."),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="feature-card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)

    st.subheader("How It Works")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", fillcolor="#0f172a", fontcolor="white", color="#38bdf8"];
          Phone [label="Phone / Student Device\\nBrowser opens LAN/ngrok URL"];
          Streamlit [label="Laptop / Streamlit Host\\napp/main.py"];
          Simulation [label="Simulation Lab\\nUpload call recording"];
          Chunks [label="5-10s Audio Chunks\\nMFCC extraction"];
          Models [label="AI Models / Demo Logic\\nSVM + Text Classifiers"];
          Explain [label="Dashboard + Explanation\\nConfidence, motive, defense"];
          Phone -> Streamlit -> Simulation -> Chunks -> Models -> Explain;
        }
        """
    )

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
