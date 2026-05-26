"""Explainability and source-code page."""

from __future__ import annotations

import os
from pathlib import Path

import plotly.express as px
import streamlit as st

from app.ui_components import get_demo_data, get_model_status, source_code_url


def render_explainability_page(root: Path, history: list[dict[str, object]]) -> None:
    st.subheader("Transparency & Explainability Hub")
    st.write(
        "This page explains the backend architecture, chosen models, training flow, and why the system "
        "stays focused on educational scam awareness."
    )

    st.markdown(
        """
        1. Text is cleaned by replacing URLs, emails, phone numbers, and money amounts with special tokens.
        2. TF-IDF converts email or transcript text into numeric word and phrase features.
        3. Naive Bayes and Decision Tree models classify text as legitimate or suspicious.
        4. Audio files are converted into MFCC features that summarize speech frequency patterns.
        5. The SVM audio model classifies uploaded speech as real human speech or possible AI-generated speech.
        6. The interface explains results with confidence scores, warning phrase highlights, charts, and session history.
        """
    )

    st.subheader("Pipeline")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          TextInput [label="Email / Transcript"];
          AudioInput [label="Uploaded Audio"];
          CleanText [label="Text Cleaning"];
          TFIDF [label="TF-IDF"];
          NB [label="Naive Bayes"];
          DT [label="Decision Tree"];
          MFCC [label="MFCC Extraction"];
          SVM [label="SVM"];
          UI [label="Streamlit Explanations"];

          TextInput -> CleanText -> TFIDF -> NB -> UI;
          TFIDF -> DT -> UI;
          AudioInput -> MFCC -> SVM -> UI;
        }
        """
    )

    st.subheader("Model Comparison")
    model_table = get_demo_data()["models"]
    st.dataframe(model_table, hide_index=True, use_container_width=True)
    status = get_model_status(str(root))
    status_rows = [{"Artifact": name, "Ready": "Yes" if exists else "No"} for name, exists in status.items()]
    fig = px.bar(status_rows, x="Artifact", color="Ready", title="Model artifact readiness")
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Development Timeline")
    timeline = [
        {"Phase": "Requirements", "Order": 1},
        {"Phase": "Data collection", "Order": 2},
        {"Phase": "Preprocessing", "Order": 3},
        {"Phase": "EDA", "Order": 4},
        {"Phase": "Model training", "Order": 5},
        {"Phase": "Simulation + UI", "Order": 6},
        {"Phase": "Evaluation", "Order": 7},
        {"Phase": "Deployment", "Order": 8},
    ]
    st.plotly_chart(px.line(timeline, x="Order", y="Phase", markers=True, title="8-phase methodology map"), use_container_width=True)

    st.subheader("Open Source Code")
    repo_url = os.environ.get("PROJECT_REPO_URL", "").strip()
    if repo_url:
        st.link_button("Open GitHub repository", repo_url)
    else:
        st.warning(
            "No GitHub URL is configured yet. After publishing this folder to GitHub, set "
            "PROJECT_REPO_URL to show a clickable open-source link here."
        )
        st.link_button("Open local source folder", source_code_url(str(root)))
    st.code(str(root), language="text")
    st.caption(f"Local source reference: {source_code_url(str(root))}")

    st.subheader("Important Limitation")
    st.write(
        "The app explains suspicious patterns, but it cannot guarantee that a message, caller, "
        "or recording is safe. It is designed for student awareness and capstone demonstration."
    )
