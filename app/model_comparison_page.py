"""Model comparison page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_components import get_demo_data, get_model_status


def render_model_comparison_page(root: Path, history: list[dict[str, object]]) -> None:
    st.subheader("AI Model Comparison")
    st.write(
        "The project stays with lightweight, explainable machine learning models so it remains "
        "capstone-safe and runnable on a local Kali Linux environment."
    )

    model_table = get_demo_data()["models"]
    st.dataframe(model_table, hide_index=True, use_container_width=True)

    status = get_model_status(str(root))
    artifact_df = pd.DataFrame(
        [{"Artifact": name, "Ready": "Yes" if exists else "No"} for name, exists in status.items()]
    )
    fig = px.bar(artifact_df, x="Artifact", color="Ready", title="Saved model artifacts")
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why not deep learning here?")
    st.info(
        "Deep learning could be future work, but the current objective is an educational prototype "
        "with transparent models, faster training, and lower hardware requirements."
    )
