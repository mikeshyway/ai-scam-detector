"""Session-only detection history tab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_history_tab(history: list[dict[str, object]]) -> None:
    st.header("Session Detection History")

    if not history:
        st.info("No detections have been run in this session yet.")
        return

    df = pd.DataFrame(history)
    top_controls = st.columns([0.75, 0.25])
    with top_controls[1]:
        if st.button("Clear history", use_container_width=True):
            history.clear()
            st.rerun()

    st.dataframe(df, hide_index=True, use_container_width=True)

    chart_col, type_col = st.columns(2)
    with chart_col:
        fig = px.histogram(
            df,
            x="confidence",
            nbins=10,
            title="Confidence distribution",
            labels={"confidence": "Confidence (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with type_col:
        counts = df.groupby(["type", "prediction"]).size().reset_index(name="count")
        fig = px.bar(
            counts,
            x="type",
            y="count",
            color="prediction",
            title="Detections by input type",
        )
        st.plotly_chart(fig, use_container_width=True)
