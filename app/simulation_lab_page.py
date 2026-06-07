"""Uploaded-recording scam simulation lab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_components import render_info_banner, render_section_header
from src.explainability import find_suspicious_phrases, highlighted_html
from src.recording_audio_simulation import (
    analyze_audio_chunks,
    load_optional_audio_model,
    spectrogram_db,
)
from src.time_utils import formatted_now


@st.cache_data(show_spinner=False, max_entries=3)
def _run_chunk_analysis(audio_bytes: bytes, suffix: str, chunk_seconds: int, root: str):
    model = load_optional_audio_model(root)
    return analyze_audio_chunks(
        audio_bytes,
        suffix,
        chunk_seconds=chunk_seconds,
        model=model,
    )


def _waveform_figure(y: np.ndarray, sr: int) -> go.Figure:
    times = np.arange(len(y)) / sr
    fig = go.Figure(
        go.Scatter(
            x=times,
            y=y,
            mode="lines",
            line=dict(color="#3d8ee8", width=1),
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
    )
    return fig


def _spectrogram_figure(y: np.ndarray, sr: int) -> go.Figure:
    db, times, freqs = spectrogram_db(y, sr)
    fig = go.Figure(
        data=go.Heatmap(
            x=times,
            y=freqs,
            z=db,
            colorscale="Magma",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
    )
    return fig


def _read_transcript(uploaded_transcript) -> str:
    if uploaded_transcript is None:
        return ""
    suffix = Path(uploaded_transcript.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_transcript.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".csv":
        return pd.read_csv(uploaded_transcript).astype(str).head(20).to_string(index=False)
    return ""


def render_simulation_lab_page(root: Path, history: list[dict[str, object]]) -> None:
    render_section_header(
        "Uploaded call and meeting recording analysis",
        "Simulate how suspicious voice evidence can be reviewed chunk by chunk after a call or online meeting.",
        "Scam simulation lab",
    )
    render_info_banner(
        "This page analyzes uploaded recordings only. It does not intercept live calls, phone audio, Zoom, Teams, or Google Meet streams.",
        kind="info",
        code="SCOPE",
    )

    upload_col, settings_col = st.columns([0.62, 0.38])
    with upload_col:
        uploaded_audio = st.file_uploader(
            "Upload exported meeting or call recording audio",
            type=["wav", "flac", "mp3", "m4a"],
            key="sim_recording_audio",
        )
        uploaded_transcript = st.file_uploader(
            "Optional transcript file",
            type=["txt", "csv"],
            key="sim_recording_transcript",
        )
    with settings_col:
        chunk_seconds = st.slider("Chunk size", min_value=5, max_value=10, value=5, step=1)
        render_info_banner(
            "The model analyzes short audio chunks and plots confidence over the uploaded timeline.",
            kind="success",
            code="FLOW",
        )

    transcript_text = _read_transcript(uploaded_transcript)
    transcript_text = st.text_area(
        "Paste or edit transcript text",
        value=transcript_text,
        height=140,
        placeholder="Optional: paste a call or meeting transcript to highlight suspicious language.",
    )

    if uploaded_audio is None:
        st.info("Upload an audio recording to run chunk-by-chunk analysis.")
        return

    audio_bytes = uploaded_audio.getvalue()
    suffix = Path(uploaded_audio.name).suffix.lower()
    st.audio(audio_bytes)

    if not st.button("Run rolling chunk analysis", type="primary", use_container_width=True):
        return

    try:
        results, y, sr = _run_chunk_analysis(audio_bytes, suffix, chunk_seconds, str(root))
    except Exception as exc:
        st.error(f"Chunk analysis failed: {exc}")
        st.caption("For MP3/M4A files, the host may require ffmpeg support.")
        return

    if results.empty:
        st.warning("No usable chunks were extracted from the recording.")
        return

    render_section_header("Rolling detection dashboard", eyebrow="Chunk analysis")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Chunks", len(results))
    metric_cols[1].metric("Peak confidence", f"{results['confidence'].max():.1f}%")
    metric_cols[2].metric("Average confidence", f"{results['confidence'].mean():.1f}%")
    metric_cols[3].metric("Engine", str(results["engine"].iloc[0]))

    line = go.Figure(
        go.Scatter(
            x=results["end_sec"],
            y=results["confidence"],
            mode="lines+markers",
            line=dict(color="#ef5b5b", width=2),
        )
    )
    line.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Call time (seconds)",
        yaxis_title="Risk confidence (%)",
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(line, use_container_width=True)
    st.dataframe(results, hide_index=True, use_container_width=True)

    render_section_header("Waveform", eyebrow="Audio shape")
    st.plotly_chart(_waveform_figure(y, sr), use_container_width=True)

    render_section_header("Spectrogram", eyebrow="Frequency view")
    st.plotly_chart(_spectrogram_figure(y, sr), use_container_width=True)

    if transcript_text.strip():
        findings = find_suspicious_phrases(transcript_text)
        render_section_header("Transcript warning indicators", eyebrow="Text context")
        st.markdown(highlighted_html(transcript_text, findings), unsafe_allow_html=True)
        if findings:
            st.dataframe(
                pd.DataFrame(findings)[["phrase", "category", "reason"]],
                hide_index=True,
                use_container_width=True,
            )

    history.insert(
        0,
        {
            "time": formatted_now(),
            "type": "Simulation",
            "prediction": "Uploaded recording chunk analysis",
            "confidence": round(float(results["confidence"].max()), 2),
            "model": str(results["engine"].iloc[0]),
            "preview": uploaded_audio.name,
        },
    )
