"""Audio deepfake detection and uploaded-recording simulation tab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_components import (
    apply_chart_theme,
    render_analysis_ready,
    render_content_card_close,
    render_content_card_open,
    render_demo_notice,
    render_info_banner,
    render_result_card,
    render_section_header,
)
from src.audio.audio_classifier import load_audio_model
from src.audio.audio_preprocessor import (
    audio_arrays_from_bytes,
    extract_mfcc_from_bytes,
    spectrogram_db,
)
from src.text.explainability import (
    educational_summary,
    find_suspicious_phrases,
    highlighted_html,
)
from src.audio.recording_audio_simulation import analyze_audio_chunks, load_optional_audio_model
from src.utils.time_utils import formatted_now


@st.cache_resource(show_spinner=False)
def _load_audio_classifier(root: str):
    return load_audio_model(Path(root) / "models" / "audio_svm.pkl")


@st.cache_data(show_spinner=False, max_entries=3)
def _load_uploaded_audio(audio_bytes: bytes, suffix: str) -> tuple[np.ndarray, int]:
    return audio_arrays_from_bytes(audio_bytes, suffix=suffix)


@st.cache_data(show_spinner=False, max_entries=3)
def _run_chunk_analysis(
    audio_bytes: bytes,
    suffix: str,
    chunk_seconds: int,
    root: str,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    model = load_optional_audio_model(root)
    return analyze_audio_chunks(
        audio_bytes,
        suffix,
        chunk_seconds=chunk_seconds,
        model=model,
    )


def _probability_chart(probabilities: dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    values = [round(float(value) * 100, 2) for value in probabilities.values()]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#14b8a6", "#dc2626"],
        )
    )
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Confidence (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
    )
    return apply_chart_theme(fig)


def _waveform_figure(y: np.ndarray, sr: int) -> go.Figure:
    times = np.arange(len(y)) / sr
    fig = go.Figure(
        go.Scatter(
            x=times,
            y=y,
            mode="lines",
            line=dict(color="#2563eb", width=1),
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
    )
    return apply_chart_theme(fig)


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
        height=360,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
    )
    return apply_chart_theme(fig)


def _rolling_confidence_figure(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=results["end_sec"],
            y=results["confidence"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=3),
            marker=dict(
                size=8,
                color=results["confidence"],
                colorscale=[[0, "#10b981"], [0.55, "#f59e0b"], [1, "#ef4444"]],
                cmin=0,
                cmax=100,
            ),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.10)",
        )
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="Watch")
    fig.add_hline(y=80, line_dash="dash", line_color="#ef4444", annotation_text="High risk")
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Call time (seconds)",
        yaxis_title="Risk confidence (%)",
        yaxis=dict(range=[0, 100]),
    )
    return apply_chart_theme(fig)


def _read_transcript(uploaded_transcript) -> str:
    if uploaded_transcript is None:
        return ""
    suffix = Path(uploaded_transcript.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_transcript.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".csv":
        return pd.read_csv(uploaded_transcript).astype(str).head(20).to_string(index=False)
    return ""


def _record_audio_result(
    history: list[dict[str, object]],
    result: dict[str, object],
    filename: str,
) -> None:
    history.insert(
        0,
        {
            "time": formatted_now(),
            "type": "Audio",
            "prediction": result["label_name"],
            "confidence": round(float(result["confidence"]) * 100, 2),
            "model": "Audio SVM",
            "preview": filename,
        },
    )


def _record_simulation_result(
    history: list[dict[str, object]],
    results: pd.DataFrame,
    filename: str,
) -> None:
    history.insert(
        0,
        {
            "time": formatted_now(),
            "type": "Audio Simulation",
            "prediction": "Uploaded recording chunk analysis",
            "confidence": round(float(results["confidence"].max()), 2),
            "model": str(results["engine"].iloc[0]),
            "preview": filename,
            "chunks": int(len(results)),
        },
    )


def _render_single_audio_checker(root: Path, history: list[dict[str, object]]) -> None:
    render_section_header(
        "Inspect an uploaded voice recording",
        "Review waveform and spectrogram evidence before running the MFCC and SVM classifier.",
        "Audio evidence",
    )

    render_content_card_open("violet")
    uploaded_file = st.file_uploader(
        "Upload a voice recording",
        type=["wav", "flac"],
        key="audio_upload",
    )
    render_content_card_close()

    if uploaded_file is None:
        st.info("Upload a .wav or .flac file to view waveform and spectrogram analysis.")
        return

    suffix = Path(uploaded_file.name).suffix.lower()
    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes)

    try:
        y, sr = _load_uploaded_audio(audio_bytes, suffix)
        render_section_header("Waveform", eyebrow="Audio shape")
        render_content_card_open("violet")
        st.plotly_chart(_waveform_figure(y, sr), use_container_width=True)
        render_content_card_close()

        render_section_header("Spectrogram", eyebrow="Frequency view")
        render_content_card_open("violet")
        st.plotly_chart(_spectrogram_figure(y, sr), use_container_width=True)
        render_content_card_close()
    except Exception as exc:
        st.error(f"Audio visualization failed: {exc}")
        return

    if st.button("Analyze audio", type="primary", use_container_width=True):
        try:
            features = extract_mfcc_from_bytes(audio_bytes, suffix=suffix)
            classifier = _load_audio_classifier(str(root))
            prediction = classifier.predict_one(features)
            result = {
                "label": prediction.label,
                "label_name": prediction.label_name,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
            }
        except FileNotFoundError:
            st.warning(
                "Audio SVM artifact was not found. Run scripts/01_prepare_audio.py "
                "and scripts/04_train_audio_model.py first."
            )
            return
        except Exception as exc:
            st.error(f"Audio analysis failed: {exc}")
            return

        confidence = float(result["confidence"])
        risk_score = float(
            result["probabilities"].get(
                "Possible AI-generated speech",
                confidence if int(result["label"]) == 1 else 1 - confidence,
            )
        ) * 100
        render_analysis_ready("Audio analysis complete - results ready below")
        render_result_card(
            str(result["label_name"]),
            risk_score,
            educational_summary(str(result["label_name"]), confidence, []),
        )

        render_content_card_open("violet")
        st.plotly_chart(_probability_chart(result["probabilities"]), use_container_width=True)
        render_content_card_close()

        feature_df = pd.DataFrame(
            {
                "feature_index": list(range(len(features))),
                "mfcc_value": features,
            }
        )
        render_section_header("MFCC feature summary", eyebrow="Audio features")
        render_content_card_open("green")
        st.dataframe(feature_df.describe().T, use_container_width=True)
        render_content_card_close()
        _record_audio_result(history, result, uploaded_file.name)


def _render_recording_simulation(root: Path, history: list[dict[str, object]]) -> None:
    render_section_header(
        "Uploaded call and meeting recording analysis",
        "Simulate chunk-by-chunk review of suspicious voice evidence after a call or online meeting.",
        "Rolling audio simulation",
    )
    render_info_banner(
        "This workflow analyzes uploaded recordings only. It does not intercept live calls, phone audio, Zoom, Teams, or Google Meet streams.",
        kind="info",
        code="SCOPE",
    )

    render_content_card_open("violet")
    upload_col, settings_col = st.columns([0.62, 0.38])
    with upload_col:
        uploaded_audio = st.file_uploader(
            "Upload exported meeting or call recording audio",
            type=["wav", "flac", "mp3", "m4a"],
            key="sim_recording_audio_merged",
        )
        uploaded_transcript = st.file_uploader(
            "Optional transcript file",
            type=["txt", "csv"],
            key="sim_recording_transcript_merged",
        )
    with settings_col:
        chunk_seconds = st.slider(
            "Chunk size",
            min_value=5,
            max_value=10,
            value=5,
            step=1,
            key="sim_chunk_seconds_merged",
        )
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
        key="sim_transcript_text_merged",
    )
    render_content_card_close()

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

    render_analysis_ready(f"Analysis complete - {len(results)} chunks processed")
    render_section_header("Rolling detection dashboard", eyebrow="Chunk analysis")
    render_result_card(
        "Uploaded recording risk summary",
        float(results["confidence"].max()),
        f"Peak chunk confidence reached {results['confidence'].max():.1f}% using {results['engine'].iloc[0]}.",
    )

    render_content_card_open("red")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Chunks", len(results))
    metric_cols[1].metric("Peak confidence", f"{results['confidence'].max():.1f}%")
    metric_cols[2].metric("Average confidence", f"{results['confidence'].mean():.1f}%")
    metric_cols[3].metric("Engine", str(results["engine"].iloc[0]))
    st.plotly_chart(_rolling_confidence_figure(results), use_container_width=True)
    st.dataframe(results, hide_index=True, use_container_width=True)
    render_content_card_close()

    render_section_header("Waveform", eyebrow="Audio shape")
    render_content_card_open("violet")
    st.plotly_chart(_waveform_figure(y, sr), use_container_width=True)
    render_content_card_close()

    render_section_header("Spectrogram", eyebrow="Frequency view")
    render_content_card_open("violet")
    st.plotly_chart(_spectrogram_figure(y, sr), use_container_width=True)
    render_content_card_close()

    if transcript_text.strip():
        findings = find_suspicious_phrases(transcript_text)
        render_section_header("Transcript warning indicators", eyebrow="Text context")
        render_content_card_open("green")
        st.markdown(highlighted_html(transcript_text, findings), unsafe_allow_html=True)
        if findings:
            st.dataframe(
                pd.DataFrame(findings)[["phrase", "category", "reason"]],
                hide_index=True,
                use_container_width=True,
            )
        render_content_card_close()

    _record_simulation_result(history, results, uploaded_audio.name)


def render_audio_deepseek_tab(root: Path, history: list[dict[str, object]]) -> None:
    render_demo_notice(root)
    single_tab, simulation_tab = st.tabs(
        [
            "Single voice checker",
            "Uploaded recording simulation",
        ]
    )
    with single_tab:
        _render_single_audio_checker(root, history)
    with simulation_tab:
        _render_recording_simulation(root, history)


def render_audio_tab(root: Path, history: list[dict[str, object]]) -> None:
    """Backward-compatible wrapper for older imports."""

    render_audio_deepseek_tab(root, history)
