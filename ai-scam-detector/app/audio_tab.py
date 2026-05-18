"""Audio deepfake detection tab."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.audio_classifier import load_audio_model
from src.audio_preprocessor import audio_arrays_from_bytes, extract_mfcc_from_bytes, spectrogram_db
from src.explainability import educational_summary


@st.cache_resource(show_spinner=False)
def _load_audio_classifier(root: str):
    return load_audio_model(Path(root) / "models" / "audio_svm.pkl")


def _probability_chart(probabilities: dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    values = [round(float(value) * 100, 2) for value in probabilities.values()]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=["#14b8a6", "#dc2626"]))
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Confidence (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
    )
    return fig


def _waveform_figure(y: np.ndarray, sr: int) -> go.Figure:
    times = np.arange(len(y)) / sr
    fig = go.Figure(go.Scatter(x=times, y=y, mode="lines", line=dict(color="#2563eb", width=1)))
    fig.update_layout(
        height=240,
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
        height=360,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
    )
    return fig


def _record(history: list[dict[str, object]], result: dict[str, object], filename: str) -> None:
    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Audio",
            "prediction": result["label_name"],
            "confidence": round(float(result["confidence"]) * 100, 2),
            "model": "Audio SVM",
            "preview": filename,
        },
    )


def render_audio_tab(root: Path, history: list[dict[str, object]]) -> None:
    st.header("AI-Generated Speech Detection")

    uploaded_file = st.file_uploader(
        "Upload a voice recording",
        type=["wav", "flac"],
        key="audio_upload",
    )

    if uploaded_file is None:
        st.info("Upload a .wav or .flac file to view waveform and spectrogram analysis.")
        return

    suffix = Path(uploaded_file.name).suffix.lower()
    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes)

    try:
        y, sr = audio_arrays_from_bytes(audio_bytes, suffix=suffix)
        st.subheader("Waveform")
        st.plotly_chart(_waveform_figure(y, sr), use_container_width=True)

        st.subheader("Spectrogram")
        fig = _spectrogram_figure(y, sr)
        st.plotly_chart(fig, use_container_width=True)
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
            st.warning("Audio SVM artifact was not found. Run scripts/01_prepare_audio.py and scripts/04_train_audio_model.py first.")
            return
        except Exception as exc:
            st.error(f"Audio analysis failed: {exc}")
            return

        confidence = float(result["confidence"])
        if int(result["label"]) == 1:
            st.error(f"{result['label_name']} - {confidence * 100:.1f}% confidence")
        else:
            st.success(f"{result['label_name']} - {confidence * 100:.1f}% confidence")

        st.write(educational_summary(str(result["label_name"]), confidence, []))
        st.plotly_chart(_probability_chart(result["probabilities"]), use_container_width=True)

        feature_df = pd.DataFrame(
            {
                "feature_index": list(range(len(features))),
                "mfcc_value": features,
            }
        )
        st.subheader("MFCC feature summary")
        st.dataframe(feature_df.describe().T, use_container_width=True)
        _record(history, result, uploaded_file.name)
