"""Call and meeting transcript scam detection tab with uploaded audio support."""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import os
import re
import tempfile
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_components import (
    apply_chart_theme,
    get_demo_data,
    render_analysis_ready,
    render_content_card_close,
    render_content_card_open,
    render_detection_tool_intro,
    render_metric_row,
    render_result_card,
    render_section_header,
)
from src.text.explainability import (
    educational_summary,
    find_suspicious_phrases,
    highlighted_html,
)
try:
    from src.reporting.history_db import record_history_item
except ImportError:
    import importlib
    import src.reporting.history_db as history_db

    record_history_item = importlib.reload(history_db).record_history_item
import src.audio.live_audio_analysis as live_audio_analysis
from src.text.rule_demo import rule_based_text_prediction

if not all(
    hasattr(live_audio_analysis, name)
    for name in (
        "BEHAVIORAL_FEATURE_NAMES",
        "assess_speech_quality",
        "analyse_live_chunk",
        "transcribe_with_whisper_details",
    )
):
    live_audio_analysis = importlib.reload(live_audio_analysis)

BEHAVIORAL_FEATURE_NAMES = live_audio_analysis.BEHAVIORAL_FEATURE_NAMES
assess_speech_quality = live_audio_analysis.assess_speech_quality
analyse_live_chunk = live_audio_analysis.analyse_live_chunk
transcribe_with_whisper_details = live_audio_analysis.transcribe_with_whisper_details


WHISPER_MODEL_LABELS = {
    "tiny.en": "tiny.en - fastest English, lowest accuracy",
    "tiny": "tiny - fastest multilingual, lowest accuracy",
    "base.en": "base.en - fast English, better demo default",
    "base": "base - fast multilingual",
    "small.en": "small.en - slower English, stronger accuracy",
    "small": "small - slower multilingual, stronger accuracy",
    "medium.en": "medium.en - high accuracy, high memory",
    "medium": "medium - high accuracy multilingual, high memory",
    "large-v3-turbo": "large-v3-turbo - strongest local option, very high memory",
    "turbo": "turbo - strongest local option, very high memory",
}
WHISPER_MODEL_ORDER = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v3-turbo",
    "turbo",
]
WHISPER_INITIAL_PROMPT = (
    "This is an English call or meeting transcript about student safety, banking, "
    "verification, accounts, payments, OTPs, passwords, universities, and scam prevention."
)
TRANSCRIPT_MODEL_FILES = {
    "nb": ("Naive Bayes", "transcript_nb.pkl"),
    "svm": ("SVM", "transcript_svm.pkl"),
    "distilbert": ("DistilBERT", "transcript_distilbert"),
}
TRANSCRIPT_TRANSFORMER_MODEL_KEYS = {"distilbert"}
LOCAL_TRANSFORMER_ARCHIVE = Path("archive") / "local_models" / "models"


@st.cache_resource(show_spinner=False)
def _load_audio_classifier(root: str):
    try:
        from src.audio.audio_classifier import load_audio_model

        return load_audio_model(Path(root) / "models" / "audio_svm.pkl")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_behavioral_classifier(root: str):
    try:
        from src.audio.audio_classifier import load_audio_behavior_model

        return load_audio_behavior_model(Path(root) / "models" / "audio_behavior_rf.pkl")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_transcript_classifier(root: str, model_key: str = "nb"):
    model_path, model_name = _transcript_model_artifact(Path(root), model_key)
    if model_key in TRANSCRIPT_TRANSFORMER_MODEL_KEYS:
        from src.text.transformer_classifier import load_transformer_text_artifacts

        return load_transformer_text_artifacts(model_path, model_name=model_name)
    from src.text.text_classifier import load_text_artifacts

    return load_text_artifacts(
        Path(root) / "models" / "transcript_vectorizer.pkl",
        model_path,
        model_name=model_name,
    )


@st.cache_resource(show_spinner=False)
def _load_transcript_classifier_safe(root: str, model_key: str = "nb"):
    try:
        return _load_transcript_classifier(root, model_key)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_whisper_model(model_size: str):
    whisper_module = _load_whisper_module()
    if whisper_module is None:
        return None
    return whisper_module.load_model(model_size)


def _load_whisper_module() -> Any | None:
    try:
        import whisper
    except Exception:
        return None
    return whisper


def _transcript_model_candidates(root: Path, model_key: str) -> list[Path]:
    _label, filename = TRANSCRIPT_MODEL_FILES.get(
        model_key,
        TRANSCRIPT_MODEL_FILES["nb"],
    )
    candidates = [root / "models" / filename]
    if model_key in TRANSCRIPT_TRANSFORMER_MODEL_KEYS:
        env_root = os.environ.get("AIFDS_LOCAL_TRANSFORMER_MODELS_DIR", "").strip()
        if env_root:
            env_path = Path(env_root)
            candidates.extend([env_path / filename, env_path])
        candidates.append(root / LOCAL_TRANSFORMER_ARCHIVE / filename)
    return candidates


def _transcript_model_path(root: Path, model_key: str) -> Path:
    candidates = _transcript_model_candidates(root, model_key)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _transcript_model_label(root: Path, model_key: str) -> str:
    label, _filename = TRANSCRIPT_MODEL_FILES.get(
        model_key,
        TRANSCRIPT_MODEL_FILES["nb"],
    )
    return label


def _transcript_model_artifact(root: Path, model_key: str = "nb") -> tuple[Path, str]:
    """Return the selected transcript model artifact, falling back safely."""

    label, filename = TRANSCRIPT_MODEL_FILES.get(
        model_key,
        TRANSCRIPT_MODEL_FILES["nb"],
    )
    model_path = _transcript_model_path(root, model_key)
    if model_path.exists():
        return model_path, f"Transcript {label}"

    for fallback_key, (fallback_label, fallback_filename) in TRANSCRIPT_MODEL_FILES.items():
        fallback_path = _transcript_model_path(root, fallback_key)
        if fallback_path.exists():
            return fallback_path, f"Transcript {fallback_label}"

    return root / "models" / "transcript_nb.pkl", "Transcript Naive Bayes"


def _available_transcript_models(root: Path) -> list[str]:
    options = [
        key
        for key, (_label, filename) in TRANSCRIPT_MODEL_FILES.items()
        if _transcript_model_path(root, key).exists()
    ]
    return options or ["nb"]


def _default_transcript_model_keys(options: list[str]) -> list[str]:
    transformer_defaults = [key for key in ("distilbert",) if key in options]
    if transformer_defaults:
        anchors = [key for key in ("svm", "nb") if key in options]
        return (transformer_defaults + anchors)[:4]

    preferred = ["nb", "svm"]
    defaults = [key for key in preferred if key in options]
    return defaults or options[:1]


def _available_whisper_models() -> list[str]:
    return ["tiny.en", "tiny", "base.en", "base", "small.en", "small"]


def _cached_whisper_models() -> set[str]:
    cache_dir = Path.home() / ".cache" / "whisper"
    if not cache_dir.exists():
        return set()
    return {path.stem for path in cache_dir.glob("*.pt")}


def _whisper_model_label(model_name: str) -> str:
    label = WHISPER_MODEL_LABELS.get(model_name, model_name)
    cached = _cached_whisper_models()
    if model_name in cached:
        return f"{label} | cached"
    return f"{label} | first use may download"


def _default_whisper_model(options: list[str]) -> str:
    cached = _cached_whisper_models()
    for model_name in ("base.en", "base", "tiny.en", "tiny"):
        if model_name in options and model_name in cached:
            return model_name
    for model_name in ("base.en", "base", "tiny.en", "tiny"):
        if model_name in options:
            return model_name
    return options[0]


def _init_transcript_voice_state() -> None:
    defaults: dict[str, Any] = {
        "transcript_use_uploaded_audio": False,
        "transcript_use_text": False,
        "transcript_text_preview": "",
        "transcript_uploaded_audio_file_name": None,
        "transcript_uploaded_audio_file_bytes": None,
        "transcript_uploaded_audio_file_suffix": "",
        "transcript_uploaded_audio_file_signature": None,
        "transcript_uploaded_audio_last_processed_signature": None,
        "transcript_uploaded_audio_results": [],
        "transcript_uploaded_audio_error": "",
        "transcript_uploaded_audio_carousel_index": 0,
        "transcript_pending_uploaded_audio_analysis": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key in (
        "transcript_use_voice",
        "transcript_voice_sessions",
        "transcript_voice_active_session_id",
        "transcript_voice_active_index",
        "transcript_voice_selector_generation",
        "transcript_voice_mode",
        "transcript_recorder_generation",
        "transcript_recorder_error",
        "transcript_recorder_carousel_index",
        "transcript_pending_voice_analysis",
    ):
        st.session_state.pop(key, None)


def _clear_uploaded_audio_state(
    *,
    clear_file: bool = False,
) -> None:
    """Clear uploaded-audio analysis without affecting speaker recordings."""

    st.session_state["transcript_uploaded_audio_results"] = []
    st.session_state["transcript_uploaded_audio_error"] = ""
    st.session_state["transcript_pending_uploaded_audio_analysis"] = False
    st.session_state["transcript_uploaded_audio_last_processed_signature"] = None

    if clear_file:
        st.session_state["transcript_uploaded_audio_file_name"] = None
        st.session_state["transcript_uploaded_audio_file_bytes"] = None
        st.session_state["transcript_uploaded_audio_file_suffix"] = ""
        st.session_state["transcript_uploaded_audio_file_signature"] = None


def _recording_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: int,
) -> list[np.ndarray]:
    chunk_size = max(1, int(sample_rate * chunk_seconds))
    chunks = []
    for start in range(0, audio.size, chunk_size):
        chunk = audio[start : start + chunk_size]
        if chunk.size >= int(sample_rate * 0.75):
            chunks.append(chunk.astype(np.float32))
    return chunks or [audio.astype(np.float32)]


def _process_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    *,
    chunk_seconds: int,
    transcript_source: str,
    manual_transcript: str,
    whisper_model: Any | None,
    whisper_language: str | None = "en",
    whisper_task: str = "transcribe",
    audio_classifier: Any | None,
    text_classifier: Any | None,
    behavioral_classifier: Any | None,
) -> list[dict[str, object]]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError("The audio file contained no usable samples.")

    processed = []
    for index, chunk in enumerate(_recording_chunks(audio, sample_rate, chunk_seconds)):
        audio_quality = assess_speech_quality(chunk, sample_rate=sample_rate)
        transcription_details: dict[str, object] = {
            "text": "",
            "usable": False,
            "quality_label": str(audio_quality.get("reason", "Not transcribed")),
            "warnings": [],
            "language": whisper_language or "",
            "language_confidence": None,
            "forced_language": whisper_language or "",
            "no_speech_probability": None,
            "avg_logprob": None,
            "compression_ratio": None,
        }
        if transcript_source == "Local Whisper":
            if not bool(audio_quality.get("usable_speech", True)):
                transcript = ""
            else:
                transcription_details = transcribe_with_whisper_details(
                    chunk,
                    whisper_model,
                    language=whisper_language,
                    task=whisper_task,
                    initial_prompt=WHISPER_INITIAL_PROMPT,
                )
                transcript = str(transcription_details.get("text", "")).strip()
        elif transcript_source == "Manual transcript" and index == 0:
            transcript = manual_transcript
            transcription_details = {
                **transcription_details,
                "text": transcript,
                "raw_text": transcript,
                "usable": bool(transcript.strip()),
                "quality_label": "Manual transcript supplied",
            }
        else:
            transcript = ""

        result = analyse_live_chunk(
            chunk,
            transcript=transcript,
            audio_classifier=audio_classifier,
            text_classifier=text_classifier,
            behavioral_classifier=behavioral_classifier,
            sample_rate=sample_rate,
        )
        result["pre_transcription_quality"] = audio_quality
        result["transcription"] = transcription_details
        result["transcription_status"] = str(
            transcription_details.get("quality_label")
            or audio_quality.get("reason")
            or "Not transcribed"
        )
        result["quality_warnings"] = [
            *list(result.get("quality_warnings", [])),
            *[
                str(message)
                for message in transcription_details.get("warnings", [])
                if str(message).strip()
            ],
        ]
        processed.append(result)
    return processed


def _process_uploaded_audio(
    audio_bytes: bytes,
    suffix: str,
    *,
    chunk_seconds: int,
    transcript_source: str,
    manual_transcript: str,
    whisper_model: Any | None,
    whisper_language: str | None = "en",
    whisper_task: str = "transcribe",
    audio_classifier: Any | None,
    text_classifier: Any | None,
    behavioral_classifier: Any | None,
) -> list[dict[str, object]]:
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError("librosa is required to decode uploaded audio files.") from exc

    suffix = suffix if suffix in {".wav", ".mp3", ".flac"} else ".wav"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        audio, sample_rate = librosa.load(str(temp_path), sr=16_000, mono=True)
        return _process_audio_array(
            np.asarray(audio, dtype=np.float32),
            int(sample_rate),
            chunk_seconds=chunk_seconds,
            transcript_source=transcript_source,
            manual_transcript=manual_transcript,
            whisper_model=whisper_model,
            whisper_language=whisper_language,
            whisper_task=whisper_task,
            audio_classifier=audio_classifier,
            text_classifier=text_classifier,
            behavioral_classifier=behavioral_classifier,
        )
    except Exception as exc:
        if suffix == ".mp3":
            raise RuntimeError(
                "MP3 decoding may require ffmpeg. Try uploading WAV or FLAC, "
                "or install ffmpeg and add it to PATH."
            ) from exc
        raise
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _analyse_selected_uploaded_audio(
    root: Path,
    *,
    chunk_seconds: int,
    whisper_size: str,
    whisper_language: str | None = "en",
    whisper_task: str = "transcribe",
    transcript_model_key: str = "nb",
    analysis_signature: str | None = None,
) -> list[dict[str, object]]:
    """Analyze only the currently selected uploaded-audio file."""

    audio_bytes = st.session_state.get("transcript_uploaded_audio_file_bytes")
    suffix = str(st.session_state.get("transcript_uploaded_audio_file_suffix", ""))
    signature = st.session_state.get("transcript_uploaded_audio_file_signature")

    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise RuntimeError("Upload an audio recording before running the analysis.")

    if suffix not in {".wav", ".mp3", ".flac"}:
        raise RuntimeError("The selected uploaded audio must be WAV, MP3, or FLAC.")

    whisper_model = _load_whisper_model(whisper_size)
    if whisper_model is None:
        st.session_state["transcript_uploaded_audio_whisper_notice"] = (
            "Local Whisper is unavailable in this environment, so uploaded audio was analysed without speech-to-text."
        )
    else:
        st.session_state.pop("transcript_uploaded_audio_whisper_notice", None)

    transcript_source = "Local Whisper" if whisper_model is not None else "Audio only"

    processed = _process_uploaded_audio(
        audio_bytes,
        suffix,
        chunk_seconds=chunk_seconds,
        transcript_source=transcript_source,
        manual_transcript="",
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        whisper_task=whisper_task,
        audio_classifier=_load_audio_classifier(str(root)),
        text_classifier=_load_transcript_classifier_safe(str(root), transcript_model_key),
        behavioral_classifier=_load_behavioral_classifier(str(root)),
    )

    filename = str(
        st.session_state.get(
            "transcript_uploaded_audio_file_name",
            "Uploaded audio",
        )
    )

    for chunk_index, result in enumerate(processed, start=1):
        result["clip"] = 1
        result["clip_chunk"] = chunk_index
        result["capture_mode"] = "Uploaded Audio Recording"
        result["source_filename"] = filename
        result["source_signature"] = signature

    st.session_state["transcript_uploaded_audio_results"] = processed
    st.session_state["transcript_uploaded_audio_last_processed_signature"] = (
        analysis_signature or signature
    )
    st.session_state["transcript_uploaded_audio_error"] = ""
    st.session_state["transcript_pending_uploaded_audio_analysis"] = True
    st.session_state["transcript_uploaded_audio_carousel_index"] = 0

    return processed


def _timeline_figure(results: list[dict[str, object]], threshold: int) -> go.Figure:
    x_values = list(range(1, len(results) + 1))
    risks = [float(item.get("risk", 0)) for item in results]
    labels = [
        (
            f"Clip {item.get('clip', 1)}, chunk {item.get('clip_chunk', index + 1)}"
            f"<br>{item.get('risk_level', '')}"
        )
        for index, item in enumerate(results)
    ]
    colors = [
        "#DC2626" if value >= threshold else "#D97706" if value >= 40 else "#0891B2"
        for value in risks
    ]
    fig = go.Figure(
        go.Scatter(
            x=x_values,
            y=risks,
            mode="lines+markers",
            text=labels,
            hovertemplate="Session point %{x}<br>Risk %{y:.1f}%<br>%{text}<extra></extra>",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=8, color=colors),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.08)",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Alert threshold {threshold}%",
    )
    fig.update_layout(
        height=285,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Processed chunk",
        yaxis_title="Combined risk (%)",
        yaxis=dict(range=[0, 100]),
    )
    return apply_chart_theme(fig)


def _mfcc_figure(results: list[dict[str, object]]) -> go.Figure | None:
    columns = []
    for result in results[-30:]:
        features = result.get("features")
        if not isinstance(features, dict):
            continue
        mfcc = features.get("mfcc_mean")
        if isinstance(mfcc, list) and len(mfcc) == 40:
            columns.append(mfcc)
    if not columns:
        return None

    matrix = np.asarray(columns, dtype=float).T
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            colorscale="RdBu_r",
            colorbar=dict(title="MFCC"),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        height=285,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Recent chunk",
        yaxis_title="MFCC coefficient",
    )
    return apply_chart_theme(fig)


def _frequency_figure(result: dict[str, object] | None) -> go.Figure | None:
    if not result:
        return None
    features = result.get("features")
    if not isinstance(features, dict):
        return None
    frequencies = features.get("spectrum_frequencies")
    decibels = features.get("spectrum_db")
    if not isinstance(frequencies, list) or not isinstance(decibels, list) or not frequencies:
        return None

    fig = go.Figure(
        go.Scatter(
            x=frequencies,
            y=decibels,
            mode="lines",
            line=dict(color="#0891B2", width=2),
            fill="tozeroy",
            fillcolor="rgba(8,145,178,0.10)",
            hovertemplate="%{x:.0f} Hz<br>%{y:.1f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Relative level (dB)",
        yaxis=dict(range=[-80, 5]),
    )
    return apply_chart_theme(fig)


def _cumulative_transcript(results: list[dict[str, object]]) -> str:
    lines = []
    for result in results:
        transcript = str(result.get("transcript", "")).strip()
        if transcript:
            lines.append(
                f"[Clip {result.get('clip', 1)} | {result.get('time', '--:--:--')}] "
                f"{transcript}"
            )
    return "\n".join(lines)


def _result_table(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in reversed(results[-20:]):
        transcript = str(result.get("transcript", "")).strip()
        quality = result.get("audio_quality", {})
        if not isinstance(quality, dict):
            quality = {}
        transcription = result.get("transcription", {})
        if not isinstance(transcription, dict):
            transcription = {}
        detected_language = str(
            transcription.get("detected_language")
            or transcription.get("language")
            or "-"
        )
        language_confidence = transcription.get("language_confidence")
        if isinstance(language_confidence, (int, float)):
            detected_language = f"{detected_language} ({float(language_confidence) * 100:.0f}%)"
        rows.append(
            {
                "Clip": int(result.get("clip", 1)),
                "Chunk": int(result.get("clip_chunk", 1)),
                "Time": result.get("time", "-"),
                "Risk": f"{float(result.get('risk', 0)):.1f}%",
                "Voice": f"{float(result.get('voice_risk', 0)):.1f}%",
                "Transcript": f"{float(result.get('transcript_risk', 0)):.1f}%",
                "Behavioral": _risk_value_text(result.get("behavioral_risk")),
                "Speech quality": quality.get("reason", "Usable speech-like audio"),
                "Whisper": result.get("transcription_status", "Not transcribed"),
                "Language check": detected_language,
                "Detected text": transcript or "No usable speech text",
                "Flags": ", ".join(result.get("flags", [])) or "-",
            }
        )
    return pd.DataFrame(rows)


def _risk_value_text(value: object) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "Unavailable"


def _behavioral_feature_rows(result: dict[str, object]) -> pd.DataFrame:
    features = result.get("behavioral_features", {})
    if not isinstance(features, dict):
        return pd.DataFrame()

    rows = []
    for feature_name in BEHAVIORAL_FEATURE_NAMES:
        value = features.get(feature_name)
        if value is None:
            continue
        rows.append(
            {
                "Behavior Feature": feature_name.replace("_", " ").title(),
                "Value": f"{float(value):.4f}",
            }
        )
    return pd.DataFrame(rows)


def _render_live_dashboard(
    results: list[dict[str, object]],
    threshold: int,
    *,
    latest_title: str,
    empty_message: str,
    metrics_placeholder,
    result_placeholder,
    timeline_placeholder,
    transcript_placeholder,
    mfcc_placeholder,
    frequency_placeholder,
    features_placeholder,
) -> None:
    latest = results[-1] if results else None
    peak = max((float(item.get("risk", 0)) for item in results), default=0.0)
    average = (
        sum(float(item.get("risk", 0)) for item in results) / len(results)
        if results
        else 0.0
    )
    alert_count = sum(1 for item in results if float(item.get("risk", 0)) >= threshold)

    with metrics_placeholder.container():
        render_metric_row(
            [
                {"label": "Chunks Analysed", "value": len(results), "color": "#2563EB"},
                {"label": "Current Risk", "value": f"{float(latest.get('risk', 0)):.0f}%" if latest else "0%", "color": "#D97706"},
                {"label": "Behavioral Risk", "value": _risk_value_text(latest.get("behavioral_risk")) if latest else "Unavailable", "color": "#7C3AED"},
                {"label": "Peak Risk", "value": f"{peak:.0f}%", "color": "#DC2626"},
                {"label": "Average Risk", "value": f"{average:.0f}%", "color": "#0891B2"},
                {"label": "Alerts", "value": alert_count, "color": "#DC2626"},
            ]
        )

    with result_placeholder.container():
        if latest:
            render_result_card(
                latest_title.format(chunk=latest.get("clip_chunk", 1)),
                float(latest.get("risk", 0)),
                str(latest.get("explanation", "")),
            )
            if float(latest.get("risk", 0)) >= threshold:
                st.error(
                    f"Alert threshold reached. This chunk scored {float(latest.get('risk', 0)):.1f}% combined risk."
                )
            quality_messages = [
                str(message)
                for message in latest.get("quality_warnings", [])
                if str(message).strip()
            ]
            audio_quality = latest.get("audio_quality", {})
            if isinstance(audio_quality, dict) and not bool(
                audio_quality.get("usable_speech", True)
            ):
                quality_messages.insert(
                    0,
                    str(audio_quality.get("reason", "No usable speech detected")),
                )
            if quality_messages:
                st.warning("Audio/transcript quality: " + " ".join(dict.fromkeys(quality_messages)))
        else:
            st.info(empty_message)

    with timeline_placeholder.container():
        st.plotly_chart(_timeline_figure(results, threshold), use_container_width=True)

    with transcript_placeholder.container():
        if results:
            transcript_text = _cumulative_transcript(results)
            st.text_area(
                "Live transcript",
                value=transcript_text or "No speech text yet. Enable Whisper for automatic transcription.",
                height=145,
                disabled=True,
            )
            st.dataframe(_result_table(results), hide_index=True, use_container_width=True)
            transcript = str(latest.get("transcript", "")).strip() if latest else ""
            findings = latest.get("findings", []) if latest else []
            if transcript and isinstance(findings, list):
                st.markdown(highlighted_html(transcript, findings), unsafe_allow_html=True)
        else:
            st.text_area("Live transcript", value=empty_message, height=145, disabled=True)
            st.caption("No transcript or audio chunk results yet.")

    with mfcc_placeholder.container():
        figure = _mfcc_figure(results)
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.caption("MFCC heatmap appears after the first processed chunk.")

    with frequency_placeholder.container():
        frequency_figure = _frequency_figure(latest)
        if frequency_figure is not None:
            st.plotly_chart(frequency_figure, use_container_width=True)
        else:
            st.caption("Frequency spectrum appears after the first processed chunk.")

    with features_placeholder.container():
        if not latest:
            st.caption("Acoustic feature values appear after the first processed chunk.")
        else:
            features = latest.get("features", {})
            if not isinstance(features, dict):
                features = {}
            feature_rows = pd.DataFrame(
                [
                    {"Feature": "Combined risk", "Value": f"{float(latest.get('risk', 0)):.2f}%"},
                    {"Feature": "Voice AI risk", "Value": f"{float(latest.get('voice_risk', 0)):.2f}%"},
                    {"Feature": "Transcript scam risk", "Value": f"{float(latest.get('transcript_risk', 0)):.2f}%"},
                    {"Feature": "Behavioral risk", "Value": _risk_value_text(latest.get("behavioral_risk"))},
                    {"Feature": "Speech quality", "Value": str(audio_quality.get("reason", "Usable speech-like audio")) if isinstance(audio_quality, dict) else "Unknown"},
                    {"Feature": "Pitch variance", "Value": f"{float(features.get('pitch_variance', 0)):.2f} Hz"},
                    {"Feature": "Spectral centroid", "Value": f"{float(features.get('spectral_centroid', 0)) / 1000:.2f} kHz"},
                    {"Feature": "Dominant frequency", "Value": f"{float(features.get('dominant_frequency', 0)):.1f} Hz"},
                    {"Feature": "Zero crossing rate", "Value": f"{float(features.get('zero_crossing_rate', 0)):.4f}"},
                    {"Feature": "RMS energy", "Value": f"{float(features.get('rms_energy', 0)):.4f}"},
                ]
            )
            st.dataframe(feature_rows, hide_index=True, use_container_width=True)

            behavioral_rows = _behavioral_feature_rows(latest)
            if behavioral_rows.empty:
                st.caption("Behavioral feature values appear after the first processed chunk.")
            else:
                st.caption(str(latest.get("behavioral_engine", "Behavioral model unavailable")))
                st.dataframe(behavioral_rows, hide_index=True, use_container_width=True)


def _render_dashboard_section(
    results: list[dict[str, object]],
    risk_threshold: int,
    *,
    transcript_heading: str,
    frequency_heading: str,
    latest_title: str,
) -> None:
    if not results:
        return

    metrics_placeholder = st.empty()
    result_placeholder = st.empty()
    timeline_placeholder = st.empty()
    display_a, display_b = st.columns([0.62, 0.38])
    with display_a:
        render_section_header(transcript_heading, eyebrow="Analysis evidence")
        transcript_placeholder = st.empty()
        render_section_header("MFCC feature heatmap", eyebrow="Audio pattern")
        mfcc_placeholder = st.empty()
    with display_b:
        render_section_header(frequency_heading, eyebrow="Frequency analysis")
        frequency_placeholder = st.empty()
        render_section_header("Latest acoustic features", eyebrow="Voice indicators")
        features_placeholder = st.empty()

    _render_live_dashboard(
        results,
        risk_threshold,
        latest_title=latest_title,
        empty_message="No processed audio yet.",
        metrics_placeholder=metrics_placeholder,
        result_placeholder=result_placeholder,
        timeline_placeholder=timeline_placeholder,
        transcript_placeholder=transcript_placeholder,
        mfcc_placeholder=mfcc_placeholder,
        frequency_placeholder=frequency_placeholder,
        features_placeholder=features_placeholder,
    )


def _recording_groups(results: list[dict[str, object]]) -> list[tuple[int, list[dict[str, object]]]]:
    groups: dict[int, list[dict[str, object]]] = {}
    for result in results:
        clip_number = int(result.get("clip", 1))
        groups.setdefault(clip_number, []).append(result)
    return sorted(groups.items(), key=lambda item: item[0])


def _render_recording_carousel(
    results: list[dict[str, object]],
    risk_threshold: int,
    *,
    state_key: str,
    title: str,
    transcript_heading: str,
    frequency_heading: str,
    latest_title: str,
) -> None:
    groups = _recording_groups(results)
    if not groups:
        return

    current_index = int(st.session_state.get(state_key, len(groups) - 1))
    current_index = max(0, min(current_index, len(groups) - 1))
    st.session_state[state_key] = current_index
    clip_number, clip_results = groups[current_index]
    peak = max(float(item.get("risk", 0)) for item in clip_results)
    flags = sorted(
        {
            str(flag)
            for item in clip_results
            for flag in item.get("flags", [])
            if str(flag).strip()
        }
    )

    render_section_header(
        title,
        f"Recording {current_index + 1} of {len(groups)} | Clip {clip_number} | Peak risk {peak:.1f}%",
        "Recording carousel",
    )
    nav_left, nav_mid, nav_right = st.columns([0.2, 0.6, 0.2])
    with nav_left:
        if st.button("Previous", use_container_width=True, disabled=current_index == 0, key=f"{state_key}_prev"):
            st.session_state[state_key] = current_index - 1
            st.rerun()
    with nav_mid:
        st.markdown(
            f"**Clip {clip_number}** | {len(clip_results)} chunk(s) | "
            f"Flags: {', '.join(flags) if flags else 'none'}"
        )
    with nav_right:
        if st.button("Next", use_container_width=True, disabled=current_index >= len(groups) - 1, key=f"{state_key}_next"):
            st.session_state[state_key] = current_index + 1
            st.rerun()

    if peak >= risk_threshold:
        st.warning(
            "Recommendation: pause the conversation, do not share OTP/passwords/payment details, "
            "and verify the request through an official channel."
        )
    elif peak >= 40:
        st.info(
            "Recommendation: treat this as needing review. Ask for written confirmation and "
            "check the sender/caller through a trusted source."
        )
    else:
        st.success(
            "Recommendation: no strong scam indicators were found in this recording, but continue "
            "to verify unexpected requests."
        )
    _render_student_ctas(peak, title="Audio response checklist")

    _render_dashboard_section(
        clip_results,
        risk_threshold,
        transcript_heading=transcript_heading,
        frequency_heading=frequency_heading,
        latest_title=latest_title,
    )


def _uploaded_audio_transcript_text() -> str:
    """Return the usable transcript text generated from uploaded audio recordings."""

    results = st.session_state.get("transcript_uploaded_audio_results", [])
    return _transcript_text_from_results(results)


def _transcript_text_from_results(results: object) -> str:
    """Return combined transcript text from analysed audio chunk results."""

    if not isinstance(results, list):
        return ""

    lines = []
    for result in results:
        if not isinstance(result, dict):
            continue
        transcript = str(result.get("transcript", "")).strip()
        if transcript:
            lines.append(transcript)
    return "\n".join(lines).strip()


def _read_upload(uploaded_file) -> str | pd.DataFrame | None:
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    st.warning("Only .txt and .csv files are supported in this tab.")
    return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _load_demo_examples(root: str) -> pd.DataFrame | None:
    root_path = Path(root)
    path = root_path / "data" / "raw" / "transcripts" / "youtube_scam_transcripts.csv"
    if not path.exists():
        return get_demo_data()["transcripts"][["sample_id", "transcript", "label"]]
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _confidence_chart(probabilities: dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    values = [round(float(value) * 100, 2) for value in probabilities.values()]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=["#22c55e", "#f97316"]))
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Confidence (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
    )
    return apply_chart_theme(fig)


def _predict(
    root: Path,
    text: str,
    *,
    transcript_model_key: str = "nb",
) -> tuple[dict[str, object], object | None]:
    try:
        classifier = _load_transcript_classifier(str(root), transcript_model_key)
        prediction = classifier.predict_one(text)
        findings = find_suspicious_phrases(text)
        return (
            {
                "label": prediction.label,
                "label_name": prediction.label_name,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
                "model_name": prediction.model_name,
                "findings": findings,
            },
            classifier,
        )
    except FileNotFoundError:
        result = rule_based_text_prediction(text)
        st.warning("Transcript model artifacts were not found, so this result uses educational demo rules.")
        return result, None


def _risk_score(result: dict[str, object]) -> float:
    probabilities = dict(result.get("probabilities", {}))
    confidence = float(result.get("confidence", 0.0))
    label = int(result.get("label", 0))

    if "Suspicious" in probabilities:
        return max(0.0, min(100.0, float(probabilities["Suspicious"]) * 100))

    return max(0.0, min(100.0, confidence * 100 if label == 1 else (1 - confidence) * 100))


def _is_suspicious_prediction(value: object) -> bool:
    return "suspicious" in str(value).casefold()


def _label_from_verdict(verdict: str) -> int:
    return 1 if _is_suspicious_prediction(verdict) else 0


@st.cache_data(show_spinner=False, ttl=300, max_entries=4)
def _load_transcript_metrics(root: str) -> dict[str, object]:
    metrics_path = Path(root) / "reports" / "metrics" / "transcript_model_metrics.json"
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _transcript_metrics_name(root: Path, model_key: str) -> str:
    return {
        "nb": "Naive Bayes",
        "svm": "SVM",
        "distilbert": "DistilBERT",
    }.get(model_key, "")


def _training_time_value(values: dict[str, object]) -> float | None:
    for key in ("training_time", "training_time_seconds", "training_seconds"):
        if key in values and values[key] is not None:
            return round(float(values[key]), 3)
    return None


def _prediction_time_value(values: dict[str, object]) -> float | None:
    if values.get("prediction_time_ms") is not None:
        return round(float(values["prediction_time_ms"]), 4)
    if values.get("prediction_time_seconds") is not None:
        return round(float(values["prediction_time_seconds"]) * 1000, 4)
    return None


def _clean_text_for_training_similarity(text: str) -> str:
    lines = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def _load_transcript_training_dataset(root: str) -> pd.DataFrame:
    dataset_path = Path(root) / "data" / "processed" / "transcript" / "transcript_dataset.csv"
    try:
        dataset = pd.read_csv(dataset_path)
    except Exception:
        return pd.DataFrame()

    required_columns = {"transcript", "label"}
    if not required_columns.issubset(dataset.columns):
        return pd.DataFrame()

    dataset = dataset.dropna(subset=["transcript", "label"]).copy()
    dataset["transcript"] = dataset["transcript"].astype(str)
    dataset["label"] = dataset["label"].astype(int)
    if "source" not in dataset.columns:
        dataset["source"] = "Training dataset"
    return dataset[dataset["transcript"].str.strip().str.len() > 0].reset_index(drop=True)


def _nearest_training_examples(root: Path, text: str, *, top_n: int = 5) -> pd.DataFrame:
    dataset = _load_transcript_training_dataset(str(root))
    clean_text = _clean_text_for_training_similarity(text)
    if dataset.empty or not clean_text:
        return pd.DataFrame()

    corpus = dataset["transcript"].astype(str).tolist()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([clean_text, *corpus])
        similarities = cosine_similarity(matrix[0], matrix[1:]).ravel()
    except Exception:
        return pd.DataFrame()

    if similarities.size == 0:
        return pd.DataFrame()

    order = np.argsort(similarities)[::-1][:top_n]
    rows = []
    for index in order:
        training_row = dataset.iloc[int(index)]
        snippet = " ".join(str(training_row["transcript"]).split())
        rows.append(
            {
                "Similarity": f"{float(similarities[index]) * 100:.1f}%",
                "Training Label": "Suspicious" if int(training_row["label"]) == 1 else "Legitimate",
                "Closest Training Snippet": snippet[:220] + ("..." if len(snippet) > 220 else ""),
                "Source": str(training_row.get("source", "Training dataset")),
            }
        )
    return pd.DataFrame(rows)


def _baseline_weight_vector(model: object) -> tuple[np.ndarray | None, str]:
    if hasattr(model, "feature_log_prob_") and model.feature_log_prob_.shape[0] >= 2:
        return model.feature_log_prob_[1] - model.feature_log_prob_[0], "naive_bayes_log_probability_delta"

    coefficients = []
    for calibrated_model in getattr(model, "calibrated_classifiers_", []) or []:
        estimator = getattr(calibrated_model, "estimator", None)
        if estimator is not None and hasattr(estimator, "coef_"):
            coefficients.append(np.ravel(estimator.coef_))
    if coefficients:
        return np.mean(coefficients, axis=0), "svm_linear_coefficient"

    if hasattr(model, "coef_"):
        return np.ravel(model.coef_), "linear_coefficient"

    return None, ""


def _baseline_vocabulary_terms(
    text: str,
    classifier: object,
    model_label: str,
    *,
    top_n: int = 8,
) -> list[dict[str, object]]:
    vectorizer = getattr(classifier, "vectorizer", None)
    model = getattr(classifier, "model", None)
    if vectorizer is None or model is None:
        return []

    try:
        X = vectorizer.transform([text])
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        active = X.toarray()[0]
        active_indices = np.flatnonzero(active)
        weights, method = _baseline_weight_vector(model)
        if weights is None or len(active_indices) == 0:
            return []

        scores = active[active_indices] * weights[active_indices]
        order = np.argsort(np.abs(scores))[::-1][:top_n]
    except Exception:
        return []

    rows = []
    for index in order:
        score = float(scores[index])
        rows.append(
            {
                "Model": model_label,
                "Term": str(feature_names[active_indices[index]]),
                "Direction": "Suspicious wording" if score > 0 else "Legitimate wording",
                "Strength": round(abs(score), 4),
                "Method": method,
            }
        )
    return rows


def _clean_vocabulary_token(value: str) -> str:
    return value.strip(".,!?;:()[]{}\"'`<>").casefold()


def _baseline_signal_map(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    signals: dict[str, dict[str, object]] = {}
    for row in rows:
        term = str(row.get("Term", "")).strip()
        if not term:
            continue

        try:
            strength = float(row.get("Strength", 0.0))
        except Exception:
            strength = 0.0

        signal = {
            "model": str(row.get("Model", "Baseline model")),
            "direction": str(row.get("Direction", "Vocabulary signal")),
            "strength": strength,
        }
        candidate_tokens = {_clean_vocabulary_token(term)}
        candidate_tokens.update(_clean_vocabulary_token(part) for part in term.split())
        for token in candidate_tokens:
            if not token:
                continue
            current = signals.get(token)
            if current is None or strength > float(current.get("strength", 0.0)):
                signals[token] = signal
    return signals


def _baseline_vocabulary_highlight_html(text: str, rows: list[dict[str, object]]) -> str:
    signals = _baseline_signal_map(rows)
    if not signals:
        return html.escape(text).replace("\n", "<br>")

    html_tokens = []
    for token in re.split(r"(\s+)", text):
        if not token:
            continue
        if token.isspace():
            html_tokens.append("<br>" if "\n" in token else " ")
            continue

        clean = _clean_vocabulary_token(token)
        escaped = html.escape(token)
        signal = signals.get(clean)
        if not signal:
            html_tokens.append(escaped)
            continue

        direction = str(signal.get("direction", "Vocabulary signal"))
        is_suspicious = direction.casefold().startswith("suspicious")
        style = ";".join(
            [
                "background:rgba(245,158,11,0.24)" if is_suspicious else "background:rgba(34,197,94,0.18)",
                "border-bottom:2px solid #F59E0B" if is_suspicious else "border-bottom:2px solid #22C55E",
                "padding:1px 4px",
                "border-radius:4px",
                "line-height:1.75",
            ]
        )
        title = html.escape(
            f"{signal.get('model')}: {direction}, strength {float(signal.get('strength', 0.0)):.4f}",
            quote=True,
        )
        html_tokens.append(f'<span title="{title}" style="{style}">{escaped}</span>')

    return (
        '<div style="font-size:0.88rem;line-height:1.75;padding:1rem;'
        'border:1px solid rgba(148,163,184,0.18);border-radius:8px;'
        'background:rgba(15,23,42,0.32);color:var(--text-secondary);">'
        f'{"".join(html_tokens)}</div>'
    )


def _transcript_metric_values(root: Path, model_key: str) -> dict[str, object]:
    metrics = _load_transcript_metrics(str(root))
    model_metrics = metrics.get("models", {})
    metric_name = _transcript_metrics_name(root, model_key)
    if isinstance(model_metrics, dict) and metric_name:
        values = model_metrics.get(metric_name, {})
        return dict(values) if isinstance(values, dict) else {}
    return {}


def _transcript_metrics_dataframe(root: Path) -> pd.DataFrame:
    metrics = _load_transcript_metrics(str(root))
    model_metrics = metrics.get("models", {})
    if not isinstance(model_metrics, dict):
        return pd.DataFrame()

    supported_metric_names = {
        _transcript_metrics_name(root, model_key)
        for model_key in TRANSCRIPT_MODEL_FILES
    }
    rows = []
    for model_name, values in model_metrics.items():
        if not isinstance(values, dict):
            continue
        if model_name not in supported_metric_names:
            continue
        cm = values.get("confusion_matrix", [[0, 0], [0, 0]])
        try:
            tn, fp = cm[0]
            fn, tp = cm[1]
        except Exception:
            tn = values.get("true_negative", 0)
            fp = values.get("false_positive", 0)
            fn = values.get("false_negative", 0)
            tp = values.get("true_positive", 0)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(float(values.get("accuracy", 0)) * 100, 2),
                "Precision": round(float(values.get("precision", 0)) * 100, 2),
                "Recall": round(float(values.get("recall", 0)) * 100, 2),
                "F1 Score": round(float(values.get("f1", 0)) * 100, 2),
                "ROC-AUC": round(float(values.get("roc_auc", 0)) * 100, 2)
                if "roc_auc" in values
                else None,
                "Training Time (s)": _training_time_value(values),
                "Prediction Time (ms)": _prediction_time_value(values),
                "True Positive": int(tp),
                "False Positive": int(fp),
                "True Negative": int(tn),
                "False Negative": int(fn),
            }
        )

    return pd.DataFrame(rows)


def _training_metrics_chart(metrics_df: pd.DataFrame) -> go.Figure:
    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    if "ROC-AUC" in metrics_df.columns and metrics_df["ROC-AUC"].notna().any():
        metric_columns.append("ROC-AUC")

    fig = go.Figure()
    for metric in metric_columns:
        fig.add_trace(go.Bar(x=metrics_df["Model"], y=metrics_df[metric], name=metric))

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=80),
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        barmode="group",
    )
    return apply_chart_theme(fig)


def _confusion_matrix_figure(metrics: dict[str, object], model_name: str) -> go.Figure | None:
    model_metrics = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    values = model_metrics.get(model_name, {}) if isinstance(model_metrics, dict) else {}
    matrix = values.get("confusion_matrix") if isinstance(values, dict) else None
    if not matrix:
        return None

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted legitimate", "Predicted suspicious"],
            y=["Actual legitimate", "Actual suspicious"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Purples",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Prediction",
        yaxis_title="Actual label",
    )
    return apply_chart_theme(fig)


def _roc_auc_curve(root: Path, model_keys: list[str]) -> go.Figure | None:
    metrics = _load_transcript_metrics(str(root))
    model_metrics = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    if not isinstance(model_metrics, dict):
        return None

    fig = go.Figure()
    has_curve = False
    seen_models: set[str] = set()
    for model_key in model_keys:
        metrics_name = _transcript_metrics_name(root, model_key)
        if not metrics_name or metrics_name in seen_models:
            continue
        seen_models.add(metrics_name)
        values = model_metrics.get(metrics_name)
        if not isinstance(values, dict):
            continue

        curve = values.get("roc_curve")
        if not isinstance(curve, dict):
            continue
        fpr = curve.get("fpr")
        tpr = curve.get("tpr")
        roc_auc = values.get("roc_auc")
        if not fpr or not tpr:
            continue

        has_curve = True
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{metrics_name} AUC={float(roc_auc):.3f}"
                if roc_auc is not None
                else metrics_name,
            )
        )

    if not has_curve:
        return None

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=40),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return apply_chart_theme(fig)


def _recommended_transcript_model(
    df_compare: pd.DataFrame,
    metrics: dict[str, object],
) -> str:
    model_metrics = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    if not df_compare.empty and isinstance(model_metrics, dict) and model_metrics:
        selected = set()
        if "Metrics Model" in df_compare.columns:
            selected = set(df_compare["Metrics Model"].dropna().astype(str).tolist())
        candidate_rows = [
            (name, float(values.get("f1", 0)))
            for name, values in model_metrics.items()
            if name in selected and isinstance(values, dict)
        ]
        if candidate_rows:
            return max(candidate_rows, key=lambda item: item[1])[0]

    for key in ("recommended_model", "top_validation_model", "best_model"):
        model_name = str(metrics.get(key, "")).strip() if isinstance(metrics, dict) else ""
        supported_metric_names = {
            _transcript_metrics_name(Path("."), model_key)
            for model_key in TRANSCRIPT_MODEL_FILES
        }
        if model_name and model_name in supported_metric_names:
            return model_name

    if not df_compare.empty:
        return str(df_compare.sort_values("Confidence", ascending=False).iloc[0]["Model"])

    return "Not available"


def _consensus_result_from_comparison(
    comparison_rows: list[dict[str, object]],
    *,
    final_verdict: str,
    average_risk: float,
    suspicious_count: int,
    total_models: int,
) -> dict[str, object]:
    first_result = dict(comparison_rows[0].get("result", {})) if comparison_rows else {}
    findings = list(first_result.get("findings", []))
    label = _label_from_verdict(final_verdict)
    suspicious_probability = max(0.0, min(1.0, average_risk / 100.0))
    confidence = suspicious_probability if label == 1 else 1.0 - suspicious_probability
    model_votes = ", ".join(
        f"{row.get('Model')}: {row.get('Prediction')} ({float(row.get('Risk Score', 0.0)):.1f}%)"
        for row in comparison_rows
    )
    model_evidence = [
        {
            "Model": str(row.get("Model", "")),
            "Prediction": str(row.get("Prediction", "")),
            "Suspicious Risk": f"{float(row.get('Risk Score', 0.0)):.1f}%",
            "Confidence": f"{float(row.get('Confidence', 0.0)):.1f}%",
        }
        for row in comparison_rows
    ]

    return {
        "label": label,
        "label_name": final_verdict,
        "confidence": confidence,
        "probabilities": {
            "Legitimate": 1.0 - suspicious_probability,
            "Suspicious": suspicious_probability,
        },
        "model_name": f"Transcript model consensus ({suspicious_count}/{total_models} suspicious)",
        "findings": findings,
        "is_consensus": True,
        "model_votes": model_votes,
        "model_evidence": model_evidence,
        "model_agreement": f"{suspicious_count}/{total_models}",
    }


def _representative_comparison_row(
    comparison_rows: list[dict[str, object]],
    *,
    final_verdict: str,
    average_risk: float,
) -> dict[str, object]:
    if not comparison_rows:
        return {}

    verdict_is_suspicious = _is_suspicious_prediction(final_verdict)
    matching_rows = [
        row
        for row in comparison_rows
        if _is_suspicious_prediction(row.get("Prediction", "")) == verdict_is_suspicious
    ]
    candidates = matching_rows or comparison_rows
    return min(
        candidates,
        key=lambda row: abs(float(row.get("Risk Score", 0.0)) - average_risk),
    )


def _render_transcript_evaluation_evidence(
    root: Path,
    metrics_df: pd.DataFrame,
    metrics: dict[str, object],
    recommended_model: str,
    model_keys: list[str],
) -> None:
    render_section_header(
        "Evaluation evidence",
        "Review saved transcript training metrics separately from this live prediction.",
        "Evaluation evidence",
    )
    render_content_card_open("violet")
    metrics_tab, confusion_tab, roc_tab = st.tabs(
        ["Performance Metrics", "Confusion Matrix Heatmap", "ROC-AUC Curve"]
    )

    with metrics_tab:
        if metrics_df.empty:
            st.warning("No saved transcript training metrics found. Run the transcript training script first.")
        else:
            st.plotly_chart(_training_metrics_chart(metrics_df), use_container_width=True)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with confusion_tab:
        figure = _confusion_matrix_figure(metrics, recommended_model)
        if figure is None:
            st.info("No confusion matrix is saved for the recommended transcript model yet.")
        else:
            st.caption(f"Confusion matrix shown for recommended model: {recommended_model}")
            st.plotly_chart(figure, use_container_width=True)

    with roc_tab:
        figure = _roc_auc_curve(root, model_keys)
        if figure is None:
            st.warning("ROC-AUC data is not available yet. Retrain the transcript models to refresh metrics.")
        else:
            st.plotly_chart(figure, use_container_width=True)
            st.caption(
                "ROC-AUC shows how well each transcript model separates legitimate and suspicious transcripts. "
                "A curve closer to the top-left corner indicates stronger classification performance."
            )

    render_content_card_close()


def _comparison_chart(rows: list[dict[str, object]]) -> go.Figure:
    labels = [str(row.get("Model", "-")) for row in rows]
    risks = [float(row.get("Risk Score", 0.0)) for row in rows]
    colors = ["#DC2626" if risk >= 70 else "#D97706" if risk >= 40 else "#22C55E" for risk in risks]
    fig = go.Figure(
        go.Bar(
            x=risks,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Risk %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(220, 48 * max(1, len(rows))),
        margin=dict(l=10, r=10, t=15, b=30),
        xaxis_title="Suspicious risk (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
    )
    return apply_chart_theme(fig)


def _render_transcript_model_comparison(
    root: Path,
    comparison_rows: list[dict[str, object]],
) -> tuple[dict[str, object] | None, object | None]:
    if not comparison_rows:
        return None, None

    df_compare = pd.DataFrame(comparison_rows)
    display_df = df_compare.drop(
        columns=["result", "classifier", "Metrics Model", "Model Key"],
        errors="ignore",
    )
    suspicious_count = int(df_compare["Prediction"].apply(_is_suspicious_prediction).sum())
    total_models = len(df_compare)
    average_risk = float(df_compare["Risk Score"].mean())
    highest_confidence = float(df_compare["Confidence"].max())
    if suspicious_count > (total_models / 2):
        final_verdict = "Suspicious"
    elif suspicious_count == (total_models / 2):
        final_verdict = "Suspicious" if average_risk >= 50 else "Legitimate"
    else:
        final_verdict = "Legitimate"

    metrics = _load_transcript_metrics(str(root))
    metrics_df = _transcript_metrics_dataframe(root)
    recommended_model = _recommended_transcript_model(df_compare, metrics)
    representative_row = _representative_comparison_row(
        comparison_rows,
        final_verdict=final_verdict,
        average_risk=average_risk,
    )
    consensus_result = _consensus_result_from_comparison(
        comparison_rows,
        final_verdict=final_verdict,
        average_risk=average_risk,
        suspicious_count=suspicious_count,
        total_models=total_models,
    )

    render_analysis_ready("Transcript model comparison complete - results ready below")
    render_section_header(
        "Transcript model agreement",
        "Compare each selected transcript model before trusting a single score.",
        "Multi-model result",
    )
    render_content_card_open("violet")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Final Verdict", final_verdict)
    col2.metric("Average Risk", f"{average_risk:.2f}%")
    col3.metric("Model Agreement", f"{suspicious_count}/{total_models}")
    col4.metric("Recommended Model", recommended_model)
    col5.metric("Highest Confidence", f"{highest_confidence:.2f}%")
    st.caption(
        "The verdict uses selected model agreement and average suspicious-risk probability. "
        "Recommended Model is chosen from saved training metrics and does not override the live consensus."
    )
    render_content_card_close()

    render_section_header(
        "AI model comparison",
        "Risk score is suspicious probability; confidence is the selected model's predicted-class confidence.",
        "Model evidence",
    )
    render_content_card_open("violet")
    st.plotly_chart(_comparison_chart(comparison_rows), use_container_width=True)
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.caption(
        "Higher agreement between independent models generally increases confidence. "
        "If models disagree, use the transcript, rule evidence, and source context before acting."
    )
    render_content_card_close()

    _render_transcript_evaluation_evidence(
        root,
        metrics_df,
        metrics,
        recommended_model,
        [str(row.get("Model Key", "")) for row in comparison_rows],
    )

    return consensus_result, representative_row.get("classifier")


def _source_labels_from_text(text: str) -> list[str]:
    labels = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            labels.append(stripped[1 : stripped.index("]")])
    return labels or ["Transcript text"]


def _student_actions(risk_score: float) -> list[str]:
    if risk_score >= 70:
        return [
            "Pause the conversation before replying.",
            "Verify through the official app, website, campus office, or published phone number.",
            "Do not share OTPs, passwords, bank details, recovery codes, or payment proof.",
            "Save screenshots, audio, phone numbers, links, timestamps, and account names.",
            "Ask a trusted person or campus support before paying or continuing.",
        ]
    if risk_score >= 40:
        return [
            "Slow down and ask for written confirmation through an official channel.",
            "Check links, sender identity, payment destination, and unusual urgency.",
            "Avoid sending sensitive information until the request is independently verified.",
            "Keep the evidence in case the pattern escalates.",
        ]
    return [
        "Continue normal caution for unexpected requests.",
        "Use official channels for payments, credentials, and account changes.",
        "Keep evidence if the conversation later becomes urgent, secretive, or payment-focused.",
    ]


def _render_student_ctas(risk_score: float, *, title: str = "Student action checklist") -> None:
    actions = _student_actions(risk_score)
    tone = st.error if risk_score >= 70 else st.warning if risk_score >= 40 else st.success
    if title:
        st.caption(title)
    tone(
        "Recommended response: "
        + ("pause and verify before acting." if risk_score >= 40 else "keep normal verification habits.")
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Step": index, "Action": action}
                for index, action in enumerate(actions, start=1)
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _render_score_flow(result: dict[str, object], text: str, findings: list[dict[str, object]]) -> None:
    risk_score = _risk_score(result)
    sources = _source_labels_from_text(text)
    source_label = ", ".join(sources[:3])
    if len(sources) > 3:
        source_label += f", +{len(sources) - 3} more"
    rule_signal = f"{len(findings)} warning pattern(s)"
    if not findings:
        rule_signal = "No rule warning patterns"

    labels = [
        f"Source: {source_label}",
        f"Transcript: {len(text.split())} word(s)",
        f"Model: {risk_score:.1f}% suspicious risk",
        f"Rules: {rule_signal}",
        "Action: verify before acting" if risk_score >= 40 else "Action: continue cautious review",
    ]
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=18,
                line=dict(color="rgba(148,163,184,.35)", width=1),
                label=labels,
                color=["#7C3AED", "#2563EB", "#D97706", "#DC2626", "#059669"],
            ),
            link=dict(
                source=[0, 1, 2, 3],
                target=[1, 2, 3, 4],
                value=[1, 1, 1, 1],
                color=["rgba(124,58,237,.20)", "rgba(37,99,235,.20)", "rgba(217,119,6,.20)", "rgba(220,38,38,.20)"],
            ),
        )
    )
    fig.update_layout(height=255, margin=dict(l=10, r=10, t=10, b=10), font_size=11)
    st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

    rows = [
        {
            "Stage": "Source",
            "Evidence": source_label,
            "Student meaning": "Where the text came from before analysis.",
        },
        {
            "Stage": "Transcript",
            "Evidence": f"{len(text.split())} word(s)",
            "Student meaning": "The exact words the model and rules inspected.",
        },
        {
            "Stage": "Model signal",
            "Evidence": f"{result.get('model_name', 'Transcript model')} | {risk_score:.1f}% suspicious risk",
            "Student meaning": "The ML probability that the wording resembles scam transcripts.",
        },
        {
            "Stage": "Rule signal",
            "Evidence": rule_signal,
            "Student meaning": "Human-readable warning patterns such as urgency, OTP, payment, secrecy, or impersonation.",
        },
        {
            "Stage": "Action",
            "Evidence": _student_actions(risk_score)[0],
            "Student meaning": "The first practical step to take before responding.",
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _record(history: list[dict[str, object]], result: dict[str, object], text: str) -> None:
    risk_score = _risk_score(result)
    findings = list(result.get("findings", []))
    record_history_item(
        history,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Transcript",
            "prediction": result["label_name"],
            "confidence": round(risk_score, 2),
            "risk_score": round(risk_score, 2),
            "model": result["model_name"],
            "preview": text.replace("\n", " ")[:160],
            "raw_input": text,
            "flags": [str(item.get("phrase", "")) for item in findings if item.get("phrase")],
            "explanation": (
                f"Transcript suspicious-risk probability: {risk_score:.1f}%. "
                f"Stored as risk so dashboard/report history does not treat legitimate confidence as threat."
            ),
        },
    )


def _transcript_result_summary(
    result: dict[str, object],
    label: str,
    confidence: float,
    findings: list[dict[str, object]],
) -> str:
    risk_score = _risk_score(result)
    count = len(findings)
    agreement = str(result.get("model_agreement", "")).strip()
    agreement_text = f" and {agreement} model agreement" if agreement else ""

    if bool(result.get("is_consensus")):
        if _is_suspicious_prediction(label):
            if count:
                return (
                    f"Selected models classified this as suspicious with {risk_score:.1f}% average suspicious risk"
                    f"{agreement_text}. It also found {count} explicit phrase-rule warning pattern(s)."
                )
            return (
                f"Selected models classified this as suspicious with {risk_score:.1f}% average suspicious risk"
                f"{agreement_text}, but no explicit phrase-rule warning was found. "
                "Treat this as a model-only warning and review the comparison table before acting."
            )
        return (
            f"Selected models classified this as lower risk with {100 - risk_score:.1f}% average legitimate confidence"
            f"{agreement_text}. Still verify identity and links before acting."
        )

    if _is_suspicious_prediction(label) and count == 0:
        percent = round(confidence * 100, 1)
        return (
            f"The model classified this as suspicious with {percent}% confidence, but no explicit phrase-rule warning "
            "was found. This is a statistical model signal, not a matched scam phrase."
        )

    return educational_summary(label, confidence, findings)


def _render_rule_evidence(result: dict[str, object], text: str, findings: list[dict[str, object]]) -> None:
    risk_score = _risk_score(result)
    render_section_header("Rule evidence", eyebrow="Explainability")
    if findings:
        render_content_card_open("red")
        st.markdown(highlighted_html(text, findings), unsafe_allow_html=True)
        findings_df = pd.DataFrame(findings)
        preferred_columns = [
            column
            for column in ["phrase", "category", "label", "specific_tactic", "reason", "intention"]
            if column in findings_df.columns
        ]
        st.dataframe(
            findings_df[preferred_columns] if preferred_columns else findings_df,
            hide_index=True,
            use_container_width=True,
        )
        render_content_card_close()
        return

    rows = [
        {
            "Evidence Layer": "Direct scam phrase rules",
            "Result": "No matched rule",
            "Student Meaning": "No explicit OTP, payment, threat, secrecy, impersonation, or urgent-action phrase was found.",
            "How To Read It": (
                "Supports lower risk, but does not override model evidence."
                if risk_score < 40
                else "Treat as a model-only warning and verify with the comparison table."
            ),
        }
    ]
    tone = st.success if risk_score < 40 else st.warning if _is_suspicious_prediction(result.get("label_name", "")) else st.info
    tone(
        "No explicit scam-rule pattern matched. This is useful evidence: the warning, if any, is coming from model similarity rather than a direct scam phrase."
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_model_agreement_evidence(result: dict[str, object]) -> None:
    model_evidence = result.get("model_evidence", [])
    if not isinstance(model_evidence, list) or not model_evidence:
        return

    render_section_header(
        "Model agreement explained",
        "Shows which models agreed and how strongly they leaned suspicious.",
        "Classifier signal",
    )
    render_content_card_open("violet")
    st.dataframe(pd.DataFrame(model_evidence), hide_index=True, use_container_width=True)
    st.caption(
        "DistilBERT is the recommended model from current training metrics. SVM and Naive Bayes are kept as transparent baselines."
    )
    render_content_card_close()


def _render_training_similarity_evidence(root: Path, text: str) -> None:
    examples = _nearest_training_examples(root, text)
    if examples.empty:
        return

    render_section_header(
        "Closest training examples",
        "These examples explain what the current transcript statistically resembles in the corrected training data.",
        "Training evidence",
    )
    render_content_card_open("green")
    st.dataframe(examples, hide_index=True, use_container_width=True)
    st.caption(
        "Similarity is a TF-IDF lookup for explanation only. It does not replace DistilBERT's prediction, but it makes the training-data comparison visible."
    )
    render_content_card_close()


def _render_baseline_vocabulary_evidence(root: Path, text: str) -> None:
    rows: list[dict[str, object]] = []
    for model_key in ("svm", "nb"):
        classifier = _load_transcript_classifier_safe(str(root), model_key)
        if classifier is None:
            continue
        rows.extend(
            _baseline_vocabulary_terms(
                text,
                classifier,
                _transcript_model_label(root, model_key),
                top_n=6,
            )
        )

    if not rows:
        return

    render_section_header(
        "Baseline vocabulary signals",
        "Transparent SVM and Naive Bayes terms that pushed the baseline models up or down.",
        "Vocabulary evidence",
    )
    render_content_card_open("green")
    st.markdown(_baseline_vocabulary_highlight_html(text, rows), unsafe_allow_html=True)
    st.caption(
        "Highlighted words come from transparent baseline models: amber leans suspicious, green leans legitimate. Hover a highlight for model and strength."
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "These are not DistilBERT word weights. They are supporting vocabulary signals from the transparent baseline models."
    )
    render_content_card_close()


def _display_result(
    root: Path,
    result: dict[str, object],
    text: str,
    classifier: object | None,
) -> None:
    confidence = float(result["confidence"])
    label = str(result["label_name"])
    findings = list(result.get("findings", []))

    probabilities = dict(result["probabilities"])
    risk_score = _risk_score(result)
    render_analysis_ready("Transcript analysis complete - results ready below")
    render_result_card(
        f"{label} transcript result",
        risk_score,
        _transcript_result_summary(result, label, confidence, findings),
    )

    render_section_header(
        "Why This Score Happened",
        "Follow the evidence from source text to model probability, rule indicators, and student action.",
        "Student view",
    )
    _render_score_flow(result, text, findings)
    _render_student_ctas(risk_score)

    render_content_card_open("violet")
    st.plotly_chart(_confidence_chart(result["probabilities"]), use_container_width=True)
    render_content_card_close()

    _render_rule_evidence(result, text, findings)
    _render_model_agreement_evidence(result)
    _render_training_similarity_evidence(root, text)
    _render_baseline_vocabulary_evidence(root, text)



def _similarity_percent(left: str, right: str) -> float:
    """Return rough text similarity for comparing Whisper text with supplied transcript."""

    left = " ".join(left.lower().split())
    right = " ".join(right.lower().split())
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio() * 100


def _render_combined_input_summary(
    *,
    use_uploaded_audio: bool,
    use_text: bool,
    uploaded_audio_text: str,
    transcript_text: str,
    uploaded_audio_results: list[dict[str, object]],
) -> None:
    """Show a compact summary of which sources are available before analysis."""

    upload_peak = max((float(item.get("risk", 0)) for item in uploaded_audio_results), default=0.0)
    upload_chunks = len(uploaded_audio_results)
    transcript_words = len(transcript_text.split())
    upload_words = len(uploaded_audio_text.split())

    rows = []
    if use_uploaded_audio:
        rows.append(
            {
                "Source": "Uploaded audio recording",
                "Status": "Ready" if uploaded_audio_results else "Waiting for upload analysis",
                "Usable text": f"{upload_words} word(s)" if uploaded_audio_text else "No transcript text yet",
                "Audio chunks": upload_chunks,
                "Peak voice risk": f"{upload_peak:.1f}%" if uploaded_audio_results else "-",
            }
        )
    if use_text:
        rows.append(
            {
                "Source": "Uploaded / pasted transcript",
                "Status": "Ready" if transcript_text.strip() else "Waiting for text",
                "Usable text": f"{transcript_words} word(s)" if transcript_text.strip() else "No text yet",
                "Audio chunks": "-",
                "Peak voice risk": "-",
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if use_uploaded_audio and use_text and uploaded_audio_text.strip() and transcript_text.strip():
        similarity = _similarity_percent(uploaded_audio_text, transcript_text)
        st.info(
            f"Audio-to-transcript similarity: {similarity:.1f}%. "
            "Use this as a rough check only; different wording, missing punctuation, or Whisper errors can lower the score."
        )


def _render_analysis_outputs(
    *,
    root: Path,
    history: list[dict[str, object]],
    transcript_model_keys: list[str],
    use_uploaded_audio: bool,
    use_text: bool,
    uploaded_audio_text: str,
    transcript_text: str,
    uploaded_audio_results: list[dict[str, object]],
    risk_threshold: int,
) -> None:
    """Render outputs for uploaded audio, transcript text, or both."""

    has_uploaded_audio_results = bool(uploaded_audio_results)
    has_uploaded_audio_text = bool(uploaded_audio_text.strip())
    has_transcript_text = bool(transcript_text.strip())

    if use_uploaded_audio and has_uploaded_audio_results:
        _render_recording_carousel(
            uploaded_audio_results,
            risk_threshold,
            state_key="transcript_uploaded_audio_carousel_index",
            title="Uploaded audio analysis",
            transcript_heading="Uploaded audio transcript and chunks",
            frequency_heading="Uploaded audio frequency spectrum",
            latest_title="Latest uploaded audio chunk {chunk}",
        )

    if use_uploaded_audio and not has_uploaded_audio_results:
        st.warning("Uploaded audio was selected, but no analysed audio file is available yet.")

    # Decide what text should be passed into transcript scam classification.
    # If both are supplied, keep the texts labelled and combined so the user can see both sources.
    text_blocks = []
    if use_uploaded_audio and has_uploaded_audio_text:
        text_blocks.append("[Uploaded audio transcript]\n" + uploaded_audio_text.strip())
    if use_text and has_transcript_text:
        text_blocks.append("[Uploaded / pasted transcript]\n" + transcript_text.strip())

    combined_text = "\n\n".join(text_blocks).strip()

    if not combined_text:
        if use_uploaded_audio and has_uploaded_audio_results:
            st.info(
                "Audio was analysed for voice authenticity and behavioral signals, but no speech transcript was available. "
                "Whisper may be unavailable, the sample may be too quiet, or no speech was detected."
            )
            return
        st.warning("No usable transcript text was available for transcript scam analysis.")
        return

    render_section_header(
        "Combined transcript analysis",
        "Transcript scam detection uses the available voice transcript, uploaded text, or both together.",
        "Unified result",
    )

    comparison_rows: list[dict[str, object]] = []
    for model_key in transcript_model_keys:
        try:
            result, classifier = _predict(
                root,
                combined_text,
                transcript_model_key=model_key,
            )
        except FileNotFoundError:
            continue
        metrics = _transcript_metric_values(root, model_key)
        comparison_rows.append(
            {
                "Model": _transcript_model_label(root, model_key),
                "Model Key": model_key,
                "Metrics Model": _transcript_metrics_name(root, model_key),
                "Prediction": result["label_name"],
                "Risk Score": round(_risk_score(result), 2),
                "Confidence": round(float(result["confidence"]) * 100, 2),
                "Accuracy": round(float(metrics.get("accuracy", 0.0)) * 100, 2) if metrics else None,
                "Precision": round(float(metrics.get("precision", 0.0)) * 100, 2) if metrics else None,
                "Recall": round(float(metrics.get("recall", 0.0)) * 100, 2) if metrics else None,
                "F1 Score": round(float(metrics.get("f1", 0.0)) * 100, 2) if metrics else None,
                "ROC-AUC": round(float(metrics.get("roc_auc", 0.0)) * 100, 2) if metrics else None,
                "Training Time (s)": _training_time_value(metrics) if metrics else None,
                "Prediction Time (ms)": _prediction_time_value(metrics) if metrics else None,
                "Engine": result["model_name"],
                "result": result,
                "classifier": classifier,
            }
        )

    if not comparison_rows:
        result, classifier = _predict(
            root,
            combined_text,
            transcript_model_key="nb",
        )
    else:
        result, classifier = _render_transcript_model_comparison(root, comparison_rows)
        if result is None:
            result = comparison_rows[0]["result"]
            classifier = comparison_rows[0]["classifier"]

    _record(history, result, combined_text)
    _display_result(root, result, combined_text, classifier)


def _inject_transcript_input_css() -> None:
    """Purple transcript-input workflow styles scoped to this tab."""
    st.markdown(
        """
        <style>
        :root {
            --transcript-accent:#A78BFA;
            --transcript-accent-strong:#8B5CF6;
            --transcript-accent-soft:rgba(167,139,250,.14);
            --transcript-border:rgba(167,139,250,.28);
            --transcript-glow:0 0 22px rgba(167,139,250,.14);
        }

        .st-key-transcript_investigation_shell
        > div[data-testid="stVerticalBlockBorderWrapper"] {
            border:1px solid rgba(167,139,250,.24)!important;
            border-radius:18px!important;
            padding:1rem!important;
            background:
                radial-gradient(circle at 90% 6%,rgba(167,139,250,.09),transparent 20rem),
                linear-gradient(145deg,rgba(17,24,39,.98),rgba(10,18,33,.98))!important;
            box-shadow:0 16px 38px rgba(0,0,0,.22),var(--transcript-glow)!important;
            overflow:hidden!important;
        }

        .transcript-step-head {
            display:flex;
            align-items:flex-start;
            gap:.65rem;
            margin:.05rem 0 .62rem;
        }

        .transcript-step-number {
            width:25px;
            height:25px;
            flex:0 0 25px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:50%;
            color:#EDE9FE;
            background:rgba(139,92,246,.18);
            border:1px solid rgba(167,139,250,.62);
            box-shadow:0 0 15px rgba(167,139,250,.15);
            font-family:'JetBrains Mono',monospace;
            font-size:.65rem;
            font-weight:850;
        }

        .transcript-step-copy strong {
            display:block;
            color:#F8FAFC;
            font-size:.88rem;
            font-weight:850;
            line-height:1.25;
        }

        .transcript-step-copy span {
            display:block;
            margin-top:2px;
            color:#7F8DA6;
            font-size:.7rem;
            line-height:1.45;
        }

        .transcript-step-divider {
            height:1px;
            margin:.8rem 0;
            background:rgba(148,163,184,.11);
        }

        .transcript-source-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.55rem;
            margin-bottom:.2rem;
        }

        .transcript-source-card {
            min-height:68px;
            padding:.7rem .78rem;
            border:1px solid rgba(167,139,250,.16);
            border-radius:12px;
            background:rgba(15,23,42,.28);
        }

        .transcript-source-card-title {
            display:flex;
            align-items:center;
            gap:.45rem;
            color:#EDE9FE;
            font-size:.72rem;
            font-weight:750;
        }

        .transcript-source-card-title::before {
            content:"";
            width:18px;
            height:18px;
            border-radius:6px;
            background:
                url("https://api.iconify.design/solar/checklist-minimalistic-bold-duotone.svg?color=%23a78bfa")
                center/13px 13px no-repeat,
                rgba(139,92,246,.12);
            border:1px solid rgba(167,139,250,.18);
        }

        /* =========================================================
           TRANSCRIPT SOURCE CARDS - SYMMETRICAL PURPLE DESIGN
           ========================================================= */

        .st-key-transcript_use_uploaded_audio_card,
        .st-key-transcript_use_text_card {
            width:100%!important;
            height:74px!important;
            min-height:74px!important;
            margin:0!important;
            padding:.7rem .78rem!important;
            box-sizing:border-box!important;
            border:1px solid rgba(167,139,250,.26)!important;
            border-radius:12px!important;
            background:
                radial-gradient(
                    circle at 8% 50%,
                    rgba(167,139,250,.12),
                    transparent 5rem
                ),
                linear-gradient(
                    145deg,
                    rgba(17,24,39,.97),
                    rgba(11,18,32,.97)
                )!important;
            box-shadow:inset 0 1px 0 rgba(255,255,255,.025)!important;
            overflow:hidden!important;
        }

        .st-key-transcript_use_uploaded_audio_card
        > div[data-testid="stHorizontalBlock"],
        .st-key-transcript_use_text_card
        > div[data-testid="stHorizontalBlock"] {
            width:100%!important;
            height:100%!important;
            min-height:0!important;
            margin:0!important;
            padding:0!important;
            gap:.45rem!important;
            align-items:center!important;
        }

        .st-key-transcript_use_uploaded_audio_card
        [data-testid="stHorizontalBlock"],
        .st-key-transcript_use_text_card
        [data-testid="stHorizontalBlock"] {
            width:100%!important;
            height:100%!important;
            margin:0!important;
            padding:0!important;
            align-items:center!important;
        }

        .st-key-transcript_use_uploaded_audio_card
        [data-testid="stElementContainer"],
        .st-key-transcript_use_text_card
        [data-testid="stElementContainer"] {
            margin:0!important;
            padding:0!important;
        }

        .st-key-transcript_use_uploaded_audio_card
        [data-testid="column"],
        .st-key-transcript_use_text_card
        [data-testid="column"] {
            min-width:0!important;
            height:100%!important;
            display:flex!important;
            align-items:center!important;
        }

        .transcript-source-icon {
            width:36px;
            height:36px;
            flex:0 0 36px;
            position:relative;
            border-radius:11px;
            background:rgba(167,139,250,.14);
            border:1px solid rgba(167,139,250,.28);
        }

        .transcript-source-icon::before {
            content:"";
            position:absolute;
            inset:8px;
            background:#A78BFA;
            -webkit-mask:var(--source-icon) center / contain no-repeat;
            mask:var(--source-icon) center / contain no-repeat;
        }

        .transcript-source-copy {
            width:100%;
            min-width:0;
            height:42px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            gap:.13rem;
            margin:0;
            padding:0;
        }

        .transcript-source-copy strong {
            display:block;
            margin:0;
            padding:0;
            color:#F8FAFC;
            font-size:.68rem;
            font-weight:800;
            line-height:1.25;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }

        .transcript-source-copy span {
            display:block;
            margin:0;
            padding:0;
            color:#8995AA;
            font-size:.54rem;
            line-height:1.3;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }

        .st-key-transcript_use_uploaded_audio_card
        [data-testid="column"]:last-child,
        .st-key-transcript_use_text_card
        [data-testid="column"]:last-child {
            justify-content:flex-end!important;
        }

        .st-key-transcript_use_uploaded_audio_card
        [data-testid="stToggle"],
        .st-key-transcript_use_text_card
        [data-testid="stToggle"] {
            width:100%!important;
            display:flex!important;
            align-items:center!important;
            justify-content:flex-end!important;
            margin:0!important;
            padding:0!important;
        }

        .st-key-transcript_use_uploaded_audio_card
        [data-testid="stToggle"] > div,
        .st-key-transcript_use_text_card
        [data-testid="stToggle"] > div {
            width:100%!important;
            display:flex!important;
            justify-content:flex-end!important;
        }

        .st-key-transcript_use_uploaded_audio_card [role="switch"],
        .st-key-transcript_use_text_card [role="switch"] {
            margin-left:auto!important;
            transform:scale(.82);
            transform-origin:right center;
        }

        .st-key-transcript_use_uploaded_audio_card:hover,
        .st-key-transcript_use_text_card:hover {
            border-color:rgba(167,139,250,.62)!important;
            box-shadow:
                0 0 20px rgba(167,139,250,.10),
                inset 0 1px 0 rgba(255,255,255,.035)!important;
        }

        .st-key-transcript_use_uploaded_audio_card:has(input:disabled) {
            opacity:.46!important;
            filter:saturate(.62);
        }

        .st-key-transcript_use_uploaded_audio_card:has(input:disabled):hover {
            border-color:rgba(167,139,250,.20)!important;
            box-shadow:none!important;
        }

        @media(max-width:760px) {
            .st-key-transcript_use_uploaded_audio_card,
            .st-key-transcript_use_text_card {
                height:78px!important;
                min-height:78px!important;
            }
        }

        .transcript-session-label {
            margin:.75rem 0 .35rem;
            color:#A78BFA;
            font-size:.66rem;
            font-weight:850;
            letter-spacing:.07em;
            text-transform:uppercase;
        }

        .transcript-subcard {
            height:100%;
            padding:.8rem;
            border:1px solid rgba(167,139,250,.15);
            border-radius:14px;
            background:
                radial-gradient(circle at 92% 8%,rgba(167,139,250,.08),transparent 9rem),
                rgba(15,23,42,.30);
        }

        .transcript-subcard-title {
            margin:0 0 .18rem;
            color:#F8FAFC;
            font-size:.8rem;
            font-weight:850;
        }

        .transcript-subcard-copy {
            margin:0 0 .58rem;
            color:#7F8DA6;
            font-size:.64rem;
            line-height:1.45;
        }

        .st-key-transcript_investigation_shell [data-testid="stCheckbox"],
        .st-key-transcript_investigation_shell [data-testid="stToggle"] {
            accent-color:#A78BFA!important;
        }

        .st-key-transcript_investigation_shell [role="checkbox"][aria-checked="true"],
        .st-key-transcript_investigation_shell [role="switch"][aria-checked="true"] {
            background:#A78BFA!important;
            border-color:#A78BFA!important;
        }

        .st-key-transcript_investigation_shell [data-testid="stSlider"] [role="slider"] {
            background:#A78BFA!important;
        }

        .st-key-transcript_investigation_shell
        [data-testid="stFileUploaderDropzone"] {
            border:1px dashed rgba(167,139,250,.42)!important;
            background:rgba(15,23,42,.26)!important;
        }

        .st-key-transcript_investigation_shell
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color:#A78BFA!important;
            background:rgba(167,139,250,.06)!important;
        }

        .st-key-transcript_text_upload {
            transition:opacity .25s ease;
        }

        .st-key-transcript_text_upload:has(input:disabled),
        .st-key-transcript_text_upload:has(button:disabled) {
            opacity:.45;
            filter:grayscale(.15);
        }

        .st-key-transcript_text_upload:has(input:disabled)
        [data-testid="stFileUploaderDropzone"] {
            cursor:not-allowed!important;
            border-color:rgba(167,139,250,.18)!important;
        }

        .st-key-transcript_investigation_shell textarea:disabled {
            opacity:.55!important;
            cursor:not-allowed!important;
        }

        .st-key-transcript_investigation_shell
        [data-testid="stFileUploaderDropzone"] button,
        .st-key-transcript_analyze_selected_sources button {
            background:linear-gradient(135deg,#8B5CF6,#A78BFA)!important;
            color:#fff!important;
            border:none!important;
            box-shadow:0 10px 24px rgba(139,92,246,.18)!important;
        }

        .transcript-review-strip {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:.45rem;
            margin:.65rem 0 .55rem;
        }

        .transcript-review-item {
            padding:.5rem .6rem;
            border:1px solid rgba(167,139,250,.13);
            border-radius:10px;
            background:rgba(15,23,42,.24);
        }

        .transcript-review-item span {
            display:block;
            color:#7F8DA6;
            font-size:.55rem;
            text-transform:uppercase;
            letter-spacing:.05em;
        }

        .transcript-review-item b {
            display:block;
            margin-top:.12rem;
            color:#F8FAFC;
            font-size:.7rem;
        }

        @media(max-width:850px) {
            .transcript-source-grid {
                grid-template-columns:1fr;
            }

            .transcript-review-strip {
                grid-template-columns:repeat(2,minmax(0,1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _transcript_step_header(number: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="transcript-step-head">
            <span class="transcript-step-number">{number}</span>
            <div class="transcript-step-copy">
                <strong>{title}</strong>
                <span>{description}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_source_choice(
    *,
    title: str,
    description: str,
    icon_url: str,
    state_key: str,
    disabled: bool = False,
    on_change=None,
) -> bool:
    with st.container(key=f"{state_key}_card"):
        icon_col, copy_col, toggle_col = st.columns(
            [0.13, 0.72, 0.15],
            gap="small",
            vertical_alignment="center",
        )

        with icon_col:
            st.markdown(
                f"""
                <div
                    class="transcript-source-icon"
                    style="--source-icon:url('{html.escape(icon_url)}')"
                    aria-hidden="true"
                ></div>
                """,
                unsafe_allow_html=True,
            )

        with copy_col:
            st.markdown(
                f"""
                <div class="transcript-source-copy">
                    <strong>{html.escape(title)}</strong>
                    <span>{html.escape(description)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with toggle_col:
            return st.toggle(
                title,
                key=state_key,
                disabled=disabled,
                on_change=on_change,
                label_visibility="collapsed",
            )


def render_transcript_tab(root: Path, history: list[dict[str, object]]) -> None:
    _init_transcript_voice_state()
    _inject_transcript_input_css()

    render_detection_tool_intro(
        title="Voice Transcript",
        description=(
            "Upload an audio recording, generate speech-to-text, upload or paste a transcript, "
            "or combine both sources for transcript scam analysis."
        ),
        icon="solar:microphone-3-bold-duotone",
        accent="purple",
    )

    uploaded: str | pd.DataFrame | None = None
    text = ""
    risk_threshold = 70

    with st.container(key="transcript_investigation_shell", border=True):
        _transcript_step_header(
            "01",
            "Choose Evidence Sources",
            "Select uploaded audio, transcript text, or both. Browser microphone recording has been removed.",
        )

        source_a, source_b = st.columns(
            2,
            gap="small",
            vertical_alignment="top",
        )
        with source_a:
            use_uploaded_audio = _render_source_choice(
                title="Uploaded Audio Recording",
                description="Upload WAV, MP3, or FLAC audio evidence.",
                icon_url="https://api.iconify.design/solar/soundwave-bold-duotone.svg",
                state_key="transcript_use_uploaded_audio",
            )
        with source_b:
            use_text = _render_source_choice(
                title="Uploaded or Pasted Transcript",
                description="Upload TXT or CSV, or paste transcript text.",
                icon_url="https://api.iconify.design/solar/document-text-bold-duotone.svg",
                state_key="transcript_use_text",
            )

        if not use_uploaded_audio and not use_text:
            st.warning("Select at least one evidence source.")

        st.markdown('<div class="transcript-step-divider"></div>', unsafe_allow_html=True)

        _transcript_step_header(
            "02",
            "Configure Audio Investigation",
            "Choose the transcript model, alert sensitivity, and Whisper settings used for recorded and uploaded audio.",
        )

        transcript_model_options = _available_transcript_models(root)
        transcript_model_keys = _default_transcript_model_keys(transcript_model_options)
        if transcript_model_options:
            transcript_model_select_key = "transcript_text_model_key"
            selected_model_keys = st.session_state.get(transcript_model_select_key)
            if not isinstance(selected_model_keys, list):
                selected_model_keys = _default_transcript_model_keys(transcript_model_options)
            had_removed_best_model = "best" in selected_model_keys
            selected_model_keys = [
                key for key in selected_model_keys if key in transcript_model_options
            ] or _default_transcript_model_keys(transcript_model_options)
            if had_removed_best_model:
                selected_model_keys = _default_transcript_model_keys(transcript_model_options)
            st.session_state[transcript_model_select_key] = selected_model_keys
            transcript_model_keys = st.multiselect(
                "Transcript scam models",
                transcript_model_options,
                key=transcript_model_select_key,
                format_func=lambda value: _transcript_model_label(root, value),
                help=(
                    "Enable multiple trained model families to compare agreement. "
                    "The recommended model is shown from saved training metrics, not as a separate runtime artifact."
                ),
            )
            if not transcript_model_keys:
                st.warning("Select at least one transcript model.")
                transcript_model_keys = _default_transcript_model_keys(transcript_model_options)

        primary_transcript_model_key = transcript_model_keys[0]

        chunk_seconds = 5
        transcript_source = "Audio only"
        whisper_size = "base.en"
        whisper_language: str | None = "en"
        whisper_task = "transcribe"
        manual_transcript = ""

        if use_uploaded_audio:
            settings_a, settings_b, settings_c, settings_d = st.columns(4, gap="small")
            chunk_key = "transcript_uploaded_audio_chunk_seconds"
            chunk_max = 30
            chunk_default = 15
            try:
                current_chunk_value = int(st.session_state.get(chunk_key, chunk_default))
            except (TypeError, ValueError):
                current_chunk_value = chunk_default
            if current_chunk_value > chunk_max:
                st.session_state[chunk_key] = chunk_default

            with settings_a:
                chunk_seconds = st.slider(
                    "Chunk length",
                    min_value=3,
                    max_value=chunk_max,
                    value=chunk_default,
                    key=chunk_key,
                    help=(
                        "Uploaded recordings can use longer chunks for better Whisper context. "
                        "Microphone recording stays shorter for faster feedback."
                    ),
                )

            with settings_b:
                risk_threshold = st.slider(
                    "Alert threshold",
                    min_value=40,
                    max_value=90,
                    value=70,
                    step=5,
                    key="transcript_uploaded_audio_risk_threshold",
                )

            with settings_c:
                whisper_options = _available_whisper_models()
                default_whisper = _default_whisper_model(whisper_options)
                whisper_key = "transcript_uploaded_audio_whisper_size"
                if st.session_state.get(whisper_key) not in whisper_options:
                    st.session_state[whisper_key] = default_whisper
                whisper_size = st.selectbox(
                    "Whisper model",
                    whisper_options,
                    key=whisper_key,
                    format_func=_whisper_model_label,
                    help=(
                        "English models are faster and less likely to drift into unrelated "
                        "languages. Larger models improve accuracy but need more CPU/RAM. "
                        "Requires requirements-local.txt; hosted cloud falls back to audio-only analysis if unavailable."
                    ),
                )

            with settings_d:
                force_english = st.checkbox(
                    "Force English",
                    value=True,
                    key="transcript_force_english_whisper",
                    help="Prevents multilingual Whisper from auto-switching to Chinese, Korean, or random gibberish on noisy chunks.",
                )
                whisper_language = "en" if force_english else None

            transcript_source = "Local Whisper"
            st.caption("Whisper loads only after Analyze. Hosted cloud can run audio-only if local Whisper is not installed.")
            if chunk_seconds < 12:
                st.info(
                    "Uploaded audio works best with 12-30 second chunks because Whisper gets more speech context."
                )

            with st.container(border=True):
                    st.markdown(
                        '<div class="transcript-subcard-title">Uploaded Audio Recording</div>'
                        '<div class="transcript-subcard-copy">Upload WAV, MP3, or FLAC evidence from an existing call.</div>',
                        unsafe_allow_html=True,
                    )

                    if use_uploaded_audio:
                        uploaded_audio = st.file_uploader(
                            "Upload audio recording",
                            type=["wav", "mp3", "flac"],
                            key="transcript_audio_upload",
                            label_visibility="collapsed",
                        )

                        if uploaded_audio is not None:
                            uploaded_audio_bytes = uploaded_audio.getvalue()
                            uploaded_audio_suffix = Path(uploaded_audio.name).suffix.lower()
                            file_signature = hashlib.sha256(
                                uploaded_audio.name.encode("utf-8")
                                + uploaded_audio_bytes
                            ).hexdigest()
                            previous_signature = st.session_state.get(
                                "transcript_uploaded_audio_file_signature"
                            )

                            if file_signature != previous_signature:
                                _clear_uploaded_audio_state(clear_file=False)
                                st.session_state[
                                    "transcript_uploaded_audio_file_name"
                                ] = uploaded_audio.name
                                st.session_state[
                                    "transcript_uploaded_audio_file_bytes"
                                ] = uploaded_audio_bytes
                                st.session_state[
                                    "transcript_uploaded_audio_file_suffix"
                                ] = uploaded_audio_suffix
                                st.session_state[
                                    "transcript_uploaded_audio_file_signature"
                                ] = file_signature

                            mime_type = {
                                ".wav": "audio/wav",
                                ".mp3": "audio/mpeg",
                                ".flac": "audio/flac",
                            }.get(uploaded_audio_suffix, "audio/wav")

                            st.audio(uploaded_audio_bytes, format=mime_type)
                            st.caption(
                                f"{uploaded_audio.name} - "
                                f"{len(uploaded_audio_bytes) / 1024:.1f} KB - "
                                "Waiting for Analyze Selected Evidence"
                            )
                        elif st.session_state.get("transcript_uploaded_audio_file_name"):
                            _clear_uploaded_audio_state(clear_file=True)

                        if st.session_state.get("transcript_uploaded_audio_error"):
                            st.error(
                                "Uploaded audio analysis failed: "
                                f"{st.session_state['transcript_uploaded_audio_error']}"
                            )
                    else:
                        st.info("Uploaded audio is not selected.")
        else:
            st.info("Audio investigation is disabled because no audio source is selected.")

        st.markdown('<div class="transcript-step-divider"></div>', unsafe_allow_html=True)

        _transcript_step_header(
            "03",
            "Review Transcript Text",
            "Upload TXT or CSV evidence, paste transcript text, or leave this step disabled for audio-only analysis.",
        )

        transcript_enabled = bool(
            st.session_state.get("transcript_use_text", False)
        )
        transcript_left, transcript_right = st.columns([0.34, 0.66], gap="small")

        with transcript_left:
            uploaded_file = st.file_uploader(
                "Upload transcript TXT or CSV",
                type=["txt", "csv"],
                key="transcript_text_upload",
                disabled=not transcript_enabled,
            )
            uploaded = _read_upload(uploaded_file) if transcript_enabled else None

        with transcript_right:
            if isinstance(uploaded, str):
                st.session_state["transcript_text_preview"] = uploaded

            text = st.text_area(
                "Transcript preview",
                height=260,
                placeholder=(
                    "Paste a call, Zoom, Teams, or Google Meet transcript here."
                    if transcript_enabled
                    else "Transcript input is disabled. Enable 'Uploaded or Pasted Transcript' above."
                ),
                disabled=not transcript_enabled,
                key="transcript_text_preview",
            )

        st.markdown('<div class="transcript-step-divider"></div>', unsafe_allow_html=True)

        with st.form("transcript_analysis_form", clear_on_submit=False):
            _transcript_step_header(
                "04",
                "Confirm and Analyze",
                "Review source readiness, then run the selected transcript and audio investigations.",
            )

            uploaded_audio_results = st.session_state.get(
                "transcript_uploaded_audio_results",
                [],
            )
            if not isinstance(uploaded_audio_results, list):
                uploaded_audio_results = []

            uploaded_audio_text_preview = _uploaded_audio_transcript_text()

            _render_combined_input_summary(
                use_uploaded_audio=use_uploaded_audio,
                use_text=use_text,
                uploaded_audio_text=uploaded_audio_text_preview,
                transcript_text=text,
                uploaded_audio_results=uploaded_audio_results,
            )

            ready_sources = sum(
                [
                    bool(
                        use_uploaded_audio
                        and st.session_state.get("transcript_uploaded_audio_file_bytes")
                    ),
                    bool(use_text and text.strip()),
                ]
            )

            st.markdown(
                '<div class="transcript-review-strip">'
                f'<div class="transcript-review-item"><span>Sources Selected</span><b>{sum([use_uploaded_audio, use_text])}</b></div>'
                f'<div class="transcript-review-item"><span>Sources Ready</span><b>{ready_sources}</b></div>'
                f'<div class="transcript-review-item"><span>Transcript Words</span><b>{len(text.split()) if use_text else 0}</b></div>'
                f'<div class="transcript-review-item"><span>Alert Threshold</span><b>{risk_threshold}%</b></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            uploaded_audio_ready = bool(
                use_uploaded_audio
                and st.session_state.get("transcript_uploaded_audio_file_bytes")
            )
            text_ready = bool(use_text and text.strip())

            analyze_button = st.form_submit_button(
                "* Analyze Selected Evidence",
                type="primary",
                use_container_width=True,
                disabled=not (uploaded_audio_ready or text_ready),
            )

    if isinstance(uploaded, pd.DataFrame) and use_text:
        render_section_header("Batch transcript CSV analysis", eyebrow="Multiple rows")
        render_content_card_open("violet")
        text_column = st.selectbox("Transcript column", uploaded.columns)

        if st.button("Analyze transcript CSV rows", use_container_width=True):
            texts = uploaded[text_column].fillna("").astype(str).tolist()
            rows = []
            try:
                for selected_model_key in transcript_model_keys:
                    classifier = _load_transcript_classifier(str(root), selected_model_key)
                    batch = classifier.predict_many(texts)
                    for row in batch.to_dict("records"):
                        rows.append(
                            {
                                "model": _transcript_model_label(root, selected_model_key),
                                **row,
                            }
                        )
                results = pd.DataFrame(rows)
            except FileNotFoundError:
                for value in texts:
                    demo = rule_based_text_prediction(value)
                    rows.append(
                        {
                            "model": "Demo rules",
                            "preview": value[:120],
                            "prediction": demo["label_name"],
                            "confidence": round(float(demo["confidence"]) * 100, 2),
                        }
                    )
                results = pd.DataFrame(rows)
                st.warning(
                    "Transcript model artifacts were not found, so batch results use demo rules."
                )

            st.dataframe(results, hide_index=True, use_container_width=True)

        render_content_card_close()

    if analyze_button:
        if use_uploaded_audio:
            current_signature = st.session_state.get(
                "transcript_uploaded_audio_file_signature"
            )
            upload_settings = json.dumps(
                [
                    chunk_seconds,
                    transcript_source,
                    whisper_size,
                    whisper_language,
                    whisper_task,
                    primary_transcript_model_key,
                ],
                ensure_ascii=True,
            ).encode("utf-8")
            current_analysis_signature = hashlib.sha256(
                str(current_signature).encode("utf-8") + upload_settings
            ).hexdigest()
            processed_signature = st.session_state.get(
                "transcript_uploaded_audio_last_processed_signature"
            )

            if not current_signature:
                st.warning("Upload an audio recording before running the analysis.")
                return

            if current_analysis_signature != processed_signature:
                with st.spinner("Analyzing uploaded audio..."):
                    try:
                        _analyse_selected_uploaded_audio(
                            root,
                            chunk_seconds=chunk_seconds,
                            whisper_size=whisper_size,
                            whisper_language=whisper_language,
                            whisper_task=whisper_task,
                            transcript_model_key=primary_transcript_model_key,
                            analysis_signature=current_analysis_signature,
                        )
                    except Exception as exc:
                        st.session_state["transcript_uploaded_audio_error"] = str(exc)
                        st.error(f"Uploaded audio analysis failed: {exc}")
                        return
                whisper_notice = st.session_state.pop("transcript_uploaded_audio_whisper_notice", None)
                if whisper_notice:
                    st.info(str(whisper_notice))

        uploaded_audio_results = (
            list(st.session_state.get("transcript_uploaded_audio_results", []))
            if use_uploaded_audio
            else []
        )

        uploaded_audio_text = (
            _uploaded_audio_transcript_text()
            if use_uploaded_audio
            else ""
        )

        _render_analysis_outputs(
            root=root,
            history=history,
            transcript_model_keys=transcript_model_keys,
            use_uploaded_audio=use_uploaded_audio,
            use_text=use_text,
            uploaded_audio_text=uploaded_audio_text,
            transcript_text=text,
            uploaded_audio_results=uploaded_audio_results,
            risk_threshold=risk_threshold,
        )

        st.session_state["transcript_pending_uploaded_audio_analysis"] = False
