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
    find_legitimate_indicators,
    find_suspicious_phrases,
    highlighted_html,
    type_intention,
)
from src.reporting.evidence_snapshot import (
    IMMEDIATE_ACTION,
    build_evidence_bundle,
    chart_artifact,
    derive_action_status,
    provenance_record,
    remediation_plan,
    table_artifact,
    text_artifact,
    xai_record,
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
def _load_voice_evidence_calibrator(root: str):
    try:
        from src.audio.voice_evidence_calibrator import load_voice_evidence_calibrator

        return load_voice_evidence_calibrator(Path(root) / "models" / "audio_voice_evidence_calibrator.pkl")
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
def _load_whisper_model(model_size: str) -> tuple[Any | None, str]:
    try:
        import whisper
    except Exception as exc:
        return (
            None,
            f"Whisper import failed: {type(exc).__name__}: {exc}",
        )

    try:
        return whisper.load_model(model_size), ""
    except Exception as exc:
        return (
            None,
            f"Whisper model '{model_size}' failed to load: {type(exc).__name__}: {exc}",
        )


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


def _is_lfs_pointer(path: Path) -> bool:
    try:
        prefix = path.read_bytes()[:40]
    except OSError:
        return False
    return prefix == b"version https://git-lfs.github.com/spec/"


def _is_transcript_model_available(root: Path, model_key: str) -> bool:
    model_path = _transcript_model_path(root, model_key)
    if not model_path.exists():
        return False
    if model_key in TRANSCRIPT_TRANSFORMER_MODEL_KEYS:
        weights_path = model_path / "model.safetensors"
        return weights_path.exists() and not _is_lfs_pointer(weights_path)
    return True


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
    if _is_transcript_model_available(root, model_key):
        return model_path, f"Transcript {label}"

    for fallback_key, (fallback_label, fallback_filename) in TRANSCRIPT_MODEL_FILES.items():
        fallback_path = _transcript_model_path(root, fallback_key)
        if _is_transcript_model_available(root, fallback_key):
            return fallback_path, f"Transcript {fallback_label}"

    return root / "models" / "transcript_nb.pkl", "Transcript Naive Bayes"


def _available_transcript_models(root: Path) -> list[str]:
    options = [
        key
        for key, (_label, filename) in TRANSCRIPT_MODEL_FILES.items()
        if _is_transcript_model_available(root, key)
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
    for model_name in ("tiny.en", "tiny", "base.en", "base"):
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
        "transcript_uploaded_audio_file_sha256": None,
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
        st.session_state["transcript_uploaded_audio_file_sha256"] = None


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
    voice_evidence_calibrator: Any | None,
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
            voice_evidence_calibrator=voice_evidence_calibrator,
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
    voice_evidence_calibrator: Any | None,
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
            voice_evidence_calibrator=voice_evidence_calibrator,
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
    audio_sha256 = st.session_state.get("transcript_uploaded_audio_file_sha256")

    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise RuntimeError("Upload an audio recording before running the analysis.")

    if suffix not in {".wav", ".mp3", ".flac"}:
        raise RuntimeError("The selected uploaded audio must be WAV, MP3, or FLAC.")

    whisper_model, whisper_error = _load_whisper_model(whisper_size)
    if whisper_model is None:
        print(f"Whisper unavailable: {whisper_error}", flush=True)
        st.session_state["transcript_uploaded_audio_whisper_notice"] = (
            "Whisper is unavailable in this environment, so uploaded audio was analysed without speech-to-text. "
            f"Details: {whisper_error}"
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
        voice_evidence_calibrator=_load_voice_evidence_calibrator(str(root)),
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
        result["source_sha256"] = audio_sha256

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
        yaxis_title="Decision risk (%)",
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


def _mfcc_explanation(results: list[dict[str, object]], latest: dict[str, object] | None) -> str:
    chunk_count = len(results)
    if not latest:
        return "MFCC evidence appears after the first processed audio chunk."
    quality = latest.get("audio_quality", {})
    if not isinstance(quality, dict):
        quality = {}
    notes = latest.get("audio_evidence_notes", [])
    note_text = ", ".join(str(note) for note in notes if str(note).strip()) if isinstance(notes, list) else ""
    conclusion = str(latest.get("authenticity_level", "voice authenticity not yet available")).lower()
    return (
        "Each heatmap column is a recent chunk, and each row is an MFCC coefficient: a compact fingerprint "
        "of vocal tone, timbre, and spectral shape used by the voice model. The heatmap supports the raw "
        "voice score, but reliability checks decide whether that pattern is strong enough to trust. "
        f"Current reading: {chunk_count} chunk(s), {float(quality.get('duration_seconds', 0.0)):.1f}s latest audio, "
        f"{conclusion}. Reliability limits: {note_text or 'none'}."
    )


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
                "Decision": str(result.get("decision_label", result.get("risk_level", "-"))),
                "Decision Risk": f"{float(result.get('risk', 0)):.1f}%",
                "Raw Blend": f"{float(result.get('raw_combined_risk', result.get('risk', 0))):.1f}%",
                "Voice Evidence": _risk_value_text(result.get("voice_evidence_risk", result.get("voice_risk"))),
                "Raw Voice AI": f"{float(result.get('voice_risk', 0)):.1f}%",
                "Transcript": f"{float(result.get('transcript_risk', 0)):.1f}%",
                "Behavioral Evidence": _risk_value_text(
                    result.get("behavioral_evidence_risk", result.get("behavioral_risk"))
                ),
                "Raw Behavioral RF": _risk_value_text(result.get("behavioral_risk")),
                "Voice Reliability": _risk_value_text(result.get("voice_reliability")),
                "Behavioral Reliability": _risk_value_text(result.get("behavioral_reliability")),
                "Content Reliability": _risk_value_text(result.get("content_reliability")),
                "Content Level": str(result.get("content_level", "-")),
                "Voice Level": str(result.get("authenticity_level", "-")),
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


def _risk_number(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip_decision_score(results: list[dict[str, object]], threshold: int) -> float:
    scores = [float(item.get("risk", 0)) for item in results]
    if not scores:
        return 0.0

    peak = max(scores)
    if peak < threshold or len(scores) < 2:
        return peak

    high_count = sum(1 for value in scores if value >= threshold)
    if high_count >= 2 or (high_count / len(scores)) >= 0.5:
        return peak

    return min(65.0, max(float(np.median(scores)), 55.0))


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


def _metric_bar_color(name: str, value: float) -> str:
    lowered = name.casefold()
    if "raw" in lowered:
        return "#7C3AED"
    if "behavioral" in lowered:
        return "#0E7490"
    if "voice" in lowered:
        return "#2563EB"
    if "content" in lowered or "transcript" in lowered:
        return "#DC2626"
    if "reliability" in lowered:
        if value >= 60:
            return "#16A34A"
        if value >= 35:
            return "#D97706"
        return "#DC2626"
    return "#0891B2"


def _percent_metric_figure(
    metrics: list[tuple[str, float | None, str]],
    *,
    height: int = 330,
) -> go.Figure | None:
    filtered = [
        (label, float(value), detail)
        for label, value, detail in metrics
        if value is not None
    ]
    if not filtered:
        return None

    labels = [item[0] for item in filtered]
    values = [max(0.0, min(100.0, item[1])) for item in filtered]
    details = [item[2] for item in filtered]
    colors = [_metric_bar_color(label, value) for label, value in zip(labels, values)]
    fig = go.Figure(
        go.Bar(
            x=list(reversed(values)),
            y=list(reversed(labels)),
            orientation="h",
            marker_color=list(reversed(colors)),
            customdata=list(reversed(details)),
            hovertemplate="%{y}<br>%{x:.1f}%<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=20, t=20, b=35),
        xaxis_title="0-100 evidence scale",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
        bargap=0.32,
    )
    return apply_chart_theme(fig)


def _voice_evidence_metric_figure(result: dict[str, object]) -> go.Figure | None:
    behavioral_evidence = result.get("behavioral_evidence_risk")
    behavioral_raw = result.get("behavioral_risk")
    behavioral_reliability = result.get("behavioral_reliability")
    metrics = [
        (
            "Decision risk",
            _risk_number(result.get("risk")),
            "Final verdict score after content, authenticity, and reliability policy.",
        ),
        (
            "Transcript scam risk",
            _risk_number(result.get("transcript_risk")),
            "Text model and rule indicators from the transcript.",
        ),
        (
            "Voice evidence risk",
            _risk_number(result.get("voice_evidence_risk")),
            f"Primary evidence score from {result.get('voice_evidence_engine', 'the available evidence path')}.",
        ),
        (
            "Rule fallback voice evidence",
            _risk_number(result.get("rule_voice_evidence_risk")),
            "Previous rule-weighted score kept for comparison and fallback.",
        ),
        (
            "Raw voice AI model score",
            _risk_number(result.get("voice_risk")),
            "MFCC + SVM probability before reliability gates.",
        ),
        (
            "Behavioral evidence risk",
            _risk_number(behavioral_evidence) if behavioral_evidence is not None else None,
            "Behavioral RF score after duration, silence, activity, and energy checks.",
        ),
        (
            "Raw behavioral RF score",
            _risk_number(behavioral_raw) if behavioral_raw is not None else None,
            "Behavioral RF probability before reliability gates.",
        ),
        (
            "Voice reliability",
            _risk_number(result.get("voice_reliability")),
            "How trustworthy the voice model is for this clip.",
        ),
        (
            "Behavioral reliability",
            _risk_number(behavioral_reliability) if behavioral_reliability is not None else None,
            "How trustworthy the behavioral RF signal is for this clip.",
        ),
        (
            "Content reliability",
            _risk_number(result.get("content_reliability")),
            "How much transcript text and rule evidence support the content verdict.",
        ),
    ]
    return _percent_metric_figure(metrics, height=360)


def _voice_evidence_explanation(result: dict[str, object]) -> str:
    raw_voice = _risk_number(result.get("voice_risk"))
    voice_evidence = _risk_number(result.get("voice_evidence_risk"))
    voice_reliability = _risk_number(result.get("voice_reliability"))
    behavioral_raw = result.get("behavioral_risk")
    behavioral_evidence = result.get("behavioral_evidence_risk")
    label = str(result.get("decision_label", "No decision"))
    engine = str(result.get("voice_evidence_engine", "the available evidence path"))
    if raw_voice >= 85.0 and voice_evidence < 40.0:
        voice_reading = "the raw voice model is high, but it is not strong evidence after reliability checks"
    elif voice_evidence >= 70.0 and voice_reliability >= 70.0:
        voice_reading = "the raw voice model and reliability gate agree, so voice authenticity can support review"
    else:
        voice_reading = "the voice signal is supporting context rather than the main verdict driver"

    behavioral_reading = "behavioral RF is unavailable"
    if behavioral_raw is not None and behavioral_evidence is not None:
        behavioral_reading = (
            f"behavioral RF moved from {_risk_value_text(behavioral_raw)} raw to "
            f"{_risk_value_text(behavioral_evidence)} evidence"
        )

    return (
        "This chart separates raw model probabilities from reliability-weighted evidence. "
        f"Primary voice evidence source: {engine}. Conclusion: {label}; {voice_reading}; {behavioral_reading}."
    )


def _normalized_signal(value: object, scale: float, *, invert: bool = False) -> float:
    number = _risk_number(value)
    score = max(0.0, min(100.0, (number / max(scale, 1e-9)) * 100.0))
    return 100.0 - score if invert else score


def _behavioral_signal_metric_figure(result: dict[str, object]) -> go.Figure | None:
    features = result.get("behavioral_features", {})
    if not isinstance(features, dict):
        return None

    quality = result.get("audio_quality", {})
    if not isinstance(quality, dict):
        quality = {}

    duration = features.get("duration_seconds", quality.get("duration_seconds"))
    speech_activity = features.get("speech_activity_ratio", quality.get("speech_activity_ratio"))
    silence = features.get("silence_ratio", quality.get("silence_ratio"))
    rms = features.get("rms_energy_mean", quality.get("rms"))
    metrics = [
        (
            "Duration coverage",
            _normalized_signal(duration, 20.0),
            f"{_risk_number(duration):.2f}s available; longer clips make RF behavior more stable.",
        ),
        (
            "Speech activity",
            _normalized_signal(speech_activity, 1.0),
            f"{_risk_number(speech_activity) * 100:.1f}% voiced frames; sparse speech weakens confidence.",
        ),
        (
            "Silence load",
            _normalized_signal(silence, 1.0),
            f"{_risk_number(silence) * 100:.1f}% silent frames; high silence is a reliability warning.",
        ),
        (
            "Recording energy",
            _normalized_signal(rms, 0.05),
            f"RMS {_risk_number(rms):.4f}; very quiet clips are less comparable to training audio.",
        ),
        (
            "Energy movement",
            _normalized_signal(features.get("rms_energy_std"), 0.04),
            f"RMS std {_risk_number(features.get('rms_energy_std')):.4f}; variation helps describe speech rhythm.",
        ),
        (
            "Zero-crossing level",
            _normalized_signal(features.get("zero_crossing_rate_mean"), 0.20),
            f"ZCR mean {_risk_number(features.get('zero_crossing_rate_mean')):.4f}; tracks noisiness and articulation.",
        ),
        (
            "Spectral centroid",
            _normalized_signal(features.get("spectral_centroid_mean"), 4_000.0),
            f"{_risk_number(features.get('spectral_centroid_mean')):.1f} Hz center of brightness.",
        ),
        (
            "Spectral bandwidth",
            _normalized_signal(features.get("spectral_bandwidth_mean"), 4_000.0),
            f"{_risk_number(features.get('spectral_bandwidth_mean')):.1f} Hz spread of frequency energy.",
        ),
        (
            "Speech rate",
            _normalized_signal(features.get("estimated_speech_rate"), 8.0),
            f"{_risk_number(features.get('estimated_speech_rate')):.2f} onset events/sec; extreme values reduce trust.",
        ),
        (
            "Pause count",
            _normalized_signal(features.get("pause_count"), 12.0),
            f"{_risk_number(features.get('pause_count')):.0f} pauses detected in the chunk.",
        ),
    ]
    return _percent_metric_figure(metrics, height=380)


def _behavioral_signal_explanation(result: dict[str, object]) -> str:
    reliability = _risk_value_text(result.get("behavioral_reliability"))
    raw = _risk_value_text(result.get("behavioral_risk"))
    evidence = _risk_value_text(result.get("behavioral_evidence_risk"))
    notes = result.get("audio_evidence_notes", [])
    note_text = ", ".join(str(note) for note in notes if str(note).strip()) if isinstance(notes, list) else ""
    return (
        "Behavioral RF looks at duration, voiced activity, silence, energy, spectral shape, pauses, and speech rate. "
        "The bars are normalized so mixed units can be compared visually; hover details keep the raw measurement. "
        f"Conclusion: RF is {raw} raw, {evidence} after {reliability} reliability. "
        f"Reliability limits: {note_text or 'none'}."
    )


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
    acoustic_metrics_placeholder,
    behavioral_metrics_placeholder,
) -> None:
    latest = results[-1] if results else None
    peak = max((float(item.get("risk", 0)) for item in results), default=0.0)
    average = (
        sum(float(item.get("risk", 0)) for item in results) / len(results)
        if results
        else 0.0
    )
    alert_count = sum(1 for item in results if float(item.get("risk", 0)) >= threshold)
    voice_peak = max(
        (
            _risk_number(item.get("voice_evidence_risk", item.get("voice_risk")))
            for item in results
        ),
        default=0.0,
    )
    behavioral_peak = max(
        (
            _risk_number(item.get("behavioral_evidence_risk", item.get("behavioral_risk")))
            for item in results
        ),
        default=0.0,
    )
    content_peak = max((float(item.get("transcript_risk", 0)) for item in results), default=0.0)

    with metrics_placeholder.container():
        render_metric_row(
            [
                {"label": "Chunks Analysed", "value": len(results), "color": "#2563EB"},
                {"label": "Current Decision", "value": f"{float(latest.get('risk', 0)):.0f}%" if latest else "0%", "color": "#D97706"},
                {"label": "Content Risk", "value": f"{content_peak:.0f}%", "color": "#DC2626"},
                {"label": "Voice Evidence", "value": f"{voice_peak:.0f}%", "color": "#7C3AED"},
                {"label": "Behavioral Evidence", "value": f"{behavioral_peak:.0f}%", "color": "#0E7490"},
                {"label": "Peak Decision", "value": f"{peak:.0f}%", "color": "#DC2626"},
                {"label": "Average Decision", "value": f"{average:.0f}%", "color": "#0891B2"},
                {"label": "Alerts", "value": alert_count, "color": "#DC2626"},
            ]
        )

    with result_placeholder.container():
        if latest:
            render_result_card(
                str(latest.get("decision_label") or latest_title.format(chunk=latest.get("clip_chunk", 1))),
                float(latest.get("risk", 0)),
                str(latest.get("explanation", "")),
            )
            if float(latest.get("risk", 0)) >= threshold:
                st.error(
                    f"Alert threshold reached. This chunk scored {float(latest.get('risk', 0)):.1f}% decision risk."
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
        st.plotly_chart(_timeline_figure(results, threshold), width="stretch")

    with transcript_placeholder.container():
        if results:
            transcript_text = _cumulative_transcript(results)
            st.text_area(
                "Live transcript",
                value=transcript_text or "No speech text yet. Enable Whisper for automatic transcription.",
                height=145,
                disabled=True,
            )
            st.dataframe(_result_table(results), hide_index=True, width="stretch")
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
            st.plotly_chart(figure, width="stretch")
            st.caption(_mfcc_explanation(results, latest))
        else:
            st.caption("MFCC heatmap appears after the first processed chunk.")

    with frequency_placeholder.container():
        frequency_figure = _frequency_figure(latest)
        if frequency_figure is not None:
            st.plotly_chart(frequency_figure, width="stretch")
        else:
            st.caption("Frequency spectrum appears after the first processed chunk.")

    with acoustic_metrics_placeholder.container():
        if not latest:
            st.caption("Acoustic feature values appear after the first processed chunk.")
        else:
            figure = _voice_evidence_metric_figure(latest)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
                st.caption(_voice_evidence_explanation(latest))
            else:
                st.caption("Voice evidence metrics appear after the first processed chunk.")

    with behavioral_metrics_placeholder.container():
        if not latest:
            st.caption("Behavioral RF metrics appear after the first processed chunk.")
        else:
            figure = _behavioral_signal_metric_figure(latest)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
                st.caption(_behavioral_signal_explanation(latest))
            else:
                st.caption("Behavioral RF metrics are unavailable for this chunk.")


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
    with display_b:
        render_section_header(frequency_heading, eyebrow="Frequency analysis")
        frequency_placeholder = st.empty()

    render_section_header(
        "MFCC feature heatmap",
        (
            "Shows the acoustic fingerprint used by the voice model. This explains what the model heard, "
            "not whether the clip is automatically fake."
        ),
        "Audio pattern",
    )
    mfcc_placeholder = st.empty()
    render_section_header(
        "Voice evidence metrics",
        (
            "Compares raw model probabilities with reliability-weighted evidence so short or sparse clips "
            "do not overrule the final verdict."
        ),
        "Voice indicators",
    )
    acoustic_metrics_placeholder = st.empty()
    render_section_header(
        "Behavioral RF signal metrics",
        (
            "Normalizes duration, silence, energy, spectral shape, and rhythm into readable signals behind "
            "the behavioral RF score."
        ),
        "Behavior indicators",
    )
    behavioral_metrics_placeholder = st.empty()

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
        acoustic_metrics_placeholder=acoustic_metrics_placeholder,
        behavioral_metrics_placeholder=behavioral_metrics_placeholder,
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
    show_navigation: bool = True,
    show_description: bool = True,
) -> None:
    groups = _recording_groups(results)
    if not groups:
        return

    current_index = int(st.session_state.get(state_key, len(groups) - 1))
    current_index = max(0, min(current_index, len(groups) - 1))
    st.session_state[state_key] = current_index
    clip_number, clip_results = groups[current_index]
    peak = _clip_decision_score(clip_results, risk_threshold)
    peak_chunk = max(float(item.get("risk", 0)) for item in clip_results)
    voice_peak = max(
        _risk_number(item.get("voice_evidence_risk", item.get("voice_risk")))
        for item in clip_results
    )
    behavioral_peak = max(
        _risk_number(item.get("behavioral_evidence_risk", item.get("behavioral_risk")))
        for item in clip_results
    )
    content_peak = max(float(item.get("transcript_risk", 0)) for item in clip_results)
    supporting_only = "supporting" in title.casefold()
    score_label = "Supporting chunk score" if supporting_only else "Decision score"
    peak_label = "Peak supporting chunk" if supporting_only else "Peak chunk"
    flags = sorted(
        {
            str(flag)
            for item in clip_results
            for flag in item.get("flags", [])
            if str(flag).strip()
        }
    )

    summary = (
        (
            f"Recording {current_index + 1} of {len(groups)} | Clip {clip_number} | "
            f"{score_label} {peak:.1f}% | {peak_label} {peak_chunk:.1f}% | "
            f"Voice evidence {voice_peak:.1f}% | Behavioral evidence {behavioral_peak:.1f}% | "
            f"Content {content_peak:.1f}%"
        )
        if show_description
        else ""
    )
    render_section_header(
        title,
        summary,
        "Recording carousel",
    )
    if show_navigation:
        nav_left, nav_mid, nav_right = st.columns([0.2, 0.6, 0.2])
        with nav_left:
            if st.button("Previous", width="stretch", disabled=current_index == 0, key=f"{state_key}_prev"):
                st.session_state[state_key] = current_index - 1
                st.rerun()
        with nav_mid:
            if show_description:
                st.markdown(
                    f"**Clip {clip_number}** | {len(clip_results)} chunk(s) | "
                    f"Flags: {', '.join(flags) if flags else 'none'}"
                )
        with nav_right:
            if st.button("Next", width="stretch", disabled=current_index >= len(groups) - 1, key=f"{state_key}_next"):
                st.session_state[state_key] = current_index + 1
                st.rerun()
    elif show_description:
        st.markdown(
            f"**Clip {clip_number}** | {len(clip_results)} chunk(s) | "
            f"Flags: {', '.join(flags) if flags else 'none'}"
        )

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


def _indicator_token_map(items: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    signals: dict[str, dict[str, object]] = {}
    for item in items:
        indicator = str(item.get("phrase", item.get("indicator", ""))).strip()
        if not indicator:
            continue
        for part in indicator.split():
            token = _clean_vocabulary_token(part)
            if token and token not in signals:
                signals[token] = item
    return signals


def _combined_transcript_evidence_html(
    text: str,
    findings: list[dict[str, object]],
    legitimate_indicators: list[dict[str, object]],
    vocabulary_rows: list[dict[str, object]],
) -> str:
    rule_signals = _indicator_token_map(findings)
    context_signals = _indicator_token_map(legitimate_indicators)
    vocabulary_signals = _baseline_signal_map(vocabulary_rows)

    html_tokens = []
    for token in re.split(r"(\s+)", text):
        if not token:
            continue
        if token.isspace():
            html_tokens.append("<br>" if "\n" in token else " ")
            continue

        clean = _clean_vocabulary_token(token)
        escaped = html.escape(token)
        tooltip_parts = []
        style_kind = ""

        if clean in rule_signals:
            signal = rule_signals[clean]
            style_kind = "rule"
            tooltip_parts.append(
                "Rule indicator: "
                f"{signal.get('category', 'Warning')} - "
                f"{signal.get('specific_tactic', signal.get('reason', ''))}"
            )

        if clean in context_signals:
            signal = context_signals[clean]
            if not style_kind:
                style_kind = "context"
            tooltip_parts.append(
                "Lower-risk context: "
                f"{signal.get('category', 'Context')} - {signal.get('reason', '')}"
            )

        if clean in vocabulary_signals:
            signal = vocabulary_signals[clean]
            direction = str(signal.get("direction", "Vocabulary signal"))
            if not style_kind:
                style_kind = "vocabulary_suspicious" if direction.casefold().startswith("suspicious") else "vocabulary_legitimate"
            tooltip_parts.append(
                f"{signal.get('model')}: {direction}, "
                f"strength {float(signal.get('strength', 0.0)):.4f}"
            )

        if not tooltip_parts:
            html_tokens.append(escaped)
            continue

        if style_kind == "rule":
            background = "rgba(239,68,68,0.24)"
            border = "#EF4444"
        elif style_kind in {"context", "vocabulary_legitimate"}:
            background = "rgba(34,197,94,0.18)"
            border = "#22C55E"
        else:
            background = "rgba(245,158,11,0.24)"
            border = "#F59E0B"

        style = ";".join(
            [
                f"background:{background}",
                f"border-bottom:2px solid {border}",
                "padding:1px 4px",
                "border-radius:4px",
                "line-height:1.75",
            ]
        )
        title = html.escape(" | ".join(tooltip_parts), quote=True)
        html_tokens.append(f'<span title="{title}" style="{style}">{escaped}</span>')

    return (
        '<div style="font-size:0.88rem;line-height:1.75;padding:1rem;'
        'border:1px solid rgba(148,163,184,0.18);border-radius:8px;'
        'background:rgba(15,23,42,0.32);color:var(--text-secondary);">'
        f'{"".join(html_tokens)}</div>'
    )


def _baseline_vocabulary_rows(root: Path, text: str, *, top_n: int = 6) -> list[dict[str, object]]:
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
                top_n=top_n,
            )
        )
    return rows


def _baseline_impact_levels(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []

    scores = []
    for row in rows:
        try:
            scores.append(abs(float(row.get("Strength", 0.0))))
        except Exception:
            scores.append(0.0)

    if not any(scores):
        return ["Low" for _row in rows]

    ordered_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    high_count = max(1, (len(scores) + 4) // 5)
    medium_end = max(high_count, (len(scores) + 1) // 2)
    labels = ["Low" for _row in rows]

    for rank, index in enumerate(ordered_indices):
        if scores[index] == 0:
            labels[index] = "Low"
        elif rank < high_count:
            labels[index] = "High"
        elif rank < medium_end:
            labels[index] = "Medium"

    return labels


def _dedupe_evidence_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped = []
    seen = set()
    for row in rows:
        indicator = str(row.get("Indicator", "")).strip().casefold()
        row_type = str(row.get("Type", "")).strip().casefold()
        if not indicator:
            continue
        key = (indicator, row_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _transcript_evidence_rows(
    findings: list[dict[str, object]],
    legitimate_indicators: list[dict[str, object]],
    vocabulary_rows: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for item in findings:
        indicator = str(item.get("phrase", "")).strip()
        if not indicator:
            continue
        category = str(item.get("category", "Rule indicator"))
        rows.append(
            {
                "Indicator": indicator,
                "Type": category,
                "Model Weight": "-",
                "Impact": "High",
                "Intention": item.get("intention", type_intention(category)),
            }
        )

    for item in legitimate_indicators:
        indicator = str(item.get("indicator", item.get("phrase", ""))).strip()
        if not indicator:
            continue
        category = str(item.get("category", item.get("Type", "Legitimate context")))
        rows.append(
            {
                "Indicator": indicator,
                "Type": category,
                "Model Weight": "-",
                "Impact": "Context",
                "Intention": item.get("intention", type_intention(category)),
            }
        )

    impact_levels = _baseline_impact_levels(vocabulary_rows)
    for index, item in enumerate(vocabulary_rows):
        indicator = str(item.get("Term", "")).strip()
        if not indicator:
            continue
        direction = str(item.get("Direction", "Vocabulary signal"))
        model_name = str(item.get("Model", "Baseline model"))
        try:
            strength = float(item.get("Strength", 0.0))
        except Exception:
            strength = 0.0
        signed_weight = strength if direction.casefold().startswith("suspicious") else -strength
        rows.append(
            {
                "Indicator": indicator,
                "Type": f"{model_name}: {direction}",
                "Model Weight": round(signed_weight, 4),
                "Impact": impact_levels[index] if index < len(impact_levels) else "Low",
                "Intention": "Shows which words pushed a transparent baseline model up or down; supporting evidence only.",
            }
        )

    return pd.DataFrame(_dedupe_evidence_rows(rows))


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
    metrics_tab, confusion_tab, roc_tab = st.tabs(
        ["Performance Metrics", "Confusion Matrix Heatmap", "ROC-AUC Curve"]
    )

    with metrics_tab:
        if metrics_df.empty:
            st.warning("No saved transcript training metrics found. Run the transcript training script first.")
        else:
            st.plotly_chart(_training_metrics_chart(metrics_df), width="stretch")
            st.dataframe(metrics_df, hide_index=True, width="stretch")

    with confusion_tab:
        figure = _confusion_matrix_figure(metrics, recommended_model)
        if figure is None:
            st.info("No confusion matrix is saved for the recommended transcript model yet.")
        else:
            st.caption(f"Confusion matrix shown for recommended model: {recommended_model}")
            st.plotly_chart(figure, width="stretch")

    with roc_tab:
        figure = _roc_auc_curve(root, model_keys)
        if figure is None:
            st.warning("ROC-AUC data is not available yet. Retrain the transcript models to refresh metrics.")
        else:
            st.plotly_chart(figure, width="stretch")
            st.caption(
                "ROC-AUC shows how well each transcript model separates legitimate and suspicious transcripts. "
                "A curve closer to the top-left corner indicates stronger classification performance."
            )


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
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Final Verdict", final_verdict)
    col2.metric("Average Risk", f"{average_risk:.2f}%")
    col3.metric("Suspicious Votes", f"{suspicious_count}/{total_models}")
    col4.metric("Recommended Model", recommended_model)
    col5.metric("Highest Confidence", f"{highest_confidence:.2f}%")
    st.caption(
        "The verdict uses selected model agreement and average suspicious-risk probability. "
        "Recommended Model is chosen from saved training metrics and does not override the live consensus."
    )

    render_section_header(
        "AI model comparison",
        "Risk score is suspicious probability; confidence is the selected model's predicted-class confidence.",
        "Model evidence",
    )
    st.plotly_chart(_comparison_chart(comparison_rows), width="stretch")
    st.dataframe(display_df, hide_index=True, width="stretch")
    st.caption(
        "Higher agreement between independent models generally increases confidence. "
        "If models disagree, use the transcript, rule evidence, and source context before acting."
    )

    _render_transcript_evaluation_evidence(
        root,
        metrics_df,
        metrics,
        recommended_model,
        [str(row.get("Model Key", "")) for row in comparison_rows],
    )

    return consensus_result, representative_row.get("classifier")


def _transcript_evidence_snapshot(
    *,
    root: Path,
    result: dict[str, object],
    text: str,
    comparison_rows: list[dict[str, object]],
    audio_results: list[dict[str, object]],
    risk_threshold: int,
) -> tuple[dict[str, object], dict[str, object], str, float, str]:
    risk_score = _risk_score(result)
    findings = list(result.get("findings", []))
    finding_labels = [
        str(item.get("phrase", ""))
        for item in findings
        if str(item.get("phrase", "")).strip()
    ]
    agreement = str(result.get("model_agreement", "")).strip()
    if not agreement and comparison_rows:
        suspicious_count = sum(
            1 for row in comparison_rows if _is_suspicious_prediction(row.get("Prediction", ""))
        )
        total = len(comparison_rows)
        agreement = f"{max(suspicious_count, total - suspicious_count)}/{total}"
    audio_peak = max((float(item.get("risk", 0.0)) for item in audio_results), default=0.0)
    concern_score = max(risk_score, audio_peak)
    action_status = derive_action_status(
        native_prediction=result.get("label_name", "Unknown"),
        evidence_type="Transcript and Audio" if audio_results else "Transcript",
        concern_score=concern_score,
        score_available=True,
        model_agreement=agreement,
        evidence_complete=bool(text.strip() or audio_results),
    )
    if audio_results and audio_peak >= max(70, risk_threshold):
        action_status = IMMEDIATE_ACTION

    artifacts: list[dict[str, object]] = []
    if audio_results:
        for clip_number, clip_results in _recording_groups(audio_results):
            latest = clip_results[-1]
            artifacts.extend(
                [
                    chart_artifact(
                        f"Recording {clip_number} - Decision Risk Timeline",
                        _timeline_figure(clip_results, risk_threshold),
                        description="Decision risk for every processed chunk in this recording.",
                        data=[
                            {
                                "clip": item.get("clip"),
                                "chunk": item.get("clip_chunk"),
                                "decision": item.get("decision_label"),
                                "risk": item.get("risk"),
                            }
                            for item in clip_results
                        ],
                    ),
                    table_artifact(
                        f"Recording {clip_number} - Chunk Investigation Table",
                        _result_table(clip_results),
                    ),
                    chart_artifact(
                        f"Recording {clip_number} - MFCC Feature Heatmap",
                        _mfcc_figure(clip_results),
                        description=_mfcc_explanation(clip_results, latest),
                    ),
                    chart_artifact(
                        f"Recording {clip_number} - Frequency Spectrum",
                        _frequency_figure(latest),
                        description="Relative frequency levels from the latest processed chunk.",
                    ),
                    chart_artifact(
                        f"Recording {clip_number} - Voice Evidence Metrics",
                        _voice_evidence_metric_figure(latest),
                        description=_voice_evidence_explanation(latest),
                    ),
                    chart_artifact(
                        f"Recording {clip_number} - Behavioral RF Signal Metrics",
                        _behavioral_signal_metric_figure(latest),
                        description=_behavioral_signal_explanation(latest),
                    ),
                    table_artifact(
                        f"Recording {clip_number} - Behavioral Feature Values",
                        _behavioral_feature_rows(latest),
                    ),
                ]
            )

    clean_comparison = [
        {
            key: value
            for key, value in row.items()
            if key not in {"result", "classifier"}
        }
        for row in comparison_rows
    ]
    if comparison_rows:
        comparison_df = pd.DataFrame(clean_comparison)
        metrics = _load_transcript_metrics(str(root))
        metrics_df = _transcript_metrics_dataframe(root)
        recommended_model = _recommended_transcript_model(comparison_df, metrics)
        model_keys = [str(row.get("Model Key", "")) for row in comparison_rows]
        artifacts.extend(
            [
                chart_artifact(
                    "Transcript Live Model Comparison",
                    _comparison_chart(comparison_rows),
                    description="Suspicious-risk probability for every selected transcript model.",
                    data=clean_comparison,
                ),
                table_artifact("Transcript Live Model Comparison Table", comparison_df),
            ]
        )
        if not metrics_df.empty:
            artifacts.extend(
                [
                    chart_artifact(
                        "Performance Metrics",
                        _training_metrics_chart(metrics_df),
                        description="Evaluation tab 1 of 3: saved transcript training metrics.",
                        data=metrics_df,
                    ),
                    table_artifact("Performance Metrics data", metrics_df),
                ]
            )
        confusion = _confusion_matrix_figure(metrics, recommended_model)
        artifacts.append(
            chart_artifact(
                "Confusion Matrix Heatmap",
                confusion,
                description=(
                    "Evaluation tab 2 of 3 for the recommended transcript model."
                    if confusion is not None
                    else "Evaluation tab 2 of 3 was unavailable."
                ),
            )
        )
        roc = _roc_auc_curve(root, model_keys)
        artifacts.append(
            chart_artifact(
                "ROC-AUC Curve",
                roc,
                description=(
                    "Evaluation tab 3 of 3 for selected transcript models."
                    if roc is not None
                    else "Evaluation tab 3 of 3 was unavailable."
                ),
            )
        )

    probabilities = dict(result.get("probabilities", {}))
    if probabilities:
        artifacts.append(
            chart_artifact(
                "Transcript Consensus Probability",
                _confidence_chart(probabilities),
                description="Legitimate and suspicious probabilities represented by the live dashboard.",
                data=probabilities,
            )
        )

    legitimate_indicators = find_legitimate_indicators(text) if text.strip() else []
    vocabulary_rows = _baseline_vocabulary_rows(root, text) if text.strip() else []
    evidence_df = _transcript_evidence_rows(findings, legitimate_indicators, vocabulary_rows)
    if not evidence_df.empty:
        artifacts.append(table_artifact("Transcript Explainability Evidence", evidence_df))
    if text.strip():
        artifacts.append(
            text_artifact(
                "Original Combined Transcript",
                text,
                description="Exact labelled transcript input submitted to the text models.",
                source="Investigation input",
            )
        )

    factors = [
        {
            "factor": item.get("phrase", ""),
            "effect": "raises concern",
            "method": "deterministic rule",
            "reason": item.get("intention") or item.get("category", ""),
        }
        for item in findings
        if item.get("phrase")
    ]
    factors.extend(
        {
            "factor": item.get("Term", ""),
            "effect": item.get("Direction", "supporting signal"),
            "strength": item.get("Strength", 0.0),
            "method": f"{item.get('Model', 'baseline')} vocabulary contribution",
        }
        for item in vocabulary_rows
        if item.get("Term")
    )
    audio_flags = sorted(
        {
            str(flag)
            for item in audio_results
            for flag in item.get("flags", [])
            if str(flag).strip()
        }
    )
    all_findings = list(dict.fromkeys(finding_labels + audio_flags))
    remediation = remediation_plan(
        evidence_type="Transcript and Audio" if audio_results else "Transcript",
        action_status=action_status,
        findings=all_findings,
    )
    source_name = (
        str(audio_results[0].get("source_filename") or "Uploaded audio")
        if audio_results
        else "Uploaded or pasted transcript"
    )
    audio_sha256 = (
        str(audio_results[0].get("source_sha256") or "")
        if audio_results
        else ""
    )
    bundle = build_evidence_bundle(
        evidence_type="Transcript and Audio" if audio_results else "Transcript",
        source_input={
            "source_name": source_name,
            "characters": len(text),
            "words": len(text.split()),
            "audio_chunks": len(audio_results),
            "audio_source_signature": (
                str(audio_results[0].get("source_signature", "")) if audio_results else ""
            ),
            "audio_sha256": audio_sha256,
            "risk_threshold": risk_threshold,
        },
        dashboard_summary={
            "final_verdict": result.get("label_name", "Unknown"),
            "transcript_suspicious_risk": round(risk_score, 2),
            "audio_peak_decision_risk": round(audio_peak, 2) if audio_results else None,
            "maximum_concern_signal": round(concern_score, 2),
            "model_agreement": agreement or "Not available",
            "model_votes": result.get("model_votes", ""),
            "action_status": action_status,
        },
        artifacts=artifacts,
        findings=all_findings,
        xai=xai_record(
            method="Direct phrase rules plus transparent baseline vocabulary and audio reliability gates",
            factors=factors,
            explanation=(
                "Rules and baseline vocabulary explain observable content signals. "
                "Audio charts separate raw model outputs from reliability-weighted evidence."
            ),
            limitations=[
                "Transformer attention is not presented as a causal explanation.",
                "No local token attribution is available for the transformer model in this snapshot.",
                "Audio detectors support review but cannot independently prove speaker identity or synthetic origin.",
            ],
        ),
        limitations=[
            "Automatic transcription can omit or mishear speech, especially in short, noisy, or multilingual clips.",
            "Model probabilities and reliability-weighted audio scores have different meanings and are labelled separately.",
        ],
        remediation=remediation,
    )
    uploaded_audio_bytes = st.session_state.get("transcript_uploaded_audio_file_bytes")
    provenance_payload: object = text if text.strip() else audio_results
    if audio_results and isinstance(uploaded_audio_bytes, bytes) and uploaded_audio_bytes:
        provenance_payload = uploaded_audio_bytes
    provenance = provenance_record(
        provenance_payload,
        source_name=source_name,
        source_kind="Transcript/audio investigation input",
        extra={
            "audio_chunks": len(audio_results),
            "audio_source_signature": (
                str(audio_results[0].get("source_signature", "")) if audio_results else ""
            ),
            "original_audio_sha256": audio_sha256,
        },
    )
    return bundle, provenance, action_status, concern_score, agreement


def _record(
    history: list[dict[str, object]],
    result: dict[str, object],
    text: str,
    *,
    evidence_bundle: dict[str, object],
    provenance: dict[str, object],
    action_status: str,
    concern_score: float,
    model_agreement: str,
    scan_type: str = "Transcript",
) -> None:
    risk_score = _risk_score(result)
    findings = list(result.get("findings", []))
    record_history_item(
        history,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": scan_type,
            "prediction": result["label_name"],
            "native_prediction": result["label_name"],
            "confidence": round(risk_score, 2),
            "risk_score": round(concern_score, 2),
            "concern_score": round(concern_score, 2),
            "score_label": "Maximum dashboard concern signal",
            "score_available": True,
            "action_status": action_status,
            "model_agreement": model_agreement,
            "model": result["model_name"],
            "source_name": str(
                dict(evidence_bundle.get("source_input", {})).get(
                    "source_name", "Uploaded or pasted transcript"
                )
            ),
            "preview": text.replace("\n", " ")[:160],
            "raw_input": (
                text
                if text
                else json.dumps(
                    {
                        "source_name": dict(evidence_bundle.get("source_input", {})).get(
                            "source_name", "Uploaded audio"
                        ),
                        "audio_sha256": dict(evidence_bundle.get("source_input", {})).get(
                            "audio_sha256", ""
                        ),
                        "audio_chunks": dict(evidence_bundle.get("source_input", {})).get(
                            "audio_chunks", 0
                        ),
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
            ),
            "flags": [
                str(item)
                for item in evidence_bundle.get("findings", [])
                if str(item).strip()
            ],
            "explanation": (
                f"Transcript suspicious-risk probability: {risk_score:.1f}%. "
                f"Maximum dashboard concern signal: {concern_score:.1f}%."
            ),
            "evidence_bundle": evidence_bundle,
            "provenance": provenance,
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


def _render_transcript_evidence_breakdown(root: Path, text: str, findings: list[dict[str, object]]) -> None:
    legitimate_indicators = find_legitimate_indicators(text)
    vocabulary_rows = _baseline_vocabulary_rows(root, text)
    evidence_df = _transcript_evidence_rows(
        findings,
        legitimate_indicators,
        vocabulary_rows,
    )

    render_section_header(
        "Transcript evidence breakdown",
        "Review direct rule matches, lower-risk context, and baseline vocabulary signals in one table.",
        "Explainability",
    )
    st.markdown(
        _combined_transcript_evidence_html(
            text,
            findings,
            legitimate_indicators,
            vocabulary_rows,
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Red highlights are direct rule matches, amber highlights are suspicious baseline vocabulary, and green highlights are lower-risk context."
    )
    if evidence_df.empty:
        st.info("No direct rules, lower-risk context indicators, or baseline vocabulary signals were found.")
    else:
        display_evidence_df = evidence_df.copy()
        display_evidence_df["Model Weight"] = display_evidence_df["Model Weight"].fillna("-").astype(str)
        st.dataframe(display_evidence_df, hide_index=True, width="stretch")
    st.caption(
        "Rule indicators are direct pattern evidence. Baseline vocabulary rows are supporting SVM/Naive Bayes signals and do not override the final verdict."
    )


def _display_result(
    root: Path,
    result: dict[str, object],
    text: str,
    classifier: object | None,
) -> None:
    confidence = float(result["confidence"])
    label = str(result["label_name"])
    findings = list(result.get("findings", []))

    risk_score = _risk_score(result)
    render_analysis_ready("Transcript analysis complete - results ready below")
    render_result_card(
        f"{label} transcript result",
        risk_score,
        _transcript_result_summary(result, label, confidence, findings),
    )

    st.plotly_chart(_confidence_chart(result["probabilities"]), width="stretch")

    _render_transcript_evidence_breakdown(root, text, findings)


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

    upload_chunks = len(uploaded_audio_results)
    transcript_words = len(transcript_text.split())
    upload_words = len(uploaded_audio_text.split())

    rows = []
    if use_uploaded_audio:
        upload_decision_peak = max((float(item.get("risk", 0)) for item in uploaded_audio_results), default=0.0)
        upload_voice_peak = max(
            (
                _risk_number(item.get("voice_evidence_risk", item.get("voice_risk")))
                for item in uploaded_audio_results
            ),
            default=0.0,
        )
        upload_behavioral_peak = max(
            (
                _risk_number(item.get("behavioral_evidence_risk", item.get("behavioral_risk")))
                for item in uploaded_audio_results
            ),
            default=0.0,
        )
        rows.append(
            {
                "Source": "Uploaded audio recording",
                "Status": "Ready" if uploaded_audio_results else "Waiting for upload analysis",
                "Usable text": f"{upload_words} word(s)" if uploaded_audio_text else "No transcript text yet",
                "Audio chunks": str(upload_chunks),
                "Peak decision risk": f"{upload_decision_peak:.1f}%" if uploaded_audio_results else "-",
                "Peak voice evidence": f"{upload_voice_peak:.1f}%" if uploaded_audio_results else "-",
                "Peak behavioral evidence": f"{upload_behavioral_peak:.1f}%" if uploaded_audio_results else "-",
            }
        )
    if use_text:
        rows.append(
            {
                "Source": "Uploaded / pasted transcript",
                "Status": "Ready" if transcript_text.strip() else "Waiting for text",
                "Usable text": f"{transcript_words} word(s)" if transcript_text.strip() else "No text yet",
                "Audio chunks": "-",
                "Peak decision risk": "-",
                "Peak voice evidence": "-",
                "Peak behavioral evidence": "-",
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

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
            _render_recording_carousel(
                uploaded_audio_results,
                risk_threshold,
                state_key="transcript_uploaded_audio_carousel_index",
                title="Uploaded audio analysis",
                transcript_heading="Uploaded audio transcript and chunks",
                frequency_heading="Uploaded audio frequency spectrum",
                latest_title="Latest uploaded audio chunk {chunk}",
                show_navigation=False,
                show_description=False,
            )
            st.info(
                "Audio was analysed for voice authenticity and behavioral signals, but no speech transcript was available. "
                "Whisper may be unavailable, the sample may be too quiet, or no speech was detected."
            )
            peak_item = max(
                uploaded_audio_results,
                key=lambda item: float(item.get("risk", 0.0)),
            )
            audio_risk = float(peak_item.get("risk", 0.0))
            audio_label = str(
                peak_item.get("decision_label")
                or peak_item.get("risk_level")
                or "Audio evidence requires review"
            )
            audio_result = {
                "label": 1 if audio_risk >= risk_threshold else 0,
                "label_name": audio_label,
                "confidence": audio_risk / 100.0,
                "probabilities": {
                    "Lower concern": max(0.0, 1.0 - audio_risk / 100.0),
                    "Suspicious": min(1.0, audio_risk / 100.0),
                },
                "model_name": "Audio authenticity and behavioral evidence pipeline",
                "findings": [
                    {"phrase": str(flag), "category": "Audio evidence"}
                    for item in uploaded_audio_results
                    for flag in item.get("flags", [])
                    if str(flag).strip()
                ],
                "model_agreement": "Not applicable",
            }
            bundle, provenance, action_status, concern_score, agreement = (
                _transcript_evidence_snapshot(
                    root=root,
                    result=audio_result,
                    text="",
                    comparison_rows=[],
                    audio_results=uploaded_audio_results,
                    risk_threshold=risk_threshold,
                )
            )
            _record(
                history,
                audio_result,
                "",
                evidence_bundle=bundle,
                provenance=provenance,
                action_status=action_status,
                concern_score=concern_score,
                model_agreement=agreement,
                scan_type="Audio",
            )
            return
        st.warning("No usable transcript text was available for transcript scam analysis.")
        return

    render_section_header(
        "Combined transcript analysis",
        (
            "This is the primary verdict when uploaded audio has usable transcript text. "
            "Voice authenticity and Behavioral RF are supporting evidence below."
        ),
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

    bundle, provenance, action_status, concern_score, agreement = (
        _transcript_evidence_snapshot(
            root=root,
            result=result,
            text=combined_text,
            comparison_rows=comparison_rows,
            audio_results=uploaded_audio_results if use_uploaded_audio else [],
            risk_threshold=risk_threshold,
        )
    )
    _record(
        history,
        result,
        combined_text,
        evidence_bundle=bundle,
        provenance=provenance,
        action_status=action_status,
        concern_score=concern_score,
        model_agreement=agreement,
    )
    _display_result(root, result, combined_text, classifier)

    if use_uploaded_audio and has_uploaded_audio_results:
        st.info(
            "The primary verdict above uses the full transcript extracted from the uploaded audio. "
            "The section below is supporting chunk-level audio evidence for voice authenticity, quality, MFCC, and Behavioral RF."
        )
        _render_recording_carousel(
            uploaded_audio_results,
            risk_threshold,
            state_key="transcript_uploaded_audio_carousel_index",
            title="Supporting audio analysis",
            transcript_heading="Supporting audio transcript and chunks",
            frequency_heading="Supporting audio frequency spectrum",
            latest_title="Supporting uploaded audio chunk {chunk}",
            show_navigation=False,
            show_description=False,
        )


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
                        "Whisper loads after Analyze; the first hosted run may download the selected model."
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
            st.caption("Whisper loads only after Analyze. The first hosted run may be slower while the selected model downloads.")
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
                            file_sha256 = hashlib.sha256(uploaded_audio_bytes).hexdigest()
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
                            st.session_state[
                                "transcript_uploaded_audio_file_sha256"
                            ] = file_sha256

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
                width="stretch",
                disabled=not (uploaded_audio_ready or text_ready),
            )

    if isinstance(uploaded, pd.DataFrame) and use_text:
        render_section_header("Batch transcript CSV analysis", eyebrow="Multiple rows")
        render_content_card_open("violet")
        text_column = st.selectbox("Transcript column", uploaded.columns)

        if st.button("Analyze transcript CSV rows", width="stretch"):
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

            st.dataframe(results, hide_index=True, width="stretch")

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
