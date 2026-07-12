"""Call and meeting transcript scam detection tab with browser voice recorder."""

from __future__ import annotations

import hashlib
import html
import json
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
from src.audio.audio_classifier import load_audio_behavior_model, load_audio_model
from src.text.explainability import (
    educational_summary,
    find_suspicious_phrases,
    highlighted_html,
    top_model_terms,
)
from src.audio.live_audio_analysis import (
    BEHAVIORAL_FEATURE_NAMES,
    analyse_live_chunk,
    transcribe_with_whisper,
    wav_bytes_to_audio,
)
from src.text.rule_demo import rule_based_text_prediction
from src.text.text_classifier import load_text_artifacts

try:
    import whisper as _whisper

    WHISPER_AVAILABLE = True
except Exception:
    _whisper = None
    WHISPER_AVAILABLE = False


@st.cache_resource(show_spinner=False)
def _load_audio_classifier(root: str):
    try:
        return load_audio_model(Path(root) / "models" / "audio_svm.pkl")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_behavioral_classifier(root: str):
    try:
        return load_audio_behavior_model(Path(root) / "models" / "audio_behavior_rf.pkl")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_transcript_classifier(root: str):
    return load_text_artifacts(
        Path(root) / "models" / "transcript_vectorizer.pkl",
        Path(root) / "models" / "transcript_nb.pkl",
        model_name="Transcript Naive Bayes",
    )


@st.cache_resource(show_spinner=False)
def _load_transcript_classifier_safe(root: str):
    try:
        return _load_transcript_classifier(root)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_whisper_model(model_size: str):
    if not WHISPER_AVAILABLE or _whisper is None:
        return None
    return _whisper.load_model(model_size)


def _init_transcript_voice_state() -> None:
    defaults: dict[str, Any] = {
        "transcript_use_voice": False,
        "transcript_use_uploaded_audio": False,
        "transcript_use_text": False,
        "transcript_text_preview": "",
        "transcript_voice_sessions": [],
        "transcript_voice_active_session_id": None,
        "transcript_voice_active_index": 0,
        "transcript_voice_mode": "record",
        "transcript_recorder_generation": 0,
        "transcript_recorder_error": "",
        "transcript_pending_voice_analysis": False,
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


def _on_voice_source_change() -> None:
    """Keep voice recording and uploaded audio mutually exclusive."""

    if st.session_state.get("transcript_use_voice", False):
        st.session_state["transcript_use_uploaded_audio"] = False


def _on_uploaded_audio_source_change() -> None:
    """Keep uploaded audio and voice recording mutually exclusive."""

    if st.session_state.get("transcript_use_uploaded_audio", False):
        st.session_state["transcript_use_voice"] = False


def _on_voice_session_selected() -> None:
    """Show selected saved recording in the shared recorder display area."""

    st.session_state["transcript_voice_mode"] = "saved"
    st.session_state["transcript_recorder_carousel_index"] = 0


def _voice_sessions() -> list[dict[str, object]]:
    sessions = st.session_state.get("transcript_voice_sessions", [])
    return sessions if isinstance(sessions, list) else []


def _active_voice_session() -> dict[str, object] | None:
    """Return the saved recording currently selected by session ID."""

    sessions = _voice_sessions()
    if not sessions:
        st.session_state["transcript_voice_active_session_id"] = None
        st.session_state["transcript_voice_active_index"] = 0
        return None

    active_id = st.session_state.get("transcript_voice_active_session_id")

    for index, session in enumerate(sessions):
        if str(session.get("session_id", "")) == str(active_id):
            st.session_state["transcript_voice_active_index"] = index
            return session

    newest_index = len(sessions) - 1
    newest_session = sessions[newest_index]

    st.session_state["transcript_voice_active_index"] = newest_index
    st.session_state["transcript_voice_active_session_id"] = str(
        newest_session.get("session_id", "")
    )

    return newest_session


def _create_voice_session(
    *,
    audio_bytes: bytes,
    signature: str,
    results: list[dict[str, object]],
) -> None:
    sessions = _voice_sessions()
    session_number = len(sessions) + 1
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    transcript_parts = [
        str(item.get("transcript", "")).strip()
        for item in results
        if str(item.get("transcript", "")).strip()
    ]

    session = {
        "session_id": datetime.now().strftime("voice_%Y%m%d_%H%M%S_%f"),
        "title": f"Recorded voice {session_number}",
        "created_at": created_at,
        "audio_bytes": audio_bytes,
        "signature": signature,
        "results": results,
        "transcript": "\n".join(transcript_parts),
    }

    sessions.append(session)
    st.session_state["transcript_voice_sessions"] = sessions
    st.session_state["transcript_voice_active_index"] = len(sessions) - 1
    st.session_state["transcript_voice_active_session_id"] = str(
        session["session_id"]
    )
    st.session_state["transcript_voice_mode"] = "saved"
    st.session_state["transcript_recorder_carousel_index"] = 0


def _delete_active_voice_session() -> None:
    sessions = _voice_sessions()
    if not sessions:
        return

    active_id = str(
        st.session_state.get("transcript_voice_active_session_id", "")
    )
    active_index = next(
        (
            index
            for index, session in enumerate(sessions)
            if str(session.get("session_id", "")) == active_id
        ),
        len(sessions) - 1,
    )

    sessions.pop(active_index)
    st.session_state["transcript_voice_sessions"] = sessions

    if sessions:
        next_index = min(active_index, len(sessions) - 1)
        next_session = sessions[next_index]
        st.session_state["transcript_voice_active_index"] = next_index
        st.session_state["transcript_voice_active_session_id"] = str(
            next_session.get("session_id", "")
        )
        st.session_state["transcript_voice_mode"] = "saved"
    else:
        st.session_state["transcript_voice_active_index"] = 0
        st.session_state["transcript_voice_active_session_id"] = None
        st.session_state["transcript_voice_mode"] = "record"
        st.session_state["transcript_recorder_generation"] = (
            int(st.session_state.get("transcript_recorder_generation", 0)) + 1
        )

    st.session_state["transcript_recorder_carousel_index"] = 0


def _clear_recorder_state() -> None:
    st.session_state["transcript_voice_sessions"] = []
    st.session_state["transcript_voice_active_index"] = 0
    st.session_state["transcript_voice_active_session_id"] = None
    st.session_state["transcript_voice_mode"] = "record"
    st.session_state["transcript_recorder_error"] = ""
    st.session_state["transcript_recorder_carousel_index"] = 0
    st.session_state["transcript_recorder_generation"] = (
        int(st.session_state.get("transcript_recorder_generation", 0)) + 1
    )


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
        if transcript_source == "Local Whisper":
            if chunk.size < sample_rate * 1.0 or float(np.abs(chunk).mean()) < 0.0005:
                transcript = ""
            else:
                transcript = transcribe_with_whisper(chunk, whisper_model)
        elif transcript_source == "Manual transcript" and index == 0:
            transcript = manual_transcript
        else:
            transcript = ""

        processed.append(
            analyse_live_chunk(
                chunk,
                transcript=transcript,
                audio_classifier=audio_classifier,
                text_classifier=text_classifier,
                behavioral_classifier=behavioral_classifier,
                sample_rate=sample_rate,
            )
        )
    return processed


def _process_recording(
    audio_bytes: bytes,
    *,
    chunk_seconds: int,
    transcript_source: str,
    manual_transcript: str,
    whisper_model: Any | None,
    audio_classifier: Any | None,
    text_classifier: Any | None,
    behavioral_classifier: Any | None,
) -> list[dict[str, object]]:
    audio, sample_rate = wav_bytes_to_audio(audio_bytes, target_sample_rate=16_000)
    return _process_audio_array(
        audio,
        sample_rate,
        chunk_seconds=chunk_seconds,
        transcript_source=transcript_source,
        manual_transcript=manual_transcript,
        whisper_model=whisper_model,
        audio_classifier=audio_classifier,
        text_classifier=text_classifier,
        behavioral_classifier=behavioral_classifier,
    )


def _process_uploaded_audio(
    audio_bytes: bytes,
    suffix: str,
    *,
    chunk_seconds: int,
    transcript_source: str,
    manual_transcript: str,
    whisper_model: Any | None,
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
) -> list[dict[str, object]]:
    """Analyze only the currently selected uploaded-audio file."""

    audio_bytes = st.session_state.get("transcript_uploaded_audio_file_bytes")
    suffix = str(st.session_state.get("transcript_uploaded_audio_file_suffix", ""))
    signature = st.session_state.get("transcript_uploaded_audio_file_signature")

    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise RuntimeError("Upload an audio recording before running the analysis.")

    if suffix not in {".wav", ".mp3", ".flac"}:
        raise RuntimeError("The selected uploaded audio must be WAV, MP3, or FLAC.")

    whisper_model = None
    if WHISPER_AVAILABLE:
        whisper_model = _load_whisper_model(whisper_size)
        if whisper_model is None:
            raise RuntimeError("The selected Whisper model could not be loaded.")

    transcript_source = "Local Whisper" if whisper_model is not None else "Audio only"

    processed = _process_uploaded_audio(
        audio_bytes,
        suffix,
        chunk_seconds=chunk_seconds,
        transcript_source=transcript_source,
        manual_transcript="",
        whisper_model=whisper_model,
        audio_classifier=_load_audio_classifier(str(root)),
        text_classifier=_load_transcript_classifier_safe(str(root)),
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
    st.session_state["transcript_uploaded_audio_last_processed_signature"] = signature
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
        rows.append(
            {
                "Clip": int(result.get("clip", 1)),
                "Chunk": int(result.get("clip_chunk", 1)),
                "Time": result.get("time", "-"),
                "Risk": f"{float(result.get('risk', 0)):.1f}%",
                "Voice": f"{float(result.get('voice_risk', 0)):.1f}%",
                "Transcript": f"{float(result.get('transcript_risk', 0)):.1f}%",
                "Behavioral": _risk_value_text(result.get("behavioral_risk")),
                "Detected text": transcript or "Audio-only chunk",
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

    _render_dashboard_section(
        clip_results,
        risk_threshold,
        transcript_heading=transcript_heading,
        frequency_heading=frequency_heading,
        latest_title=latest_title,
    )


def _recorder_transcript_text() -> str:
    """Return usable transcript text from the selected speaker voice recording."""

    session = _active_voice_session()
    if not session:
        return ""

    return str(session.get("transcript", "")).strip()


def _active_recorder_results() -> list[dict[str, object]]:
    session = _active_voice_session()
    if not session:
        return []

    results = session.get("results", [])
    return list(results) if isinstance(results, list) else []


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


def _combined_source_text(mode: str, voice_text: str, transcript_text: str) -> str:
    """Build the final text passed into the transcript classifier."""

    voice_text = voice_text.strip()
    transcript_text = transcript_text.strip()

    if mode == "Voice recording only":
        return voice_text
    if mode == "Uploaded / pasted transcript only":
        return transcript_text

    parts = []
    if voice_text:
        parts.append("[Voice recording transcript]\n" + voice_text)
    if transcript_text:
        parts.append("[Uploaded / pasted transcript]\n" + transcript_text)
    return "\n\n".join(parts).strip()


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


@st.cache_data(show_spinner=False)
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


def _predict(root: Path, text: str) -> tuple[dict[str, object], object | None]:
    try:
        classifier = _load_transcript_classifier(str(root))
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


def _record(history: list[dict[str, object]], result: dict[str, object], text: str) -> None:
    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Transcript",
            "prediction": result["label_name"],
            "confidence": round(float(result["confidence"]) * 100, 2),
            "model": result["model_name"],
            "preview": text.replace("\n", " ")[:160],
        },
    )


def _display_result(result: dict[str, object], text: str, classifier: object | None) -> None:
    confidence = float(result["confidence"])
    label = str(result["label_name"])
    findings = list(result.get("findings", []))

    probabilities = dict(result["probabilities"])
    risk_score = float(probabilities.get("Suspicious", confidence if int(result["label"]) == 1 else 1 - confidence)) * 100
    render_analysis_ready("Transcript analysis complete - results ready below")
    render_result_card(
        f"{label} transcript result",
        risk_score,
        educational_summary(label, confidence, findings),
    )

    render_content_card_open("violet")
    st.plotly_chart(_confidence_chart(result["probabilities"]), use_container_width=True)
    render_content_card_close()

    render_section_header("Suspicious transcript patterns", eyebrow="Explainability")
    if findings:
        render_content_card_open("red")
        st.markdown(highlighted_html(text, findings), unsafe_allow_html=True)
        findings_df = pd.DataFrame(findings)
        preferred_columns = [
            column
            for column in ["phrase", "category", "label", "specific_tactic", "reason", "intention"]
            if column in findings_df.columns
        ]
        st.dataframe(findings_df[preferred_columns] if preferred_columns else findings_df, hide_index=True, use_container_width=True)
        render_content_card_close()
    else:
        st.info("No suspicious phrase rules were triggered.")

    if classifier is not None:
        terms = top_model_terms(text, classifier.vectorizer, classifier.model)
        if terms:
            render_section_header("Model-influential terms", eyebrow="Classifier signal")
            render_content_card_open("green")
            st.dataframe(pd.DataFrame(terms), hide_index=True, use_container_width=True)
            render_content_card_close()



def _similarity_percent(left: str, right: str) -> float:
    """Return rough text similarity for comparing Whisper text with supplied transcript."""

    left = " ".join(left.lower().split())
    right = " ".join(right.lower().split())
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio() * 100


def _render_combined_input_summary(
    *,
    use_voice: bool,
    use_uploaded_audio: bool,
    use_text: bool,
    voice_text: str,
    uploaded_audio_text: str,
    transcript_text: str,
    recorder_results: list[dict[str, object]],
    uploaded_audio_results: list[dict[str, object]],
) -> None:
    """Show a compact summary of which sources are available before analysis."""

    voice_peak = max((float(item.get("risk", 0)) for item in recorder_results), default=0.0)
    upload_peak = max((float(item.get("risk", 0)) for item in uploaded_audio_results), default=0.0)
    voice_chunks = len(recorder_results)
    upload_chunks = len(uploaded_audio_results)
    transcript_words = len(transcript_text.split())
    voice_words = len(voice_text.split())
    upload_words = len(uploaded_audio_text.split())

    rows = []
    if use_voice:
        rows.append(
            {
                "Source": "Speaker voice recorder",
                "Status": "Ready" if recorder_results else "Waiting for recording",
                "Usable text": f"{voice_words} word(s)" if voice_text else "No transcript text yet",
                "Audio chunks": voice_chunks,
                "Peak voice risk": f"{voice_peak:.1f}%" if recorder_results else "-",
            }
        )
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

    audio_text = "\n".join(
        value for value in [voice_text.strip(), uploaded_audio_text.strip()] if value
    )
    if (use_voice or use_uploaded_audio) and use_text and audio_text and transcript_text.strip():
        similarity = _similarity_percent(audio_text, transcript_text)
        st.info(
            f"Audio-to-transcript similarity: {similarity:.1f}%. "
            "Use this as a rough check only; different wording, missing punctuation, or Whisper errors can lower the score."
        )


def _render_analysis_outputs(
    *,
    root: Path,
    history: list[dict[str, object]],
    use_voice: bool,
    use_uploaded_audio: bool,
    use_text: bool,
    voice_text: str,
    uploaded_audio_text: str,
    transcript_text: str,
    recorder_results: list[dict[str, object]],
    uploaded_audio_results: list[dict[str, object]],
    risk_threshold: int,
) -> None:
    """Render outputs for either voice, transcript, or both."""

    has_voice_results = bool(recorder_results)
    has_voice_text = bool(voice_text.strip())
    has_uploaded_audio_results = bool(uploaded_audio_results)
    has_uploaded_audio_text = bool(uploaded_audio_text.strip())
    has_transcript_text = bool(transcript_text.strip())

    if use_voice and has_voice_results:
        _render_recording_carousel(
            recorder_results,
            risk_threshold,
            state_key="transcript_recorder_carousel_index",
            title="Voice recording analysis",
            transcript_heading="Recorded transcript and chunks",
            frequency_heading="Latest frequency spectrum",
            latest_title="Latest recorded chunk {chunk}",
        )

    if use_voice and not has_voice_results:
        st.warning("Voice recording was selected, but no analysed voice sample is available yet.")

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
    if use_voice and has_voice_text:
        text_blocks.append("[Speaker voice recorder transcript]\n" + voice_text.strip())
    if use_uploaded_audio and has_uploaded_audio_text:
        text_blocks.append("[Uploaded audio transcript]\n" + uploaded_audio_text.strip())
    if use_text and has_transcript_text:
        text_blocks.append("[Uploaded / pasted transcript]\n" + transcript_text.strip())

    combined_text = "\n\n".join(text_blocks).strip()

    if not combined_text:
        if (use_voice and has_voice_results) or (use_uploaded_audio and has_uploaded_audio_results):
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

    result, classifier = _predict(root, combined_text)
    _record(history, result, combined_text)
    _display_result(result, combined_text, classifier)


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

        .st-key-transcript_use_voice_card,
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

        .st-key-transcript_use_voice_card
        > div[data-testid="stHorizontalBlock"],
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

        .st-key-transcript_use_voice_card
        [data-testid="stHorizontalBlock"],
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

        .st-key-transcript_use_voice_card
        [data-testid="stElementContainer"],
        .st-key-transcript_use_uploaded_audio_card
        [data-testid="stElementContainer"],
        .st-key-transcript_use_text_card
        [data-testid="stElementContainer"] {
            margin:0!important;
            padding:0!important;
        }

        .st-key-transcript_use_voice_card
        [data-testid="column"],
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

        .st-key-transcript_use_voice_card
        [data-testid="column"]:last-child,
        .st-key-transcript_use_uploaded_audio_card
        [data-testid="column"]:last-child,
        .st-key-transcript_use_text_card
        [data-testid="column"]:last-child {
            justify-content:flex-end!important;
        }

        .st-key-transcript_use_voice_card
        [data-testid="stToggle"],
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

        .st-key-transcript_use_voice_card
        [data-testid="stToggle"] > div,
        .st-key-transcript_use_uploaded_audio_card
        [data-testid="stToggle"] > div,
        .st-key-transcript_use_text_card
        [data-testid="stToggle"] > div {
            width:100%!important;
            display:flex!important;
            justify-content:flex-end!important;
        }

        .st-key-transcript_use_voice_card [role="switch"],
        .st-key-transcript_use_uploaded_audio_card [role="switch"],
        .st-key-transcript_use_text_card [role="switch"] {
            margin-left:auto!important;
            transform:scale(.82);
            transform-origin:right center;
        }

        .st-key-transcript_use_voice_card:hover,
        .st-key-transcript_use_uploaded_audio_card:hover,
        .st-key-transcript_use_text_card:hover {
            border-color:rgba(167,139,250,.62)!important;
            box-shadow:
                0 0 20px rgba(167,139,250,.10),
                inset 0 1px 0 rgba(255,255,255,.035)!important;
        }

        .st-key-transcript_use_voice_card:has(input:disabled),
        .st-key-transcript_use_uploaded_audio_card:has(input:disabled) {
            opacity:.46!important;
            filter:saturate(.62);
        }

        .st-key-transcript_use_voice_card:has(input:disabled):hover,
        .st-key-transcript_use_uploaded_audio_card:has(input:disabled):hover {
            border-color:rgba(167,139,250,.20)!important;
            box-shadow:none!important;
        }

        @media(max-width:760px) {
            .st-key-transcript_use_voice_card,
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


def _render_voice_session_manager() -> None:
    """Select, rename, and remove saved recordings."""

    sessions = _voice_sessions()
    if not sessions:
        return

    active_session = _active_voice_session()
    if active_session is None:
        return

    session_ids = [
        str(session.get("session_id", ""))
        for session in sessions
    ]
    active_id = str(active_session.get("session_id", ""))

    st.markdown(
        '<div class="transcript-session-label">Temporarily saved recordings</div>',
        unsafe_allow_html=True,
    )

    selected_session_id = st.selectbox(
        "Saved recording",
        options=session_ids,
        index=session_ids.index(active_id),
        format_func=lambda session_id: next(
            (
                str(session.get("title", "Recorded voice"))
                for session in sessions
                if str(session.get("session_id", "")) == session_id
            ),
            "Recorded voice",
        ),
        key="transcript_voice_active_session_id",
        on_change=_on_voice_session_selected,
        help="Choose which saved recording should be used by Analyze Selected Evidence.",
    )

    if selected_session_id != active_id:
        st.session_state["transcript_voice_active_session_id"] = selected_session_id
        st.session_state["transcript_voice_mode"] = "saved"
        st.session_state["transcript_recorder_carousel_index"] = 0
        st.rerun()

    selected_index = next(
        (
            index
            for index, session in enumerate(sessions)
            if str(session.get("session_id", "")) == str(selected_session_id)
        ),
        len(sessions) - 1,
    )
    selected_session = sessions[selected_index]
    st.session_state["transcript_voice_active_index"] = selected_index
    selected_id = str(
        selected_session.get("session_id", f"voice_{selected_index}")
    )

    new_title = st.text_input(
        "Recording title",
        value=str(
            selected_session.get(
                "title",
                f"Recorded voice {selected_index + 1}",
            )
        ),
        key=f"voice_title_{selected_id}",
        placeholder="Example: Bank verification call",
    )

    cleaned_title = new_title.strip()
    if (
        cleaned_title
        and cleaned_title != str(selected_session.get("title", ""))
    ):
        sessions[selected_index]["title"] = cleaned_title
        st.session_state["transcript_voice_sessions"] = sessions

    remove_col, information_col = st.columns(
        [0.30, 0.70],
        gap="small",
        vertical_alignment="center",
    )
    with remove_col:
        if st.button(
            "Remove recording",
            key=f"remove_voice_session_{selected_id}",
            use_container_width=True,
        ):
            _delete_active_voice_session()
            st.rerun()

    with information_col:
        st.caption(
            f"Recording {selected_index + 1} of {len(sessions)} - "
            f"{selected_session.get('created_at', '')}"
        )


def render_transcript_tab(root: Path, history: list[dict[str, object]]) -> None:
    _init_transcript_voice_state()
    _inject_transcript_input_css()

    render_detection_tool_intro(
        title="Voice Transcript",
        description=(
            "Record a speaker voice sample, upload an audio recording, upload or paste "
            "a transcript, or combine sources for transcript scam analysis."
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
            "Select one or more sources. Audio-only analysis remains available even when no transcript text exists.",
        )

        voice_enabled = bool(
            st.session_state.get("transcript_use_voice", False)
        )
        uploaded_audio_enabled = bool(
            st.session_state.get("transcript_use_uploaded_audio", False)
        )

        source_a, source_b, source_c = st.columns(
            3,
            gap="small",
            vertical_alignment="top",
        )
        with source_a:
            use_voice = _render_source_choice(
                title="Speaker Voice Recorder",
                description="Record speaker playback through the microphone.",
                icon_url="https://api.iconify.design/solar/microphone-3-bold-duotone.svg",
                state_key="transcript_use_voice",
                disabled=uploaded_audio_enabled,
                on_change=_on_voice_source_change,
            )

        with source_b:
            use_uploaded_audio = _render_source_choice(
                title="Uploaded Audio Recording",
                description="Upload WAV, MP3, or FLAC audio evidence.",
                icon_url="https://api.iconify.design/solar/soundwave-bold-duotone.svg",
                state_key="transcript_use_uploaded_audio",
                disabled=voice_enabled,
                on_change=_on_uploaded_audio_source_change,
            )
        with source_c:
            use_text = _render_source_choice(
                title="Uploaded or Pasted Transcript",
                description="Upload TXT or CSV, or paste transcript text.",
                icon_url="https://api.iconify.design/solar/document-text-bold-duotone.svg",
                state_key="transcript_use_text",
            )

        if not use_voice and not use_uploaded_audio and not use_text:
            st.warning("Select at least one evidence source.")

        st.markdown('<div class="transcript-step-divider"></div>', unsafe_allow_html=True)

        _transcript_step_header(
            "02",
            "Configure Audio Investigation",
            "Choose chunking, alert sensitivity, and the Whisper model used for recorded and uploaded audio.",
        )

        audio_classifier = _load_audio_classifier(str(root))
        behavioral_classifier = _load_behavioral_classifier(str(root))
        text_classifier = _load_transcript_classifier_safe(str(root))

        chunk_seconds = 5
        transcript_source = "Audio only"
        whisper_size = "tiny"
        manual_transcript = ""

        if use_voice or use_uploaded_audio:
            settings_a, settings_b, settings_c = st.columns(3, gap="small")

            with settings_a:
                chunk_seconds = st.slider(
                    "Chunk length",
                    min_value=3,
                    max_value=10,
                    value=5,
                    key="transcript_recorder_chunk_seconds",
                )

            with settings_b:
                risk_threshold = st.slider(
                    "Alert threshold",
                    min_value=40,
                    max_value=90,
                    value=70,
                    step=5,
                    key="transcript_recorder_risk_threshold",
                )

            with settings_c:
                whisper_size = st.selectbox(
                    "Whisper model",
                    ["tiny", "base"],
                    key="transcript_recorder_whisper_size",
                    disabled=not WHISPER_AVAILABLE,
                    help="Used for both Speaker Voice Recorder and Uploaded Audio Recording.",
                )

            transcript_source = "Local Whisper" if WHISPER_AVAILABLE else "Audio only"
            if not WHISPER_AVAILABLE:
                st.warning(
                    "Local Whisper is unavailable. Audio authenticity and behavioral analysis can still run, "
                    "but automatic transcript generation will be skipped."
                )

            audio_left, audio_right = st.columns(2, gap="small")

            with audio_left:
                with st.container(border=True):
                    st.markdown(
                        '<div class="transcript-subcard-title">Speaker Voice Recorder</div>'
                        '<div class="transcript-subcard-copy">'
                        'Record a short voice sample played from your speaker through the microphone.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    if use_voice:
                        sessions = _voice_sessions()
                        voice_mode = str(
                            st.session_state.get("transcript_voice_mode", "record")
                        )
                        active_session = _active_voice_session()

                        if voice_mode == "record" or not sessions:
                            recorded_audio = st.audio_input(
                                "Record voice sample",
                                sample_rate=16_000,
                                key=(
                                    "transcript_voice_recorder_"
                                    f"{int(st.session_state['transcript_recorder_generation'])}"
                                ),
                            )
                        else:
                            recorded_audio = None
                            selected_audio_bytes = (
                                active_session.get("audio_bytes", b"")
                                if active_session
                                else b""
                            )

                            if isinstance(selected_audio_bytes, bytes) and selected_audio_bytes:
                                st.audio(selected_audio_bytes, format="audio/wav")
                            else:
                                st.info("The selected saved recording contains no playable audio.")

                        if sessions:
                            record_col, clear_col = st.columns(2)
                            with record_col:
                                if st.button(
                                    "Record another",
                                    use_container_width=True,
                                    key="transcript_record_another",
                                ):
                                    st.session_state["transcript_voice_mode"] = "record"
                                    st.session_state["transcript_recorder_generation"] = (
                                        int(
                                            st.session_state.get(
                                                "transcript_recorder_generation",
                                                0,
                                            )
                                        )
                                        + 1
                                    )
                                    st.rerun()

                            with clear_col:
                                if st.button(
                                    "Clear recordings",
                                    use_container_width=True,
                                    key="transcript_clear_recorder",
                                ):
                                    _clear_recorder_state()
                                    st.rerun()

                        if recorded_audio is not None:
                            recorded_bytes = recorded_audio.getvalue()
                            settings = json.dumps(
                                [chunk_seconds, transcript_source, whisper_size],
                                ensure_ascii=True,
                            ).encode("utf-8")
                            signature = hashlib.sha256(recorded_bytes + settings).hexdigest()
                            signatures = {
                                str(session.get("signature", ""))
                                for session in _voice_sessions()
                            }

                            if signature not in signatures:
                                with st.spinner("Analysing voice recording..."):
                                    try:
                                        whisper_model = (
                                            _load_whisper_model(whisper_size)
                                            if WHISPER_AVAILABLE
                                            else None
                                        )
                                        active_transcript_source = (
                                            "Local Whisper"
                                            if whisper_model is not None
                                            else "Audio only"
                                        )

                                        processed = _process_recording(
                                            recorded_bytes,
                                            chunk_seconds=chunk_seconds,
                                            transcript_source=active_transcript_source,
                                            manual_transcript="",
                                            whisper_model=whisper_model,
                                            audio_classifier=audio_classifier,
                                            text_classifier=text_classifier,
                                            behavioral_classifier=behavioral_classifier,
                                        )
                                    except Exception as exc:
                                        st.session_state["transcript_recorder_error"] = str(exc)
                                    else:
                                        for chunk_index, result in enumerate(processed, 1):
                                            result["clip"] = 1
                                            result["clip_chunk"] = chunk_index
                                            result["capture_mode"] = "Speaker Voice Recorder"
                                        _create_voice_session(
                                            audio_bytes=recorded_bytes,
                                            signature=signature,
                                            results=processed,
                                        )
                                        st.session_state["transcript_recorder_error"] = ""
                                        st.session_state["transcript_pending_voice_analysis"] = True
                                        render_analysis_ready("New voice recording analysed")
                                        st.rerun()

                        if st.session_state.get("transcript_recorder_error"):
                            st.error(
                                f"Recording analysis failed: "
                                f"{st.session_state['transcript_recorder_error']}"
                            )
                        _render_voice_session_manager()
                    else:
                        st.info("Speaker Voice Recorder is not selected.")

            with audio_right:
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

        _transcript_step_header(
            "04",
            "Confirm and Analyze",
            "Review source readiness, then run the selected transcript and audio investigations.",
        )

        recorder_results = _active_recorder_results()

        uploaded_audio_results = st.session_state.get(
            "transcript_uploaded_audio_results",
            [],
        )
        if not isinstance(uploaded_audio_results, list):
            uploaded_audio_results = []

        voice_text_preview = _recorder_transcript_text()
        uploaded_audio_text_preview = _uploaded_audio_transcript_text()

        _render_combined_input_summary(
            use_voice=use_voice,
            use_uploaded_audio=use_uploaded_audio,
            use_text=use_text,
            voice_text=voice_text_preview,
            uploaded_audio_text=uploaded_audio_text_preview,
            transcript_text=text,
            recorder_results=recorder_results,
            uploaded_audio_results=uploaded_audio_results,
        )

        ready_sources = sum(
            [
                bool(use_voice and recorder_results),
                bool(
                    use_uploaded_audio
                    and st.session_state.get("transcript_uploaded_audio_file_bytes")
                ),
                bool(use_text and text.strip()),
            ]
        )

        st.markdown(
            '<div class="transcript-review-strip">'
            f'<div class="transcript-review-item"><span>Sources Selected</span><b>{sum([use_voice, use_uploaded_audio, use_text])}</b></div>'
            f'<div class="transcript-review-item"><span>Sources Ready</span><b>{ready_sources}</b></div>'
            f'<div class="transcript-review-item"><span>Transcript Words</span><b>{len(text.split()) if use_text else 0}</b></div>'
            f'<div class="transcript-review-item"><span>Alert Threshold</span><b>{risk_threshold}%</b></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        voice_ready = bool(use_voice and _active_voice_session() is not None)
        uploaded_audio_ready = bool(
            use_uploaded_audio
            and st.session_state.get("transcript_uploaded_audio_file_bytes")
        )
        text_ready = bool(use_text and text.strip())

        analyze_button = st.button(
            "* Analyze Selected Evidence",
            type="primary",
            use_container_width=True,
            disabled=not (voice_ready or uploaded_audio_ready or text_ready),
            key="transcript_analyze_selected_sources",
        )

    if isinstance(uploaded, pd.DataFrame) and use_text:
        render_section_header("Batch transcript CSV analysis", eyebrow="Multiple rows")
        render_content_card_open("violet")
        text_column = st.selectbox("Transcript column", uploaded.columns)

        if st.button("Analyze transcript CSV rows", use_container_width=True):
            texts = uploaded[text_column].fillna("").astype(str).tolist()
            try:
                classifier = _load_transcript_classifier(str(root))
                results = classifier.predict_many(texts)
            except FileNotFoundError:
                rows = []
                for value in texts:
                    demo = rule_based_text_prediction(value)
                    rows.append(
                        {
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
        if use_voice and use_uploaded_audio:
            st.error(
                "Speaker Voice Recorder and Uploaded Audio Recording cannot be analyzed together. "
                "Select only one audio source."
            )
            return

        if use_uploaded_audio:
            current_signature = st.session_state.get(
                "transcript_uploaded_audio_file_signature"
            )
            processed_signature = st.session_state.get(
                "transcript_uploaded_audio_last_processed_signature"
            )

            if not current_signature:
                st.warning("Upload an audio recording before running the analysis.")
                return

            if current_signature != processed_signature:
                with st.spinner("Analyzing uploaded audio..."):
                    try:
                        _analyse_selected_uploaded_audio(
                            root,
                            chunk_seconds=chunk_seconds,
                            whisper_size=whisper_size,
                        )
                    except Exception as exc:
                        st.session_state["transcript_uploaded_audio_error"] = str(exc)
                        st.error(f"Uploaded audio analysis failed: {exc}")
                        return

        active_voice_session = _active_voice_session() if use_voice else None
        recorder_results = (
            list(active_voice_session.get("results", []))
            if active_voice_session
            else []
        )
        uploaded_audio_results = (
            list(st.session_state.get("transcript_uploaded_audio_results", []))
            if use_uploaded_audio
            else []
        )

        voice_text = _recorder_transcript_text() if use_voice else ""
        uploaded_audio_text = (
            _uploaded_audio_transcript_text()
            if use_uploaded_audio
            else ""
        )

        _render_analysis_outputs(
            root=root,
            history=history,
            use_voice=use_voice,
            use_uploaded_audio=use_uploaded_audio,
            use_text=use_text,
            voice_text=voice_text,
            uploaded_audio_text=uploaded_audio_text,
            transcript_text=text,
            recorder_results=recorder_results,
            uploaded_audio_results=uploaded_audio_results,
            risk_threshold=risk_threshold,
        )

        st.session_state["transcript_pending_voice_analysis"] = False
        st.session_state["transcript_pending_uploaded_audio_analysis"] = False
