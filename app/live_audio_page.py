"""Local system-audio meeting scam detection demonstration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_components import (
    apply_chart_theme,
    render_analysis_ready,
    render_metric_row,
    render_result_card,
    render_section_header,
)
from src.audio.internal_capture import (
    list_internal_audio_devices,
    record_internal_chunk,
    sounddevice_available,
)
from src.audio.audio_classifier import load_audio_model
from src.text.explainability import highlighted_html
from src.reporting.history_db import DEFAULT_SESSION_ID, history_fingerprint, insert_scan
from src.audio.live_audio_analysis import (
    analyse_live_chunk,
    transcribe_with_whisper,
    wav_bytes_to_audio,
)
from src.text.text_classifier import load_text_artifacts
from src.utils.time_utils import formatted_now
from src.utils.system_check import (
    analyse_capture_test,
    build_audio_diagnostics,
    log_system_diagnostics,
)

try:
    import whisper as _whisper

    WHISPER_AVAILABLE = True
except Exception:
    _whisper = None
    WHISPER_AVAILABLE = False


def _can_capture_internal_device(device: Any) -> bool:
    """Return True when Device Audio Monitor can open this source safely."""

    return (
        bool(getattr(device, "is_internal_candidate", False))
        and not bool(getattr(device, "is_microphone", False))
        and not bool(getattr(device, "is_unsupported_backend", False))
        and int(getattr(device, "max_input_channels", 0) or 0) > 0
    )


@st.cache_resource(show_spinner=False)
def _load_audio_classifier(root: str):
    try:
        return load_audio_model(Path(root) / "models" / "audio_svm.pkl")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_transcript_classifier(root: str):
    try:
        return load_text_artifacts(
            Path(root) / "models" / "transcript_vectorizer.pkl",
            Path(root) / "models" / "transcript_nb.pkl",
            model_name="Transcript Naive Bayes",
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_whisper_model(model_size: str):
    if not WHISPER_AVAILABLE or _whisper is None:
        return None
    return _whisper.load_model(model_size)


def _init_live_state() -> None:
    defaults: dict[str, Any] = {
        "live_recorder_results": [],
        "live_recorder_signatures": [],
        "live_recorder_clip_count": 0,
        "live_recorder_generation": 0,
        "live_recorder_error": "",
        "live_recorder_saved_signature": "",
        "live_recorder_carousel_index": 0,
        "live_monitor_results": [],
        "live_monitor_signatures": [],
        "live_monitor_clip_count": 0,
        "live_monitor_generation": 0,
        "live_monitor_error": "",
        "live_monitor_saved_signature": "",
        "live_monitor_source": "",
        "live_monitor_last_wav": b"",
        "live_monitor_carousel_index": 0,
        "live_audio_diagnostics": None,
        "live_monitor_test_result": {},
        "live_monitor_test_wav": b"",
        "live_monitor_test_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_recorder_state() -> None:
    st.session_state["live_recorder_results"] = []
    st.session_state["live_recorder_signatures"] = []
    st.session_state["live_recorder_clip_count"] = 0
    st.session_state["live_recorder_error"] = ""
    st.session_state["live_recorder_saved_signature"] = ""
    st.session_state["live_recorder_carousel_index"] = 0
    st.session_state["live_recorder_generation"] = (
        int(st.session_state.get("live_recorder_generation", 0)) + 1
    )


def _clear_monitor_state() -> None:
    st.session_state["live_monitor_results"] = []
    st.session_state["live_monitor_signatures"] = []
    st.session_state["live_monitor_clip_count"] = 0
    st.session_state["live_monitor_generation"] = (
        int(st.session_state.get("live_monitor_generation", 0)) + 1
    )
    st.session_state["live_monitor_error"] = ""
    st.session_state["live_monitor_saved_signature"] = ""
    st.session_state["live_monitor_source"] = ""
    st.session_state["live_monitor_last_wav"] = b""
    st.session_state["live_monitor_carousel_index"] = 0
    st.session_state["live_monitor_test_result"] = {}
    st.session_state["live_monitor_test_wav"] = b""
    st.session_state["live_monitor_test_error"] = ""


def _prepare_next_monitor_capture() -> None:
    """Reset only the active internal-audio widget, preserving analysed clips."""

    st.session_state["live_monitor_generation"] = (
        int(st.session_state.get("live_monitor_generation", 0)) + 1
    )
    st.session_state["live_monitor_error"] = ""
    st.session_state["live_monitor_source"] = ""
    st.session_state["live_monitor_last_wav"] = b""


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


def _process_recording(
    audio_bytes: bytes,
    *,
    chunk_seconds: int,
    transcript_source: str,
    manual_transcript: str,
    whisper_model: Any | None,
    audio_classifier: Any | None,
    text_classifier: Any | None,
) -> list[dict[str, object]]:
    audio, sample_rate = wav_bytes_to_audio(audio_bytes, target_sample_rate=16_000)
    processed = []
    for index, chunk in enumerate(_recording_chunks(audio, sample_rate, chunk_seconds)):
        if transcript_source == "Local Whisper":
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
                sample_rate=sample_rate,
            )
        )
    return processed


def _transcribe_recording_file(
    audio_bytes: bytes,
    whisper_model: Any | None,
) -> str:
    """Write recorder bytes to a temporary WAV for local Whisper."""

    if whisper_model is None:
        return ""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Whisper needs ffmpeg to decode "
            "the recorded WAV before transcription. Install ffmpeg, restart the "
            "terminal/Streamlit, or use Demo fallback / Audio only."
        )
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        result = whisper_model.transcribe(
            temp_path,
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
        )
        return str(result.get("text", "")).strip()
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


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
                "Detected text": transcript or "Audio-only chunk",
                "Flags": ", ".join(result.get("flags", [])) or "-",
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
                value=transcript_text or "No speech text yet. Select Local Whisper for automatic transcription.",
                height=145,
                disabled=True,
            )
            st.dataframe(_result_table(results), hide_index=True, use_container_width=True)
            transcript = str(latest.get("transcript", "")).strip() if latest else ""
            findings = latest.get("findings", []) if latest else []
            if transcript and isinstance(findings, list):
                st.markdown(highlighted_html(transcript, findings), unsafe_allow_html=True)
        else:
            st.text_area(
                "Live transcript",
                value=empty_message,
                height=145,
                disabled=True,
            )
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
                    {"Feature": "Voice AI risk", "Value": f"{float(latest.get('voice_risk', 0)):.2f}%"},
                    {"Feature": "Transcript scam risk", "Value": f"{float(latest.get('transcript_risk', 0)):.2f}%"},
                    {"Feature": "Pitch variance", "Value": f"{float(features.get('pitch_variance', 0)):.2f} Hz"},
                    {"Feature": "Spectral centroid", "Value": f"{float(features.get('spectral_centroid', 0)) / 1000:.2f} kHz"},
                    {"Feature": "Dominant frequency", "Value": f"{float(features.get('dominant_frequency', 0)):.1f} Hz"},
                    {"Feature": "Zero crossing rate", "Value": f"{float(features.get('zero_crossing_rate', 0)):.4f}"},
                    {"Feature": "RMS energy", "Value": f"{float(features.get('rms_energy', 0)):.4f}"},
                ]
            )
            st.dataframe(feature_rows, hide_index=True, use_container_width=True)


def _save_session(
    history: list[dict[str, object]],
    results: list[dict[str, object]],
    transcript_source: str,
    capture_mode: str,
    *,
    scan_type: str,
    source_name: str,
    saved_signature_key: str,
) -> None:
    if not results:
        return

    peak_result = max(results, key=lambda item: float(item.get("risk", 0)))
    peak_risk = float(peak_result.get("risk", 0))
    transcripts = [
        str(item.get("transcript", "")).strip()
        for item in results
        if str(item.get("transcript", "")).strip()
    ]
    flags = sorted(
        {
            str(flag)
            for item in results
            for flag in item.get("flags", [])
            if str(flag).strip()
        }
    )
    timestamp = formatted_now()
    preview = " ".join(transcripts)[:800] or f"Audio-only {source_name.lower()} with {len(results)} chunk(s)."
    engines = sorted(
        {
            str(item.get("audio_engine", ""))
            for item in results
            if str(item.get("audio_engine", "")).strip()
        }
        | {
            str(item.get("text_engine", ""))
            for item in results
            if str(item.get("text_engine", "")).strip()
        }
    )
    entry: dict[str, object] = {
        "time": timestamp,
        "type": scan_type,
        "prediction": str(peak_result.get("risk_level", "Needs review")),
        "confidence": round(peak_risk, 2),
        "model": " + ".join(engines),
        "preview": preview,
        "chunks": len(results),
        "flags": flags,
        "explanation": (
            f"{capture_mode} using {transcript_source}. "
            f"Peak combined risk {peak_risk:.1f}% across {len(results)} chunk(s)."
        ),
        "raw_input": "\n".join(transcripts),
    }
    fingerprint = history_fingerprint(entry)
    entry["source_fingerprint"] = fingerprint
    history.insert(0, entry)
    insert_scan(
        session_id=DEFAULT_SESSION_ID,
        scanned_at=timestamp,
        scan_type=scan_type,
        source_name=f"{source_name} ({len(results)} chunks)",
        prediction=str(entry["prediction"]),
        confidence=float(entry["confidence"]),
        model_name=str(entry["model"]),
        preview=preview,
        flags=flags,
        explanation=str(entry["explanation"]),
        raw_input=str(entry["raw_input"]),
        source_fingerprint=fingerprint,
    )
    st.session_state[saved_signature_key] = json.dumps(
        [len(results), results[-1].get("time"), peak_risk],
        ensure_ascii=True,
    )
    render_analysis_ready(f"{source_name} saved for the AI Report Generator")


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
        if st.button(
            "Previous",
            use_container_width=True,
            disabled=current_index == 0,
            key=f"{state_key}_prev",
        ):
            st.session_state[state_key] = current_index - 1
            st.rerun()
    with nav_mid:
        st.markdown(
            f"**Clip {clip_number}** | {len(clip_results)} chunk(s) | "
            f"Flags: {', '.join(flags) if flags else 'none'}"
        )
    with nav_right:
        if st.button(
            "Next",
            use_container_width=True,
            disabled=current_index >= len(groups) - 1,
            key=f"{state_key}_next",
        ):
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
            "Recommendation: no strong scam indicators were found in this chunk, but continue "
            "to verify unexpected requests."
        )

    _render_dashboard_section(
        clip_results,
        risk_threshold,
        transcript_heading=transcript_heading,
        frequency_heading=frequency_heading,
        latest_title=latest_title,
    )


def _render_internal_audio_meter(
    slot,
    *,
    level: float,
    progress: float,
    status: str,
    duration_seconds: int = 5,
) -> None:
    """Render a compact speaker-output preview matching Streamlit's recorder style."""

    safe_level = min(1.0, max(0.0, float(level)))
    safe_progress = min(1.0, max(0.0, float(progress)))
    duration_seconds = max(1, int(duration_seconds))
    elapsed_seconds = min(duration_seconds, int(round(safe_progress * duration_seconds)))
    active_progress = int(round(safe_progress * 64))
    bars = []
    for index in range(64):
        wave = abs(np.sin((index + 1) * 0.67))
        height = 2 + int(20 * (0.18 + safe_level * 0.82) * wave)
        is_elapsed = index <= active_progress and safe_progress > 0
        color = "rgba(241,245,249,0.96)" if is_elapsed else "rgba(148,163,184,0.45)"
        bars.append(
            "<span style='"
            f"display:inline-block;width:3px;border-radius:999px;height:{height}px;"
            f"background:{color};"
            "'></span>"
        )

    slot.markdown(
        f"""
        <div style="
            background:rgba(15,23,42,0.92);
            border:1px solid rgba(30,41,59,0.72);
            border-radius:8px;
            padding:13px 16px;
            margin:8px 0 16px 0;
        ">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="
                    color:#CBD5E1;font-size:16px;line-height:1;width:18px;text-align:center;
                " title="Internal system audio">&#128266;</div>
                <div style="
                    width:0;height:0;border-top:5px solid transparent;
                    border-bottom:5px solid transparent;border-left:7px solid #CBD5E1;
                    opacity:.95;
                "></div>
                <div style="flex:1;">
                    <div style="
                        display:flex;align-items:center;gap:4px;height:26px;
                    ">
                        {''.join(bars)}
                    </div>
                </div>
                <div style="
                    min-width:90px;text-align:right;color:#CBD5E1;font-size:12px;
                    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                ">{elapsed_seconds:02d}s / {duration_seconds:02d}s</div>
            </div>
            <div style="
                margin-top:8px;color:#94A3B8;font-size:11px;
                letter-spacing:.08em;text-transform:uppercase;
            ">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_audio_diagnostics(root: Path, *, refresh: bool = False) -> dict[str, Any]:
    current = st.session_state.get("live_audio_diagnostics")
    if refresh or not isinstance(current, dict):
        try:
            current = build_audio_diagnostics()
        except Exception as exc:
            current = {
                "dependencies": [],
                "devices": [],
                "meeting_devices": [],
                "microphones": [],
                "default_input": None,
                "default_output": None,
                "capture_status": "ERROR",
                "capture_message": f"Diagnostics could not complete: {exc}",
            }
        st.session_state["live_audio_diagnostics"] = current
        log_system_diagnostics(root, current)
    return current


def _device_display_name(device: object) -> str:
    if isinstance(device, dict):
        return str(device.get("name", "Not configured"))
    return "Not configured"


def _render_audio_setup_diagnostics(
    root: Path,
    diagnostics: dict[str, Any],
    selected_device: Any | None,
    capture_problem: str,
) -> None:
    """Render compact setup checks and an actual 3-second capture test."""

    with st.expander("Audio setup and diagnostics", expanded=False):
        status = str(diagnostics.get("capture_status", "ERROR"))
        st.markdown(
            f"**Capture readiness:** `{status}`  \n"
            f"{diagnostics.get('capture_message', 'Diagnostics unavailable.')}"
        )

        dependencies = diagnostics.get("dependencies", [])
        if isinstance(dependencies, list) and dependencies:
            dependency_rows = [
                {
                    "Component": item.get("name", "Unknown"),
                    "Status": item.get("status", "ERROR"),
                    "Details": item.get("detail", ""),
                }
                for item in dependencies
                if isinstance(item, dict)
            ]
            st.dataframe(
                pd.DataFrame(dependency_rows),
                hide_index=True,
                use_container_width=True,
            )

        default_a, default_b = st.columns(2)
        with default_a:
            st.caption("Default input")
            st.write(_device_display_name(diagnostics.get("default_input")))
        with default_b:
            st.caption("Default output")
            st.write(_device_display_name(diagnostics.get("default_output")))

        meeting_devices = diagnostics.get("meeting_devices", [])
        if isinstance(meeting_devices, list) and meeting_devices:
            st.caption("Meeting Capture Devices")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Recommended": "Yes" if item.get("is_recommended") else "No",
                            "ID": item.get("index", "-"),
                            "Device": item.get("name", "Unknown"),
                            "Category": item.get("category", "System audio source"),
                            "Host API": item.get("hostapi", "Unknown"),
                            "Capture support": item.get("capture_support", "Unknown"),
                            "Rate": item.get("sample_rate", 0),
                            "Input": item.get("max_input_channels", 0),
                            "Output": item.get("max_output_channels", 0),
                            "WASAPI": "Yes" if item.get("is_wasapi") else "No",
                            "Input-ready": "Yes" if item.get("can_open_as_input") else "No",
                            "Capture-ready": "Yes" if item.get("is_capture_ready") else "No",
                        }
                        for item in meeting_devices
                        if isinstance(item, dict)
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption(
                "No meeting/system-audio source detected. On Windows enable Stereo Mix or "
                "configure VB-Cable/VoiceMeeter; on macOS use BlackHole; on Linux select a "
                "PulseAudio/PipeWire monitor source."
            )

        install_commands = [
            str(item.get("install_command", ""))
            for item in dependencies
            if isinstance(item, dict) and item.get("install_command")
        ]
        if install_commands:
            st.caption("Required installation commands")
            st.code("\n".join(install_commands), language="powershell")

        refresh_col, test_col = st.columns(2)
        with refresh_col:
            if st.button(
                "Refresh diagnostics",
                key="refresh_audio_diagnostics",
                use_container_width=True,
            ):
                _get_audio_diagnostics(root, refresh=True)
                st.rerun()
        with test_col:
            run_test = st.button(
                "Test selected device for 3 seconds",
                key="test_internal_audio_capture",
                use_container_width=True,
            )

        if run_test:
            st.session_state["live_monitor_test_error"] = ""
            if capture_problem:
                st.session_state["live_monitor_test_error"] = capture_problem
            else:
                test_meter = st.empty()
                _render_internal_audio_meter(
                    test_meter,
                    level=0.0,
                    progress=0.0,
                    status="Testing internal speaker output",
                    duration_seconds=3,
                )

                def update_test_meter(progress_value: float, level: float) -> None:
                    _render_internal_audio_meter(
                        test_meter,
                        level=min(1.0, max(0.0, level * 10.0)),
                        progress=progress_value,
                        status="Testing internal speaker output",
                        duration_seconds=3,
                    )

                try:
                    audio, sample_rate, wav_bytes = record_internal_chunk(
                        selected_device,
                        seconds=3,
                        minimum_seconds=1,
                        progress_callback=update_test_meter,
                    )
                    summary = analyse_capture_test(audio, sample_rate)
                except Exception as exc:
                    error_message = str(exc)
                    st.session_state["live_monitor_test_error"] = error_message
                    log_system_diagnostics(
                        root,
                        {
                            "event": "internal_audio_capture_test_failed",
                            "device": getattr(selected_device, "label", "Unknown"),
                            "error": error_message,
                        },
                    )
                else:
                    summary["device"] = selected_device.label
                    st.session_state["live_monitor_test_result"] = summary
                    st.session_state["live_monitor_test_wav"] = wav_bytes
                    log_system_diagnostics(
                        root,
                        {"event": "internal_audio_capture_test", **summary},
                    )

        test_error = str(st.session_state.get("live_monitor_test_error", ""))
        test_result = st.session_state.get("live_monitor_test_result", {})
        if test_error:
            st.warning(test_error)
        elif isinstance(test_result, dict) and test_result:
            status_method = {
                "PASS": st.success,
                "WARNING": st.warning,
                "ERROR": st.error,
            }.get(str(test_result.get("status", "ERROR")), st.info)
            status_method(str(test_result.get("message", "Test completed.")))
            metric_a, metric_b, metric_c, metric_d = st.columns(4)
            metric_a.metric("Duration", f"{float(test_result.get('duration_seconds', 0)):.2f}s")
            metric_b.metric("RMS level", f"{float(test_result.get('rms_db', -160)):.1f} dB")
            metric_c.metric("Peak level", f"{float(test_result.get('peak_db', -160)):.1f} dB")
            metric_d.metric("Status", str(test_result.get("status", "ERROR")))
            test_wav = st.session_state.get("live_monitor_test_wav", b"")
            if test_wav:
                st.audio(test_wav, format="audio/wav")


def _session_signature(results: list[dict[str, object]]) -> str:
    return json.dumps(
        [
            len(results),
            results[-1].get("time"),
            max(float(item.get("risk", 0)) for item in results),
        ],
        ensure_ascii=True,
    )


def _render_save_action(
    history: list[dict[str, object]],
    results: list[dict[str, object]],
    transcript_source: str,
    capture_mode: str,
    *,
    scan_type: str,
    source_name: str,
    saved_signature_key: str,
    button_label: str,
) -> None:
    if not results:
        return
    already_saved = _session_signature(results) == st.session_state.get(
        saved_signature_key
    )
    if st.button(
        button_label,
        type="primary",
        use_container_width=True,
        disabled=already_saved,
    ):
        _save_session(
            history,
            results,
            transcript_source,
            capture_mode,
            scan_type=scan_type,
            source_name=source_name,
            saved_signature_key=saved_signature_key,
        )
    if already_saved:
        st.caption("This session version is already saved.")


def _render_voice_recorder(
    root: Path,
    history: list[dict[str, object]],
) -> None:
    results: list[dict[str, object]] = st.session_state["live_recorder_results"]
    audio_classifier = _load_audio_classifier(str(root))
    text_classifier = _load_transcript_classifier(str(root))

    render_section_header(
        "Voice recorder",
        "Record a short voice sample in the browser, stop recording, and analyse it immediately.",
        "Browser recording",
    )
    with st.container(border=True):
        setting_a, setting_b, setting_c = st.columns(3)
        with setting_a:
            chunk_seconds = st.slider(
                "Analysis chunk",
                min_value=3,
                max_value=10,
                value=5,
                key="recorder_chunk_seconds",
            )
        with setting_b:
            risk_threshold = st.slider(
                "Alert threshold",
                min_value=40,
                max_value=90,
                value=70,
                step=5,
                key="recorder_risk_threshold",
            )
        with setting_c:
            transcript_options = ["Manual transcript", "Audio only"]
            if WHISPER_AVAILABLE:
                transcript_options.insert(0, "Local Whisper")
            transcript_source = st.selectbox(
                "Transcript analysis",
                transcript_options,
                key="recorder_transcript_source",
            )

        whisper_size = "tiny"
        manual_transcript = ""
        if transcript_source == "Local Whisper":
            whisper_size = st.selectbox(
                "Whisper model",
                ["tiny", "base"],
                key="recorder_whisper_size",
            )
        elif transcript_source == "Manual transcript":
            manual_transcript = st.text_area(
                "Spoken words",
                placeholder="Optional: type what was said so transcript scam analysis can run.",
                height=80,
                key="recorder_manual_transcript",
            )

        recorded_audio = st.audio_input(
            "Record voice sample",
            sample_rate=16_000,
            key=f"voice_recorder_{int(st.session_state['live_recorder_generation'])}",
        )
        action_a, action_b = st.columns(2)
        with action_a:
            if recorded_audio is not None and st.button(
                "Record another sample",
                use_container_width=True,
            ):
                st.session_state["live_recorder_generation"] += 1
                st.rerun()
        with action_b:
            if st.button("Clear recorder session", use_container_width=True):
                _clear_recorder_state()
                st.rerun()

    if not WHISPER_AVAILABLE:
        st.caption(
            "Automatic transcription is optional. Manual transcript and audio-only analysis remain available."
        )

    if recorded_audio is not None:
        recorded_bytes = recorded_audio.getvalue()
        settings = json.dumps(
            [chunk_seconds, transcript_source, whisper_size, manual_transcript],
            ensure_ascii=True,
        ).encode("utf-8")
        signature = hashlib.sha256(recorded_bytes + settings).hexdigest()
        signatures = list(st.session_state["live_recorder_signatures"])
        if signature not in signatures:
            with st.spinner("Analysing voice recording..."):
                try:
                    whisper_model = None
                    if transcript_source == "Local Whisper":
                        whisper_model = _load_whisper_model(whisper_size)
                        if whisper_model is None:
                            raise RuntimeError("Local Whisper could not be loaded.")
                    processed = _process_recording(
                        recorded_bytes,
                        chunk_seconds=chunk_seconds,
                        transcript_source=transcript_source,
                        manual_transcript=manual_transcript,
                        whisper_model=whisper_model,
                        audio_classifier=audio_classifier,
                        text_classifier=text_classifier,
                    )
                except Exception as exc:
                    st.session_state["live_recorder_error"] = str(exc)
                else:
                    clip = int(st.session_state["live_recorder_clip_count"]) + 1
                    for chunk_index, result in enumerate(processed, 1):
                        result["clip"] = clip
                        result["clip_chunk"] = chunk_index
                        result["capture_mode"] = "Browser voice recorder"
                    results.extend(processed)
                    del results[:-60]
                    signatures.append(signature)
                    st.session_state["live_recorder_signatures"] = signatures[-60:]
                    st.session_state["live_recorder_clip_count"] = clip
                    st.session_state["live_recorder_saved_signature"] = ""
                    st.session_state["live_recorder_carousel_index"] = max(
                        0,
                        len(_recording_groups(results)) - 1,
                    )
                    st.session_state["live_recorder_error"] = ""
                    render_analysis_ready(f"Voice sample {clip} analysed")

    if st.session_state.get("live_recorder_error"):
        st.error(f"Recording analysis failed: {st.session_state['live_recorder_error']}")
    elif not results:
        st.caption("Your transcript, confidence scores, and audio visualisations will appear after recording.")

    _render_recording_carousel(
        results,
        risk_threshold,
        state_key="live_recorder_carousel_index",
        title="Voice recording analysis",
        transcript_heading="Recorded transcript and chunks",
        frequency_heading="Latest frequency spectrum",
        latest_title="Latest recorded chunk {chunk}",
    )
    _render_save_action(
        history,
        results,
        transcript_source,
        "Browser voice recorder",
        scan_type="Microphone Audio",
        source_name="Voice recorder session",
        saved_signature_key="live_recorder_saved_signature",
        button_label="Save recorder session to report history",
    )


def _render_device_audio_monitor(
    root: Path,
    history: list[dict[str, object]],
) -> None:
    results: list[dict[str, object]] = st.session_state["live_monitor_results"]
    audio_classifier = _load_audio_classifier(str(root))
    text_classifier = _load_transcript_classifier(str(root))

    render_section_header(
        "Internal audio capture",
        "Capture sound produced by this device, such as Zoom, Google Meet, Teams, browser audio, or speaker output.",
        "Local-only system audio",
    )
    st.caption(
        "This does not use the physical microphone. It must run locally because hosted Streamlit cannot access your computer's speaker output."
    )
    diagnostics = _get_audio_diagnostics(root)

    with st.container(border=True):
        sounddevice_ready = sounddevice_available()
        reported_devices = list_internal_audio_devices() if sounddevice_ready else []
        unsupported_devices = [
            device
            for device in reported_devices
            if getattr(device, "is_unsupported_backend", False)
        ]
        loopback_required_devices = [
            device
            for device in reported_devices
            if getattr(device, "is_loopback_required", False)
        ]
        devices = [
            device
            for device in reported_devices
            if _can_capture_internal_device(device)
        ]
        stored_device = st.session_state.get("monitor_supported_internal_device")
        if stored_device is not None and (
            not hasattr(stored_device, "is_unsupported_backend")
            or not _can_capture_internal_device(stored_device)
        ):
            del st.session_state["monitor_supported_internal_device"]
        selected_device = None
        if not sounddevice_ready:
            st.caption(
                "Setup required: sounddevice is unavailable in this Python environment. "
                "Open Audio setup and diagnostics below."
            )
        elif not devices and loopback_required_devices and unsupported_devices:
            st.caption(
                "Speaker output and WDM-KS sources were detected, but neither can be "
                "opened by the current input-stream capture method. Use an input-capable "
                "virtual cable endpoint such as CABLE Output / VoiceMeeter Output, or upload a WAV chunk."
            )
        elif not devices and loopback_required_devices:
            st.caption(
                "A Windows speaker output was detected, but it has 0 input channels. "
                "Route meeting audio through VB-Cable/VoiceMeeter/BlackHole and select the "
                "input-capable capture endpoint."
            )
        elif not devices and unsupported_devices:
            st.caption(
                "Only Windows WDM-KS capture devices were detected. They are listed in "
                "diagnostics but excluded because this app's blocking capture flow is unsupported."
            )
        elif not devices:
            st.caption(
                "Setup required: no audio devices were reported. Open Audio setup and diagnostics below."
            )
        else:
            selected_device = st.selectbox(
                "Step 1: Select system audio device",
                devices,
                format_func=lambda device: device.label,
                key="monitor_supported_internal_device",
                help=(
                    "Only input-capable internal capture sources are selectable. "
                    "Speaker outputs and WDM-KS entries stay in diagnostics because this page "
                    "does not open them as normal input streams."
                ),
            )
            if getattr(selected_device, "is_virtual_device", False):
                recommendation_text = (
                    "Meeting Capture Device - prioritized for live transcription and scam detection"
                    if selected_device.max_input_channels > 0
                    else "Meeting Capture Device - routing endpoint; select its input-capable pair for capture"
                )
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:9px;margin:2px 0 8px 0;">
                        <span style="
                            display:inline-flex;align-items:center;padding:3px 8px;
                            border-radius:999px;background:rgba(16,185,129,.14);
                            border:1px solid rgba(52,211,153,.45);color:#6EE7B7;
                            font-size:11px;font-weight:700;letter-spacing:.04em;
                            text-transform:uppercase;
                        ">Recommended</span>
                        <span style="color:#CBD5E1;font-size:13px;">
                            {recommendation_text}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if selected_device.is_microphone:
                st.warning(
                    "This appears to be a microphone input. Device Audio Monitor is internal-audio only, so this source is blocked."
                )
            elif (
                getattr(selected_device, "is_virtual_device", False)
                and selected_device.max_input_channels <= 0
            ):
                st.warning(
                    "This is the virtual playback/routing endpoint and cannot be recorded directly. "
                    "Choose the paired CABLE/VoiceMeeter/BlackHole endpoint with input channels."
                )
            elif not selected_device.is_internal_candidate:
                st.warning(
                    "This device is not recognised as system/internal audio. Choose a speaker output, monitor source, BlackHole, Stereo Mix, or virtual cable."
                )
            else:
                st.caption(
                    f"Selected internal source: {selected_device.label} | "
                    f"ID {selected_device.index} | {selected_device.default_samplerate} Hz | "
                    f"input {selected_device.max_input_channels} / "
                    f"output {selected_device.max_output_channels} channels"
                )

        setting_a, setting_b, setting_c = st.columns(3)
        with setting_a:
            chunk_seconds = st.slider(
                "Chunk duration",
                min_value=3,
                max_value=10,
                value=5,
                key="monitor_chunk_seconds",
            )
        with setting_b:
            risk_threshold = st.slider(
                "Alert threshold",
                min_value=40,
                max_value=90,
                value=70,
                step=5,
                key="monitor_risk_threshold",
            )
        with setting_c:
            transcript_options = ["Demo fallback", "Audio only"]
            if WHISPER_AVAILABLE:
                transcript_options.insert(0, "Local Whisper")
            transcript_source = st.selectbox(
                "Transcript analysis",
                transcript_options,
                key="monitor_transcript_source",
            )

        whisper_size = "tiny"
        demo_transcript = ""
        if transcript_source == "Local Whisper":
            whisper_size = st.selectbox(
                "Whisper model",
                ["tiny", "base"],
                key="monitor_whisper_size",
            )
        elif transcript_source == "Demo fallback":
            demo_transcript = st.text_area(
                "Fallback transcript",
                value="Hello, this is your bank security team. Your account will be suspended unless you verify the OTP immediately.",
                height=88,
                key="monitor_demo_transcript",
            )

        capture_problem = ""
        if not sounddevice_ready:
            capture_problem = (
                "Internal capture setup needed: install sounddevice with "
                "`pip install sounddevice soundfile`, then restart Streamlit."
            )
        elif not devices and loopback_required_devices and unsupported_devices:
            capture_problem = (
                "Speaker outputs and Windows WDM-KS sources were detected, but neither "
                "is selectable for this capture method. Use an input-capable virtual "
                "cable endpoint such as CABLE Output / VoiceMeeter Output, or upload a WAV chunk."
            )
        elif not devices and unsupported_devices:
            capture_problem = (
                "Only Windows WDM-KS sources are available, and their blocking API is unsupported. "
                "Enable a WASAPI/DirectSound/MME duplicate or configure VB-Cable/VoiceMeeter."
            )
        elif not devices and loopback_required_devices:
            capture_problem = (
                "Only speaker/output devices were detected. They have 0 input channels in "
                "this capture workflow, so route meeting audio through VB-Cable/VoiceMeeter/"
                "BlackHole and select the input-capable endpoint."
            )
        elif not devices:
            capture_problem = (
                "Internal capture setup needed: no local audio devices were reported. "
                "Use the WAV fallback or enable a system-output/monitor source."
            )
        elif selected_device is None:
            capture_problem = "Select a system audio device before starting capture."
        elif selected_device.is_microphone:
            capture_problem = (
                "Device Audio Monitor blocks microphone inputs. Select a speaker output, "
                "Stereo Mix, BlackHole, monitor source, or virtual cable instead."
            )
        elif (
            getattr(selected_device, "is_virtual_device", False)
            and selected_device.max_input_channels <= 0
        ):
            capture_problem = (
                "Select the input-capable endpoint of this virtual audio device. The current "
                "CABLE/VoiceMeeter/BlackHole endpoint is for playback or routing only."
            )
        elif selected_device.max_input_channels <= 0:
            capture_problem = (
                "Selected device has 0 input channels and cannot be opened as an input stream. "
                "Use an input-capable virtual cable or monitor source instead."
            )
        elif not selected_device.is_internal_candidate:
            capture_problem = (
                "Selected device is not recognised as internal/system audio. Choose a "
                "WASAPI output, monitor source, BlackHole, Stereo Mix, or virtual cable."
            )

        _render_audio_setup_diagnostics(
            root,
            diagnostics,
            selected_device,
            capture_problem,
        )

        st.markdown("**Step 2: Open Zoom / Google Meet / Teams, then capture a chunk**")
        meter_slot = st.empty()
        _render_internal_audio_meter(
            meter_slot,
            level=0.0,
            progress=0.0,
            status="Internal speaker output idle",
            duration_seconds=chunk_seconds,
        )
        progress_slot = st.empty()
        action_a, action_b, action_c = st.columns(3)
        captured_wav = None
        monitor_generation = int(st.session_state["live_monitor_generation"])
        with action_a:
            if st.button(
                f"Start {chunk_seconds}s internal capture",
                type="primary",
                use_container_width=True,
                key=f"start_internal_capture_{monitor_generation}",
            ):
                if capture_problem:
                    st.session_state["live_monitor_error"] = capture_problem
                else:
                    progress = progress_slot.progress(
                        0.0,
                        text="Speaker output capture starting...",
                    )

                    def update_meter(progress_value: float, level: float) -> None:
                        meter = min(1.0, max(0.0, level * 10.0))
                        _render_internal_audio_meter(
                            meter_slot,
                            level=meter,
                            progress=progress_value,
                            status="Capturing internal speaker output",
                            duration_seconds=chunk_seconds,
                        )
                        progress.progress(
                            min(1.0, max(0.0, progress_value)),
                            text=f"Speaker output level {meter:.0%}",
                        )

                    try:
                        _audio, _sample_rate, captured_wav = record_internal_chunk(
                            selected_device,
                            seconds=chunk_seconds,
                            progress_callback=update_meter,
                        )
                        st.session_state["live_monitor_last_wav"] = captured_wav
                        st.session_state["live_monitor_source"] = selected_device.label
                    except Exception as exc:
                        error_message = str(exc)
                        st.session_state["live_monitor_error"] = error_message
                        log_system_diagnostics(
                            root,
                            {
                                "event": "internal_audio_capture_failed",
                                "device": getattr(selected_device, "label", "Unknown"),
                                "error": error_message,
                            },
                        )
                    else:
                        _render_internal_audio_meter(
                            meter_slot,
                            level=0.0,
                            progress=1.0,
                            status="Internal speaker output capture complete",
                            duration_seconds=chunk_seconds,
                        )
                        progress.progress(1.0, text="Internal capture complete")
        with action_b:
            if st.session_state.get("live_monitor_last_wav") and st.button(
                "Capture another sample",
                use_container_width=True,
                key=f"next_internal_capture_{monitor_generation}",
            ):
                _prepare_next_monitor_capture()
                st.rerun()
        with action_c:
            if st.button(
                "Clear session",
                use_container_width=True,
                key=f"clear_internal_capture_{monitor_generation}",
            ):
                _clear_monitor_state()
                st.rerun()

        uploaded_wav = st.file_uploader(
            "Fallback: upload a WAV chunk",
            type=["wav"],
            key=f"monitor_fallback_wav_{monitor_generation}",
            help="Use this when local internal capture is unavailable in the current environment.",
        )
        if uploaded_wav is not None:
            captured_wav = uploaded_wav.getvalue()
            st.session_state["live_monitor_last_wav"] = captured_wav
            st.session_state["live_monitor_source"] = uploaded_wav.name

    if not WHISPER_AVAILABLE and transcript_source == "Local Whisper":
        st.caption("Local Whisper is unavailable. Use Demo fallback or install requirements.txt.")
    if not WHISPER_AVAILABLE:
        st.caption("Demo fallback keeps the page usable when Whisper is unavailable.")

    audio_bytes = captured_wav or st.session_state.get("live_monitor_last_wav", b"")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        settings = json.dumps(
            [
                chunk_seconds,
                transcript_source,
                whisper_size,
                demo_transcript,
                st.session_state.get("live_monitor_source", ""),
            ],
            ensure_ascii=True,
        ).encode("utf-8")
        signature = hashlib.sha256(audio_bytes + settings).hexdigest()
        signatures = list(st.session_state["live_monitor_signatures"])
        if signature not in signatures:
            with st.spinner("Transcribing and analysing local audio chunk..."):
                try:
                    whisper_model = None
                    full_transcript = ""
                    warning_message = ""
                    if transcript_source == "Local Whisper":
                        whisper_model = _load_whisper_model(whisper_size)
                        if whisper_model is None:
                            raise RuntimeError("Local Whisper could not be loaded.")
                        try:
                            full_transcript = _transcribe_recording_file(
                                audio_bytes,
                                whisper_model,
                            )
                        except RuntimeError as exc:
                            warning_message = str(exc)
                            full_transcript = ""
                    elif transcript_source == "Demo fallback":
                        full_transcript = demo_transcript

                    processed = _process_recording(
                        audio_bytes,
                        chunk_seconds=chunk_seconds,
                        transcript_source="Manual transcript" if full_transcript else "Audio only",
                        manual_transcript=full_transcript,
                        whisper_model=None,
                        audio_classifier=audio_classifier,
                        text_classifier=text_classifier,
                    )
                except Exception as exc:
                    st.session_state["live_monitor_error"] = str(exc)
                else:
                    clip = int(st.session_state["live_monitor_clip_count"]) + 1
                    for chunk_index, result in enumerate(processed, 1):
                        result["clip"] = clip
                        result["clip_chunk"] = chunk_index
                        result["capture_mode"] = "Internal system audio capture"
                    results.extend(processed)
                    del results[:-60]
                    signatures.append(signature)
                    st.session_state["live_monitor_signatures"] = signatures[-60:]
                    st.session_state["live_monitor_clip_count"] = clip
                    st.session_state["live_monitor_saved_signature"] = ""
                    st.session_state["live_monitor_carousel_index"] = max(
                        0,
                        len(_recording_groups(results)) - 1,
                    )
                    st.session_state["live_monitor_error"] = warning_message
                    render_analysis_ready(f"Audio chunk {clip} analysed")

    if st.session_state.get("live_monitor_error"):
        message = str(st.session_state["live_monitor_error"])
        if "ffmpeg" in message.casefold() or "setup needed" in message.casefold():
            st.warning(message)
        else:
            st.error(f"Device audio analysis failed: {message}")
    elif not results:
        st.caption("Risk score, transcript, suspicious flags, and audio charts appear after recording.")

    _render_recording_carousel(
        results,
        risk_threshold,
        state_key="live_monitor_carousel_index",
        title="Internal system audio analysis",
        transcript_heading="Recorded device-audio transcript",
        frequency_heading="Latest in-device audio spectrum",
        latest_title="Latest local audio chunk {chunk}",
    )
    _render_save_action(
        history,
        results,
        transcript_source,
        "Internal system audio capture",
        scan_type="Internal Audio",
        source_name="Internal audio session",
        saved_signature_key="live_monitor_saved_signature",
        button_label="Save internal audio session to report history",
    )


def render_live_audio_page(root: Path, history: list[dict[str, object]]) -> None:
    _init_live_state()
    render_section_header(
        "Live audio detection",
        "Use separate tabs for the original Voice Recorder and the internal system-audio workflow.",
        "Audio analysis",
    )
    voice_tab, device_tab = st.tabs(["Voice Recorder", "Device Audio Monitor"])
    with voice_tab:
        _render_voice_recorder(root, history)
    with device_tab:
        _render_device_audio_monitor(root, history)
