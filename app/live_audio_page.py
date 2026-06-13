"""Browser-microphone live audio scam detection demonstration."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_components import (
    apply_chart_theme,
    render_analysis_ready,
    render_info_banner,
    render_metric_row,
    render_result_card,
    render_section_header,
)
from src.audio_classifier import load_audio_model
from src.explainability import highlighted_html
from src.history_db import DEFAULT_SESSION_ID, history_fingerprint, insert_scan
from src.live_audio_analysis import (
    AudioChunkBuffer,
    analyse_live_chunk,
    transcribe_with_whisper,
    wav_bytes_to_audio,
)
from src.text_classifier import load_text_artifacts
from src.time_utils import formatted_now
from src.webrtc_config import (
    build_rtc_configuration,
    fetch_twilio_ice_servers,
    static_turn_servers,
)


try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = ""
except Exception as exc:
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = str(exc)

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


def _secret_value(name: str, *, section: str = "webrtc") -> object:
    environment_value = os.environ.get(name)
    if environment_value is not None:
        return environment_value
    try:
        section_values = st.secrets.get(section, {})
        if name in section_values:
            return section_values[name]
        return st.secrets.get(name, "")
    except Exception:
        return ""


@st.cache_data(show_spinner=False, ttl=3600)
def _twilio_turn_servers(account_sid: str, auth_token: str) -> list[dict[str, object]]:
    return fetch_twilio_ice_servers(account_sid, auth_token)


def _rtc_settings() -> tuple[dict[str, object], str, str]:
    turn_servers = static_turn_servers(
        urls=_secret_value("AIFDS_TURN_URLS") or _secret_value("turn_urls"),
        username=str(_secret_value("AIFDS_TURN_USERNAME") or _secret_value("turn_username")),
        credential=str(_secret_value("AIFDS_TURN_CREDENTIAL") or _secret_value("turn_credential")),
    )
    provider_error = ""

    if not turn_servers:
        account_sid = str(
            _secret_value("TWILIO_ACCOUNT_SID")
            or _secret_value("twilio_account_sid")
        )
        auth_token = str(
            _secret_value("TWILIO_AUTH_TOKEN")
            or _secret_value("twilio_auth_token")
        )
        if account_sid and auth_token:
            try:
                turn_servers = _twilio_turn_servers(account_sid, auth_token)
            except Exception as exc:
                provider_error = f"Twilio TURN token request failed: {exc}"

    configuration, mode = build_rtc_configuration(turn_servers=turn_servers)
    return configuration, mode, provider_error


def _init_live_state() -> None:
    defaults: dict[str, Any] = {
        "live_audio_results": [],
        "live_audio_buffer": AudioChunkBuffer(),
        "live_audio_error": "",
        "live_audio_saved_signature": "",
        "live_recording_signatures": [],
        "live_clip_count": 0,
        "live_recorder_generation": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_live_state() -> None:
    buffer = st.session_state.get("live_audio_buffer")
    if isinstance(buffer, AudioChunkBuffer):
        buffer.clear()
    st.session_state["live_audio_results"] = []
    st.session_state["live_audio_error"] = ""
    st.session_state["live_audio_saved_signature"] = ""
    st.session_state["live_recording_signatures"] = []
    st.session_state["live_clip_count"] = 0
    st.session_state["live_recorder_generation"] = (
        int(st.session_state.get("live_recorder_generation", 0)) + 1
    )


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


def _recording_chunks(audio: np.ndarray, sample_rate: int, chunk_seconds: int) -> list[np.ndarray]:
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
    chunks = _recording_chunks(audio, sample_rate, chunk_seconds)
    for index, chunk in enumerate(chunks):
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
    metrics_placeholder,
    result_placeholder,
    timeline_placeholder,
    transcript_placeholder,
    mfcc_placeholder,
    frequency_placeholder,
    features_placeholder,
) -> None:
    latest = results[-1] if results else None
    clip_count = max((int(item.get("clip", 0)) for item in results), default=0)
    peak = max((float(item.get("risk", 0)) for item in results), default=0.0)
    alert_count = sum(1 for item in results if float(item.get("risk", 0)) >= threshold)

    with metrics_placeholder.container():
        render_metric_row(
            [
                {"label": "Clips Recorded", "value": clip_count, "color": "#2563EB"},
                {"label": "Chunks Processed", "value": len(results), "color": "#0891B2"},
                {"label": "Current Risk", "value": f"{float(latest.get('risk', 0)):.0f}%" if latest else "0%", "color": "#D97706"},
                {"label": "Peak Risk", "value": f"{peak:.0f}%", "color": "#DC2626"},
                {"label": "Alerts", "value": alert_count, "color": "#DC2626"},
            ]
        )

    with result_placeholder.container():
        if latest:
            render_result_card(
                f"Latest result: clip {latest.get('clip', 1)}, chunk {latest.get('clip_chunk', 1)}",
                float(latest.get("risk", 0)),
                str(latest.get("explanation", "")),
            )
            if float(latest.get("risk", 0)) >= threshold:
                st.error(
                    f"Alert threshold reached. This chunk scored {float(latest.get('risk', 0)):.1f}% combined risk."
                )
        else:
            st.info("Record a short clip and stop the recorder. Analysis begins automatically.")

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
                value="Record and stop a clip. Local Whisper converts the completed recording into text.",
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
    preview = " ".join(transcripts)[:800] or f"Audio-only live microphone session with {len(results)} chunk(s)."
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
        "type": "Microphone Audio",
        "prediction": str(peak_result.get("risk_level", "Needs review")),
        "confidence": round(peak_risk, 2),
        "model": " + ".join(engines),
        "preview": preview,
        "chunks": len(results),
        "flags": flags,
        "explanation": (
            f"Browser microphone demonstration using {capture_mode} and {transcript_source}. "
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
        scan_type="Microphone Audio",
        source_name=f"Microphone session ({len(results)} chunks)",
        prediction=str(entry["prediction"]),
        confidence=float(entry["confidence"]),
        model_name=str(entry["model"]),
        preview=preview,
        flags=flags,
        explanation=str(entry["explanation"]),
        raw_input=str(entry["raw_input"]),
        source_fingerprint=fingerprint,
    )
    st.session_state["live_audio_saved_signature"] = json.dumps(
        [len(results), results[-1].get("time"), peak_risk],
        ensure_ascii=True,
    )
    render_analysis_ready("Live audio session saved for the AI Report Generator")


def render_live_audio_page(root: Path, history: list[dict[str, object]]) -> None:
    _init_live_state()
    results: list[dict[str, object]] = st.session_state["live_audio_results"]
    buffer: AudioChunkBuffer = st.session_state["live_audio_buffer"]

    render_section_header(
        "Live audio detection",
        "Record short microphone clips, convert speech to text, and combine transcript scam indicators with explainable voice features.",
        "Near-real-time clip analysis",
    )
    render_info_banner(
        "This page records only the microphone selected in the browser. It does not intercept phone calls, "
        "Zoom, Teams, Google Meet, or protected system audio.",
        kind="warning",
        code="SCOPE",
    )
    render_info_banner(
        "Recommended flow: record 5-10 seconds, stop, review the transcript and risk result, then record the next clip. "
        "The session builds without WebRTC, STUN, or TURN.",
        kind="success",
        code="RELIABLE",
    )

    audio_classifier = _load_audio_classifier(str(root))
    text_classifier = _load_transcript_classifier(str(root))
    render_metric_row(
        [
            {"label": "Microphone Recorder", "value": "Ready" if hasattr(st, "audio_input") else "Upgrade required", "color": "#059669" if hasattr(st, "audio_input") else "#DC2626"},
            {"label": "Audio SVM", "value": "Ready" if audio_classifier else "Heuristic", "color": "#059669" if audio_classifier else "#D97706"},
            {"label": "Transcript Model", "value": "Ready" if text_classifier else "Rule Demo", "color": "#059669" if text_classifier else "#D97706"},
            {"label": "Local Whisper", "value": "Ready" if WHISPER_AVAILABLE else "Optional install", "color": "#2563EB" if WHISPER_AVAILABLE else "#6B7280"},
        ]
    )

    with st.container(border=True):
        st.subheader("Recording and analysis settings")
        setting_a, setting_b, setting_c = st.columns(3)
        with setting_a:
            chunk_seconds = st.slider(
                "Analysis chunk",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                help="Longer recordings are divided into chunks of this duration.",
            )
        with setting_b:
            risk_threshold = st.slider(
                "Alert threshold",
                min_value=40,
                max_value=90,
                value=70,
                step=5,
            )
        with setting_c:
            transcript_options = ["Manual transcript", "Audio only"]
            if WHISPER_AVAILABLE:
                transcript_options.insert(0, "Local Whisper")
            transcript_source = st.selectbox("Transcript source", transcript_options)

        whisper_size = "tiny"
        manual_transcript = ""
        if transcript_source == "Local Whisper":
            whisper_size = st.selectbox(
                "Whisper model size",
                ["tiny", "base"],
                index=0,
                help="Tiny is fastest on CPU. Base can improve transcription at the cost of more processing time.",
            )
            st.caption("The first use downloads and caches the selected model. Later clips reuse it.")
        elif transcript_source == "Manual transcript":
            manual_transcript = st.text_area(
                "Spoken words for this clip",
                placeholder="Type what was said, then record the matching voice clip.",
                height=90,
                help="Use this fallback when local Whisper is not installed.",
            )

        note_column, clear_column = st.columns([0.72, 0.28])
        with note_column:
            st.caption(
                "Recordings are processed in memory. Only the analysis summary is added to report history when you choose Save."
            )
        with clear_column:
            if st.button("Clear recording session", use_container_width=True):
                _clear_live_state()
                st.rerun()

    buffer.configure(chunk_seconds)

    whisper_model = None

    render_section_header(
        "Record the next conversation clip",
        "Speak for roughly 5-10 seconds, stop the recorder, and wait for the transcript and risk panels to update.",
        "Primary microphone input",
    )

    recorded_audio = None
    if not hasattr(st, "audio_input"):
        render_info_banner(
            "This recorder requires Streamlit 1.48 or newer. Reinstall requirements.txt and restart the app.",
            kind="danger",
            code="UPGRADE",
        )
    else:
        recorder_generation = int(st.session_state.get("live_recorder_generation", 0))
        recorded_audio = st.audio_input(
            "Microphone recording",
            sample_rate=16_000,
            key=f"aifds_microphone_{recorder_generation}",
            help="Stop the recording when finished. The new clip is analysed automatically.",
        )

    if recorded_audio is not None:
        recorded_bytes = recorded_audio.getvalue()
        recording_signature = hashlib.sha256(recorded_bytes).hexdigest()
        processed_signatures = list(
            st.session_state.get("live_recording_signatures", [])
        )

        if recording_signature not in processed_signatures:
            with st.spinner("Transcribing and analysing the recorded clip..."):
                try:
                    if transcript_source == "Local Whisper":
                        whisper_model = _load_whisper_model(whisper_size)
                        if whisper_model is None:
                            raise RuntimeError(
                                "Local Whisper is unavailable. Install requirements-live.txt "
                                "or select Manual transcript."
                            )
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
                    st.session_state["live_audio_error"] = str(exc)
                    st.error(f"Microphone clip analysis failed: {exc}")
                else:
                    clip_number = int(st.session_state.get("live_clip_count", 0)) + 1
                    for chunk_index, result in enumerate(processed, 1):
                        result["clip"] = clip_number
                        result["clip_chunk"] = chunk_index
                        result["capture_mode"] = "Short microphone recording"
                    results.extend(processed)
                    if len(results) > 60:
                        del results[:-60]

                    processed_signatures.append(recording_signature)
                    st.session_state["live_recording_signatures"] = processed_signatures[-60:]
                    st.session_state["live_clip_count"] = clip_number
                    st.session_state["live_audio_saved_signature"] = ""
                    st.session_state["live_audio_error"] = ""
                    render_analysis_ready(
                        f"Clip {clip_number} analysed in {len(processed)} chunk(s)"
                    )

        next_column, session_column = st.columns([0.34, 0.66])
        with next_column:
            if st.button(
                "Record next clip",
                type="primary",
                use_container_width=True,
                help="Resets the recorder while keeping the current session analysis.",
            ):
                st.session_state["live_recorder_generation"] = (
                    int(st.session_state.get("live_recorder_generation", 0)) + 1
                )
                st.rerun()
        with session_column:
            st.caption(
                f"{int(st.session_state.get('live_clip_count', 0))} clip(s) currently stored in this analysis session."
            )
    else:
        st.caption("No microphone clip has been submitted yet.")

    webrtc_context = None
    advanced_webrtc_enabled = False
    with st.expander("Advanced local WebRTC experiment", expanded=False):
        st.caption(
            "Optional continuous streaming for a local demonstration. Hosted or restricted networks may still require TURN. "
            "The short recorder above is the supported reliable path."
        )
        advanced_webrtc_enabled = st.toggle(
            "Enable advanced WebRTC controls",
            value=False,
            help="Leave this off unless you are intentionally testing continuous local streaming.",
        )
        if advanced_webrtc_enabled:
            if not WEBRTC_AVAILABLE:
                render_info_banner(
                    "streamlit-webrtc is unavailable. Install requirements.txt or continue with the recorder above.",
                    kind="danger",
                    code="SETUP",
                )
            else:
                if transcript_source == "Local Whisper":
                    with st.spinner(f"Loading local Whisper {whisper_size} model..."):
                        try:
                            whisper_model = _load_whisper_model(whisper_size)
                        except Exception as exc:
                            st.session_state["live_audio_error"] = str(exc)
                            st.error(f"Whisper could not be loaded: {exc}")
                            transcript_source = "Audio only"

                rtc_configuration, rtc_mode, rtc_error = _rtc_settings()
                if rtc_error:
                    render_info_banner(rtc_error, kind="danger", code="TURN ERROR")
                elif rtc_mode == "STUN only":
                    render_info_banner(
                        "No TURN relay is configured. Use this only on a network where a direct WebRTC connection succeeds.",
                        kind="warning",
                        code="STUN ONLY",
                    )
                else:
                    render_info_banner(
                        "A TURN relay is configured for this optional WebRTC test.",
                        kind="success",
                        code="TURN READY",
                    )

                def audio_frame_callback(frame):
                    return buffer.push_frame(frame)

                webrtc_context = webrtc_streamer(
                    key="aifds-live-audio-advanced",
                    mode=WebRtcMode.SENDONLY,
                    audio_frame_callback=audio_frame_callback,
                    media_stream_constraints={
                        "audio": {
                            "echoCancellation": True,
                            "noiseSuppression": True,
                            "channelCount": 1,
                        },
                        "video": False,
                    },
                    rtc_configuration=rtc_configuration,
                )

    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    result_placeholder = st.empty()
    timeline_placeholder = st.empty()

    display_a, display_b = st.columns([0.62, 0.38])
    with display_a:
        render_section_header("Session transcript and chunks", eyebrow="Conversation evidence")
        transcript_placeholder = st.empty()
        render_section_header("MFCC feature heatmap", eyebrow="Audio pattern")
        mfcc_placeholder = st.empty()
    with display_b:
        render_section_header("Latest frequency spectrum", eyebrow="Audio frequency")
        frequency_placeholder = st.empty()
        render_section_header("Latest acoustic features", eyebrow="Voice indicators")
        features_placeholder = st.empty()

    _render_live_dashboard(
        results,
        risk_threshold,
        metrics_placeholder=metrics_placeholder,
        result_placeholder=result_placeholder,
        timeline_placeholder=timeline_placeholder,
        transcript_placeholder=transcript_placeholder,
        mfcc_placeholder=mfcc_placeholder,
        frequency_placeholder=frequency_placeholder,
        features_placeholder=features_placeholder,
    )

    if webrtc_context is not None and webrtc_context.state.playing:
        stream_clip = int(st.session_state.get("live_clip_count", 0)) + 1
        status_placeholder.success(
            f"Advanced microphone stream active. Processing approximately every {chunk_seconds} seconds."
        )
        while webrtc_context.state.playing:
            chunk = buffer.get_chunk(timeout=0.4)
            if chunk is None:
                time.sleep(0.05)
                continue

            try:
                if transcript_source == "Local Whisper":
                    transcript = transcribe_with_whisper(chunk, whisper_model)
                elif transcript_source == "Manual transcript":
                    transcript = manual_transcript
                else:
                    transcript = ""

                result = analyse_live_chunk(
                    chunk,
                    transcript=transcript,
                    audio_classifier=audio_classifier,
                    text_classifier=text_classifier,
                )
                result["clip"] = stream_clip
                result["clip_chunk"] = (
                    sum(1 for item in results if int(item.get("clip", 0)) == stream_clip) + 1
                )
                result["capture_mode"] = "Advanced local WebRTC"
            except Exception as exc:
                st.session_state["live_audio_error"] = str(exc)
                status_placeholder.error(f"WebRTC chunk analysis failed: {exc}")
                time.sleep(0.15)
                continue

            results.append(result)
            st.session_state["live_clip_count"] = stream_clip
            st.session_state["live_audio_saved_signature"] = ""
            if len(results) > 60:
                del results[:-60]

            _render_live_dashboard(
                results,
                risk_threshold,
                metrics_placeholder=metrics_placeholder,
                result_placeholder=result_placeholder,
                timeline_placeholder=timeline_placeholder,
                transcript_placeholder=transcript_placeholder,
                mfcc_placeholder=mfcc_placeholder,
                frequency_placeholder=frequency_placeholder,
                features_placeholder=features_placeholder,
            )
    elif advanced_webrtc_enabled:
        status_placeholder.info("Advanced WebRTC is enabled but stopped.")
    elif results:
        status_placeholder.success(
            "The recorded session is ready. Record another clip above to continue the conversation."
        )
    else:
        status_placeholder.info("Record and stop the first microphone clip to begin analysis.")

    if st.session_state.get("live_audio_error"):
        st.warning(f"Most recent processing issue: {st.session_state['live_audio_error']}")

    if results:
        render_section_header(
            "Save session evidence",
            "Store the current microphone-session summary so it appears in the AI Report Generator.",
            "Report integration",
        )
        current_signature = json.dumps(
            [
                len(results),
                results[-1].get("time"),
                max(float(item.get("risk", 0)) for item in results),
            ],
            ensure_ascii=True,
        )
        already_saved = current_signature == st.session_state.get("live_audio_saved_signature")
        capture_modes = {
            str(item.get("capture_mode", "Short microphone recording"))
            for item in results
        }
        capture_mode = " + ".join(sorted(capture_modes))
        if st.button(
            "Save microphone session to report history",
            type="primary",
            use_container_width=True,
            disabled=already_saved,
        ):
            _save_session(history, results, transcript_source, capture_mode)
        if already_saved:
            st.caption("This session version is already saved. Record another clip to create a new version.")
