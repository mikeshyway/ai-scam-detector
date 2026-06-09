"""Browser-microphone live audio scam detection demonstration."""

from __future__ import annotations

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


def _timeline_figure(results: list[dict[str, object]], threshold: int) -> go.Figure:
    x_values = list(range(1, len(results) + 1))
    risks = [float(item.get("risk", 0)) for item in results]
    labels = [str(item.get("risk_level", "")) for item in results]
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
            hovertemplate="Chunk %{x}<br>Risk %{y:.1f}%<br>%{text}<extra></extra>",
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
            lines.append(f"[{result.get('time', '--:--:--')}] {transcript}")
    return "\n".join(lines)


def _result_table(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for index, result in enumerate(reversed(results[-20:]), 1):
        transcript = str(result.get("transcript", "")).strip()
        rows.append(
            {
                "Chunk": len(results) - index + 1,
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
                {"label": "Chunks Processed", "value": len(results), "color": "#2563EB"},
                {"label": "Current Risk", "value": f"{float(latest.get('risk', 0)):.0f}%" if latest else "0%", "color": "#D97706"},
                {"label": "Peak Risk", "value": f"{peak:.0f}%", "color": "#DC2626"},
                {"label": "Average Risk", "value": f"{average:.0f}%", "color": "#0891B2"},
                {"label": "Alerts", "value": alert_count, "color": "#DC2626"},
            ]
        )

    with result_placeholder.container():
        if latest:
            render_result_card(
                "Latest microphone chunk",
                float(latest.get("risk", 0)),
                str(latest.get("explanation", "")),
            )
            if float(latest.get("risk", 0)) >= threshold:
                st.error(
                    f"Alert threshold reached. This chunk scored {float(latest.get('risk', 0)):.1f}% combined risk."
                )
        else:
            st.info("Start the microphone stream. Results appear after the first complete audio chunk.")

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
                value="Start the microphone stream. Local Whisper converts completed chunks into text.",
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
        "type": "Live Audio",
        "prediction": str(peak_result.get("risk_level", "Needs review")),
        "confidence": round(peak_risk, 2),
        "model": " + ".join(engines),
        "preview": preview,
        "chunks": len(results),
        "flags": flags,
        "explanation": (
            f"Live browser microphone demonstration using {transcript_source}. "
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
        scan_type="Live Audio",
        source_name=f"Live microphone session ({len(results)} chunks)",
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
        "Analyse browser microphone audio in short chunks using voice features, optional local transcription, and explainable scam-language indicators.",
        "Microphone demonstration",
    )
    render_info_banner(
        "This educational page captures only the microphone selected in the browser. It does not intercept phone calls, "
        "Zoom, Teams, Google Meet, or system audio. A speaker or virtual audio cable is required to include other participants.",
        kind="warning",
        code="SCOPE",
    )
    render_info_banner(
        "Microphone access works on localhost. Remote deployments require HTTPS, and restrictive networks may also require a TURN server.",
        kind="info",
        code="WEBRTC",
    )

    audio_classifier = _load_audio_classifier(str(root))
    text_classifier = _load_transcript_classifier(str(root))
    rtc_configuration, rtc_mode, rtc_error = _rtc_settings()
    render_metric_row(
        [
            {"label": "WebRTC", "value": "Ready" if WEBRTC_AVAILABLE else "Missing", "color": "#059669" if WEBRTC_AVAILABLE else "#DC2626"},
            {"label": "ICE Relay", "value": "TURN" if rtc_mode.startswith("TURN") else "STUN only", "color": "#059669" if rtc_mode.startswith("TURN") else "#D97706"},
            {"label": "Audio SVM", "value": "Ready" if audio_classifier else "Heuristic", "color": "#059669" if audio_classifier else "#D97706"},
            {"label": "Transcript Model", "value": "Ready" if text_classifier else "Rule Demo", "color": "#059669" if text_classifier else "#D97706"},
            {"label": "Local Whisper", "value": "Ready" if WHISPER_AVAILABLE else "Optional install", "color": "#2563EB" if WHISPER_AVAILABLE else "#6B7280"},
        ]
    )
    if rtc_error:
        render_info_banner(rtc_error, kind="danger", code="TURN ERROR")
    elif rtc_mode == "STUN only":
        render_info_banner(
            "STUN-only mode is active. This commonly stalls on Streamlit Community Cloud, university Wi-Fi, VPNs, "
            "corporate firewalls, and carrier-grade NAT. Configure TURN secrets before relying on the live demonstration.",
            kind="warning",
            code="TURN NEEDED",
        )
    else:
        render_info_banner(
            "TURN relay credentials are configured. Hosted WebRTC can relay microphone media when direct ICE paths fail.",
            kind="success",
            code="TURN READY",
        )

    with st.container(border=True):
        st.subheader("Live session settings")
        setting_a, setting_b, setting_c = st.columns(3)
        with setting_a:
            chunk_seconds = st.slider("Analysis chunk", min_value=3, max_value=10, value=4, step=1)
        with setting_b:
            risk_threshold = st.slider("Alert threshold", min_value=40, max_value=90, value=70, step=5)
        with setting_c:
            transcript_options = ["Audio only", "Manual demo text"]
            if WHISPER_AVAILABLE:
                transcript_options.insert(0, "Local Whisper")
            transcript_source = st.selectbox("Transcript source", transcript_options)

        whisper_size = "tiny"
        manual_transcript = ""
        if transcript_source == "Local Whisper":
            whisper_size = st.selectbox("Whisper model size", ["tiny", "base"], index=0)
            st.caption("The first use may download the selected Whisper model and can be slow on CPU.")
        elif transcript_source == "Manual demo text":
            manual_transcript = st.text_area(
                "Manual transcript overlay",
                placeholder="Enter a sentence used only for educational text-risk fusion. It is not transcribed from the microphone.",
                height=90,
            )

        action_left, action_right = st.columns([0.72, 0.28])
        with action_left:
            st.caption(
                "Audio is processed in memory as short chunks. Raw microphone audio is not saved by this page."
            )
        with action_right:
            if st.button("Clear live session", use_container_width=True):
                _clear_live_state()
                st.rerun()

    buffer.configure(chunk_seconds)

    if not WEBRTC_AVAILABLE:
        render_info_banner(
            "Live microphone transport is unavailable. Install requirements.txt, then restart Streamlit. "
            f"Import detail: {WEBRTC_IMPORT_ERROR or 'dependency missing'}",
            kind="danger",
            code="SETUP",
        )
        return

    whisper_model = None
    if transcript_source == "Local Whisper":
        with st.spinner(f"Loading local Whisper {whisper_size} model..."):
            try:
                whisper_model = _load_whisper_model(whisper_size)
            except Exception as exc:
                st.error(f"Whisper could not be loaded: {exc}")
                transcript_source = "Audio only"

    render_section_header(
        "Microphone stream",
        "Select Start below and grant browser microphone permission. The first result appears after one complete chunk.",
        "WebRTC input",
    )

    def audio_frame_callback(frame):
        return buffer.push_frame(frame)

    webrtc_context = webrtc_streamer(
        key="aifds-live-audio",
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
        render_section_header("Recent chunks and transcript", eyebrow="Live evidence")
        transcript_placeholder = st.empty()
        render_section_header("MFCC feature heatmap", eyebrow="Audio pattern")
        mfcc_placeholder = st.empty()
    with display_b:
        render_section_header("Live frequency spectrum", eyebrow="Audio frequency")
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

    if webrtc_context.state.playing:
        status_placeholder.success(
            f"Microphone active. Processing approximately every {chunk_seconds} seconds."
        )
        while webrtc_context.state.playing:
            chunk = buffer.get_chunk(timeout=0.4)
            if chunk is None:
                time.sleep(0.05)
                continue

            try:
                if transcript_source == "Local Whisper":
                    transcript = transcribe_with_whisper(chunk, whisper_model)
                elif transcript_source == "Manual demo text":
                    transcript = manual_transcript
                else:
                    transcript = ""

                result = analyse_live_chunk(
                    chunk,
                    transcript=transcript,
                    audio_classifier=audio_classifier,
                    text_classifier=text_classifier,
                )
            except Exception as exc:
                st.session_state["live_audio_error"] = str(exc)
                status_placeholder.error(f"Chunk analysis failed: {exc}")
                time.sleep(0.15)
                continue

            results.append(result)
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
    else:
        status_placeholder.info("Microphone is stopped. Start the stream to continue analysis.")

    if st.session_state.get("live_audio_error"):
        st.warning(f"Most recent processing issue: {st.session_state['live_audio_error']}")

    if results:
        render_section_header(
            "Save session evidence",
            "Store the current live-session summary so it appears in the AI Report Generator.",
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
        if st.button(
            "Save live session to report history",
            type="primary",
            use_container_width=True,
            disabled=already_saved,
        ):
            _save_session(history, results, transcript_source)
        if already_saved:
            st.caption("This live-session state has already been saved. Process another chunk to create a new version.")
