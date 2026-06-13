"""Local system-audio meeting scam detection demonstration."""

from __future__ import annotations

import json
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
    analyse_live_chunk,
    transcribe_with_whisper,
)
from src.system_audio_capture import (
    CaptureDevice,
    LocalSystemAudioMonitor,
    list_capture_devices,
    soundcard_available,
)
from src.text_classifier import load_text_artifacts
from src.time_utils import formatted_now

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


def _init_live_state() -> None:
    defaults: dict[str, Any] = {
        "live_audio_results": [],
        "live_system_monitor": LocalSystemAudioMonitor(),
        "live_audio_error": "",
        "live_audio_saved_signature": "",
        "live_monitor_chunk_index": 0,
        "live_monitor_source": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_live_state() -> None:
    monitor = st.session_state.get("live_system_monitor")
    if isinstance(monitor, LocalSystemAudioMonitor):
        monitor.stop()
        monitor.clear()
    st.session_state["live_audio_results"] = []
    st.session_state["live_audio_error"] = ""
    st.session_state["live_audio_saved_signature"] = ""
    st.session_state["live_monitor_chunk_index"] = 0
    st.session_state["live_monitor_source"] = ""


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
                f"Latest system-audio chunk {latest.get('clip_chunk', 1)}",
                float(latest.get("risk", 0)),
                str(latest.get("explanation", "")),
            )
            if float(latest.get("risk", 0)) >= threshold:
                st.error(
                    f"Alert threshold reached. This chunk scored {float(latest.get('risk', 0)):.1f}% combined risk."
                )
        else:
            st.info("Start the local meeting monitor. Results appear after the first complete audio chunk.")

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
                value="Start monitoring. Local Whisper converts completed system-audio chunks into text.",
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
    preview = " ".join(transcripts)[:800] or f"Audio-only local meeting monitor session with {len(results)} chunk(s)."
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
        "type": "System Audio",
        "prediction": str(peak_result.get("risk_level", "Needs review")),
        "confidence": round(peak_risk, 2),
        "model": " + ".join(engines),
        "preview": preview,
        "chunks": len(results),
        "flags": flags,
        "explanation": (
            f"Local meeting monitor using {capture_mode} and {transcript_source}. "
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
        scan_type="System Audio",
        source_name=f"Local meeting monitor ({len(results)} chunks)",
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
    render_analysis_ready("Local meeting-monitor session saved for the AI Report Generator")


@st.fragment(run_every=1.0)
def _render_monitor_fragment(
    root: Path,
    history: list[dict[str, object]],
    risk_threshold: int,
    transcript_source: str,
    whisper_size: str,
) -> None:
    """Drain local capture chunks, analyse them, and refresh only this panel."""

    monitor: LocalSystemAudioMonitor = st.session_state["live_system_monitor"]
    results: list[dict[str, object]] = st.session_state["live_audio_results"]
    audio_classifier = _load_audio_classifier(str(root))
    text_classifier = _load_transcript_classifier(str(root))
    chunks = monitor.drain_chunks(limit=2)

    whisper_model = None
    if chunks and transcript_source == "Local Whisper":
        try:
            whisper_model = _load_whisper_model(whisper_size)
            if whisper_model is None:
                raise RuntimeError(
                    "Local Whisper is unavailable. Install requirements-live.txt."
                )
        except Exception as exc:
            st.session_state["live_audio_error"] = str(exc)

    for chunk in chunks:
        try:
            energy = float(np.sqrt(np.mean(np.square(chunk))))
            transcript = ""
            if transcript_source == "Local Whisper" and whisper_model is not None:
                if energy >= 0.002:
                    transcript = transcribe_with_whisper(chunk, whisper_model)

            result = analyse_live_chunk(
                chunk,
                transcript=transcript,
                audio_classifier=audio_classifier,
                text_classifier=text_classifier,
            )
            chunk_index = int(st.session_state.get("live_monitor_chunk_index", 0)) + 1
            result["clip"] = 1
            result["clip_chunk"] = chunk_index
            result["capture_mode"] = "Local system-audio capture"
            result["source_device"] = monitor.device_name
            st.session_state["live_monitor_chunk_index"] = chunk_index
            st.session_state["live_audio_saved_signature"] = ""
            st.session_state["live_audio_error"] = ""
            results.append(result)
            if len(results) > 60:
                del results[:-60]
        except Exception as exc:
            st.session_state["live_audio_error"] = str(exc)

    stats = monitor.stats()
    if monitor.error:
        st.error(f"Local audio capture stopped: {monitor.error}")
    elif monitor.running:
        st.success(
            f"Monitoring {monitor.device_name}. "
            f"Captured {stats['captured_chunks']} chunk(s); "
            f"{stats['queued_chunks']} waiting for analysis."
        )
    elif results:
        st.info("Monitoring is stopped. The captured session remains available below.")
    else:
        st.info("Select a local system-output or virtual-cable device and start monitoring.")

    if int(stats["dropped_chunks"]) > 0:
        st.warning(
            f"{stats['dropped_chunks']} audio chunk(s) were dropped because transcription "
            "could not keep pace. Select the tiny Whisper model or increase chunk duration."
        )

    if st.session_state.get("live_audio_error"):
        st.warning(f"Most recent processing issue: {st.session_state['live_audio_error']}")

    metrics_placeholder = st.empty()
    result_placeholder = st.empty()
    timeline_placeholder = st.empty()

    display_a, display_b = st.columns([0.62, 0.38])
    with display_a:
        render_section_header("Meeting transcript and chunks", eyebrow="Local evidence")
        transcript_placeholder = st.empty()
        render_section_header("MFCC feature heatmap", eyebrow="Audio pattern")
        mfcc_placeholder = st.empty()
    with display_b:
        render_section_header("Latest frequency spectrum", eyebrow="System audio")
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

    if results:
        render_section_header(
            "Save session evidence",
            "Store the current local meeting-monitor summary in the AI Report Generator.",
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
        already_saved = current_signature == st.session_state.get(
            "live_audio_saved_signature"
        )
        capture_mode = (
            f"Local system-audio capture from "
            f"{st.session_state.get('live_monitor_source', monitor.device_name)}"
        )
        if st.button(
            "Save meeting-monitor session to report history",
            type="primary",
            use_container_width=True,
            disabled=already_saved,
        ):
            _save_session(history, results, transcript_source, capture_mode)
        if already_saved:
            st.caption(
                "This session version is already saved. Capture another chunk to create a new version."
            )


def _discover_devices() -> tuple[list[CaptureDevice], str]:
    if not soundcard_available():
        return [], "The SoundCard package is not installed."
    try:
        return list_capture_devices(), ""
    except Exception as exc:
        return [], str(exc)


def render_live_audio_page(root: Path, history: list[dict[str, object]]) -> None:
    _init_live_state()
    monitor: LocalSystemAudioMonitor = st.session_state["live_system_monitor"]
    audio_classifier = _load_audio_classifier(str(root))
    text_classifier = _load_transcript_classifier(str(root))
    devices, device_error = _discover_devices()

    render_section_header(
        "Live audio detection",
        "Capture Zoom, Google Meet, Teams, or other speaker output on this computer and analyse it in short transcript and voice-feature chunks.",
        "Local meeting monitor",
    )
    render_info_banner(
        "This feature captures audio on the computer running Streamlit. It must be launched locally; "
        "a hosted Streamlit server cannot hear your laptop's meeting audio.",
        kind="warning",
        code="LOCAL ONLY",
    )
    render_info_banner(
        "Use a Windows WASAPI loopback device, Linux monitor source, or virtual cable. "
        "Audio stays on the local computer and no meeting-platform API is required.",
        kind="success",
        code="SYSTEM AUDIO",
    )

    recommended_devices = [
        device for device in devices if device.kind in {"System output", "Virtual cable"}
    ]
    render_metric_row(
        [
            {"label": "Local Capture", "value": "Ready" if recommended_devices else "Unavailable", "color": "#059669" if recommended_devices else "#DC2626"},
            {"label": "System/Cable Inputs", "value": len(recommended_devices), "color": "#2563EB"},
            {"label": "Audio SVM", "value": "Ready" if audio_classifier else "Heuristic", "color": "#059669" if audio_classifier else "#D97706"},
            {"label": "Transcript Model", "value": "Ready" if text_classifier else "Rule Demo", "color": "#059669" if text_classifier else "#D97706"},
            {"label": "Local Whisper", "value": "Ready" if WHISPER_AVAILABLE else "Install required", "color": "#059669" if WHISPER_AVAILABLE else "#DC2626"},
        ]
    )

    if device_error:
        render_info_banner(
            f"Audio device discovery failed: {device_error}",
            kind="danger",
            code="DEVICE ERROR",
        )
    elif not devices:
        render_info_banner(
            "No local audio inputs were found. Run the app on your own computer and install or enable "
            "a loopback/monitor source such as VB-Cable, VoiceMeeter, PulseAudio Monitor, or BlackHole.",
            kind="danger",
            code="NO DEVICE",
        )
    elif not recommended_devices:
        render_info_banner(
            "Only physical microphone inputs were found, and they are intentionally excluded. "
            "Enable a system-output loopback/monitor source or install a virtual audio cable.",
            kind="danger",
            code="NO SYSTEM AUDIO",
        )
    if not WHISPER_AVAILABLE:
        render_info_banner(
            "Automatic transcript analysis is not installed. Run "
            "pip install -r requirements-live.txt locally, then restart Streamlit.",
            kind="warning",
            code="TRANSCRIPT SETUP",
        )

    with st.container(border=True):
        st.subheader("Local meeting monitor setup")
        device_column, chunk_column, risk_column = st.columns([1.5, 0.75, 0.75])
        with device_column:
            selected_device = st.selectbox(
                "System-audio input",
                recommended_devices,
                format_func=lambda device: device.label,
                disabled=monitor.running or not recommended_devices,
                placeholder="No local capture device available",
                help="Choose System output first. Use Virtual cable when the meeting app is routed through VB-Cable, VoiceMeeter, or BlackHole.",
            )
        with chunk_column:
            chunk_seconds = st.slider(
                "Chunk duration",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                disabled=monitor.running,
            )
        with risk_column:
            risk_threshold = st.slider(
                "Alert threshold",
                min_value=40,
                max_value=90,
                value=70,
                step=5,
            )

        transcript_column, model_column = st.columns(2)
        with transcript_column:
            transcript_options = ["Audio only"]
            if WHISPER_AVAILABLE:
                transcript_options.insert(0, "Local Whisper")
            transcript_source = st.selectbox(
                "Transcript analysis",
                transcript_options,
                disabled=monitor.running,
                help="Install requirements-live.txt to enable automatic local transcription.",
            )
        with model_column:
            whisper_size = st.selectbox(
                "Whisper model",
                ["tiny", "base"],
                index=0,
                disabled=monitor.running or transcript_source != "Local Whisper",
                help="Tiny is recommended for keeping up with rolling chunks on CPU.",
            )

        consent = st.checkbox(
            "I have permission from meeting participants to record and analyse this audio.",
            value=False,
            disabled=monitor.running,
        )

        start_column, stop_column, clear_column = st.columns(3)
        with start_column:
            if st.button(
                "Start local monitor",
                type="primary",
                use_container_width=True,
                disabled=monitor.running or selected_device is None or not consent,
            ):
                model_ready = True
                if transcript_source == "Local Whisper":
                    with st.spinner(
                        f"Preparing local Whisper {whisper_size} before capture..."
                    ):
                        try:
                            model_ready = _load_whisper_model(whisper_size) is not None
                        except Exception as exc:
                            model_ready = False
                            st.session_state["live_audio_error"] = str(exc)
                            st.error(f"Whisper setup failed: {exc}")
                if model_ready:
                    monitor.start(selected_device, chunk_seconds=chunk_seconds)
                    st.session_state["live_monitor_source"] = selected_device.label
                    st.session_state["live_audio_error"] = ""
                    st.rerun()
        with stop_column:
            if st.button(
                "Stop monitor",
                use_container_width=True,
                disabled=not monitor.running,
            ):
                monitor.stop()
                st.rerun()
        with clear_column:
            if st.button("Clear monitor session", use_container_width=True):
                _clear_live_state()
                st.rerun()

        st.caption(
            "The Python process captures locally at 48 kHz and resamples completed chunks to "
            "16 kHz for Whisper. Raw chunks stay in memory and are discarded after analysis; "
            "only summaries are saved when you choose Save."
        )

    with st.expander("Operating-system audio routing", expanded=False):
        st.markdown(
            """
**Windows:** Choose a device labelled **System output** for WASAPI loopback. If none appears,
install VB-Cable or VoiceMeeter, route Zoom/Meet/Teams output to the cable input, then select
the matching cable output here.

**Linux/Kali:** Select the PulseAudio/PipeWire source ending in **monitor**. If it is missing,
enable the monitor source in `pavucontrol` or your PipeWire audio controls.

**macOS:** CoreAudio does not provide native loopback capture. Install BlackHole, create an
appropriate multi-output device, route meeting audio through it, and select BlackHole here.
"""
        )

    _render_monitor_fragment(
        root,
        history,
        risk_threshold,
        transcript_source,
        whisper_size,
    )
