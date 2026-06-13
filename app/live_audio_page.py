"""Local system-audio meeting scam detection demonstration."""

from __future__ import annotations

import hashlib
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
    wav_bytes_to_audio,
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
        "live_audio_mode": "Voice recorder",
        "live_recorder_results": [],
        "live_recorder_signatures": [],
        "live_recorder_clip_count": 0,
        "live_recorder_generation": 0,
        "live_recorder_error": "",
        "live_recorder_saved_signature": "",
        "live_monitor_results": [],
        "live_system_monitor": LocalSystemAudioMonitor(),
        "live_monitor_error": "",
        "live_monitor_saved_signature": "",
        "live_monitor_chunk_index": 0,
        "live_monitor_source": "",
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
    st.session_state["live_recorder_generation"] = (
        int(st.session_state.get("live_recorder_generation", 0)) + 1
    )


def _clear_monitor_state() -> None:
    monitor = st.session_state.get("live_system_monitor")
    if isinstance(monitor, LocalSystemAudioMonitor):
        monitor.stop()
        monitor.clear()
    st.session_state["live_monitor_results"] = []
    st.session_state["live_monitor_error"] = ""
    st.session_state["live_monitor_saved_signature"] = ""
    st.session_state["live_monitor_chunk_index"] = 0
    st.session_state["live_monitor_source"] = ""


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
                    st.session_state["live_recorder_error"] = ""
                    render_analysis_ready(f"Voice sample {clip} analysed")

    if st.session_state.get("live_recorder_error"):
        st.error(f"Recording analysis failed: {st.session_state['live_recorder_error']}")
    elif not results:
        st.caption("Your transcript, confidence scores, and audio visualisations will appear after recording.")

    _render_dashboard_section(
        results,
        risk_threshold,
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


@st.fragment(run_every=1.0)
def _render_monitor_fragment(
    root: Path,
    history: list[dict[str, object]],
    risk_threshold: int,
    transcript_source: str,
    whisper_size: str,
) -> None:
    monitor: LocalSystemAudioMonitor = st.session_state["live_system_monitor"]
    results: list[dict[str, object]] = st.session_state["live_monitor_results"]
    chunks = monitor.drain_chunks(limit=2)
    audio_classifier = _load_audio_classifier(str(root))
    text_classifier = _load_transcript_classifier(str(root))

    whisper_model = None
    if chunks and transcript_source == "Local Whisper":
        try:
            whisper_model = _load_whisper_model(whisper_size)
        except Exception as exc:
            st.session_state["live_monitor_error"] = str(exc)

    for chunk in chunks:
        try:
            energy = float(np.sqrt(np.mean(np.square(chunk))))
            transcript = ""
            if whisper_model is not None and energy >= 0.002:
                transcript = transcribe_with_whisper(chunk, whisper_model)
            result = analyse_live_chunk(
                chunk,
                transcript=transcript,
                audio_classifier=audio_classifier,
                text_classifier=text_classifier,
            )
            index = int(st.session_state["live_monitor_chunk_index"]) + 1
            result["clip"] = 1
            result["clip_chunk"] = index
            result["capture_mode"] = "Local device-audio monitor"
            result["source_device"] = monitor.device_name
            st.session_state["live_monitor_chunk_index"] = index
            st.session_state["live_monitor_saved_signature"] = ""
            st.session_state["live_monitor_error"] = ""
            results.append(result)
            del results[:-60]
        except Exception as exc:
            st.session_state["live_monitor_error"] = str(exc)

    stats = monitor.stats()
    if monitor.error:
        st.error(f"Device capture stopped: {monitor.error}")
    elif st.session_state.get("live_monitor_error"):
        st.error(f"Audio analysis failed: {st.session_state['live_monitor_error']}")
    elif monitor.running:
        st.caption(
            f"Monitoring {monitor.device_name} | "
            f"{stats['captured_chunks']} captured | {stats['queued_chunks']} queued"
        )
    elif results:
        st.caption("Monitor stopped. The captured analysis remains available.")
    else:
        st.caption("Results will appear after the first completed audio chunk.")

    if int(stats["dropped_chunks"]) > 0:
        st.caption(
            f"{stats['dropped_chunks']} chunk(s) were dropped. Use tiny Whisper or a longer chunk."
        )

    _render_dashboard_section(
        results,
        risk_threshold,
        transcript_heading="Meeting transcript and chunks",
        frequency_heading="Latest system-audio spectrum",
        latest_title="Latest device-audio chunk {chunk}",
    )
    _render_save_action(
        history,
        results,
        transcript_source,
        f"Local device-audio monitor from {st.session_state.get('live_monitor_source', monitor.device_name)}",
        scan_type="System Audio",
        source_name="Device audio monitor",
        saved_signature_key="live_monitor_saved_signature",
        button_label="Save device-audio session to report history",
    )


def _discover_devices() -> tuple[list[CaptureDevice], str]:
    if not soundcard_available():
        return [], (
            "For local device capture, install requirements-device-audio.txt "
            "and restart Streamlit."
        )
    try:
        return list_capture_devices(), ""
    except Exception as exc:
        return [], str(exc)


def _render_device_audio_monitor(
    root: Path,
    history: list[dict[str, object]],
) -> None:
    monitor: LocalSystemAudioMonitor = st.session_state["live_system_monitor"]
    devices, device_error = _discover_devices()
    devices = [
        device for device in devices if device.kind in {"System output", "Virtual cable"}
    ]

    render_section_header(
        "Device audio monitor",
        "Analyse Zoom, Google Meet, Teams, or other audio playing on the same computer.",
        "Local system audio",
    )
    st.caption(
        "This optional mode runs only on a local Streamlit installation and does not affect the Voice Recorder."
    )

    if device_error:
        st.info(f"Device audio is not available here. {device_error}")
        with st.expander("Local setup instructions"):
            st.markdown(
                """
Install the project requirements on the computer running Streamlit. On Windows, select a
WASAPI system-output device or route meeting audio through VB-Cable/VoiceMeeter. On Linux,
select a PulseAudio/PipeWire monitor source. On macOS, use BlackHole.
"""
            )
        return
    if not devices:
        st.info(
            "No system-output or virtual-cable input was found. Enable a loopback/monitor source or install a virtual audio cable."
        )
        return

    with st.container(border=True):
        selected_device = st.selectbox(
            "Device audio source",
            devices,
            format_func=lambda device: device.label,
            disabled=monitor.running,
        )
        setting_a, setting_b, setting_c = st.columns(3)
        with setting_a:
            chunk_seconds = st.slider(
                "Chunk duration",
                min_value=3,
                max_value=10,
                value=5,
                disabled=monitor.running,
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
            transcript_options = ["Audio only"]
            if WHISPER_AVAILABLE:
                transcript_options.insert(0, "Local Whisper")
            transcript_source = st.selectbox(
                "Transcript analysis",
                transcript_options,
                disabled=monitor.running,
                key="monitor_transcript_source",
            )
        whisper_size = st.selectbox(
            "Whisper model",
            ["tiny", "base"],
            disabled=monitor.running or transcript_source != "Local Whisper",
            key="monitor_whisper_size",
        )
        consent = st.checkbox(
            "I have permission from meeting participants to record and analyse this audio.",
            disabled=monitor.running,
            key="monitor_consent",
        )

        start_column, stop_column, clear_column = st.columns(3)
        with start_column:
            if st.button(
                "Start monitor",
                type="primary",
                use_container_width=True,
                disabled=monitor.running or not consent,
            ):
                model_ready = True
                if transcript_source == "Local Whisper":
                    try:
                        model_ready = _load_whisper_model(whisper_size) is not None
                    except Exception as exc:
                        model_ready = False
                        st.session_state["live_monitor_error"] = str(exc)
                if model_ready:
                    monitor.start(selected_device, chunk_seconds=chunk_seconds)
                    st.session_state["live_monitor_source"] = selected_device.label
                    st.session_state["live_monitor_error"] = ""
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
            if st.button("Clear session", use_container_width=True):
                _clear_monitor_state()
                st.rerun()

    if not WHISPER_AVAILABLE:
        st.caption("Install requirements-live.txt locally to add automatic transcription.")
    with st.expander("Audio routing help"):
        st.markdown(
            """
**Windows:** choose a System output device, or route the meeting through VB-Cable/VoiceMeeter.

**Linux/Kali:** choose the PulseAudio/PipeWire source ending in `monitor`.

**macOS:** route audio through BlackHole and select that virtual device here.
"""
        )

    _render_monitor_fragment(
        root,
        history,
        risk_threshold,
        transcript_source,
        whisper_size,
    )


def render_live_audio_page(root: Path, history: list[dict[str, object]]) -> None:
    _init_live_state()
    render_section_header(
        "Live audio detection",
        "Choose a browser voice recording or monitor audio playing on the local computer.",
        "Audio analysis",
    )
    mode = st.segmented_control(
        "Audio source",
        ["Voice recorder", "Device audio monitor"],
        key="live_audio_mode",
        label_visibility="collapsed",
    )
    if mode == "Device audio monitor":
        _render_device_audio_monitor(root, history)
    else:
        monitor = st.session_state.get("live_system_monitor")
        if isinstance(monitor, LocalSystemAudioMonitor) and monitor.running:
            monitor.stop()
        _render_voice_recorder(root, history)
