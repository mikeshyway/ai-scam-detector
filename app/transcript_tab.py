"""Call and meeting transcript scam detection tab with browser voice recorder."""

from __future__ import annotations

import hashlib
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
        "transcript_recorder_results": [],
        "transcript_recorder_signatures": [],
        "transcript_recorder_clip_count": 0,
        "transcript_recorder_generation": 0,
        "transcript_recorder_error": "",
        "transcript_recorder_carousel_index": 0,
        "transcript_pending_voice_analysis": False,
        "transcript_uploaded_audio_results": [],
        "transcript_uploaded_audio_signatures": [],
        "transcript_uploaded_audio_clip_count": 0,
        "transcript_uploaded_audio_error": "",
        "transcript_uploaded_audio_carousel_index": 0,
        "transcript_pending_uploaded_audio_analysis": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_recorder_state() -> None:
    st.session_state["transcript_recorder_results"] = []
    st.session_state["transcript_recorder_signatures"] = []
    st.session_state["transcript_recorder_clip_count"] = 0
    st.session_state["transcript_recorder_error"] = ""
    st.session_state["transcript_recorder_carousel_index"] = 0
    st.session_state["transcript_recorder_generation"] = (
        int(st.session_state.get("transcript_recorder_generation", 0)) + 1
    )


def _clear_uploaded_audio_state() -> None:
    st.session_state["transcript_uploaded_audio_results"] = []
    st.session_state["transcript_uploaded_audio_signatures"] = []
    st.session_state["transcript_uploaded_audio_clip_count"] = 0
    st.session_state["transcript_uploaded_audio_error"] = ""
    st.session_state["transcript_uploaded_audio_carousel_index"] = 0
    st.session_state["transcript_pending_uploaded_audio_analysis"] = False


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


def _render_voice_recorder(root: Path) -> int:
    results: list[dict[str, object]] = st.session_state["transcript_recorder_results"]
    audio_classifier = _load_audio_classifier(str(root))
    behavioral_classifier = _load_behavioral_classifier(str(root))
    text_classifier = _load_transcript_classifier_safe(str(root))

    render_section_header(
        "Voice recorder",
        "Record a short voice sample in the browser, stop recording, and analyse it immediately.",
        "Browser recording",
    )
    with st.container(border=True):
        setting_a, setting_b, setting_c = st.columns(3)
        with setting_a:
            chunk_seconds = st.slider("Analysis chunk", min_value=3, max_value=10, value=5, key="transcript_recorder_chunk_seconds")
        with setting_b:
            risk_threshold = st.slider("Alert threshold", min_value=40, max_value=90, value=70, step=5, key="transcript_recorder_risk_threshold")
        with setting_c:
            transcript_options = ["Manual transcript", "Audio only"]
            if WHISPER_AVAILABLE:
                transcript_options.insert(0, "Local Whisper")
            transcript_source = st.selectbox("Transcript analysis", transcript_options, key="transcript_recorder_transcript_source")

        whisper_size = "tiny"
        manual_transcript = ""
        if transcript_source == "Local Whisper":
            whisper_size = st.selectbox("Whisper model", ["tiny", "base"], key="transcript_recorder_whisper_size")
        elif transcript_source == "Manual transcript":
            manual_transcript = st.text_area(
                "Spoken words",
                placeholder="Optional: type what was said so transcript scam analysis can run.",
                height=80,
                key="transcript_recorder_manual_transcript",
            )

        recorded_audio = st.audio_input(
            "Record voice sample",
            sample_rate=16_000,
            key=f"transcript_voice_recorder_{int(st.session_state['transcript_recorder_generation'])}",
        )

        action_a, action_b = st.columns(2)
        with action_a:
            if recorded_audio is not None and st.button("Record another sample", use_container_width=True, key="transcript_record_another"):
                st.session_state["transcript_recorder_generation"] += 1
                st.rerun()
        with action_b:
            if st.button("Clear recorder session", use_container_width=True, key="transcript_clear_recorder"):
                _clear_recorder_state()
                st.rerun()

    if not WHISPER_AVAILABLE:
        st.caption("Automatic transcription is optional. Manual transcript and audio-only analysis remain available.")

    if recorded_audio is not None:
        recorded_bytes = recorded_audio.getvalue()
        settings = json.dumps([chunk_seconds, transcript_source, whisper_size, manual_transcript], ensure_ascii=True).encode("utf-8")
        signature = hashlib.sha256(recorded_bytes + settings).hexdigest()
        signatures = list(st.session_state["transcript_recorder_signatures"])

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
                        behavioral_classifier=behavioral_classifier,
                    )
                except Exception as exc:
                    st.session_state["transcript_recorder_error"] = str(exc)
                else:
                    clip = int(st.session_state["transcript_recorder_clip_count"]) + 1
                    for chunk_index, result in enumerate(processed, 1):
                        result["clip"] = clip
                        result["clip_chunk"] = chunk_index
                        result["capture_mode"] = "Browser voice recorder"
                    results.extend(processed)
                    del results[:-60]
                    signatures.append(signature)
                    st.session_state["transcript_recorder_signatures"] = signatures[-60:]
                    st.session_state["transcript_recorder_clip_count"] = clip
                    st.session_state["transcript_recorder_carousel_index"] = max(0, len(_recording_groups(results)) - 1)
                    st.session_state["transcript_recorder_error"] = ""
                    st.session_state["transcript_pending_voice_analysis"] = True
                    render_analysis_ready(f"Voice sample {clip} analysed")

    if st.session_state.get("transcript_recorder_error"):
        st.error(f"Recording analysis failed: {st.session_state['transcript_recorder_error']}")
    elif not results:
        st.caption("Voice recording results appear here after recording.")

    return risk_threshold


def _recorder_transcript_text() -> str:
    """Return the usable transcript text generated from browser voice recordings."""

    results = st.session_state.get("transcript_recorder_results", [])
    return _transcript_text_from_results(results)


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
                "Source": "Browser voice recording",
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
        text_blocks.append("[Browser voice transcript]\n" + voice_text.strip())
    if use_uploaded_audio and has_uploaded_audio_text:
        text_blocks.append("[Uploaded audio transcript]\n" + uploaded_audio_text.strip())
    if use_text and has_transcript_text:
        text_blocks.append("[Uploaded / pasted transcript]\n" + transcript_text.strip())

    combined_text = "\n\n".join(text_blocks).strip()

    if not combined_text:
        if (use_voice and has_voice_results) or (use_uploaded_audio and has_uploaded_audio_results):
            st.info(
                "Audio was analysed for voice authenticity and behavioral signals, but no speech transcript was available. "
                "Choose Local Whisper or Manual transcript if you also want transcript scam analysis."
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

def render_transcript_tab(root: Path, history: list[dict[str, object]]) -> None:
    _init_transcript_voice_state()

    render_detection_tool_intro(
        title="Voice Transcript",
        description=(
            "Record speech directly in the browser, upload an audio recording, upload/paste "
            "a transcript, or combine sources for transcript scam analysis."
        ),
        icon="solar:microphone-3-bold-duotone",
        accent="purple",
    )

    render_section_header(
        "Transcript analysis input",
        "Choose one or both sources. Recording-only still produces voice/audio results even when no text transcript exists.",
        "Unified input",
    )

    render_content_card_open("violet")

    use_voice = st.checkbox(
        "Use voice recording",
        value=True,
        key="transcript_use_voice",
    )
    use_uploaded_audio = st.checkbox(
        "Use uploaded audio recording",
        value=False,
        key="transcript_use_uploaded_audio",
    )
    use_text = st.checkbox(
        "Use uploaded / pasted transcript",
        value=False,
        key="transcript_use_text",
    )

    if not use_voice and not use_uploaded_audio and not use_text:
        st.warning("Select at least one source to analyse.")

    st.markdown("<hr style='border-color:rgba(148,163,184,.18);margin:0.8rem 0 1rem 0;'>", unsafe_allow_html=True)

    top_left, top_right = st.columns([0.50, 0.50])

    risk_threshold = 70

    with top_left:
        if use_voice or use_uploaded_audio:
            st.markdown("#### Audio evidence")
            st.caption("Record in the browser or upload an existing call/voice recording for the same audio analysis pipeline.")

            setting_a, setting_b = st.columns(2)
            with setting_a:
                chunk_seconds = st.slider(
                    "Chunk",
                    min_value=3,
                    max_value=10,
                    value=5,
                    key="transcript_recorder_chunk_seconds",
                )
            with setting_b:
                risk_threshold = st.slider(
                    "Alert",
                    min_value=40,
                    max_value=90,
                    value=70,
                    step=5,
                    key="transcript_recorder_risk_threshold",
                )

            transcript_options = ["Manual transcript", "Audio only"]
            if WHISPER_AVAILABLE:
                transcript_options.insert(0, "Local Whisper")
            transcript_source = st.selectbox(
                "Transcript analysis",
                transcript_options,
                key="transcript_recorder_transcript_source",
            )

            whisper_size = "tiny"
            manual_transcript = ""
            if transcript_source == "Local Whisper":
                whisper_size = st.selectbox(
                    "Whisper model",
                    ["tiny", "base"],
                    key="transcript_recorder_whisper_size",
                )
            elif transcript_source == "Manual transcript":
                manual_transcript = st.text_area(
                    "Spoken words",
                    placeholder="Optional: type what was said.",
                    height=68,
                    key="transcript_recorder_manual_transcript",
                )

            if not WHISPER_AVAILABLE:
                st.caption("Whisper unavailable. Use Manual transcript or Audio only.")

            audio_classifier = _load_audio_classifier(str(root))
            behavioral_classifier = _load_behavioral_classifier(str(root))
            text_classifier = _load_transcript_classifier_safe(str(root))

            if use_voice:
                st.markdown("#### Voice recorder")
                recorded_audio = st.audio_input(
                    "Record voice sample",
                    sample_rate=16_000,
                    key=f"transcript_voice_recorder_{int(st.session_state['transcript_recorder_generation'])}",
                )

                action_a, action_b = st.columns(2)
                with action_a:
                    if recorded_audio is not None and st.button(
                        "Record another",
                        use_container_width=True,
                        key="transcript_record_another",
                    ):
                        st.session_state["transcript_recorder_generation"] += 1
                        st.rerun()
                with action_b:
                    if st.button(
                        "Clear recording",
                        use_container_width=True,
                        key="transcript_clear_recorder",
                    ):
                        _clear_recorder_state()
                        st.rerun()

                if recorded_audio is not None:
                    recorded_bytes = recorded_audio.getvalue()
                    settings = json.dumps(
                        [chunk_seconds, transcript_source, whisper_size, manual_transcript],
                        ensure_ascii=True,
                    ).encode("utf-8")
                    signature = hashlib.sha256(recorded_bytes + settings).hexdigest()
                    signatures = list(st.session_state["transcript_recorder_signatures"])

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
                                    behavioral_classifier=behavioral_classifier,
                                )
                            except Exception as exc:
                                st.session_state["transcript_recorder_error"] = str(exc)
                            else:
                                results = st.session_state["transcript_recorder_results"]
                                clip = int(st.session_state["transcript_recorder_clip_count"]) + 1
                                for chunk_index, result in enumerate(processed, 1):
                                    result["clip"] = clip
                                    result["clip_chunk"] = chunk_index
                                    result["capture_mode"] = "Browser voice recorder"
                                results.extend(processed)
                                del results[:-60]
                                signatures.append(signature)
                                st.session_state["transcript_recorder_signatures"] = signatures[-60:]
                                st.session_state["transcript_recorder_clip_count"] = clip
                                st.session_state["transcript_recorder_carousel_index"] = max(
                                    0,
                                    len(_recording_groups(results)) - 1,
                                )
                                st.session_state["transcript_recorder_error"] = ""
                                st.session_state["transcript_pending_voice_analysis"] = True
                                render_analysis_ready(f"Voice sample {clip} analysed")

                if st.session_state.get("transcript_recorder_error"):
                    st.error(f"Recording analysis failed: {st.session_state['transcript_recorder_error']}")

            if use_uploaded_audio:
                st.markdown("#### Uploaded audio recording")
                st.caption("Upload a recorded call or voice sample. WAV and FLAC are most reliable; MP3 may require ffmpeg.")
                uploaded_audio = st.file_uploader(
                    "Upload audio recording",
                    type=["wav", "mp3", "flac"],
                    key="transcript_audio_upload",
                )

                if uploaded_audio is not None:
                    uploaded_audio_bytes = uploaded_audio.getvalue()
                    suffix = Path(uploaded_audio.name).suffix.lower()
                    mime_type = {
                        ".wav": "audio/wav",
                        ".mp3": "audio/mpeg",
                        ".flac": "audio/flac",
                    }.get(suffix, "audio/wav")

                    st.audio(uploaded_audio_bytes, format=mime_type)
                    st.caption(
                        f"File: `{uploaded_audio.name}` | Type: `{suffix.lstrip('.').upper()}` | "
                        f"Size: {len(uploaded_audio_bytes) / 1024:.1f} KB"
                    )

                    upload_action_a, upload_action_b = st.columns(2)
                    with upload_action_a:
                        analyze_uploaded_audio = st.button(
                            "Analyze uploaded audio",
                            use_container_width=True,
                            type="primary",
                            key="transcript_analyze_uploaded_audio",
                        )
                    with upload_action_b:
                        if st.button(
                            "Clear uploaded audio results",
                            use_container_width=True,
                            key="transcript_clear_uploaded_audio",
                        ):
                            _clear_uploaded_audio_state()
                            st.rerun()

                    if analyze_uploaded_audio:
                        settings = json.dumps(
                            [
                                uploaded_audio.name,
                                chunk_seconds,
                                transcript_source,
                                whisper_size,
                                manual_transcript,
                            ],
                            ensure_ascii=True,
                        ).encode("utf-8")
                        signature = hashlib.sha256(uploaded_audio_bytes + settings).hexdigest()
                        signatures = list(st.session_state["transcript_uploaded_audio_signatures"])

                        if not uploaded_audio_bytes:
                            st.session_state["transcript_uploaded_audio_error"] = "The uploaded audio file was empty."
                        elif signature in signatures:
                            st.session_state["transcript_pending_uploaded_audio_analysis"] = True
                            render_analysis_ready("Uploaded audio already analysed")
                        else:
                            with st.spinner("Analyzing uploaded audio..."):
                                try:
                                    whisper_model = None
                                    if transcript_source == "Local Whisper":
                                        whisper_model = _load_whisper_model(whisper_size)
                                        if whisper_model is None:
                                            raise RuntimeError("Local Whisper could not be loaded.")

                                    processed = _process_uploaded_audio(
                                        uploaded_audio_bytes,
                                        suffix,
                                        chunk_seconds=chunk_seconds,
                                        transcript_source=transcript_source,
                                        manual_transcript=manual_transcript,
                                        whisper_model=whisper_model,
                                        audio_classifier=audio_classifier,
                                        text_classifier=text_classifier,
                                        behavioral_classifier=behavioral_classifier,
                                    )
                                except Exception as exc:
                                    st.session_state["transcript_uploaded_audio_error"] = str(exc)
                                else:
                                    uploaded_results = st.session_state["transcript_uploaded_audio_results"]
                                    clip = int(st.session_state["transcript_uploaded_audio_clip_count"]) + 1
                                    for chunk_index, result in enumerate(processed, 1):
                                        result["clip"] = clip
                                        result["clip_chunk"] = chunk_index
                                        result["capture_mode"] = "Uploaded audio"
                                        result["source_filename"] = uploaded_audio.name
                                    uploaded_results.extend(processed)
                                    del uploaded_results[:-60]
                                    signatures.append(signature)
                                    st.session_state["transcript_uploaded_audio_signatures"] = signatures[-60:]
                                    st.session_state["transcript_uploaded_audio_clip_count"] = clip
                                    st.session_state["transcript_uploaded_audio_carousel_index"] = max(
                                        0,
                                        len(_recording_groups(uploaded_results)) - 1,
                                    )
                                    st.session_state["transcript_uploaded_audio_error"] = ""
                                    st.session_state["transcript_pending_uploaded_audio_analysis"] = True
                                    render_analysis_ready("Uploaded audio analysis complete - results ready below")

                if st.session_state.get("transcript_uploaded_audio_error"):
                    st.error(f"Uploaded audio analysis failed: {st.session_state['transcript_uploaded_audio_error']}")

        else:
            st.info("Audio input is disabled for this analysis.")

    with top_right:
        st.markdown("#### Transcript text")
        st.caption("Paste/upload transcript text, or leave disabled for recording-only analysis.")

        uploaded_file = None

        if use_text:
            uploaded_file = st.file_uploader(
                "Upload TXT/CSV",
                type=["txt", "csv"],
                key="transcript_upload",
            )

            uploaded = _read_upload(uploaded_file)
            if isinstance(uploaded, str):
                default_text = uploaded
            else:
                default_text = ""

            text = st.text_area(
                "Paste transcript",
                value=default_text,
                height=210,
                placeholder="Paste a call, Zoom, Teams, or Google Meet transcript here.",
            )
        else:
            uploaded = None
            text = ""
            st.text_area(
                "Paste transcript",
                value="",
                height=210,
                placeholder="Enable uploaded / pasted transcript to use this field.",
                disabled=True,
            )

        recorder_results = st.session_state.get("transcript_recorder_results", [])
        if not isinstance(recorder_results, list):
            recorder_results = []
        uploaded_audio_results = st.session_state.get("transcript_uploaded_audio_results", [])
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

        analyze_button = st.button(
            "Analyze selected source(s)",
            type="primary",
            use_container_width=True,
            disabled=not (use_voice or use_uploaded_audio or use_text),
        )

    render_content_card_close()

    if isinstance(uploaded, pd.DataFrame) and use_text:
        render_section_header("Batch transcript CSV analysis", eyebrow="Multiple rows")
        render_content_card_open("green")
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
                st.warning("Transcript model artifacts were not found, so batch results use demo rules.")
            st.dataframe(results, hide_index=True, use_container_width=True)
        render_content_card_close()

    should_auto_show_voice = (
        (use_voice or use_uploaded_audio)
        and not use_text
        and (
            bool(st.session_state.get("transcript_pending_voice_analysis"))
            or bool(st.session_state.get("transcript_pending_uploaded_audio_analysis"))
        )
    )

    if analyze_button or should_auto_show_voice:
        recorder_results = st.session_state.get("transcript_recorder_results", [])
        if not isinstance(recorder_results, list):
            recorder_results = []
        uploaded_audio_results = st.session_state.get("transcript_uploaded_audio_results", [])
        if not isinstance(uploaded_audio_results, list):
            uploaded_audio_results = []

        voice_text = _recorder_transcript_text()
        uploaded_audio_text = _uploaded_audio_transcript_text()

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
