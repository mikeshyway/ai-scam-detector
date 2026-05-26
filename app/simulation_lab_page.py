"""Turn-based scam simulation lab."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html

from src.explainability import (
    educational_summary,
    find_suspicious_phrases,
    highlighted_html,
    rule_based_text_prediction,
)
from src.scenarios import ScamScenario, get_scenario, get_scenarios
from src.text_classifier import load_text_artifacts
from src.recording_audio_simulation import analyze_audio_chunks, load_optional_audio_model, spectrogram_db


PHASES = ("encounter", "investigate", "defend", "complete")


@st.cache_data(show_spinner=False, max_entries=3)
def _run_chunk_analysis(audio_bytes: bytes, suffix: str, chunk_seconds: int, root: str):
    model = load_optional_audio_model(root)
    return analyze_audio_chunks(audio_bytes, suffix, chunk_seconds=chunk_seconds, model=model)


def _waveform_figure(y: np.ndarray, sr: int) -> go.Figure:
    times = np.arange(len(y)) / sr
    fig = go.Figure(go.Scatter(x=times, y=y, mode="lines", line=dict(color="#38bdf8", width=1)))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=20, b=35), xaxis_title="Time (seconds)", yaxis_title="Amplitude")
    return fig


def _spectrogram_figure(y: np.ndarray, sr: int) -> go.Figure:
    db, times, freqs = spectrogram_db(y, sr)
    fig = go.Figure(data=go.Heatmap(x=times, y=freqs, z=db, colorscale="Magma", colorbar=dict(title="dB")))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=35), xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)")
    return fig


def _init_simulation_state() -> None:
    defaults = {
        "sim_scenario_id": get_scenarios()[0].scenario_id,
        "sim_phase": "encounter",
        "sim_checkpoint": "encounter",
        "sim_score": 0,
        "sim_decision": "",
        "sim_free_text": "",
        "sim_deadline": 0.0,
        "sim_failed": False,
        "sim_feedback": "",
        "sim_started": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _set_phase(phase: str, scenario: ScamScenario) -> None:
    st.session_state.sim_phase = phase
    st.session_state.sim_checkpoint = phase
    st.session_state.sim_failed = False
    st.session_state.sim_feedback = ""
    st.session_state.sim_deadline = time.time() + scenario.time_limit_seconds


def _reset_scenario(scenario: ScamScenario) -> None:
    st.session_state.sim_scenario_id = scenario.scenario_id
    st.session_state.sim_score = 0
    st.session_state.sim_decision = ""
    st.session_state.sim_free_text = ""
    st.session_state.sim_started = True
    _set_phase("encounter", scenario)


def _time_remaining() -> int:
    return max(0, int(st.session_state.sim_deadline - time.time()))


def _expired() -> bool:
    return _time_remaining() <= 0


def _countdown(seconds: int) -> None:
    duration = max(seconds, 1)
    html(
        f"""
        <style>
        .sim-timer {{
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 12px;
            padding: 0.75rem;
            background: rgba(15, 23, 42, 0.05);
            font-family: sans-serif;
        }}
        .timer-label {{ color: #64748b; font-size: 0.78rem; }}
        .timer-value {{ font-size: 1.45rem; font-weight: 800; margin: 0.2rem 0; }}
        .timer-track {{ height: 8px; border-radius: 99px; background: rgba(148, 163, 184, 0.28); overflow: hidden; }}
        .timer-fill {{
            height: 100%;
            width: 100%;
            background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
            transform-origin: left center;
            animation: drain {duration}s linear forwards;
        }}
        @keyframes drain {{ from {{ transform: scaleX(1); }} to {{ transform: scaleX(0); }} }}
        </style>
        <div class="sim-timer">
          <div class="timer-label">Turn countdown</div>
          <div class="timer-value" id="timer-value">{seconds}s</div>
          <div class="timer-track"><div class="timer-fill" id="timer-fill"></div></div>
        </div>
        <script>
        const total = {duration};
        const started = Date.now();
        const value = document.getElementById("timer-value");
        function tick() {{
          const elapsed = (Date.now() - started) / 1000;
          const remain = Math.max(total - elapsed, 0);
          value.textContent = remain.toFixed(1) + "s";
          if (remain > 0) window.requestAnimationFrame(tick);
        }}
        tick();
        </script>
        """,
        height=105,
    )


def _predict_scenario(root: Path, scenario: ScamScenario) -> tuple[dict[str, object], object | None]:
    try:
        if scenario.channel == "Email":
            classifier = load_text_artifacts(
                root / "models" / "email_vectorizer.pkl",
                root / "models" / "email_nb.pkl",
                model_name="Email Naive Bayes",
            )
        else:
            classifier = load_text_artifacts(
                root / "models" / "transcript_vectorizer.pkl",
                root / "models" / "transcript_nb.pkl",
                model_name="Transcript Naive Bayes",
            )
        prediction = classifier.predict_one(scenario.content)
        findings = find_suspicious_phrases(scenario.content)
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
        return rule_based_text_prediction(scenario.content), None


def _probability_chart(probabilities: dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    values = [round(float(value) * 100, 2) for value in probabilities.values()]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=["#22c55e", "#ef4444"]))
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis_title="Confidence (%)",
        xaxis=dict(range=[0, 100]),
    )
    return fig


def _safe_free_text(value: str) -> bool:
    lowered = value.lower()
    safe_terms = ("verify", "official", "report", "hang up", "call back", "portal", "do not click", "bank")
    unsafe_terms = ("otp", "password", "pay", "transfer", "click")
    return any(term in lowered for term in safe_terms) and not any(term in lowered for term in unsafe_terms)


def _fail_turn() -> None:
    st.session_state.sim_failed = True
    st.session_state.sim_feedback = "Time ran out. Retry from your last checkpoint."


def _render_scenario_card(scenario: ScamScenario) -> None:
    st.markdown(
        f"""
        <div class="scenario-panel">
          <div class="scenario-meta">{scenario.channel} · {scenario.difficulty}</div>
          <h2>{scenario.title}</h2>
          <pre>{scenario.content}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_encounter(scenario: ScamScenario) -> None:
    st.subheader("Phase 1 - Encounter")
    st.write("You are the student receiving this message. Choose what you would do before time runs out.")
    _countdown(_time_remaining())
    _render_scenario_card(scenario)

    option_labels = [option.label for option in scenario.options] + ["Write my own decision"]
    choice = st.radio("Your move", option_labels, key="sim_decision_choice")
    free_text = ""
    if choice == "Write my own decision":
        free_text = st.text_area("Type your decision", key="sim_free_decision", height=100)

    if st.button("Submit decision", type="primary"):
        if _expired():
            _fail_turn()
            return

        is_safe = False
        feedback = ""
        if choice == "Write my own decision":
            is_safe = _safe_free_text(free_text)
            feedback = "Your written decision shows safe verification behavior." if is_safe else "Your written decision needs clearer verification or reporting steps."
        else:
            selected = next(option for option in scenario.options if option.label == choice)
            is_safe = selected.safe
            feedback = selected.feedback

        st.session_state.sim_score += 1 if is_safe else 0
        st.session_state.sim_decision = choice if choice != "Write my own decision" else free_text
        st.session_state.sim_feedback = feedback
        _set_phase("investigate", scenario)
        st.rerun()


def _render_investigate(root: Path, scenario: ScamScenario) -> None:
    st.subheader("Phase 2 - Investigate")
    st.write("Now the AI analysis explains what happened and why the message is risky.")
    _countdown(_time_remaining())

    st.info(st.session_state.sim_feedback or "Decision recorded.")
    result, _classifier = _predict_scenario(root, scenario)
    findings = list(result.get("findings", []))
    st.write(educational_summary(str(result["label_name"]), float(result["confidence"]), findings))
    st.plotly_chart(_probability_chart(result["probabilities"]), use_container_width=True)

    st.markdown(highlighted_html(scenario.content, findings), unsafe_allow_html=True)
    if findings:
        st.dataframe(pd.DataFrame(findings)[["phrase", "category", "reason"]], hide_index=True, use_container_width=True)

    st.subheader("Attacker Motive")
    st.write(scenario.attacker_motive)
    st.write("Key indicators: " + ", ".join(scenario.indicators))

    if st.button("Continue to defense phase", type="primary"):
        if _expired():
            _fail_turn()
            return
        _set_phase("defend", scenario)
        st.rerun()


def _render_defend(history: list[dict[str, object]], scenario: ScamScenario) -> None:
    st.subheader("Phase 3 - Defend")
    st.write("Apply what you learned. Choose the safest answers before the countdown ends.")
    _countdown(_time_remaining())

    st.markdown("**Defender steps for this scenario:**")
    for step in scenario.defense_steps:
        st.write(f"- {step}")

    answers: list[tuple[object, str]] = []
    for index, question in enumerate(scenario.quiz):
        st.markdown(f"**Check {index + 1}: {question.question}**")
        choice = st.radio(
            "Choose one",
            question.options,
            key=f"sim_quiz_{scenario.scenario_id}_{index}",
            horizontal=True,
            label_visibility="collapsed",
        )
        answers.append((question, choice))

    if st.button("Lock in defense answers", type="primary"):
        if _expired():
            _fail_turn()
            return
        correct = sum(question.answer == choice for question, choice in answers)
        st.session_state.sim_score += correct
        st.session_state.sim_feedback = f"Defense quiz score: {correct}/{len(answers)}"
        st.session_state.sim_phase = "complete"
        history.insert(
            0,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "Simulation",
                "prediction": "Completed",
                "confidence": round(st.session_state.sim_score / (len(answers) + 1) * 100, 2),
                "model": "Scam Simulation Lab",
                "preview": scenario.title,
            },
        )
        st.rerun()


def _render_complete(scenario: ScamScenario) -> None:
    st.subheader("Scenario Complete")
    st.success(st.session_state.sim_feedback)
    st.metric("Simulation score", st.session_state.sim_score)
    st.write("You moved from encounter, to AI investigation, to defense practice.")
    if st.button("Retry this scenario"):
        _reset_scenario(scenario)
        st.rerun()


def _render_recording_simulation(root: Path, history: list[dict[str, object]]) -> None:
    st.subheader("Uploaded Recording Scam Voice Detection")
    st.write(
        "Upload a meeting or call recording and optionally a transcript. The app splits audio into "
        "5-10 second chunks, extracts MFCC features per chunk, and shows how risk confidence changes "
        "across the uploaded file."
    )

    upload_col, settings_col = st.columns([0.62, 0.38])
    with upload_col:
        uploaded_audio = st.file_uploader(
            "Upload exported Zoom/Teams/Google Meet or phone-call recording audio",
            type=["wav", "flac", "mp3", "m4a"],
            key="sim_recording_audio",
        )
        uploaded_transcript = st.file_uploader(
            "Upload meeting transcript file",
            type=["txt", "csv"],
            key="sim_recording_transcript",
        )
    with settings_col:
        chunk_seconds = st.slider("Chunk size", min_value=5, max_value=10, value=5, step=1)
        st.info("This uses uploaded-file analysis only.")

    transcript_text = ""
    if uploaded_transcript is not None:
        if Path(uploaded_transcript.name).suffix.lower() == ".txt":
            transcript_text = uploaded_transcript.getvalue().decode("utf-8", errors="ignore")
        else:
            transcript_text = pd.read_csv(uploaded_transcript).astype(str).head(20).to_string(index=False)
    transcript_text = st.text_area("Paste or edit transcript text", value=transcript_text, height=140)

    if uploaded_audio is None:
        st.info("Upload an audio recording to run chunk-by-chunk analysis.")
        return

    audio_bytes = uploaded_audio.getvalue()
    suffix = Path(uploaded_audio.name).suffix.lower()
    st.audio(audio_bytes)

    if st.button("Run rolling chunk analysis", type="primary", use_container_width=True):
        try:
            results, y, sr = _run_chunk_analysis(audio_bytes, suffix, chunk_seconds, str(root))
        except Exception as exc:
            st.error(f"Chunk analysis failed: {exc}")
            st.caption("For MP3/M4A files, PyDub or librosa may require ffmpeg on the host machine.")
            return

        if results.empty:
            st.warning("No usable chunks were extracted from the recording.")
            return

        st.subheader("Chunk Detection Dashboard")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Chunks", len(results))
        metric_cols[1].metric("Peak confidence", f"{results['confidence'].max():.1f}%")
        metric_cols[2].metric("Average confidence", f"{results['confidence'].mean():.1f}%")
        metric_cols[3].metric("Engine", str(results["engine"].iloc[0]))

        line = go.Figure(
            go.Scatter(
                x=results["end_sec"],
                y=results["confidence"],
                mode="lines+markers",
                line=dict(color="#ef4444", width=2),
            )
        )
        line.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=35), xaxis_title="Call time (seconds)", yaxis_title="Risk confidence (%)", yaxis=dict(range=[0, 100]))
        st.plotly_chart(line, use_container_width=True)
        st.dataframe(results, hide_index=True, use_container_width=True)

        st.subheader("Waveform")
        st.plotly_chart(_waveform_figure(y, sr), use_container_width=True)
        st.subheader("Spectrogram")
        st.plotly_chart(_spectrogram_figure(y, sr), use_container_width=True)

        if transcript_text.strip():
            findings = find_suspicious_phrases(transcript_text)
            st.subheader("Transcript Warning Indicators")
            st.markdown(highlighted_html(transcript_text, findings), unsafe_allow_html=True)
            if findings:
                st.dataframe(pd.DataFrame(findings)[["phrase", "category", "reason"]], hide_index=True, use_container_width=True)

        history.insert(
            0,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "Simulation",
                "prediction": "Uploaded recording chunk analysis",
                "confidence": round(float(results["confidence"].max()), 2),
                "model": str(results["engine"].iloc[0]),
                "preview": uploaded_audio.name,
            },
        )


def _render_decision_simulation(root: Path, history: list[dict[str, object]]) -> None:
    _init_simulation_state()
    st.write(
        "A turn-based training mode where students experience scam pressure, investigate it with AI, "
        "then practice the safest defense."
    )

    scenarios = get_scenarios()
    scenario_titles = {scenario.title: scenario.scenario_id for scenario in scenarios}
    current = get_scenario(st.session_state.sim_scenario_id)
    selected_title = st.selectbox(
        "Choose a case scenario",
        list(scenario_titles.keys()),
        index=list(scenario_titles.values()).index(current.scenario_id),
    )
    selected = get_scenario(scenario_titles[selected_title])

    if not st.session_state.sim_started:
        st.markdown(
            f"""
            <div class="scenario-panel">
              <div class="scenario-meta">{selected.channel} - {selected.difficulty}</div>
              <h2>{selected.title}</h2>
              <p>Select this case and press start. The scenario content, countdown, and quiz only appear after the session begins.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start session", type="primary", use_container_width=True):
            _reset_scenario(selected)
            st.rerun()
        return

    if selected.scenario_id != current.scenario_id:
        st.session_state.sim_scenario_id = selected.scenario_id
        st.session_state.sim_started = False
        st.rerun()

    if st.session_state.sim_deadline <= 0:
        _set_phase(st.session_state.sim_phase, selected)

    if st.session_state.sim_failed:
        st.error(st.session_state.sim_feedback)
        if st.button("Retry from checkpoint", type="primary"):
            _set_phase(st.session_state.sim_checkpoint, selected)
            st.rerun()
        return

    phase = st.session_state.sim_phase
    phase_index = PHASES.index(phase) + 1
    st.progress(phase_index / len(PHASES), text=f"Phase: {phase.title()}")

    if phase == "encounter":
        _render_encounter(selected)
    elif phase == "investigate":
        _render_investigate(root, selected)
    elif phase == "defend":
        _render_defend(history, selected)
    else:
        _render_complete(selected)


def render_simulation_lab_page(root: Path, history: list[dict[str, object]]) -> None:
    st.subheader("Scam Simulation Lab")
    recording_tab, scenario_tab = st.tabs(["Uploaded Recording Analysis", "Turn-Based Scenario"])
    with recording_tab:
        _render_recording_simulation(root, history)
    with scenario_tab:
        _render_decision_simulation(root, history)
