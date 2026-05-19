"""Turn-based scam simulation lab."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

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


PHASES = ("encounter", "investigate", "defend", "complete")


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
    _set_phase("encounter", scenario)


def _time_remaining() -> int:
    return max(0, int(st.session_state.sim_deadline - time.time()))


def _expired() -> bool:
    return _time_remaining() <= 0


def _countdown(seconds: int) -> None:
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
        .timer-fill {{ height: 100%; width: 100%; background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444); transition: width 0.2s linear; }}
        </style>
        <div class="sim-timer">
          <div class="timer-label">Turn countdown</div>
          <div class="timer-value" id="timer-value">{seconds}s</div>
          <div class="timer-track"><div class="timer-fill" id="timer-fill"></div></div>
        </div>
        <script>
        const total = {max(seconds, 1)};
        const started = Date.now();
        const value = document.getElementById("timer-value");
        const fill = document.getElementById("timer-fill");
        function tick() {{
          const elapsed = Math.floor((Date.now() - started) / 1000);
          const remain = Math.max(total - elapsed, 0);
          value.textContent = remain + "s";
          fill.style.width = (remain / total * 100) + "%";
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


def render_simulation_lab_page(root: Path, history: list[dict[str, object]]) -> None:
    _init_simulation_state()
    st.subheader("Scam Simulation Lab")
    st.write(
        "A turn-based training mode where students experience scam pressure, investigate it with AI, "
        "then practice the safest defense."
    )

    scenarios = get_scenarios()
    scenario_titles = {scenario.title: scenario.scenario_id for scenario in scenarios}
    current = get_scenario(st.session_state.sim_scenario_id)
    selected_title = st.selectbox(
        "Choose scenario",
        list(scenario_titles.keys()),
        index=list(scenario_titles.values()).index(current.scenario_id),
    )
    selected = get_scenario(scenario_titles[selected_title])
    if selected.scenario_id != current.scenario_id:
        _reset_scenario(selected)
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
