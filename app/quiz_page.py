"""Interactive student quiz page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui_components import get_demo_data


def render_quiz_page(root: Path, history: list[dict[str, object]]) -> None:
    st.subheader("Student Scam Awareness Quiz 🎓")
    st.write("Try to identify whether each scenario is likely scam behavior or normal communication.")

    questions = get_demo_data()["quiz"]
    answers: list[tuple[object, str]] = []
    for index, question in enumerate(questions):
        st.markdown(f"**{index + 1}. {question.question}**")
        choice = st.radio(
            "Choose one",
            question.options,
            key=f"quiz_{index}",
            horizontal=True,
            label_visibility="collapsed",
        )
        answers.append((question, choice))

    if st.button("Check quiz score", type="primary"):
        score = sum(choice == question.answer for question, choice in answers)
        st.success(f"Score: {score}/{len(questions)}")
        for index, (question, choice) in enumerate(answers):
            if choice == question.answer:
                st.write(f"✅ Q{index + 1}: Correct. {question.explanation}")
            else:
                st.write(f"❌ Q{index + 1}: The better answer is **{question.answer}**. {question.explanation}")
