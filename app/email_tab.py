"""Email phishing detection tab."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_components import get_demo_data, render_demo_notice, render_section_header
from src.explainability import (
    educational_summary,
    find_suspicious_phrases,
    highlighted_html,
    rule_based_text_prediction,
    top_model_terms,
)
from src.text_classifier import load_text_artifacts


MODEL_FILES = {
    "Naive Bayes": ("email_vectorizer.pkl", "email_nb.pkl"),
    "Decision Tree": ("email_vectorizer.pkl", "email_dt.pkl"),
}


@st.cache_resource(show_spinner=False)
def _load_email_classifier(root: str, model_choice: str):
    vectorizer_name, model_name = MODEL_FILES[model_choice]
    return load_text_artifacts(
        Path(root) / "models" / vectorizer_name,
        Path(root) / "models" / model_name,
        model_name=f"Email {model_choice}",
    )


def _read_uploaded_text(uploaded_file) -> str | pd.DataFrame | None:
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    st.warning("Only .txt and .csv files are supported in this tab.")
    return None


def _predict_text(root: Path, text: str, model_choice: str) -> tuple[dict[str, object], object | None]:
    try:
        classifier = _load_email_classifier(str(root), model_choice)
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
        st.warning("Email model artifacts were not found, so this result uses educational demo rules.")
        return result, None


def _confidence_chart(probabilities: dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    values = [round(float(value) * 100, 2) for value in probabilities.values()]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=["#3b82f6", "#ef4444"]))
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Confidence (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
    )
    return fig


def _record(history: list[dict[str, object]], result: dict[str, object], text: str) -> None:
    history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Email",
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

    if int(result["label"]) == 1:
        st.error(f"{label} - {confidence * 100:.1f}% confidence")
    else:
        st.success(f"{label} - {confidence * 100:.1f}% confidence")

    st.write(educational_summary(label, confidence, findings))
    st.plotly_chart(_confidence_chart(result["probabilities"]), use_container_width=True)

    if findings:
        st.subheader("Highlighted warning signs")
        st.markdown(highlighted_html(text, findings), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(findings)[["phrase", "category", "reason"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No suspicious phrase rules were triggered.")

    if classifier is not None:
        terms = top_model_terms(text, classifier.vectorizer, classifier.model)
        if terms:
            st.subheader("Model-influential terms")
            st.dataframe(pd.DataFrame(terms), hide_index=True, use_container_width=True)


def render_email_tab(root: Path, history: list[dict[str, object]]) -> None:
    render_demo_notice(root)
    render_section_header(
        "Scan email or message content",
        "Paste one message, upload a text file, or inspect multiple CSV rows.",
        "Text evidence",
    )
    demo_emails = get_demo_data()["emails"]

    controls, input_area = st.columns([0.28, 0.72])
    with controls:
        model_choice = st.radio("Text model", list(MODEL_FILES.keys()), horizontal=False)
        demo_choice = st.selectbox(
            "Try synthetic demo email",
            ["None"] + demo_emails["sample_id"].tolist(),
        )
        uploaded_file = st.file_uploader("Upload email text or CSV", type=["txt", "csv"], key="email_upload")
        analyze_button = st.button("Analyze email", type="primary", use_container_width=True)

    uploaded = _read_uploaded_text(uploaded_file)
    if demo_choice != "None":
        selected = demo_emails.loc[demo_emails["sample_id"] == demo_choice].iloc[0]
        default_text = str(selected["text"])
    elif isinstance(uploaded, str):
        default_text = uploaded
    else:
        default_text = ""
    with input_area:
        text = st.text_area(
            "Paste suspicious email or message",
            value=default_text,
            height=260,
            placeholder="Paste an email, campus message, scholarship offer, or job/internship message here.",
        )

    if isinstance(uploaded, pd.DataFrame):
        st.subheader("Batch CSV analysis")
        text_column = st.selectbox("Text column", uploaded.columns)
        if st.button("Analyze CSV rows", use_container_width=True):
            texts = uploaded[text_column].fillna("").astype(str).tolist()
            try:
                classifier = _load_email_classifier(str(root), model_choice)
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
                st.warning("Email model artifacts were not found, so batch results use educational demo rules.")
            st.dataframe(results, hide_index=True, use_container_width=True)

    if analyze_button:
        if not text.strip():
            st.warning("Paste text or upload a .txt file first.")
            return
        result, classifier = _predict_text(root, text, model_choice)
        _record(history, result, text)
        _display_result(result, text, classifier)
