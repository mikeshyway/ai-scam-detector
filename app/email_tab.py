"""Email phishing detection tab."""

from __future__ import annotations

from datetime import datetime
import html
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_components import (
    apply_chart_theme,
    get_demo_data,
    render_analysis_ready,
    render_content_card_close,
    render_content_card_open,
    render_demo_notice,
    render_section_header,
    render_soft_panel,
)
from src.explainability import (
    find_legitimate_indicators,
    find_suspicious_phrases,
    type_intention,
    top_model_terms,
)
from src.text_classifier import load_text_artifacts


MODEL_FILES = {
    "Naive Bayes": ("email_vectorizer.pkl", "email_nb.pkl"),
    "Decision Tree": ("email_vectorizer.pkl", "email_dt.pkl"),
    "Support Vector Machine": ("email_vectorizer.pkl", "email_svm.pkl"),
    "Random Forest": ("email_vectorizer.pkl", "email_rf.pkl"),
    "XGBoost": ("email_vectorizer.pkl", "email_xgb.pkl"),
    "Best Model": ("email_vectorizer.pkl", "email_best.pkl"),
}

DISPLAY_MODELS = [
    "Naive Bayes",
    "Decision Tree",
    "Support Vector Machine",
    "Random Forest",
    "XGBoost",
]


@st.cache_resource(show_spinner=False)
def _load_email_classifier(root: str, model_choice: str):
    vectorizer_name, model_name = MODEL_FILES[model_choice]
    return load_text_artifacts(
        Path(root) / "models" / vectorizer_name,
        Path(root) / "models" / model_name,
        model_name=f"Email {model_choice}",
    )


def _model_status(root: Path) -> pd.DataFrame:
    rows = []
    for model_name, (vectorizer_file, model_file) in MODEL_FILES.items():
        vectorizer_exists = (root / "models" / vectorizer_file).exists()
        model_exists = (root / "models" / model_file).exists()
        rows.append(
            {
                "Model": model_name,
                "Vectorizer": "Ready" if vectorizer_exists else "Missing",
                "Model File": "Ready" if model_exists else "Missing",
                "Usable": "Yes" if vectorizer_exists and model_exists else "No",
            }
        )
    return pd.DataFrame(rows)


def _available_models(root: Path) -> list[str]:
    available = []
    for model_name, (vectorizer_file, model_file) in MODEL_FILES.items():
        if (root / "models" / vectorizer_file).exists() and (root / "models" / model_file).exists():
            available.append(model_name)
    return available


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


def _predict_text(root: Path, text: str, model_choice: str) -> tuple[dict[str, object], Any | None]:
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
        raise RuntimeError(
            f"Email model artifacts for {model_choice} were not found. "
            "Train or insert the model files before running email prediction."
        )

    except Exception as exc:
        raise RuntimeError(f"The selected email model could not be loaded or used: {exc}") from exc


def _confidence_chart(probabilities: dict[str, float]) -> go.Figure:
    labels = list(probabilities.keys())
    values = [round(float(value) * 100, 2) for value in probabilities.values()]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#3b82f6", "#ef4444"][: len(labels)],
        )
    )
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Confidence (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
    )
    return apply_chart_theme(fig)


def _model_comparison_chart(rows: list[dict[str, object]]) -> go.Figure:
    df = pd.DataFrame(rows)

    fig = go.Figure(
        go.Bar(
            x=df["Risk Score"],
            y=df["Model"],
            orientation="h",
            text=df["Risk Score"].round(2).astype(str) + "%",
            textposition="auto",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=30),
        xaxis_title="Risk Score (%)",
        yaxis_title="Model",
        xaxis=dict(range=[0, 100]),
    )

    return apply_chart_theme(fig)

def _metrics_dataframe(root: Path) -> pd.DataFrame:
    metrics = _load_training_metrics(root)
    if not metrics or "models" not in metrics:
        return pd.DataFrame()

    rows = []
    for model_name, values in metrics["models"].items():
        cm = values.get("confusion_matrix", [[0, 0], [0, 0]])

        try:
            tn, fp = cm[0]
            fn, tp = cm[1]
        except Exception:
            tn = fp = fn = tp = 0

        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(values.get("accuracy", 0) * 100, 2),
                "Precision": round(values.get("precision", 0) * 100, 2),
                "Recall": round(values.get("recall", 0) * 100, 2),
                "F1 Score": round(values.get("f1", 0) * 100, 2),
                "ROC-AUC": round(values.get("roc_auc", 0) * 100, 2)
                if "roc_auc" in values
                else None,
                "Training Time (s)": round(values.get("training_time", 0), 3)
                if "training_time" in values
                else None,
                "Prediction Time (ms)": round(values.get("prediction_time_ms", 0), 4)
                if "prediction_time_ms" in values
                else None,
                "True Positive": tp,
                "False Positive": fp,
                "True Negative": tn,
                "False Negative": fn,
            }
        )

    return pd.DataFrame(rows)


def _training_metrics_chart(metrics_df: pd.DataFrame) -> go.Figure:
    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    if "ROC-AUC" in metrics_df.columns and metrics_df["ROC-AUC"].notna().any():
        metric_columns.append("ROC-AUC")
    chart_df = metrics_df[["Model", *metric_columns]]

    fig = go.Figure()

    for metric in metric_columns:
        fig.add_trace(
            go.Bar(
                x=chart_df["Model"],
                y=chart_df[metric],
                name=metric,
            )
        )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=80),
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        barmode="group",
    )

    return apply_chart_theme(fig)


def _confusion_summary_chart(metrics_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for metric in ["True Positive", "False Positive", "True Negative", "False Negative"]:
        fig.add_trace(
            go.Bar(
                x=metrics_df["Model"],
                y=metrics_df[metric],
                name=metric,
            )
        )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=80),
        yaxis_title="Count",
        barmode="group",
    )

    return apply_chart_theme(fig)


def _metrics_model_name(model_name: str, metrics: dict[str, object]) -> str:
    if model_name == "Support Vector Machine":
        return "SVM"
    if model_name == "Best Model":
        return str(metrics.get("best_model", "Best Model"))
    return model_name


def _is_suspicious_prediction(prediction: object) -> bool:
    text = str(prediction).casefold()
    return any(term in text for term in ("suspicious", "phishing", "scam"))


def _comparison_with_metrics(
    df_compare: pd.DataFrame,
    metrics: dict[str, object],
) -> pd.DataFrame:
    model_metrics = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    rows = []

    for row in df_compare.to_dict("records"):
        metrics_name = _metrics_model_name(str(row["Model"]), metrics)
        values = model_metrics.get(metrics_name, {}) if isinstance(model_metrics, dict) else {}
        rows.append(
            {
                "Model": row["Model"],
                "Prediction": row["Prediction"],
                "Risk Score": row["Risk Score"],
                "Confidence": row["Confidence"],
                "Accuracy": round(float(values.get("accuracy", 0)) * 100, 2)
                if values
                else None,
                "Precision": round(float(values.get("precision", 0)) * 100, 2)
                if values
                else None,
                "Recall": round(float(values.get("recall", 0)) * 100, 2)
                if values
                else None,
                "F1 Score": round(float(values.get("f1", 0)) * 100, 2)
                if values
                else None,
                "ROC-AUC": round(float(values.get("roc_auc", 0)) * 100, 2)
                if values and "roc_auc" in values
                else None,
                "Training Time (s)": round(float(values.get("training_time", 0)), 3)
                if values and "training_time" in values
                else None,
                "Prediction Time (ms)": round(float(values.get("prediction_time_ms", 0)), 4)
                if values and "prediction_time_ms" in values
                else None,
            }
        )

    return pd.DataFrame(rows)


def _recommended_model(df_compare: pd.DataFrame, metrics: dict[str, object]) -> str:
    model_metrics = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    if not df_compare.empty and isinstance(model_metrics, dict) and model_metrics:
        selected = {
            _metrics_model_name(str(model), metrics)
            for model in df_compare["Model"].astype(str).tolist()
        }
        candidate_rows = [
            (name, float(values.get("f1", 0)))
            for name, values in model_metrics.items()
            if name in selected and isinstance(values, dict)
        ]
        if candidate_rows:
            return max(candidate_rows, key=lambda item: item[1])[0]

    best_model = str(metrics.get("best_model", "")).strip() if isinstance(metrics, dict) else ""
    if best_model:
        return best_model

    if not df_compare.empty:
        return str(df_compare.sort_values("Confidence", ascending=False).iloc[0]["Model"])

    return "Not available"


def _confusion_matrix_figure(metrics: dict[str, object], model_name: str) -> go.Figure | None:
    model_metrics = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    values = model_metrics.get(model_name, {}) if isinstance(model_metrics, dict) else {}
    matrix = values.get("confusion_matrix") if isinstance(values, dict) else None
    if not matrix:
        return None

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted safe", "Predicted suspicious"],
            y=["Actual safe", "Actual suspicious"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="Prediction",
        yaxis_title="Actual label",
    )
    return apply_chart_theme(fig)


def _roc_auc_figure(metrics: dict[str, object], model_name: str) -> go.Figure | None:
    model_metrics = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    values = model_metrics.get(model_name, {}) if isinstance(model_metrics, dict) else {}
    if not isinstance(values, dict):
        return None

    curve = values.get("roc_curve", {})
    if not isinstance(curve, dict):
        curve = {}
    fpr = curve.get("fpr") or values.get("fpr")
    tpr = curve.get("tpr") or values.get("tpr")
    roc_auc = values.get("roc_auc") or values.get("auc")
    if not fpr or not tpr:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC-AUC {float(roc_auc):.3f}" if roc_auc is not None else "ROC curve",
            line=dict(color="#2563eb", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(color="#94a3b8", width=1, dash="dash"),
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=35),
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return apply_chart_theme(fig)


def _roc_auc_curve(root: Path, model_names: list[str]) -> go.Figure | None:
    metrics = _load_training_metrics(root)

    if not metrics or "models" not in metrics:
        return None

    fig = go.Figure()
    has_curve = False
    seen_models: set[str] = set()

    for model_name in model_names:
        metrics_name = _metrics_model_name(str(model_name), metrics)
        if metrics_name in seen_models:
            continue
        seen_models.add(metrics_name)

        model_metrics = metrics["models"].get(metrics_name)
        if not model_metrics:
            continue

        curve = model_metrics.get("roc_curve")
        roc_auc = model_metrics.get("roc_auc")

        if not curve:
            continue

        fpr = curve.get("fpr", [])
        tpr = curve.get("tpr", [])

        if not fpr or not tpr:
            continue

        has_curve = True

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=(
                    f"{metrics_name} AUC={float(roc_auc):.3f}"
                    if roc_auc is not None
                    else metrics_name
                ),
            )
        )

    if not has_curve:
        return None

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=40),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )

    return apply_chart_theme(fig)


def _render_evaluation_evidence(
    root: Path,
    metrics_df: pd.DataFrame,
    metrics: dict[str, object],
    recommended_model: str,
    model_choices: list[str],
) -> None:
    render_section_header(
        "Evaluation evidence",
        "Review saved training metrics separately from the live email prediction.",
        "Evaluation evidence",
    )
    render_content_card_open("violet")
    metrics_tab, confusion_tab, roc_tab = st.tabs(
        ["Performance Metrics", "Confusion Matrix Heatmap", "ROC-AUC Curve"]
    )

    with metrics_tab:
        if metrics_df.empty:
            st.warning("No saved training metrics found. Run the email training script first.")
        else:
            st.plotly_chart(_training_metrics_chart(metrics_df), use_container_width=True)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with confusion_tab:
        figure = _confusion_matrix_figure(metrics, recommended_model)
        if figure is None:
            st.info("No confusion matrix is saved for the recommended model yet.")
        else:
            st.caption(f"Confusion matrix shown for recommended model: {recommended_model}")
            st.plotly_chart(figure, use_container_width=True)

    with roc_tab:
        figure = _roc_auc_curve(root, model_choices)
        if figure is None:
            st.warning(
                "ROC-AUC data is not available yet. Retrain the email models after "
                "updating `src/training/train_email_model.py`."
            )
        else:
            st.plotly_chart(figure, use_container_width=True)
            st.caption(
                "ROC-AUC shows how well each model separates safe and suspicious emails. "
                "A curve closer to the top-left corner indicates stronger classification performance."
            )

    render_content_card_close()


def _clean_token(value: str) -> str:
    return value.strip(".,!?;:()[]{}\"'`<>").casefold()


def _term_score(item: dict[str, object]) -> float:
    try:
        return float(item.get("score", item.get("weight", 0)))
    except Exception:
        return 0.0


def _is_directional_term(item: dict[str, object]) -> bool:
    return bool(item.get("directional", True))


def _term_signal_map(terms: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    signals: dict[str, dict[str, object]] = {}
    for item in terms:
        term = str(item.get("term", item.get("phrase", item.get("feature", "")))).strip().casefold()
        if not term:
            continue
        score = _term_score(item)
        signal = {
            "score": score,
            "directional": _is_directional_term(item),
        }
        candidate_tokens = {_clean_token(term)}
        candidate_tokens.update(_clean_token(part) for part in term.split())
        for token in candidate_tokens:
            if not token:
                continue
            if token not in signals or abs(score) > abs(float(signals[token]["score"])):
                signals[token] = signal
    return signals


def _finding_token_map(findings: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    evidence: dict[str, list[dict[str, object]]] = {}
    for item in findings:
        phrase = str(item.get("phrase", "")).strip()
        if not phrase:
            continue
        for part in phrase.split():
            token = _clean_token(part)
            if token:
                evidence.setdefault(token, []).append(item)
    return evidence


def _merged_explainability_html(
    text: str,
    findings: list[dict[str, object]],
    terms: list[dict[str, object]],
) -> str:
    term_signals = _term_signal_map(terms)
    finding_terms = _finding_token_map(findings)
    html_tokens = []

    for token in re.split(r"(\s+)", text):
        if not token:
            continue
        if token.isspace():
            html_tokens.append("<br>" if "\n" in token else " ")
            continue

        clean = _clean_token(token)
        escaped = html.escape(token)
        tooltip_parts = []

        if clean in finding_terms:
            details = finding_terms[clean][0]
            tooltip_parts.append(
                "Rule indicator: "
                f"{details.get('category', 'Warning')} - "
                f"{details.get('specific_tactic', details.get('reason', ''))}"
            )

        if clean in term_signals:
            signal = term_signals[clean]
            score = float(signal["score"])
            if bool(signal.get("directional", True)):
                direction = "higher scam risk" if score > 0 else "lower scam risk" if score < 0 else "neutral risk"
            else:
                direction = "context-dependent tree feature"
            tooltip_parts.append(f"Model weight: {score:.4f} ({direction})")

        if tooltip_parts:
            style = ";".join(
                [
                    "background:rgba(250,204,21,0.22)",
                    "border-bottom:2px solid #facc15",
                    "padding:1px 4px",
                    "border-radius:4px",
                    "line-height:1.7",
                ]
            )
            title = html.escape(" | ".join(tooltip_parts), quote=True)
            html_tokens.append(f'<span title="{title}" style="{style}">{escaped}</span>')
        else:
            html_tokens.append(escaped)

    return (
        '<div style="font-size:0.88rem;line-height:1.75;padding:1rem;'
        'border:1px solid rgba(148,163,184,0.18);border-radius:12px;'
        'background:rgba(15,23,42,0.32);color:var(--text-secondary);">'
        f'{"".join(html_tokens)}</div>'
    )


def _relative_impact_levels(terms: list[dict[str, object]]) -> list[str]:
    if not terms:
        return []

    scores = [abs(_term_score(item)) for item in terms]
    if not any(scores):
        return ["Low" for _item in terms]

    ordered_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    high_count = max(1, math.ceil(len(scores) * 0.20))
    medium_end = max(high_count, math.ceil(len(scores) * 0.50))
    labels = ["Low" for _item in terms]

    for rank, index in enumerate(ordered_indices):
        if scores[index] == 0:
            labels[index] = "Low"
        elif rank < high_count:
            labels[index] = "High"
        elif rank < medium_end:
            labels[index] = "Medium"

    return labels


def _model_term_type(indicator: str, directional: bool) -> str:
    return "Model Term"


def _impact_guide_text() -> str:
    return (
        "Impact is a relative contribution label for the current prediction, not a universal danger rating. "
        "For model terms, the app ranks the displayed terms by absolute model weight: top 20% = High, "
        "next 30% = Medium, remaining terms = Low. Rule indicators stay High as explicit evidence, "
        "while legitimate context rows are marked Context. Model weights are not directly comparable across algorithms."
    )


def _dedupe_table_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped = []
    seen = set()

    for row in rows:
        indicator = str(row.get("Indicator", "")).strip().lower()
        row_type = str(row.get("Type", "")).strip().lower()

        if not indicator:
            continue

        key = (indicator, row_type)

        if key in seen:
            continue

        seen.add(key)
        deduped.append(row)

    return deduped


def _explainability_rows(
    _text: str,
    findings: list[dict[str, object]],
    legitimate_indicators: list[dict[str, object]],
    terms: list[dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for item in findings:
        indicator = str(item.get("phrase", "")).strip()
        if not indicator:
            continue
        rows.append(
            {
                "Indicator": indicator,
                "Type": item.get("category", "Rule indicator"),
                "Model Weight": "-",
                "Impact": "High",
                "Intention": item.get("intention", type_intention(item.get("category", ""))),
            }
        )

    for item in legitimate_indicators:
        indicator = str(item.get("phrase", item.get("indicator", ""))).strip()
        if not indicator:
            continue
        row_type = item.get("category", item.get("Type", "Legitimate Context"))
        rows.append(
            {
                "Indicator": indicator,
                "Type": row_type,
                "Model Weight": "-",
                "Impact": "Context",
                "Intention": item.get("intention", type_intention(row_type)),
            }
        )

    impact_levels = _relative_impact_levels(terms)
    for index, item in enumerate(terms):
        indicator = str(item.get("term", item.get("phrase", item.get("feature", ""))))
        score = _term_score(item)
        impact = impact_levels[index] if index < len(impact_levels) else "Low"
        directional = _is_directional_term(item)
        indicator_type = _model_term_type(indicator, directional)

        rows.append(
            {
                "Indicator": indicator,
                "Type": indicator_type,
                "Model Weight": round(score, 4),
                "Impact": impact,
                "Intention": type_intention(indicator_type),
            }
        )

    return pd.DataFrame(_dedupe_table_rows(rows))


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


def _risk_score(result: dict[str, object]) -> float:
    probabilities = dict(result.get("probabilities", {}))
    confidence = float(result.get("confidence", 0.0))
    label = int(result.get("label", 0))

    if "Suspicious" in probabilities:
        return float(probabilities["Suspicious"]) * 100

    return confidence * 100 if label == 1 else (1 - confidence) * 100


def _display_result(result: dict[str, object], text: str, classifier: Any | None) -> None:
    confidence = float(result["confidence"])
    label = str(result["label_name"])
    findings = list(result.get("findings", []))
    legitimate_indicators = find_legitimate_indicators(text)
    risk_score = _risk_score(result)
    terms = []
    if classifier is not None:
        try:
            terms = top_model_terms(text, classifier.vectorizer, classifier.model)
        except Exception:
            terms = []

    render_section_header(
        "Explainability evidence",
        "Review rule indicators and model-influential terms in one place. Hover highlighted words to see scores.",
        "Why the model decided this",
    )
    render_soft_panel(
        "Yellow highlights combine rule-based warning indicators with model-influential terms. "
        "Hover each highlighted word to inspect the underlying evidence signal."
    )
    st.markdown(
        _merged_explainability_html(text, findings, terms),
        unsafe_allow_html=True,
    )
    if not findings and terms:
        st.caption(
            "No rule indicators were triggered, but model terms are still shown so safe or ordinary emails remain explainable."
        )

    with st.expander("Rule indicator details", expanded=False):
        st.caption(_impact_guide_text())
        evidence_df = _explainability_rows(text, findings, legitimate_indicators, terms)
        if evidence_df.empty:
            st.info("No rule indicators, legitimate context indicators, or model-influential terms were detected.")
        else:
            st.dataframe(evidence_df, hide_index=True, use_container_width=True)

    render_section_header(
        "Technical details",
        "Inspect the selected model's probability distribution and raw confidence values.",
        "Model internals",
    )
    render_content_card_open("violet")
    st.plotly_chart(_confidence_chart(dict(result["probabilities"])), use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Selected model engine": result.get("model_name", "Unknown"),
                    "Selected prediction": label,
                    "Selected confidence": round(confidence * 100, 2),
                    "Selected risk score": round(risk_score, 2),
                    "Rule indicator count": len(findings),
                    "Legitimate context count": len(legitimate_indicators),
                }
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    render_content_card_close()


def _load_training_metrics(root: Path) -> dict[str, object]:
    metrics_path = root / "reports" / "metrics" / "email_model_metrics.json"
    if not metrics_path.exists():
        return {}

    try:
        import json
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def render_email_tab(root: Path, history: list[dict[str, object]]) -> None:
    render_demo_notice(root)

    render_section_header(
        "Scan email or message content",
        "Paste one message, upload a text file, or inspect multiple CSV rows.",
        "Text evidence",
    )

    demo_emails = get_demo_data()["emails"]
    available_models = _available_models(root)
    model_options = [model for model in DISPLAY_MODELS if model in available_models]

    with st.expander("Email model status", expanded=False):
        st.dataframe(_model_status(root), hide_index=True, use_container_width=True)
        if not available_models:
            st.warning(
                "No trained email models were found. Run `py src/training/train_email_model.py` first. "
                "Email prediction requires trained ML model artifacts."
            )

        metrics_df = _metrics_dataframe(root)

        with st.expander("Trained AI model evaluation metrics", expanded=False):
            if not metrics_df.empty:
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)

                st.info(
                    "Accuracy measures overall correctness. Precision shows how many flagged emails were truly suspicious. "
                    "Recall shows how many suspicious emails were caught. F1 balances precision and recall. "
                    "ROC-AUC measures class separation. Training time and prediction time are saved after retraining with the latest script. "
                    "False positives are safe emails wrongly flagged as suspicious. False negatives are suspicious emails missed by the model."
                )
            else:
                st.warning("No saved training metrics found. Run the email training script first.")

    render_content_card_open("violet")
    controls, input_area = st.columns([0.30, 0.70])

    with controls:
        select_all = st.checkbox("Select all trained models", value=True)

        if select_all:
            model_choices = model_options
        else:
            model_choices = st.multiselect(
                "Select AI models to compare",
                model_options,
                default=[model_options[0]] if model_options else [],
                help="Select one or more trained models to compare their predictions.",
            )

        demo_choice = st.selectbox(
            "Try synthetic demo email",
            ["None"] + demo_emails["sample_id"].tolist(),
        )

        uploaded_file = st.file_uploader(
            "Upload email text or CSV",
            type=["txt", "csv"],
            key="email_upload",
        )

        analyze_button = st.button(
            "Analyze email",
            type="primary",
            use_container_width=True,
        )

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
            placeholder="Paste an email, campus message, scholarship offer, job offer, or banking message here.",
        )

    render_content_card_close()

    if isinstance(uploaded, pd.DataFrame):
        render_section_header("Batch CSV analysis", eyebrow="Multiple rows")
        render_content_card_open("violet")

        text_column = st.selectbox("Text column", uploaded.columns)

        if st.button("Analyze CSV rows", use_container_width=True):
            texts = uploaded[text_column].fillna("").astype(str).tolist()
            rows = []

            for value in texts:
                for selected_model in model_choices:
                    try:
                        result, _classifier = _predict_text(root, value, selected_model)
                    except RuntimeError as exc:
                        st.error(str(exc))
                        return
                    rows.append(
                        {
                            "model": selected_model,
                            "preview": value[:120],
                            "prediction": result["label_name"],
                            "risk_score": round(_risk_score(result), 2),
                            "confidence": round(float(result["confidence"]) * 100, 2),
                            "engine": result["model_name"],
                        }
                    )

            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        render_content_card_close()

    if analyze_button:
        if not text.strip():
            st.warning("Paste text or upload a .txt file first.")
            return

        if not model_choices:
            st.warning("Select at least one model first.")
            return

        comparison_rows = []
        first_result = None
        first_classifier = None

        for selected_model in model_choices:
            try:
                result, classifier = _predict_text(root, text, selected_model)
            except RuntimeError as exc:
                st.error(str(exc))
                return

            if first_result is None:
                first_result = result
                first_classifier = classifier

            comparison_rows.append(
                {
                    "Model": selected_model,
                    "Prediction": result["label_name"],
                    "Risk Score": round(_risk_score(result), 2),
                    "Confidence": round(float(result["confidence"]) * 100, 2),
                    "Engine": result["model_name"],
                }
            )

        df_compare = pd.DataFrame(comparison_rows)
        suspicious_count = int(df_compare["Prediction"].apply(_is_suspicious_prediction).sum())
        total_models = len(df_compare)
        average_risk = float(df_compare["Risk Score"].mean())
        highest_confidence = float(df_compare["Confidence"].max())
        metrics = _load_training_metrics(root)
        metrics_df = _metrics_dataframe(root)
        recommended_model = _recommended_model(df_compare, metrics)
        df_compare_metrics = _comparison_with_metrics(df_compare, metrics)
        if suspicious_count > (total_models / 2):
            final_verdict = "Suspicious"
        elif suspicious_count == (total_models / 2):
            final_verdict = "Suspicious" if average_risk >= 50 else "Safe"
        else:
            final_verdict = "Safe"

        render_analysis_ready("Email analysis complete - results ready below")

        render_section_header(
            "Analysis summary",
            "Overall AI verdict and agreement across the selected email models.",
            "Executive result",
        )
        render_content_card_open("violet")
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Final Verdict",
            final_verdict,
        )
        col2.metric(
            "Average Risk",
            f"{average_risk:.2f}%",
        )
        col3.metric(
            "Model Agreement",
            f"{suspicious_count}/{total_models}",
        )
        col4.metric(
            "Recommended Model",
            recommended_model,
        )
        col5.metric(
            "Highest Confidence",
            f"{highest_confidence:.2f}%",
        )
        st.caption(
            "This summary is based only on trained ML model predictions. Explainability evidence below does not "
            "adjust the verdict, average risk, or confidence."
        )
        render_content_card_close()

        render_section_header(
            "AI model comparison",
            "Compare each selected model's live prediction beside its saved training performance.",
            "Model evidence",
        )
        render_content_card_open("violet")

        st.plotly_chart(
            _model_comparison_chart(comparison_rows),
            use_container_width=True,
        )

        st.dataframe(
            df_compare_metrics,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Higher agreement between independent models generally increases confidence in the prediction. "
            "If models disagree significantly, the message should be reviewed manually."
        )

        render_content_card_close()

        _render_evaluation_evidence(root, metrics_df, metrics, recommended_model, model_choices)

        if first_result is not None:
            _record(history, first_result, text)
            _display_result(first_result, text, first_classifier)
