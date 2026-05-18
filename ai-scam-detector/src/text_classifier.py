"""Training and inference helpers for text scam detection models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from .text_preprocessor import build_tfidf_vectorizer


TEXT_LABEL_NAMES = {
    0: "Legitimate",
    1: "Suspicious",
}


@dataclass
class TextPrediction:
    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str


class TextScamClassifier:
    """Thin inference wrapper around a saved vectorizer and scikit-learn model."""

    def __init__(self, vectorizer: Any, model: Any, model_name: str = "Text model") -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.model_name = model_name

    def predict_one(self, text: str) -> TextPrediction:
        X = self.vectorizer.transform([text])
        prediction = int(self.model.predict(X)[0])
        probabilities = _probabilities(self.model, X)
        confidence = float(probabilities.get(prediction, max(probabilities.values())))

        return TextPrediction(
            label=prediction,
            label_name=TEXT_LABEL_NAMES.get(prediction, str(prediction)),
            confidence=confidence,
            probabilities={
                TEXT_LABEL_NAMES.get(label, str(label)): float(prob)
                for label, prob in sorted(probabilities.items())
            },
            model_name=self.model_name,
        )

    def predict_many(self, texts: list[str]) -> pd.DataFrame:
        X = self.vectorizer.transform(texts)
        labels = self.model.predict(X).astype(int)
        proba = _probability_matrix(self.model, X)

        rows = []
        for text, label, row in zip(texts, labels, proba):
            confidence = float(row[label]) if label < len(row) else float(row.max())
            rows.append(
                {
                    "preview": text[:120],
                    "prediction": TEXT_LABEL_NAMES.get(int(label), str(label)),
                    "confidence": round(confidence * 100, 2),
                }
            )
        return pd.DataFrame(rows)


def _probability_matrix(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = [int(label) for label in getattr(model, "classes_", [0, 1])]
        out = np.zeros((X.shape[0], 2), dtype=float)
        for index, label in enumerate(classes):
            if label in (0, 1):
                out[:, label] = probabilities[:, index]
        missing = out.sum(axis=1) == 0
        out[missing] = np.array([0.5, 0.5])
        return out

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-scores))
            return np.vstack([1.0 - positive, positive]).T

    predictions = model.predict(X).astype(int)
    out = np.zeros((X.shape[0], 2), dtype=float)
    out[np.arange(X.shape[0]), predictions] = 1.0
    return out


def _probabilities(model: Any, X: Any) -> dict[int, float]:
    row = _probability_matrix(model, X)[0]
    return {0: float(row[0]), 1: float(row[1])}


def train_text_models(
    texts: list[str] | pd.Series,
    labels: list[int] | pd.Series,
    *,
    include_decision_tree: bool = True,
    max_features: int = 8000,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any], dict[str, dict[str, Any]]]:
    """Fit TF-IDF + Naive Bayes and optionally a Decision Tree model."""

    labels_array = np.asarray(labels, dtype=int)
    if len(set(labels_array.tolist())) < 2:
        raise ValueError("Training requires both class labels: 0 and 1.")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        list(texts),
        labels_array,
        test_size=0.2,
        random_state=random_state,
        stratify=labels_array,
    )

    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models: dict[str, Any] = {
        "nb": MultinomialNB(alpha=0.5),
    }
    if include_decision_tree:
        models["dt"] = DecisionTreeClassifier(
            max_depth=25,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        )

    metrics: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=["Legitimate", "Suspicious"],
                zero_division=0,
                output_dict=True,
            ),
        }

    return vectorizer, models, metrics


def save_text_artifacts(
    vectorizer: Any,
    model: Any,
    vectorizer_path: str | Path,
    model_path: str | Path,
) -> None:
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)


def load_text_artifacts(
    vectorizer_path: str | Path,
    model_path: str | Path,
    *,
    model_name: str,
) -> TextScamClassifier:
    vectorizer_file = Path(vectorizer_path)
    model_file = Path(model_path)
    if not vectorizer_file.exists() or not model_file.exists():
        missing = [str(path) for path in (vectorizer_file, model_file) if not path.exists()]
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing))

    return TextScamClassifier(
        vectorizer=joblib.load(vectorizer_file),
        model=joblib.load(model_file),
        model_name=model_name,
    )

