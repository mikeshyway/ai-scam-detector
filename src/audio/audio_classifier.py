"""SVM training and inference for AI-generated speech detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


AUDIO_LABEL_NAMES = {0: "Real human speech", 1: "Possible AI-generated speech"}


@dataclass
class AudioPrediction:
    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]


class AudioDeepfakeClassifier:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_one(self, mfcc_features: np.ndarray) -> AudioPrediction:
        row = np.asarray(mfcc_features, dtype=float).reshape(1, -1)
        label = int(self.model.predict(row)[0])
        probabilities = _probabilities(self.model, row)
        confidence = float(probabilities.get(label, max(probabilities.values())))
        return AudioPrediction(
            label=label,
            label_name=AUDIO_LABEL_NAMES.get(label, str(label)),
            confidence=confidence,
            probabilities={
                AUDIO_LABEL_NAMES.get(label_id, str(label_id)): float(probability)
                for label_id, probability in sorted(probabilities.items())
            },
        )


def _probabilities(model: Any, X: np.ndarray) -> dict[int, float]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = [int(label) for label in getattr(model, "classes_", [0, 1])]
        out = {0: 0.0, 1: 0.0}
        for index, label in enumerate(classes):
            if label in out:
                out[label] = float(proba[index])
        return out

    if hasattr(model, "decision_function"):
        score = float(np.ravel(model.decision_function(X))[0])
        positive = 1.0 / (1.0 + np.exp(-score))
        return {0: 1.0 - positive, 1: positive}

    label = int(model.predict(X)[0])
    return {0: float(label == 0), 1: float(label == 1)}


def train_audio_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, object]]:
    if len(set(y.tolist())) < 2:
        raise ValueError("Audio training requires both labels: 0 real and 1 fake.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Real human speech", "Possible AI-generated speech"],
            zero_division=0,
            output_dict=True,
        ),
    }
    return model, metrics


def save_audio_model(model: Any, model_path: str | Path) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_audio_model(model_path: str | Path) -> AudioDeepfakeClassifier:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)
    model = joblib.load(path)
    _force_single_worker_prediction(model)
    return AudioDeepfakeClassifier(model)


def load_audio_behavior_model(model_path: str | Path) -> AudioDeepfakeClassifier:
    """Load the optional behavioral metadata model.

    The behavioral model uses the same binary label mapping as the MFCC SVM:
    0 = bonafide / lower-risk speech, 1 = spoof / possible AI-generated speech.
    """

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)
    model = joblib.load(path)
    _force_single_worker_prediction(model)
    return AudioDeepfakeClassifier(model)


def _force_single_worker_prediction(model: Any) -> None:
    """Avoid local runtime failures from joblib worker pools during prediction."""

    if hasattr(model, "n_jobs"):
        try:
            setattr(model, "n_jobs", 1)
        except Exception:
            pass

    for attribute in ("estimator", "base_estimator", "final_estimator"):
        child = getattr(model, attribute, None)
        if child is not None and child is not model:
            _force_single_worker_prediction(child)

    for attribute in ("estimators_", "calibrated_classifiers_"):
        children = getattr(model, attribute, None)
        if not children:
            continue
        for child in children:
            if child is not model:
                _force_single_worker_prediction(child)
