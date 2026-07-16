"""Train behavioral metadata model for voice spoof / deepfake risk.

Input expected from prepare_audio_dataset.py:
    data/processed/audio/
    ├── train/
    ├── dev/
    └── labels.csv

Output:
    models/audio_behavior_rf.pkl
    reports/metrics/audio_behavior_metrics.json

Run:
    py scripts/07_train_audio_behavior_model.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.live_audio_analysis import (  # noqa: E402
    BEHAVIORAL_FEATURE_NAMES,
    TARGET_SAMPLE_RATE,
    extract_behavioral_features,
)


DATA_DIR = ROOT / "data" / "processed" / "audio"
LABELS_PATH = DATA_DIR / "labels.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "metrics"

MODEL_PATH = MODELS_DIR / "audio_behavior_rf.pkl"
METRICS_PATH = REPORTS_DIR / "audio_behavior_metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_features_from_path(audio_path: Path) -> np.ndarray:
    audio, sample_rate = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
    if audio.size == 0:
        audio = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
    features = extract_behavioral_features(audio.astype(np.float32), sample_rate)
    return np.asarray(features["feature_vector"], dtype=np.float32)


def load_split(labels: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    split_rows = labels[labels["split"] == split].copy()
    if split_rows.empty:
        raise ValueError(f"No rows found for split={split!r} in {LABELS_PATH}")

    X: list[np.ndarray] = []
    y: list[int] = []
    paths: list[str] = []
    total = len(split_rows)

    for index, row in enumerate(split_rows.to_dict(orient="records"), start=1):
        relative_path = str(row["relative_path"])
        audio_path = DATA_DIR / relative_path

        if not audio_path.exists():
            print(f"Skipping missing file: {audio_path}")
            continue

        try:
            features = extract_features_from_path(audio_path)
        except Exception as exc:
            print(f"Skipping unreadable file {audio_path}: {exc}")
            continue

        label_text = str(row["label"]).lower()
        label = 1 if label_text == "spoof" else 0

        X.append(features)
        y.append(label)
        paths.append(relative_path)

        if index % 100 == 0 or index == total:
            print(f"[{split}] processed {index}/{total}")

    if not X:
        raise RuntimeError(f"No usable audio files found for split={split!r}")

    return np.vstack(X), np.asarray(y, dtype=int), paths


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["bonafide", "spoof"],
            zero_division=0,
            output_dict=True,
        ),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
    }


def main() -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}\n"
            "Run prepare_audio_dataset.py first."
        )

    labels = pd.read_csv(LABELS_PATH)

    required_columns = {"split", "relative_path", "label"}
    missing = required_columns - set(labels.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")

    print("Loading behavioral audio features...")
    start_time = time.time()
    X_train, y_train, train_paths = load_split(labels, "train")
    X_dev, y_dev, dev_paths = load_split(labels, "dev")
    feature_seconds = time.time() - start_time

    if X_train.shape[1] != len(BEHAVIORAL_FEATURE_NAMES):
        raise ValueError(
            f"Feature mismatch: extracted {X_train.shape[1]} features, "
            f"expected {len(BEHAVIORAL_FEATURE_NAMES)}."
        )

    print("\nDataset loaded")
    print(f"Train samples: {len(y_train)}")
    print(f"Dev samples: {len(y_dev)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print("Train label counts:", {int(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()})
    print("Dev label counts:", {int(k): int(v) for k, v in pd.Series(y_dev).value_counts().sort_index().items()})
    print(f"Feature extraction time: {feature_seconds:.2f}s")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining behavioral Random Forest...")
    train_start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - train_start

    print("Evaluating on dev split...")
    metrics = evaluate_model(model, X_dev, y_dev)

    print("\nBehavioral model metrics")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    joblib.dump(model, MODEL_PATH)

    importances = [
        {
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in sorted(
            zip(BEHAVIORAL_FEATURE_NAMES, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    summary = {
        "dataset_path": str(DATA_DIR),
        "labels_path": str(LABELS_PATH),
        "model_path": str(MODEL_PATH),
        "task": "ASVspoof2019 LA behavioral metadata spoof detection",
        "label_mapping": {
            "bonafide": 0,
            "spoof": 1,
        },
        "sample_rate": TARGET_SAMPLE_RATE,
        "feature_dimension": int(X_train.shape[1]),
        "feature_names": BEHAVIORAL_FEATURE_NAMES,
        "feature_importances": importances,
        "train_samples": int(len(y_train)),
        "dev_samples": int(len(y_dev)),
        "train_distribution": {
            "bonafide": int((y_train == 0).sum()),
            "spoof": int((y_train == 1).sum()),
        },
        "dev_distribution": {
            "bonafide": int((y_dev == 0).sum()),
            "spoof": int((y_dev == 1).sum()),
        },
        "feature_extraction_seconds": float(feature_seconds),
        "training_seconds": float(training_seconds),
        "training_time_seconds": float(training_seconds),
        "training_time": float(training_seconds),
        "model": "Random Forest on audio behavior metadata",
        "metrics": metrics,
        "dev_paths": dev_paths,
        "train_paths_sample": train_paths[:20],
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
