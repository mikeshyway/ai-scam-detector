"""Train an ASVspoof audio classifier for AI-generated / spoofed speech detection.

Input expected from prepare_audio_dataset.py:
    data/processed/audio/
    ├── train/
    │   └── *.flac
    ├── dev/
    │   └── *.flac
    └── labels.csv

Output:
    models/audio_svm.pkl
    reports/metrics/audio_model_metrics.json

Run:
    py src/training/train_audio_model.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import joblib
import librosa
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIR = ROOT / "data" / "processed" / "audio"
LABELS_PATH = DATA_DIR / "labels.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "metrics"

MODEL_PATH = MODELS_DIR / "audio_svm.pkl"
METRICS_PATH = REPORTS_DIR / "audio_model_metrics.json"

TARGET_SAMPLE_RATE = 16_000
N_MFCC = 40

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def feature_names() -> list[str]:
    """Return names for the 251-dimensional audio feature vector."""

    names: list[str] = []

    names += [f"mfcc_{i + 1}_mean" for i in range(N_MFCC)]
    names += [f"mfcc_{i + 1}_std" for i in range(N_MFCC)]

    names += [f"delta_mfcc_{i + 1}_mean" for i in range(N_MFCC)]
    names += [f"delta_mfcc_{i + 1}_std" for i in range(N_MFCC)]

    names += [f"delta2_mfcc_{i + 1}_mean" for i in range(N_MFCC)]
    names += [f"delta2_mfcc_{i + 1}_std" for i in range(N_MFCC)]

    names += [
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
        "rms_energy_mean",
        "rms_energy_std",
        "duration_seconds",
    ]

    return names


def extract_audio_features(audio_path: Path) -> np.ndarray:
    """Extract compact MFCC-based features from one audio file."""

    y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)

    if y.size == 0:
        y = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta2.mean(axis=1),
            delta2.std(axis=1),
            [
                float(spectral_centroid.mean()),
                float(spectral_centroid.std()),
                float(spectral_bandwidth.mean()),
                float(spectral_bandwidth.std()),
                float(spectral_rolloff.mean()),
                float(spectral_rolloff.std()),
                float(zero_crossing.mean()),
                float(zero_crossing.std()),
                float(rms.mean()),
                float(rms.std()),
                float(len(y) / sr),
            ],
        ]
    )

    return features.astype(np.float32)


def load_split(labels: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features and labels for train or dev split."""

    split_rows = labels[labels["split"] == split].copy()

    if split_rows.empty:
        raise ValueError(f"No rows found for split={split!r} in {LABELS_PATH}")

    X = []
    y = []
    paths = []

    total = len(split_rows)

    for index, row in enumerate(split_rows.to_dict(orient="records"), start=1):
        relative_path = str(row["relative_path"])
        audio_path = DATA_DIR / relative_path

        if not audio_path.exists():
            print(f"Skipping missing file: {audio_path}")
            continue

        try:
            features = extract_audio_features(audio_path)
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


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    """Evaluate classifier and return JSON-serialisable metrics."""

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_score = model.decision_function(X_test)
        y_score = (raw_score - raw_score.min()) / max(1e-9, raw_score.max() - raw_score.min())
    else:
        y_score = y_pred

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
            f"Labels file not found: {LABELS_PATH}\\n"
            "Run prepare_audio_dataset.py first."
        )

    labels = pd.read_csv(LABELS_PATH)

    required_columns = {"split", "relative_path", "label"}
    missing = required_columns - set(labels.columns)

    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")

    print("Loading audio features...")
    start_time = time.time()

    X_train, y_train, train_paths = load_split(labels, "train")
    X_dev, y_dev, dev_paths = load_split(labels, "dev")

    feature_seconds = time.time() - start_time
    names = feature_names()

    if X_train.shape[1] != len(names):
        raise ValueError(
            f"Feature name mismatch: extracted {X_train.shape[1]} features, "
            f"but generated {len(names)} names."
        )

    print("\\nDataset loaded")
    print(f"Train samples: {len(y_train)}")
    print(f"Dev samples: {len(y_dev)}")
    print("Train label counts:", {int(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()})
    print("Dev label counts:", {int(k): int(v) for k, v in pd.Series(y_dev).value_counts().sort_index().items()})
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Feature extraction time: {feature_seconds:.2f}s")

    base_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=3.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # CalibratedClassifierCV provides predict_proba() for confidence display.
    # SVC(probability=True) is intentionally avoided.
    model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=3,
    )

    print("\\nTraining calibrated SVM audio classifier...")
    train_start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - train_start

    print("Evaluating on dev split...")
    metrics = evaluate_model(model, X_dev, y_dev)

    print("\\nAudio model metrics")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    joblib.dump(model, MODEL_PATH)

    summary = {
        "dataset_path": str(DATA_DIR),
        "labels_path": str(LABELS_PATH),
        "model_path": str(MODEL_PATH),
        "task": "ASVspoof2019 LA binary spoof detection",
        "label_mapping": {
            "bonafide": 0,
            "spoof": 1,
        },
        "sample_rate": TARGET_SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "feature_dimension": int(X_train.shape[1]),
        "feature_names": names,
        "feature_groups": {
            "mfcc": "MFCC mean and standard deviation",
            "delta_mfcc": "First-order MFCC temporal change statistics",
            "delta2_mfcc": "Second-order MFCC temporal change statistics",
            "spectral": "Centroid, bandwidth, and rolloff statistics",
            "energy_timing": "Zero-crossing rate, RMS energy, and duration",
        },
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
        "model": "Calibrated SVM with MFCC/statistical audio features",
        "calibration": {
            "method": "sigmoid",
            "cv": 3,
            "purpose": "Provides probability-like confidence scores for Streamlit display.",
        },
        "metrics": metrics,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
