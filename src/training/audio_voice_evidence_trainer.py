"""Train the second-stage voice evidence calibrator.

The raw MFCC SVM remains the source model. This trainer learns a user-facing
voice-evidence score from raw voice probability, Behavioral RF probability, and
audio quality/acoustic context so short or sparse clips do not look conclusive.

Output:
    models/audio_voice_evidence_calibrator.pkl
    reports/metrics/audio_voice_evidence_metrics.json
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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.audio_classifier import (  # noqa: E402
    load_audio_behavior_model,
    load_audio_model,
)
from src.audio.live_audio_analysis import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    assess_speech_quality,
    extract_behavioral_features,
    extract_live_features,
    score_audio_chunk,
    score_behavioral_chunk,
)
from src.audio.voice_evidence_calibrator import (  # noqa: E402
    VOICE_EVIDENCE_FEATURE_NAMES,
    build_voice_evidence_feature_values,
    voice_evidence_feature_vector,
)


DATA_DIR = ROOT / "data" / "processed" / "audio"
LABELS_PATH = DATA_DIR / "labels.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "metrics"

RAW_AUDIO_MODEL_PATH = MODELS_DIR / "audio_svm.pkl"
BEHAVIORAL_MODEL_PATH = MODELS_DIR / "audio_behavior_rf.pkl"
MODEL_PATH = MODELS_DIR / "audio_voice_evidence_calibrator.pkl"
METRICS_PATH = REPORTS_DIR / "audio_voice_evidence_metrics.json"

TRAIN_FULL_PER_LABEL = int(os.getenv("VOICE_EVIDENCE_TRAIN_FULL_PER_LABEL", "80"))
TRAIN_AUGMENT_PER_LABEL = int(os.getenv("VOICE_EVIDENCE_TRAIN_AUGMENT_PER_LABEL", "50"))
DEV_FULL_PER_LABEL = int(os.getenv("VOICE_EVIDENCE_DEV_FULL_PER_LABEL", "100"))

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _bounded_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(100.0, value)))


def _audio_rms(audio: np.ndarray) -> float:
    finite = np.asarray(audio, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(finite))))


def _scaled_to_rms(audio: np.ndarray, target_rms: float = 0.011) -> np.ndarray:
    current = _audio_rms(audio)
    if current <= 1e-8:
        return audio.astype(np.float32)
    scaled = np.asarray(audio, dtype=np.float32) * (target_rms / current)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def _slice_audio(audio: np.ndarray, sample_rate: int, seconds: float, offset_ratio: float) -> np.ndarray:
    window = max(1, int(sample_rate * seconds))
    if audio.size <= window:
        return audio.astype(np.float32)
    start = int((audio.size - window) * max(0.0, min(1.0, offset_ratio)))
    return audio[start : start + window].astype(np.float32)


def _variants(
    audio: np.ndarray,
    sample_rate: int,
    *,
    include_augmented: bool,
) -> list[tuple[str, np.ndarray]]:
    variants = [("full", audio.astype(np.float32))]
    if include_augmented:
        short = _slice_audio(audio, sample_rate, 3.0, 0.45)
        variants.extend(
            [
                ("short_3s", short),
                ("quiet_short_3s", _scaled_to_rms(short)),
            ]
        )
    return variants


def _quality_weight(speech_quality: dict[str, object]) -> float:
    duration = float(speech_quality.get("duration_seconds", 0.0))
    speech_activity = float(speech_quality.get("speech_activity_ratio", 0.0))
    silence = float(speech_quality.get("silence_ratio", 1.0))
    rms = float(speech_quality.get("rms", 0.0))
    density = max(speech_activity, 1.0 - silence)

    duration_score = max(0.0, min(1.0, (duration - 2.0) / 8.0))
    density_score = max(0.0, min(1.0, (density - 0.20) / 0.45))
    energy_score = max(0.0, min(1.0, (rms - 0.008) / 0.035))
    score = (0.45 * duration_score) + (0.35 * density_score) + (0.20 * energy_score)

    if duration < 3.0:
        score *= 0.35
    elif duration < 6.0:
        score *= 0.70
    if silence > 0.65:
        score *= 0.50
    if speech_activity < 0.35:
        score *= 0.65
    if rms < 0.015:
        score *= 0.60

    return max(0.0, min(1.0, score))


def _evidence_target(label: int, speech_quality: dict[str, object], variant_name: str) -> float:
    if label == 0 or not bool(speech_quality.get("usable_speech", True)):
        return 0.0
    if variant_name == "full":
        return 100.0
    return 100.0 * _quality_weight(speech_quality)


def _row_to_sample(
    audio_path: Path,
    label: int,
    split: str,
    audio_classifier,
    behavioral_classifier,
    include_augmented: bool,
) -> list[dict[str, object]]:
    audio, sample_rate = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
    if audio.size == 0:
        audio = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)

    samples: list[dict[str, object]] = []
    for variant_name, variant_audio in _variants(
        np.asarray(audio, dtype=np.float32),
        sample_rate,
        include_augmented=include_augmented,
    ):
        try:
            features = extract_live_features(variant_audio, sample_rate=sample_rate)
            behavioral_features = extract_behavioral_features(variant_audio, sample_rate=sample_rate)
            speech_quality = assess_speech_quality(
                variant_audio,
                sample_rate=sample_rate,
                features=features,
                behavioral_features=behavioral_features,
            )
            features["speech_quality"] = speech_quality
            raw_voice_risk, _voice_label, _voice_engine = score_audio_chunk(features, audio_classifier)
            raw_behavioral_risk, _behavior_label, _behavior_engine = score_behavioral_chunk(
                behavioral_features,
                behavioral_classifier,
            )
            feature_values = build_voice_evidence_feature_values(
                raw_voice_risk=raw_voice_risk,
                raw_behavioral_risk=raw_behavioral_risk,
                features=features,
                behavioral_features=behavioral_features,
                speech_quality=speech_quality,
            )
        except Exception as exc:
            print(f"Skipping {audio_path.name} {variant_name}: {exc}")
            continue

        target = _evidence_target(label, speech_quality, variant_name)
        samples.append(
            {
                "split": split,
                "variant": variant_name,
                "label": label,
                "target": target,
                "feature_vector": voice_evidence_feature_vector(feature_values),
                "feature_values": feature_values,
            }
        )
    return samples


def _stratified_sample(rows: pd.DataFrame, per_label: int | None) -> pd.DataFrame:
    if per_label is None:
        return rows
    sampled = []
    for _label, group in rows.groupby("label", sort=True):
        sampled.append(
            group.sample(
                n=min(per_label, len(group)),
                random_state=42,
            )
        )
    return pd.concat(sampled).sample(frac=1.0, random_state=42).reset_index(drop=True)


def _load_samples(
    labels: pd.DataFrame,
    split: str,
    audio_classifier,
    behavioral_classifier,
    *,
    full_per_label: int | None,
    augment_per_label: int | None,
) -> list[dict[str, object]]:
    rows = labels[labels["split"] == split].copy()
    full_rows = _stratified_sample(rows, full_per_label)
    augment_paths = set(
        _stratified_sample(rows, augment_per_label)["relative_path"].astype(str).tolist()
        if augment_per_label is not None
        else []
    )
    samples: list[dict[str, object]] = []
    total = len(full_rows)
    for index, row in enumerate(full_rows.to_dict(orient="records"), start=1):
        audio_path = DATA_DIR / str(row["relative_path"])
        if not audio_path.exists():
            print(f"Skipping missing file: {audio_path}")
            continue
        label = 1 if str(row["label"]).lower() == "spoof" else 0
        include_augmented = str(row["relative_path"]) in augment_paths
        samples.extend(
            _row_to_sample(
                audio_path,
                label,
                split,
                audio_classifier,
                behavioral_classifier,
                include_augmented,
            )
        )
        if index % 100 == 0 or index == total:
            print(f"[{split}] processed {index}/{total} files, {len(samples)} calibrator rows", flush=True)
    return samples


def _matrix(samples: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X = np.vstack([np.asarray(item["feature_vector"], dtype=np.float32) for item in samples])
    y = np.asarray([float(item["target"]) for item in samples], dtype=np.float32)
    labels = np.asarray([int(item["label"]) for item in samples], dtype=int)
    variants = [str(item["variant"]) for item in samples]
    return X, y, labels, variants


def _evaluate_predictions(y_target: np.ndarray, y_label: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    y_pred = np.asarray([_bounded_score(float(value)) for value in y_pred], dtype=float)
    y_binary = (y_pred >= 50.0).astype(int)
    metrics = {
        "mae": float(mean_absolute_error(y_target, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_target, y_pred))),
        "mean_prediction": float(np.mean(y_pred)),
        "mean_target": float(np.mean(y_target)),
        "threshold_accuracy": float(accuracy_score(y_label, y_binary)),
        "threshold_precision": float(precision_score(y_label, y_binary, zero_division=0)),
        "threshold_recall": float(recall_score(y_label, y_binary, zero_division=0)),
        "threshold_f1": float(f1_score(y_label, y_binary, zero_division=0)),
        "threshold_confusion_matrix": confusion_matrix(y_label, y_binary).tolist(),
    }
    if len(set(y_label.tolist())) > 1:
        metrics["truth_roc_auc"] = float(roc_auc_score(y_label, y_pred))
    return metrics


def _predict_evidence_scores(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = [int(label) for label in getattr(model, "classes_", [0, 1])]
        if 1 in classes:
            return np.asarray(probabilities[:, classes.index(1)] * 100.0, dtype=float)
    return np.asarray(model.predict(X) * 100.0, dtype=float)


def _evaluate_by_variant(
    model: RandomForestClassifier,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    labels_dev: np.ndarray,
    variants_dev: list[str],
) -> dict[str, object]:
    predictions = np.asarray([_bounded_score(float(value)) for value in _predict_evidence_scores(model, X_dev)])
    out: dict[str, object] = {
        "all_dev_variants": _evaluate_predictions(y_dev, labels_dev, predictions),
    }
    for variant in sorted(set(variants_dev)):
        mask = np.asarray([item == variant for item in variants_dev], dtype=bool)
        if not mask.any():
            continue
        out[variant] = _evaluate_predictions(y_dev[mask], labels_dev[mask], predictions[mask])
    return out


def main() -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}\n"
            "Run prepare_audio_dataset.py first."
        )
    if not RAW_AUDIO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Raw audio model not found: {RAW_AUDIO_MODEL_PATH}")
    if not BEHAVIORAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Behavioral model not found: {BEHAVIORAL_MODEL_PATH}")

    labels = pd.read_csv(LABELS_PATH)
    required_columns = {"split", "relative_path", "label"}
    missing = required_columns - set(labels.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")

    print("Loading source models...")
    audio_classifier = load_audio_model(RAW_AUDIO_MODEL_PATH)
    behavioral_classifier = load_audio_behavior_model(BEHAVIORAL_MODEL_PATH)

    print("Building voice-evidence calibration rows...")
    feature_start = time.time()
    train_samples = _load_samples(
        labels,
        "train",
        audio_classifier,
        behavioral_classifier,
        full_per_label=TRAIN_FULL_PER_LABEL,
        augment_per_label=TRAIN_AUGMENT_PER_LABEL,
    )
    dev_samples = _load_samples(
        labels,
        "dev",
        audio_classifier,
        behavioral_classifier,
        full_per_label=DEV_FULL_PER_LABEL,
        augment_per_label=None,
    )
    feature_seconds = time.time() - feature_start

    if not train_samples or not dev_samples:
        raise RuntimeError("No usable rows were built for voice-evidence calibration.")

    X_train, y_train, labels_train, variants_train = _matrix(train_samples)
    X_dev, y_dev, labels_dev, variants_dev = _matrix(dev_samples)

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    print("\nTraining voice-evidence calibrator...")
    training_start = time.time()
    y_train_evidence_class = (y_train >= 50.0).astype(int)
    model.fit(X_train, y_train_evidence_class)
    training_seconds = time.time() - training_start

    print("Evaluating voice-evidence calibrator...")
    metrics = _evaluate_by_variant(model, X_dev, y_dev, labels_dev, variants_dev)

    importances = [
        {"feature": name, "importance": float(importance)}
        for name, importance in sorted(
            zip(VOICE_EVIDENCE_FEATURE_NAMES, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    payload = {
        "model": model,
        "feature_names": VOICE_EVIDENCE_FEATURE_NAMES,
        "target": "voice evidence risk percentage",
        "target_policy": (
            "bonafide clips target 0; spoof clips target label strength scaled by speech-evidence quality "
            "so short, quiet, or sparse clips train toward limited evidence instead of conclusive deepfake risk."
        ),
        "version": 1,
    }
    joblib.dump(payload, MODEL_PATH)

    summary = {
        "dataset_path": str(DATA_DIR),
        "labels_path": str(LABELS_PATH),
        "raw_audio_model_path": str(RAW_AUDIO_MODEL_PATH),
        "behavioral_model_path": str(BEHAVIORAL_MODEL_PATH),
        "model_path": str(MODEL_PATH),
        "task": "Second-stage trained voice evidence calibration",
        "feature_names": VOICE_EVIDENCE_FEATURE_NAMES,
        "feature_dimension": len(VOICE_EVIDENCE_FEATURE_NAMES),
        "train_rows": int(len(y_train)),
        "dev_rows": int(len(y_dev)),
        "training_limits": {
            "train_full_per_label": TRAIN_FULL_PER_LABEL,
            "train_augment_per_label": TRAIN_AUGMENT_PER_LABEL,
            "dev_full_per_label": DEV_FULL_PER_LABEL,
        },
        "train_distribution": {
            "bonafide": int((labels_train == 0).sum()),
            "spoof": int((labels_train == 1).sum()),
        },
        "dev_distribution": {
            "bonafide": int((labels_dev == 0).sum()),
            "spoof": int((labels_dev == 1).sum()),
        },
        "train_variants": {
            variant: int(variants_train.count(variant))
            for variant in sorted(set(variants_train))
        },
        "dev_variants": {
            variant: int(variants_dev.count(variant))
            for variant in sorted(set(variants_dev))
        },
        "feature_extraction_seconds": float(feature_seconds),
        "training_seconds": float(training_seconds),
        "model": "RandomForestClassifier trained as voice evidence calibrator",
        "metrics": metrics,
        "feature_importances": importances,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nVoice evidence calibration complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("Dev full metrics:", json.dumps(metrics.get("full", {}), indent=2))


if __name__ == "__main__":
    main()
