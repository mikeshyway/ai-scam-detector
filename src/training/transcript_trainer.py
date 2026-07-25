"""Train transcript scam classifiers for call / meeting transcript analysis.

Input:
    data/processed/transcript/transcript_dataset.csv

Expected columns:
    transcript,label,source

Output:
    models/transcript_vectorizer.pkl
    models/transcript_nb.pkl
    models/transcript_svm.pkl
    models/transcript_distilbert/  (optional)
    models/transcript_bert/        (optional)
    reports/metrics/transcript_model_metrics.json

Run:
    py scripts/05_train_transcript_model.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.training.transcript_transformer import (
    TRANSFORMER_MODEL_CONFIGS,
    TransformerTrainingConfig,
    train_transformer_model,
)


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data" / "processed" / "transcript" / "transcript_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODELS_DIR / "transcript_vectorizer.pkl"
METRICS_PATH = REPORTS_DIR / "transcript_model_metrics.json"

LABEL_NAMES = ["Legitimate", "Suspicious"]


def get_scores(model, X_test):
    """Return suspicious-class scores for ROC-AUC."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        min_score = scores.min()
        max_score = scores.max()
        return (scores - min_score) / max(1e-9, max_score - min_score)

    return model.predict(X_test)


def evaluate_model(name, model, X_test, y_test) -> dict[str, object]:
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - start_pred
    y_score = get_scores(model, X_test)

    fpr, tpr, _thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "training_time_seconds": None,
        "prediction_time_seconds": float(prediction_time),
        "prediction_time_ms": float(prediction_time / max(1, X_test.shape[0]) * 1000),
        "confusion_matrix": cm.tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=LABEL_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
    }


def build_models() -> dict[str, object]:
    models: dict[str, object] = {
        "Naive Bayes": MultinomialNB(alpha=0.8),
        "SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                random_state=42,
                class_weight="balanced",
                C=0.8,
            ),
            method="sigmoid",
            cv=5,
        ),
    }

    return models


def _normalise_transformer_keys(model_keys: list[str] | None) -> list[str]:
    if not model_keys:
        return ["distilbert"]
    valid_keys = []
    for key in model_keys:
        normalised = str(key).strip().casefold()
        if normalised in TRANSFORMER_MODEL_CONFIGS and normalised not in valid_keys:
            valid_keys.append(normalised)
    return valid_keys or ["distilbert"]


def _load_existing_transformer_metrics(model_keys: list[str]) -> dict[str, dict[str, object]]:
    if not METRICS_PATH.exists():
        return {}
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    model_metrics = metrics.get("models", {})
    if not isinstance(model_metrics, dict):
        return {}

    existing: dict[str, dict[str, object]] = {}
    for key in model_keys:
        values = TRANSFORMER_MODEL_CONFIGS[key]
        display_name = values["display_name"]
        artifact_dir = MODELS_DIR / values["artifact_dir"]
        metric_values = model_metrics.get(display_name)
        if artifact_dir.exists() and isinstance(metric_values, dict):
            existing[display_name] = dict(metric_values)
    return existing


def _train_optional_transformers(
    *,
    model_keys: list[str],
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    allow_download: bool,
) -> tuple[dict[str, object], dict[str, str]]:
    transformer_metrics: dict[str, object] = {}
    transformer_errors: dict[str, str] = {}

    for key in _normalise_transformer_keys(model_keys):
        config_values = TRANSFORMER_MODEL_CONFIGS[key]
        display_name = config_values["display_name"]
        print(f"\nTraining {display_name} transformer...")
        config = TransformerTrainingConfig(
            key=key,
            display_name=display_name,
            checkpoint=config_values["checkpoint"],
            artifact_dir=MODELS_DIR / config_values["artifact_dir"],
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_length,
            learning_rate=learning_rate,
            allow_download=allow_download,
        )
        try:
            metrics = train_transformer_model(
                config,
                X_train.tolist(),
                y_train.tolist(),
                X_test.tolist(),
                y_test.tolist(),
            )
        except Exception as exc:
            message = str(exc)
            transformer_errors[display_name] = message
            print(f"Skipping {display_name}: {message}")
            continue

        transformer_metrics[display_name] = metrics
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"Training : {metrics['training_time_seconds']:.2f}s")

    return transformer_metrics, transformer_errors


def main(
    *,
    include_transformers: bool = False,
    transformer_models: list[str] | None = None,
    transformer_epochs: int = 2,
    transformer_batch_size: int = 8,
    transformer_max_length: int = 256,
    transformer_learning_rate: float = 2e-5,
    allow_transformer_download: bool = True,
) -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run prepare_transcript_dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    if "transcript" not in df.columns or "label" not in df.columns:
        raise ValueError("transcript_dataset.csv must contain transcript and label columns.")

    df = df.dropna(subset=["transcript", "label"]).copy()
    df["transcript"] = df["transcript"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["transcript"].str.strip().str.len() > 0]

    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both labels: 0 legitimate and 1 suspicious.")

    print("Transcript dataset loaded")
    print(f"Total samples: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    source_distribution = (
        df["source"].value_counts().to_dict()
        if "source" in df.columns
        else {}
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["transcript"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 3),
        stop_words="english",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    print("\nVectorising transcripts...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    models = build_models()
    all_metrics: dict[str, object] = {}
    training_times: dict[str, float] = {}
    active_transformer_keys = _normalise_transformer_keys(transformer_models)
    existing_transformer_metrics = _load_existing_transformer_metrics(active_transformer_keys)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train_vec, y_train)
        elapsed = time.time() - start
        training_times[name] = elapsed

        metrics = evaluate_model(name, model, X_test_vec, y_test)
        metrics["training_seconds"] = float(elapsed)
        metrics["training_time_seconds"] = float(elapsed)
        metrics["training_time"] = float(elapsed)
        all_metrics[name] = metrics

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"Training : {elapsed:.2f}s")

    transformer_errors: dict[str, str] = {}
    if include_transformers:
        transformer_metrics, transformer_errors = _train_optional_transformers(
            model_keys=active_transformer_keys,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=transformer_epochs,
            batch_size=transformer_batch_size,
            max_length=transformer_max_length,
            learning_rate=transformer_learning_rate,
            allow_download=allow_transformer_download,
        )
        all_metrics.update(transformer_metrics)

    preserved_transformer_metrics = {}
    for model_name, metric_values in existing_transformer_metrics.items():
        if model_name not in all_metrics:
            all_metrics[model_name] = metric_values
            preserved_transformer_metrics[model_name] = "Preserved from previous transformer training run."

    recommended_model_name = max(
        all_metrics,
        key=lambda model_name: all_metrics[model_name]["f1"],
    )

    print(f"\nTop validation benchmark: {recommended_model_name}")
    print(f"Recommended F1: {all_metrics[recommended_model_name]['f1']:.4f}")

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(models["Naive Bayes"], MODELS_DIR / "transcript_nb.pkl")
    joblib.dump(models["SVM"], MODELS_DIR / "transcript_svm.pkl")

    for stale_filename in (
        "transcript_best.pkl",
        "transcript_dt.pkl",
        "transcript_rf.pkl",
        "transcript_xgb.pkl",
    ):
        stale_path = MODELS_DIR / stale_filename
        if stale_path.exists():
            stale_path.unlink()

    summary = {
        "dataset_path": str(DATASET_PATH),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "label_distribution": {
            "legitimate": int((df["label"] == 0).sum()),
            "suspicious": int((df["label"] == 1).sum()),
        },
        "source_distribution": source_distribution,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "max_features": 20000,
            "ngram_range": [1, 3],
            "stop_words": "english",
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "vocabulary_size": int(len(vectorizer.get_feature_names_out())),
        },
        "recommended_model": recommended_model_name,
        "top_validation_model": recommended_model_name,
        "recommended_f1": float(all_metrics[recommended_model_name]["f1"]),
        "models": all_metrics,
        "saved_files": {
            "vectorizer": "models/transcript_vectorizer.pkl",
            "naive_bayes": "models/transcript_nb.pkl",
            "svm": "models/transcript_svm.pkl",
            "distilbert": "models/transcript_distilbert"
            if (MODELS_DIR / "transcript_distilbert").exists()
            else None,
        },
        "transformer_training": {
            "enabled": bool(include_transformers),
            "requested_models": _normalise_transformer_keys(transformer_models),
            "epochs": int(transformer_epochs),
            "batch_size": int(transformer_batch_size),
            "max_length": int(transformer_max_length),
            "learning_rate": float(transformer_learning_rate),
            "allow_download": bool(allow_transformer_download),
            "errors": transformer_errors,
            "preserved_metrics": preserved_transformer_metrics,
        },
        "note": (
            "Recommended model is stored as training metadata only. "
            "Live transcript analysis should use selected model families and consensus, not a duplicate best-model artifact."
        ),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print("\nTraining complete.")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
