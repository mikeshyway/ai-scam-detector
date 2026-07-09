"""Train transcript scam classifiers for call / meeting transcript analysis.

Input:
    data/processed/transcript/transcript_dataset.csv

Expected columns:
    transcript,label,source

Output:
    models/transcript_vectorizer.pkl
    models/transcript_nb.pkl
    models/transcript_dt.pkl
    models/transcript_svm.pkl
    models/transcript_rf.pkl
    models/transcript_xgb.pkl  (if xgboost is installed)
    models/transcript_best.pkl
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
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


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
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)

    fpr, tpr, _thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
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
        "Decision Tree": CalibratedClassifierCV(
            estimator=DecisionTreeClassifier(
                random_state=42,
                max_depth=8,
                min_samples_leaf=10,
                min_samples_split=20,
                ccp_alpha=0.002,
                class_weight="balanced",
            ),
            method="sigmoid",
            cv=5,
        ),
        "SVM": CalibratedClassifierCV(
            estimator=LinearSVC(
                random_state=42,
                class_weight="balanced",
                C=0.8,
            ),
            method="sigmoid",
            cv=5,
        ),
        "Random Forest": CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=5,
                min_samples_split=10,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            method="sigmoid",
            cv=5,
        ),
    }

    if XGBOOST_AVAILABLE and XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=1.5,
            gamma=0.2,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        print("XGBoost not installed. Skipping XGBoost.")

    return models


def main() -> None:
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

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train_vec, y_train)
        elapsed = time.time() - start
        training_times[name] = elapsed

        metrics = evaluate_model(name, model, X_test_vec, y_test)
        metrics["training_seconds"] = float(elapsed)
        all_metrics[name] = metrics

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"Training : {elapsed:.2f}s")

    best_model_name = max(all_metrics, key=lambda model_name: all_metrics[model_name]["f1"])
    best_model = models[best_model_name]

    print(f"\nTop validation model: {best_model_name}")
    print(f"Best F1: {all_metrics[best_model_name]['f1']:.4f}")

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(models["Naive Bayes"], MODELS_DIR / "transcript_nb.pkl")
    joblib.dump(models["Decision Tree"], MODELS_DIR / "transcript_dt.pkl")
    joblib.dump(models["SVM"], MODELS_DIR / "transcript_svm.pkl")
    joblib.dump(models["Random Forest"], MODELS_DIR / "transcript_rf.pkl")

    if "XGBoost" in models:
        joblib.dump(models["XGBoost"], MODELS_DIR / "transcript_xgb.pkl")

    joblib.dump(best_model, MODELS_DIR / "transcript_best.pkl")

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
        "top_validation_model": best_model_name,
        "best_model": best_model_name,
        "best_f1": float(all_metrics[best_model_name]["f1"]),
        "models": all_metrics,
        "saved_files": {
            "vectorizer": "models/transcript_vectorizer.pkl",
            "naive_bayes": "models/transcript_nb.pkl",
            "decision_tree": "models/transcript_dt.pkl",
            "svm": "models/transcript_svm.pkl",
            "random_forest": "models/transcript_rf.pkl",
            "xgboost": "models/transcript_xgb.pkl" if "XGBoost" in models else None,
            "best_model": "models/transcript_best.pkl",
        },
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nTraining complete.")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
