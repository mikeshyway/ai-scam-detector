from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data" / "processed" / "email_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - min_score) / (max_score - min_score)

    return model.predict(X_test)


def evaluate_model(name, model, X_test, y_test, training_time: float):
    prediction_start = perf_counter()
    y_pred = model.predict(X_test)
    prediction_total_time = perf_counter() - prediction_start
    y_score = get_scores(model, X_test)
    fpr, tpr, _thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    sample_count = max(1, int(getattr(X_test, "shape", [len(y_test)])[0]))

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc": float(roc_auc),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "training_time": float(training_time),
        "prediction_time_ms": float((prediction_total_time / sample_count) * 1000),
        "prediction_total_time_ms": float(prediction_total_time * 1000),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Safe", "Suspicious"],
            zero_division=0,
            output_dict=True,
        ),
    }


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run prepare_email_dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    print("Dataset loaded")
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=20,
            ccp_alpha=0.002,
            class_weight="balanced",
        ),
        "SVM": LinearSVC(
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        print("XGBoost not installed. Skipping XGBoost.")

    all_metrics = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        training_start = perf_counter()
        model.fit(X_train_vec, y_train)
        training_time = perf_counter() - training_start

        metrics = evaluate_model(name, model, X_test_vec, y_test, training_time)
        all_metrics[name] = metrics

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"Train sec: {metrics['training_time']:.2f}")
        print(f"Predict : {metrics['prediction_time_ms']:.4f} ms/sample")

    best_model_name = max(all_metrics, key=lambda name: all_metrics[name]["f1"])
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print(f"Best F1: {all_metrics[best_model_name]['f1']:.4f}")

    joblib.dump(vectorizer, MODELS_DIR / "email_vectorizer.pkl")

    joblib.dump(models["Naive Bayes"], MODELS_DIR / "email_nb.pkl")
    joblib.dump(models["Decision Tree"], MODELS_DIR / "email_dt.pkl")
    joblib.dump(models["SVM"], MODELS_DIR / "email_svm.pkl")
    joblib.dump(models["Random Forest"], MODELS_DIR / "email_rf.pkl")

    if "XGBoost" in models:
        joblib.dump(models["XGBoost"], MODELS_DIR / "email_xgb.pkl")

    joblib.dump(best_model, MODELS_DIR / "email_best.pkl")

    summary = {
        "schema_version": 2,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_labels": {"0": "Safe", "1": "Suspicious"},
        "best_model": best_model_name,
        "best_f1": float(all_metrics[best_model_name]["f1"]),
        "models": all_metrics,
        "saved_files": {
            "vectorizer": "models/email_vectorizer.pkl",
            "naive_bayes": "models/email_nb.pkl",
            "decision_tree": "models/email_dt.pkl",
            "svm": "models/email_svm.pkl",
            "random_forest": "models/email_rf.pkl",
            "xgboost": "models/email_xgb.pkl" if XGBOOST_AVAILABLE else None,
            "best_model": "models/email_best.pkl",
        },
    }

    with open(REPORTS_DIR / "email_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Vectorizer saved to: {MODELS_DIR / 'email_vectorizer.pkl'}")
    print(f"Metrics saved to: {REPORTS_DIR / 'email_model_metrics.json'}")


if __name__ == "__main__":
    main()
