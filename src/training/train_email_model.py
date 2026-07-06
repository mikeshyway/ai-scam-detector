from pathlib import Path
import json
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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

DATASET_PATH = ROOT / "data" / "processed" / "email" / "email_dataset.csv"
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
            return scores

        return (scores - min_score) / (max_score - min_score)

    return model.predict(X_test)


def evaluate_model(name, model, X_test, y_test):
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
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "training_time_seconds": None,
        "prediction_time_seconds": prediction_time,
        "prediction_time_ms": prediction_time / max(1, X_test.shape[0]) * 1000,
        "confusion_matrix": cm.tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Suspicious"],
            zero_division=0,
            output_dict=True,
        ),
    }


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run src/preprocessing/prepare_email_dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    print("Dataset loaded")
    print(f"Total samples: {len(df)}")
    print("\nLabel counts:")
    print(df["label"].value_counts())

    source_counts = (
        df["source"].value_counts().to_dict()
        if "source" in df.columns
        else {}
    )

    if source_counts:
        print("\nSource counts:")
        print(df["source"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        max_features=15000,
        sublinear_tf=True,
    )

    print("\nVectorizing text...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
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
            ),
            method="sigmoid",
            cv=5,
        ),
        "Random Forest": CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=3,
                min_samples_split=10,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            method="sigmoid",
            cv=5,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=1,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        print("XGBoost not installed. Skipping XGBoost.")

    all_metrics = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        start_train = time.perf_counter()
        model.fit(X_train_vec, y_train)
        training_time = time.perf_counter() - start_train

        metrics = evaluate_model(name, model, X_test_vec, y_test)
        metrics["training_time_seconds"] = training_time
        all_metrics[name] = metrics

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"Train s  : {training_time:.3f}")

    best_model_name = max(all_metrics, key=lambda name: all_metrics[name]["f1"])
    best_model = models[best_model_name]

    print(f"\nTop validation benchmark: {best_model_name}")
    print(f"Best F1: {all_metrics[best_model_name]['f1']:.4f}")

    joblib.dump(vectorizer, MODELS_DIR / "email_vectorizer.pkl")

    file_map = {
        "Naive Bayes": "email_nb.pkl",
        "Decision Tree": "email_dt.pkl",
        "SVM": "email_svm.pkl",
        "Random Forest": "email_rf.pkl",
        "XGBoost": "email_xgb.pkl",
    }

    for name, filename in file_map.items():
        if name in models:
            joblib.dump(models[name], MODELS_DIR / filename)

    joblib.dump(best_model, MODELS_DIR / "email_best.pkl")

    summary = {
        "dataset_path": str(DATASET_PATH),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "label_distribution": {
            str(label): int(count)
            for label, count in df["label"].value_counts().to_dict().items()
        },
        "dataset_sources": {
            str(source): int(count)
            for source, count in source_counts.items()
        },
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 3],
            "max_features": 15000,
            "min_df": 2,
            "max_df": 0.95,
            "stop_words": "english",
            "sublinear_tf": True,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "top_validation_model": best_model_name,
        "best_model": best_model_name,
        "best_f1": all_metrics[best_model_name]["f1"],
        "models": all_metrics,
        "saved_files": {
            "vectorizer": "models/email_vectorizer.pkl",
            "naive_bayes": "models/email_nb.pkl",
            "decision_tree": "models/email_dt.pkl",
            "svm": "models/email_svm.pkl",
            "random_forest": "models/email_rf.pkl",
            "xgboost": "models/email_xgb.pkl" if XGBOOST_AVAILABLE else None,
            "benchmark_model": "models/email_best.pkl",
        },
        "note": (
            "Best model is saved as a validation benchmark. "
            "Live analysis should use multi-model consensus, not only email_best.pkl."
        ),
    }

    with open(REPORTS_DIR / "email_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Vectorizer saved to: {MODELS_DIR / 'email_vectorizer.pkl'}")
    print(f"Metrics saved to: {REPORTS_DIR / 'email_model_metrics.json'}")

    print("\nSummary:")
    for name, metrics in all_metrics.items():
        print(
            f"{name:18} "
            f"F1={metrics['f1']:.4f} "
            f"ROC-AUC={metrics['roc_auc']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f}"
        )


if __name__ == "__main__":
    main()