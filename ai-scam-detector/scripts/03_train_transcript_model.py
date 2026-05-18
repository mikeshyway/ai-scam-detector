"""Train transcript scam classifier from a labeled scam/non-scam CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_classifier import save_text_artifacts, train_text_models
from src.text_preprocessor import load_labeled_text_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=ROOT / "data" / "raw" / "transcripts" / "scam_nonscam_calls.csv",
    )
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--metrics-out", default=ROOT / "models" / "transcript_metrics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_labeled_text_csv(
        args.csv,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    print(f"Loaded {len(df)} transcript samples.")
    print(df["label"].value_counts().rename(index={0: "non-scam", 1: "scam"}))

    vectorizer, models, metrics = train_text_models(
        df["text"],
        df["label"],
        include_decision_tree=False,
        max_features=args.max_features,
    )
    save_text_artifacts(
        vectorizer,
        models["nb"],
        ROOT / "models" / "transcript_vectorizer.pkl",
        ROOT / "models" / "transcript_nb.pkl",
    )

    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({name: {"accuracy": m["accuracy"], "f1": m["f1"]} for name, m in metrics.items()}, indent=2))
    print("Saved transcript_vectorizer.pkl and transcript_nb.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

