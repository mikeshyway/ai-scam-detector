"""Train email phishing classifiers from SpamAssassin spam/ham folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_classifier import save_text_artifacts, train_text_models
from src.text_preprocessor import load_spamassassin_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=ROOT / "data" / "raw" / "spamassassin")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--metrics-out", default=ROOT / "models" / "email_metrics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_spamassassin_dataset(args.data_dir)
    print(f"Loaded {len(df)} email samples.")
    print(df["label"].value_counts().rename(index={0: "ham", 1: "spam"}))

    vectorizer, models, metrics = train_text_models(
        df["text"],
        df["label"],
        include_decision_tree=True,
        max_features=args.max_features,
    )

    save_text_artifacts(
        vectorizer,
        models["nb"],
        ROOT / "models" / "email_vectorizer.pkl",
        ROOT / "models" / "email_nb.pkl",
    )
    save_text_artifacts(
        vectorizer,
        models["dt"],
        ROOT / "models" / "email_vectorizer.pkl",
        ROOT / "models" / "email_dt.pkl",
    )

    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({name: {"accuracy": m["accuracy"], "f1": m["f1"]} for name, m in metrics.items()}, indent=2))
    print("Saved email_vectorizer.pkl, email_nb.pkl, and email_dt.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
