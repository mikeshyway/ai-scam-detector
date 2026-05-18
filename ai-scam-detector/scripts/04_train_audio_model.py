"""Train the MFCC + SVM AI-generated speech detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_classifier import save_audio_model, train_audio_svm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", default=ROOT / "data" / "processed" / "audio_X.npy")
    parser.add_argument("--y", default=ROOT / "data" / "processed" / "audio_y.npy")
    parser.add_argument("--model-out", default=ROOT / "models" / "audio_svm.pkl")
    parser.add_argument("--metrics-out", default=ROOT / "models" / "audio_metrics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    X = np.load(args.x)
    y = np.load(args.y)
    print(f"Loaded audio features: {X.shape}")
    print(f"Labels: {np.bincount(y).tolist()} [real, fake]")

    model, metrics = train_audio_svm(X, y)
    save_audio_model(model, args.model_out)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({"accuracy": metrics["accuracy"], "f1": metrics["f1"]}, indent=2))
    print(f"Saved {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

