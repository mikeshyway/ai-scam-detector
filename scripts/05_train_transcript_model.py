"""Train and evaluate transcript classifiers."""

import argparse

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.transcript_trainer import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transcript scam classifiers.")
    parser.add_argument(
        "--include-transformers",
        action="store_true",
        help="Also fine-tune transformer models and include them in transcript metrics.",
    )
    parser.add_argument(
        "--transformer-models",
        default="distilbert",
        help="Comma-separated transformer keys: distilbert,bert.",
    )
    parser.add_argument("--transformer-epochs", type=int, default=2)
    parser.add_argument("--transformer-batch-size", type=int, default=8)
    parser.add_argument("--transformer-max-length", type=int, default=256)
    parser.add_argument("--transformer-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--no-transformer-download",
        action="store_true",
        help="Use only locally cached Hugging Face checkpoints.",
    )
    args = parser.parse_args()

    main(
        include_transformers=args.include_transformers,
        transformer_models=[
            key.strip()
            for key in args.transformer_models.split(",")
            if key.strip()
        ],
        transformer_epochs=args.transformer_epochs,
        transformer_batch_size=args.transformer_batch_size,
        transformer_max_length=args.transformer_max_length,
        transformer_learning_rate=args.transformer_learning_rate,
        allow_transformer_download=not args.no_transformer_download,
    )
