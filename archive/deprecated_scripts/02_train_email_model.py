"""Compatibility wrapper for the canonical email training pipeline.

The real email model training flow lives in src/training/train_email_model.py.
Keeping this script as a wrapper prevents older capstone instructions from
writing stale metrics to models/email_metrics.json or training only the legacy
SpamAssassin subset.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_email_model import main as train_email_models


def main() -> int:
    train_email_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
