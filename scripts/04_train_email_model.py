"""Train and evaluate all email classifiers."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.email_trainer import main


if __name__ == "__main__":
    main()
