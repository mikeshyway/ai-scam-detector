"""Train the second-stage voice evidence calibrator."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.audio_voice_evidence_trainer import main


if __name__ == "__main__":
    main()
