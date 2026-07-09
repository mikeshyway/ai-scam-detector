"""Verify the local Python environment and expected project folders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "sounddevice",
    "sklearn",
    "nltk",
    "librosa",
    "pydub",
    "soundfile",
    "scipy",
    "matplotlib",
    "plotly",
    "requests",
    "whisper",
    "transformers",
    "torch",
    "joblib",
    "reportlab",
    "docx",
    "pypdf",
    "extract_msg",
]

OPTIONAL_PACKAGES = [
]

EXPECTED_DIRS = [
    "app",
    "src",
    "scripts",
    "models",
    "data/raw/email",
    "data/raw/voice_transcript",
    "data/raw/asvspoof_2019_dataset_subset",
    "data/processed/email",
    "data/processed/transcript",
    "data/processed/audio",
    "data/processed/phone",
    "notebooks",
    "reports/metrics",
    "tests",
]


def main() -> int:
    print(f"Python: {sys.version}")
    print(f"Project root: {ROOT}")

    missing_packages = []
    for package in REQUIRED_PACKAGES:
        available = importlib.util.find_spec(package) is not None
        print(f"{'OK' if available else 'MISSING':7} {package}")
        if not available:
            missing_packages.append(package)

    print("\nOptional packages")
    for package, note in OPTIONAL_PACKAGES:
        available = importlib.util.find_spec(package) is not None
        print(f"{'OK' if available else 'OPTIONAL':7} {package} - {note}")

    print("\nFolders")
    for relative in EXPECTED_DIRS:
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        print(f"OK      {relative}")

    if missing_packages:
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return 1

    print("\nSetup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
