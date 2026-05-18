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
    "sklearn",
    "nltk",
    "librosa",
    "soundfile",
    "matplotlib",
    "plotly",
    "joblib",
]

EXPECTED_DIRS = [
    "app",
    "src",
    "scripts",
    "models",
    "data/raw/spamassassin/spam",
    "data/raw/spamassassin/ham",
    "data/raw/transcripts",
    "data/raw/asvspoof_subset",
    "data/processed",
    "notebooks",
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
