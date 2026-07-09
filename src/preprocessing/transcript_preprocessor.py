"""Prepare a unified English transcript dataset for scam classification.

Combines multiple transcript sources into:
    data/processed/transcript/transcript_dataset.csv

Supported sources (optional):
- data/raw/voice_transcript/call_transcripts_scam_determinations/
- data/raw/voice_transcript/youtube_scam_phone_call_transcripts/

Output columns:
    transcript,label,source
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw" / "voice_transcript"
OUT_DIR = ROOT / "data" / "processed" / "transcript"
OUTPUT = OUT_DIR / "transcript_dataset.csv"

MIN_WORDS = 8


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "transcript", "text", "content", "message",
        "call_transcript", "conversation", "speech"
    ]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return df.columns[0]


def detect_label_column(df: pd.DataFrame) -> str | None:
    candidates = ["label", "class", "target", "scam"]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def normalise_label(value) -> int:
    s = str(value).strip().lower()
    if s in {"1", "spam", "scam", "fraud", "phishing", "yes", "true", "suspicious"}:
        return 1
    return 0


def prepare_transcript_dataset(
    raw_dir: Path = RAW,
    output_file: Path = OUTPUT,
) -> pd.DataFrame:
    """Build and save the unified transcript dataset."""
    rows: list[dict[str, object]] = []
    sources = [
        ("Call Transcripts Scam Determinations", raw_dir / "call_transcripts_scam_determinations"),
        ("YouTube Scam Phone Call Transcripts", raw_dir / "youtube_scam_phone_call_transcripts"),
    ]

    for source_name, folder in sources:
        if not folder.exists():
            print(f"Skipping missing source: {folder}")
            continue

        for csv_file in folder.rglob("*.csv"):
            print("Reading", csv_file.name)
            df = pd.read_csv(csv_file)
            text_col = detect_text_column(df)
            label_col = detect_label_column(df)

            for _, row in df.iterrows():
                text = clean_text(row[text_col])
                if len(text.split()) < MIN_WORDS:
                    continue

                rows.append(
                    {
                        "transcript": text,
                        "label": normalise_label(row[label_col]) if label_col else 1,
                        "source": source_name,
                    }
                )

        for txt_file in folder.rglob("*.txt"):
            text = clean_text(txt_file.read_text(encoding="utf-8", errors="ignore"))
            if len(text.split()) < MIN_WORDS:
                continue

            name = txt_file.stem.lower()
            rows.append(
                {
                    "transcript": text,
                    "label": 1 if any(key in name for key in ["scam", "fraud", "phishing"]) else 0,
                    "source": source_name,
                }
            )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise RuntimeError("No transcript samples found. Check raw dataset folders.")

    dataset.drop_duplicates(subset="transcript", inplace=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_file, index=False)
    return dataset


def main() -> None:
    dataset = prepare_transcript_dataset()
    print("\nTranscript dataset prepared.")
    print("Saved to:", OUTPUT)
    print("Samples:", len(dataset))
    print(dataset["label"].value_counts())
    print(dataset["source"].value_counts())


if __name__ == "__main__":
    main()
