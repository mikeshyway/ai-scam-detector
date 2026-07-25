"""Prepare a unified English transcript dataset for scam classification.

Combines multiple transcript sources into:
    data/processed/transcript/transcript_dataset.csv

Supported sources (optional):
- data/raw/voice_transcript/call_transcripts_scam_determinations/
- data/raw/voice_transcript/youtube_scam_phone_call_transcripts/

Supervised output columns:
    transcript,label,source

Unlabeled rows are written separately for manual review instead of being
silently treated as suspicious training examples.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw" / "voice_transcript"
OUT_DIR = ROOT / "data" / "processed" / "transcript"
OUTPUT = OUT_DIR / "transcript_dataset.csv"
UNLABELED_REVIEW_OUTPUT = OUT_DIR / "transcript_unlabeled_review.csv"

MIN_WORDS = 8

SUSPICIOUS_LABELS = {
    "1",
    "spam",
    "scam",
    "fraud",
    "phishing",
    "yes",
    "true",
    "suspicious",
    "potential_scam",
    "slightly_suspicious",
    "highly_suspicious",
}

LEGITIMATE_LABELS = {
    "0",
    "ham",
    "legit",
    "legitimate",
    "neutral",
    "no",
    "false",
    "benign",
    "safe",
    "scam_response",
    "polite_ending",
    "adhering_to_protocols",
    "adhering to protocols",
    "ready_for_further_engagement",
    "ready for further engagement",
}

SUSPICIOUS_LABEL_HINTS = (
    "urgency",
    "dangerous",
    "dismissing official",
    "identification_request",
    "identification request",
    "security and compliance",
)


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


def normalise_label(value) -> int | None:
    s = str(value).strip().lower().strip('"').strip("'")
    normalised = re.sub(r"\s+", "_", s)

    if s in SUSPICIOUS_LABELS or normalised in SUSPICIOUS_LABELS:
        return 1
    if s in LEGITIMATE_LABELS or normalised in LEGITIMATE_LABELS:
        return 0
    if any(hint in s or hint in normalised for hint in SUSPICIOUS_LABEL_HINTS):
        return 1

    return None


def prepare_transcript_dataset(
    raw_dir: Path = RAW,
    output_file: Path = OUTPUT,
    unlabeled_review_file: Path = UNLABELED_REVIEW_OUTPUT,
) -> pd.DataFrame:
    """Build and save the unified transcript dataset."""
    rows: list[dict[str, object]] = []
    unlabeled_rows: list[dict[str, object]] = []
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

                if not label_col:
                    unlabeled_rows.append(
                        {
                            "transcript": text,
                            "source": source_name,
                            "raw_file": csv_file.name,
                            "raw_label": "",
                            "reason": "missing_label_column",
                        }
                    )
                    continue

                label = normalise_label(row[label_col])
                if label is None:
                    unlabeled_rows.append(
                        {
                            "transcript": text,
                            "source": source_name,
                            "raw_file": csv_file.name,
                            "raw_label": str(row[label_col]),
                            "reason": "unmapped_label",
                        }
                    )
                    continue

                rows.append(
                    {
                        "transcript": text,
                        "label": label,
                        "source": source_name,
                    }
                )

        for txt_file in folder.rglob("*.txt"):
            text = clean_text(txt_file.read_text(encoding="utf-8", errors="ignore"))
            if len(text.split()) < MIN_WORDS:
                continue

            name = txt_file.stem.lower()
            label = 1 if any(key in name for key in ["scam", "fraud", "phishing"]) else None
            if label is None and any(key in name for key in ["legit", "neutral", "benign", "safe"]):
                label = 0
            if label is None:
                unlabeled_rows.append(
                    {
                        "transcript": text,
                        "source": source_name,
                        "raw_file": txt_file.name,
                        "raw_label": "",
                        "reason": "missing_label_in_filename",
                    }
                )
                continue

            rows.append(
                {
                    "transcript": text,
                    "label": label,
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

    review_dataset = pd.DataFrame(unlabeled_rows)
    if not review_dataset.empty:
        review_dataset.drop_duplicates(subset="transcript", inplace=True)
    unlabeled_review_file.parent.mkdir(parents=True, exist_ok=True)
    review_dataset.to_csv(unlabeled_review_file, index=False)

    return dataset


def main() -> None:
    dataset = prepare_transcript_dataset()
    print("\nTranscript dataset prepared.")
    print("Saved to:", OUTPUT)
    print("Samples:", len(dataset))
    print(dataset["label"].value_counts())
    print(dataset["source"].value_counts())
    if UNLABELED_REVIEW_OUTPUT.exists():
        review_dataset = pd.read_csv(UNLABELED_REVIEW_OUTPUT)
        print("Unlabeled review rows:", len(review_dataset))
        if not review_dataset.empty:
            print(review_dataset["source"].value_counts())


if __name__ == "__main__":
    main()
