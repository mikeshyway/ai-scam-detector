"""Prepare a unified English transcript dataset for scam classification.

Combines multiple transcript sources into:
    data/processed/transcript/transcript_dataset.csv

Supported sources (optional):
- data/raw/call_transcripts_scam_determinations/
- data/raw/youtube_scam_transcripts/

Output columns:
    transcript,label,source
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "transcript"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    if s in {"1","spam","scam","fraud","phishing","yes","true","suspicious"}:
        return 1
    return 0


rows = []

sources = [
    ("Call Transcripts Scam Determinations", RAW / "call_transcripts_scam_determinations"),
    ("YouTube Scam Phone Call Transcripts", RAW / "youtube_scam_transcripts"),
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

        for _, r in df.iterrows():
            text = clean_text(r[text_col])

            if len(text.split()) < MIN_WORDS:
                continue

            label = normalise_label(r[label_col]) if label_col else 1

            rows.append({
                "transcript": text,
                "label": label,
                "source": source_name,
            })

    for txt_file in folder.rglob("*.txt"):
        text = clean_text(txt_file.read_text(encoding="utf-8", errors="ignore"))
        if len(text.split()) < MIN_WORDS:
            continue

        # Heuristic: filenames containing scam/fraud/phishing are suspicious.
        name = txt_file.stem.lower()
        label = 1 if any(k in name for k in ["scam","fraud","phishing"]) else 0

        rows.append({
            "transcript": text,
            "label": label,
            "source": source_name,
        })

dataset = pd.DataFrame(rows)

if dataset.empty:
    raise RuntimeError("No transcript samples found. Check raw dataset folders.")

dataset.drop_duplicates(subset="transcript", inplace=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

dataset.to_csv(OUTPUT, index=False)

print("\nTranscript dataset prepared.")
print("Saved to:", OUTPUT)
print("Samples:", len(dataset))
print(dataset["label"].value_counts())
print(dataset["source"].value_counts())
