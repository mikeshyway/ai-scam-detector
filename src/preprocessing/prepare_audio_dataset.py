"""Prepare a balanced ASVspoof 2019 LA subset for capstone audio training.

This script reads the official ASVspoof 2019 LA protocol files, samples a balanced
number of bonafide and spoof audio files, copies them into one subset folder, and
creates labels.csv for downstream training.

Recommended use:
    py src/preprocessing/prepare_audio_dataset.py
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

# Change this if your ASVspoof folder is somewhere else.
SOURCE_ROOT = ROOT / "data" / "raw" / "asvspoof_2019_dataset_subset"

TRAIN_AUDIO_DIR = SOURCE_ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO_DIR = SOURCE_ROOT / "ASVspoof2019_LA_dev" / "flac"
PROTOCOL_DIR = SOURCE_ROOT / "ASVspoof2019_LA_cm_protocols"

TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"

OUTPUT_DIR = ROOT / "data" / "processed" / "audio"
OUTPUT_TRAIN_DIR = OUTPUT_DIR / "train"
OUTPUT_DEV_DIR = OUTPUT_DIR / "dev"
OUTPUT_LABELS = OUTPUT_DIR / "labels.csv"

RANDOM_STATE = 42

TRAIN_BONAFIDE_SAMPLES = 1000
TRAIN_SPOOF_SAMPLES = 1000
DEV_BONAFIDE_SAMPLES = 300
DEV_SPOOF_SAMPLES = 300

STRATIFY_SPOOF_BY_ATTACK = True


@dataclass(frozen=True)
class SplitConfig:
    split: str
    protocol_path: Path
    audio_dir: Path
    output_dir: Path
    bonafide_samples: int
    spoof_samples: int


def read_protocol(protocol_path: Path) -> pd.DataFrame:
    """Read ASVspoof LA protocol into a dataframe."""

    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    rows = []

    with protocol_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            speaker_id = parts[0]
            file_id = parts[1]
            system_id = parts[2]
            attack_id = parts[3]
            label = parts[-1].lower()

            if label not in {"bonafide", "spoof"}:
                continue

            rows.append(
                {
                    "speaker_id": speaker_id,
                    "file_id": file_id,
                    "system_id": system_id,
                    "attack_id": attack_id,
                    "label": label,
                    "file_name": f"{file_id}.flac",
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No valid rows found in protocol: {protocol_path}")

    return df


def sample_spoof_by_attack(df_spoof: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    """Sample spoof rows across attack IDs."""

    if sample_count <= 0 or df_spoof.empty:
        return df_spoof.iloc[0:0].copy()

    attack_groups = [
        group
        for _attack_id, group in df_spoof.groupby("attack_id")
        if not group.empty
    ]

    if not attack_groups:
        return df_spoof.sample(
            n=min(sample_count, len(df_spoof)),
            random_state=RANDOM_STATE,
        )

    per_attack = max(1, sample_count // len(attack_groups))
    sampled_parts = []

    for index, group in enumerate(attack_groups):
        n = min(per_attack, len(group))
        sampled_parts.append(group.sample(n=n, random_state=RANDOM_STATE + index))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    remaining = sample_count - len(sampled)

    if remaining > 0:
        unused = df_spoof[~df_spoof["file_id"].isin(sampled["file_id"])]
        if not unused.empty:
            sampled_extra = unused.sample(
                n=min(remaining, len(unused)),
                random_state=RANDOM_STATE + 999,
            )
            sampled = pd.concat([sampled, sampled_extra], ignore_index=True)

    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def sample_split(df: pd.DataFrame, bonafide_count: int, spoof_count: int) -> pd.DataFrame:
    """Create a balanced subset for one split."""

    bonafide = df[df["label"] == "bonafide"]
    spoof = df[df["label"] == "spoof"]

    if bonafide.empty:
        raise ValueError("No bonafide rows found in protocol.")

    if spoof.empty:
        raise ValueError("No spoof rows found in protocol.")

    bonafide_sample = bonafide.sample(
        n=min(bonafide_count, len(bonafide)),
        random_state=RANDOM_STATE,
    )

    if STRATIFY_SPOOF_BY_ATTACK:
        spoof_sample = sample_spoof_by_attack(spoof, min(spoof_count, len(spoof)))
    else:
        spoof_sample = spoof.sample(
            n=min(spoof_count, len(spoof)),
            random_state=RANDOM_STATE,
        )

    sampled = pd.concat([bonafide_sample, spoof_sample], ignore_index=True)

    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def copy_audio_files(sampled: pd.DataFrame, audio_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Copy selected .flac files and return rows that were successfully copied."""

    output_dir.mkdir(parents=True, exist_ok=True)

    copied_rows = []
    missing_files = []

    for row in sampled.to_dict(orient="records"):
        src = audio_dir / str(row["file_name"])
        dst = output_dir / str(row["file_name"])

        if not src.exists():
            missing_files.append(str(src))
            continue

        shutil.copy2(src, dst)
        row["relative_path"] = str(dst.relative_to(OUTPUT_DIR)).replace("\\", "/")
        copied_rows.append(row)

    if missing_files:
        print(f"Warning: {len(missing_files)} selected files were missing.")
        print(f"First missing file: {missing_files[0]}")

    if not copied_rows:
        raise RuntimeError(f"No files were copied from {audio_dir}")

    return pd.DataFrame(copied_rows)


def prepare_split(config: SplitConfig) -> pd.DataFrame:
    """Prepare one train/dev subset."""

    print(f"\nPreparing {config.split} split")
    print(f"Protocol: {config.protocol_path}")
    print(f"Audio dir: {config.audio_dir}")

    df = read_protocol(config.protocol_path)

    sampled = sample_split(
        df,
        bonafide_count=config.bonafide_samples,
        spoof_count=config.spoof_samples,
    )

    copied = copy_audio_files(sampled, config.audio_dir, config.output_dir)
    copied["split"] = config.split

    print(f"Copied {len(copied)} files:")
    print(copied["label"].value_counts().to_string())

    attack_counts = copied[copied["label"] == "spoof"]["attack_id"].value_counts()
    if not attack_counts.empty:
        print("Spoof attack distribution:")
        print(attack_counts.to_string())

    return copied


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        SplitConfig(
            split="train",
            protocol_path=TRAIN_PROTOCOL,
            audio_dir=TRAIN_AUDIO_DIR,
            output_dir=OUTPUT_TRAIN_DIR,
            bonafide_samples=TRAIN_BONAFIDE_SAMPLES,
            spoof_samples=TRAIN_SPOOF_SAMPLES,
        ),
        SplitConfig(
            split="dev",
            protocol_path=DEV_PROTOCOL,
            audio_dir=DEV_AUDIO_DIR,
            output_dir=OUTPUT_DEV_DIR,
            bonafide_samples=DEV_BONAFIDE_SAMPLES,
            spoof_samples=DEV_SPOOF_SAMPLES,
        ),
    ]

    all_rows = []

    for config in configs:
        all_rows.append(prepare_split(config))

    labels = pd.concat(all_rows, ignore_index=True)

    labels = labels[
        [
            "split",
            "file_name",
            "relative_path",
            "label",
            "attack_id",
            "speaker_id",
            "system_id",
        ]
    ]

    labels.to_csv(OUTPUT_LABELS, index=False)

    print("\nASVspoof subset preparation complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Labels saved to: {OUTPUT_LABELS}")
    print("\nFinal counts:")
    print(labels.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
