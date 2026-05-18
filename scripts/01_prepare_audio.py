"""Pre-extract MFCC features from the ASVspoof subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_preprocessor import balanced_audio_sample, load_audio_labels, prepare_audio_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default=ROOT / "data" / "raw" / "asvspoof_subset")
    parser.add_argument("--labels-csv", default=ROOT / "data" / "raw" / "asvspoof_subset" / "labels.csv")
    parser.add_argument("--filename-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--max-real", type=int, default=300)
    parser.add_argument("--max-fake", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--out-x", default=ROOT / "data" / "processed" / "audio_X.npy")
    parser.add_argument("--out-y", default=ROOT / "data" / "processed" / "audio_y.npy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_audio_labels(
        args.labels_csv,
        filename_column=args.filename_column,
        label_column=args.label_column,
    )
    labels = balanced_audio_sample(labels, max_real=args.max_real, max_fake=args.max_fake)
    X, y, used_files = prepare_audio_features(
        args.audio_dir,
        labels,
        n_mfcc=args.n_mfcc,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
    )

    Path(args.out_x).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_x, X)
    np.save(args.out_y, y)

    print(f"Extracted features: {X.shape}")
    print(f"Labels: {np.bincount(y).tolist()} [real, fake]")
    print(f"Used audio files: {len(used_files)}")
    print(f"Saved {args.out_x}")
    print(f"Saved {args.out_y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
