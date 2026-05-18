"""Audio feature extraction utilities for synthetic speech detection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .text_preprocessor import label_to_binary


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac"}


def _require_librosa():
    try:
        import librosa

        return librosa
    except Exception as exc:
        raise RuntimeError(
            "librosa is required for audio preprocessing. Install requirements.txt first."
        ) from exc


def load_audio(
    file_path: str | Path,
    *,
    sample_rate: int = 16000,
    max_seconds: float | None = 20.0,
) -> tuple[np.ndarray, int]:
    """Load mono audio for model inference and visualization."""

    librosa = _require_librosa()
    duration = max_seconds if max_seconds and max_seconds > 0 else None
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError(f"No audio samples could be loaded from {file_path}.")
    return y.astype(np.float32), int(sr)


def extract_mfcc(
    file_path: str | Path,
    *,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    max_seconds: float | None = 20.0,
) -> np.ndarray:
    """Extract a fixed-size MFCC feature vector from one audio file."""

    librosa = _require_librosa()
    y, sr = load_audio(file_path, sample_rate=sample_rate, max_seconds=max_seconds)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)


def extract_mfcc_from_bytes(
    data: bytes,
    *,
    suffix: str,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    max_seconds: float | None = 20.0,
) -> np.ndarray:
    """Extract MFCC features from a Streamlit-uploaded audio file."""

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        return extract_mfcc(
            tmp_path,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            max_seconds=max_seconds,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def audio_arrays_from_bytes(
    data: bytes,
    *,
    suffix: str,
    sample_rate: int = 16000,
    max_seconds: float | None = 20.0,
) -> tuple[np.ndarray, int]:
    """Load waveform arrays from uploaded bytes for plotting."""

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        return load_audio(tmp_path, sample_rate=sample_rate, max_seconds=max_seconds)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def spectrogram_db(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spectrogram values in decibels plus time/frequency axes."""

    librosa = _require_librosa()
    stft = librosa.stft(y, n_fft=1024, hop_length=256)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sr, hop_length=256)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    return db, times, freqs


def _infer_audio_columns(df: pd.DataFrame) -> tuple[str, str]:
    lowered = {column.lower().strip(): column for column in df.columns}
    filename_candidates = ["filename", "file", "path", "audio", "utt_id", "utterance_id"]
    label_candidates = ["label", "class", "target", "is_fake", "spoof", "type"]

    filename_col = next((lowered[name] for name in filename_candidates if name in lowered), None)
    label_col = next((lowered[name] for name in label_candidates if name in lowered), None)

    if filename_col is None:
        filename_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]
    return filename_col, label_col


def load_audio_labels(
    labels_csv: str | Path,
    *,
    filename_column: str | None = None,
    label_column: str | None = None,
) -> pd.DataFrame:
    """Load ASVspoof subset labels into filename + binary label columns."""

    df = pd.read_csv(labels_csv)
    inferred_file, inferred_label = _infer_audio_columns(df)
    filename_col = filename_column or inferred_file
    label_col = label_column or inferred_label

    labels = pd.DataFrame(
        {
            "filename": df[filename_col].astype(str),
            "label": df[label_col].map(label_to_binary),
        }
    )
    labels = labels.dropna()
    if labels.empty:
        raise ValueError(f"No usable audio labels found in {labels_csv}.")
    return labels


def balanced_audio_sample(
    labels: pd.DataFrame,
    *,
    max_real: int | None = 300,
    max_fake: int | None = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Subsample real and fake rows to keep training practical on a laptop."""

    real = labels[labels["label"] == 0]
    fake = labels[labels["label"] == 1]
    if real.empty or fake.empty:
        raise ValueError("Audio training requires both real and synthetic labels.")

    if max_real:
        real = real.sample(n=min(max_real, len(real)), random_state=random_state)
    if max_fake:
        fake = fake.sample(n=min(max_fake, len(fake)), random_state=random_state)

    return (
        pd.concat([real, fake], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def prepare_audio_features(
    audio_dir: str | Path,
    labels: pd.DataFrame,
    *,
    n_mfcc: int = 40,
    sample_rate: int = 16000,
    max_seconds: float | None = 20.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract MFCC features for every labeled audio file that exists."""

    root = Path(audio_dir)
    features: list[np.ndarray] = []
    y: list[int] = []
    used_files: list[str] = []

    for row in labels.itertuples(index=False):
        filename = str(row.filename)
        path = root / filename
        if not path.exists() and path.suffix == "":
            for ext in SUPPORTED_AUDIO_EXTENSIONS:
                candidate = root / f"{filename}{ext}"
                if candidate.exists():
                    path = candidate
                    break

        if not path.exists() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue

        try:
            features.append(
                extract_mfcc(
                    path,
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    max_seconds=max_seconds,
                )
            )
            y.append(int(row.label))
            used_files.append(str(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    if not features:
        raise ValueError(f"No audio features extracted from {root}.")
    return np.vstack(features), np.asarray(y, dtype=int), used_files

