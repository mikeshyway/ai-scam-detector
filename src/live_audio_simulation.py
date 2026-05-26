"""Chunk-based audio analysis for the live-call simulation page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audio_preprocessor import audio_arrays_from_bytes, spectrogram_db


def chunk_waveform(y: np.ndarray, sr: int, chunk_seconds: int) -> list[tuple[int, float, float, np.ndarray]]:
    chunk_size = max(1, int(sr * chunk_seconds))
    chunks: list[tuple[int, float, float, np.ndarray]] = []
    for index, start in enumerate(range(0, len(y), chunk_size), start=1):
        end = min(start + chunk_size, len(y))
        if end - start < int(sr * 0.75):
            continue
        chunks.append((index, start / sr, end / sr, y[start:end]))
    return chunks


def extract_mfcc_from_array(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError("librosa is required for MFCC chunk extraction.") from exc

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).astype(np.float32)


def demo_chunk_confidence(chunk: np.ndarray) -> float:
    energy = float(np.sqrt(np.mean(np.square(chunk))) if chunk.size else 0.0)
    variation = float(np.std(chunk) if chunk.size else 0.0)
    score = min(0.92, max(0.08, 0.35 + energy * 2.2 + variation * 1.4))
    return score


def predict_chunk_features(model: Any | None, features: np.ndarray, chunk: np.ndarray) -> tuple[str, float, str]:
    if model is None:
        confidence = demo_chunk_confidence(chunk)
        label = "Possible AI-generated / scam-risk audio" if confidence >= 0.55 else "Lower-risk audio segment"
        return label, confidence, "Demo heuristic"

    row = features.reshape(1, -1)
    label_id = int(model.predict(row)[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = [int(item) for item in getattr(model, "classes_", [0, 1])]
        class_index = classes.index(label_id) if label_id in classes else int(label_id)
        confidence = float(probabilities[class_index])
    else:
        confidence = 0.5
    label = "Possible AI-generated speech" if label_id == 1 else "Real human speech"
    return label, confidence, "SVM"


def analyze_audio_chunks(
    audio_bytes: bytes,
    suffix: str,
    *,
    chunk_seconds: int = 5,
    model: Any | None = None,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    y, sr = audio_arrays_from_bytes(audio_bytes, suffix=suffix, max_seconds=None)
    rows = []
    for chunk_id, start, end, chunk in chunk_waveform(y, sr, chunk_seconds):
        features = extract_mfcc_from_array(chunk, sr)
        label, confidence, engine = predict_chunk_features(model, features, chunk)
        rows.append(
            {
                "chunk": chunk_id,
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "prediction": label,
                "confidence": round(confidence * 100, 2),
                "engine": engine,
            }
        )
    return pd.DataFrame(rows), y, sr


def load_optional_audio_model(root: str | Path) -> Any | None:
    model_path = Path(root) / "models" / "audio_svm.pkl"
    if not model_path.exists():
        return None
    try:
        import joblib

        return joblib.load(model_path)
    except Exception:
        return None


__all__ = [
    "analyze_audio_chunks",
    "load_optional_audio_model",
    "spectrogram_db",
]
