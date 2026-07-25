"""Trained voice-evidence calibration utilities.

The raw MFCC SVM estimates whether audio resembles spoofed speech. This module
builds a second-stage feature vector that lets a lightweight trained calibrator
estimate how much voice evidence is strong enough to show to users.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


VOICE_EVIDENCE_FEATURE_NAMES = [
    "raw_voice_risk",
    "raw_behavioral_risk",
    "behavioral_available",
    "raw_model_gap",
    "raw_model_mean",
    "duration_seconds",
    "speech_activity_ratio",
    "silence_ratio",
    "estimated_speech_rate",
    "rms_energy",
    "peak_amplitude",
    "rms_energy_std",
    "zero_crossing_rate",
    "zero_crossing_rate_std",
    "spectral_centroid",
    "spectral_centroid_std",
    "spectral_bandwidth",
    "spectral_bandwidth_std",
    "spectral_rolloff",
    "spectral_rolloff_std",
    "pitch_variance",
    "mfcc_dynamics",
    "mfcc_available",
    "usable_speech",
    "quality_warning_count",
    "short_under_3s",
    "short_under_6s",
    "high_silence",
    "sparse_speech",
    "low_energy",
]


@dataclass
class VoiceEvidencePrediction:
    evidence_risk: float
    label_name: str
    confidence: float


class TrainedVoiceEvidenceCalibrator:
    """Wrapper around the saved voice-evidence regressor/classifier."""

    def __init__(self, payload: Any) -> None:
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.feature_names = list(payload.get("feature_names", VOICE_EVIDENCE_FEATURE_NAMES))
            self.metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"model", "feature_names"}
            }
        else:
            self.model = payload
            self.feature_names = VOICE_EVIDENCE_FEATURE_NAMES
            self.metadata = {}

    def predict_one(self, feature_vector: np.ndarray) -> VoiceEvidencePrediction:
        row = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        if row.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Voice evidence feature mismatch: got {row.shape[1]}, "
                f"expected {len(self.feature_names)}."
            )

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(row)[0]
            classes = [int(label) for label in getattr(self.model, "classes_", [0, 1])]
            probability_map = {
                label: float(probabilities[index])
                for index, label in enumerate(classes)
            }
            score = probability_map.get(1, 0.0) * 100.0
        elif hasattr(self.model, "predict"):
            score = float(np.ravel(self.model.predict(row))[0])
        else:
            raise TypeError("Voice evidence calibrator does not provide predict().")

        if 0.0 <= score <= 1.0:
            score *= 100.0
        risk = _bounded_score(score)
        label_name = (
            "Trained AI-voice evidence"
            if risk >= 60.0
            else "Lower trained voice evidence"
        )
        confidence = max(risk, 100.0 - risk) / 100.0
        return VoiceEvidencePrediction(
            evidence_risk=risk,
            label_name=label_name,
            confidence=confidence,
        )


def load_voice_evidence_calibrator(model_path: str | Path) -> TrainedVoiceEvidenceCalibrator:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return TrainedVoiceEvidenceCalibrator(joblib.load(path))


def build_voice_evidence_feature_values(
    *,
    raw_voice_risk: float,
    raw_behavioral_risk: float | None,
    features: dict[str, object],
    behavioral_features: dict[str, object],
    speech_quality: dict[str, object],
) -> dict[str, float]:
    behavioral_available = raw_behavioral_risk is not None
    behavioral_value = float(raw_behavioral_risk) if behavioral_available else 50.0
    duration = _first_number(
        speech_quality.get("duration_seconds"),
        behavioral_features.get("duration_seconds"),
        features.get("duration_seconds"),
    )
    speech_activity = _first_number(
        speech_quality.get("speech_activity_ratio"),
        behavioral_features.get("speech_activity_ratio"),
    )
    silence_ratio = _first_number(
        speech_quality.get("silence_ratio"),
        behavioral_features.get("silence_ratio"),
        1.0,
    )
    rms = _first_number(
        speech_quality.get("rms"),
        features.get("rms_energy"),
        behavioral_features.get("rms_energy_mean"),
    )

    return {
        "raw_voice_risk": _bounded_score(raw_voice_risk),
        "raw_behavioral_risk": _bounded_score(behavioral_value),
        "behavioral_available": float(behavioral_available),
        "raw_model_gap": abs(_bounded_score(raw_voice_risk) - _bounded_score(behavioral_value)),
        "raw_model_mean": (_bounded_score(raw_voice_risk) + _bounded_score(behavioral_value)) / 2.0,
        "duration_seconds": duration,
        "speech_activity_ratio": speech_activity,
        "silence_ratio": silence_ratio,
        "estimated_speech_rate": _first_number(
            speech_quality.get("estimated_speech_rate"),
            behavioral_features.get("estimated_speech_rate"),
        ),
        "rms_energy": rms,
        "peak_amplitude": _first_number(speech_quality.get("peak")),
        "rms_energy_std": _first_number(
            features.get("rms_energy_std"),
            behavioral_features.get("rms_energy_std"),
        ),
        "zero_crossing_rate": _first_number(
            features.get("zero_crossing_rate"),
            behavioral_features.get("zero_crossing_rate_mean"),
        ),
        "zero_crossing_rate_std": _first_number(
            features.get("zero_crossing_rate_std"),
            behavioral_features.get("zero_crossing_rate_std"),
        ),
        "spectral_centroid": _first_number(
            features.get("spectral_centroid"),
            behavioral_features.get("spectral_centroid_mean"),
        ),
        "spectral_centroid_std": _first_number(
            features.get("spectral_centroid_std"),
            behavioral_features.get("spectral_centroid_std"),
        ),
        "spectral_bandwidth": _first_number(
            features.get("spectral_bandwidth"),
            behavioral_features.get("spectral_bandwidth_mean"),
        ),
        "spectral_bandwidth_std": _first_number(features.get("spectral_bandwidth_std")),
        "spectral_rolloff": _first_number(
            features.get("spectral_rolloff"),
            behavioral_features.get("spectral_rolloff_mean"),
        ),
        "spectral_rolloff_std": _first_number(features.get("spectral_rolloff_std")),
        "pitch_variance": _first_number(features.get("pitch_variance")),
        "mfcc_dynamics": _first_number(features.get("mfcc_dynamics")),
        "mfcc_available": float(bool(features.get("mfcc_available", False))),
        "usable_speech": float(bool(speech_quality.get("usable_speech", True))),
        "quality_warning_count": float(len(speech_quality.get("warnings", []))),
        "short_under_3s": float(duration < 3.0),
        "short_under_6s": float(duration < 6.0),
        "high_silence": float(silence_ratio > 0.65),
        "sparse_speech": float(speech_activity < 0.35),
        "low_energy": float(rms < 0.015),
    }


def voice_evidence_feature_vector(feature_values: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [float(feature_values.get(name, 0.0)) for name in VOICE_EVIDENCE_FEATURE_NAMES],
        dtype=np.float32,
    )


def _first_number(*values: object) -> float:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return 0.0


def _bounded_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(100.0, value)))
