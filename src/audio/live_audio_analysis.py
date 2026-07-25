"""Near-real-time audio analysis shared by local capture and upload flows."""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

from src.audio.voice_evidence_calibrator import (
    build_voice_evidence_feature_values,
    voice_evidence_feature_vector,
)
from src.text.explainability import find_suspicious_phrases
from src.text.rule_demo import rule_based_text_prediction
from src.utils.time_utils import now_for_app


TARGET_SAMPLE_RATE = 16_000
N_MFCC = 40
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

MIN_USABLE_SECONDS = 0.75
MIN_USABLE_PEAK = 0.002
MIN_USABLE_RMS = 0.0008
MIN_SPEECH_RMS = 0.003
MIN_SPEECH_ACTIVITY_RATIO = 0.08

BEHAVIORAL_FEATURE_NAMES = [
    "duration_seconds",
    "rms_energy_mean",
    "rms_energy_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "silence_ratio",
    "speech_activity_ratio",
    "pause_count",
    "estimated_speech_rate",
]


def wav_bytes_to_audio(
    data: bytes,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Decode the PCM WAV returned by Streamlit's voice recorder."""

    with wave.open(io.BytesIO(data), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width * 8}-bit.")

    samples = np.frombuffer(raw, dtype=dtype_map[sample_width])
    if sample_width == 1:
        audio = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        info = np.iinfo(dtype_map[sample_width])
        audio = samples.astype(np.float32) / float(max(abs(info.min), info.max))

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if audio.size == 0:
        raise ValueError("The recording contained no audio samples.")

    if source_rate != target_sample_rate and audio.size > 1:
        target_size = max(1, int(round(audio.size * target_sample_rate / source_rate)))
        source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)
        source_rate = target_sample_rate

    return np.clip(audio, -1.0, 1.0).astype(np.float32), int(source_rate)


def _spectrum_summary(
    audio: np.ndarray,
    sample_rate: int,
    *,
    points: int = 160,
) -> tuple[list[float], list[float], float]:
    windowed = np.asarray(audio, dtype=np.float32) * np.hanning(audio.size)
    magnitudes = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(windowed.size, d=1.0 / sample_rate)
    if magnitudes.size == 0 or float(np.max(magnitudes)) <= 0:
        return [], [], 0.0

    decibels = 20.0 * np.log10(np.maximum(magnitudes, 1e-12) / np.max(magnitudes))
    max_frequency = min(sample_rate / 2.0, 8_000.0)
    usable = frequencies <= max_frequency
    frequencies = frequencies[usable]
    decibels = decibels[usable]
    dominant_frequency = float(frequencies[int(np.argmax(decibels))]) if frequencies.size else 0.0

    if frequencies.size > points:
        indices = np.linspace(0, frequencies.size - 1, num=points, dtype=int)
        frequencies = frequencies[indices]
        decibels = decibels[indices]
    return (
        np.round(frequencies, 2).astype(float).tolist(),
        np.round(decibels, 2).astype(float).tolist(),
        dominant_frequency,
    )


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _count_pauses(silent_frames: np.ndarray, frame_seconds: float, min_pause_seconds: float = 0.25) -> int:
    silent_frames = np.asarray(silent_frames, dtype=bool)
    if silent_frames.size == 0:
        return 0

    pause_count = 0
    run_length = 0
    min_frames = max(1, int(round(min_pause_seconds / max(frame_seconds, 1e-6))))

    for is_silent in silent_frames:
        if is_silent:
            run_length += 1
            continue
        if run_length >= min_frames:
            pause_count += 1
        run_length = 0

    if run_length >= min_frames:
        pause_count += 1

    return pause_count


def _basic_behavioral_features(audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    duration_seconds = float(audio.size / max(sample_rate, 1))
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.signbit(audio))).astype(np.float32))) if audio.size > 1 else 0.0
    values = {
        "duration_seconds": duration_seconds,
        "rms_energy_mean": rms,
        "rms_energy_std": 0.0,
        "zero_crossing_rate_mean": zcr,
        "zero_crossing_rate_std": 0.0,
        "spectral_centroid_mean": 0.0,
        "spectral_centroid_std": 0.0,
        "spectral_bandwidth_mean": 0.0,
        "spectral_rolloff_mean": 0.0,
        "silence_ratio": 1.0 if rms < 0.003 else 0.0,
        "speech_activity_ratio": 0.0 if rms < 0.003 else 1.0,
        "pause_count": 0.0,
        "estimated_speech_rate": 0.0,
    }
    values["feature_names"] = BEHAVIORAL_FEATURE_NAMES
    values["feature_vector"] = np.asarray(
        [float(values[name]) for name in BEHAVIORAL_FEATURE_NAMES],
        dtype=np.float32,
    )
    return values


def _basic_live_features(
    audio: np.ndarray,
    sample_rate: int,
    spectrum_frequencies: list[float],
    spectrum_db: list[float],
    dominant_frequency: float,
) -> dict[str, object]:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.signbit(audio))).astype(np.float32))) if audio.size > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    spectrum_sum = float(np.sum(spectrum))
    centroid = float(np.sum(frequencies * spectrum) / spectrum_sum) if spectrum_sum else 0.0
    return {
        "feature_vector": np.zeros((N_MFCC * 6) + 11, dtype=np.float32),
        "mfcc_mean": np.zeros(N_MFCC, dtype=float).tolist(),
        "mfcc_dynamics": 20.0,
        "spectral_centroid": centroid,
        "spectral_centroid_std": 0.0,
        "spectral_bandwidth": 0.0,
        "spectral_bandwidth_std": 0.0,
        "zero_crossing_rate": zcr,
        "zero_crossing_rate_std": 0.0,
        "rms_energy": rms,
        "rms_energy_std": 0.0,
        "pitch_variance": 20.0,
        "spectral_rolloff": 0.0,
        "spectral_rolloff_std": 0.0,
        "duration_seconds": float(audio.size / max(sample_rate, 1)),
        "mfcc_available": False,
        "spectrum_frequencies": spectrum_frequencies,
        "spectrum_db": spectrum_db,
        "dominant_frequency": dominant_frequency,
    }


def extract_behavioral_features(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict[str, object]:
    """Extract local speech-behavior metadata used by the optional RF model."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        audio = np.zeros(sample_rate, dtype=np.float32)

    duration_seconds = float(audio.size / max(sample_rate, 1))

    try:
        import librosa
    except Exception:
        return _basic_behavioral_features(audio, sample_rate)

    try:
        frame_length = min(2048, max(512, int(sample_rate * 0.08)))
        hop_length = max(128, frame_length // 4)

        rms_values = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        zcr_values = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        centroid_values = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=hop_length)[0]
        bandwidth_values = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, hop_length=hop_length)[0]
        rolloff_values = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, hop_length=hop_length)[0]

        rms_mean, rms_std = _mean_std(rms_values)
        zcr_mean, zcr_std = _mean_std(zcr_values)
        centroid_mean, centroid_std = _mean_std(centroid_values)
        bandwidth_mean, _bandwidth_std = _mean_std(bandwidth_values)
        rolloff_mean, _rolloff_std = _mean_std(rolloff_values)

        if rms_values.size:
            silence_threshold = max(0.004, float(np.percentile(rms_values, 20)) * 1.8)
            silent_frames = rms_values <= silence_threshold
            silence_ratio = float(np.mean(silent_frames))
        else:
            silent_frames = np.asarray([], dtype=bool)
            silence_ratio = 1.0

        speech_activity_ratio = float(max(0.0, min(1.0, 1.0 - silence_ratio)))
        frame_seconds = hop_length / max(sample_rate, 1)
        pause_count = float(_count_pauses(silent_frames, frame_seconds))

        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length)
            estimated_speech_rate = float(len(onset_frames) / max(duration_seconds, 1e-6))
        except Exception:
            estimated_speech_rate = 0.0
    except Exception:
        return _basic_behavioral_features(audio, sample_rate)

    values = {
        "duration_seconds": duration_seconds,
        "rms_energy_mean": rms_mean,
        "rms_energy_std": rms_std,
        "zero_crossing_rate_mean": zcr_mean,
        "zero_crossing_rate_std": zcr_std,
        "spectral_centroid_mean": centroid_mean,
        "spectral_centroid_std": centroid_std,
        "spectral_bandwidth_mean": bandwidth_mean,
        "spectral_rolloff_mean": rolloff_mean,
        "silence_ratio": silence_ratio,
        "speech_activity_ratio": speech_activity_ratio,
        "pause_count": pause_count,
        "estimated_speech_rate": estimated_speech_rate,
    }
    values["feature_names"] = BEHAVIORAL_FEATURE_NAMES
    values["feature_vector"] = np.asarray(
        [float(values[name]) for name in BEHAVIORAL_FEATURE_NAMES],
        dtype=np.float32,
    )
    return values


def extract_live_features(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, object]:
    """Extract MFCC and lightweight acoustic indicators from one chunk."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < 512:
        raise ValueError("Audio chunk is too short for feature extraction.")
    spectrum_frequencies, spectrum_db, dominant_frequency = _spectrum_summary(audio, sample_rate)

    try:
        import librosa
    except Exception:
        return _basic_live_features(audio, sample_rate, spectrum_frequencies, spectrum_db, dominant_frequency)

    try:
        mfcc_full = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc_full)
        delta2 = librosa.feature.delta(mfcc_full, order=2)
        mfcc_mean = np.mean(mfcc_full, axis=1)
        mfcc_std = np.std(mfcc_full, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        zero_crossing = librosa.feature.zero_crossing_rate(audio)
        rms_values = librosa.feature.rms(y=audio)

        centroid = float(np.mean(spectral_centroid))
        centroid_std = float(np.std(spectral_centroid))
        bandwidth = float(np.mean(spectral_bandwidth))
        bandwidth_std = float(np.std(spectral_bandwidth))
        zcr = float(np.mean(zero_crossing))
        zcr_std = float(np.std(zero_crossing))
        rms = float(np.mean(rms_values))
        rms_std = float(np.std(rms_values))
        rolloff = float(np.mean(spectral_rolloff))
        rolloff_std = float(np.std(spectral_rolloff))
        duration_seconds = float(audio.size / max(sample_rate, 1))

        feature_vector = np.concatenate(
            [
                mfcc_mean,
                mfcc_std,
                delta.mean(axis=1),
                delta.std(axis=1),
                delta2.mean(axis=1),
                delta2.std(axis=1),
                [
                    centroid,
                    centroid_std,
                    bandwidth,
                    bandwidth_std,
                    rolloff,
                    rolloff_std,
                    zcr,
                    zcr_std,
                    rms,
                    rms_std,
                    duration_seconds,
                ],
            ]
        ).astype(np.float32)

        pitch_variance = 0.0
        try:
            pitches = librosa.yin(audio, fmin=55, fmax=400, sr=sample_rate)
            finite = pitches[np.isfinite(pitches)]
            if finite.size:
                pitch_variance = float(np.std(finite))
        except Exception:
            pitch_variance = 0.0
    except Exception:
        return _basic_live_features(audio, sample_rate, spectrum_frequencies, spectrum_db, dominant_frequency)

    return {
        "feature_vector": feature_vector,
        "mfcc_mean": mfcc_mean.astype(float).tolist(),
        "mfcc_dynamics": float(np.mean(mfcc_std)),
        "spectral_centroid": centroid,
        "spectral_centroid_std": centroid_std,
        "spectral_bandwidth": bandwidth,
        "spectral_bandwidth_std": bandwidth_std,
        "zero_crossing_rate": zcr,
        "zero_crossing_rate_std": zcr_std,
        "rms_energy": rms,
        "rms_energy_std": rms_std,
        "pitch_variance": pitch_variance,
        "spectral_rolloff": rolloff,
        "spectral_rolloff_std": rolloff_std,
        "duration_seconds": duration_seconds,
        "mfcc_available": True,
        "spectrum_frequencies": spectrum_frequencies,
        "spectrum_db": spectrum_db,
        "dominant_frequency": dominant_frequency,
    }


def assess_speech_quality(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    features: dict[str, object] | None = None,
    behavioral_features: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return a lightweight gate for chunks that should not be audio-SVM scored."""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    sample_rate = max(1, int(sample_rate))
    duration_seconds = float(samples.size / sample_rate)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        finite = np.zeros(0, dtype=np.float32)

    rms = (
        float(features.get("rms_energy", 0.0))
        if isinstance(features, dict) and "rms_energy" in features
        else float(np.sqrt(np.mean(np.square(finite)))) if finite.size else 0.0
    )
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    speech_activity = (
        float(behavioral_features.get("speech_activity_ratio", 0.0))
        if isinstance(behavioral_features, dict)
        else 0.0
    )
    silence_ratio = (
        float(behavioral_features.get("silence_ratio", 1.0))
        if isinstance(behavioral_features, dict)
        else 1.0
    )
    estimated_speech_rate = (
        float(behavioral_features.get("estimated_speech_rate", 0.0))
        if isinstance(behavioral_features, dict)
        else 0.0
    )
    zcr = (
        float(features.get("zero_crossing_rate", 0.0))
        if isinstance(features, dict)
        else 0.0
    )
    centroid = (
        float(features.get("spectral_centroid", 0.0))
        if isinstance(features, dict)
        else 0.0
    )
    bandwidth = (
        float(features.get("spectral_bandwidth", 0.0))
        if isinstance(features, dict)
        else 0.0
    )
    mfcc_available = bool(features.get("mfcc_available", False)) if isinstance(features, dict) else False
    has_behavioral_features = isinstance(behavioral_features, dict)

    usable = True
    reason = "Usable speech-like audio"
    warnings: list[str] = []

    if duration_seconds < MIN_USABLE_SECONDS:
        usable = False
        reason = "Audio chunk is too short for reliable speech analysis"
    elif peak < MIN_USABLE_PEAK or rms < MIN_USABLE_RMS:
        usable = False
        reason = "Audio is too quiet to score reliably"
    elif has_behavioral_features and rms < MIN_SPEECH_RMS and (
        silence_ratio >= 0.85 or speech_activity <= MIN_SPEECH_ACTIVITY_RATIO
    ):
        usable = False
        reason = "No usable speech detected"
    elif has_behavioral_features and speech_activity < 0.05 and silence_ratio > 0.90:
        usable = False
        reason = "Mostly silence or background room tone"
    elif mfcc_available and zcr > 0.32 and centroid > 3_000:
        usable = False
        reason = "Broadband noise detected instead of speech"
    elif mfcc_available and bandwidth < 80 and estimated_speech_rate < 0.2:
        usable = False
        reason = "Tonal or non-speech audio detected"

    if usable and rms < MIN_SPEECH_RMS:
        warnings.append("Low speech energy; audio model confidence may be weak.")
    if usable and speech_activity < 0.20:
        warnings.append("Limited speech activity; verify the transcript before acting.")

    return {
        "usable_speech": usable,
        "reason": reason,
        "warnings": warnings,
        "duration_seconds": round(duration_seconds, 3),
        "rms": rms,
        "peak": peak,
        "speech_activity_ratio": speech_activity,
        "silence_ratio": silence_ratio,
        "estimated_speech_rate": estimated_speech_rate,
    }


def _heuristic_audio_risk(features: dict[str, object]) -> tuple[float, str]:
    """Educational fallback when the trained audio SVM is unavailable."""

    rms = float(features.get("rms_energy", 0.0))
    if rms < 0.003:
        return 5.0, "Insufficient speech energy"

    score = 18.0
    pitch_variance = float(features.get("pitch_variance", 0.0))
    mfcc_dynamics = float(features.get("mfcc_dynamics", 0.0))
    zcr = float(features.get("zero_crossing_rate", 0.0))

    if pitch_variance < 8:
        score += 28
    elif pitch_variance < 16:
        score += 16

    if mfcc_dynamics < 8:
        score += 22
    elif mfcc_dynamics < 14:
        score += 10

    if zcr < 0.025:
        score += 14
    elif zcr > 0.24:
        score += 8

    return min(92.0, score), "Educational acoustic heuristic"


def score_audio_chunk(
    features: dict[str, object],
    audio_classifier: Any | None,
) -> tuple[float, str, str]:
    """Return AI-voice risk percentage, prediction label, and engine name."""

    speech_quality = features.get("speech_quality")
    if isinstance(speech_quality, dict) and not bool(
        speech_quality.get("usable_speech", True)
    ):
        return 0.0, str(speech_quality.get("reason", "No usable speech detected")), "Audio quality gate"

    if audio_classifier is None or not bool(features.get("mfcc_available", False)):
        risk, engine = _heuristic_audio_risk(features)
        if not bool(features.get("mfcc_available", False)):
            engine = "Basic acoustic fallback"
        label = "Possible AI-generated speech" if risk >= 60 else "Lower-risk voice characteristics"
        return risk, label, engine

    try:
        prediction = audio_classifier.predict_one(np.asarray(features["feature_vector"], dtype=np.float32))
    except Exception:
        risk, engine = _heuristic_audio_risk(features)
        return risk, "Lower-risk voice characteristics" if risk < 60 else "Possible AI-generated speech", (
            "Acoustic fallback; retrain audio_svm.pkl for the current feature shape"
        )

    risk = float(prediction.probabilities.get("Possible AI-generated speech", 0.0)) * 100
    return risk, prediction.label_name, "MFCC + SVM"


def score_behavioral_chunk(
    behavior_features: dict[str, object],
    behavioral_classifier: Any | None,
) -> tuple[float | None, str, str]:
    """Return optional behavioral metadata risk percentage."""

    if behavioral_classifier is None:
        return None, "Behavioral model unavailable", "Behavioral model unavailable"

    try:
        feature_vector = np.asarray(behavior_features["feature_vector"], dtype=np.float32)
        prediction = behavioral_classifier.predict_one(feature_vector)
    except Exception:
        return None, "Behavioral model unavailable", "Behavioral model unavailable"

    risk = float(prediction.probabilities.get("Possible AI-generated speech", 0.0)) * 100
    return risk, prediction.label_name, "Behavioral RF"


def score_voice_evidence_chunk(
    *,
    raw_voice_risk: float,
    raw_behavioral_risk: float | None,
    features: dict[str, object],
    behavioral_features: dict[str, object],
    speech_quality: dict[str, object],
    voice_evidence_calibrator: Any | None,
) -> tuple[float | None, str, str, dict[str, float]]:
    """Return trained voice-evidence risk when the second-stage calibrator exists."""

    feature_values = build_voice_evidence_feature_values(
        raw_voice_risk=raw_voice_risk,
        raw_behavioral_risk=raw_behavioral_risk,
        features=features,
        behavioral_features=behavioral_features,
        speech_quality=speech_quality,
    )
    if voice_evidence_calibrator is None:
        return None, "Rule reliability weighting", "Rule reliability weighting", feature_values

    try:
        vector = voice_evidence_feature_vector(feature_values)
        prediction = voice_evidence_calibrator.predict_one(vector)
    except Exception:
        return None, "Rule reliability weighting", "Rule reliability weighting", feature_values

    return (
        float(prediction.evidence_risk),
        str(prediction.label_name),
        "Trained voice evidence calibrator",
        feature_values,
    )


def score_transcript(
    transcript: str,
    text_classifier: Any | None,
) -> tuple[float, str, str, list[dict[str, object]]]:
    """Return scam-language risk percentage and explainable findings."""

    transcript = transcript.strip()
    if not transcript:
        return 0.0, "No transcript", "Audio-only analysis", []

    findings = find_suspicious_phrases(transcript)
    if text_classifier is None:
        result = rule_based_text_prediction(transcript)
        risk = float(result["probabilities"]["Suspicious"]) * 100
        return risk, str(result["label_name"]), str(result["model_name"]), findings

    prediction = text_classifier.predict_one(transcript)
    risk = float(prediction.probabilities.get("Suspicious", 0.0)) * 100
    return risk, prediction.label_name, prediction.model_name, findings


def combined_risk(
    *,
    voice_risk: float,
    transcript_risk: float,
    has_transcript: bool,
    behavioral_risk: float | None = None,
) -> float:
    """Average the available voice, semantic, and behavioral risk signals."""

    scores = [float(voice_risk)]
    if has_transcript:
        scores.append(float(transcript_risk))
    if behavioral_risk is not None:
        scores.append(float(behavioral_risk))

    score = sum(scores) / max(1, len(scores))
    return max(0.0, min(100.0, score))


def _bounded_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def _content_reliability_score(
    transcript: str,
    transcript_risk: float,
    findings: list[dict[str, object]],
    text_engine: str,
) -> float:
    words = _word_count(transcript)
    if words == 0:
        return 0.0

    word_factor = _clamp_ratio(words / 35.0)
    rule_factor = _clamp_ratio(len(findings) / 3.0)
    certainty_factor = _clamp_ratio(abs(float(transcript_risk) - 50.0) / 50.0)
    engine_factor = 0.75 if "rule" in text_engine.casefold() else 1.0

    score = (
        (word_factor * 45.0)
        + (rule_factor * 25.0)
        + (certainty_factor * 15.0)
        + (engine_factor * 15.0)
    )
    return _bounded_score(score)


def _content_level(
    transcript: str,
    transcript_risk: float,
    findings: list[dict[str, object]],
    text_engine: str,
) -> str:
    if not transcript.strip():
        return "Inconclusive"
    reliability = _content_reliability_score(
        transcript,
        transcript_risk,
        findings,
        text_engine,
    )
    if (transcript_risk >= 70 and reliability >= 45) or len(findings) >= 3:
        return "High scam-content concern"
    if findings:
        return "Needs scam-content review"
    if transcript_risk >= 55 and reliability >= 35:
        return "Needs scam-content review"
    if reliability < 35:
        return "Limited scam-content evidence"
    return "Lower scam-content concern"


def _authenticity_score(voice_risk: float, behavioral_risk: float | None) -> float:
    if behavioral_risk is None:
        return _bounded_score(voice_risk)
    return _bounded_score((float(voice_risk) * 0.7) + (float(behavioral_risk) * 0.3))


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _speech_quality_notes(speech_quality: dict[str, object]) -> list[str]:
    notes: list[str] = []
    duration = float(speech_quality.get("duration_seconds", 0.0))
    speech_activity = float(speech_quality.get("speech_activity_ratio", 0.0))
    silence_ratio = float(speech_quality.get("silence_ratio", 1.0))
    rms = _float_or_none(speech_quality.get("rms"))

    if duration < 3.0:
        notes.append("clip is under 3 seconds")
    elif duration < 6.0:
        notes.append("clip is shorter than the preferred voice-authenticity window")
    if silence_ratio > 0.65:
        notes.append("silence ratio is high")
    if speech_activity < 0.35:
        notes.append("speech activity is sparse")
    if rms is not None and rms < 0.015:
        notes.append("recording energy is below the reliable comparison range")

    return notes


def _voice_reliability_score(
    speech_quality: dict[str, object],
    voice_risk: float,
    behavioral_risk: float | None,
    audio_engine: str,
) -> float:
    if not bool(speech_quality.get("usable_speech", True)):
        return 0.0

    duration = float(speech_quality.get("duration_seconds", 0.0))
    speech_activity = float(speech_quality.get("speech_activity_ratio", 0.0))
    silence_ratio = float(speech_quality.get("silence_ratio", 1.0))
    speech_density = max(speech_activity, 1.0 - silence_ratio)
    speech_rate = float(speech_quality.get("estimated_speech_rate", 0.0))
    rms = _float_or_none(speech_quality.get("rms"))
    peak = _float_or_none(speech_quality.get("peak"))

    if duration >= 20.0:
        duration_points = 35.0
    elif duration >= 10.0:
        duration_points = 28.0
    elif duration >= 6.0:
        duration_points = 22.0
    elif duration >= 3.0:
        duration_points = 16.0
    elif duration >= 1.5:
        duration_points = 8.0
    else:
        duration_points = 0.0

    density_points = _clamp_ratio((speech_density - 0.08) / 0.42) * 25.0
    if 1.0 <= speech_rate <= 8.0:
        rate_points = 10.0
    elif speech_rate > 0.0:
        rate_points = 5.0
    else:
        rate_points = 0.0

    if behavioral_risk is None:
        agreement_points = 6.0
    else:
        agreement_points = _clamp_ratio(1.0 - abs(float(voice_risk) - float(behavioral_risk)) / 100.0) * 15.0

    engine_points = 10.0 if "MFCC + SVM" in audio_engine else 6.0
    warning_penalty = min(10.0, 5.0 * len(speech_quality.get("warnings", [])))
    evidence_penalty = 0.0
    if duration < 3.0:
        evidence_penalty += 10.0
    elif duration < 6.0:
        evidence_penalty += 4.0
    if silence_ratio > 0.65:
        evidence_penalty += 15.0
    elif silence_ratio > 0.55:
        evidence_penalty += 6.0
    if speech_activity < 0.35:
        evidence_penalty += 12.0
    elif speech_activity < 0.45:
        evidence_penalty += 5.0
    if rms is not None:
        if rms < 0.006:
            evidence_penalty += 18.0
        elif rms < 0.015:
            evidence_penalty += 12.0
        elif rms < 0.025:
            evidence_penalty += 5.0
    if peak is not None and peak < 0.04:
        evidence_penalty += 8.0

    return _bounded_score(
        duration_points
        + density_points
        + rate_points
        + agreement_points
        + engine_points
        + 5.0
        - warning_penalty
        - evidence_penalty
    )


def _behavioral_reliability_score(
    speech_quality: dict[str, object],
    behavioral_features: dict[str, object] | None,
    behavioral_risk: float | None,
    behavioral_engine: str,
) -> float:
    if behavioral_risk is None or "unavailable" in behavioral_engine.casefold():
        return 0.0
    if not bool(speech_quality.get("usable_speech", True)):
        return 0.0

    feature_source = behavioral_features if isinstance(behavioral_features, dict) else {}
    duration = float(speech_quality.get("duration_seconds", feature_source.get("duration_seconds", 0.0)))
    speech_activity = float(speech_quality.get("speech_activity_ratio", feature_source.get("speech_activity_ratio", 0.0)))
    silence_ratio = float(speech_quality.get("silence_ratio", feature_source.get("silence_ratio", 1.0)))
    speech_density = max(speech_activity, 1.0 - silence_ratio)
    speech_rate = float(speech_quality.get("estimated_speech_rate", feature_source.get("estimated_speech_rate", 0.0)))
    rms = _float_or_none(speech_quality.get("rms"))
    if rms is None:
        rms = _float_or_none(feature_source.get("rms_energy_mean"))

    if duration >= 20.0:
        duration_points = 25.0
    elif duration >= 10.0:
        duration_points = 21.0
    elif duration >= 6.0:
        duration_points = 16.0
    elif duration >= 3.0:
        duration_points = 12.0
    elif duration >= 1.5:
        duration_points = 8.0
    else:
        duration_points = 0.0

    density_points = _clamp_ratio((speech_density - 0.30) / 0.35) * 25.0
    if 1.5 <= speech_rate <= 7.0:
        rate_points = 10.0
    elif 0.8 <= speech_rate <= 9.0:
        rate_points = 6.0
    elif speech_rate > 0.0:
        rate_points = 3.0
    else:
        rate_points = 0.0

    if rms is None:
        energy_points = 8.0
    elif rms >= 0.030:
        energy_points = 15.0
    elif rms >= 0.020:
        energy_points = 10.0
    elif rms >= 0.012:
        energy_points = 5.0
    else:
        energy_points = 0.0

    engine_points = 10.0 if "Behavioral RF" in behavioral_engine else 5.0
    evidence_penalty = 0.0
    if duration < 3.0:
        evidence_penalty += 8.0
    if silence_ratio > 0.65:
        evidence_penalty += 18.0
    elif silence_ratio > 0.55:
        evidence_penalty += 8.0
    if speech_activity < 0.35:
        evidence_penalty += 15.0
    elif speech_activity < 0.45:
        evidence_penalty += 6.0
    if rms is not None and rms < 0.015:
        evidence_penalty += 12.0

    return _bounded_score(
        duration_points
        + density_points
        + rate_points
        + energy_points
        + engine_points
        + 5.0
        - evidence_penalty
    )


def _weighted_authenticity_evidence(
    voice_evidence_risk: float,
    behavioral_evidence_risk: float | None,
) -> float:
    if behavioral_evidence_risk is None:
        return _bounded_score(voice_evidence_risk)
    return _bounded_score((float(voice_evidence_risk) * 0.75) + (float(behavioral_evidence_risk) * 0.25))


def _authenticity_level(
    voice_risk: float,
    behavioral_risk: float | None,
    speech_quality: dict[str, object],
    audio_engine: str,
    behavioral_features: dict[str, object] | None = None,
    behavioral_engine: str = "",
    trained_voice_evidence_risk: float | None = None,
) -> str:
    if not bool(speech_quality.get("usable_speech", True)):
        return "Inconclusive"

    raw_score = _authenticity_score(voice_risk, behavioral_risk)
    voice_reliability = _voice_reliability_score(
        speech_quality,
        voice_risk,
        behavioral_risk,
        audio_engine,
    )
    behavioral_reliability = _behavioral_reliability_score(
        speech_quality,
        behavioral_features,
        behavioral_risk,
        behavioral_engine,
    )
    voice_evidence_risk = _bounded_score(voice_risk * (voice_reliability / 100.0))
    behavioral_evidence_risk = (
        _bounded_score(float(behavioral_risk) * (behavioral_reliability / 100.0))
        if behavioral_risk is not None
        else None
    )
    if trained_voice_evidence_risk is not None:
        voice_evidence_risk = _bounded_score(float(trained_voice_evidence_risk))
    effective_score = _weighted_authenticity_evidence(
        voice_evidence_risk,
        behavioral_evidence_risk,
    )
    has_trained_voice_evidence = trained_voice_evidence_risk is not None

    if voice_reliability < 35 and behavioral_reliability < 35 and voice_evidence_risk < 35:
        if raw_score >= 85:
            return "Weak voice-authenticity evidence"
        return "Inconclusive voice-authenticity evidence"
    if raw_score >= 90 and effective_score >= 70 and (
        has_trained_voice_evidence or voice_reliability >= 70
    ) and (
        behavioral_risk is None or behavioral_reliability >= 55
    ):
        return "High voice-authenticity concern"
    if raw_score >= 70 and effective_score >= 40 and (
        has_trained_voice_evidence or max(voice_reliability, behavioral_reliability) >= 45
    ):
        return "Needs voice-authenticity review"
    if raw_score >= 85:
        return "Weak voice-authenticity evidence"
    return "Lower voice-authenticity concern"


def decision_layer(
    *,
    transcript: str,
    voice_risk: float,
    transcript_risk: float,
    behavioral_risk: float | None,
    speech_quality: dict[str, object],
    findings: list[dict[str, object]],
    audio_engine: str,
    text_engine: str,
    behavioral_features: dict[str, object] | None = None,
    behavioral_engine: str = "",
    trained_voice_evidence_risk: float | None = None,
    voice_evidence_engine: str = "Rule reliability weighting",
) -> dict[str, object]:
    """Convert model scores into reliability-weighted evidence decisions."""

    has_transcript = bool(transcript.strip())
    content_reliability = _content_reliability_score(
        transcript,
        transcript_risk,
        findings,
        text_engine,
    )
    content_level = _content_level(
        transcript,
        transcript_risk,
        findings,
        text_engine,
    )
    authenticity_score = _authenticity_score(voice_risk, behavioral_risk)
    voice_reliability = _voice_reliability_score(
        speech_quality,
        voice_risk,
        behavioral_risk,
        audio_engine,
    )
    behavioral_reliability = _behavioral_reliability_score(
        speech_quality,
        behavioral_features,
        behavioral_risk,
        behavioral_engine,
    )
    authenticity_level = _authenticity_level(
        voice_risk,
        behavioral_risk,
        speech_quality,
        audio_engine,
        behavioral_features,
        behavioral_engine,
        trained_voice_evidence_risk,
    )
    usable_speech = bool(speech_quality.get("usable_speech", True))
    effective_content_risk = _bounded_score(transcript_risk * (content_reliability / 100.0))
    rule_voice_evidence_risk = _bounded_score(voice_risk * (voice_reliability / 100.0))
    voice_evidence_risk = (
        _bounded_score(float(trained_voice_evidence_risk))
        if trained_voice_evidence_risk is not None
        else rule_voice_evidence_risk
    )
    behavioral_evidence_risk = (
        _bounded_score(float(behavioral_risk) * (behavioral_reliability / 100.0))
        if behavioral_risk is not None
        else None
    )
    effective_authenticity_risk = _weighted_authenticity_evidence(
        voice_evidence_risk,
        behavioral_evidence_risk,
    )

    if not usable_speech:
        decision_score = 0.0
        label = "Inconclusive audio"
        summary = "Audio quality was not sufficient for a voice-authenticity or transcript-content verdict."
        action = "Use a clearer recording or a written transcript before acting."
    elif not has_transcript:
        if authenticity_level == "High voice-authenticity concern":
            decision_score = min(65.0, max(50.0, effective_authenticity_risk * 0.65))
            label = "Authenticity concern - transcript unavailable"
            summary = "Voice evidence is strong enough for authenticity review, but scam intent cannot be judged without transcript content."
            action = "Verify through an official channel before trusting the caller."
        elif authenticity_level == "Needs voice-authenticity review":
            decision_score = min(52.0, max(38.0, effective_authenticity_risk * 0.62))
            label = "Voice authenticity review - transcript unavailable"
            summary = "Voice evidence needs review, but there is no transcript evidence of threat, payment, or credential pressure."
            action = "Review the transcript or request written confirmation before acting."
        elif authenticity_level in {"Weak voice-authenticity evidence", "Inconclusive voice-authenticity evidence"}:
            decision_score = min(35.0, max(20.0, effective_authenticity_risk * 0.55))
            label = "Audio evidence limited - transcript unavailable"
            summary = "The raw voice score is not reliable enough by itself because the available speech evidence is limited."
            action = "Use a longer recording or transcript before making a deepfake judgement."
        else:
            decision_score = 20.0
            label = "Audio-only lower concern"
            summary = "No transcript was available, and the voice-authenticity signal did not reach review level."
            action = "Continue cautious review if the request was unexpected."
    elif content_level == "High scam-content concern":
        if authenticity_level == "High voice-authenticity concern":
            decision_score = 90.0
            label = "High scam and voice-authenticity concern"
            summary = "Transcript content shows scam pressure, and the voice signal also raises synthetic-voice concern."
        else:
            decision_score = 82.0
            label = "High scam-content concern"
            summary = "Transcript content shows strong scam indicators; voice authenticity is supporting context, not the main reason."
        action = "Pause, do not share secrets or money, and verify through an official channel."
    elif content_level == "Needs scam-content review":
        if authenticity_level == "High voice-authenticity concern":
            decision_score = 68.0
            label = "Scam-content and authenticity review"
            summary = "Some transcript indicators need review and the voice also raises authenticity concern."
        elif authenticity_level == "Needs voice-authenticity review":
            decision_score = 58.0
            label = "Scam-content and voice review"
            summary = "Transcript content and voice authenticity both need review, but neither is strong enough alone for a high-risk verdict."
        else:
            decision_score = 52.0
            label = "Scam-content review"
            summary = "Transcript content has review-level indicators, but not enough for a high-risk verdict."
        action = "Ask for confirmation outside the call before responding."
    elif authenticity_level == "High voice-authenticity concern":
        decision_score = min(58.0, max(45.0, effective_authenticity_risk * 0.55))
        label = "Voice authenticity concern - content lower risk"
        summary = "The voice signal raises synthetic-voice concern, but the transcript content does not show scam pressure."
        action = "Verify identity, but do not treat the content as a proven scam from voice alone."
    elif authenticity_level == "Needs voice-authenticity review":
        decision_score = min(45.0, max(35.0, effective_authenticity_risk * 0.55))
        label = "Voice authenticity review - content lower risk"
        summary = "The wording appears lower risk, while the voice signal should be reviewed separately."
        action = "Use normal verification if the request was unexpected."
    elif authenticity_level in {"Weak voice-authenticity evidence", "Inconclusive voice-authenticity evidence"}:
        decision_score = min(35.0, max(20.0, effective_authenticity_risk * 0.55))
        label = "Lower threat - limited voice evidence"
        summary = "Transcript content is not scam-like, and the high raw voice score is not reliable enough to drive the verdict."
        action = "Use a longer recording if voice cloning is the concern."
    else:
        decision_score = 20.0
        label = "Lower concern"
        summary = "Transcript content and voice-authenticity evidence are both lower concern."
        action = "Continue cautious review for unexpected requests."

    return {
        "decision_score": round(_bounded_score(decision_score), 2),
        "decision_label": label,
        "decision_summary": summary,
        "action_recommendation": action,
        "content_level": content_level,
        "authenticity_level": authenticity_level,
        "authenticity_score": round(authenticity_score, 2),
        "content_reliability": round(content_reliability, 2),
        "voice_reliability": round(voice_reliability, 2),
        "behavioral_reliability": round(behavioral_reliability, 2),
        "voice_evidence_risk": round(voice_evidence_risk, 2),
        "rule_voice_evidence_risk": round(rule_voice_evidence_risk, 2),
        "trained_voice_evidence_risk": round(_bounded_score(float(trained_voice_evidence_risk)), 2)
        if trained_voice_evidence_risk is not None
        else None,
        "voice_evidence_engine": voice_evidence_engine,
        "behavioral_evidence_risk": round(behavioral_evidence_risk, 2) if behavioral_evidence_risk is not None else None,
        "effective_content_risk": round(effective_content_risk, 2),
        "effective_authenticity_risk": round(effective_authenticity_risk, 2),
        "audio_evidence_notes": _speech_quality_notes(speech_quality),
        "evidence_policy": "Reliability-weighted content/authenticity evidence jury",
    }


def risk_level(score: float) -> str:
    if score >= 70:
        return "High risk"
    if score >= 40:
        return "Needs review"
    return "Lower risk"


def analyse_live_chunk(
    audio: np.ndarray,
    *,
    transcript: str = "",
    audio_classifier: Any | None = None,
    text_classifier: Any | None = None,
    behavioral_classifier: Any | None = None,
    voice_evidence_calibrator: Any | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    audio_quality: dict[str, object] | None = None,
) -> dict[str, object]:
    """Analyse one microphone chunk and return a UI/report-ready result."""

    features = extract_live_features(audio, sample_rate=sample_rate)
    behavioral_features = extract_behavioral_features(audio, sample_rate=sample_rate)
    speech_quality = audio_quality or assess_speech_quality(
        audio,
        sample_rate=sample_rate,
        features=features,
        behavioral_features=behavioral_features,
    )
    features["speech_quality"] = speech_quality
    usable_speech = bool(speech_quality.get("usable_speech", True))

    voice_risk, voice_label, audio_engine = score_audio_chunk(features, audio_classifier)
    if usable_speech:
        behavioral_risk, behavioral_label, behavioral_engine = score_behavioral_chunk(
            behavioral_features,
            behavioral_classifier,
        )
    else:
        behavioral_risk = None
        behavioral_label = str(speech_quality.get("reason", "No usable speech detected"))
        behavioral_engine = "Skipped by audio quality gate"

    transcript_risk, transcript_label, text_engine, findings = score_transcript(
        transcript,
        text_classifier,
    )
    (
        trained_voice_evidence_risk,
        trained_voice_evidence_label,
        voice_evidence_engine,
        voice_evidence_features,
    ) = score_voice_evidence_chunk(
        raw_voice_risk=voice_risk,
        raw_behavioral_risk=behavioral_risk,
        features=features,
        behavioral_features=behavioral_features,
        speech_quality=speech_quality,
        voice_evidence_calibrator=voice_evidence_calibrator,
    )
    raw_combined_risk = combined_risk(
        voice_risk=voice_risk,
        transcript_risk=transcript_risk,
        has_transcript=bool(transcript.strip()),
        behavioral_risk=behavioral_risk,
    )
    flags = [str(item.get("phrase", "")) for item in findings if item.get("phrase")]
    decision = decision_layer(
        transcript=transcript.strip(),
        voice_risk=voice_risk,
        transcript_risk=transcript_risk,
        behavioral_risk=behavioral_risk,
        speech_quality=speech_quality,
        findings=findings,
        audio_engine=audio_engine,
        text_engine=text_engine,
        behavioral_features=behavioral_features,
        behavioral_engine=behavioral_engine,
        trained_voice_evidence_risk=trained_voice_evidence_risk,
        voice_evidence_engine=voice_evidence_engine,
    )
    total_risk = float(decision["decision_score"])
    level = risk_level(total_risk)
    behavioral_evidence = decision.get("behavioral_evidence_risk")
    behavioral_text = (
        f"{behavioral_risk:.1f}% raw using {behavioral_engine}; "
        f"{float(behavioral_evidence):.1f}% evidence after "
        f"{decision['behavioral_reliability']:.1f}% reliability"
        if behavioral_risk is not None and behavioral_evidence is not None
        else behavioral_engine
    )
    quality_text = (
        str(speech_quality.get("reason", "Usable speech-like audio"))
        if not usable_speech
        else "usable speech-like audio"
    )
    voice_evidence = float(decision.get("voice_evidence_risk", voice_risk))
    rule_voice_evidence = float(decision.get("rule_voice_evidence_risk", voice_evidence))
    trained_evidence_text = (
        f"{voice_evidence:.1f}% trained evidence using {voice_evidence_engine}; "
        f"rule-weighted fallback would be {rule_voice_evidence:.1f}%"
        if trained_voice_evidence_risk is not None
        else f"{voice_evidence:.1f}% evidence after {decision['voice_reliability']:.1f}% reliability"
    )
    audio_text = (
        f"Voice signal {voice_risk:.1f}% raw using {audio_engine}; "
        f"{trained_evidence_text}"
        if usable_speech
        else f"Voice signal skipped by {audio_engine}: {quality_text}"
    )
    evidence_notes = decision.get("audio_evidence_notes", [])
    evidence_note_text = (
        "; ".join(str(note) for note in evidence_notes if str(note).strip())
        if isinstance(evidence_notes, list)
        else ""
    )

    return {
        "time": now_for_app().strftime("%H:%M:%S"),
        "transcript": transcript.strip(),
        "risk": round(total_risk, 2),
        "risk_level": level,
        "raw_combined_risk": round(raw_combined_risk, 2),
        "voice_risk": round(voice_risk, 2),
        "voice_label": voice_label,
        "transcript_risk": round(transcript_risk, 2),
        "transcript_label": transcript_label,
        "behavioral_risk": round(behavioral_risk, 2) if behavioral_risk is not None else None,
        "behavioral_label": behavioral_label,
        "trained_voice_evidence_label": trained_voice_evidence_label,
        "audio_engine": audio_engine,
        "text_engine": text_engine,
        "behavioral_engine": behavioral_engine,
        "voice_evidence_features": voice_evidence_features,
        "flags": flags,
        "findings": findings,
        "features": features,
        "behavioral_features": behavioral_features,
        "audio_quality": speech_quality,
        "quality_warnings": list(speech_quality.get("warnings", [])),
        **decision,
        "explanation": (
            f"{decision['decision_label']} ({total_risk:.1f}%). "
            f"{decision['decision_summary']} "
            f"{audio_text}; "
            f"transcript signal {transcript_risk:.1f}% using {text_engine}. "
            f"Behavioral signal {behavioral_text}. "
            f"Authenticity evidence {decision['effective_authenticity_risk']:.1f}%; "
            f"content reliability {decision['content_reliability']:.1f}%. "
            f"Reliability limits: {evidence_note_text or 'none'}. "
            f"Detected phrase indicators: {', '.join(flags) if flags else 'none'}. "
            f"Raw blended model score for evidence: {raw_combined_risk:.1f}%."
        ),
    }


def _detect_whisper_language(
    audio: np.ndarray,
    whisper_model: Any,
) -> tuple[str, float] | None:
    """Run Whisper's language detector for diagnostics without trusting it blindly."""

    try:
        import whisper
        import torch
    except Exception:
        return None

    try:
        samples = np.asarray(audio, dtype=np.float32)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(samples))
        device = getattr(whisper_model, "device", None)
        if device is not None:
            mel = mel.to(device)
        with torch.no_grad():
            _, probabilities = whisper_model.detect_language(mel)
        if not probabilities:
            return None
        language = max(probabilities, key=probabilities.get)
        return str(language), float(probabilities[language])
    except Exception:
        return None


def _latin_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 1.0
    latin = [character for character in letters if character.isascii()]
    return len(latin) / max(1, len(letters))


def transcribe_with_whisper_details(
    audio: np.ndarray,
    whisper_model: Any | None,
    *,
    language: str | None = "en",
    task: str = "transcribe",
    initial_prompt: str | None = None,
    detect_language: bool = True,
) -> dict[str, object]:
    """Transcribe one chunk and return text plus quality diagnostics."""

    if whisper_model is None:
        return {
            "text": "",
            "usable": False,
            "quality_label": "Whisper unavailable",
            "warnings": ["Local Whisper is unavailable."],
            "language": language or "",
            "language_confidence": None,
            "forced_language": language or "",
            "no_speech_probability": None,
            "avg_logprob": None,
            "compression_ratio": None,
        }

    detected = _detect_whisper_language(audio, whisper_model) if detect_language else None
    detected_language = detected[0] if detected else ""
    language_confidence = detected[1] if detected else None

    decode_options: dict[str, object] = {
        "fp16": False,
        "verbose": None,
        "condition_on_previous_text": False,
        "task": task,
        "temperature": 0.0,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
    }
    if language:
        decode_options["language"] = language
    if initial_prompt:
        decode_options["initial_prompt"] = initial_prompt

    result = whisper_model.transcribe(np.asarray(audio, dtype=np.float32), **decode_options)
    text = str(result.get("text", "")).strip()
    segments = result.get("segments", [])
    segment_values = segments if isinstance(segments, list) else []

    no_speech_values = [
        float(segment.get("no_speech_prob"))
        for segment in segment_values
        if isinstance(segment, dict) and segment.get("no_speech_prob") is not None
    ]
    logprob_values = [
        float(segment.get("avg_logprob"))
        for segment in segment_values
        if isinstance(segment, dict) and segment.get("avg_logprob") is not None
    ]
    compression_values = [
        float(segment.get("compression_ratio"))
        for segment in segment_values
        if isinstance(segment, dict) and segment.get("compression_ratio") is not None
    ]
    no_speech_probability = max(no_speech_values) if no_speech_values else None
    avg_logprob = sum(logprob_values) / len(logprob_values) if logprob_values else None
    compression_ratio = max(compression_values) if compression_values else None

    warnings: list[str] = []
    usable = bool(text)
    quality_label = "Transcript accepted" if usable else "No speech text produced"
    expected_language = str(language or "").strip().casefold()

    if not text:
        warnings.append("Whisper did not produce usable speech text for this chunk.")
    if (
        expected_language
        and detected_language
        and detected_language.casefold() != expected_language
        and language_confidence is not None
        and language_confidence >= 0.55
    ):
        warnings.append(
            f"Language check leaned {detected_language} ({language_confidence:.0%}) while English transcription is forced."
        )
    if no_speech_probability is not None and no_speech_probability >= 0.75:
        warnings.append("Whisper marked this chunk as likely no-speech.")
        if not text:
            usable = False
            quality_label = "Likely no speech"
    if avg_logprob is not None and avg_logprob < -1.2:
        warnings.append("Whisper confidence was low; verify the transcript before using it.")
    if compression_ratio is not None and compression_ratio > 2.6:
        warnings.append("Whisper output may be repetitive or hallucinated.")
    if expected_language == "en" and text and _latin_ratio(text) < 0.80:
        usable = False
        quality_label = "Non-English characters detected"
        warnings.append("Transcript contains many non-Latin characters while English mode is selected.")

    return {
        "text": text if usable else "",
        "raw_text": text,
        "usable": usable,
        "quality_label": quality_label,
        "warnings": warnings,
        "language": str(result.get("language", detected_language or language or "")),
        "detected_language": detected_language,
        "language_confidence": language_confidence,
        "forced_language": language or "",
        "no_speech_probability": no_speech_probability,
        "avg_logprob": avg_logprob,
        "compression_ratio": compression_ratio,
    }


def transcribe_with_whisper(
    audio: np.ndarray,
    whisper_model: Any | None,
    *,
    language: str | None = "en",
    task: str = "transcribe",
) -> str:
    """Transcribe one chunk with an optional local Whisper model."""

    result = transcribe_with_whisper_details(
        audio,
        whisper_model,
        language=language,
        task=task,
    )
    return str(result.get("text", "")).strip()


__all__ = [
    "BEHAVIORAL_FEATURE_NAMES",
    "TARGET_SAMPLE_RATE",
    "assess_speech_quality",
    "analyse_live_chunk",
    "decision_layer",
    "extract_behavioral_features",
    "extract_live_features",
    "transcribe_with_whisper_details",
    "transcribe_with_whisper",
    "wav_bytes_to_audio",
]
