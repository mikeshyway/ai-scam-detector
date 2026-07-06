"""Near-real-time audio analysis shared by local capture and upload flows."""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

from src.text.explainability import find_suspicious_phrases
from src.text.rule_demo import rule_based_text_prediction
from src.utils.time_utils import now_for_app


TARGET_SAMPLE_RATE = 16_000
N_MFCC = 40
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

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
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict[str, object]:
    """Analyse one microphone chunk and return a UI/report-ready result."""

    features = extract_live_features(audio, sample_rate=sample_rate)
    behavioral_features = extract_behavioral_features(audio, sample_rate=sample_rate)
    voice_risk, voice_label, audio_engine = score_audio_chunk(features, audio_classifier)
    behavioral_risk, behavioral_label, behavioral_engine = score_behavioral_chunk(
        behavioral_features,
        behavioral_classifier,
    )
    transcript_risk, transcript_label, text_engine, findings = score_transcript(
        transcript,
        text_classifier,
    )
    total_risk = combined_risk(
        voice_risk=voice_risk,
        transcript_risk=transcript_risk,
        has_transcript=bool(transcript.strip()),
        behavioral_risk=behavioral_risk,
    )
    level = risk_level(total_risk)
    flags = [str(item.get("phrase", "")) for item in findings if item.get("phrase")]
    behavioral_text = (
        f"{behavioral_risk:.1f}% using {behavioral_engine}"
        if behavioral_risk is not None
        else behavioral_engine
    )

    return {
        "time": now_for_app().strftime("%H:%M:%S"),
        "transcript": transcript.strip(),
        "risk": round(total_risk, 2),
        "risk_level": level,
        "voice_risk": round(voice_risk, 2),
        "voice_label": voice_label,
        "transcript_risk": round(transcript_risk, 2),
        "transcript_label": transcript_label,
        "behavioral_risk": round(behavioral_risk, 2) if behavioral_risk is not None else None,
        "behavioral_label": behavioral_label,
        "audio_engine": audio_engine,
        "text_engine": text_engine,
        "behavioral_engine": behavioral_engine,
        "flags": flags,
        "findings": findings,
        "features": features,
        "behavioral_features": behavioral_features,
        "explanation": (
            f"Combined educational risk {total_risk:.1f}%. "
            f"Voice signal {voice_risk:.1f}% using {audio_engine}; "
            f"transcript signal {transcript_risk:.1f}% using {text_engine}. "
            f"Behavioral signal {behavioral_text}. "
            f"Detected phrase indicators: {', '.join(flags) if flags else 'none'}."
        ),
    }


def transcribe_with_whisper(audio: np.ndarray, whisper_model: Any | None) -> str:
    """Transcribe one chunk with an optional local Whisper model."""

    if whisper_model is None:
        return ""
    result = whisper_model.transcribe(
        np.asarray(audio, dtype=np.float32),
        fp16=False,
        verbose=False,
        condition_on_previous_text=False,
    )
    return str(result.get("text", "")).strip()


__all__ = [
    "BEHAVIORAL_FEATURE_NAMES",
    "TARGET_SAMPLE_RATE",
    "analyse_live_chunk",
    "extract_behavioral_features",
    "extract_live_features",
    "transcribe_with_whisper",
    "wav_bytes_to_audio",
]
