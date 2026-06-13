"""Near-real-time audio analysis shared by local capture and upload flows."""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np

from src.explainability import find_suspicious_phrases, rule_based_text_prediction
from src.time_utils import now_for_app


TARGET_SAMPLE_RATE = 16_000


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


def extract_live_features(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, object]:
    """Extract MFCC and lightweight acoustic indicators from one chunk."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < 512:
        raise ValueError("Audio chunk is too short for feature extraction.")
    spectrum_frequencies, spectrum_db, dominant_frequency = _spectrum_summary(audio, sample_rate)

    try:
        import librosa
    except Exception:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio))).astype(np.float32)))
        spectrum = np.abs(np.fft.rfft(audio))
        frequencies = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
        spectrum_sum = float(np.sum(spectrum))
        centroid = float(np.sum(frequencies * spectrum) / spectrum_sum) if spectrum_sum else 0.0
        return {
            "feature_vector": np.zeros(80, dtype=np.float32),
            "mfcc_mean": np.zeros(40, dtype=float).tolist(),
            "mfcc_dynamics": 20.0,
            "spectral_centroid": centroid,
            "zero_crossing_rate": zcr,
            "rms_energy": rms,
            "pitch_variance": 20.0,
            "spectral_rolloff": 0.0,
            "mfcc_available": False,
            "spectrum_frequencies": spectrum_frequencies,
            "spectrum_db": spectrum_db,
            "dominant_frequency": dominant_frequency,
        }

    mfcc_full = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc_full, axis=1)
    mfcc_std = np.std(mfcc_full, axis=1)
    feature_vector = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)))

    pitch_variance = 0.0
    try:
        pitches = librosa.yin(audio, fmin=55, fmax=400, sr=sample_rate)
        finite = pitches[np.isfinite(pitches)]
        if finite.size:
            pitch_variance = float(np.std(finite))
    except Exception:
        pitch_variance = 0.0

    return {
        "feature_vector": feature_vector,
        "mfcc_mean": mfcc_mean.astype(float).tolist(),
        "mfcc_dynamics": float(np.mean(mfcc_std)),
        "spectral_centroid": centroid,
        "zero_crossing_rate": zcr,
        "rms_energy": rms,
        "pitch_variance": pitch_variance,
        "spectral_rolloff": rolloff,
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

    prediction = audio_classifier.predict_one(np.asarray(features["feature_vector"], dtype=np.float32))
    risk = float(prediction.probabilities.get("Possible AI-generated speech", 0.0)) * 100
    return risk, prediction.label_name, "MFCC + SVM"


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
    finding_count: int,
) -> float:
    """Fuse voice and transcript signals without presenting either as proof."""

    if not has_transcript:
        return max(0.0, min(100.0, voice_risk))
    score = transcript_risk * 0.68 + voice_risk * 0.32
    score += min(8.0, finding_count * 1.5)
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
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict[str, object]:
    """Analyse one microphone chunk and return a UI/report-ready result."""

    features = extract_live_features(audio, sample_rate=sample_rate)
    voice_risk, voice_label, audio_engine = score_audio_chunk(features, audio_classifier)
    transcript_risk, transcript_label, text_engine, findings = score_transcript(
        transcript,
        text_classifier,
    )
    total_risk = combined_risk(
        voice_risk=voice_risk,
        transcript_risk=transcript_risk,
        has_transcript=bool(transcript.strip()),
        finding_count=len(findings),
    )
    level = risk_level(total_risk)
    flags = [str(item.get("phrase", "")) for item in findings if item.get("phrase")]

    return {
        "time": now_for_app().strftime("%H:%M:%S"),
        "transcript": transcript.strip(),
        "risk": round(total_risk, 2),
        "risk_level": level,
        "voice_risk": round(voice_risk, 2),
        "voice_label": voice_label,
        "transcript_risk": round(transcript_risk, 2),
        "transcript_label": transcript_label,
        "audio_engine": audio_engine,
        "text_engine": text_engine,
        "flags": flags,
        "findings": findings,
        "features": features,
        "explanation": (
            f"Combined educational risk {total_risk:.1f}%. "
            f"Voice signal {voice_risk:.1f}% using {audio_engine}; "
            f"transcript signal {transcript_risk:.1f}% using {text_engine}. "
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
    "TARGET_SAMPLE_RATE",
    "analyse_live_chunk",
    "extract_live_features",
    "transcribe_with_whisper",
    "wav_bytes_to_audio",
]
