"""Local internal/system audio capture using sounddevice.

This module is intentionally local-only. It records audio from operating-system
speaker output, monitor sources, or virtual devices such as BlackHole. It should
not be used on Streamlit Cloud because cloud servers cannot access the user's
laptop audio output.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import numpy as np


TARGET_SAMPLE_RATE = 16_000
_MICROPHONE_TERMS = ("mic", "microphone", "array", "webcam", "camera")
_INTERNAL_TERMS = (
    "blackhole",
    "cable",
    "loopback",
    "monitor",
    "stereo mix",
    "voicemeeter",
    "what u hear",
)


@dataclass(frozen=True)
class InternalAudioDevice:
    """One selectable local audio device."""

    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: int
    kind: str
    is_microphone: bool
    is_internal_candidate: bool

    @property
    def label(self) -> str:
        return f"{self.name} [{self.kind} | {self.hostapi}]"


def _normalise_name(name: str) -> str:
    return name.casefold()


def classify_device(
    name: str,
    hostapi: str,
    *,
    max_input_channels: int,
    max_output_channels: int,
) -> tuple[str, bool, bool]:
    """Classify a sounddevice entry for the Streamlit picker."""

    lowered = _normalise_name(name)
    host = hostapi.casefold()
    has_internal_name = any(term in lowered for term in _INTERNAL_TERMS)
    is_microphone = (
        max_input_channels > 0
        and any(term in lowered for term in _MICROPHONE_TERMS)
        and not has_internal_name
    )

    if "wasapi" in host and max_output_channels > 0:
        return "Windows WASAPI output", is_microphone, True
    if has_internal_name and max_input_channels > 0:
        return "Internal monitor input", False, True
    if max_output_channels > 0 and max_input_channels == 0:
        return "Output device", False, False
    if max_input_channels > 0:
        return "Microphone/input", is_microphone, False
    return "Unavailable", False, False


def sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except Exception:
        return False
    return True


def list_internal_audio_devices() -> list[InternalAudioDevice]:
    """Return available devices with internal-audio candidates first."""

    try:
        import sounddevice as sd
    except Exception:
        return []

    hostapis = sd.query_hostapis()
    devices = []
    for device in sd.query_devices():
        hostapi = str(hostapis[int(device["hostapi"])]["name"])
        max_input = int(device["max_input_channels"])
        max_output = int(device["max_output_channels"])
        kind, is_microphone, is_internal = classify_device(
            str(device["name"]),
            hostapi,
            max_input_channels=max_input,
            max_output_channels=max_output,
        )
        if max_input <= 0 and max_output <= 0:
            continue
        devices.append(
            InternalAudioDevice(
                index=int(device["index"]),
                name=str(device["name"]),
                hostapi=hostapi,
                max_input_channels=max_input,
                max_output_channels=max_output,
                default_samplerate=max(8_000, int(float(device["default_samplerate"]))),
                kind=kind,
                is_microphone=is_microphone,
                is_internal_candidate=is_internal,
            )
        )

    return sorted(
        devices,
        key=lambda item: (
            not item.is_internal_candidate,
            item.is_microphone,
            item.name.casefold(),
        ),
    )


def normalise_audio(audio: np.ndarray) -> np.ndarray:
    """Convert captured samples to finite mono float32."""

    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    elif data.ndim != 1:
        data = np.ravel(data)
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def resample_audio(
    audio: np.ndarray,
    *,
    source_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample mono audio with NumPy to keep the capture helper lightweight."""

    mono = normalise_audio(audio)
    if source_rate == target_rate or mono.size <= 1:
        return mono
    target_size = max(1, int(round(mono.size * target_rate / source_rate)))
    source_positions = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, mono).astype(np.float32)


def wav_bytes_from_audio(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Encode mono float32 audio as WAV bytes."""

    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile is required to write captured WAV chunks.") from exc

    buffer = io.BytesIO()
    sf.write(buffer, normalise_audio(audio), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _stream_settings(device: InternalAudioDevice):
    """Return sounddevice stream settings for internal capture."""

    import sounddevice as sd

    extra_settings = None
    channels = max(1, min(2, device.max_input_channels or device.max_output_channels))
    if "wasapi" in device.hostapi.casefold() and device.max_output_channels > 0:
        extra_settings = sd.WasapiSettings(auto_convert=True)
        channels = max(1, min(2, device.max_output_channels))
    return extra_settings, channels


def record_internal_chunk(
    device: InternalAudioDevice,
    *,
    seconds: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    progress_callback: Callable[[float, float], None] | None = None,
) -> tuple[np.ndarray, int, bytes]:
    """Record one 5-10 second internal-audio chunk.

    The selected device must be a system-output/monitor candidate. Physical
    microphones are rejected so this feature does not capture room noise.
    """

    if device.is_microphone or not device.is_internal_candidate:
        raise RuntimeError(
            "Selected device is not an internal/system-audio source. Choose a "
            "WASAPI output, monitor source, BlackHole, Stereo Mix, or virtual cable."
        )

    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("sounddevice is required for internal audio capture.") from exc

    seconds = max(5, min(10, int(seconds)))
    source_rate = int(device.default_samplerate or 48_000)
    extra_settings, channels = _stream_settings(device)
    block_frames = max(512, int(source_rate * 0.25))
    total_frames = int(source_rate * seconds)
    captured = []
    captured_frames = 0

    try:
        with sd.InputStream(
            samplerate=source_rate,
            device=device.index,
            channels=channels,
            dtype="float32",
            extra_settings=extra_settings,
            blocksize=block_frames,
        ) as stream:
            while captured_frames < total_frames:
                frames_to_read = min(block_frames, total_frames - captured_frames)
                block, _overflowed = stream.read(frames_to_read)
                block = np.asarray(block, dtype=np.float32)
                captured.append(block)
                captured_frames += block.shape[0]
                if progress_callback is not None:
                    level = float(np.sqrt(np.mean(np.square(normalise_audio(block)))))
                    progress_callback(captured_frames / total_frames, level)
    except Exception as exc:
        raise RuntimeError(
            "Internal audio capture could not start for this device. On Windows, "
            "choose a WASAPI speaker/output device. On macOS, route audio through "
            "BlackHole. On Linux, choose a PulseAudio/PipeWire monitor source."
        ) from exc

    raw = normalise_audio(np.concatenate(captured)) if captured else np.empty(0, dtype=np.float32)
    if raw.size == 0:
        raise RuntimeError("The internal audio capture returned no samples.")
    audio = resample_audio(raw, source_rate=source_rate, target_rate=target_sample_rate)
    return audio, target_sample_rate, wav_bytes_from_audio(audio, target_sample_rate)


__all__ = [
    "InternalAudioDevice",
    "TARGET_SAMPLE_RATE",
    "classify_device",
    "list_internal_audio_devices",
    "normalise_audio",
    "record_internal_chunk",
    "resample_audio",
    "sounddevice_available",
    "wav_bytes_from_audio",
]
