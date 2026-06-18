"""Local internal/system audio capture using sounddevice.

This module is intentionally local-only. It records audio from operating-system
speaker output, monitor sources, or virtual devices such as BlackHole. It should
not be used on Streamlit Cloud because cloud servers cannot access the user's
laptop audio output.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Callable

import numpy as np


TARGET_SAMPLE_RATE = 16_000
_MICROPHONE_TERMS = ("mic", "microphone", "array", "webcam", "camera")
_VIRTUAL_DEVICE_TERMS = (
    "cable",
    "vb-cable",
    "vb cable",
    "voicemeeter",
    "blackhole",
)
_INTERNAL_TERMS = (
    *_VIRTUAL_DEVICE_TERMS,
    "loopback",
    "monitor",
    "stereo mix",
    "what u hear",
)


def is_unsupported_capture_host(hostapi: str) -> bool:
    """Return True for Windows backends incompatible with this blocking workflow."""

    return "wdm-ks" in hostapi.casefold()


def host_api_priority(hostapi: str) -> int:
    """Prefer stable Windows capture APIs and place WDM-KS last."""

    lowered = hostapi.casefold()
    if "wdm-ks" in lowered:
        return 99
    if "wasapi" in lowered:
        return 0
    if "directsound" in lowered:
        return 1
    if "mme" in lowered:
        return 2
    return 10


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
    is_virtual_device: bool = False
    is_recommended: bool = False
    is_unsupported_backend: bool = False
    host_priority: int = 10

    @property
    def label(self) -> str:
        if getattr(self, "is_unsupported_backend", False):
            return (
                f"{self.name} [WDM-KS unsupported for blocking capture | "
                f"{self.hostapi}]"
            )
        if getattr(self, "is_virtual_device", False):
            readiness = "Input-ready" if self.max_input_channels > 0 else "Routing endpoint"
            return (
                f"[Recommended] {self.name} "
                f"[Meeting Capture Device | {readiness} | {self.hostapi}]"
            )
        return f"{self.name} [{self.kind} | {self.hostapi}]"


def _normalise_name(name: str) -> str:
    return name.casefold()


def is_virtual_audio_device(name: str) -> bool:
    """Identify virtual routing devices suited to meeting-audio capture."""

    lowered = _normalise_name(name)
    return any(term in lowered for term in _VIRTUAL_DEVICE_TERMS)


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
        is_virtual = is_virtual_audio_device(str(device["name"]))
        unsupported_backend = is_unsupported_capture_host(hostapi)
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
                is_virtual_device=is_virtual,
                is_recommended=is_virtual and not unsupported_backend,
                is_unsupported_backend=unsupported_backend,
                host_priority=host_api_priority(hostapi),
            )
        )

    return sorted(
        devices,
        key=lambda item: (
            getattr(item, "is_unsupported_backend", False),
            0
            if getattr(item, "is_virtual_device", False) and item.max_input_channels > 0
            else 1
            if item.is_internal_candidate
            else 2
            if getattr(item, "is_virtual_device", False)
            else 3,
            getattr(item, "host_priority", 10),
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
    """Encode mono float32 audio as WAV bytes with a standard-library fallback."""

    mono = normalise_audio(audio)
    buffer = io.BytesIO()
    try:
        import soundfile as sf
    except Exception:
        pcm = np.round(mono * 32_767.0).astype("<i2").tobytes()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm)
        return buffer.getvalue()

    sf.write(buffer, mono, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _stream_settings(device: InternalAudioDevice):
    """Return platform settings for internal capture."""

    import sounddevice as sd

    extra_settings = None
    channels = max(1, min(2, device.max_input_channels or device.max_output_channels))
    if "wasapi" in device.hostapi.casefold() and device.max_output_channels > 0:
        extra_settings = sd.WasapiSettings(auto_convert=True)
        channels = max(1, min(2, device.max_output_channels))
    return extra_settings, channels


def _capture_configurations(device: InternalAudioDevice) -> list[dict[str, object]]:
    """Build conservative stream combinations for picky Windows host APIs."""

    extra_settings, preferred_channels = _stream_settings(device)
    rates = [int(device.default_samplerate or 48_000), 48_000, 44_100]
    channels = [preferred_channels]
    if preferred_channels > 1:
        channels.append(1)

    configurations = []
    seen = set()
    for sample_rate in rates:
        for channel_count in channels:
            for latency in ("high", None):
                signature = (sample_rate, channel_count, latency)
                if signature in seen:
                    continue
                seen.add(signature)
                configurations.append(
                    {
                        "samplerate": sample_rate,
                        "channels": channel_count,
                        "latency": latency,
                        "extra_settings": extra_settings,
                    }
                )
    return configurations


def record_internal_chunk(
    device: InternalAudioDevice,
    *,
    seconds: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    progress_callback: Callable[[float, float], None] | None = None,
    minimum_seconds: int = 5,
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
    if getattr(device, "is_unsupported_backend", False) or is_unsupported_capture_host(
        device.hostapi
    ):
        raise RuntimeError(
            f"'{device.name}' uses {device.hostapi}, which does not support the blocking "
            "capture workflow used by this app. Select the WASAPI, DirectSound, or MME "
            "version of the device, or use an input-capable virtual cable."
        )

    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("sounddevice is required for internal audio capture.") from exc

    minimum_seconds = max(1, min(5, int(minimum_seconds)))
    seconds = max(minimum_seconds, min(10, int(seconds)))
    stream = None
    selected_configuration: dict[str, object] | None = None
    attempt_errors = []
    for configuration in _capture_configurations(device):
        candidate = None
        stream_kwargs = {
            "samplerate": int(configuration["samplerate"]),
            "device": device.index,
            "channels": int(configuration["channels"]),
            "dtype": "float32",
            "blocksize": 0,
        }
        if configuration["latency"] is not None:
            stream_kwargs["latency"] = configuration["latency"]
        if configuration["extra_settings"] is not None:
            stream_kwargs["extra_settings"] = configuration["extra_settings"]
        try:
            candidate = sd.InputStream(**stream_kwargs)
            candidate.start()
        except Exception as exc:
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass
            attempt_errors.append(
                f"{configuration['samplerate']} Hz/{configuration['channels']} ch/"
                f"{configuration['latency'] or 'default'}: {type(exc).__name__}: {exc}"
            )
            continue
        stream = candidate
        selected_configuration = configuration
        break

    if stream is None or selected_configuration is None:
        details = " | ".join(attempt_errors[-6:]) or "No compatible stream configuration."
        raise RuntimeError(
            f"Could not open '{device.name}' (ID {device.index}, {device.hostapi}) as an "
            f"internal-audio input. PortAudio attempts: {details}"
        )

    source_rate = int(selected_configuration["samplerate"])
    block_frames = max(512, int(source_rate * 0.1))
    total_frames = int(source_rate * seconds)
    captured = []
    captured_frames = 0
    try:
        while captured_frames < total_frames:
            frames_to_read = min(block_frames, total_frames - captured_frames)
            block, overflowed = stream.read(frames_to_read)
            block = np.asarray(block, dtype=np.float32)
            captured.append(block)
            captured_frames += block.shape[0]
            if progress_callback is not None:
                level = float(np.sqrt(np.mean(np.square(normalise_audio(block)))))
                progress_callback(captured_frames / total_frames, level)
            if overflowed:
                continue
    except Exception as exc:
        raise RuntimeError(
            f"Capture started on '{device.name}' but reading failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            stream.stop()
        except Exception:
            pass
        stream.close()

    raw = normalise_audio(np.concatenate(captured)) if captured else np.empty(0, dtype=np.float32)
    if raw.size == 0:
        raise RuntimeError("The internal audio capture returned no samples.")
    audio = resample_audio(raw, source_rate=source_rate, target_rate=target_sample_rate)
    return audio, target_sample_rate, wav_bytes_from_audio(audio, target_sample_rate)


__all__ = [
    "InternalAudioDevice",
    "TARGET_SAMPLE_RATE",
    "classify_device",
    "host_api_priority",
    "is_unsupported_capture_host",
    "is_virtual_audio_device",
    "list_internal_audio_devices",
    "normalise_audio",
    "record_internal_chunk",
    "resample_audio",
    "sounddevice_available",
    "wav_bytes_from_audio",
]
