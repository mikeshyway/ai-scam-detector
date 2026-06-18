"""Graceful local audio dependency and device diagnostics.

The checks in this module intentionally avoid importing optional dependencies at
module load. A missing audio package therefore produces a diagnostic result
instead of preventing the Streamlit application from starting.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _package_check(
    module_name: str,
    display_name: str,
    install_command: str,
) -> dict[str, Any]:
    available = importlib.util.find_spec(module_name) is not None
    return {
        "name": display_name,
        "available": available,
        "status": "PASS" if available else "ERROR",
        "detail": "Available" if available else "Not installed in this Python environment",
        "install_command": "" if available else install_command,
    }


def check_sounddevice() -> dict[str, Any]:
    return _package_check(
        "sounddevice",
        "sounddevice",
        "python -m pip install sounddevice",
    )


def check_soundfile() -> dict[str, Any]:
    return _package_check(
        "soundfile",
        "soundfile",
        "python -m pip install soundfile",
    )


def check_ffmpeg() -> dict[str, Any]:
    executable = shutil.which("ffmpeg")
    system = platform.system().casefold()
    if "windows" in system:
        install_command = "winget install Gyan.FFmpeg"
    elif "darwin" in system:
        install_command = "brew install ffmpeg"
    else:
        install_command = "sudo apt update && sudo apt install ffmpeg"
    return {
        "name": "ffmpeg",
        "available": executable is not None,
        "status": "PASS" if executable else "WARNING",
        "detail": executable or "Executable not found on PATH; Whisper transcription is unavailable",
        "install_command": "" if executable else install_command,
    }


def check_whisper() -> dict[str, Any]:
    return _package_check(
        "whisper",
        "openai-whisper",
        "python -m pip install openai-whisper",
    )


def get_audio_devices() -> list[dict[str, Any]]:
    """Return audio devices without raising when PortAudio is unavailable."""

    if not check_sounddevice()["available"]:
        return []
    try:
        import sounddevice as sd

        hostapis = sd.query_hostapis()
        devices = []
        for index, device in enumerate(sd.query_devices()):
            host_index = int(device.get("hostapi", 0))
            hostapi = str(hostapis[host_index]["name"])
            name = str(device.get("name", f"Device {index}"))
            lowered = name.casefold()
            max_input = int(device.get("max_input_channels", 0))
            max_output = int(device.get("max_output_channels", 0))
            internal_terms = (
                "blackhole",
                "cable",
                "loopback",
                "monitor",
                "stereo mix",
                "voicemeeter",
                "what u hear",
            )
            microphone_terms = ("mic", "microphone", "array", "webcam", "camera")
            has_internal_name = any(term in lowered for term in internal_terms)
            is_microphone = (
                max_input > 0
                and any(term in lowered for term in microphone_terms)
                and not has_internal_name
            )
            is_windows_output = "wasapi" in hostapi.casefold() and max_output > 0
            is_internal = is_windows_output or (has_internal_name and max_input > 0)
            devices.append(
                {
                    "index": int(device.get("index", index)),
                    "name": name,
                    "hostapi": hostapi,
                    "max_input_channels": max_input,
                    "max_output_channels": max_output,
                    "sample_rate": int(float(device.get("default_samplerate", 0) or 0)),
                    "is_microphone": is_microphone,
                    "is_internal_candidate": is_internal,
                    "category": (
                        "Meeting capture"
                        if is_internal
                        else "Microphone"
                        if is_microphone
                        else "Audio device"
                    ),
                }
            )
        return devices
    except Exception:
        return []


def _default_device(kind: str) -> dict[str, Any] | None:
    if not check_sounddevice()["available"]:
        return None
    try:
        import sounddevice as sd

        default_pair = sd.default.device
        position = 0 if kind == "input" else 1
        device_index = int(default_pair[position])
        if device_index < 0:
            return None
        device = dict(sd.query_devices(device_index))
        device["index"] = device_index
        return device
    except Exception:
        return None


def get_default_input_device() -> dict[str, Any] | None:
    return _default_device("input")


def get_default_output_device() -> dict[str, Any] | None:
    return _default_device("output")


def build_audio_diagnostics() -> dict[str, Any]:
    """Build a serialisable snapshot for Streamlit and diagnostic logs."""

    dependencies = [
        check_sounddevice(),
        check_soundfile(),
        check_ffmpeg(),
        check_whisper(),
    ]
    devices = get_audio_devices()
    meeting_devices = [item for item in devices if item["is_internal_candidate"]]
    microphones = [item for item in devices if item["is_microphone"]]
    required_ready = all(
        item["available"]
        for item in dependencies
        if item["name"] in {"sounddevice", "soundfile"}
    )
    if required_ready and meeting_devices:
        capture_status = "PASS"
        capture_message = "Internal system-audio capture is ready for a device test."
    elif not required_ready:
        capture_status = "ERROR"
        capture_message = "Install the missing Python audio dependencies, then refresh checks."
    else:
        capture_status = "WARNING"
        capture_message = (
            "No internal/loopback source was detected. Enable Stereo Mix or install/configure "
            "VB-Cable, VoiceMeeter, BlackHole, or a PulseAudio/PipeWire monitor source."
        )
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "dependencies": dependencies,
        "devices": devices,
        "meeting_devices": meeting_devices,
        "microphones": microphones,
        "default_input": get_default_input_device(),
        "default_output": get_default_output_device(),
        "capture_status": capture_status,
        "capture_message": capture_message,
    }


def analyse_capture_test(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Summarise a short capture and flag empty or effectively silent audio."""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    sample_rate = max(1, int(sample_rate))
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        finite = np.zeros(0, dtype=np.float32)
    duration = float(finite.size / sample_rate)
    rms = float(np.sqrt(np.mean(np.square(finite)))) if finite.size else 0.0
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    rms_db = float(20.0 * np.log10(max(rms, 1e-8)))
    peak_db = float(20.0 * np.log10(max(peak, 1e-8)))
    is_silent = rms < 0.0005 or peak < 0.002
    duration_ok = duration >= 2.5
    if duration_ok and not is_silent:
        status = "PASS"
        message = "System audio was captured and contains an audible signal."
    elif not duration_ok:
        status = "ERROR"
        message = "Capture ended before the 3-second device test completed."
    else:
        status = "WARNING"
        message = "Capture completed, but the selected source appears silent. Play meeting audio and test again."
    return {
        "status": status,
        "message": message,
        "duration_seconds": round(duration, 3),
        "rms": rms,
        "peak": peak,
        "rms_db": round(rms_db, 2),
        "peak_db": round(peak_db, 2),
        "is_silent": is_silent,
        "sample_rate": sample_rate,
    }


def log_system_diagnostics(project_root: Path, diagnostics: dict[str, Any]) -> Path | None:
    """Append a JSON diagnostics snapshot without breaking the application."""

    try:
        log_dir = Path(project_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "system_diagnostics.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(diagnostics, ensure_ascii=True, default=str) + "\n")
        return log_path
    except OSError:
        return None


__all__ = [
    "analyse_capture_test",
    "build_audio_diagnostics",
    "check_ffmpeg",
    "check_sounddevice",
    "check_soundfile",
    "check_whisper",
    "get_audio_devices",
    "get_default_input_device",
    "get_default_output_device",
    "log_system_diagnostics",
]
