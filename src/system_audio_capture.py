"""Local system-output capture for the Live Audio Detection page.

SoundCard exposes Windows WASAPI loopback devices and Linux PulseAudio/PipeWire
monitor sources as microphone-like inputs. Virtual audio cable devices are also
listed so users can route Zoom, Google Meet, or Teams output into the monitor.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


TARGET_SAMPLE_RATE = 16_000
DEVICE_SAMPLE_RATE = 48_000
_VIRTUAL_DEVICE_TERMS = (
    "blackhole",
    "cable",
    "monitor",
    "stereo mix",
    "voicemeeter",
    "virtual",
)


@dataclass(frozen=True)
class CaptureDevice:
    """One operating-system audio input suitable for local capture."""

    device_id: str
    name: str
    kind: str
    is_loopback: bool

    @property
    def label(self) -> str:
        return f"{self.name} [{self.kind}]"


def classify_capture_device(name: str, is_loopback: bool) -> str:
    """Return a concise device type for setup guidance and sorting."""

    if is_loopback:
        return "System output"
    lowered = name.casefold()
    if any(term in lowered for term in _VIRTUAL_DEVICE_TERMS):
        return "Virtual cable"
    return "Microphone input"


def normalise_captured_audio(data: np.ndarray) -> np.ndarray:
    """Downmix captured frames to finite mono float32 audio."""

    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim != 1:
        audio = np.ravel(audio)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def resample_captured_audio(
    audio: np.ndarray,
    *,
    source_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample mono audio without adding another runtime dependency."""

    mono = normalise_captured_audio(audio)
    if source_rate == target_rate or mono.size <= 1:
        return mono
    target_size = max(1, int(round(mono.size * target_rate / source_rate)))
    source_positions = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, mono).astype(np.float32)


def soundcard_available() -> bool:
    try:
        import soundcard  # noqa: F401
    except Exception:
        return False
    return True


def list_capture_devices() -> list[CaptureDevice]:
    """List local inputs, including loopback/monitor sources when supported."""

    try:
        import soundcard as sc
    except Exception:
        return []

    devices: list[CaptureDevice] = []
    seen: set[str] = set()
    for microphone in sc.all_microphones(include_loopback=True):
        device_id = str(microphone.id)
        if device_id in seen:
            continue
        seen.add(device_id)
        name = str(microphone.name)
        is_loopback = bool(getattr(microphone, "isloopback", False))
        devices.append(
            CaptureDevice(
                device_id=device_id,
                name=name,
                kind=classify_capture_device(name, is_loopback),
                is_loopback=is_loopback,
            )
        )

    priority = {"System output": 0, "Virtual cable": 1, "Microphone input": 2}
    return sorted(
        devices,
        key=lambda item: (priority.get(item.kind, 9), item.name.casefold()),
    )


def _resolve_device(device_id: str) -> Any:
    import soundcard as sc

    for microphone in sc.all_microphones(include_loopback=True):
        if str(microphone.id) == device_id:
            return microphone
    raise RuntimeError(
        "The selected audio device is no longer available. Refresh the device list."
    )


class LocalSystemAudioMonitor:
    """Capture local audio continuously and emit fixed-duration mono chunks."""

    def __init__(
        self,
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
        device_sample_rate: int = DEVICE_SAMPLE_RATE,
        queue_size: int = 12,
    ):
        self.sample_rate = int(sample_rate)
        self.device_sample_rate = int(device_sample_rate)
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error = ""
        self._device_id = ""
        self._device_name = ""
        self._chunk_seconds = 5
        self._captured_chunks = 0
        self._dropped_chunks = 0

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive() and not self._stop_event.is_set())

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def device_name(self) -> str:
        with self._lock:
            return self._device_name

    @property
    def chunk_seconds(self) -> int:
        with self._lock:
            return self._chunk_seconds

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "running": self.running,
                "device_name": self._device_name,
                "chunk_seconds": self._chunk_seconds,
                "captured_chunks": self._captured_chunks,
                "dropped_chunks": self._dropped_chunks,
                "queued_chunks": self._chunks.qsize(),
                "error": self._error,
            }

    def start(self, device: CaptureDevice, *, chunk_seconds: int = 5) -> None:
        """Start local capture from one loopback, cable, or microphone input."""

        self.stop()
        self.clear()
        with self._lock:
            self._device_id = device.device_id
            self._device_name = device.name
            self._chunk_seconds = max(3, min(10, int(chunk_seconds)))
            self._captured_chunks = 0
            self._dropped_chunks = 0
            self._error = ""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="aifds-system-audio-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def clear(self) -> None:
        while True:
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                return

    def get_chunk(self) -> np.ndarray | None:
        try:
            return self._chunks.get_nowait()
        except queue.Empty:
            return None

    def drain_chunks(self, *, limit: int = 2) -> list[np.ndarray]:
        chunks = []
        for _ in range(max(1, int(limit))):
            chunk = self.get_chunk()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    def _capture_loop(self) -> None:
        try:
            device = _resolve_device(self._device_id)
            read_frames = max(1_024, self.device_sample_rate // 4)
            chunk_frames = self.device_sample_rate * self.chunk_seconds
            pending = np.empty(0, dtype=np.float32)

            # Capture all available channels and downmix after recording. SoundCard
            # documents unreliable single-channel WASAPI capture on some devices.
            with device.recorder(
                samplerate=self.device_sample_rate,
                blocksize=read_frames * 2,
            ) as recorder:
                while not self._stop_event.is_set():
                    frames = recorder.record(numframes=read_frames)
                    mono = normalise_captured_audio(frames)
                    if mono.size == 0:
                        continue
                    pending = np.concatenate([pending, mono])
                    while pending.size >= chunk_frames:
                        chunk = pending[:chunk_frames].copy()
                        pending = pending[chunk_frames:]
                        self._put_chunk(
                            resample_captured_audio(
                                chunk,
                                source_rate=self.device_sample_rate,
                                target_rate=self.sample_rate,
                            )
                        )
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            self._stop_event.set()

    def _put_chunk(self, chunk: np.ndarray) -> None:
        try:
            self._chunks.put_nowait(chunk)
        except queue.Full:
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                pass
            self._chunks.put_nowait(chunk)
            with self._lock:
                self._dropped_chunks += 1
        with self._lock:
            self._captured_chunks += 1


__all__ = [
    "CaptureDevice",
    "DEVICE_SAMPLE_RATE",
    "LocalSystemAudioMonitor",
    "TARGET_SAMPLE_RATE",
    "classify_capture_device",
    "list_capture_devices",
    "normalise_captured_audio",
    "resample_captured_audio",
    "soundcard_available",
]
