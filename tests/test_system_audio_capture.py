"""Tests for local system-output capture helpers."""

from __future__ import annotations

import unittest
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np

from src.system_audio_capture import (
    CaptureDevice,
    LocalSystemAudioMonitor,
    classify_capture_device,
    normalise_captured_audio,
    resample_captured_audio,
)


class SystemAudioCaptureTests(unittest.TestCase):
    def test_device_classification_prioritises_loopback_and_virtual_cables(self) -> None:
        self.assertEqual(classify_capture_device("Speakers", True), "System output")
        self.assertEqual(classify_capture_device("CABLE Output", False), "Virtual cable")
        self.assertEqual(classify_capture_device("Built-in microphone", False), "Microphone input")

    def test_stereo_audio_is_downmixed_and_resampled_for_whisper(self) -> None:
        left = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)
        right = np.linspace(0.5, -0.5, 48_000, dtype=np.float32)
        stereo = np.column_stack([left, right + 0.2])

        mono = normalise_captured_audio(stereo)
        resampled = resample_captured_audio(
            mono,
            source_rate=48_000,
            target_rate=16_000,
        )

        self.assertEqual(mono.shape, (48_000,))
        self.assertEqual(resampled.shape, (16_000,))
        self.assertTrue(np.all(np.isfinite(resampled)))

    def test_monitor_queue_returns_captured_chunks(self) -> None:
        monitor = LocalSystemAudioMonitor(queue_size=2)
        first = np.ones(16_000, dtype=np.float32) * 0.1
        second = np.ones(16_000, dtype=np.float32) * 0.2

        monitor._put_chunk(first)
        monitor._put_chunk(second)
        chunks = monitor.drain_chunks(limit=2)

        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(float(np.mean(chunks[0])), 0.1, places=5)
        self.assertAlmostEqual(float(np.mean(chunks[1])), 0.2, places=5)

    def test_background_monitor_emits_resampled_chunks(self) -> None:
        class FakeRecorder:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def record(self, numframes: int) -> np.ndarray:
                sleep(0.001)
                return np.full((numframes, 2), 0.1, dtype=np.float32)

        class FakeDevice:
            def recorder(self, **kwargs):
                return FakeRecorder()

        monitor = LocalSystemAudioMonitor(
            sample_rate=4_000,
            device_sample_rate=8_000,
            queue_size=2,
        )
        device = CaptureDevice(
            device_id="fake-loopback",
            name="Fake speakers",
            kind="System output",
            is_loopback=True,
        )

        with patch("src.system_audio_capture._resolve_device", return_value=FakeDevice()):
            monitor.start(device, chunk_seconds=3)
            deadline = monotonic() + 1.0
            while monitor.stats()["captured_chunks"] == 0 and monotonic() < deadline:
                sleep(0.01)
            monitor.stop()

        chunk = monitor.get_chunk()
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.shape, (12_000,))
        self.assertAlmostEqual(float(np.mean(chunk)), 0.1, places=4)


if __name__ == "__main__":
    unittest.main()
