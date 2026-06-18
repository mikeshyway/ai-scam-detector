"""Tests for internal system-audio capture helpers."""

from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from src.audio.internal_capture import (
    classify_device,
    normalise_audio,
    resample_audio,
    wav_bytes_from_audio,
)


class InternalCaptureTests(unittest.TestCase):
    def test_classifies_internal_sources_and_microphones(self) -> None:
        kind, is_mic, is_internal = classify_device(
            "Speakers",
            "Windows WASAPI",
            max_input_channels=0,
            max_output_channels=2,
        )
        self.assertEqual(kind, "Windows WASAPI output")
        self.assertFalse(is_mic)
        self.assertTrue(is_internal)

        kind, is_mic, is_internal = classify_device(
            "Microphone Array",
            "MME",
            max_input_channels=2,
            max_output_channels=0,
        )
        self.assertEqual(kind, "Microphone/input")
        self.assertTrue(is_mic)
        self.assertFalse(is_internal)

        kind, is_mic, is_internal = classify_device(
            "BlackHole 2ch",
            "Core Audio",
            max_input_channels=2,
            max_output_channels=0,
        )
        self.assertEqual(kind, "Internal monitor input")
        self.assertFalse(is_mic)
        self.assertTrue(is_internal)

    def test_downmixes_and_resamples_audio(self) -> None:
        stereo = np.column_stack(
            [
                np.linspace(-0.5, 0.5, 48_000, dtype=np.float32),
                np.linspace(0.5, -0.5, 48_000, dtype=np.float32),
            ]
        )

        mono = normalise_audio(stereo)
        resampled = resample_audio(mono, source_rate=48_000, target_rate=16_000)

        self.assertEqual(mono.shape, (48_000,))
        self.assertEqual(resampled.shape, (16_000,))
        self.assertTrue(np.all(np.isfinite(resampled)))

    def test_wav_encoding_has_standard_library_fallback(self) -> None:
        sample_rate = 16_000
        audio = np.linspace(-0.2, 0.2, sample_rate, dtype=np.float32)

        encoded = wav_bytes_from_audio(audio, sample_rate)

        with wave.open(io.BytesIO(encoded), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), sample_rate)
            self.assertEqual(wav_file.getnframes(), sample_rate)


if __name__ == "__main__":
    unittest.main()
