"""Focused tests for the reliable microphone recording pipeline."""

from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from src.live_audio_analysis import analyse_live_chunk, wav_bytes_to_audio


def _wav_bytes(*, sample_rate: int = 8_000, duration_seconds: float = 1.0) -> bytes:
    sample_count = int(sample_rate * duration_seconds)
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    mono = 0.2 * np.sin(2 * np.pi * 220 * time_axis)
    stereo = np.column_stack([mono, mono * 0.8])
    pcm = (stereo * 32_767).astype(np.int16)

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


class LiveAudioAnalysisTests(unittest.TestCase):
    def test_wav_decoder_converts_to_mono_16khz_audio(self) -> None:
        audio, sample_rate = wav_bytes_to_audio(_wav_bytes())

        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.size, 16_000)
        self.assertGreater(float(np.max(np.abs(audio))), 0.05)

    def test_analysis_combines_voice_and_transcript_indicators(self) -> None:
        audio, sample_rate = wav_bytes_to_audio(
            _wav_bytes(sample_rate=16_000, duration_seconds=2.0)
        )
        result = analyse_live_chunk(
            audio,
            transcript="Send the OTP now or your account will be suspended.",
            sample_rate=sample_rate,
        )

        self.assertIn("OTP", result["flags"])
        self.assertGreater(float(result["transcript_risk"]), 0.0)
        self.assertIn(result["risk_level"], {"Lower risk", "Needs review", "High risk"})
        self.assertEqual(result["features"]["dominant_frequency"], 220.0)


if __name__ == "__main__":
    unittest.main()
