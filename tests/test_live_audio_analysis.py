"""Focused tests for the reliable microphone recording pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from src.live_audio_analysis import analyse_live_chunk


class LiveAudioAnalysisTests(unittest.TestCase):
    def test_analysis_combines_voice_and_transcript_indicators(self) -> None:
        sample_rate = 16_000
        time_axis = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = 0.2 * np.sin(2 * np.pi * 220 * time_axis)
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
