"""Focused tests for the reliable microphone recording pipeline."""

from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from src.audio.live_audio_analysis import (
    analyse_live_chunk,
    transcribe_with_whisper_details,
    wav_bytes_to_audio,
)


class LiveAudioAnalysisTests(unittest.TestCase):
    def test_streamlit_wav_is_decoded_and_resampled(self) -> None:
        sample_rate = 8_000
        time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = (0.2 * np.sin(2 * np.pi * 220 * time_axis) * 32_767).astype(
            np.int16
        )
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

        audio, decoded_rate = wav_bytes_to_audio(output.getvalue())

        self.assertEqual(decoded_rate, 16_000)
        self.assertEqual(audio.shape, (16_000,))

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

    def test_silent_audio_is_gated_before_audio_classifier(self) -> None:
        class ExplodingAudioClassifier:
            def predict_one(self, _features):
                raise AssertionError("silent chunks should not reach the audio SVM")

        sample_rate = 16_000
        audio = np.zeros(sample_rate * 2, dtype=np.float32)

        result = analyse_live_chunk(
            audio,
            audio_classifier=ExplodingAudioClassifier(),
            sample_rate=sample_rate,
        )

        self.assertEqual(float(result["voice_risk"]), 0.0)
        self.assertEqual(result["audio_engine"], "Audio quality gate")
        self.assertFalse(result["audio_quality"]["usable_speech"])
        self.assertLess(float(result["risk"]), 1.0)

    def test_whisper_transcription_forces_english_task(self) -> None:
        class FakeWhisperModel:
            def __init__(self) -> None:
                self.kwargs = {}

            def transcribe(self, _audio, **kwargs):
                self.kwargs = kwargs
                return {
                    "text": "Please verify through the official office.",
                    "language": "en",
                    "segments": [
                        {
                            "no_speech_prob": 0.05,
                            "avg_logprob": -0.2,
                            "compression_ratio": 1.1,
                        }
                    ],
                }

        model = FakeWhisperModel()
        details = transcribe_with_whisper_details(
            np.ones(16_000, dtype=np.float32) * 0.01,
            model,
            language="en",
            task="transcribe",
            detect_language=False,
        )

        self.assertTrue(details["usable"])
        self.assertEqual(details["text"], "Please verify through the official office.")
        self.assertEqual(model.kwargs["language"], "en")
        self.assertEqual(model.kwargs["task"], "transcribe")
        self.assertFalse(model.kwargs["condition_on_previous_text"])

    def test_whisper_non_latin_output_is_not_used_in_english_mode(self) -> None:
        class FakeWhisperModel:
            def transcribe(self, _audio, **_kwargs):
                return {
                    "text": "你好你好",
                    "language": "zh",
                    "segments": [
                        {
                            "no_speech_prob": 0.1,
                            "avg_logprob": -0.3,
                            "compression_ratio": 1.0,
                        }
                    ],
                }

        details = transcribe_with_whisper_details(
            np.ones(16_000, dtype=np.float32) * 0.01,
            FakeWhisperModel(),
            language="en",
            detect_language=False,
        )

        self.assertFalse(details["usable"])
        self.assertEqual(details["text"], "")
        self.assertIn("non-Latin", " ".join(details["warnings"]))


if __name__ == "__main__":
    unittest.main()
