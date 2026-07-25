"""Focused tests for the reliable microphone recording pipeline."""

from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from src.audio.live_audio_analysis import (
    analyse_live_chunk,
    decision_layer,
    transcribe_with_whisper_details,
    wav_bytes_to_audio,
)


class LiveAudioAnalysisTests(unittest.TestCase):
    @staticmethod
    def _speech_like_audio(seconds: int = 4, sample_rate: int = 16_000) -> np.ndarray:
        time_axis = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
        rng = np.random.default_rng(42)
        base_voice = 0.08 * np.sin(
            2 * np.pi * (160 + 40 * np.sin(2 * np.pi * 2 * time_axis)) * time_axis
        )
        breath_noise = 0.05 * rng.normal(size=time_axis.size).astype(np.float32)
        envelope = (
            (np.sin(2 * np.pi * 4 * time_axis) > -0.2).astype(np.float32) * 0.8
            + 0.2
        )
        return ((base_voice + breath_noise) * envelope).astype(np.float32)

    @staticmethod
    def _fixed_audio_classifier(ai_probability: float):
        class FixedAudioClassifier:
            def predict_one(self, _features):
                class Prediction:
                    label = 1 if ai_probability >= 0.5 else 0
                    label_name = (
                        "Possible AI-generated speech"
                        if ai_probability >= 0.5
                        else "Real human speech"
                    )
                    confidence = ai_probability if ai_probability >= 0.5 else 1.0 - ai_probability
                    probabilities = {
                        "Real human speech": 1.0 - ai_probability,
                        "Possible AI-generated speech": ai_probability,
                    }

                return Prediction()

        return FixedAudioClassifier()

    @staticmethod
    def _fixed_voice_evidence_calibrator(evidence_risk: float):
        class FixedVoiceEvidenceCalibrator:
            def predict_one(self, _features):
                class Prediction:
                    label_name = (
                        "Trained AI-voice evidence"
                        if evidence_risk >= 60
                        else "Lower trained voice evidence"
                    )
                    confidence = max(evidence_risk, 100.0 - evidence_risk) / 100.0

                    def __init__(self) -> None:
                        self.evidence_risk = evidence_risk

                return Prediction()

        return FixedVoiceEvidenceCalibrator()

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

    def test_high_voice_score_without_transcript_is_not_high_scam_risk(self) -> None:
        result = analyse_live_chunk(
            self._speech_like_audio(),
            audio_classifier=self._fixed_audio_classifier(0.99),
            behavioral_classifier=self._fixed_audio_classifier(0.90),
            sample_rate=16_000,
        )

        self.assertGreater(float(result["raw_combined_risk"]), 90.0)
        self.assertLess(float(result["risk"]), 70.0)
        self.assertEqual(result["risk_level"], "Needs review")
        self.assertEqual(result["content_level"], "Inconclusive")
        self.assertEqual(result["authenticity_level"], "High voice-authenticity concern")
        self.assertIn("transcript unavailable", str(result["decision_label"]).lower())

    def test_high_voice_score_with_benign_transcript_stays_review_not_high(self) -> None:
        result = analyse_live_chunk(
            self._speech_like_audio(),
            transcript="Please confirm this in the official portal before the meeting agenda.",
            audio_classifier=self._fixed_audio_classifier(0.99),
            behavioral_classifier=self._fixed_audio_classifier(0.90),
            sample_rate=16_000,
        )

        self.assertLess(float(result["transcript_risk"]), 40.0)
        self.assertLess(float(result["risk"]), 70.0)
        self.assertEqual(result["risk_level"], "Needs review")
        self.assertEqual(result["content_level"], "Lower scam-content concern")
        self.assertEqual(result["authenticity_level"], "High voice-authenticity concern")

    def test_short_benign_clip_downweights_raw_voice_deepfake_score(self) -> None:
        result = decision_layer(
            transcript="Interesting, interesting.",
            voice_risk=99.9,
            transcript_risk=30.9,
            behavioral_risk=76.3,
            speech_quality={
                "usable_speech": True,
                "duration_seconds": 2.904,
                "speech_activity_ratio": 0.55,
                "silence_ratio": 0.20,
                "estimated_speech_rate": 3.0,
                "warnings": [],
            },
            findings=[],
            audio_engine="MFCC + SVM",
            text_engine="Transcript DistilBERT",
        )

        self.assertLess(float(result["decision_score"]), 40.0)
        self.assertEqual(result["content_level"], "Limited scam-content evidence")
        self.assertEqual(result["authenticity_level"], "Needs voice-authenticity review")
        self.assertLess(float(result["content_reliability"]), 35.0)
        self.assertIn("content lower risk", str(result["decision_label"]).lower())

    def test_short_rule_free_clip_does_not_promote_borderline_text_model_score(self) -> None:
        result = decision_layer(
            transcript="Interesting, interesting.",
            voice_risk=99.9,
            transcript_risk=48.2,
            behavioral_risk=76.3,
            speech_quality={
                "usable_speech": True,
                "duration_seconds": 2.904,
                "speech_activity_ratio": 0.315,
                "silence_ratio": 0.685,
                "estimated_speech_rate": 2.066,
                "warnings": [],
            },
            findings=[],
            audio_engine="MFCC + SVM",
            text_engine="Transcript SVM",
        )

        self.assertLess(float(result["decision_score"]), 40.0)
        self.assertEqual(result["content_level"], "Limited scam-content evidence")
        self.assertEqual(result["authenticity_level"], "Weak voice-authenticity evidence")
        self.assertIn("limited voice evidence", str(result["decision_label"]).lower())

    def test_short_sparse_audio_downweights_voice_and_behavioral_rf_evidence(self) -> None:
        result = decision_layer(
            transcript="Interesting, interesting.",
            voice_risk=99.9,
            transcript_risk=30.9,
            behavioral_risk=76.3,
            speech_quality={
                "usable_speech": True,
                "duration_seconds": 2.904,
                "speech_activity_ratio": 0.315,
                "silence_ratio": 0.685,
                "estimated_speech_rate": 2.066,
                "rms": 0.0117,
                "peak": 0.2166,
                "warnings": [],
            },
            behavioral_features={"rms_energy_mean": 0.0117},
            findings=[],
            audio_engine="MFCC + SVM",
            text_engine="Transcript DistilBERT",
            behavioral_engine="Behavioral RF",
        )

        self.assertLess(float(result["voice_reliability"]), 15.0)
        self.assertLess(float(result["behavioral_reliability"]), 5.0)
        self.assertLess(float(result["voice_evidence_risk"]), 15.0)
        self.assertLess(float(result["behavioral_evidence_risk"]), 5.0)
        self.assertLess(float(result["effective_authenticity_risk"]), 15.0)
        self.assertEqual(result["authenticity_level"], "Weak voice-authenticity evidence")
        self.assertIn("clip is under 3 seconds", result["audio_evidence_notes"])

    def test_trained_voice_evidence_calibrator_replaces_rule_weighted_voice_evidence(self) -> None:
        result = analyse_live_chunk(
            self._speech_like_audio(),
            audio_classifier=self._fixed_audio_classifier(0.99),
            behavioral_classifier=self._fixed_audio_classifier(0.90),
            voice_evidence_calibrator=self._fixed_voice_evidence_calibrator(12.0),
            sample_rate=16_000,
        )

        self.assertEqual(float(result["trained_voice_evidence_risk"]), 12.0)
        self.assertEqual(float(result["voice_evidence_risk"]), 12.0)
        self.assertGreater(float(result["rule_voice_evidence_risk"]), 12.0)
        self.assertEqual(result["voice_evidence_engine"], "Trained voice evidence calibrator")
        self.assertLess(float(result["risk"]), 40.0)

    def test_high_scam_transcript_still_drives_high_risk(self) -> None:
        result = analyse_live_chunk(
            self._speech_like_audio(),
            transcript=(
                "Urgently send the OTP password today or your account will be "
                "suspended and legal action will begin."
            ),
            sample_rate=16_000,
        )

        self.assertGreaterEqual(float(result["transcript_risk"]), 70.0)
        self.assertGreaterEqual(float(result["risk"]), 70.0)
        self.assertEqual(result["risk_level"], "High risk")
        self.assertEqual(result["content_level"], "High scam-content concern")

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
