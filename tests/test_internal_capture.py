"""Tests for internal system-audio capture helpers."""

from __future__ import annotations

import io
import sys
import types
import unittest
import wave
from unittest.mock import patch

import numpy as np

from src.audio.internal_capture import (
    InternalAudioDevice,
    can_capture_internal_device,
    classify_device,
    host_api_priority,
    is_virtual_audio_device,
    list_internal_audio_devices,
    normalise_audio,
    record_internal_chunk,
    resample_audio,
    requires_wasapi_loopback,
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

    def test_capture_retries_after_incompatible_stream_configuration(self) -> None:
        attempts = []

        class FakeStream:
            def __init__(self, channels: int) -> None:
                self.channels = channels

            def start(self) -> None:
                return None

            def read(self, frames: int):
                return np.full((frames, self.channels), 0.05, dtype=np.float32), False

            def stop(self) -> None:
                return None

            def close(self) -> None:
                return None

        def input_stream(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise RuntimeError("unsupported latency")
            return FakeStream(int(kwargs["channels"]))

        fake_sounddevice = types.SimpleNamespace(InputStream=input_stream)
        device = InternalAudioDevice(
            index=7,
            name="Stereo Mix",
            hostapi="Windows DirectSound",
            max_input_channels=2,
            max_output_channels=0,
            default_samplerate=44_100,
            kind="Internal monitor input",
            is_microphone=False,
            is_internal_candidate=True,
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
            audio, sample_rate, wav_bytes = record_internal_chunk(
                device,
                seconds=1,
                minimum_seconds=1,
            )

        self.assertGreaterEqual(len(attempts), 2)
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(audio.size, 16_000)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))

    def test_wasapi_speaker_output_is_diagnostics_only_without_loopback_input(self) -> None:
        self.assertTrue(
            requires_wasapi_loopback(
                "Windows WASAPI",
                max_input_channels=0,
                max_output_channels=2,
            )
        )

        fake_devices = [
            {
                "index": 8,
                "name": "Speakers (2- Realtek(R) Audio)",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000,
            },
            {
                "index": 9,
                "name": "CABLE Output (VB-Audio Virtual Cable)",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            },
            {
                "index": 10,
                "name": "CABLE Input (VB-Audio Virtual Cable)",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000,
            },
        ]
        fake_sounddevice = types.SimpleNamespace(
            query_hostapis=lambda: [{"name": "Windows WASAPI"}],
            query_devices=lambda: fake_devices,
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
            devices = list_internal_audio_devices()

        cable = devices[0]
        speakers = next(device for device in devices if device.index == 8)
        cable_input = next(device for device in devices if device.index == 10)
        self.assertEqual(cable.index, 9)
        self.assertTrue(can_capture_internal_device(cable))
        self.assertTrue(cable.is_recommended)
        self.assertFalse(can_capture_internal_device(speakers))
        self.assertTrue(speakers.is_loopback_required)
        self.assertIn("WASAPI loopback required", speakers.label)
        self.assertFalse(can_capture_internal_device(cable_input))
        self.assertFalse(cable_input.is_loopback_required)
        self.assertIn("Routing endpoint", cable_input.label)

        with self.assertRaisesRegex(RuntimeError, "0 input channels"):
            record_internal_chunk(speakers, seconds=1, minimum_seconds=1)

    def test_virtual_meeting_devices_are_recommended_and_prioritized(self) -> None:
        for name in (
            "CABLE Output (VB-Audio Virtual Cable)",
            "VB-Cable Monitor",
            "VoiceMeeter Output",
            "BlackHole 2ch",
        ):
            self.assertTrue(is_virtual_audio_device(name))

        fake_devices = [
            {
                "index": 3,
                "name": "Stereo Mix",
                "hostapi": 2,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 44_100,
            },
            {
                "index": 9,
                "name": "CABLE Output (VB-Audio Virtual Cable)",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            },
        ]
        fake_sounddevice = types.SimpleNamespace(
            query_hostapis=lambda: [
                {"name": "Windows WASAPI"},
                {"name": "Windows DirectSound"},
                {"name": "MME"},
            ],
            query_devices=lambda: fake_devices,
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
            devices = list_internal_audio_devices()

        self.assertEqual(devices[0].index, 9)
        self.assertTrue(devices[0].is_recommended)
        self.assertIn("[Recommended]", devices[0].label)
        self.assertIn("Meeting Capture Device", devices[0].label)

    def test_wdm_ks_is_unsupported_and_sorted_after_stable_backends(self) -> None:
        fake_devices = [
            {
                "index": 14,
                "name": "Stereo Mix",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            },
            {
                "index": 15,
                "name": "Stereo Mix",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            },
            {
                "index": 16,
                "name": "Stereo Mix",
                "hostapi": 2,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 44_100,
            },
        ]
        fake_sounddevice = types.SimpleNamespace(
            query_hostapis=lambda: [
                {"name": "Windows WDM-KS"},
                {"name": "Windows WASAPI"},
                {"name": "MME"},
            ],
            query_devices=lambda: fake_devices,
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
            devices = list_internal_audio_devices()

        self.assertEqual([device.index for device in devices], [15, 16, 14])
        self.assertFalse(devices[0].is_unsupported_backend)
        self.assertTrue(devices[-1].is_unsupported_backend)
        self.assertIn("WDM-KS unsupported", devices[-1].label)
        self.assertLess(host_api_priority("Windows WASAPI"), host_api_priority("MME"))

        with self.assertRaisesRegex(RuntimeError, "blocking capture workflow"):
            record_internal_chunk(devices[-1], seconds=1, minimum_seconds=1)


if __name__ == "__main__":
    unittest.main()
