"""Tests for self-diagnosing local audio setup helpers."""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.utils import system_check
from src.utils.system_check import analyse_capture_test, get_audio_devices, log_system_diagnostics


class SystemCheckTests(unittest.TestCase):
    def test_capture_test_distinguishes_signal_and_silence(self) -> None:
        sample_rate = 16_000
        seconds = 3
        positions = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
        tone = (0.15 * np.sin(2 * np.pi * 440 * positions)).astype(np.float32)

        audible = analyse_capture_test(tone, sample_rate)
        silent = analyse_capture_test(np.zeros_like(tone), sample_rate)

        self.assertEqual(audible["status"], "PASS")
        self.assertFalse(audible["is_silent"])
        self.assertEqual(silent["status"], "WARNING")
        self.assertTrue(silent["is_silent"])

    def test_short_capture_fails_duration_check(self) -> None:
        summary = analyse_capture_test(np.ones(8_000, dtype=np.float32) * 0.1, 16_000)
        self.assertEqual(summary["status"], "ERROR")

    def test_device_diagnostics_mark_speaker_outputs_as_loopback_required(self) -> None:
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

        with (
            patch.object(system_check, "check_sounddevice", return_value={"available": True}),
            patch.dict(sys.modules, {"sounddevice": fake_sounddevice}),
        ):
            devices = get_audio_devices()

        speaker = next(item for item in devices if item["index"] == 8)
        cable = next(item for item in devices if item["index"] == 9)
        cable_input = next(item for item in devices if item["index"] == 10)

        self.assertTrue(speaker["is_loopback_required"])
        self.assertFalse(speaker["is_capture_ready"])
        self.assertEqual(
            speaker["capture_support"],
            "Diagnostics only: WASAPI loopback required",
        )
        self.assertTrue(cable["is_capture_ready"])
        self.assertTrue(cable["is_recommended"])
        self.assertFalse(cable_input["is_capture_ready"])
        self.assertFalse(cable_input["is_loopback_required"])
        self.assertEqual(
            cable_input["capture_support"],
            "Routing endpoint only: select paired capture input",
        )

    def test_diagnostics_log_is_json_lines(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp_system_check"
        root.mkdir(parents=True, exist_ok=True)
        path = None
        try:
            path = log_system_diagnostics(root, {"capture_status": "PASS"})

            self.assertIsNotNone(path)
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["capture_status"], "PASS")
        finally:
            if path is not None and path.exists():
                path.unlink()
            log_dir = root / "logs"
            if log_dir.exists():
                log_dir.rmdir()
            if root.exists():
                root.rmdir()


if __name__ == "__main__":
    unittest.main()
