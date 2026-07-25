"""Tests for audio classifier runtime loading behavior."""

from __future__ import annotations

import unittest

from src.audio.audio_classifier import _force_single_worker_prediction


class AudioClassifierRuntimeTests(unittest.TestCase):
    def test_runtime_loader_forces_nested_models_to_single_worker(self) -> None:
        class FakeEstimator:
            def __init__(self, *children) -> None:
                self.n_jobs = -1
                self.estimators_ = list(children)

        child = FakeEstimator()
        parent = FakeEstimator(child)

        _force_single_worker_prediction(parent)

        self.assertEqual(parent.n_jobs, 1)
        self.assertEqual(child.n_jobs, 1)


if __name__ == "__main__":
    unittest.main()
