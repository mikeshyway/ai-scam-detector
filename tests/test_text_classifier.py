"""Tests for text classifier runtime helpers."""

from __future__ import annotations

import unittest

from src.text.text_classifier import _force_single_worker_prediction


class TextClassifierRuntimeTests(unittest.TestCase):
    def test_runtime_loader_forces_nested_models_to_single_worker(self) -> None:
        class FakeEstimator:
            def __init__(self, *children) -> None:
                self.n_jobs = -1
                self.calibrated_classifiers_ = list(children)

        child = FakeEstimator()
        parent = FakeEstimator(child)

        _force_single_worker_prediction(parent)

        self.assertEqual(parent.n_jobs, 1)
        self.assertEqual(child.n_jobs, 1)


if __name__ == "__main__":
    unittest.main()
