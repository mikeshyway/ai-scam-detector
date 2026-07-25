"""Tests for transcript dataset labeling safeguards."""

from __future__ import annotations

import unittest
import shutil
from pathlib import Path

import pandas as pd

from src.preprocessing.transcript_preprocessor import (
    normalise_label,
    prepare_transcript_dataset,
)


class TranscriptPreprocessorTests(unittest.TestCase):
    def test_suspicious_variant_labels_are_not_mapped_legitimate(self) -> None:
        self.assertEqual(normalise_label("potential_scam"), 1)
        self.assertEqual(normalise_label("slightly_suspicious"), 1)
        self.assertEqual(normalise_label("highly_suspicious"), 1)
        self.assertEqual(normalise_label('citing urgency"'), 1)
        self.assertEqual(normalise_label("scam_response"), 0)

    def test_unlabeled_youtube_rows_are_written_for_review_not_training(self) -> None:
        temp_parent = Path.cwd() / "tests_tmp"
        root = temp_parent / "transcript_preprocessor_case"
        if root.exists():
            shutil.rmtree(root)
        try:
            call_dir = root / "call_transcripts_scam_determinations"
            youtube_dir = root / "youtube_scam_phone_call_transcripts"
            call_dir.mkdir(parents=True)
            youtube_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "TEXT": "Please send the verification code immediately to avoid account suspension.",
                        "LABEL": "scam",
                    },
                    {
                        "TEXT": "Wednesday at 2 PM works for our normal project meeting.",
                        "LABEL": "legitimate",
                    },
                ]
            ).to_csv(call_dir / "labeled.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Content": "Thank you for calling support, just let me know when you are ready.",
                        "Source": "https://www.youtube.com/watch?v=test",
                    }
                ]
            ).to_csv(youtube_dir / "unlabeled.csv", index=False)

            output = root / "processed" / "transcript_dataset.csv"
            review = root / "processed" / "transcript_unlabeled_review.csv"
            dataset = prepare_transcript_dataset(root, output, review)
            review_dataset = pd.read_csv(review)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(set(dataset["label"].tolist()), {0, 1})
            self.assertEqual(len(review_dataset), 1)
            self.assertEqual(review_dataset.iloc[0]["reason"], "missing_label_column")
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
