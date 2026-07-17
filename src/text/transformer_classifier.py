"""Runtime helpers for transformer-based transcript scam classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .text_classifier import TEXT_LABEL_NAMES, TextPrediction


class TransformerTextScamClassifier:
    """Small wrapper that matches the TextScamClassifier runtime interface."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        model_name: str,
        device: Any,
        max_length: int = 384,
        stride: int = 96,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.explainability_mode = "transformer_attention"
        self.model.eval()

    def predict_one(self, text: str) -> TextPrediction:
        probabilities = self._predict_probability_matrix([str(text)])[0]
        label = int(np.argmax(probabilities))
        confidence = float(probabilities[label])
        return TextPrediction(
            label=label,
            label_name=TEXT_LABEL_NAMES.get(label, str(label)),
            confidence=confidence,
            probabilities={
                TEXT_LABEL_NAMES.get(index, str(index)): float(value)
                for index, value in enumerate(probabilities)
            },
            model_name=self.model_name,
        )

    def predict_many(self, texts: list[str]) -> pd.DataFrame:
        clean_texts = [str(text) for text in texts]
        probability_matrix = self._predict_probability_matrix(clean_texts)

        rows = []
        for text, probabilities in zip(clean_texts, probability_matrix):
            label = int(np.argmax(probabilities))
            suspicious_probability = float(probabilities[1])
            confidence = float(probabilities[label])
            rows.append(
                {
                    "preview": text[:120],
                    "prediction": TEXT_LABEL_NAMES.get(label, str(label)),
                    "risk_score": round(suspicious_probability * 100, 2),
                    "confidence": round(confidence * 100, 2),
                }
            )

        return pd.DataFrame(rows)

    def _predict_probability_matrix(self, texts: list[str]) -> np.ndarray:
        import torch

        rows: list[np.ndarray] = []
        for text in texts:
            encoded = self.tokenizer(
                str(text),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            encoded.pop("overflow_to_sample_mapping", None)
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if hasattr(value, "to")
            }
            with torch.no_grad():
                outputs = self.model(**encoded)
                chunk_probabilities = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            if chunk_probabilities.size == 0:
                rows.append(np.array([0.5, 0.5], dtype=float))
            else:
                rows.append(chunk_probabilities.mean(axis=0).astype(float))

        matrix = np.vstack(rows)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return matrix / row_sums


def load_transformer_text_artifacts(
    model_dir: str | Path,
    *,
    model_name: str,
    max_length: int | None = None,
    stride: int | None = None,
) -> TransformerTextScamClassifier:
    """Load a locally saved Hugging Face sequence classifier.

    Streamlit runtime intentionally uses local_files_only so opening the app never
    surprises the user with a model download.
    """

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing transformer model artifact: {model_path}")

    metadata_path = model_path / "training_metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    resolved_max_length = int(max_length or metadata.get("max_length") or 384)
    resolved_stride = int(stride or max(32, min(96, resolved_max_length // 4)))

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Transformer transcript models require torch and transformers."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return TransformerTextScamClassifier(
        tokenizer,
        model,
        model_name=model_name,
        device=device,
        max_length=resolved_max_length,
        stride=resolved_stride,
    )
