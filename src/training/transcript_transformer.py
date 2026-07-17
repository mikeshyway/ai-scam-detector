"""Fine-tuning helpers for transformer transcript scam classifiers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


LABEL_NAMES = ["Legitimate", "Suspicious"]

TRANSFORMER_MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "distilbert": {
        "display_name": "DistilBERT",
        "checkpoint": "distilbert-base-uncased",
        "artifact_dir": "transcript_distilbert",
    },
    "bert": {
        "display_name": "BERT",
        "checkpoint": "bert-base-uncased",
        "artifact_dir": "transcript_bert",
    },
}


@dataclass(frozen=True)
class TransformerTrainingConfig:
    key: str
    display_name: str
    checkpoint: str
    artifact_dir: Path
    epochs: int = 2
    batch_size: int = 8
    max_length: int = 256
    learning_rate: float = 2e-5
    allow_download: bool = True


class _TranscriptTorchDataset:
    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        tokenizer,
        *,
        max_length: int,
    ) -> None:
        import torch

        self.encodings = tokenizer(
            [str(text) for text in texts],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = torch.tensor([int(label) for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


def _evaluate_transformer(
    name: str,
    model,
    dataloader,
    y_test: list[int],
    device,
) -> dict[str, object]:
    import torch

    model.eval()
    probabilities: list[np.ndarray] = []

    start_pred = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")
            del labels
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            probabilities.append(batch_probs)
    prediction_time = time.perf_counter() - start_pred

    probability_matrix = np.vstack(probabilities)
    y_score = probability_matrix[:, 1]
    y_pred = probability_matrix.argmax(axis=1)

    fpr, tpr, _thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "training_time_seconds": None,
        "prediction_time_seconds": float(prediction_time),
        "prediction_time_ms": float(prediction_time / max(1, len(y_test)) * 1000),
        "confusion_matrix": cm.tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=LABEL_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
    }


def train_transformer_model(
    config: TransformerTrainingConfig,
    X_train: Iterable[str],
    y_train: Iterable[int],
    X_test: Iterable[str],
    y_test: Iterable[int],
) -> dict[str, object]:
    """Fine-tune one Hugging Face sequence classifier and save it locally."""

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_files_only = not config.allow_download
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.checkpoint,
            local_files_only=local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.checkpoint,
            num_labels=2,
            id2label={0: "Legitimate", 1: "Suspicious"},
            label2id={"Legitimate": 0, "Suspicious": 1},
            local_files_only=local_files_only,
        )
    except Exception as exc:
        source_hint = (
            "cached locally"
            if local_files_only
            else "downloadable from Hugging Face"
        )
        raise RuntimeError(
            f"{config.display_name} checkpoint '{config.checkpoint}' is not {source_hint}. "
            "Run training with network access once, or use --no-transformer-download only after the checkpoint is cached."
        ) from exc

    train_dataset = _TranscriptTorchDataset(
        X_train,
        y_train,
        tokenizer,
        max_length=config.max_length,
    )
    test_dataset = _TranscriptTorchDataset(
        X_test,
        y_test,
        tokenizer,
        max_length=config.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    start_train = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        average_loss = running_loss / max(1, len(train_loader))
        print(
            f"{config.display_name} epoch {epoch + 1}/{config.epochs} "
            f"loss={average_loss:.4f}"
        )

    training_time = time.perf_counter() - start_train
    metrics = _evaluate_transformer(
        config.display_name,
        model,
        test_loader,
        [int(label) for label in y_test],
        device,
    )
    metrics["training_seconds"] = float(training_time)
    metrics["training_time_seconds"] = float(training_time)
    metrics["training_time"] = float(training_time)
    metrics["transformer_checkpoint"] = config.checkpoint
    metrics["max_length"] = int(config.max_length)
    metrics["epochs"] = int(config.epochs)
    metrics["batch_size"] = int(config.batch_size)
    metrics["learning_rate"] = float(config.learning_rate)

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.artifact_dir)
    tokenizer.save_pretrained(config.artifact_dir)
    metadata = {
        "model": config.display_name,
        "checkpoint": config.checkpoint,
        "labels": {"0": "Legitimate", "1": "Suspicious"},
        "max_length": config.max_length,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "training_time_seconds": training_time,
    }
    (config.artifact_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return metrics
