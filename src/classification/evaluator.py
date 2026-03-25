"""Classification evaluation — accuracy, F1, MCC, AUC."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np


def compute_classification_metrics(
    labels,
    predictions,
    proba: np.ndarray | None = None,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision_macro": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "precision_micro": precision_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "precision_weighted": precision_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall_micro": recall_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "recall_weighted": recall_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_weighted": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "mcc": matthews_corrcoef(labels, predictions),
        "quadratic_kappa": cohen_kappa_score(labels, predictions, weights="quadratic"),
    }

    if proba is None:
        metrics["roc_auc_ovo"] = float("nan")
        metrics["roc_auc_ovr"] = float("nan")
        return metrics

    try:
        if proba.ndim != 2 or proba.shape[1] < 2:
            metrics["roc_auc_ovo"] = float("nan")
            metrics["roc_auc_ovr"] = float("nan")
        elif proba.shape[1] == 2:
            metrics["roc_auc_ovo"] = roc_auc_score(labels, proba[:, 1])
            metrics["roc_auc_ovr"] = metrics["roc_auc_ovo"]
        else:
            metrics["roc_auc_ovo"] = roc_auc_score(
                labels,
                proba,
                multi_class="ovo",
                average="macro",
            )
            metrics["roc_auc_ovr"] = roc_auc_score(
                labels,
                proba,
                multi_class="ovr",
                average="macro",
            )
    except Exception:
        metrics["roc_auc_ovo"] = float("nan")
        metrics["roc_auc_ovr"] = float("nan")

    return metrics


def evaluate(
    test_csv: Path,
    images_dir: Path,
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    num_threads: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate AutoGluon predictor. Returns dict of metric name → value."""
    from autogluon.multimodal import MultiModalPredictor
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = pd.read_csv(test_csv)
    df["image"] = df["image"].apply(lambda p: str(images_dir / p))

    predictor = MultiModalPredictor.load(str(model_dir))
    predictions = predictor.predict(df).values
    labels = df["label"].values

    proba = None
    try:
        proba = predictor.predict_proba(df).values
    except Exception:
        proba = None

    return compute_classification_metrics(
        labels=labels,
        predictions=predictions,
        proba=proba,
    )


def evaluate_onnx(
    test_csv: Path,
    images_dir: Path,
    onnx_path: Path,
    batch_size: int,
    num_threads: int,
) -> Dict[str, float]:
    """Evaluate ONNX model using predict_onnx + sklearn metrics."""
    from src.classification.predictor import predict_onnx

    df = pd.read_csv(test_csv)
    result = predict_onnx(df, images_dir, onnx_path, batch_size, num_threads)

    labels = df["label"].values
    predictions = result["prediction"].values

    proba_cols = sorted(
        [col for col in result.columns if col.startswith("proba_")],
        key=lambda x: int(x.split("_")[1]),
    )
    proba = result[proba_cols].to_numpy() if proba_cols else None

    return compute_classification_metrics(
        labels=labels,
        predictions=predictions,
        proba=proba,
    )
