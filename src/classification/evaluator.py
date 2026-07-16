"""Classification evaluation — accuracy, F1, MCC, AUC."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np


def _predictor_class_labels(predictor) -> list[str] | None:
    for attr in ("classes_", "classes", "class_labels"):
        value = getattr(predictor, attr, None)
        if value is None or isinstance(value, str):
            continue
        try:
            return list(value)
        except TypeError:
            continue
    return None


def _resolve_class_labels(labels, predictions, class_labels=None) -> list:
    if class_labels is not None:
        return list(class_labels)
    return sorted(set(labels) | set(predictions))


def _align_proba_to_metric_labels(proba, ordered_labels, metric_labels):
    if proba is None or ordered_labels == metric_labels:
        return proba
    if proba.ndim != 2 or proba.shape[1] != len(ordered_labels):
        return proba
    order = {label: idx for idx, label in enumerate(ordered_labels)}
    try:
        return proba[:, [order[label] for label in metric_labels]]
    except KeyError:
        return proba


def build_evaluation_result(
    labels,
    predictions,
    proba: np.ndarray | None = None,
    class_labels=None,
) -> dict[str, object]:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    ordered_labels = _resolve_class_labels(
        labels, predictions, class_labels=class_labels
    )
    metric_labels = ordered_labels
    if class_labels is not None:
        class_index = list(range(len(ordered_labels)))
        if set(labels).issubset(class_index) and set(predictions).issubset(class_index):
            metric_labels = class_index
    auc_labels = metric_labels
    if class_labels is not None and metric_labels == ordered_labels:
        auc_labels = sorted(set(labels))

    metrics = compute_classification_metrics(
        labels=labels,
        predictions=predictions,
        proba=_align_proba_to_metric_labels(proba, ordered_labels, auc_labels),
    )

    confusion = confusion_matrix(labels, predictions, labels=metric_labels)
    confusion_df = pd.DataFrame(confusion, index=ordered_labels, columns=ordered_labels)

    normalized = confusion.astype(float)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(
        normalized,
        row_sums,
        out=np.zeros_like(normalized),
        where=row_sums != 0,
    )
    normalized_df = pd.DataFrame(
        normalized,
        index=ordered_labels,
        columns=ordered_labels,
    )

    precision, recall, f1_score, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=metric_labels,
        zero_division=0,
    )
    per_class_df = pd.DataFrame(
        [
            {
                "label": label,
                "precision": precision[idx],
                "recall": recall[idx],
                "f1-score": f1_score[idx],
                "support": support[idx],
            }
            for idx, label in enumerate(ordered_labels)
        ]
    )

    return {
        "metrics": metrics,
        "class_labels": ordered_labels,
        "confusion_matrix": confusion_df,
        "confusion_matrix_normalized": normalized_df,
        "per_class_metrics": per_class_df,
    }


def _write_confusion_matrix_pdf(result: dict[str, object], pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    class_labels = result["class_labels"]
    class_count = len(class_labels)
    cell_size = 0.45
    fig, ax = plt.subplots(
        figsize=(max(6, class_count * cell_size + 3),
                 max(5, class_count * cell_size + 1.5))
    )
    heatmap = ax.imshow(
        result["confusion_matrix_normalized"].to_numpy(),
        cmap="Blues",
        aspect="equal",
        vmin=0, vmax=1,
    )
    ax.set_xticks(range(class_count), class_labels, rotation=45, ha="right")
    ax.set_yticks(range(class_count), class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)


def write_evaluation_outputs(
    result: dict[str, object],
    out_dir: Path,
    pdf_class_limit: int = 50,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in result["metrics"].items()]
    )
    metrics_csv = out_dir / "evaluations.csv"
    confusion_csv = out_dir / "confusion_matrix.csv"
    normalized_csv = out_dir / "confusion_matrix_normalized.csv"
    per_class_csv = out_dir / "per_class_metrics.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    result["confusion_matrix"].to_csv(confusion_csv, index_label="label")
    result["confusion_matrix_normalized"].to_csv(normalized_csv, index_label="label")
    result["per_class_metrics"].to_csv(per_class_csv, index=False)

    written = {
        "evaluations_csv": metrics_csv,
        "confusion_matrix_csv": confusion_csv,
        "confusion_matrix_normalized_csv": normalized_csv,
        "per_class_metrics_csv": per_class_csv,
    }

    if len(result["class_labels"]) <= pdf_class_limit:
        pdf_path = out_dir / "confusion_matrix.pdf"
        _write_confusion_matrix_pdf(result, pdf_path)
        written["confusion_matrix_pdf"] = pdf_path

    return written


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
) -> dict[str, object]:
    """Evaluate AutoGluon predictor and return metrics plus per-class artifacts."""
    from autogluon.multimodal import MultiModalPredictor
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = pd.read_csv(test_csv)
    df["image"] = df["image"].apply(lambda p: str(images_dir / p))

    predictor = MultiModalPredictor.load(str(model_dir))
    predictions = predictor.predict(df).values
    labels = df["label"].values
    class_labels = _predictor_class_labels(predictor)

    proba = None
    try:
        proba = predictor.predict_proba(df).values
    except Exception:
        proba = None

    return build_evaluation_result(
        labels=labels,
        predictions=predictions,
        proba=proba,
        class_labels=class_labels,
    )


def evaluate_onnx(
    test_csv: Path,
    images_dir: Path,
    onnx_path: Path,
    batch_size: int,
    num_threads: int,
) -> dict[str, object]:
    """Evaluate ONNX model and return metrics plus per-class artifacts."""
    from src.classification.predictor import predict_onnx, load_onnx_class_labels

    df = pd.read_csv(test_csv)
    result = predict_onnx(
        df,
        images_dir,
        onnx_path,
        batch_size,
        num_threads,
        show_progress=True,
    )

    labels = df["label"].values
    predictions = result["prediction"].values

    class_labels = load_onnx_class_labels(onnx_path)

    proba_cols = [col for col in result.columns if col.startswith("proba_")]
    if class_labels:
        # Prefer class-name columns (new output); fall back to integer-indexed (proba_0, proba_1, ...)
        named_cols = [f"proba_{cls}" for cls in class_labels if f"proba_{cls}" in result.columns]
        if named_cols:
            proba_cols = named_cols
        else:
            # Legacy ONNX output with proba_0, proba_1, ... — sort numerically
            proba_cols = sorted(
                [c for c in proba_cols if c.split("_", 1)[1].isdigit()],
                key=lambda c: int(c.split("_", 1)[1]),
            )
    else:
        proba_cols = sorted(proba_cols)
    proba = result[proba_cols].to_numpy() if proba_cols else None

    return build_evaluation_result(
        labels=labels,
        predictions=predictions,
        proba=proba,
        class_labels=class_labels,
    )
