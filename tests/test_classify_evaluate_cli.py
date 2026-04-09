"""Tests for classify evaluate CLI outputs and metrics."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pandas as pd


def test_compute_classification_metrics_includes_common_metrics() -> None:
    from src.classification.evaluator import compute_classification_metrics

    labels = [0, 1, 2, 0, 1, 2]
    predictions = [0, 1, 1, 0, 2, 2]

    metrics = compute_classification_metrics(labels, predictions)

    assert "balanced_accuracy" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "f1_weighted" in metrics
    assert "quadratic_kappa" in metrics


def test_classify_evaluate_run_writes_csv_in_out_dir(
    tmp_path,
    monkeypatch,
) -> None:
    from entomokit.classify import evaluate as evaluate_cli

    out_dir = tmp_path / "eval_out"

    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.classification.utils.select_device",
        lambda _device: SimpleNamespace(type="cpu"),
    )
    monkeypatch.setattr("src.classification.utils.ag_device_map", lambda _d: "cpu")
    monkeypatch.setattr(
        "src.classification.evaluator.evaluate",
        lambda **_kwargs: {
            "accuracy": 0.9,
            "balanced_accuracy": 0.88,
            "f1_weighted": 0.91,
        },
    )

    args = argparse.Namespace(
        test_csv="test.csv",
        images_dir="images",
        model_dir="model_dir",
        onnx_model=None,
        out_dir=str(out_dir),
        batch_size=32,
        num_workers=2,
        num_threads=0,
        device="auto",
    )

    evaluate_cli.run(args)

    metrics_csv = out_dir / "evaluations.csv"
    assert metrics_csv.exists()

    data = pd.read_csv(metrics_csv)
    assert list(data.columns) == ["metric", "value"]
    assert set(data["metric"].tolist()) == {
        "accuracy",
        "balanced_accuracy",
        "f1_weighted",
    }

    assert not (out_dir / "logs" / "evaluations.txt").exists()


def test_evaluate_onnx_maps_string_labels_using_sidecar(
    tmp_path,
    monkeypatch,
) -> None:
    from src.classification.evaluator import evaluate_onnx

    test_csv = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "image": ["a.jpg", "b.jpg"],
            "label": ["Epidorcus_gracilis", "Epidorcus_tonkinensis"],
        }
    ).to_csv(test_csv, index=False)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    (tmp_path / "label_classes.json").write_text(
        json.dumps(
            {
                "class_labels": [
                    "Epidorcus_gracilis",
                    "Epidorcus_tonkinensis",
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.classification.predictor.predict_onnx",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "image": ["a.jpg", "b.jpg"],
                "prediction": ["Epidorcus_gracilis", "Epidorcus_tonkinensis"],
                "prediction_index": [0, 1],
                "proba_0": [0.9, 0.1],
                "proba_1": [0.1, 0.9],
            }
        ),
    )

    metrics = evaluate_onnx(
        test_csv=test_csv,
        images_dir=tmp_path,
        onnx_path=onnx_path,
        batch_size=2,
        num_threads=0,
    )

    assert metrics["accuracy"] == 1.0


def test_evaluate_onnx_enables_progress_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    from src.classification.evaluator import evaluate_onnx

    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"image": ["a.jpg"], "label": ["cls_a"]}).to_csv(test_csv, index=False)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    captured: dict[str, object] = {}

    def fake_predict_onnx(*_args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "image": ["a.jpg"],
                "prediction": ["cls_a"],
                "prediction_index": [0],
                "proba_0": [1.0],
            }
        )

    monkeypatch.setattr("src.classification.predictor.predict_onnx", fake_predict_onnx)

    evaluate_onnx(
        test_csv=test_csv,
        images_dir=tmp_path,
        onnx_path=onnx_path,
        batch_size=1,
        num_threads=0,
    )

    assert captured["show_progress"] is True
