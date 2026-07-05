"""Tests for classify evaluate CLI outputs and metrics."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
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


def test_classify_evaluate_run_writes_all_outputs(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from entomokit.classify import evaluate as evaluate_cli

    out_dir = tmp_path / "eval_out"

    confusion = pd.DataFrame([[1, 0], [0, 1]], index=["a", "b"], columns=["a", "b"])
    normalized = confusion.astype(float)
    per_class = pd.DataFrame(
        [
            {"label": "a", "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
            {"label": "b", "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
        ]
    )

    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.classification.utils.select_device",
        lambda _device: SimpleNamespace(type="cpu"),
    )
    monkeypatch.setattr("src.classification.utils.ag_device_map", lambda _d: "cpu")
    monkeypatch.setattr(
        "src.classification.evaluator.evaluate",
        lambda **_kwargs: {
            "metrics": {
                "accuracy": 0.9,
                "balanced_accuracy": 0.88,
                "f1_weighted": 0.91,
            },
            "class_labels": ["a", "b"],
            "confusion_matrix": confusion,
            "confusion_matrix_normalized": normalized,
            "per_class_metrics": per_class,
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
    confusion_csv = out_dir / "confusion_matrix.csv"
    normalized_csv = out_dir / "confusion_matrix_normalized.csv"
    per_class_csv = out_dir / "per_class_metrics.csv"
    pdf_path = out_dir / "confusion_matrix.pdf"

    assert metrics_csv.exists()
    assert confusion_csv.exists()
    assert normalized_csv.exists()
    assert per_class_csv.exists()
    assert pdf_path.exists()

    data = pd.read_csv(metrics_csv)
    assert list(data.columns) == ["metric", "value"]
    assert set(data["metric"].tolist()) == {
        "accuracy",
        "balanced_accuracy",
        "f1_weighted",
    }

    confusion_data = pd.read_csv(confusion_csv)
    assert list(confusion_data.columns) == ["label", "a", "b"]

    normalized_data = pd.read_csv(normalized_csv)
    assert list(normalized_data.columns) == ["label", "a", "b"]

    per_class_data = pd.read_csv(per_class_csv)
    assert list(per_class_data["label"]) == ["a", "b"]

    out = capsys.readouterr().out
    assert "accuracy: 0.900000" in out
    assert f"Results saved to: {metrics_csv}" in out

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

    result = evaluate_onnx(
        test_csv=test_csv,
        images_dir=tmp_path,
        onnx_path=onnx_path,
        batch_size=2,
        num_threads=0,
    )

    assert result["metrics"]["accuracy"] == 1.0


def test_evaluate_onnx_enables_progress_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    from src.classification.evaluator import evaluate_onnx

    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"image": ["a.jpg", "b.jpg"], "label": ["cls_a", "cls_b"]}).to_csv(test_csv, index=False)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    captured: dict[str, object] = {}

    def fake_predict_onnx(*_args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "image": ["a.jpg", "b.jpg"],
                "prediction": ["cls_a", "cls_b"],
                "prediction_index": [0, 1],
                "proba_0": [1.0, 0.1],
                "proba_1": [0.0, 0.9],
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


def test_build_evaluation_result_returns_confusion_and_per_class_tables() -> None:
    from src.classification.evaluator import build_evaluation_result

    result = build_evaluation_result(
        labels=["b", "a", "a", "b"],
        predictions=["b", "b", "a", "a"],
        class_labels=["b", "a"],
    )

    confusion = result["confusion_matrix"]
    per_class = result["per_class_metrics"]
    assert list(confusion.index) == ["b", "a"]
    assert list(confusion.columns) == ["b", "a"]
    assert list(per_class["label"]) == ["b", "a"]


def test_build_evaluation_result_aligns_auc_proba_to_class_labels() -> None:
    from src.classification.evaluator import build_evaluation_result

    result = build_evaluation_result(
        labels=["beta", "alpha", "beta", "alpha"],
        predictions=["beta", "alpha", "beta", "alpha"],
        proba=np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        ),
        class_labels=["beta", "alpha"],
    )

    assert result["metrics"]["roc_auc_ovr"] == 1.0


def test_write_evaluation_outputs_skips_pdf_when_class_count_exceeds_limit(tmp_path) -> None:
    from src.classification.evaluator import (
        build_evaluation_result,
        write_evaluation_outputs,
    )

    labels = [f"cls_{i}" for i in range(51)]
    result = build_evaluation_result(labels=labels, predictions=labels)

    paths = write_evaluation_outputs(result, tmp_path, pdf_class_limit=50)

    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "confusion_matrix_normalized.csv").exists()
    assert (tmp_path / "per_class_metrics.csv").exists()
    assert "confusion_matrix_pdf" not in paths
    assert not (tmp_path / "confusion_matrix.pdf").exists()


def test_evaluate_onnx_preserves_sidecar_label_order(tmp_path, monkeypatch) -> None:
    from src.classification.evaluator import evaluate_onnx

    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"image": ["a.jpg", "b.jpg"], "label": ["beta", "alpha"]}).to_csv(
        test_csv, index=False
    )
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    (tmp_path / "label_classes.json").write_text(
        json.dumps({"class_labels": ["beta", "alpha"]}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "src.classification.predictor.predict_onnx",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "image": ["a.jpg", "b.jpg"],
                "prediction": ["beta", "alpha"],
                "prediction_index": [0, 1],
                "proba_0": [0.9, 0.1],
                "proba_1": [0.1, 0.9],
            }
        ),
    )

    result = evaluate_onnx(
        test_csv=test_csv,
        images_dir=tmp_path,
        onnx_path=onnx_path,
        batch_size=2,
        num_threads=0,
    )

    assert list(result["per_class_metrics"]["label"]) == ["beta", "alpha"]


def test_evaluate_uses_predictor_class_order_when_available(tmp_path, monkeypatch) -> None:
    from src.classification.evaluator import evaluate

    test_csv = tmp_path / "test.csv"
    pd.DataFrame(
        {"image": ["a.jpg", "b.jpg"], "label": ["beta", "alpha"]}
    ).to_csv(test_csv, index=False)

    captured: dict[str, object] = {}

    class FakePredictResult:
        def __init__(self, values) -> None:
            self.values = values

    class FakePredictor:
        classes_ = ["beta", "alpha"]

        def predict(self, _df):
            return FakePredictResult(pd.Series(["beta", "alpha"]).values)

        def predict_proba(self, _df):
            return FakePredictResult(pd.DataFrame([[0.9, 0.1], [0.2, 0.8]]).values)

    class FakeMultiModalPredictor:
        @staticmethod
        def load(_path: str):
            return FakePredictor()

    def fake_build_evaluation_result(**kwargs):
        captured.update(kwargs)
        return {
            "metrics": {},
            "class_labels": list(kwargs["class_labels"]),
            "confusion_matrix": pd.DataFrame(),
            "confusion_matrix_normalized": pd.DataFrame(),
            "per_class_metrics": pd.DataFrame(),
        }

    monkeypatch.setattr("src.classification.evaluator.build_evaluation_result", fake_build_evaluation_result)
    monkeypatch.setattr("src.classification.utils.set_num_threads", lambda _n: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "autogluon.multimodal",
        SimpleNamespace(MultiModalPredictor=FakeMultiModalPredictor),
    )

    result = evaluate(
        test_csv=test_csv,
        images_dir=tmp_path,
        model_dir=tmp_path / "model",
        batch_size=2,
        num_workers=0,
        num_threads=0,
        device="cpu",
    )

    assert list(captured["class_labels"]) == ["beta", "alpha"]
    assert result["class_labels"] == ["beta", "alpha"]
