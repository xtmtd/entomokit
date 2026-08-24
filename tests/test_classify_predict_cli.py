"""Tests for classify predict CLI input resolution."""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
import pytest


def test_classify_predict_parser_accepts_input_csv_and_images_dir_together() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "classify",
            "predict",
            "--input-csv",
            "test.csv",
            "--images-dir",
            "images",
            "--model-dir",
            "model",
            "--out-dir",
            "out",
        ]
    )

    assert args.input_csv == "test.csv"
    assert args.images_dir == "images"


def test_resolve_predict_inputs_uses_images_dir_for_relative_csv_names(
    tmp_path: Path,
) -> None:
    from entomokit.classify import predict as predict_cli

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"fake")

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"image": ["a.jpg"]}).to_csv(input_csv, index=False)

    df, resolved_images_dir = predict_cli._resolve_predict_inputs(
        input_csv=input_csv,
        images_dir=images_dir,
    )

    assert df["image"].tolist() == ["a.jpg"]
    assert resolved_images_dir == images_dir


def test_resolve_predict_inputs_requires_images_dir_when_csv_has_no_paths(
    tmp_path: Path,
) -> None:
    from entomokit.classify import predict as predict_cli

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"image": ["a.jpg"]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="--images-dir"):
        predict_cli._resolve_predict_inputs(input_csv=input_csv, images_dir=None)


def test_resolve_predict_inputs_rejects_redundant_images_dir_for_path_csv(
    tmp_path: Path,
) -> None:
    from entomokit.classify import predict as predict_cli

    csv_image_dir = tmp_path / "csv_images"
    csv_image_dir.mkdir()
    csv_image = csv_image_dir / "a.jpg"
    csv_image.write_bytes(b"fake")

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "b.jpg").write_bytes(b"fake")

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"image": [str(csv_image)]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="already contains readable image paths"):
        predict_cli._resolve_predict_inputs(input_csv=input_csv, images_dir=images_dir)


def test_resolve_predict_inputs_scans_images_dir_without_csv(tmp_path: Path) -> None:
    from entomokit.classify import predict as predict_cli

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "b.jpg").write_bytes(b"fake")
    (images_dir / "a.png").write_bytes(b"fake")
    (images_dir / "note.txt").write_text("x", encoding="utf-8")

    df, resolved_images_dir = predict_cli._resolve_predict_inputs(
        input_csv=None,
        images_dir=images_dir,
    )

    assert df["image"].tolist() == ["a.png", "b.jpg"]
    assert resolved_images_dir == images_dir


def test_predict_run_writes_missing_images_log(tmp_path: Path, monkeypatch) -> None:
    from entomokit.classify import predict as predict_cli

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"image": ["missing.jpg"]}).to_csv(input_csv, index=False)
    (tmp_path / "images").mkdir()

    out_dir = tmp_path / "out"

    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)

    args = argparse.Namespace(
        input_csv=str(input_csv),
        images_dir=str(tmp_path / "images"),
        model_dir="model",
        onnx_model=None,
        out_dir=str(out_dir),
        batch_size=32,
        num_workers=4,
        num_threads=0,
        device="auto",
        overwrite=False,
        recursive=False,
    )

    with pytest.raises(SystemExit) as exc:
        predict_cli.run(args)

    assert exc.value.code == 2
    missing_log = out_dir / "logs" / "missing_images.txt"
    assert missing_log.exists()
    assert missing_log.read_text(encoding="utf-8").splitlines() == ["missing.jpg"]


def test_predict_run_onnx_enables_progress_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from entomokit.classify import predict as predict_cli

    out_dir = tmp_path / "out"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.classification.utils.select_device",
        lambda _device: type("Dev", (), {"type": "cpu"})(),
    )

    df = pd.DataFrame({"image": ["a.jpg"]})
    monkeypatch.setattr(
        "entomokit.classify.predict._resolve_predict_inputs",
        lambda **_kwargs: (df, images_dir),
    )

    captured: dict[str, object] = {}

    def fake_predict_onnx(*_args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "image": ["a.jpg"],
                "prediction": [0],
                "proba_0": [1.0],
            }
        )

    monkeypatch.setattr("src.classification.predictor.predict_onnx", fake_predict_onnx)

    args = argparse.Namespace(
        input_csv=None,
        images_dir=str(images_dir),
        model_dir=None,
        onnx_model=str(tmp_path / "model.onnx"),
        out_dir=str(out_dir),
        batch_size=32,
        num_workers=4,
        num_threads=0,
        device="auto",
        overwrite=False,
        recursive=False,
    )

    predict_cli.run(args)

    assert captured["show_progress"] is True
