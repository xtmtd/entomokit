"""Tests for CAM utility helpers and CLI behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def test_prepare_output_dirs_skips_arrays_when_save_npy_false(tmp_path: Path) -> None:
    from src.classification.cam import prepare_output_dirs

    out_dirs = prepare_output_dirs(tmp_path, save_npy=False)

    assert (tmp_path / "figures").is_dir()
    assert not (tmp_path / "arrays").exists()
    assert out_dirs["array"] is None


def test_collect_image_label_rows_uses_images_dir_without_label_csv(
    tmp_path: Path,
) -> None:
    from src.classification.cam import collect_image_label_rows

    for name in ["a.jpg", "nested/b.png", "nested/c.txt"]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    rows = collect_image_label_rows(images_dir=tmp_path, label_csv=None)

    assert list(rows.columns) == ["image", "label"]
    assert rows["image"].tolist() == ["a.jpg", "nested/b.png"]
    assert rows["label"].tolist() == ["", ""]


def test_run_accepts_missing_label_csv_and_forwards_dump_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from entomokit.classify import cam as cam_cli

    captured: dict[str, object] = {}

    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.classification.utils.select_device",
        lambda _device: type("_D", (), {"type": "cpu"})(),
    )
    monkeypatch.setattr("src.classification.utils.set_num_threads", lambda *_args: None)

    def _fake_run_cam(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("src.classification.cam.run_cam", _fake_run_cam)

    args = argparse.Namespace(
        label_csv=None,
        images_dir=str(tmp_path / "images"),
        out_dir=str(tmp_path / "out"),
        model_dir=str(tmp_path / "model"),
        base_model=None,
        checkpoint_path=None,
        num_classes=2,
        no_pretrained=False,
        cam_method="scorecam",
        arch=None,
        target_layer_name=None,
        image_weight=0.5,
        fig_format="png",
        save_npy=False,
        max_images=None,
        cam_batch_size=8,
        num_workers=0,
        num_threads=0,
        device="auto",
        dump_model_structure=True,
        overwrite=False,
    )

    cam_cli.run(args)

    assert captured["label_csv"] is None
    assert captured["dump_model_structure"] is True


def test_cam_register_allows_missing_label_csv() -> None:
    from entomokit.classify import cam as cam_cli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cam_cli.register(sub)

    args = parser.parse_args(
        [
            "cam",
            "--images-dir",
            "images",
            "--out-dir",
            "out",
            "--model-dir",
            "model",
        ]
    )

    assert args.label_csv is None


def test_run_cam_dump_model_structure_writes_layers_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.classification import cam

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, kernel_size=3), torch.nn.ReLU())

    monkeypatch.setattr(
        cam,
        "collect_image_label_rows",
        lambda **_kwargs: pd.DataFrame({"image": [], "label": []}),
    )
    monkeypatch.setattr(cam, "load_model_from_args", lambda **_kwargs: model)
    monkeypatch.setattr(cam, "build_eval_transforms", lambda _m: (None, None))
    monkeypatch.setattr(
        cam,
        "prepare_cam",
        lambda *_args, **_kwargs: (lambda **_k: None, [model[0]], None),
    )

    cam.run_cam(
        label_csv=None,
        images_dir=tmp_path,
        out_dir=tmp_path,
        model_dir=tmp_path / "model",
        base_model=None,
        checkpoint_path=None,
        num_classes=2,
        pretrained=True,
        cam_method="scorecam",
        arch="cnn",
        target_layer_name=None,
        image_weight=0.5,
        fig_format="png",
        save_npy=False,
        dump_model_structure=True,
        max_images=None,
        cam_batch_size=8,
        device=torch.device("cpu"),
    )

    layers_file = tmp_path / "model_layers.txt"
    assert layers_file.exists()
    content = layers_file.read_text(encoding="utf-8")
    assert "# Named modules for --target-layer-name" in content
