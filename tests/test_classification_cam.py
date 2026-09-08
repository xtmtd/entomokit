"""Tests for CAM utility helpers and CLI behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


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
        num_threads=0,
        device="auto",
        dump_model_structure=True,
        eval_transform="center-crop",
        overwrite=False,
    )

    cam_cli.run(args)

    assert captured["label_csv"] is None
    assert captured["dump_model_structure"] is True
    assert captured["eval_transform"] == "center-crop"


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
    assert args.eval_transform == "center-crop"


def test_cam_register_accepts_whole_specimen_pad() -> None:
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
            "--eval-transform",
            "whole-specimen-pad",
        ]
    )

    assert args.eval_transform == "whole-specimen-pad"


def test_load_model_from_args_uses_autogluon_classifier_head(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace
    import sys

    class Backbone(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 4))

    class Head(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 1.0]], device=x.device).repeat(x.shape[0], 1)

    backbone = Backbone()
    ag_model = SimpleNamespace(
        model=backbone,
        head=Head(),
        image_size=224,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    )

    class ImageProcessor:
        val_transforms = ["resize_shorter_side", "center_crop"]

        def construct_image_processor(self, transforms, size, normalization):
            self.args = transforms, size, normalization
            return torch.nn.Identity()

    image_processor = ImageProcessor()
    predictor = SimpleNamespace(
        _learner=SimpleNamespace(
            _model=ag_model,
            _data_processors={"image": [image_processor]},
        ),
        class_labels=["gracilis", "tonkinensis"],
    )

    class FakePredictor:
        @staticmethod
        def load(_path):
            return predictor

    monkeypatch.setitem(
        sys.modules,
        "autogluon.multimodal",
        SimpleNamespace(MultiModalPredictor=FakePredictor),
    )

    from src.classification.cam import load_model_from_args

    model = load_model_from_args(
        load_ag=str(tmp_path / "model"),
        base_model=None,
        checkpoint_path=None,
        num_classes=None,
        pretrained=False,
        device=torch.device("cpu"),
    )

    output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 2)
    assert int(output.argmax(dim=1).item()) == 0
    assert model.class_labels == ["gracilis", "tonkinensis"]
    assert model.image_processor is image_processor


def test_infer_architecture_uses_autogluon_backbone() -> None:
    from types import SimpleNamespace

    from src.classification.cam import infer_architecture

    class ViTBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])

    assert infer_architecture("", SimpleNamespace(backbone=ViTBackbone())) == "vit"


def test_prepare_cam_uses_convnext_block_instead_of_pointwise_conv() -> None:
    from src.classification.cam import prepare_cam

    class ConvNeXtBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_dw = torch.nn.Conv2d(8, 8, 3, padding=1)
            self.mlp = torch.nn.Sequential(torch.nn.Conv2d(8, 8, 1))

    class ConvNeXtStage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([ConvNeXtBlock(), ConvNeXtBlock()])

    class ConvNeXt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = torch.nn.ModuleList([ConvNeXtStage(), ConvNeXtStage()])

    model = ConvNeXt()
    _cam, target_layers, _reshape_transform = prepare_cam(
        model, "cnn", None, "gradcam", torch.device("cpu"), 1
    )

    assert target_layers == [model.stages[-1].blocks[-1]]


def test_prepare_cam_keeps_last_conv_fallback_for_plain_cnn() -> None:
    from src.classification.cam import prepare_cam

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 4, 1),
    )
    _cam, target_layers, _reshape_transform = prepare_cam(
        model, "cnn", None, "gradcam", torch.device("cpu"), 1
    )

    assert target_layers == [model[-1]]


def test_prepare_cam_uses_last_swin_block_and_spatial_reshape() -> None:
    from src.classification.cam import infer_architecture, prepare_cam

    class SwinBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(768)

    class SwinStage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([SwinBlock(), SwinBlock()])

    class SwinBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([SwinStage(), SwinStage()])

    model = SwinBackbone()

    assert infer_architecture("", model) == "vit"
    _cam, target_layers, reshape_transform = prepare_cam(
        model, "vit", None, "gradcam", torch.device("cpu"), 1
    )
    ablation_cam, _target_layers, _reshape_transform = prepare_cam(
        model, "vit", None, "ablationcam", torch.device("cpu"), 1
    )

    assert target_layers == [model.layers[-1].blocks[-1].norm1]
    assert reshape_transform(torch.zeros(2, 7, 7, 768)).shape == (2, 768, 7, 7)
    assert ablation_cam.ablation_layer.__class__.__name__ == "AblationLayerSwin"


def test_build_cam_transforms_uses_autogluon_processor_size() -> None:
    from src.classification.cam import _AutoGluonCamModel, build_cam_transforms

    class ImageProcessor:
        val_transforms = ["resize_shorter_side", "center_crop"]

        def construct_image_processor(self, _transforms, size, normalization):
            return transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    normalization,
                ]
            )

        def get_image_transform_funcs(self, _transforms, size):
            return [transforms.Resize(size), transforms.CenterCrop(size)]

    backbone = torch.nn.Identity()
    model = _AutoGluonCamModel(
        type("Model", (), {
            "model": backbone,
            "head": torch.nn.Identity(),
            "image_size": 384,
            "image_mean": (0.1, 0.2, 0.3),
            "image_std": (0.4, 0.5, 0.6),
        })(),
        [],
        ImageProcessor(),
    )
    preprocess, size, resize_size = build_cam_transforms(model, "center-crop")

    assert size == 384
    assert resize_size == 384
    assert preprocess.transforms[0].size == 384
    assert preprocess.transforms[-1].mean == (0.1, 0.2, 0.3)


def test_build_cam_transforms_uses_saved_size_for_whole_specimen_pad() -> None:
    from src.classification.cam import _AutoGluonCamModel, build_cam_transforms

    class ImageProcessor:
        val_transforms = ["resize_to_square"]

        def construct_image_processor(self, _transforms, _size, normalization):
            return transforms.Compose([transforms.ToTensor(), normalization])

        def get_image_transform_funcs(self, _transforms, _size):
            return []

    model = _AutoGluonCamModel(
        type("Model", (), {
            "model": torch.nn.Identity(),
            "head": torch.nn.Identity(),
            "image_size": 384,
            "image_mean": (0.1, 0.2, 0.3),
            "image_std": (0.4, 0.5, 0.6),
        })(),
        [],
        ImageProcessor(),
    )

    _preprocess, size, resize_size = build_cam_transforms(model, "whole-specimen-pad")

    assert size == 384
    assert resize_size == 384


def test_build_cam_transforms_returns_non_ag_resize_size(monkeypatch) -> None:
    from src.classification import cam

    preprocess = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    display = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    monkeypatch.setattr(cam, "build_eval_transforms", lambda _model: (preprocess, display))

    _preprocess, model_size, resize_size = cam.build_cam_transforms(
        torch.nn.Identity(), "center-crop"
    )

    assert model_size == 224
    assert resize_size == 256


def test_map_cam_to_original_preserves_whole_specimen_geometry() -> None:
    from src.classification.cam import map_cam_to_original

    cam = np.zeros((32, 32), dtype=np.float32)
    cam[14:18, 14:18] = 1.0

    mapped, fov = map_cam_to_original(
        cam,
        (100, 200),
        model_size=32,
        resize_size=32,
        eval_transform="whole-specimen-pad",
    )

    assert mapped.shape == (200, 100)
    assert fov.all()
    ys, xs = np.nonzero(mapped > 0.9 * mapped.max())
    assert 90 <= ys.mean() <= 110
    assert 40 <= xs.mean() <= 60


def test_map_cam_to_original_preserves_center_crop_geometry() -> None:
    from src.classification.cam import map_cam_to_original

    cam = np.zeros((32, 32), dtype=np.float32)
    cam[14:18, 14:18] = 1.0

    mapped, fov = map_cam_to_original(
        cam, (100, 200), model_size=32, resize_size=32
    )

    assert mapped.shape == (200, 100)
    assert not fov[20, 50]
    assert fov[100, 50]
    ys, xs = np.nonzero(mapped > 0.9 * mapped.max())
    assert 90 <= ys.mean() <= 110
    assert 40 <= xs.mean() <= 60


def test_map_cam_to_original_uses_resize_size_for_center_crop_fov() -> None:
    from src.classification.cam import map_cam_to_original

    mapped, fov = map_cam_to_original(
        np.ones((224, 224), dtype=np.float32),
        (100, 200),
        model_size=224,
        resize_size=256,
    )

    assert mapped.shape == (200, 100)
    assert not fov[50, 50]
    assert fov[57, 50]
    assert not fov[144, 50]


def test_process_image_does_not_double_scale_uint8_overlay(
    tmp_path: Path, monkeypatch
) -> None:
    from src.classification import cam

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 1.0]], device=x.device).repeat(x.shape[0], 1)

    monkeypatch.setattr(
        cam,
        "show_cam_on_image",
        lambda *_args, **_kwargs: np.full((32, 32, 3), 240, dtype=np.uint8),
    )

    class IdentityTransform:
        def __call__(self, _image):
            return torch.zeros(3, 32, 32)

    cam.process_image(
        img_path=image_path,
        label="x",
        model=FakeModel(),
        preprocess=IdentityTransform(),
        cam_extractor=lambda **_kwargs: [np.ones((32, 32), dtype=np.float32)],
        device=torch.device("cpu"),
        fig_dir=tmp_path,
        array_dir=None,
        image_weight=0.5,
        fig_format="png",
        save_npy=False,
    )

    output = np.asarray(Image.open(tmp_path / "image_cam.png"))
    assert output[:, :, 0].max() >= 230


def test_process_image_records_relative_source_and_class_label(
    tmp_path: Path, monkeypatch
) -> None:
    from src.classification import cam

    image_path = tmp_path / "nested" / "image.png"
    image_path.parent.mkdir()
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)

    class FakeModel(torch.nn.Module):
        class_labels = ["gracilis", "tonkinensis"]

        def forward(self, x):
            return torch.tensor([[1.0, 2.0]], device=x.device).repeat(x.shape[0], 1)

    monkeypatch.setattr(
        cam,
        "show_cam_on_image",
        lambda *_args, **_kwargs: np.full((32, 32, 3), 240, dtype=np.uint8),
    )
    record = cam.process_image(
        img_path=image_path,
        label="x",
        model=FakeModel(),
        preprocess=lambda _image: torch.zeros(3, 32, 32),
        cam_extractor=lambda **_kwargs: [np.ones((32, 32), dtype=np.float32)],
        device=torch.device("cpu"),
        fig_dir=tmp_path,
        array_dir=None,
        image_weight=0.5,
        fig_format="png",
        save_npy=False,
        source_image="nested/image.png",
    )

    assert record["image"] == "nested/image.png"
    assert record["pred_class"] == "tonkinensis"


def test_output_stem_keeps_recursive_inputs_unique() -> None:
    from src.classification.cam import output_stem

    assert output_stem("one/a.jpg") != output_stem("two/a.jpg")


def test_write_model_structure_uses_backbone_relative_layer_names(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from src.classification.cam import write_model_structure

    model = SimpleNamespace(backbone=torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3)))
    content = write_model_structure(model, tmp_path).read_text(encoding="utf-8")

    assert "0\tConv2d" in content
    assert "backbone.0\tConv2d" not in content


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
    monkeypatch.setattr(cam, "build_cam_transforms", lambda *_args: (None, 224, 224))
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
