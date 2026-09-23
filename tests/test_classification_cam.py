"""Tests for CAM utility helpers and CLI behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image


def test_prepare_output_dirs_skips_arrays_when_save_npy_none(tmp_path: Path) -> None:
    from src.classification.cam import prepare_output_dirs

    out_dirs = prepare_output_dirs(tmp_path, save_npy="none")

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
        save_npy="none",
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
        save_npy="none",
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
        save_npy="none",
        source_image="nested/image.png",
    )

    assert record["image"] == "nested/image.png"
    assert record["pred_class"] == "tonkinensis"


@pytest.mark.parametrize(
    "save_npy, arrays_expected", [("raw", True), ("none", False)]
)
def test_run_cam_logs_where_outputs_were_written(
    tmp_path: Path, monkeypatch, caplog, save_npy: str, arrays_expected: bool
) -> None:
    """The run must report on screen where figures, summary and arrays went.

    The arrays line is conditional, so both branches are covered.
    """
    import logging

    from src.classification import cam

    caplog.set_level(logging.INFO)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "a.png")

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, kernel_size=3), torch.nn.ReLU())
    monkeypatch.setattr(
        cam,
        "collect_image_label_rows",
        lambda **_kwargs: pd.DataFrame({"image": ["a.png"], "label": ["x"]}),
    )
    monkeypatch.setattr(cam, "load_model_from_args", lambda **_kwargs: model)
    monkeypatch.setattr(cam, "build_cam_transforms", lambda *_args: (None, 224, 224))
    monkeypatch.setattr(
        cam,
        "prepare_cam",
        lambda *_args, **_kwargs: (lambda **_k: None, [model[0]], None),
    )
    monkeypatch.setattr(
        cam,
        "process_image",
        lambda **_kwargs: {
            "image": "a.png",
            "label": "x",
            "pred_class": "x",
            "pred_prob": 0.9,
            "figure_path": "f",
            "cam_array_path": "",
        },
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
        arch=None,
        target_layer_name=None,
        image_weight=0.5,
        fig_format="png",
        save_npy=save_npy,
        dump_model_structure=False,
        max_images=None,
        cam_batch_size=8,
        device=torch.device("cpu"),
    )

    assert f"CAM heatmaps written to: {tmp_path / 'figures'}" in caplog.text
    assert f"Summary: {tmp_path / 'cam_summary.csv'}" in caplog.text
    assert ("CAM arrays written to:" in caplog.text) is arrays_expected
    if arrays_expected:
        assert f"({save_npy})" in caplog.text


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
        save_npy="none",
        dump_model_structure=True,
        max_images=None,
        cam_batch_size=8,
        device=torch.device("cpu"),
    )

    layers_file = tmp_path / "model_layers.txt"
    assert layers_file.exists()
    content = layers_file.read_text(encoding="utf-8")
    assert "# Named modules for --target-layer-name" in content


def _cam_args(extra: list[str]) -> argparse.Namespace:
    from entomokit.classify import cam as cam_cli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cam_cli.register(sub)
    return parser.parse_args(
        ["cam", "--images-dir", "images", "--out-dir", "out",
         "--model-dir", "model", *extra]
    )


def test_cam_save_npy_defaults_to_none() -> None:
    assert _cam_args([]).save_npy == "none"


def test_cam_save_npy_accepts_explicit_values() -> None:
    assert _cam_args(["--save-npy", "none"]).save_npy == "none"
    assert _cam_args(["--save-npy", "raw"]).save_npy == "raw"
    assert _cam_args(["--save-npy", "normalized"]).save_npy == "normalized"


def test_cam_save_npy_requires_a_value() -> None:
    with pytest.raises(SystemExit):
        _cam_args(["--save-npy"])


def test_cam_save_npy_rejects_unknown_value() -> None:
    with pytest.raises(SystemExit):
        _cam_args(["--save-npy", "bogus"])


CAM_METHOD_NAMES = (
    "ablationcam",
    "eigencam",
    "gradcam",
    "gradcampp",
    "layercam",
    "scorecam",
)


def _tiny_conv_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 3),
    )


def _fixed_cam_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 3, 96, 96) * 2 - 1


def test_cam_methods_all_use_raw_classes() -> None:
    from src.classification.cam import CAM_METHODS, UnnormalizedCAMMixin

    assert sorted(CAM_METHODS) == list(CAM_METHOD_NAMES)
    for name, cls in CAM_METHODS.items():
        assert issubclass(cls, UnnormalizedCAMMixin), name


@pytest.mark.parametrize("name", CAM_METHOD_NAMES)
def test_every_cam_method_returns_unnormalized_map(name: str) -> None:
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.classification.cam import CAM_METHODS

    torch.manual_seed(0)
    model = _tiny_conv_model().eval()
    cam = CAM_METHODS[name](model=model, target_layers=[model[4]])

    raw = cam(input_tensor=_fixed_cam_input(), targets=[ClassifierOutputTarget(1)])[0]

    # Structural invariants: no value here is an acceptance criterion.
    assert raw.shape == (96, 96)
    assert raw.dtype == np.float32
    assert np.isfinite(raw).all()
    assert raw.min() >= 0.0
    # Positivity holds for this fixture, not for CAM methods in general: a fully
    # ReLU'd-zero map is legal, and eigencam is additionally exposed to the SVD sign
    # (D5). This is the only magnitude-related assertion in the suite and it is a
    # fixture-scoped diagnostic, not acceptance evidence. The binding evidence is:
    # test_cam_methods_all_use_raw_classes (per-method mapping),
    # test_raw_cam_does_not_call_scale_cam_image (structural), and
    # test_process_image_saves_raw_array_by_default (save path, test-owned array).
    assert raw.max() > 0.0


def test_raw_cam_does_not_call_scale_cam_image(monkeypatch) -> None:
    """Structural proof: the raw path never reaches upstream's normalization.

    The official class must raise under the same patch, otherwise this test
    would pass vacuously.
    """
    import pytorch_grad_cam.base_cam as base_cam_module
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.classification.cam import CAM_METHODS

    def _boom(*_args, **_kwargs):
        raise AssertionError("scale_cam_image must not be called")

    monkeypatch.setattr(base_cam_module, "scale_cam_image", _boom)
    tensor = _fixed_cam_input()
    targets = [ClassifierOutputTarget(1)]

    torch.manual_seed(0)
    official_model = _tiny_conv_model().eval()
    with pytest.raises(AssertionError):
        GradCAM(model=official_model, target_layers=[official_model[4]])(
            input_tensor=tensor, targets=targets
        )

    torch.manual_seed(0)
    raw_model = _tiny_conv_model().eval()
    raw = CAM_METHODS["gradcam"](model=raw_model, target_layers=[raw_model[4]])(
        input_tensor=tensor, targets=targets
    )[0]

    assert raw.shape == (96, 96)


def test_raw_cam_agrees_with_official_cam_after_min_max() -> None:
    """Protect display equivalence; does not pin any CAM magnitude.

    Catches: changed resize, changed ReLU policy, changed aggregation, a new
    non-affine upstream normalization, and an upstream that stops normalizing
    (via the official.max() check). Does NOT catch a pure magnitude scaling of
    raw CAMs, which min-max would hide.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.classification.cam import CAM_METHODS

    torch.manual_seed(0)
    model = _tiny_conv_model().eval()
    tensor = _fixed_cam_input()
    targets = [ClassifierOutputTarget(1)]

    official = GradCAM(model=model, target_layers=[model[4]])(
        input_tensor=tensor, targets=targets
    )[0]
    raw = CAM_METHODS["gradcam"](model=model, target_layers=[model[4]])(
        input_tensor=tensor, targets=targets
    )[0]

    display = raw - raw.min()
    display = display / display.max()

    # min-max is affine-invariant and cv2.INTER_LINEAR is affine, so both agree to
    # float32 precision. A new non-affine step upstream, a changed resize, a changed
    # ReLU policy, or a different aggregation breaks this.
    assert official.max() == pytest.approx(1.0, abs=1e-6)
    assert np.abs(display - official).max() < 1e-6


@pytest.mark.parametrize("name", ["gradcam", "scorecam"])
def test_overlay_figure_matches_between_official_and_raw_cam(
    name: str,
    tmp_path: Path,
) -> None:
    """The rendered overlay must not change beyond uint8 quantization.

    Coverage is gradcam and scorecam. `scorecam` is included deliberately: it is
    the only method whose overlay measurably differs (44 of 73728 pixels by at
    most 2 LSB on this fixture), so a gradcam-only check could not detect the
    regression this test exists for. The other four methods keep their
    development-time measurements only.
    """
    from pytorch_grad_cam import GradCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.classification import cam as cam_mod
    from src.classification.cam import CAM_METHODS

    official_cls = {"gradcam": GradCAM, "scorecam": ScoreCAM}[name]
    tensor = _fixed_cam_input()
    targets = [ClassifierOutputTarget(1)]

    def render(cls, stem: str) -> np.ndarray:
        image_path = tmp_path / f"{stem}.png"
        Image.new("RGB", (128, 96), (200, 180, 160)).save(image_path)
        torch.manual_seed(0)
        model = _tiny_conv_model().eval()
        extractor = cls(model=model, target_layers=[model[4]])
        record = cam_mod.process_image(
            img_path=image_path,
            label="x",
            model=model,
            preprocess=lambda _image: tensor[0],
            cam_extractor=lambda **kwargs: extractor(**kwargs),
            device=torch.device("cpu"),
            fig_dir=tmp_path,
            array_dir=None,
            image_weight=0.5,
            fig_format="png",
            save_npy="none",
            model_size=96,
            resize_size=96,
        )
        return np.asarray(Image.open(record["figure_path"])).astype(np.int16)

    difference = np.abs(
        render(official_cls, f"{name}-official") - render(CAM_METHODS[name], f"{name}-raw")
    )

    assert difference.max() <= 4
    assert (difference > 0).mean() < 0.01


FAKE_CAM = np.array([[0.0, 2.0], [1.0, 0.5]], dtype=np.float32)


def _run_process_image(tmp_path: Path, monkeypatch, save_npy: str):
    from src.classification import cam

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)
    array_dir = tmp_path / "arrays"
    array_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(
        cam,
        "show_cam_on_image",
        lambda *_a, **_k: np.full((32, 32, 3), 240, dtype=np.uint8),
    )

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 1.0]], device=x.device).repeat(x.shape[0], 1)

    return cam.process_image(
        img_path=image_path,
        label="x",
        model=FakeModel(),
        preprocess=lambda _image: torch.zeros(3, 32, 32),
        cam_extractor=lambda **_k: [FAKE_CAM.copy()],
        device=torch.device("cpu"),
        fig_dir=tmp_path,
        array_dir=array_dir,
        image_weight=0.5,
        fig_format="png",
        save_npy=save_npy,
    )


def test_process_image_saves_raw_array_by_default(tmp_path: Path, monkeypatch) -> None:
    record = _run_process_image(tmp_path, monkeypatch, "raw")

    saved = np.load(tmp_path / "arrays" / "image.npy")
    assert saved.dtype == np.float32
    np.testing.assert_allclose(saved, FAKE_CAM)
    assert saved.max() != pytest.approx(1.0)
    assert record["cam_array_path"].endswith("image.npy")


def test_process_image_saves_normalized_array_on_request(
    tmp_path: Path, monkeypatch
) -> None:
    _run_process_image(tmp_path, monkeypatch, "normalized")

    saved = np.load(tmp_path / "arrays" / "image.npy")
    np.testing.assert_allclose(saved, np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32))


def test_process_image_writes_no_array_when_disabled(tmp_path: Path, monkeypatch) -> None:
    record = _run_process_image(tmp_path, monkeypatch, "none")

    assert record["cam_array_path"] == ""
    assert list((tmp_path / "arrays").iterdir()) == []


def test_overlay_receives_normalized_mask_not_raw(
    tmp_path: Path, monkeypatch
) -> None:
    """The overlay must always receive the normalized, mapped copy.

    Measured on this fixture: the captured mask is `(32, 32)` with min/max
    `0.0 / 1.0`, and it matches `map_cam_to_original(min-max(FAKE_CAM), ...)`
    exactly. `FAKE_CAM` peaks at `2.0`, so if the raw array leaked into the
    overlay the mapped mask would peak at `2.0` and both the bound and the
    reference comparison below would fail. Note the mask is at the original
    image size, not `FAKE_CAM`'s `2x2`, so it must be compared against a
    same-shaped reference.
    """
    from src.classification import cam

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)
    array_dir = tmp_path / "arrays"
    array_dir.mkdir()

    captured: dict[str, np.ndarray] = {}

    def _capture(_rgb, mask, **_kwargs):
        captured["mask"] = np.asarray(mask).copy()
        return np.full((32, 32, 3), 240, dtype=np.uint8)

    monkeypatch.setattr(cam, "show_cam_on_image", _capture)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 1.0]], device=x.device).repeat(x.shape[0], 1)

    cam.process_image(
        img_path=image_path,
        label="x",
        model=FakeModel(),
        preprocess=lambda _image: torch.zeros(3, 32, 32),
        cam_extractor=lambda **_kwargs: [FAKE_CAM.copy()],
        device=torch.device("cpu"),
        fig_dir=tmp_path,
        array_dir=array_dir,
        image_weight=0.5,
        fig_format="png",
        save_npy="raw",
        model_size=32,
        resize_size=32,
    )

    display = FAKE_CAM - FAKE_CAM.min()
    display = display / display.max()
    expected, _fov = cam.map_cam_to_original(
        display, (32, 32), model_size=32, resize_size=32
    )

    mask = captured["mask"]
    assert mask.shape == (32, 32)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    np.testing.assert_allclose(mask, expected)


def test_process_image_writes_figure_under_encoded_nested_name(
    tmp_path: Path, monkeypatch
) -> None:
    from src.classification import cam

    image_path = tmp_path / "nested" / "image.png"
    image_path.parent.mkdir()
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()

    class FakeModel(torch.nn.Module):
        class_labels = ["a", "b"]

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
        fig_dir=fig_dir,
        array_dir=None,
        image_weight=0.5,
        fig_format="png",
        save_npy="none",
        output_name=cam.output_stem("nested/image.png"),
        source_image="nested/image.png",
    )

    assert (fig_dir / "nested__image_cam.png").exists()
    assert record["figure_path"].endswith("nested__image_cam.png")
