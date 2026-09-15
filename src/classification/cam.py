"""GradCAM heatmap generation for CNN and ViT backbones.

Supports timm backbones and AutoGluon MultiModalPredictor checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    EigenCAM,
    GradCAMPlusPlus,
    LayerCAM,
    AblationCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm.data import resolve_model_data_config
from torchvision.transforms.functional import InterpolationMode
import cv2


class UnnormalizedCAMMixin:
    """Keep CAM magnitude by skipping BaseCAM's scale_cam_image calls.

    Mirrors pytorch-grad-cam 1.5.5 ``BaseCAM.compute_cam_per_layer`` and
    ``BaseCAM.aggregate_multi_layers``. ReLU, resize-to-input-size and the
    per-layer mean are identical to upstream; only the two min-max
    normalizations are removed.

    ``test_raw_cam_agrees_with_official_cam_after_min_max`` detects upstream
    drift: if grad-cam adds a non-affine step, changes its resize, or changes
    its ReLU or aggregation policy, that test fails.
    """

    @staticmethod
    def _resize_batch(cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        return np.stack(
            [
                cv2.resize(
                    np.float32(image), target_size, interpolation=cv2.INTER_LINEAR
                )
                for image in cam
            ]
        )

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets, eigen_smooth: bool
    ) -> list:
        if self.detach:
            activations_list = [
                activation.cpu().data.numpy()
                for activation in self.activations_and_grads.activations
            ]
            grads_list = [
                gradient.cpu().data.numpy()
                for gradient in self.activations_and_grads.gradients
            ]
        else:
            activations_list = list(self.activations_and_grads.activations)
            grads_list = list(self.activations_and_grads.gradients)
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for index, target_layer in enumerate(self.target_layers):
            activations = (
                activations_list[index] if index < len(activations_list) else None
            )
            gradients = grads_list[index] if index < len(grads_list) else None
            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                activations,
                gradients,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            cam_per_target_layer.append(self._resize_batch(cam, target_size)[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: list) -> np.ndarray:
        stacked = np.concatenate(cam_per_target_layer, axis=1)
        stacked = np.maximum(stacked, 0)
        return np.mean(stacked, axis=1)


class RawGradCAM(UnnormalizedCAMMixin, GradCAM):
    pass


class RawGradCAMPlusPlus(UnnormalizedCAMMixin, GradCAMPlusPlus):
    pass


class RawLayerCAM(UnnormalizedCAMMixin, LayerCAM):
    pass


class RawScoreCAM(UnnormalizedCAMMixin, ScoreCAM):
    pass


class RawEigenCAM(UnnormalizedCAMMixin, EigenCAM):
    pass


class RawAblationCAM(UnnormalizedCAMMixin, AblationCAM):
    pass


CAM_METHODS = {
    "gradcam": RawGradCAM,
    "gradcampp": RawGradCAMPlusPlus,
    "layercam": RawLayerCAM,
    "ablationcam": RawAblationCAM,
    "scorecam": RawScoreCAM,
    "eigencam": RawEigenCAM,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SaveMode = Literal["none", "raw", "normalized"]


class AblationLayerSwin(AblationLayerVit):
    """Ablate channels in channel-last Swin activations (B, H, W, C)."""

    def set_next_batch(
        self, input_batch_index: int, activations: torch.Tensor, num_channels_to_ablate: int
    ) -> None:
        self.activations = activations[input_batch_index].clone().unsqueeze(0).repeat(
            num_channels_to_ablate, *([1] * (activations.ndim - 1))
        )


def load_label_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns named 'image' and 'label'.")
    return df[["image", "label"]]


def collect_image_label_rows(
    *,
    images_dir: Path,
    label_csv: Optional[Path],
) -> pd.DataFrame:
    if label_csv is not None:
        return load_label_file(label_csv)

    images = [
        p.relative_to(images_dir).as_posix()
        for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return pd.DataFrame({"image": images, "label": ["" for _ in images]})


class _AutoGluonCamModel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        class_labels: list[str],
        image_processor,
    ):
        super().__init__()
        self.backbone = model.model
        self.head = model.head
        self.image_size = model.image_size
        self.image_mean = model.image_mean
        self.image_std = model.image_std
        self.class_labels = class_labels
        self.image_processor = image_processor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_model_from_args(
    load_ag: Optional[str],
    base_model: Optional[str],
    checkpoint_path: Optional[str],
    num_classes: Optional[int],
    pretrained: bool,
    device: torch.device,
) -> torch.nn.Module:
    if load_ag:
        try:
            from autogluon.multimodal import MultiModalPredictor
        except ImportError as exc:
            raise ImportError(
                "AutoGluon is not installed. pip install autogluon.multimodal"
            ) from exc
        predictor = MultiModalPredictor.load(load_ag)
        torch_model = _AutoGluonCamModel(
            predictor._learner._model,
            [str(label) for label in predictor.class_labels],
            predictor._learner._data_processors["image"][0],
        )
        logging.info("Loaded AutoGluon classifier from %s", load_ag)
    else:
        if base_model is None:
            raise ValueError("Either model_dir or base_model must be provided.")
        import timm

        torch_model = timm.create_model(
            base_model,
            pretrained=pretrained,
            num_classes=num_classes if num_classes is not None else None,
        )
        logging.info(
            "Instantiated timm backbone %s (pretrained=%s, num_classes=%s)",
            base_model,
            pretrained,
            num_classes,
        )
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            torch_model.load_state_dict(state_dict, strict=False)
            logging.info("Loaded checkpoint weights from %s", checkpoint_path)
    torch_model.eval().to(device)
    return torch_model


def build_eval_transforms(
    model: torch.nn.Module,
) -> Tuple[transforms.Compose, transforms.Compose]:
    if isinstance(model, _AutoGluonCamModel):
        processor = model.image_processor
        preprocess = getattr(processor, "val_processor", None)
        if preprocess is None:
            preprocess = processor.construct_image_processor(
                processor.val_transforms,
                model.image_size,
                transforms.Normalize(model.image_mean, model.image_std),
            )
        geometric = processor.get_image_transform_funcs(
            processor.val_transforms, model.image_size
        )
        return preprocess, transforms.Compose(geometric)
    if all(hasattr(model, attr) for attr in ("image_size", "image_mean", "image_std")):
        crop_size = model.image_size
        mean = model.image_mean
        std = model.image_std
        resize_shorter = crop_size
        interpolation = InterpolationMode.BICUBIC
    else:
        cfg = resolve_model_data_config(model)
        crop_tuple = cfg.get("test_input_size", cfg.get("input_size"))
        crop_size = crop_tuple[1]
        crop_pct = cfg.get("test_crop_pct", cfg.get("crop_pct", 1.0))
        resize_shorter = int(round(crop_size / crop_pct))
        mean = cfg.get("mean")
        std = cfg.get("std")
        interpolation_name = cfg.get("interpolation", "bicubic").upper()
        interpolation = getattr(
            InterpolationMode, interpolation_name, InterpolationMode.BICUBIC
        )
    geometric = [
        transforms.Resize(resize_shorter, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
    ]
    return (
        transforms.Compose(geometric + [transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose(geometric),
    )


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    module = model
    for attr in name.split("."):
        if attr.isdigit():
            idx = int(attr)
            try:
                module = module[idx]
            except Exception as exc:
                raise AttributeError(
                    f"Module '{module.__class__.__name__}' has no index '{attr}'"
                ) from exc
            continue
        if isinstance(module, torch.nn.ModuleDict) and attr in module:
            module = module[attr]
            continue
        if not hasattr(module, attr):
            raise AttributeError(
                f"Module '{module.__class__.__name__}' has no attribute '{attr}'"
            )
        module = getattr(module, attr)
    return module


def find_last_conv_module(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for _, module in getattr(model, "backbone", model).named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found. Specify --target-layer-name manually."
        )
    return last_conv


def default_cnn_target(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "stages") and len(backbone.stages) > 0:
        stage = backbone.stages[-1]
        if hasattr(stage, "blocks") and len(stage.blocks) > 0:
            return stage.blocks[-1]
    return find_last_conv_module(model)


def is_swin_backbone(model: torch.nn.Module) -> bool:
    backbone = getattr(model, "backbone", model)
    return "swin" in backbone.__class__.__name__.lower() or (
        hasattr(backbone, "layers")
        and len(backbone.layers) > 0
        and hasattr(backbone.layers[-1], "blocks")
    )


def default_swin_target(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "layers") and len(backbone.layers) > 0:
        stage = backbone.layers[-1]
        if hasattr(stage, "blocks") and len(stage.blocks) > 0:
            block = stage.blocks[-1]
            return getattr(block, "norm1", block)
    raise RuntimeError(
        "Could not automatically find a Swin block. Specify --target-layer-name."
    )


def default_vit_target(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        block = backbone.blocks[-1]
        for candidate in ["norm1", "ln1", "ln"]:
            if hasattr(block, candidate):
                return getattr(block, candidate)
        # fallback to block itself
        return block
    raise RuntimeError(
        "Could not automatically find a ViT block. Specify --target-layer-name."
    )


def infer_architecture(base_model_name: str, model: torch.nn.Module) -> str:
    backbone = getattr(model, "backbone", model)
    name = (base_model_name or backbone.__class__.__name__).lower()
    if "swin" in name or is_swin_backbone(backbone):
        return "vit"
    if "vit" in name or "transformer" in name or hasattr(backbone, "blocks"):
        return "vit"
    return "cnn"


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Convert Swin (B, H, W, C) or (B, N, C) activations to (B, C, H, W)."""
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        spatial_dim = int(tokens**0.5)
        if spatial_dim * spatial_dim == tokens:
            return tensor.permute(0, 2, 1).reshape(
                batch, channels, spatial_dim, spatial_dim
            )
    raise ValueError(
        f"Expected Swin activations (B, H, W, C) or (B, N, C). Got shape {tensor.shape}."
    )


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT tokens (B, N, C) into feature maps (B, C, H, W)."""
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected ViT token tensor (B, N, C). Got shape {tensor.shape}."
        )
    tensor = tensor[:, 1:, :]  # drop CLS token
    batch, tokens, channels = tensor.shape
    spatial_dim = int(tokens**0.5)
    if spatial_dim * spatial_dim != tokens:
        raise ValueError("Token count cannot form a square grid.")
    tensor = tensor.permute(0, 2, 1).reshape(batch, channels, spatial_dim, spatial_dim)
    return tensor


def prepare_cam(
    model: torch.nn.Module,
    arch: str,
    target_layer_name: Optional[str],
    cam_name: str,
    device: torch.device,
    cam_batch_size: int,
) -> Tuple:
    if target_layer_name:
        target_layers = [get_module_by_name(getattr(model, "backbone", model), target_layer_name)]
    else:
        target_layers = (
            [default_cnn_target(model)]
            if arch == "cnn"
            else [default_swin_target(model)]
            if is_swin_backbone(model)
            else [default_vit_target(model)]
        )

    reshape_transform = (
        swin_reshape_transform
        if arch == "vit" and is_swin_backbone(model)
        else vit_reshape_transform
        if arch == "vit"
        else None
    )
    cam_kwargs = {
        "model": model,
        "target_layers": target_layers,
        "reshape_transform": reshape_transform,
    }
    if cam_name == "ablationcam" and arch == "vit" and is_swin_backbone(model):
        cam_kwargs["ablation_layer"] = AblationLayerSwin()
    elif cam_name == "ablationcam" and arch == "vit":
        cam_kwargs["ablation_layer"] = AblationLayerVit()
    cam = CAM_METHODS[cam_name](**cam_kwargs)
    if isinstance(cam, BaseCAM):
        cam.batch_size = cam_batch_size  # ScoreCAM/EigenCAM
    return cam, target_layers, reshape_transform


def map_cam_to_original(
    cam_display: np.ndarray,
    original_size: tuple[int, int],
    *,
    model_size: int,
    resize_size: Optional[int] = None,
    eval_transform: str = "center-crop",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the original-coordinate CAM and the model field-of-view mask."""
    width, height = original_size
    resize_size = resize_size or model_size
    cam_display = cv2.resize(cam_display, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
    if eval_transform == "whole-specimen-pad":
        side = max(width, height)
        padded = cv2.resize(cam_display, (side, side), interpolation=cv2.INTER_LINEAR)
        if width >= height:
            top = (side - height) // 2
            mapped = padded[top : top + height, :]
        else:
            left = (side - width) // 2
            mapped = padded[:, left : left + width]
        return cv2.resize(mapped, (width, height), interpolation=cv2.INTER_LINEAR), np.ones(
            (height, width), dtype=bool
        )
    if eval_transform != "center-crop":
        raise ValueError("eval_transform must be 'center-crop' or 'whole-specimen-pad'.")

    if width >= height:
        resized_width, resized_height = int(resize_size * width / height), resize_size
    else:
        resized_width, resized_height = resize_size, int(resize_size * height / width)
    left = int(round((resized_width - model_size) / 2))
    top = int(round((resized_height - model_size) / 2))
    canvas = np.zeros((resized_height, resized_width), dtype=np.float32)
    fov = np.zeros((resized_height, resized_width), dtype=bool)
    canvas[top : top + model_size, left : left + model_size] = cam_display
    fov[top : top + model_size, left : left + model_size] = True
    mapped = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_LINEAR)
    fov = cv2.resize(fov.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return mapped, fov > 0.5


def _pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    pixels = np.asarray(image.convert("RGB"))
    edges = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    fill = tuple(np.median(edges, axis=0).astype(np.uint8).tolist())
    side = max(width, height)
    result = Image.new("RGB", (side, side), fill)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


def build_cam_transforms(
    model: torch.nn.Module, eval_transform: str
) -> Tuple[transforms.Compose, int, int]:
    preprocess, _display = build_eval_transforms(model)
    if isinstance(model, _AutoGluonCamModel):
        crop_size = model.image_size
    else:
        crop_size = preprocess.transforms[1].size
        crop_size = crop_size[0] if isinstance(crop_size, tuple) else crop_size
    if eval_transform == "center-crop":
        resize_size = preprocess.transforms[0].size
        resize_size = resize_size[0] if isinstance(resize_size, tuple) else resize_size
        return preprocess, crop_size, resize_size
    if eval_transform == "whole-specimen-pad":
        normalize = preprocess.transforms[-1]
        geometric = transforms.Compose(
            [
                transforms.Lambda(_pad_to_square),
                transforms.Resize((crop_size, crop_size), interpolation=InterpolationMode.BICUBIC),
            ]
        )
        return (
            transforms.Compose(geometric.transforms + [transforms.ToTensor(), normalize]),
            crop_size,
            crop_size,
        )
    raise ValueError("eval_transform must be 'center-crop' or 'whole-specimen-pad'.")


def output_stem(image: str) -> str:
    return Path(image).with_suffix("").as_posix().replace("/", "__")


def prepare_output_dirs(out_dir: Path, save_npy: SaveMode) -> Dict[str, Optional[Path]]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    array_dir: Optional[Path] = None
    if save_npy != "none":
        array_dir = out_dir / "arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
    return {"fig": fig_dir, "array": array_dir}


def write_model_structure(model: torch.nn.Module, out_dir: Path) -> Path:
    path = out_dir / "model_layers.txt"
    lines = [
        "# Named modules for --target-layer-name",
        "# Format: <name>\t<class>",
    ]
    for name, module in getattr(model, "backbone", model).named_modules():
        module_name = name if name else "<root>"
        lines.append(f"{module_name}\t{module.__class__.__name__}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def process_image(
    img_path: Path,
    label: str,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    cam_extractor,
    device: torch.device,
    fig_dir: Path,
    array_dir: Optional[Path],
    image_weight: float,
    fig_format: str,
    save_npy: SaveMode,
    eval_transform: str = "center-crop",
    model_size: int = 224,
    resize_size: Optional[int] = None,
    output_name: Optional[str] = None,
    source_image: Optional[str] = None,
) -> Dict[str, str]:
    pil_img = Image.open(img_path).convert("RGB")
    original_img = pil_img.copy()
    rgb_display = np.array(original_img).astype(np.float32) / 255.0

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits, dim=1)
        pred_score = float(probs[0, pred_idx].cpu())

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

    # Display copy: show_cam_on_image needs [0, 1]. The saved array keeps the
    # unnormalized CAM unless save_npy == "normalized".
    cam_display = grayscale_cam - grayscale_cam.min()
    if cam_display.max() > 0:
        cam_display = cam_display / cam_display.max()
    else:
        cam_display = np.zeros_like(cam_display)

    cam_on_full, fov_mask = map_cam_to_original(
        cam_display,
        original_img.size,
        model_size=model_size,
        resize_size=resize_size,
        eval_transform=eval_transform,
    )

    overlay = show_cam_on_image(
        rgb_display,
        cam_on_full,
        use_rgb=True,
        image_weight=image_weight,
    )
    overlay = overlay.astype(np.float32) / 255.0
    if not fov_mask.all():
        grayscale = np.dot(rgb_display, [0.299, 0.587, 0.114])
        dimmed = np.repeat(grayscale[..., None], 3, axis=2) * 0.5
        overlay = np.where(fov_mask[..., None], overlay, dimmed)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    combined = Image.new("RGB", (original_img.width * 2, original_img.height))
    combined.paste(original_img, (0, 0))
    combined.paste(overlay_img, (original_img.width, 0))

    stem = output_name or img_path.stem
    fig_path = fig_dir / f"{stem}_cam.{fig_format}"
    combined.save(fig_path)

    cam_array_path = ""
    if save_npy != "none" and array_dir is not None:
        npy_path = array_dir / f"{stem}.npy"
        cam_array = cam_display if save_npy == "normalized" else grayscale_cam
        np.save(npy_path, cam_array.astype(np.float32))
        cam_array_path = str(npy_path)

    class_labels = getattr(model, "class_labels", None)
    pred_class = class_labels[pred_idx] if class_labels else pred_idx
    return {
        "image": source_image or img_path.name,
        "label": label,
        "pred_class": pred_class,
        "pred_prob": pred_score,
        "figure_path": str(fig_path),
        "cam_array_path": cam_array_path,
    }


def run_cam(
    *,
    label_csv: Optional[Path],
    images_dir: Path,
    out_dir: Path,
    model_dir: Optional[Path],
    base_model: Optional[str],
    checkpoint_path: Optional[Path],
    num_classes: Optional[int],
    pretrained: bool,
    cam_method: str,
    arch: Optional[str],
    target_layer_name: Optional[str],
    image_weight: float,
    fig_format: str,
    save_npy: SaveMode,
    dump_model_structure: bool,
    max_images: Optional[int],
    cam_batch_size: int,
    device: torch.device,
    eval_transform: str = "center-crop",
) -> None:
    """Run CAM heatmap generation for all images in label_csv."""
    out_dirs = prepare_output_dirs(out_dir, save_npy=save_npy)
    df = collect_image_label_rows(images_dir=images_dir, label_csv=label_csv)
    model = load_model_from_args(
        load_ag=str(model_dir) if model_dir else None,
        base_model=base_model,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
    )
    preprocess, model_size, resize_size = build_cam_transforms(
        model, eval_transform
    )
    inferred_arch = arch or infer_architecture(base_model or "", model)
    cam_extractor, target_layers, reshape_transform = prepare_cam(
        model, inferred_arch, target_layer_name, cam_method, device, cam_batch_size
    )

    logging.info("Architecture inferred as %s", inferred_arch)
    logging.info(
        "Using target layer(s): %s",
        ", ".join([layer.__class__.__name__ for layer in target_layers]),
    )
    if reshape_transform:
        logging.info("Enabled Transformer reshape_transform for CAM.")
    if dump_model_structure:
        layers_path = write_model_structure(model, out_dir)
        logging.info("Model layer names written to %s", layers_path)

    records = []
    for idx, row in df.iterrows():
        if max_images and idx >= max_images:
            break
        img_path = (images_dir / row["image"]).resolve()
        if not img_path.exists():
            logging.error("Image not found: %s", img_path)
            continue
        try:
            record = process_image(
                img_path=img_path,
                label=str(row["label"]),
                model=model,
                preprocess=preprocess,
                cam_extractor=cam_extractor,
                device=device,
                fig_dir=out_dirs["fig"],
                array_dir=out_dirs["array"],
                image_weight=image_weight,
                fig_format=fig_format,
                save_npy=save_npy,
                eval_transform=eval_transform,
                model_size=model_size,
                resize_size=resize_size,
                output_name=output_stem(str(row["image"])),
                source_image=str(row["image"]),
            )
            records.append(record)
        except Exception as exc:
            logging.exception("Failed on %s: %s", img_path, exc)
    if records:
        pd.DataFrame(records).to_csv(out_dir / "cam_summary.csv", index=False)
        logging.info("CAM heatmaps written to: %s", out_dirs["fig"])
        logging.info("Summary: %s", out_dir / "cam_summary.csv")
        if out_dirs["array"] is not None:
            logging.info(
                "CAM arrays written to: %s (%s)", out_dirs["array"], save_npy
            )
