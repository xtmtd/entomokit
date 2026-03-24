#!/usr/bin/env python3
"""
cam_visualizer.py

Generate CAM (GradCAM / GradCAM++ / ScoreCAM / EigenCAM / LayerCAM / AblationCAM) visualizations for image classification models.
Supports timm backbones as well as AutoGluon MultiModalPredictor checkpoints.

Example:
    python cam_visualizer.py \
        --label test.csv \
        --input_images_dir ./images \
        --out_dir results \
        --base_model convnextv2_femto \
        --cnn \
        --cam gradcam

"""
import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, EigenCAM, GradCAMPlusPlus, LayerCAM, AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm.data import resolve_model_data_config
from torchvision.transforms.functional import InterpolationMode
import cv2

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcampp": GradCAMPlusPlus,
    "layercam": LayerCAM,
    "ablationcam": AblationCAM,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAM heatmaps for CNN or ViT backbones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label",
        required=True,
        help="CSV file with two columns: 'image' and 'label'.",
    )
    parser.add_argument(
        "--input_images_dir",
        required=True,
        help="Directory that stores all images referenced in the CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="results",
        help="Directory to store every output artifact (figures, arrays, logs).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--load_AutogluonModel",
        default=None,
        help="Path to an AutoGluon MultiModalPredictor (e.g., AutogluonModels/best_model).",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="timm backbone name. Use together with optional --checkpoint_path if you need custom weights.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional PyTorch checkpoint (.pth) to load weights for the timm backbone.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Override number of classes when instantiating a timm backbone.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Do NOT load pretrained weights from timm when creating a backbone.",
    )
    arch_group = parser.add_mutually_exclusive_group(required=False)
    arch_group.add_argument(
        "--cnn",
        action="store_true",
        help="Flag indicating that the backbone is CNN-like.",
    )
    arch_group.add_argument(
        "--vit",
        action="store_true",
        help="Flag indicating that the backbone is ViT/transformer-like.",
    )
    parser.add_argument(
        "--target_layer_name",
        default=None,
        help="Dot-separated path to a specific module to use as CAM target layer (e.g., blocks.11.norm1).",
    )
    parser.add_argument(
        "--cam",
        default="gradcam",
        choices=list(CAM_METHODS.keys()),
        help="CAM algorithm to use.",
    )
    parser.add_argument(
        "--image_weight",
        type=float,
        default=0.5,
        help="Weight for the original image when blending with the CAM heatmap (0~1).",
    )
    parser.add_argument(
        "--fig_format",
        default="png",
        choices=["png", "jpg", "jpeg", "pdf"],
        help="File format for the side-by-side visualization.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Save normalized CAM arrays as .npy files.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optionally cap the number of images processed (useful for quick tests).",
    )
    parser.add_argument(
        "--cam_batch_size",
        type=int,
        default=32,
        help="Internal batch size used by pytorch-grad-cam (relevant for ScoreCAM/EigenCAM).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logger(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=getattr(logging, level),
    )


def select_device(device_str: str) -> torch.device:
    """Select torch device with availability checks."""
    device_str = device_str.lower()
    if device_str == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            logging.warning("MPS not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device("cpu")


def load_label_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns named 'image' and 'label'.")
    return df[["image", "label"]]


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.load_AutogluonModel:
        try:
            from autogluon.multimodal import MultiModalPredictor
        except ImportError as exc:
            raise ImportError("AutoGluon is not installed. pip install autogluon.multimodal") from exc
        predictor = MultiModalPredictor.load(args.load_AutogluonModel)
        torch_model = predictor._learner._model.model
        logging.info("Loaded AutoGluon model from %s", args.load_AutogluonModel)
    else:
        if args.base_model is None:
            raise ValueError("Either --load_AutogluonModel or --base_model must be provided.")
        import timm

        pretrained = not args.no_pretrained
        torch_model = timm.create_model(
            args.base_model,
            pretrained=pretrained,
            num_classes=args.num_classes if args.num_classes is not None else None,
        )
        logging.info(
            "Instantiated timm backbone %s (pretrained=%s, num_classes=%s)",
            args.base_model,
            pretrained,
            args.num_classes,
        )
        if args.checkpoint_path:
            state_dict = torch.load(args.checkpoint_path, map_location="cpu")
            torch_model.load_state_dict(state_dict, strict=False)
            logging.info("Loaded checkpoint weights from %s", args.checkpoint_path)
    torch_model.eval()
    torch_model.to(device)
    return torch_model


def build_eval_transforms(model: torch.nn.Module) -> Tuple[transforms.Compose, transforms.Compose]:
    cfg = resolve_model_data_config(model)
    crop_tuple = cfg.get("test_input_size", cfg.get("input_size"))
    crop_size = crop_tuple[1]
    crop_pct = cfg.get("test_crop_pct", cfg.get("crop_pct", 1.0))
    resize_shorter = int(round(crop_size / crop_pct))
    mean = cfg.get("mean")
    std = cfg.get("std")
    interpolation = cfg.get("interpolation", "bicubic").upper()
    interpolation = getattr(InterpolationMode, interpolation, InterpolationMode.BICUBIC)
    preprocess = transforms.Compose([
        transforms.Resize(resize_shorter, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    display_transform = transforms.Compose([
        transforms.Resize(resize_shorter, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
    ])
    return preprocess, display_transform


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    module = model
    for attr in name.split("."):
        if not hasattr(module, attr):
            raise AttributeError(f"Module '{module.__class__.__name__}' has no attribute '{attr}'")
        module = getattr(module, attr)
    return module


def find_last_conv_module(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found. Specify --target_layer_name manually.")
    return last_conv


def default_vit_target(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        block = model.blocks[-1]
        for candidate in ["norm1", "ln1", "ln"]:
            if hasattr(block, candidate):
                return getattr(block, candidate)
        # fallback to block itself
        return block
    raise RuntimeError("Could not automatically find a ViT block. Specify --target_layer_name.")


def infer_architecture(args: argparse.Namespace, model: torch.nn.Module) -> str:
    if args.cnn:
        return "cnn"
    if args.vit:
        return "vit"
    name = args.base_model or model.__class__.__name__
    name = name.lower()
    if "vit" in name or "transformer" in name:
        return "vit"
    return "cnn"


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT tokens (B, N, C) into feature maps (B, C, H, W)."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected ViT token tensor (B, N, C). Got shape {tensor.shape}.")
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
        target_layers = [get_module_by_name(model, target_layer_name)]
    else:
        target_layers = (
            [find_last_conv_module(model)] if arch == "cnn" else [default_vit_target(model)]
        )

    reshape_transform = vit_reshape_transform if arch == "vit" else None
    cam_kwargs = {
        "model": model,
        "target_layers": target_layers,
        "reshape_transform": reshape_transform,
    }
    if cam_name == "ablationcam" and arch == "vit":
        cam_kwargs["ablation_layer"] = AblationLayerVit()
    cam = CAM_METHODS[cam_name](**cam_kwargs)
    if isinstance(cam, BaseCAM):
        cam.batch_size = cam_batch_size  # ScoreCAM/EigenCAM
    return cam, target_layers, reshape_transform


def prepare_output_dirs(out_dir: Path) -> Dict[str, Path]:
    fig_dir = out_dir / "figures"
    array_dir = out_dir / "arrays"
    for d in [fig_dir, array_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"fig": fig_dir, "array": array_dir}


def process_image(
    img_path: Path,
    label: str,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    display_transform: transforms.Compose,
    cam_extractor,
    device: torch.device,
    fig_dir: Path,
    array_dir: Path,
    args: argparse.Namespace,
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

    cam_norm = grayscale_cam - grayscale_cam.min()
    if cam_norm.max() > 0:
        cam_norm = cam_norm / cam_norm.max()
    else:
        cam_norm = np.zeros_like(cam_norm)

    cam_on_full = cv2.resize(
        cam_norm,
        (original_img.width, original_img.height),
        interpolation=cv2.INTER_LINEAR,
    )

    overlay = show_cam_on_image(
        rgb_display,
        cam_on_full,
        use_rgb=True,
        image_weight=args.image_weight,
    )
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    combined = Image.new("RGB", (original_img.width * 2, original_img.height))
    combined.paste(original_img, (0, 0))
    combined.paste(overlay_img, (original_img.width, 0))

    fig_path = fig_dir / f"{img_path.stem}_cam.{args.fig_format}"
    combined.save(fig_path)

    cam_array_path = ""
    if args.save_npy:
        cam_array_path = array_dir / f"{img_path.stem}.npy"
        np.save(cam_array_path, cam_norm.astype(np.float32))
        cam_array_path = str(cam_array_path)

    return {
        "image": img_path.name,
        "label": label,
        "pred_class": pred_idx,
        "pred_prob": pred_score,
        "figure_path": str(fig_path),
        "cam_array_path": cam_array_path,
    }


def main():
    args = parse_args()
    setup_logger(args.log_level)
    csv_path = Path(args.label)
    img_root = Path(args.input_images_dir)
    out_dir = Path(args.out_dir)
    out_dirs = prepare_output_dirs(out_dir)

    df = load_label_file(csv_path)
    device = select_device(args.device)
    model = load_model(args, device)
    preprocess, display_transform = build_eval_transforms(model)
    arch = infer_architecture(args, model)
    cam_extractor, target_layers, reshape_transform = prepare_cam(
        model, arch, args.target_layer_name, args.cam, device, args.cam_batch_size
    )

    logging.info("Architecture inferred as %s", arch)
    logging.info(
        "Using target layer(s): %s",
        ", ".join([layer.__class__.__name__ for layer in target_layers]),
    )
    if reshape_transform:
        logging.info("Enabled ViT reshape_transform for CAM.")

    records = []
    total_images = len(df)
    logging.info("Starting CAM generation for %d images...", total_images)

    for idx, row in df.iterrows():
        if args.max_images and idx >= args.max_images:
            break
        img_rel_path = row["image"]
        label = str(row["label"])
        img_path = (img_root / img_rel_path).expanduser().resolve()

        if not img_path.exists():
            logging.error("Image %s not found. Skipping.", img_path)
            continue

        try:
            record = process_image(
                img_path=img_path,
                label=label,
                model=model,
                preprocess=preprocess,
                display_transform=display_transform,
                cam_extractor=cam_extractor,
                device=device,
                fig_dir=out_dirs["fig"],
                array_dir=out_dirs["array"],
                args=args,
            )
            records.append(record)
            if (idx + 1) % 10 == 0:
                logging.info("Processed %d/%d images", idx + 1, total_images)
        except Exception as exc:
            logging.exception("Failed on image %s: %s", img_path, exc)

    if records:
        df_out = pd.DataFrame(records)
        summary_path = out_dir / "cam_summary.csv"
        df_out.to_csv(summary_path, index=False)
        logging.info("Saved summary CSV to %s", summary_path)
    else:
        logging.warning("No CAM results were generated.")

    logging.info("Finished.")


if __name__ == "__main__":
    main()

