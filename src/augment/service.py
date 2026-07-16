"""Application service for image augmentation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.augment.compiler import build_pipeline
from src.augment.runner import run_pipeline


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class AugmentResult:
    success: bool
    manifest: dict = field(default_factory=dict)
    error: Optional[str] = None


def _list_images(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    )


def run_augment(
    input_dir,
    out_dir,
    preset: Optional[str] = "light",
    custom: Optional[dict] = None,
    seed: int = 42,
    multiply: int = 1,
    args: Optional[dict] = None,
    skip_existing: bool = False,
    shutdown_flag: Optional[Callable[[], bool]] = None,
) -> AugmentResult:
    """Run augmentation on an image directory."""
    src = Path(input_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if multiply < 1:
        raise ValueError(f"multiply must be >= 1, got {multiply}")
    if not src.exists() or not src.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {src}")

    image_paths = _list_images(src)
    if not image_paths:
        raise ValueError(f"No images found in '{src}'.")

    images_out = dst / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(preset=preset, custom=custom, args=args)
    n_copies = max(1, multiply)
    idx_width = len(str(n_copies))

    augmented_images: list[dict] = []
    processed_count = 0

    images = tqdm(image_paths, desc="Augmenting") if tqdm else image_paths
    for img_path in images:
        if shutdown_flag is not None and shutdown_flag():
            break
        if skip_existing and any(images_out.glob(f"{img_path.stem}*")):
            continue
        img_array = cv2.imread(str(img_path))
        if img_array is None:
            continue

        stem = img_path.stem
        suffix = img_path.suffix or ".jpg"

        for copy_idx in range(n_copies):
            copy_seed = seed + processed_count * n_copies + copy_idx
            result = run_pipeline(pipeline, img_array, seed=copy_seed)

            out_name = (
                img_path.name
                if multiply <= 1
                else f"{stem}_aug{copy_idx + 1:0{idx_width}d}{suffix}"
            )
            cv2.imwrite(str(images_out / out_name), result["image"])
            augmented_images.append(
                {
                    "original": img_path.name,
                    "augmented": out_name,
                    "copy_index": copy_idx + 1,
                }
            )

        processed_count += 1

    manifest = {
        "preset": preset,
        "multiply": multiply,
        "seed": seed,
        "images_processed": processed_count,
        "augmented_images_created": len(augmented_images),
    }
    (dst / "augment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return AugmentResult(success=True, manifest=manifest)
