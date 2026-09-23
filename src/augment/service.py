"""Application service for image augmentation."""

from __future__ import annotations

import json
import re
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
from src.common.files import IMAGE_EXTENSIONS, iter_files


@dataclass
class AugmentResult:
    success: bool
    manifest: dict = field(default_factory=dict)
    error: Optional[str] = None


def _list_images(root: Path) -> list[Path]:
    return iter_files(root, IMAGE_EXTENSIONS)


def _expected_augment_names(
    stem: str, suffix: str, n_copies: int, idx_width: int
) -> list[str]:
    return [
        f"{stem}_aug{index:0{idx_width}d}{suffix}" for index in range(1, n_copies + 1)
    ]


def _remove_stale_augmentations(
    target_dir: Path, stem: str, suffix: str, keep: set[str]
) -> None:
    """Delete a source's augmentations that are not part of the current set.

    ``{stem}_aug<digits>{suffix}`` is the only naming this service writes, so a
    leftover copy from a run with a larger ``--multiply`` (or an interrupted
    run) is removed instead of being reported as part of the dataset.
    """
    for candidate in _existing_augmentations(target_dir, stem, suffix):
        if candidate.name not in keep:
            candidate.unlink()


def _existing_augmentations(target_dir: Path, stem: str, suffix: str) -> list[Path]:
    """Return this source's existing ``{stem}_aug<digits>{suffix}`` files."""
    if not target_dir.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(stem)}_aug\d+{re.escape(suffix)}$")
    return [
        candidate
        for candidate in target_dir.iterdir()
        if candidate.is_file() and pattern.match(candidate.name)
    ]


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
    for source_index, img_path in enumerate(images):
        if shutdown_flag is not None and shutdown_flag():
            break
        rel_image = img_path.relative_to(src)
        target_dir = images_out / rel_image.parent
        stem = img_path.stem
        suffix = img_path.suffix or ".jpg"
        expected_names = _expected_augment_names(stem, suffix, n_copies, idx_width)
        existing_names = {
            candidate.name
            for candidate in _existing_augmentations(target_dir, stem, suffix)
        }
        if skip_existing and existing_names == set(expected_names):
            continue
        img_array = cv2.imread(str(img_path))
        if img_array is None:
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        # Drop leftover copies before writing this run's exact set.
        _remove_stale_augmentations(target_dir, stem, suffix, set(expected_names))
        for copy_idx in range(n_copies):
            # Seed by the source's stable position, not the running processed
            # count, so resuming does not change a source's output.
            copy_seed = seed + source_index * n_copies + copy_idx
            result = run_pipeline(pipeline, img_array, seed=copy_seed)

            out_name = expected_names[copy_idx]
            cv2.imwrite(str(target_dir / out_name), result["image"])
            augmented_images.append(
                {
                    "original": rel_image.as_posix(),
                    "augmented": (rel_image.parent / out_name).as_posix(),
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
