from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io
from skimage.measure import label

from src.common.files import IMAGE_EXTENSIONS, iter_files


VALID_IMAGE_EXTENSIONS = set(IMAGE_EXTENSIONS)


def iter_mask_files(mask_dir: Path) -> list[Path]:
    return iter_files(mask_dir, IMAGE_EXTENSIONS)


def load_binary_mask(path: Path) -> np.ndarray:
    image = io.imread(str(path))
    if image.ndim == 3:
        image = image[..., 0]
    return (image > 0).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    labels = label(mask_u8, connectivity=2)
    if labels.max() == 0:
        return mask_u8

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    target = int(np.argmax(counts))
    return (labels == target).astype(np.uint8)
