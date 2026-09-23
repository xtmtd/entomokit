"""Synthesis processor for compositing target objects onto background images."""

import hashlib
import logging
import math
import multiprocessing
import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image
from skimage import exposure
from skimage.color import rgb2lab, lab2rgb

from src.common.files import IMAGE_EXTENSIONS, iter_files

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon

logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="skimage.color.colorconv"
)


def _annotation_stem(output_filename: str) -> str:
    """Preserve relative parents while stripping any file extension."""
    rel = Path(output_filename)
    return str(rel.parent / rel.stem)


def _image_mode(image_path: Path) -> str:
    """Return the PIL mode of an image, or 'unreadable'."""
    try:
        with Image.open(image_path) as img:
            return img.mode
    except Exception:
        return "unreadable"


def resolve_target_stems(target_paths: List[Path], target_dir: Path) -> dict:
    """Return a stable output stem per target, disambiguating same-dir clashes.

    Different directories are already separated by mirrored output paths, so
    only targets sharing a directory *and* a stem (for example ``a.png`` and
    ``a.tif``) need a digest suffix.
    """
    groups: dict = {}
    for path in target_paths:
        groups.setdefault((path.parent, path.stem), []).append(path)

    stems: dict = {}
    for paths in groups.values():
        if len(paths) == 1:
            stems[paths[0]] = paths[0].stem
            continue
        for path in paths:
            rel = path.relative_to(target_dir).as_posix()
            digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:8]
            stems[path] = f"{path.stem}__{digest}"
    return stems


def derive_seed(base: int, index: int) -> int:
    """Derive a stable task seed from a base seed and an index."""
    digest = hashlib.sha256(f"{base}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _existing_synthesis_outputs(
    target_out_dir: Path, stem: str, extension: str
) -> dict:
    """Map copy index -> existing ``{stem}_{NN}{ext}`` file for one target."""
    found: dict = {}
    if not target_out_dir.is_dir():
        return found
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(extension)}$")
    for candidate in target_out_dir.iterdir():
        if not candidate.is_file():
            continue
        match = pattern.match(candidate.name)
        if match:
            digits = match.group(1)
            # The generator writes ``{index:02d}``; reject non-canonical names
            # such as ``a_1.png`` so they are not mistaken for ``a_01.png``.
            if digits != f"{int(digits):02d}":
                continue
            found[int(digits)] = candidate
    return found


def resolve_num_syntheses(
    value: float, target_count: int, seed: int = 42
) -> Tuple[List[int], int]:
    """Resolve --num-syntheses into selected target indices and a per-target count.

    Positive integer-valued input selects every target and returns that count.
    A value strictly between 0 and 1 selects ``floor(target_count * value)``
    targets (at least one when targets exist) and returns one synthesis each.
    """
    if target_count <= 0:
        return [], 0

    if isinstance(value, float):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"num_syntheses must be positive, got {value}")
        if 0 < value < 1:
            count = min(target_count, max(1, math.floor(target_count * value)))
            indices = sorted(random.Random(seed).sample(range(target_count), count))
            return indices, 1
        if not value.is_integer():
            raise ValueError(
                f"num_syntheses must be a positive integer or a fraction in (0, 1), got {value}"
            )

    count = int(value)
    if count <= 0:
        raise ValueError(f"num_syntheses must be positive, got {value}")
    return list(range(target_count)), count


class SynthesisProcessor:
    """Processor for compositing target objects onto background images."""

    def __init__(
        self,
        output_format: str = "png",
        area_ratio_min: float = 0.05,
        area_ratio_max: float = 0.20,
        color_match_strength: float = 0.5,
        avoid_black_regions: bool = False,
        rotate_degrees: float = 30.0,
        output_subdir: str = "images",
        annotation_format: str = "coco",
        coco_output_mode: str = "unified",
        coco_bbox_format: str = "xywh",
    ):
        """Initialize synthesis processor."""
        self.output_format = output_format.lower()
        self.area_ratio_min = max(0.01, min(0.50, area_ratio_min))
        self.area_ratio_max = max(0.01, min(0.50, area_ratio_max))
        self.color_match_strength = color_match_strength
        self.avoid_black_regions = avoid_black_regions
        self.rotate_degrees = max(0.0, rotate_degrees)
        self.output_subdir = output_subdir
        self.annotation_format = annotation_format.lower()
        self.coco_output_mode = coco_output_mode.lower()
        self.coco_bbox_format = coco_bbox_format
        self.metadata_manager = COCOMetadataManager()
        self.insect_category_id = self.metadata_manager.add_category("insect")
        self.synthesis_metadata: List[Dict[str, Any]] = []
        self._current_image_width: Optional[int] = None
        self._current_image_height: Optional[int] = None
        # Accumulators for unified COCO output via annotation_writer
        self._ann_image_paths: list = []
        self._ann_detections: dict = {}

    def load_images_from_directory(
        self, directory: Path, desc: str = "Loading images"
    ) -> List[np.ndarray]:
        """Load all images from directory recursively."""
        image_paths = iter_files(directory, IMAGE_EXTENSIONS)

        images = []
        for img_path in image_paths:
            try:
                img = self._load_image(img_path)
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")

        logger.info(f"Loaded {len(images)} images from {directory}")
        return images

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from file path."""
        img = Image.open(image_path)
        if img.mode == "RGBA":
            return np.array(img)
        elif img.mode == "RGB":
            return np.array(img)
        else:
            img = img.convert("RGB")
            return np.array(img)

    def _load_target_image(self, image_path: Path) -> np.ndarray:
        """Load a foreground (target) image, requiring RGBA (alpha channel).

        Raises ValueError with a clear message if the image lacks an alpha channel,
        since synthesis requires the alpha mask to composite the insect onto backgrounds.
        """
        img = Image.open(image_path)
        if img.mode != "RGBA":
            raise ValueError(
                f"Target image '{image_path.name}' is {img.mode} (no alpha channel). "
                "Synthesis requires RGBA images with a transparency mask. "
                "Please use PNG format with an alpha channel (e.g. segmented insect cutouts)."
            )
        return np.array(img)

    def _save_image(
        self, image: np.ndarray, output_path: Path, quality: int = 90
    ) -> None:
        """Save image to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if image.shape[2] == 4 and self.output_format == "jpg":
            img_pil = Image.fromarray(image[:, :, :3])
        else:
            img_pil = Image.fromarray(image)
        if self.output_format == "jpg":
            img_pil.save(output_path, "JPEG", quality=quality, optimize=True)
        else:
            img_pil.save(
                output_path, "PNG", optimize=True, compress_level=9 - (quality // 11)
            )

    def _calculate_scale_factor(
        self, background_shape: Tuple[int, ...], mask_area: int, scale_ratio: float
    ) -> float:
        """Calculate scale factor to achieve target area ratio."""
        bg_area = background_shape[0] * background_shape[1]
        target_pixel_area = int(bg_area * scale_ratio)
        scale_factor = np.sqrt(target_pixel_area / mask_area)
        return float(scale_factor)

    def _is_region_black(
        self,
        background: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        threshold: int = 15,
    ) -> bool:
        """Check if region contains any near-black pixels (RGB values below threshold).

        Args:
            background: Background image
            x, y: Top-left position
            w, h: Width and height of region to check
            threshold: Maximum RGB value to consider as "black" (default 15)

        Returns:
            True if region contains ANY near-black pixels, False otherwise
        """
        region = background[y : y + h, x : x + w]
        if region.shape[2] == 4:
            rgb = region[:, :, :3]
        else:
            rgb = region
        is_near_black = np.all(rgb <= threshold, axis=-1)
        return bool(np.any(is_near_black))

    def _random_position_with_constraint(
        self,
        background: np.ndarray,
        target_shape: Tuple[int, ...],
        edge_margin: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, int, float]:
        """Find random position with constraint (avoid edges and optionally black regions).

        Returns:
            Tuple of (x, y, scale_factor) where scale_factor is the downscale applied (1.0 if none)
        """
        bg_h, bg_w = background.shape[:2]
        target_h, target_w = target_shape[:2]
        rng = rng if rng is not None else np.random.default_rng()
        margin_x = int(bg_w * edge_margin)
        margin_y = int(bg_h * edge_margin)
        min_x = margin_x
        max_x = bg_w - target_w - margin_x
        min_y = margin_y
        max_y = bg_h - target_h - margin_y
        if max_x <= min_x:
            min_x = 0
            max_x = max(1, bg_w - target_w)
        if max_y <= min_y:
            min_y = 0
            max_y = max(1, bg_h - target_h)
        max_attempts = 100
        if not self.avoid_black_regions:
            x = int(rng.integers(min_x, max_x)) if max_x > min_x else min_x
            y = int(rng.integers(min_y, max_y)) if max_y > min_y else min_y
            return x, y, 1.0
        downscale_factor = 1.0
        while downscale_factor >= 0.1:
            scaled_target_h = int(target_h * downscale_factor)
            scaled_target_w = int(target_w * downscale_factor)
            for _ in range(max_attempts):
                x = int(rng.integers(min_x, max_x)) if max_x > min_x else min_x
                y = int(rng.integers(min_y, max_y)) if max_y > min_y else min_y
                if not self._is_region_black(
                    background, x, y, scaled_target_w, scaled_target_h
                ):
                    return x, y, downscale_factor
            downscale_factor -= 0.1
        x = int(rng.integers(min_x, max_x)) if max_x > min_x else min_x
        y = int(rng.integers(min_y, max_y)) if max_y > min_y else min_y
        return x, y, 1.0

    def _get_random_position_no_constraint(
        self,
        background: np.ndarray,
        target_shape: Tuple[int, ...],
        edge_margin: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, int]:
        """Get random position without black region checking."""
        bg_h, bg_w = background.shape[:2]
        target_h, target_w = target_shape[:2]
        rng = rng if rng is not None else np.random.default_rng()
        margin_x = int(bg_w * edge_margin)
        margin_y = int(bg_h * edge_margin)
        min_x = margin_x
        max_x = bg_w - target_w - margin_x
        min_y = margin_y
        max_y = bg_h - target_h - margin_y
        if max_x <= min_x:
            min_x = 0
            max_x = max(1, bg_w - target_w)
        if max_y <= min_y:
            min_y = 0
            max_y = max(1, bg_h - target_h)
        x = int(rng.integers(min_x, max_x)) if max_x > min_x else min_x
        y = int(rng.integers(min_y, max_y)) if max_y > min_y else min_y
        return x, y

    def _paste_with_alpha(
        self, background: np.ndarray, target_rgba: np.ndarray, x: int, y: int
    ) -> np.ndarray:
        """Paste target with alpha blending onto background."""
        result = background.copy()
        h, w = target_rgba.shape[:2]
        bg_h, bg_w = background.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + w)
        y2 = min(bg_h, y + h)
        src_x1 = x1 - x
        src_y1 = y1 - y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        if x1 >= x2 or y1 >= y2:
            return result
        bg_region = result[y1:y2, x1:x2].astype(np.float32)
        target_region = target_rgba[src_y1:src_y2, src_x1:src_x2]
        alpha = target_region[:, :, 3:4].astype(np.float32) / 255.0
        mask = alpha > 0.1
        if np.any(mask):
            target_rgb = target_region[:, :, :3].astype(np.float32)
            blended = (alpha * target_rgb + (1 - alpha) * bg_region).astype(np.uint8)
            result[y1:y2, x1:x2] = np.where(mask, blended, bg_region.astype(np.uint8))
        return result

    def _rotate_image(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> np.ndarray:
        """Rotate image by random angle within specified degrees."""
        if self.rotate_degrees <= 0:
            return image
        if angle is None:
            rng = rng if rng is not None else random.Random()
            angle = rng.uniform(-self.rotate_degrees, self.rotate_degrees)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_abs = abs(M[0, 0])
        sin_abs = abs(M[0, 1])
        new_w = int((h * sin_abs) + (w * cos_abs))
        new_h = int((h * cos_abs) + (w * sin_abs))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return rotated

    def _calculate_max_scale_to_fit(
        self, target_shape: Tuple[int, ...], background_shape: Tuple[int, ...]
    ) -> float:
        """Calculate maximum scale factor to ensure target fits within background."""
        target_h, target_w = target_shape[:2]
        bg_h, bg_w = background_shape[:2]
        scale_x = bg_w / target_w if target_w > 0 else 1.0
        scale_y = bg_h / target_h if target_h > 0 else 1.0
        return min(scale_x, scale_y)

    def _get_target_filename(
        self, target_image_path: Path, counter: int, target_stem: Optional[str] = None
    ) -> str:
        """Generate output filename from target filename and counter."""
        base_name = target_stem if target_stem else target_image_path.stem
        counter_str = f"{counter:02d}"
        return f"{base_name}_{counter_str}"

    def _match_lab_histograms(
        self, image: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Match LAB histograms between image and reference."""
        image_lab = rgb2lab(image)
        reference_lab = rgb2lab(reference)
        image_lab = np.nan_to_num(image_lab, nan=0.0, posinf=100.0, neginf=-100.0)
        reference_lab = np.nan_to_num(
            reference_lab, nan=0.0, posinf=100.0, neginf=-100.0
        )
        result_lab = image_lab.copy()
        for channel in range(3):
            image_channel = image_lab[:, :, channel]
            ref_channel = reference_lab[:, :, channel]
            matched = exposure.match_histograms(image_channel, ref_channel)
            result_lab[:, :, channel] = (
                1 - self.color_match_strength
            ) * image_channel + self.color_match_strength * matched
        result = (lab2rgb(result_lab) * 255).astype(np.uint8)
        return result

    def synthesize_single(
        self,
        target_image: np.ndarray,
        background: np.ndarray,
        scale_ratio: Optional[float] = None,
        target_path: Optional[Path] = None,
        counter: int = 0,
        background_path: Optional[Path] = None,
        rotation_angle: Optional[float] = None,
        np_seed: Optional[int] = None,
        target_stem: Optional[str] = None,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[str],
        float,
        Optional[float],
        Optional[Path],
        Optional[Path],
        int,
        int,
        np.ndarray,
    ]:
        """Perform single synthesis.

        Args:
            target_image: Target image with alpha channel
            background: Background image
            scale_ratio: Target area ratio (0.01-0.50)
            target_path: Original target file path for naming output
            counter: Counter for output filename
            background_path: Original background file path
            rotation_angle: Pre-generated rotation angle (task-local randomness)
            np_seed: Seed for the task-local NumPy generator

        Returns:
            Tuple of (result_image, output_filename, scale_ratio, rotation_angle, target_path, background_path, position_x, position_y, final_target)
        """
        try:
            np_rng = np.random.default_rng(np_seed)
            py_rng = random.Random(np_seed)
            if target_image.shape[2] == 4:
                mask = target_image[:, :, 3]
            else:
                mask = np.ones(target_image.shape[:2], dtype=np.uint8) * 255
            mask_area = int(np.sum(mask > 0))
            if scale_ratio is None:
                scale_ratio = py_rng.uniform(
                    self.area_ratio_min, self.area_ratio_max
                )
            scale_factor = self._calculate_scale_factor(
                background.shape, mask_area, scale_ratio
            )
            new_h = int(target_image.shape[0] * scale_factor)
            new_w = int(target_image.shape[1] * scale_factor)
            if new_h < 10 or new_w < 10:
                logger.warning(f"Target too small after scaling: {new_w}x{new_h}")
                return (
                    None,
                    None,
                    scale_ratio,
                    None,
                    target_path,
                    background_path,
                    0,
                    0,
                    target_image,
                )
            target_scaled = cv2.resize(
                target_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            max_scale = self._calculate_max_scale_to_fit(
                target_scaled.shape, background.shape
            )
            if max_scale < 1.0:
                logger.debug(
                    f"Auto-downscaling target to fit background (scale: {max_scale:.2f})"
                )
                new_h = int(new_h * max_scale)
                new_w = int(new_w * max_scale)
            if new_h < 10 or new_w < 10:
                logger.warning(
                    f"Target too small after auto-downscaling: {new_w}x{new_h}"
                )
                return (
                    None,
                    None,
                    scale_ratio,
                    None,
                    target_path,
                    background_path,
                    0,
                    0,
                    target_scaled,
                )
            angle = None
            if self.rotate_degrees > 0:
                if rotation_angle is not None:
                    angle = rotation_angle
                else:
                    angle = py_rng.uniform(-self.rotate_degrees, self.rotate_degrees)
                logger.debug(f"Selected rotation angle: {angle:.2f} degrees")
            target_rotated = self._rotate_image(target_scaled, angle, rng=py_rng)
            downscale_factor = 1.0
            while (
                target_rotated.shape[0] > background.shape[0]
                or target_rotated.shape[1] > background.shape[1]
            ):
                if downscale_factor <= 0.1:
                    logger.warning(
                        f"Target (even after downscaling) exceeds background dimensions, skipping"
                    )
                    return (
                        None,
                        None,
                        scale_ratio,
                        angle,
                        target_path,
                        background_path,
                        0,
                        0,
                        target_scaled,
                    )
                downscale_factor *= 0.9
                new_h = int(target_scaled.shape[0] * downscale_factor)
                new_w = int(target_scaled.shape[1] * downscale_factor)
                if new_h < 10 or new_w < 10:
                    logger.warning(
                        f"Target too small after downscaling: {new_w}x{new_h}"
                    )
                    return (
                        None,
                        None,
                        scale_ratio,
                        angle,
                        target_path,
                        background_path,
                        0,
                        0,
                        target_scaled,
                    )
                target_scaled = cv2.resize(
                    target_scaled, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
                logger.debug(
                    f"Auto-downscaling target to fit (scale: {downscale_factor:.2f})"
                )
                target_rotated = self._rotate_image(target_scaled, angle, rng=py_rng)
            final_target = target_rotated
            final_target_for_check = final_target.copy()
            x, y = 0, 0
            if self.avoid_black_regions:
                found_valid_position = False
                best_position = None
                best_black_ratio = 1.0
                original_target = final_target_for_check.copy()
                target_h, target_w = original_target.shape[:2]
                min_target_size = 50

                scale_factors = [1.0, 0.85, 0.7, 0.55, 0.4]
                attempts_per_scale = [300, 400, 500, 600, 800]

                for scale_idx, scale_factor in enumerate(scale_factors):
                    if found_valid_position:
                        break

                    current_h = int(target_h * scale_factor)
                    current_w = int(target_w * scale_factor)

                    if current_h < min_target_size or current_w < min_target_size:
                        continue

                    if scale_factor < 1.0:
                        final_target_for_check = cv2.resize(
                            original_target,
                            (current_w, current_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        final_target_for_check = original_target.copy()

                    max_attempts = attempts_per_scale[scale_idx]

                    for attempt in range(max_attempts):
                        x, y = self._get_random_position_no_constraint(
                            background,
                            (current_h, current_w, 4),
                            edge_margin=0.05,
                            rng=np_rng,
                        )

                        region = background[y : y + current_h, x : x + current_w]
                        if region.shape[0] > 0 and region.shape[1] > 0:
                            if region.shape[2] == 4:
                                rgb = region[:, :, :3]
                            else:
                                rgb = region
                            black_pixels = np.sum(np.all(rgb <= 15, axis=-1))
                            total_pixels = region.shape[0] * region.shape[1]
                            black_ratio = black_pixels / total_pixels

                            if black_ratio == 0:
                                found_valid_position = True
                                logger.debug(
                                    f"Found perfect position at scale {scale_factor:.2f} after {attempt + 1} attempts"
                                )
                                break
                            elif black_ratio < best_black_ratio:
                                best_black_ratio = black_ratio
                                best_position = (x, y)

                if not found_valid_position:
                    if best_position is not None:
                        x, y = best_position
                        logger.debug(
                            f"Using best position with {best_black_ratio * 100:.1f}% black pixels"
                        )
                    else:
                        x, y = self._get_random_position_no_constraint(
                            background,
                            final_target_for_check.shape,
                            edge_margin=0.05,
                            rng=np_rng,
                        )
            else:
                x, y = self._get_random_position_no_constraint(
                    background,
                    final_target_for_check.shape,
                    edge_margin=0.05,
                    rng=np_rng,
                )
            result = self._paste_with_alpha(background, final_target_for_check, x, y)
            if self.color_match_strength > 0:
                result = self._match_lab_histograms(result, background)
            output_filename = None
            if target_path is not None and counter > 0:
                output_filename = self._get_target_filename(
                    target_path, counter, target_stem
                )

            return (
                result,
                output_filename,
                scale_ratio,
                angle,
                target_path,
                background_path,
                x,
                y,
                final_target_for_check,
            )
        except Exception:
            logger.exception("Synthesis failed")
            return (
                None,
                None,
                scale_ratio if scale_ratio is not None else 0.0,
                None,
                target_path,
                background_path,
                0,
                0,
                target_image,
            )

    def _synthesize_single_wrapper(self, args):
        """Wrapper for multiprocessing - calls synthesize_single with args tuple."""
        (
            target_img,
            background,
            scale_ratio,
            target_path,
            counter,
            background_path,
            rotation_angle,
            np_seed,
            target_stem,
        ) = args
        result = self.synthesize_single(
            target_img,
            background,
            scale_ratio,
            target_path,
            counter,
            background_path,
            rotation_angle=rotation_angle,
            np_seed=np_seed,
            target_stem=target_stem,
        )
        return (
            result[0],
            result[1],
            result[2],
            None if result[0] is None else result[3],
            str(result[4]) if result[4] is not None else None,
            str(result[5]) if result[5] is not None else None,
            result[6] if len(result) > 6 else None,
            result[7] if len(result) > 7 else None,
            result[8] if len(result) > 8 else None,
        )

    def _add_synthesis_metadata(
        self,
        output_filename: str,
        result: np.ndarray,
        scale_ratio: float,
        rotation_angle: Optional[float],
        position_x: Optional[int] = None,
        position_y: Optional[int] = None,
    ) -> None:
        """Add synthesis metadata for COCO annotation generation.

        Args:
            output_filename: Output filename (without extension)
            result: SYNTHESIZED image (H, W, 4) with alpha channel
            scale_ratio: Scale ratio used for this synthesis
            rotation_angle: Rotation angle in degrees
            position_x: X position in synthesized image (optional)
            position_y: Y position in synthesized image (optional)
        """
        if result.shape[2] == 4:
            result_mask = result[:, :, 3]
        else:
            result_mask = None

        if result_mask is not None and np.any(result_mask):
            bbox = mask_to_bbox(result_mask)
            polygon = mask_to_polygon(result_mask)
            mask_area = int(np.sum(result_mask > 0))
            area = float(bbox[2] * bbox[3])
        else:
            bbox = [0, 0, result.shape[1], result.shape[0]]
            polygon = []
            mask_area = 0
            area = float(result.shape[1] * result.shape[0])

        self.synthesis_metadata.append(
            {
                "image": {
                    "file_name": f"{output_filename}.{self.output_format}",
                    "width": result.shape[1],
                    "height": result.shape[0],
                    "position_x": position_x,
                    "position_y": position_y,
                },
                "annotation": {
                    "bbox": [int(x) for x in bbox],
                    "segmentation": polygon,
                    "area": area,
                    "mask_area": mask_area,
                    "scale_ratio": scale_ratio,
                    "rotation_angle": rotation_angle
                    if rotation_angle is not None
                    else 0.0,
                    "position_x": position_x,
                    "position_y": position_y,
                },
            }
        )

    def _save_annotation_for_image(
        self,
        output_filename: str,
        result: np.ndarray,
        scale_ratio: float,
        rotation_angle: Optional[float],
        position_x: Optional[int],
        position_y: Optional[int],
        output_dir: Optional[Path] = None,
        target_rgba: Optional[np.ndarray] = None,
    ) -> None:
        """Save / accumulate annotation after each image synthesis."""
        if self.annotation_format == "coco":
            # Accumulate for unified write via annotation_writer at end of process_directory()
            self._accumulate_coco_single(
                output_filename,
                result,
                position_x,
                position_y,
                output_dir,
                target_rgba,
            )
        elif self.annotation_format == "voc":
            self._save_voc_single(
                output_filename,
                result,
                scale_ratio,
                rotation_angle,
                position_x,
                position_y,
                output_dir,
                target_rgba,
            )
        elif self.annotation_format == "yolo":
            self._save_yolo_single(
                output_filename,
                result,
                scale_ratio,
                rotation_angle,
                position_x,
                position_y,
                output_dir,
                target_rgba,
            )

    def _accumulate_coco_single(
        self,
        output_filename: str,
        result: np.ndarray,
        position_x: Optional[int],
        position_y: Optional[int],
        output_dir: Optional[Path] = None,
        target_rgba: Optional[np.ndarray] = None,
    ) -> None:
        """Accumulate detection for a single synthesized image into annotation_writer state."""
        import supervision as sv

        # Determine image path in the output images dir
        effective_output_dir = (
            output_dir if output_dir else Path(self.output_subdir).parent
        )
        img_filename = f"{output_filename}.{self.output_format}"
        img_path = effective_output_dir / self.output_subdir / img_filename

        # Calculate bbox from target_rgba or result alpha
        pos_x = position_x if position_x is not None else 0
        pos_y = position_y if position_y is not None else 0

        if target_rgba is not None and target_rgba.shape[2] == 4:
            target_mask = target_rgba[:, :, 3]
            bbox = mask_to_bbox(target_mask)
        elif result.shape[2] == 4:
            result_mask = result[:, :, 3]
            bbox = mask_to_bbox(result_mask)
        else:
            bbox = [0, 0, result.shape[1], result.shape[0]]

        # Offset bbox by position in the background image
        x1 = float(bbox[0] + pos_x)
        y1 = float(bbox[1] + pos_y)
        x2 = float(bbox[0] + pos_x + bbox[2])
        y2 = float(bbox[1] + pos_y + bbox[3])

        xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        class_ids = np.array([0], dtype=int)
        dets = sv.Detections(xyxy=xyxy, class_id=class_ids)

        self._ann_image_paths.append(img_path)
        self._ann_detections[str(img_path)] = dets

    def _save_coco_single(
        self,
        output_filename: str,
        result: np.ndarray,
        scale_ratio: float,
        rotation_angle: Optional[float],
        position_x: Optional[int],
        position_y: Optional[int],
        output_dir: Optional[Path] = None,
        target_rgba: Optional[np.ndarray] = None,
    ) -> None:
        """Save single COCO JSON file per image."""
        if output_dir:
            annotations_dir = self._get_annotation_output_dir(output_dir)
        else:
            annotations_dir = self._get_annotation_output_dir(
                Path(self.output_subdir).parent
            )
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Calculate bbox and segmentation from target_rgba alpha channel
        if target_rgba is not None and target_rgba.shape[2] == 4:
            target_mask = target_rgba[:, :, 3]
            bbox = mask_to_bbox(target_mask)
            polygon = mask_to_polygon(target_mask)
            area = float(bbox[2] * bbox[3])
        elif result.shape[2] == 4:
            # Fallback: use result alpha channel
            result_mask = result[:, :, 3]
            bbox = mask_to_bbox(result_mask)
            polygon = mask_to_polygon(result_mask)
            area = float(bbox[2] * bbox[3])
        else:
            # Fallback: use full image if no alpha channel
            bbox = [0, 0, result.shape[1], result.shape[0]]
            polygon = []
            area = float(result.shape[1] * result.shape[0])

        # Apply position offset to bbox and polygon
        pos_x = position_x if position_x is not None else 0
        pos_y = position_y if position_y is not None else 0

        adjusted_bbox = [
            bbox[0] + pos_x,  # x
            bbox[1] + pos_y,  # y
            bbox[2],  # width (unchanged)
            bbox[3],  # height (unchanged)
        ]

        adjusted_polygon = []
        for poly in polygon:
            adjusted_poly = []
            for i in range(0, len(poly), 2):
                adjusted_poly.append(poly[i] + pos_x)  # x
                adjusted_poly.append(poly[i + 1] + pos_y)  # y
            adjusted_polygon.append(adjusted_poly)

        manager = COCOMetadataManager()
        category_id = manager.add_category("insect")

        image_id = manager.add_image(
            file_name=output_filename, width=result.shape[1], height=result.shape[0]
        )

        manager.add_annotation(
            image_id=image_id,
            category_id=category_id,
            bbox=[int(x) for x in adjusted_bbox],
            segmentation=adjusted_polygon,
            area=area,
            scale_ratio=scale_ratio,
            rotation_angle=rotation_angle if rotation_angle is not None else 0.0,
        )

        annotations_path = annotations_dir / f"{_annotation_stem(output_filename)}.json"
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        manager.save(annotations_path)

    def _save_voc_single(
        self,
        output_filename: str,
        result: np.ndarray,
        scale_ratio: float,
        rotation_angle: Optional[float],
        position_x: Optional[int],
        position_y: Optional[int],
        output_dir: Optional[Path] = None,
        target_rgba: Optional[np.ndarray] = None,
    ) -> None:
        """Save single VOC XML file per image."""
        if output_dir:
            annotations_dir = self._get_annotation_output_dir(output_dir)
        else:
            annotations_dir = self._get_annotation_output_dir(
                Path(self.output_subdir).parent
            )
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Calculate bbox and segmentation from target_rgba alpha channel
        if target_rgba is not None and target_rgba.shape[2] == 4:
            target_mask = target_rgba[:, :, 3]
            bbox = mask_to_bbox(target_mask)
            polygon = mask_to_polygon(target_mask)
            area = float(bbox[2] * bbox[3])
        elif result.shape[2] == 4:
            # Fallback: use result alpha channel
            result_mask = result[:, :, 3]
            bbox = mask_to_bbox(result_mask)
            polygon = mask_to_polygon(result_mask)
            area = float(bbox[2] * bbox[3])
        else:
            # Fallback: use full image if no alpha channel
            bbox = [0, 0, result.shape[1], result.shape[0]]
            polygon = []
            area = float(result.shape[1] * result.shape[0])

        # Apply position offset to bbox and polygon
        pos_x = position_x if position_x is not None else 0
        pos_y = position_y if position_y is not None else 0

        adjusted_bbox = [
            bbox[0] + pos_x,  # x
            bbox[1] + pos_y,  # y
            bbox[2],  # width (unchanged)
            bbox[3],  # height (unchanged)
        ]

        adjusted_polygon = []
        for poly in polygon:
            adjusted_poly = []
            for i in range(0, len(poly), 2):
                adjusted_poly.append(poly[i] + pos_x)  # x
                adjusted_poly.append(poly[i + 1] + pos_y)  # y
            adjusted_polygon.append(adjusted_poly)

        manager = COCOMetadataManager()
        manager.add_category("insect")

        manager.add_annotation(
            image_id=1,
            category_id=1,
            bbox=[int(x) for x in adjusted_bbox],
            segmentation=adjusted_polygon,
            area=area,
            scale_ratio=scale_ratio,
            rotation_angle=rotation_angle if rotation_angle is not None else 0.0,
        )

        xml_content = manager.to_voc_xml(
            output_filename,
            result.shape[1],
            result.shape[0],
            segmentation=adjusted_polygon if adjusted_polygon else None,
        )

        annotations_path = annotations_dir / f"{_annotation_stem(output_filename)}.xml"
        annotations_path.parent.mkdir(parents=True, exist_ok=True)

        with open(annotations_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

    def _save_yolo_single(
        self,
        output_filename: str,
        result: np.ndarray,
        scale_ratio: float,
        rotation_angle: Optional[float],
        position_x: Optional[int],
        position_y: Optional[int],
        output_dir: Optional[Path] = None,
        target_rgba: Optional[np.ndarray] = None,
    ) -> None:
        """Save single YOLO TXT file per image."""
        if output_dir:
            labels_dir = self._get_annotation_output_dir(output_dir)
        else:
            labels_dir = self._get_annotation_output_dir(
                Path(self.output_subdir).parent
            )
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Calculate bbox and segmentation from target_rgba alpha channel
        if target_rgba is not None and target_rgba.shape[2] == 4:
            target_mask = target_rgba[:, :, 3]
            bbox = mask_to_bbox(target_mask)
            polygon = mask_to_polygon(target_mask)
            area = float(bbox[2] * bbox[3])
        elif result.shape[2] == 4:
            # Fallback: use result alpha channel
            result_mask = result[:, :, 3]
            bbox = mask_to_bbox(result_mask)
            polygon = mask_to_polygon(result_mask)
            area = float(bbox[2] * bbox[3])
        else:
            # Fallback: use full image if no alpha channel
            bbox = [0, 0, result.shape[1], result.shape[0]]
            polygon = []
            area = float(result.shape[1] * result.shape[0])

        # Apply position offset to bbox and polygon
        pos_x = position_x if position_x is not None else 0
        pos_y = position_y if position_y is not None else 0

        adjusted_bbox = [
            bbox[0] + pos_x,  # x
            bbox[1] + pos_y,  # y
            bbox[2],  # width (unchanged)
            bbox[3],  # height (unchanged)
        ]

        adjusted_polygon = []
        for poly in polygon:
            adjusted_poly = []
            for i in range(0, len(poly), 2):
                adjusted_poly.append(poly[i] + pos_x)  # x
                adjusted_poly.append(poly[i + 1] + pos_y)  # y
            adjusted_polygon.append(adjusted_poly)

        manager = COCOMetadataManager()
        manager.add_category("insect")

        manager.add_annotation(
            image_id=1,
            category_id=1,
            bbox=[int(x) for x in adjusted_bbox],
            segmentation=adjusted_polygon,
            area=area,
            scale_ratio=scale_ratio,
            rotation_angle=rotation_angle if rotation_angle is not None else 0.0,
        )

        yolo_content = manager.to_yolo_txt(
            result.shape[1],
            result.shape[0],
            segmentation=adjusted_polygon if adjusted_polygon else None,
        )

        labels_path = labels_dir / f"{_annotation_stem(output_filename)}.txt"
        labels_path.parent.mkdir(parents=True, exist_ok=True)

        with open(labels_path, "w", encoding="utf-8") as f:
            f.write(yolo_content)

        yaml_dir = output_dir if output_dir else Path(self.output_subdir).parent
        yaml_path = yaml_dir / "data.yaml"
        yaml_path.write_text(
            'train: images\nnc: 1\nnames: ["insect"]\n', encoding="utf-8"
        )

    def _get_annotation_output_dir(self, output_dir: Path) -> Path:
        """Get annotation output directory based on format (detcli-aligned paths)."""
        if self.annotation_format == "yolo":
            return output_dir / "labels"
        elif self.annotation_format == "voc":
            return output_dir / "Annotations"
        else:
            return output_dir / "annotations"

    def _remove_synthesis_annotation(
        self, output_dir: Path, rel_parent: Path, stem: str, index: int
    ) -> None:
        """Remove the per-synthesis annotation files for one stale copy.

        Unified COCO has no per-copy file; its stale entries are dropped from
        the prior payload via ``drop_coco_images()`` instead.
        """
        name = (rel_parent / f"{stem}_{index:02d}").as_posix()
        for directory, suffix in (
            ("Annotations", ".xml"),
            ("labels", ".txt"),
            ("annotations", ".json"),
        ):
            path = output_dir / directory / f"{name}{suffix}"
            if path.is_file():
                path.unlink()

    def _finalize_result(
        self,
        result,
        image_output_dir: Path,
        target_dir: Path,
        output_dir: Path,
        synthesis_id: int,
    ) -> None:
        """Write one synthesis and its annotation using target-relative paths."""
        target_path = Path(result[4]) if result[4] is not None else None
        rel_parent = Path()
        if target_path is not None:
            try:
                rel_parent = target_path.parent.relative_to(target_dir)
            except ValueError:
                rel_parent = Path()

        if result[1]:
            out_dir = image_output_dir / rel_parent
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{result[1]}.{self.output_format}"
            annotation_name = (rel_parent / result[1]).as_posix()
        else:
            output_path = (
                image_output_dir / f"synth_{synthesis_id:06d}.{self.output_format}"
            )
            annotation_name = f"synth_{synthesis_id:06d}"

        self._save_image(result[0], output_path)
        if result[1] is not None:
            self._save_annotation_for_image(
                output_filename=annotation_name,
                result=result[0],
                scale_ratio=result[2],
                rotation_angle=result[3],
                position_x=result[6] if len(result) > 6 else None,
                position_y=result[7] if len(result) > 7 else None,
                output_dir=output_dir,
                target_rgba=result[8] if len(result) > 8 else None,
            )

    def process_directory(
        self,
        target_dir: Path,
        background_dir: Path,
        output_dir: Path,
        num_syntheses: float = 1,
        disable_tqdm: bool = False,
        threads: int = 1,
        skip_existing: bool = False,
        shutdown_flag: Optional[Callable[[], bool]] = None,
        seed: int = 42,
    ) -> dict:
        """Process all images in directories with recursive scanning and seeded tasks."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_output_dir = output_dir / self.output_subdir
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the unified COCO file before this run overwrites it, so
        # resume-skipped samples keep their annotations.
        prior_coco = None
        if skip_existing and self.annotation_format == "coco":
            from src.common.annotation_writer import (
                coco_bbox_format_of,
                load_coco_json,
            )

            prior_coco = load_coco_json(output_dir / "annotations.coco.json")
            if prior_coco is not None and prior_coco.get("annotations"):
                prior_format = coco_bbox_format_of(prior_coco)
                if prior_format != self.coco_bbox_format:
                    raise ValueError(
                        "Cannot change --coco-bbox-format on --resume: existing "
                        f"annotations.coco.json uses {prior_format!r} but this run "
                        f"requested {self.coco_bbox_format!r}. Delete the output "
                        "directory or keep the original format."
                    )

        target_paths = iter_files(target_dir, IMAGE_EXTENSIONS)
        background_paths = iter_files(background_dir, IMAGE_EXTENSIONS)

        if not target_paths:
            logger.error("No target images found!")
            raise ValueError(f"No target images found in {target_dir}")
        if not background_paths:
            logger.error("No background images found!")
            raise ValueError(f"No background images found in {background_dir}")

        logger.info(f"Loaded {len(target_paths)} targets from {target_dir}")
        logger.info(f"Loaded {len(background_paths)} backgrounds from {background_dir}")

        selected_indices, per_target = resolve_num_syntheses(
            num_syntheses, len(target_paths), seed=seed
        )
        total_syntheses = len(selected_indices) * per_target
        synthesis_id = 0
        skipped_syntheses = 0
        target_stems = resolve_target_stems(target_paths, target_dir)
        stale_coco_file_names: set = set()
        # Leftover copies are removed only after this target actually produced a
        # new result, so a run that fails earlier (unreadable target or
        # background) leaves the previous output untouched.
        pending_stale_cleanup: dict = {}
        written_targets: set = set()

        tasks = []
        failed_targets = 0
        failed_modes: Counter = Counter()
        for target_idx in selected_indices:
            target_path = target_paths[target_idx]
            if shutdown_flag is not None and shutdown_flag():
                logger.info("Shutdown requested. Stopping synthesis.")
                break
            rel_parent = target_path.relative_to(target_dir).parent
            target_stem = target_stems[target_path]
            extension = f".{self.output_format}"
            target_out_dir = image_output_dir / rel_parent
            existing_outputs = _existing_synthesis_outputs(
                target_out_dir, target_stem, extension
            )
            expected_indices = set(range(1, per_target + 1))
            if skip_existing and set(existing_outputs) == expected_indices:
                skipped_syntheses += per_target
                continue
            try:
                target_img = self._load_target_image(target_path)
            except ValueError as e:
                failed_modes[_image_mode(target_path)] += 1
                logger.error(str(e))
                print(f"Error: {e}", file=__import__("sys").stderr)
                failed_targets += 1
                continue
            except Exception as e:
                failed_modes["unreadable"] += 1
                logger.warning(f"Failed to load {target_path}: {e}")
                failed_targets += 1
                continue

            # Defer cleanup of leftover copies from a larger --num-syntheses
            # until this target produces a replacement (see post-loop pass).
            stale_indices = sorted(set(existing_outputs) - expected_indices)
            if stale_indices:
                pending_stale_cleanup[str(target_path)] = [
                    (existing_outputs[index], rel_parent, target_stem, index)
                    for index in stale_indices
                ]
            target_seed = derive_seed(seed, target_idx)
            for syn_idx in range(per_target):
                task_seed = derive_seed(target_seed, syn_idx)
                task_rng = random.Random(task_seed)
                background_path = task_rng.choice(background_paths)
                try:
                    background = self._load_image(background_path)
                except Exception as e:
                    logger.warning(f"Failed to load background {background_path}: {e}")
                    continue

                scale_ratio = task_rng.uniform(
                    self.area_ratio_min, self.area_ratio_max
                )
                rotation_angle = (
                    task_rng.uniform(-self.rotate_degrees, self.rotate_degrees)
                    if self.rotate_degrees > 0
                    else None
                )
                tasks.append(
                    (
                        target_img,
                        background,
                        scale_ratio,
                        target_path,
                        syn_idx + 1,
                        background_path,
                        rotation_angle,
                        task_seed,
                        target_stem,
                    )
                )

        if failed_modes:
            mode_summary = ", ".join(
                f"{count} {mode}" for mode, count in failed_modes.most_common()
            )
            logger.warning(
                f"{failed_targets} target image(s) failed to load by mode: "
                f"{mode_summary}"
            )

        if failed_targets > 0 and len(tasks) == 0:
            mode_summary = ", ".join(
                f"{count} {mode}" for mode, count in failed_modes.most_common()
            )
            raise ValueError(
                f"Synthesis produced no tasks: {failed_targets} of "
                f"{len(selected_indices)} selected target image(s) failed to load "
                f"(observed modes: {mode_summary or 'unknown'}). "
                "Synthesis requires RGBA cutouts with an alpha channel. Use "
                "mask-mode segmentation output (segment without '-bbox', e.g. "
                "--segmentation-method sam3|otsu|grabcut) or another RGBA source; "
                "bbox-mode crop output and raw photos are RGB."
            )

        def _consume(result) -> None:
            nonlocal synthesis_id
            if result[0] is None:
                # Failed synthesis: counted by the ``failed`` total below.
                return
            self._finalize_result(
                result, image_output_dir, target_dir, output_dir, synthesis_id
            )
            if result[4] is not None:
                written_targets.add(str(result[4]))
            synthesis_id += 1

        if threads > 1 and len(tasks) > 0:
            with multiprocessing.Pool(processes=threads) as pool:
                if TQDM_AVAILABLE and not disable_tqdm:
                    results = tqdm(
                        pool.imap(self._synthesize_single_wrapper, tasks),
                        total=len(tasks),
                        desc="Synthesizing",
                    )
                else:
                    results = pool.imap(self._synthesize_single_wrapper, tasks)
                for result in results:
                    _consume(result)
        else:
            iterator = tasks
            if TQDM_AVAILABLE and not disable_tqdm:
                iterator = tqdm(tasks, desc="Synthesizing")
            for task in iterator:
                result = self.synthesize_single(
                    task[0],
                    task[1],
                    task[2],
                    task[3],
                    task[4],
                    task[5],
                    rotation_angle=task[6],
                    np_seed=task[7],
                    target_stem=task[8],
                )
                _consume(result)

        # Remove leftover copies only for targets that produced a new result.
        for target_key in written_targets:
            for path, rel_parent, stem, index in pending_stale_cleanup.pop(
                target_key, []
            ):
                if path.exists():
                    path.unlink()
                self._remove_synthesis_annotation(output_dir, rel_parent, stem, index)
                stale_coco_file_names.add(f"{stem}_{index:02d}.{self.output_format}")

        # COCO: flush accumulated detections via annotation_writer once
        if self.annotation_format == "coco" and self._ann_image_paths:
            from src.common.annotation_writer import (
                merge_coco_json,
                write_annotations,
            )

            write_annotations(
                image_paths=self._ann_image_paths,
                detections_per_image=self._ann_detections,
                class_names=["insect"],
                out_dir=output_dir,
                fmt="coco",
                coco_bbox_format=self.coco_bbox_format,
            )
            # Merge after the bbox rewrite; write_annotations() recorded this
            # run's convention, and a resume with a different one is rejected
            # above before any synthesis is created.
            if prior_coco:
                if stale_coco_file_names:
                    from src.common.annotation_writer import drop_coco_images

                    prior_coco = drop_coco_images(prior_coco, stale_coco_file_names)
                merge_coco_json(prior_coco, output_dir / "annotations.coco.json")
            logger.info(
                f"Saved COCO annotations to {output_dir / 'annotations.coco.json'}"
            )
            self._ann_image_paths = []
            self._ann_detections = {}

        created_tasks = len(tasks)
        return {
            "processed": synthesis_id,
            "failed": created_tasks - synthesis_id,
            "uncreated": total_syntheses - skipped_syntheses - created_tasks,
            "output_files": synthesis_id,
            "skipped": skipped_syntheses,
        }
