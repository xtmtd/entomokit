"""Synthesis processor for compositing target objects onto background images."""

import logging
import multiprocessing
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image
from skimage import exposure
from skimage.color import rgb2lab, lab2rgb

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
        """Load all images from directory."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = [
            p
            for p in directory.iterdir()
            if p.suffix.lower() in image_extensions and p.is_file()
        ]

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

    def _save_image(
        self, image: np.ndarray, output_path: Path, quality: int = 90
    ) -> None:
        """Save image to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if image.shape[2] == 4 and self.output_format == "jpg":
            img_pil = Image.fromarray(image[:, :, :3], mode="RGB")
        else:
            img_pil = Image.fromarray(
                image, mode="RGBA" if image.shape[2] == 4 else "RGB"
            )
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
    ) -> Tuple[int, int, float]:
        """Find random position with constraint (avoid edges and optionally black regions).

        Returns:
            Tuple of (x, y, scale_factor) where scale_factor is the downscale applied (1.0 if none)
        """
        bg_h, bg_w = background.shape[:2]
        target_h, target_w = target_shape[:2]
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
            x = np.random.randint(min_x, max_x) if max_x > min_x else min_x
            y = np.random.randint(min_y, max_y) if max_y > min_y else min_y
            return x, y, 1.0
        downscale_factor = 1.0
        while downscale_factor >= 0.1:
            scaled_target_h = int(target_h * downscale_factor)
            scaled_target_w = int(target_w * downscale_factor)
            for _ in range(max_attempts):
                x = np.random.randint(min_x, max_x) if max_x > min_x else min_x
                y = np.random.randint(min_y, max_y) if max_y > min_y else min_y
                if not self._is_region_black(
                    background, x, y, scaled_target_w, scaled_target_h
                ):
                    return x, y, downscale_factor
            downscale_factor -= 0.1
        x = np.random.randint(min_x, max_x) if max_x > min_x else min_x
        y = np.random.randint(min_y, max_y) if max_y > min_y else min_y
        return x, y, 1.0

    def _get_random_position_no_constraint(
        self,
        background: np.ndarray,
        target_shape: Tuple[int, ...],
        edge_margin: float = 0.1,
    ) -> Tuple[int, int]:
        """Get random position without black region checking."""
        bg_h, bg_w = background.shape[:2]
        target_h, target_w = target_shape[:2]
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
        x = np.random.randint(min_x, max_x) if max_x > min_x else min_x
        y = np.random.randint(min_y, max_y) if max_y > min_y else min_y
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
        self, image: np.ndarray, angle: Optional[float] = None
    ) -> np.ndarray:
        """Rotate image by random angle within specified degrees."""
        if self.rotate_degrees <= 0:
            return image
        if angle is None:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
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

    def _get_target_filename(self, target_image_path: Path, counter: int) -> str:
        """Generate output filename from target filename and counter."""
        base_name = target_image_path.stem
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

        Returns:
            Tuple of (result_image, output_filename, scale_ratio, rotation_angle, target_path, background_path, position_x, position_y, final_target)
        """
        try:
            if target_image.shape[2] == 4:
                mask = target_image[:, :, 3]
            else:
                mask = np.ones(target_image.shape[:2], dtype=np.uint8) * 255
            mask_area = int(np.sum(mask > 0))
            if scale_ratio is None:
                scale_ratio = random.uniform(self.area_ratio_min, self.area_ratio_max)
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
                angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
                logger.debug(f"Selected rotation angle: {angle:.2f} degrees")
            target_rotated = self._rotate_image(target_scaled, angle)
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
                target_rotated = self._rotate_image(target_scaled, angle)
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
                            background, (current_h, current_w, 4), edge_margin=0.05
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
                            background, final_target_for_check.shape, edge_margin=0.05
                        )
            else:
                x, y = self._get_random_position_no_constraint(
                    background, final_target_for_check.shape, edge_margin=0.05
                )
            result = self._paste_with_alpha(background, final_target_for_check, x, y)
            if self.color_match_strength > 0:
                result = self._match_lab_histograms(result, background)
            output_filename = None
            if target_path is not None and counter > 0:
                output_filename = self._get_target_filename(target_path, counter)

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
        target_img, background, scale_ratio, target_path, counter, background_path = (
            args
        )
        result = self.synthesize_single(
            target_img, background, scale_ratio, target_path, counter, background_path
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

        base_name = Path(output_filename).stem
        annotations_path = annotations_dir / f"{base_name}.json"
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

        base_name = Path(output_filename).stem
        annotations_path = annotations_dir / f"{base_name}.xml"

        with open(annotations_path, "w") as f:
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

        base_name = Path(output_filename).stem
        labels_path = labels_dir / f"{base_name}.txt"

        with open(labels_path, "w") as f:
            f.write(yolo_content)

        yaml_dir = output_dir if output_dir else Path(self.output_subdir).parent
        yaml_path = yaml_dir / "data.yaml"
        yaml_path.write_text('train: images\nnc: 1\nnames: ["insect"]\n')

    def _get_annotation_output_dir(self, output_dir: Path) -> Path:
        """Get annotation output directory based on format (detcli-aligned paths)."""
        if self.annotation_format == "yolo":
            return output_dir / "labels"
        elif self.annotation_format == "voc":
            return output_dir / "Annotations"
        else:
            return output_dir / "annotations"

    def process_directory(
        self,
        target_dir: Path,
        background_dir: Path,
        output_dir: Path,
        num_syntheses: int = 10,
        disable_tqdm: bool = False,
        threads: int = 1,
    ) -> dict:
        """Process all images in directories."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_output_dir = output_dir / self.output_subdir
        image_output_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        target_paths = [
            p
            for p in target_dir.iterdir()
            if p.suffix.lower() in image_extensions and p.is_file()
        ]
        background_paths = [
            p
            for p in background_dir.iterdir()
            if p.suffix.lower() in image_extensions and p.is_file()
        ]

        if not target_paths:
            logger.error("No target images found!")
            raise ValueError(f"No target images found in {target_dir}")
        if not background_paths:
            logger.error("No background images found!")
            raise ValueError(f"No background images found in {background_dir}")

        logger.info(f"Loaded {len(target_paths)} targets from {target_dir}")
        logger.info(f"Loaded {len(background_paths)} backgrounds from {background_dir}")

        total_syntheses = len(target_paths) * num_syntheses
        synthesis_id = 0
        skipped_images = 0

        tasks = []
        for target_path in target_paths:
            try:
                target_img = self._load_image(target_path)
            except Exception as e:
                logger.warning(f"Failed to load {target_path}: {e}")
                continue

            for syn_idx in range(num_syntheses):
                background_path = random.choice(background_paths)
                try:
                    background = self._load_image(background_path)
                except Exception as e:
                    logger.warning(f"Failed to load background {background_path}: {e}")
                    continue

                scale_ratio = random.uniform(self.area_ratio_min, self.area_ratio_max)
                tasks.append(
                    (
                        target_img,
                        background,
                        scale_ratio,
                        target_path,
                        syn_idx + 1,
                        background_path,
                    )
                )

        if threads > 1 and len(tasks) > 0:
            if TQDM_AVAILABLE and not disable_tqdm:
                with multiprocessing.Pool(processes=threads) as pool:
                    for result in tqdm(
                        pool.imap(self._synthesize_single_wrapper, tasks),
                        total=len(tasks),
                        desc="Synthesizing",
                    ):
                        if result[0] is None:
                            if result[1] is None:
                                skipped_images += 1
                            continue
                        if result[1]:
                            output_path = (
                                image_output_dir / f"{result[1]}.{self.output_format}"
                            )
                        else:
                            output_path = (
                                image_output_dir
                                / f"synth_{synthesis_id:06d}.{self.output_format}"
                            )
                        self._save_image(result[0], output_path)
                        if result[1] is not None:
                            self._save_annotation_for_image(
                                output_filename=result[1],
                                result=result[0],
                                scale_ratio=result[2],
                                rotation_angle=result[3],
                                position_x=result[6] if len(result) > 6 else None,
                                position_y=result[7] if len(result) > 7 else None,
                                output_dir=output_dir,
                                target_rgba=result[8] if len(result) > 8 else None,
                            )
                        synthesis_id += 1
            else:
                with multiprocessing.Pool(processes=threads) as pool:
                    results = pool.map(self._synthesize_single_wrapper, tasks)
                for result in results:
                    if result[0] is None:
                        if result[1] is None:
                            skipped_images += 1
                        continue
                    if result[1]:
                        output_path = (
                            image_output_dir / f"{result[1]}.{self.output_format}"
                        )
                    else:
                        output_path = (
                            image_output_dir
                            / f"synth_{synthesis_id:06d}.{self.output_format}"
                        )
                    self._save_image(result[0], output_path)
                    if result[1] is not None:
                        self._save_annotation_for_image(
                            output_filename=result[1],
                            result=result[0],
                            scale_ratio=result[2],
                            rotation_angle=result[3],
                            position_x=result[6] if len(result) > 6 else None,
                            position_y=result[7] if len(result) > 7 else None,
                            output_dir=output_dir,
                            target_rgba=result[8] if len(result) > 8 else None,
                        )
                    synthesis_id += 1
        else:
            if TQDM_AVAILABLE and not disable_tqdm:
                for task in tqdm(tasks, desc="Synthesizing"):
                    result = self.synthesize_single(
                        task[0], task[1], task[2], task[3], task[4], task[5]
                    )
                    if result[0] is None:
                        if result[1] is None:
                            skipped_images += 1
                        continue
                    if result[1]:
                        output_path = (
                            image_output_dir / f"{result[1]}.{self.output_format}"
                        )
                    else:
                        output_path = (
                            image_output_dir
                            / f"synth_{synthesis_id:06d}.{self.output_format}"
                        )
                    self._save_image(result[0], output_path)
                    if result[1] is not None:
                        self._save_annotation_for_image(
                            output_filename=result[1],
                            result=result[0],
                            scale_ratio=task[2],
                            rotation_angle=result[3],
                            position_x=result[6] if len(result) > 6 else None,
                            position_y=result[7] if len(result) > 7 else None,
                            output_dir=output_dir,
                            target_rgba=result[8] if len(result) > 8 else None,
                        )
                    synthesis_id += 1
            else:
                for task in tasks:
                    result = self.synthesize_single(
                        task[0], task[1], task[2], task[3], task[4], task[5]
                    )
                    if result[0] is None:
                        if result[1] is None:
                            skipped_images += 1
                        continue
                    if result[1]:
                        output_path = (
                            image_output_dir / f"{result[1]}.{self.output_format}"
                        )
                    else:
                        output_path = (
                            image_output_dir
                            / f"synth_{synthesis_id:06d}.{self.output_format}"
                        )
                    self._save_image(result[0], output_path)
                    if result[1] is not None:
                        self._save_annotation_for_image(
                            output_filename=result[1],
                            result=result[0],
                            scale_ratio=task[2],
                            rotation_angle=result[3],
                            position_x=result[6] if len(result) > 6 else None,
                            position_y=result[7] if len(result) > 7 else None,
                            output_dir=output_dir,
                            target_rgba=result[8] if len(result) > 8 else None,
                        )
                    synthesis_id += 1

        # COCO: flush accumulated detections via annotation_writer once
        if self.annotation_format == "coco" and self._ann_image_paths:
            from src.common.annotation_writer import write_annotations

            write_annotations(
                image_paths=self._ann_image_paths,
                detections_per_image=self._ann_detections,
                class_names=["insect"],
                out_dir=output_dir,
                fmt="coco",
                coco_bbox_format=self.coco_bbox_format,
            )
            logger.info(
                f"Saved COCO annotations to {output_dir / 'annotations.coco.json'}"
            )
            self._ann_image_paths = []
            self._ann_detections = {}

        return {
            "processed": synthesis_id,
            "failed": total_syntheses - synthesis_id - skipped_images,
            "output_files": synthesis_id,
            "skipped": skipped_images,
        }
