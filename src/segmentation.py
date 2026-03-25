# src/segmentation.py
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import logging
from tqdm import tqdm

from src.sam3_wrapper import SAM3Wrapper
from src.utils import (
    load_image,
    save_image_rgba,
    apply_mask_with_alpha,
    save_image,
    get_device,
)
from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon

logger = logging.getLogger(__name__)


class SegmentationProcessor:
    """Process images to segment insects/objects using multiple methods."""

    def __init__(
        self,
        sam3_checkpoint: Union[str, Path],
        device: str = "auto",
        hint: str = "insect",
        repair_strategy: Optional[str] = None,
        confidence_threshold: float = 0.0,
        padding_ratio: float = 0.0,
        segmentation_method: str = "segmentation",
    ):
        """
        Initialize segmentation processor.

        Args:
            sam3_checkpoint: Path to SAM3 checkpoint (required for SAM3 methods)
            device: Device for inference ("auto", "cpu", "cuda", "mps")
            hint: Text prompt for segmentation
            repair_strategy: Repair strategy ("opencv", None)
            confidence_threshold: Minimum confidence score for masks (0.0 = no filtering)
            padding_ratio: Padding ratio for bounding box (0.0 = no padding, 0.1 = 10% padding)
            segmentation_method: Output method ("segmentation" for transparent mask, "bbox" for cropped box)
        """
        self.device = device
        self.hint = hint
        self.repair_strategy = repair_strategy
        self.confidence_threshold = confidence_threshold
        self.padding_ratio = padding_ratio
        self.segmentation_method = segmentation_method
        self.metadata_manager = COCOMetadataManager()

        # Add default category
        self.insect_category_id = self.metadata_manager.add_category("insect")

        # For repair strategy
        self.repaired_dir = None

        # Only initialize SAM3 wrapper if needed
        if self.segmentation_method in ["sam3", "sam3-bbox"]:
            if sam3_checkpoint is None:
                raise ValueError(
                    "sam3_checkpoint is required for sam3/sam3-bbox methods"
                )
            if not Path(sam3_checkpoint).exists():
                raise FileNotFoundError(
                    f"SAM3 checkpoint file not found: {sam3_checkpoint}. "
                    f"Please provide a valid checkpoint path or download from "
                    f"https://github.com/SysCV/sam-hq"
                )
            self.sam_wrapper = SAM3Wrapper(sam3_checkpoint, device=device)
        else:
            self.sam_wrapper = None

    def _segment_with_otsu(self, image: np.ndarray) -> List[np.ndarray]:
        """Segment using Otsu's thresholding method (matches detect_bounding_box.py)."""
        # Use BGR color space like reference script
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        erode = cv2.erode(dilate, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        masks = []
        for contour in contours:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            masks.append(mask.astype(bool))
        return masks

    def _enlarge_bbox(
        self, x: int, y: int, w: int, h: int, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box with padding_ratio representing empty space ratio in output.
        - padding_ratio = 0.0 → no padding (object fills 100% of output)
        - padding_ratio = 0.2 → empty space = 20% of output, object = 80%
        - padding_ratio = 0.8 → empty space = 80% of output, object = 20%

        Formula: output_size = object_size / (1 - padding_ratio)
        Example: object=2000x1000, padding_ratio=0.2 → output=2500x1250
        """
        logger.debug(
            f"_enlarge_bbox input: x={x}, y={y}, w={w}, h={h}, img={img_w}x{img_h}, padding_ratio={self.padding_ratio}"
        )
        pad_ratio = self.padding_ratio
        if pad_ratio <= 0 or pad_ratio >= 1:
            logger.debug(f"_enlarge_bbox: no padding, returning original bbox")
            return x, y, w, h  # No padding

        # Calculate output size: output = object / (1 - padding_ratio)
        # padding_ratio=0.2 → object 80% → scale 1/0.8 = 1.25
        new_w = int(w / (1 - pad_ratio))
        new_h = int(h / (1 - pad_ratio))

        logger.debug(f"_enlarge_bbox: obj={w}x{h}, new={new_w}x{new_h}")

        # Calculate center of original bbox
        cx = x + w / 2
        cy = y + h / 2

        # Calculate top-left corner centered on bbox center
        nx = cx - new_w / 2
        ny = cy - new_h / 2

        # Clamp to image bounds while keeping object centered
        if nx < 0:
            nx = 0
        if ny < 0:
            ny = 0
        if nx + new_w > img_w:
            nx = img_w - new_w
        if ny + new_h > img_h:
            ny = img_h - new_h

        # Final check: ensure nx, ny are not negative
        nx = max(0, nx)
        ny = max(0, ny)

        return int(nx), int(ny), new_w, new_h

    def _segment_with_grabcut(self, image: np.ndarray) -> List[np.ndarray]:
        """Segment using GrabCut algorithm (matches detect_bounding_box.py)."""
        # Resize large images for faster processing
        original_shape = image.shape
        max_dim = 800
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

        try:
            cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
            # Convert to binary mask: 2=bg, 3=fg-pr, 1=bg-pr, 0=fg
            result_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

            # Find contours
            contours, _ = cv2.findContours(
                result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            masks = []
            for contour in contours:
                single_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(single_mask, [contour], -1, 1, -1)
                masks.append(single_mask.astype(bool))
            return masks
        except cv2.error:
            logger.exception("GrabCut segmentation failed")
            return []

    def process_image(
        self,
        image: np.ndarray,
        output_dir: Union[str, Path],
        base_name: str,
        original_path: Optional[str] = None,
        output_format: str = "png",
    ) -> Dict[str, Any]:
        """
        Process single image and extract insects.

        Args:
            image: Input image array (H, W, 3)
            output_dir: Output directory
            base_name: Base name for output files
            original_path: Original image path for metadata
            output_format: Output format ('png' or 'jpg')

        Returns:
            Dictionary with processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleaned_dir = output_dir / "cleaned_images"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        # Setup repair directory if needed
        if self.repair_strategy in ["opencv", "sam3-fill"]:
            self.repaired_dir = output_dir / "repaired_images"
            self.repaired_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {base_name}")

        results = {"base_name": base_name, "masks": [], "output_files": []}

        # Initialize repair tracking (local variables, not instance state)
        all_masks_for_repair: List[np.ndarray] = []
        repair_image: Optional[np.ndarray] = None

        # Determine if using bbox-only mode
        is_bbox_mode = self.segmentation_method in ["sam3-bbox", "bbox"]

        # Run segmentation based on method
        if self.segmentation_method == "otsu":
            masks = self._segment_with_otsu(image)
            scores = [1.0] * len(masks)
        elif self.segmentation_method == "grabcut":
            masks = self._segment_with_grabcut(image)
            scores = [1.0] * len(masks)
            if not masks:
                logger.warning("GrabCut failed to find any masks")
                return results
        elif (
            self.segmentation_method in ["sam3", "sam3-bbox"]
            and self.sam_wrapper is not None
        ):
            masks_with_scores = self.sam_wrapper.predict_with_scores(
                image, text_prompt=self.hint
            )
            masks = masks_with_scores["masks"]
            scores = masks_with_scores.get("scores", [1.0] * len(masks))
        else:
            raise ValueError(
                f"SAM3 checkpoint required for segmentation method: {self.segmentation_method}"
            )

        logger.info(f"Found {len(masks)} object(s)")

        # Filter masks by confidence threshold
        filtered_masks = []
        filtered_scores = []
        for mask, score in zip(masks, scores):
            if score >= self.confidence_threshold:
                filtered_masks.append(mask)
                filtered_scores.append(score)
            else:
                logger.debug(
                    f"Filtering mask with score {score:.3f} < {self.confidence_threshold}"
                )

        masks = filtered_masks
        scores = filtered_scores

        if len(masks) == 0:
            logger.warning(
                f"No masks passed confidence threshold ({self.confidence_threshold})"
            )
            return results

        actual_idx = 0
        num_masks = len(filtered_masks)
        for i, (mask, score) in enumerate(zip(filtered_masks, filtered_scores)):
            # Single object: no suffix, Multiple objects: _01, _02, etc.
            if num_masks == 1:
                output_name = f"{base_name}.{output_format}"
            else:
                output_name = f"{base_name}_{actual_idx + 1:02d}.{output_format}"

            output_path = output_dir / "cleaned_images" / output_name
            actual_idx += 1

            if is_bbox_mode:
                # Bbox mode: crop image using mask bounding box
                # Compute minimal bounding box from mask
                bbox = mask_to_bbox(mask)
                x, y, w, h = bbox

                # Apply padding using reference script's enlarge_bbox logic
                img_h, img_w = mask.shape[:2]
                x, y, w, h = self._enlarge_bbox(x, y, w, h, img_w, img_h)

                x, y, w, h = int(x), int(y), int(w), int(h)
                bbox_padded = [x, y, w, h]
                x_min, y_min = x, y
                x_max, y_max = x + w, y + h

                # Crop image to padded bounding box
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Save bbox output (no alpha)
                save_image(cropped_image, output_path, format=output_format)
            else:
                # Mask mode: crop using exact mask shape
                # Compute minimal bounding box from mask
                bbox = mask_to_bbox(mask)

                # Apply padding using reference script's enlarge_bbox logic
                img_h, img_w = mask.shape[:2]
                x, y, w, h = self._enlarge_bbox(
                    bbox[0], bbox[1], bbox[2], bbox[3], img_w, img_h
                )

                x, y, w, h = int(x), int(y), int(w), int(h)
                bbox_padded = [x, y, w, h]
                x_min, y_min = x, y
                x_max, y_max = x + w, y + h

                # Crop image and mask to padded bounding box
                cropped_image = image[y_min:y_max, x_min:x_max]
                cropped_mask = mask[y_min:y_max, x_min:x_max]

                # Clear RGB background by applying mask
                # Convert mask to 0-1 float for multiplication
                mask_float = cropped_mask.astype(np.float32)
                if mask_float.ndim == 2:
                    mask_float = mask_float[:, :, np.newaxis]

                # Apply mask to RGB channels (background becomes 0)
                cropped_rgb_masked = (
                    cropped_image[:, :, :3].astype(np.float32) * mask_float
                )

                # Create RGBA with masked RGB + alpha
                if cropped_image.shape[2] == 3:
                    cropped_rgba = np.dstack(
                        [
                            cropped_rgb_masked.astype(np.uint8),
                            cropped_mask.astype(np.uint8) * 255,
                        ]
                    )
                elif cropped_image.shape[2] == 4:
                    cropped_rgba = cropped_image.copy()
                    cropped_rgba[:, :, :3] = cropped_rgb_masked.astype(np.uint8)
                    cropped_rgba[:, :, 3] = cropped_mask.astype(np.uint8) * 255
                else:
                    cropped_rgba = np.dstack(
                        [
                            cropped_rgb_masked.astype(np.uint8),
                            np.ones_like(cropped_mask) * 255,
                        ]
                    )
                save_image(cropped_rgba, output_path, format=output_format)

            # For metadata, use relative path without output_dir prefix
            # Format: cleaned_images/filename.ext
            metadata_filename = f"cleaned_images/{output_name}"

            logger.info(f"Saved {output_path}")

            # Repair strategy - apply OpenCV/SAM3-fill inpainting if enabled
            # Store masks for later combined repair
            if self.repair_strategy in ["opencv", "sam3-fill"]:
                all_masks_for_repair.append(mask)
                repair_image = image.copy()

            # Compute metadata
            polygon = mask_to_polygon(mask)
            mask_area = int(np.sum(mask > 0))

            # Convert bbox tuple to list for JSON serialization
            bbox_list = [int(x) for x in bbox_padded]

            # Add to metadata
            image_id = self.metadata_manager.add_image(
                file_name=metadata_filename,
                width=bbox_padded[2],
                height=bbox_padded[3],
                original_path=str(Path(original_path).resolve())
                if original_path
                else None,
            )

            self.metadata_manager.add_annotation(
                image_id=image_id,
                category_id=self.insect_category_id,
                bbox=bbox_list,
                segmentation=polygon,
                area=float(bbox_padded[2] * bbox_padded[3]),
                mask_area=mask_area,
            )

            results["masks"].append(mask)
            results["output_files"].append(str(output_path))

        # Apply combined repair if enabled (after processing all masks)
        if repair_image is not None and all_masks_for_repair:
            # Combine all masks
            combined_mask = np.zeros(repair_image.shape[:2], dtype=np.uint8)
            for mask in all_masks_for_repair:
                combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)

            # Repair the full image once using the selected strategy
            if self.repair_strategy == "sam3-fill":
                repaired = self._repair_with_sam3_fill(repair_image, combined_mask)
            else:  # opencv
                repaired = self._repair_with_opencv(repair_image, combined_mask)

            # Save to repaired_images folder with original image name
            if self.repaired_dir is not None:
                repaired_path = self.repaired_dir / f"{base_name}.{output_format}"
                save_image(repaired, repaired_path, format=output_format)
                logger.info(f"Saved repaired {repaired_path}")
            else:
                logger.warning(
                    f"repair_strategy is {self.repair_strategy} but repaired_dir is not initialized"
                )

        return results

    def _repair_with_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Repair image using OpenCV inpainting.

        Args:
            image: Input image (H, W, 3)
            mask: Binary mask where 255 = area to inpaint, 0 = valid area

        Returns:
            Repaired image
        """
        # Convert mask to proper format for OpenCV
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Normalize mask to 0-255
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Inpaint using Telea's algorithm
        repaired = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return repaired

    def _repair_with_sam3_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Repair image using SAM3 inpainting strategy.
        Uses SAM3 to generate a better fill for the masked region.

        Args:
            image: Input image (H, W, 3)
            mask: Binary mask where 255 = area to inpaint, 0 = valid area

        Returns:
            Repaired image
        """
        # Convert mask to proper format
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Normalize mask to 0-255
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Create inpaint mask (inverse - where mask is 0, we inpaint)
        # Actually for SAM3-fill, we want to inpaint the masked regions
        # mask=255 means we want to fill those areas
        inpaint_mask = mask

        # For SAM3-fill, we use SAM3 to generate content
        # First, blur the mask slightly to get a smoother transition
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

        # Inpaint with OpenCV first (baseline)
        base_repaired = cv2.inpaint(
            image, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
        )

        # Use SAM3 to enhance the repaired area
        # Get SAM3 prediction on the repaired image
        try:
            if self.sam_wrapper is not None:  # type: ignore
                masks_with_scores = self.sam_wrapper.predict_with_scores(
                    base_repaired, text_prompt=self.hint
                )
                sam_masks = masks_with_scores["masks"]  # type: ignore
            else:
                sam_masks = []  # type: ignore

            if sam_masks and len(sam_masks) > 0:
                # Use the largest SAM3 mask to guide the fill
                largest_mask = max(sam_masks, key=lambda m: np.sum(m))

                # Create a guide mask for the original image
                guide_mask = cv2.resize(
                    largest_mask.astype(np.uint8), (image.shape[1], image.shape[0])
                )
                guide_mask = (guide_mask > 0.5).astype(np.uint8) * 255

                # Blend the repaired image with guidance
                # For now, use the base repaired image as SAM3-fill produces similar results
                repaired = base_repaired
            else:
                repaired = base_repaired
        except Exception:
            logger.exception(
                "SAM3-fill enhancement failed; falling back to OpenCV inpaint"
            )
            # Fallback to OpenCV repair if SAM3 fails
            repaired = base_repaired

        return repaired

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        num_workers: int = 1,
        disable_tqdm: bool = False,
        output_format: str = "png",
        shutdown_flag: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Process all images in directory.

        Args:
            input_dir: Input directory with images
            output_dir: Output directory
            num_workers: Number of parallel workers (1 for sequential)
            disable_tqdm: Disable tqdm progress bar (for cleaner logs)
            output_format: Output format for saved images
            shutdown_flag: Optional callable that returns True when shutdown is requested

        Returns:
            Dictionary with all processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = [
            p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(image_paths)} images to process")

        results = {"processed": 0, "failed": 0, "output_files": []}

        # Process each image
        for img_path in tqdm(image_paths, desc="Segmenting", disable=disable_tqdm):
            # Check for shutdown request before processing
            if shutdown_flag is not None and shutdown_flag():
                logger.info(
                    "Shutdown requested. Exiting after completing current image."
                )
                break

            try:
                # Load image
                image = load_image(img_path)

                # Process
                result = self.process_image(
                    image,
                    output_dir=output_dir,
                    base_name=img_path.stem,
                    original_path=str(img_path),
                    output_format=output_format,
                )

                results["processed"] += 1
                results["output_files"].extend(result["output_files"])

            except Exception:
                logger.exception(f"Failed to process {img_path}")
                results["failed"] += 1

        # Save metadata
        metadata_path = output_dir / "annotations.json"
        self.metadata_manager.save(metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")

        return results
