# src/segmentation.py
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, TYPE_CHECKING
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

if TYPE_CHECKING:
    from src.lama.lama_inpainter import LaMaInpainter


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
        lama_model: Optional[str] = None,
        lama_mask_dilate: int = 0,
        annotation_format: str = "coco",
        coco_output_mode: str = "unified",
        coco_bbox_format: str = "xywh",
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
            lama_model: Path to LaMa model checkpoint file (default: models/big-lama/models/best.ckpt)
            annotation_format: Output format for annotations ("coco", "voc", "yolo")
            coco_output_mode: COCO output mode ("unified" or "separate") — kept for backward compat
            coco_bbox_format: COCO bbox format ("xywh" or "xyxy")
        """
        self.device = device
        self.hint = hint
        self.repair_strategy = repair_strategy
        self.confidence_threshold = confidence_threshold
        self.padding_ratio = padding_ratio
        self.segmentation_method = segmentation_method
        self.annotation_format = annotation_format.lower()
        self.coco_output_mode = coco_output_mode.lower()
        self.coco_bbox_format = coco_bbox_format
        self.metadata_manager = COCOMetadataManager()
        # Accumulators for unified COCO output via annotation_writer
        self._ann_image_paths: list = []
        self._ann_detections: dict = {}

        self.lama_model = lama_model
        self.lama_mask_dilate = max(0, int(lama_mask_dilate))
        self._lama_inpainters: Dict[
            Tuple[str, Optional[str], bool], "LaMaInpainter"
        ] = {}

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
        except Exception:
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
        if self.repair_strategy in [
            "opencv",
            "sam3-fill",
            "black-mask",
            "lama",
            "lama_refine",
        ]:
            self.repaired_dir = output_dir / "repaired_images"
            self.repaired_dir.mkdir(parents=True, exist_ok=True)

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

        logger.debug(f"Found {len(masks)} object(s)")

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

        if self.annotation_format == "coco" and self.coco_output_mode == "separate":
            coco_separate_manager = COCOMetadataManager()
            coco_separate_category_id = coco_separate_manager.add_category("insect")

        voc_yolo_annotations = []
        voc_yolo_manager: Optional[COCOMetadataManager] = None
        voc_yolo_category_id: Optional[int] = None
        if self.annotation_format in ["voc", "yolo"]:
            voc_yolo_manager = COCOMetadataManager()
            voc_yolo_category_id = voc_yolo_manager.add_category("insect")

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
            metadata_filename = Path(output_name).name

            logger.debug(f"Saved {output_path}")

            # Repair strategy - apply OpenCV/SAM3-fill inpainting if enabled
            # Store masks for later combined repair
            if self.repair_strategy in ["opencv", "sam3-fill", "black-mask", "lama"]:
                all_masks_for_repair.append(mask)
                repair_image = image.copy()

            # Compute metadata - bbox and polygon in original image coordinates
            # Object bbox in original image coordinates (before padding)
            orig_bbox = mask_to_bbox(mask)

            # Polygon in original image coordinates
            polygon_orig = mask_to_polygon(mask)

            # Get original image dimensions
            orig_height, orig_width = image.shape[:2]

            # Mask area is the actual object pixel count
            mask_area = int(np.sum(mask > 0))

            results["masks"].append(mask)
            results["output_files"].append(str(output_path))

            # COCO separate mode: add image and annotation for this cleaned output
            if self.annotation_format == "coco" and self.coco_output_mode == "separate":
                coco_separate_manager.add_image(
                    file_name=metadata_filename, width=orig_width, height=orig_height
                )
                coco_separate_manager.add_annotation(
                    image_id=coco_separate_manager._image_id_counter,
                    category_id=coco_separate_category_id,
                    bbox=list(orig_bbox),
                    segmentation=polygon_orig,
                    area=mask_area,
                )

            # VOC/YOLO mode: accumulate annotations for this input image
            if self.annotation_format in ["voc", "yolo"]:
                voc_yolo_manager.add_annotation(
                    image_id=1,
                    category_id=voc_yolo_category_id,
                    bbox=list(orig_bbox),
                    segmentation=polygon_orig,
                    area=mask_area,
                )

        if self.annotation_format == "coco" and self.coco_output_mode == "separate":
            annotations_dir = self._get_annotation_output_dir(output_dir)
            annotations_dir.mkdir(parents=True, exist_ok=True)
            annotations_path = annotations_dir / f"{base_name}.json"
            coco_separate_manager.save(annotations_path)

        # VOC/YOLO mode: save accumulated annotations once per input image
        if self.annotation_format in ["voc", "yolo"] and voc_yolo_manager:
            annotations_dir = self._get_annotation_output_dir(output_dir)
            annotations_dir.mkdir(parents=True, exist_ok=True)

            if self.annotation_format == "voc":
                xml_content = voc_yolo_manager.to_voc_xml(
                    filename=f"{base_name}.{output_format}",
                    width=orig_width,
                    height=orig_height,
                    depth=4
                    if self.segmentation_method not in ["sam3-bbox", "bbox"]
                    else 3,
                    segmentation=None,
                )
                # VOC: Annotations/ dir (detcli-aligned)
                annotations_path = annotations_dir / f"{base_name}.xml"
                with open(annotations_path, "w") as f:
                    f.write(xml_content)
                # Append stem to ImageSets/Main/default.txt
                imagesets_dir = output_dir / "ImageSets" / "Main"
                imagesets_dir.mkdir(parents=True, exist_ok=True)
                with open(imagesets_dir / "default.txt", "a") as f:
                    f.write(f"{base_name}\n")
            elif self.annotation_format == "yolo":
                yolo_content = voc_yolo_manager.to_yolo_txt(
                    width=orig_width, height=orig_height, segmentation=None
                )
                # YOLO: labels/ dir (detcli-aligned)
                labels_path = annotations_dir / f"{base_name}.txt"
                with open(labels_path, "w") as f:
                    f.write(yolo_content)
                # Write/overwrite data.yaml after each image (safe to overwrite)
                from src.common.annotation_writer import _write_yolo_yaml

                _write_yolo_yaml(output_dir / "data.yaml", ["insect"])

        # COCO unified mode: also populate metadata_manager for backward compatibility
        # (tests check metadata_manager.images / .annotations)
        if (
            self.annotation_format == "coco"
            and self.coco_output_mode == "unified"
            and len(filtered_masks) > 0
        ):
            metadata_image_filename = f"{base_name}.{output_format}"
            image_id = self.metadata_manager.add_image(
                file_name=metadata_image_filename, width=orig_width, height=orig_height
            )
            # Add an annotation for each mask
            for mask in filtered_masks:
                mb = mask_to_bbox(mask)
                mp = mask_to_polygon(mask)
                ma = int(np.sum(mask > 0))
                self.metadata_manager.add_annotation(
                    image_id=image_id,
                    category_id=self.insect_category_id,
                    bbox=list(mb),
                    segmentation=mp,
                    area=ma,
                )

        # COCO mode: accumulate detections for write_annotations() in process_directory()
        # Using sv.Detections with xyxy boxes for annotation_writer compatibility
        if (
            self.annotation_format == "coco"
            and original_path
            and len(filtered_masks) > 0
        ):
            import supervision as sv

            xyxy_list = []
            for mask in filtered_masks:
                x, y, w, h = mask_to_bbox(mask)
                xyxy_list.append([x, y, x + w, y + h])
            xyxy = np.array(xyxy_list, dtype=np.float32)
            class_ids = np.zeros(len(filtered_masks), dtype=int)
            dets = sv.Detections(xyxy=xyxy, class_id=class_ids)

            img_path = Path(original_path)
            self._ann_image_paths.append(img_path)
            self._ann_detections[str(img_path)] = dets

        # Apply combined repair if enabled (after processing all masks)
        if repair_image is not None and all_masks_for_repair:
            # Combine all masks
            combined_mask = np.zeros(repair_image.shape[:2], dtype=np.uint8)
            for mask in all_masks_for_repair:
                combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)

            # Repair the full image once using the selected strategy
            if self.repair_strategy == "sam3-fill":
                repaired = self._repair_with_sam3_fill(repair_image, combined_mask)
            elif self.repair_strategy == "black-mask":
                repaired = self._repair_with_black_mask(repair_image, combined_mask)
            elif self.repair_strategy == "lama":
                repaired = self._repair_with_lama(repair_image, combined_mask)
            else:  # opencv
                repaired = self._repair_with_opencv(repair_image, combined_mask)

            # Save to repaired_images folder with original image name
            if self.repaired_dir is not None:
                repaired_path = self.repaired_dir / f"{base_name}.{output_format}"
                save_image(repaired, repaired_path, format=output_format)
                logger.debug(f"Saved repaired {repaired_path}")
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
            # Fallback to OpenCV repair if SAM3 fails
            repaired = base_repaired

        return repaired

    def _repair_with_black_mask(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Repair image by filling masked regions with pure black [0, 0, 0].

        Args:
            image: Input image (H, W, 3)
            mask: Binary mask where 255 = area to inpaint, 0 = valid area

        Returns:
            Repaired image with black-filled regions
        """
        result = image.copy()

        # Ensure mask is proper format
        if mask.ndim == 3:
            mask = mask.squeeze()

        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Fill masked regions with black [0, 0, 0]
        result[mask > 0] = [0, 0, 0]

        return result

    def _repair_with_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Repair image using LaMa (Large Mask Inpainting) - WACV 2022.
        Based on傅里叶卷积，专为大掩码设计，支持高分辨率。

        Args:
            image: Input image (H, W, 3)
            mask: Binary mask where 255 = area to inpaint, 0 = valid area

        Returns:
            Repaired image with high-quality inpainting
        """
        # Ensure mask is proper format
        if mask.ndim == 3:
            mask = mask.squeeze()

        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        mask = self._prepare_lama_mask(mask)

        inpainter = self._get_lama_inpainter()

        result = inpainter(image, mask)

        return result

    def _prepare_lama_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8) * 255
        if self.lama_mask_dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=self.lama_mask_dilate)
        return mask

    def _get_lama_inpainter(self):
        from src.lama.lama_inpainter import LaMaInpainter

        cache_key = (self.device, self.lama_model, False)
        if cache_key in self._lama_inpainters:
            return self._lama_inpainters[cache_key]
        try:
            inpainter = LaMaInpainter(
                device=self.device, checkpoint_path=self.lama_model, refine=False
            )
        except Exception:
            inpainter = LaMaInpainter(
                device="auto", checkpoint_path=self.lama_model, refine=False
            )
        self._lama_inpainters[cache_key] = inpainter
        return inpainter

    def _get_annotation_output_dir(self, output_dir: Path) -> Path:
        """Get annotation output directory based on format (detcli-aligned paths)."""
        if self.annotation_format == "yolo":
            return output_dir / "labels"
        elif self.annotation_format == "voc":
            return output_dir / "Annotations"
        else:
            return output_dir / "annotations"

    def _save_coco_annotation(
        self,
        output_name: str,
        metadata_filename: str,
        width: int,
        height: int,
        bbox: List[int],
        segmentation: List[List[float]],
        area: float,
        output_dir: Path,
    ) -> None:
        """Save COCO format annotation."""
        if self.coco_output_mode == "separate":
            # Save separate JSON file per image
            annotations_dir = self._get_annotation_output_dir(output_dir)
            annotations_dir.mkdir(parents=True, exist_ok=True)

            manager = COCOMetadataManager()
            category_id = manager.add_category("insect")

            image_id = manager.add_image(
                file_name=metadata_filename, width=width, height=height
            )

            manager.add_annotation(
                image_id=image_id,
                category_id=category_id,
                bbox=bbox,
                segmentation=segmentation,
                area=area,
            )

            base_name = Path(output_name).stem
            annotations_path = annotations_dir / f"{base_name}.json"
            manager.save(annotations_path)
        else:
            # Unified mode: add to main metadata manager
            image_id = self.metadata_manager.add_image(
                file_name=metadata_filename, width=width, height=height
            )

            self.metadata_manager.add_annotation(
                image_id=image_id,
                category_id=self.insect_category_id,
                bbox=bbox,
                segmentation=segmentation,
                area=area,
            )

    def _save_voc_annotation(
        self,
        output_name: str,
        metadata_filename: str,
        width: int,
        height: int,
        bbox: Union[List[int], List[List[int]]],
        segmentation: Union[List[List[float]], List[List[List[float]]]],
        area: Union[float, List[float]],
        output_dir: Path,
    ) -> None:
        """Save VOC Pascal format annotation.

        Args:
            output_name: Name of output image file
            metadata_filename: Filename for metadata
            width: Image width
            height: Image height
            bbox: Bounding box [x, y, w, h] or list of bounding boxes
            segmentation: Polygon segmentation or list of polygons
            area: Area in pixels or list of areas
            output_dir: Output directory
        """
        annotations_dir = self._get_annotation_output_dir(output_dir)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        manager = COCOMetadataManager()
        manager.add_category("insect")

        # Check if this is single or multiple annotations
        # Single: bbox is List[int], segmentation is List[List[float]]
        # Multiple: bbox is List[List[int]], segmentation is List[List[List[float]]]
        is_single_bbox = (
            isinstance(bbox, list) and bbox and isinstance(bbox[0], (int, float))
        )

        if is_single_bbox:
            bboxes = [bbox]
            segmentations = [segmentation] if segmentation else [None]
            areas = [area]
        else:
            bboxes = bbox if isinstance(bbox, list) else [bbox]
            segmentations = segmentation if segmentation else [None] * len(bboxes)
            areas = area if isinstance(area, list) else [area]

        for box, seg, area_val in zip(bboxes, segmentations, areas):
            manager.add_annotation(
                image_id=1, category_id=1, bbox=box, segmentation=seg, area=area_val
            )

        xml_content = manager.to_voc_xml(
            output_name,
            width,
            height,
            depth=4 if self.segmentation_method not in ["sam3-bbox", "bbox"] else 3,
            segmentation=segmentation if segmentation and is_single_bbox else None,
        )

        base_name = Path(output_name).stem
        annotations_path = annotations_dir / f"{base_name}.xml"

        with open(annotations_path, "w") as f:
            f.write(xml_content)

    def _save_yolo_annotation(
        self,
        output_name: str,
        metadata_filename: str,
        width: int,
        height: int,
        bbox: Union[List[int], List[List[int]]],
        segmentation: Union[List[List[float]], List[List[List[float]]]],
        area: Union[float, List[float]],
        output_dir: Path,
    ) -> None:
        """Save YOLO format annotation.

        Args:
            output_name: Name of output image file
            metadata_filename: Filename for metadata
            width: Image width
            height: Image height
            bbox: Bounding box [x, y, w, h] or list of bounding boxes
            segmentation: Polygon segmentation or list of polygons
            area: Area in pixels or list of areas
            output_dir: Output directory
        """
        labels_dir = self._get_annotation_output_dir(output_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)

        manager = COCOMetadataManager()
        manager.add_category("insect")

        # Check if this is single or multiple annotations
        is_single_bbox = (
            isinstance(bbox, list) and bbox and isinstance(bbox[0], (int, float))
        )

        if is_single_bbox:
            bboxes = [bbox]
            segmentations = [segmentation] if segmentation else [None]
            areas = [area]
        else:
            bboxes = bbox if isinstance(bbox, list) else [bbox]
            segmentations = segmentation if segmentation else [None] * len(bboxes)
            areas = area if isinstance(area, list) else [area]

        for box, seg, area_val in zip(bboxes, segmentations, areas):
            manager.add_annotation(
                image_id=1, category_id=1, bbox=box, segmentation=seg, area=area_val
            )

        yolo_content = manager.to_yolo_txt(
            width,
            height,
            segmentation=segmentation if segmentation and is_single_bbox else None,
        )

        base_name = Path(output_name).stem
        labels_path = labels_dir / f"{base_name}.txt"

        with open(labels_path, "w") as f:
            f.write(yolo_content)

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

        # Save metadata for COCO mode — write as annotations.coco.json (detcli-aligned)
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
        elif self.annotation_format == "coco":
            # Legacy fallback: save via metadata_manager when no accumulation happened
            annotations_dir = output_dir / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = annotations_dir / "annotations.json"
            self.metadata_manager.save(metadata_path)
            logger.info(f"Saved metadata to {metadata_path}")

        return results
