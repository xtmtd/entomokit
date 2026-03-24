# src/metadata.py
"""
COCO metadata utilities for the Insect Synthesizer project.

This module provides utilities for managing COCO format metadata:
- mask_to_bbox: Extract bounding box from binary mask
- mask_to_polygon: Extract polygon from binary mask
- COCOMetadataManager: Manage COCO format metadata

Example usage:
    >>> manager = COCOMetadataManager()
    >>> cat_id = manager.add_category("insect")
    >>> img_id = manager.add_image("test.png", 100, 100)
    >>> manager.add_annotation(img_id, cat_id, [10, 20, 50, 30], [[10, 20, 60, 20, 60, 50, 10, 50]], 1500, 1500)
    >>> manager.save("output/annotations.json")
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import cv2


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Extract bounding box from binary mask.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
    
    Returns:
        Bounding box (x, y, width, height)
    
    Raises:
        TypeError: If mask is not a numpy array
    
    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[20:50, 30:80] = 255
        >>> mask_to_bbox(mask)
        (30, 20, 50, 30)
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(mask)}")
    
    # Find non-zero pixels
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0:
        return (0, 0, 0, 0)
    
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    return (
        int(x_min),
        int(y_min),
        int(x_max - x_min + 1),
        int(y_max - y_min + 1)
    )


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    Extract polygon from binary mask.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
    
    Returns:
        List of polygon coordinates [[x1, y1, x2, y2, ...]]
    
    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[20:50, 30:80] = 255
        >>> polygon = mask_to_polygon(mask)
        >>> isinstance(polygon, list)
        True
    """
    # Validate mask
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(mask)}")
    
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.ndim}D")
    
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return []
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Flatten to [x1, y1, x2, y2, ...] format
    polygon = largest_contour.flatten().tolist()
    
    return [polygon]


class COCOMetadataManager:
    """Manager for COCO format metadata.
    
    This class manages COCO format metadata including images, annotations,
    and categories. It handles ID generation and provides methods to save
    the metadata to JSON files.
    
    Attributes:
        images: List of image metadata dictionaries
        annotations: List of annotation dictionaries
        categories: List of category dictionaries
    
    Example:
        >>> manager = COCOMetadataManager()
        >>> cat_id = manager.add_category("insect")
        >>> img_id = manager.add_image("test.png", 100, 100)
        >>> bbox = [10, 20, 50, 30]
        >>> seg = [[10, 20, 60, 20, 60, 50, 10, 50]]
        >>> manager.add_annotation(img_id, cat_id, bbox, seg, 1500, 1500)
        >>> manager.save("output/annotations.json")
    """
    
    def __init__(self):
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.categories: List[Dict[str, Any]] = []
        self._image_id_counter = 0
        self._annotation_id_counter = 0
        self._category_id_counter = 0
    
    def add_category(self, name: str, supercategory: str = "") -> int:
        """Add a category and return its ID.
        
        Args:
            name: Category name (e.g., "insect")
            supercategory: Optional supercategory name
        
        Returns:
            Category ID (1-indexed)
        
        Raises:
            TypeError: If name is not a string
        """
        if not isinstance(name, str):
            raise TypeError(f"Category name must be string, got {type(name)}")
        
        self._category_id_counter += 1
        category = {
            "id": self._category_id_counter,
            "name": name,
            "supercategory": supercategory
        }
        self.categories.append(category)
        return self._category_id_counter
    
    def add_image(
        self,
        file_name: str,
        width: int,
        height: int
    ) -> int:
        """Add image metadata and return its ID.
        
        Args:
            file_name: Image filename
            width: Image width in pixels
            height: Image height in pixels
        
        Returns:
            Image ID (1-indexed)
        
        Raises:
            TypeError: If file_name, width, or height are wrong types
            ValueError: If width or height are non-positive
        """
        if not isinstance(file_name, str):
            raise TypeError(f"file_name must be string, got {type(file_name)}")
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"width must be positive integer, got {width}")
        if not isinstance(height, int) or height <= 0:
            raise ValueError(f"height must be positive integer, got {height}")
        
        self._image_id_counter += 1
        image = {
            "id": self._image_id_counter,
            "file_name": Path(file_name).name,
            "width": width,
            "height": height
        }
        
        self.images.append(image)
        return self._image_id_counter
    
    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        bbox: List[int],
        segmentation: List[List[float]],
        area: float,
        scale_ratio: Optional[float] = None,
        rotation_angle: Optional[float] = None
    ) -> int:
        """Add annotation and return its ID.
        
        Args:
            image_id: Parent image ID
            category_id: Category ID
            bbox: Bounding box [x, y, width, height]
            segmentation: Polygon segmentation [[x1, y1, x2, y2, ...]]
            area: Area in pixels (mask pixel count for segmentation)
            scale_ratio: Optional scale ratio used for synthesis
            rotation_angle: Optional rotation angle in degrees
        
        Returns:
            Annotation ID (1-indexed)
        
        Raises:
            TypeError: If any parameter has wrong type
            ValueError: If IDs are non-positive
        """
        if not isinstance(image_id, int) or image_id <= 0:
            raise ValueError(f"image_id must be positive integer, got {image_id}")
        if not isinstance(category_id, int) or category_id <= 0:
            raise ValueError(f"category_id must be positive integer, got {category_id}")
        
        self._annotation_id_counter += 1
        annotation = {
            "id": self._annotation_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0
        }
        if scale_ratio is not None:
            annotation["scale_ratio"] = scale_ratio
        if rotation_angle is not None:
            annotation["rotation_angle"] = rotation_angle
        
        self.annotations.append(annotation)
        return self._annotation_id_counter
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to COCO format dictionary.
        
        Returns:
            Dictionary with images, annotations, and categories
        """
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save metadata to JSON file.
        
        Args:
            output_path: Output file path
        
        Raises:
            IOError: If file cannot be written
            TypeError: If output_path is not string or Path
        
        Example:
            >>> manager = COCOMetadataManager()
            >>> manager.save("output/annotations.json")
        """
        if not isinstance(output_path, (str, Path)):
            raise TypeError(f"output_path must be string or Path, got {type(output_path)}")
        
        output_path = Path(output_path)
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except OSError as e:
            raise IOError(f"Failed to save COCO metadata to {output_path}: {e}")
    
    def to_voc_xml(self, filename: str, width: int, height: int, depth: int = 3,
                   segmentation: Optional[List[List[float]]] = None) -> str:
        """Convert metadata to VOC XML format.
        
        Args:
            filename: Image filename
            width: Image width
            height: Image height
            depth: Image depth (default 3 for RGB)
            segmentation: Deprecated - kept for API compatibility but not used
        
        Returns:
            XML string in VOC Pascal format
        """
        xml_parts = [
            '<annotation>',
            f'    <folder>annotations</folder>',
            f'    <filename>{filename}</filename>',
            '    <size>',
            f'        <width>{width}</width>',
            f'        <height>{height}</height>',
            f'        <depth>{depth}</depth>',
            '    </size>',
        ]
        
        for idx, annotation in enumerate(self.annotations):
            bbox = annotation['bbox']
            xml_parts.extend([
                '    <object>',
                '        <name>insect</name>',
                '        <pose>Unspecified</pose>',
                '        <truncated>0</truncated>',
                '        <difficult>0</difficult>',
                '        <bndbox>',
                f'            <xmin>{bbox[0]}</xmin>',
                f'            <ymin>{bbox[1]}</ymin>',
                f'            <xmax>{bbox[0] + bbox[2]}</xmax>',
                f'            <ymax>{bbox[1] + bbox[3]}</ymax>',
                '        </bndbox>',
            ])
            
            xml_parts.append('    </object>')
        
        xml_parts.append('</annotation>')
        return '\n'.join(xml_parts)
    
    def to_yolo_txt(self, width: int, height: int,
                      segmentation: Optional[List[List[float]]] = None) -> str:
        """Convert metadata to YOLO TXT format.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            segmentation: Optional polygon segmentation [[x1, y1, x2, y2, ...]]
        
        Returns:
            YOLO format string (class_id x1 y1 x2 y2 ... for polygon, or class_id cx cy w h for bbox)
        """
        lines = []
        for idx, annotation in enumerate(self.annotations):
            bbox = annotation['bbox']
            class_id = 0
            
            if segmentation and idx < len(segmentation) and segmentation[idx]:
                poly = segmentation[idx]  # [x1, y1, x2, y2, ...] in pixels
                line = f"{class_id}"
                for i in range(0, len(poly), 2):
                    x_norm = poly[i] / width
                    y_norm = poly[i+1] / height
                    line += f" {x_norm:.6f} {y_norm:.6f}"
                lines.append(line)
            else:
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                yolo_w = bbox[2] / width
                yolo_h = bbox[3] / height
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}")
        
        return '\n'.join(lines)
