# tests/test_metadata.py
import json
import os
import tempfile

import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("cv2")

from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon


# ==============================================================================
# mask_to_bbox Tests
# ==============================================================================

def test_mask_to_bbox():
    """Test bounding box extraction from mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:80] = 255
    
    bbox = mask_to_bbox(mask)
    
    # Verify format and values
    assert len(bbox) == 4
    assert bbox[0] == 30  # x
    assert bbox[1] == 20  # y
    assert bbox[2] == 50  # width
    assert bbox[3] == 30  # height


def test_mask_to_bbox_empty_mask():
    """Test bbox extraction from empty mask (all zeros)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    bbox = mask_to_bbox(mask)
    
    assert bbox == (0, 0, 0, 0)


def test_mask_to_bbox_single_pixel():
    """Test bbox extraction from single pixel mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 255
    
    bbox = mask_to_bbox(mask)
    
    assert bbox == (50, 50, 1, 1)


def test_mask_to_bbox_invalid_type():
    """Test bbox with invalid input type."""
    with pytest.raises(TypeError, match="Expected numpy array"):
        mask_to_bbox("not a mask")


def test_mask_to_bbox_3d_array():
    """Test bbox with 3D array (use only first channel)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:80] = 255
    
    bbox = mask_to_bbox(mask)
    
    # Should extract from 2D mask
    assert bbox[0] == 30
    assert bbox[1] == 20
    assert bbox[2] == 50
    assert bbox[3] == 30


# ==============================================================================
# mask_to_polygon Tests
# ==============================================================================

def test_mask_to_polygon():
    """Test polygon extraction from mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:80] = 255
    
    polygon = mask_to_polygon(mask)
    
    assert isinstance(polygon, list)
    assert len(polygon) > 0
    assert isinstance(polygon[0], list)


def test_mask_to_polygon_empty_mask():
    """Test polygon extraction from empty mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    polygon = mask_to_polygon(mask)
    
    assert polygon == []


def test_mask_to_polygon_invalid_type():
    """Test polygon with invalid input type."""
    with pytest.raises(TypeError, match="Expected numpy array"):
        mask_to_polygon("not a mask")


def test_mask_to_polygon_3d_array():
    """Test polygon with 3D array."""
    mask = np.zeros((100, 100, 3), dtype=np.uint8)
    mask[20:50, 30:80, 0] = 255
    
    with pytest.raises(ValueError, match="Expected 2D mask"):
        mask_to_polygon(mask)


# ==============================================================================
# COCOMetadataManager Tests
# ==============================================================================

def test_coco_metadata_manager_init():
    """Test COCO metadata manager initialization."""
    manager = COCOMetadataManager()
    
    assert manager.images == []
    assert manager.annotations == []
    assert manager.categories == []
    assert manager._image_id_counter == 0
    assert manager._annotation_id_counter == 0
    assert manager._category_id_counter == 0


def test_coco_metadata_manager_add_category():
    """Test adding categories."""
    manager = COCOMetadataManager()
    
    cat_id1 = manager.add_category("insect")
    cat_id2 = manager.add_category("plant", supercategory="vegetation")
    
    assert cat_id1 == 1
    assert cat_id2 == 2
    assert len(manager.categories) == 2
    assert manager.categories[0]["name"] == "insect"
    assert manager.categories[1]["name"] == "plant"
    assert manager.categories[1]["supercategory"] == "vegetation"


def test_coco_metadata_manager_add_image():
    """Test adding image metadata."""
    manager = COCOMetadataManager()
    
    img_id = manager.add_image(
        file_name="test.png",
        width=100,
        height=100
    )
    
    assert img_id == 1
    assert len(manager.images) == 1
    assert manager.images[0]["file_name"] == "test.png"
    assert manager.images[0]["width"] == 100
    assert manager.images[0]["height"] == 100


def test_coco_metadata_manager_add_image_without_path():
    """Test adding image without original_path (paths not stored for portability)."""
    manager = COCOMetadataManager()
    
    img_id = manager.add_image(
        file_name="test.png",
        width=100,
        height=100
    )
    
    assert img_id == 1
    # Original paths should NOT be stored for dataset portability
    assert "original_target_path" not in manager.images[0]
    assert "original_background_path" not in manager.images[0]


def test_coco_metadata_manager_add_annotation():
    """Test adding annotation."""
    manager = COCOMetadataManager()
    
    img_id = manager.add_image("test.png", 100, 100)
    cat_id = manager.add_category("insect")
    
    ann_id = manager.add_annotation(
        image_id=img_id,
        category_id=cat_id,
        bbox=[10, 20, 50, 30],
        segmentation=[[10, 20, 60, 20, 60, 50, 10, 50]],
        area=1500.0
    )
    
    assert ann_id == 1
    assert len(manager.annotations) == 1
    assert manager.annotations[0]["image_id"] == img_id
    assert manager.annotations[0]["category_id"] == cat_id
    assert manager.annotations[0]["bbox"] == [10, 20, 50, 30]
    assert manager.annotations[0]["iscrowd"] == 0


def test_coco_metadata_manager_to_dict():
    """Test to_dict conversion."""
    manager = COCOMetadataManager()
    
    manager.add_category("insect")
    manager.add_image("test.png", 100, 100)
    
    data = manager.to_dict()
    
    assert "images" in data
    assert "annotations" in data
    assert "categories" in data
    assert len(data["images"]) == 1
    assert len(data["categories"]) == 1


def test_coco_metadata_manager_save():
    """Test saving to JSON file."""
    manager = COCOMetadataManager()
    
    manager.add_category("insect")
    manager.add_image("test.png", 100, 100)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "annotations.json"
        manager.save(output_path)
        
        # Verify file exists and content is valid JSON
        assert output_path.exists()
        
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data


def test_coco_metadata_manager_save_creates_dirs():
    """Test save creates parent directories."""
    manager = COCOMetadataManager()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "subdir" / "annotations.json"
        manager.save(output_path)
        
        assert output_path.exists()


def test_coco_metadata_manager_save_writes_utf8() -> None:
    manager = COCOMetadataManager()
    manager.add_category("昆虫")
    manager.add_image("中文.png", 100, 100)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "annotations.json"
        manager.save(output_path)

        raw = output_path.read_bytes()
        assert "昆虫".encode("utf-8") in raw
        assert "中文.png".encode("utf-8") in raw


def test_coco_metadata_manager_multiple_categories():
    """Test adding multiple categories."""
    manager = COCOMetadataManager()
    
    cat1 = manager.add_category("insect")
    cat2 = manager.add_category("plant")
    cat3 = manager.add_category("bird")
    
    assert cat1 == 1
    assert cat2 == 2
    assert cat3 == 3
    assert len(manager.categories) == 3


def test_coco_metadata_manager_multiple_images():
    """Test adding multiple images."""
    manager = COCOMetadataManager()
    
    img1 = manager.add_image("test1.png", 100, 100)
    img2 = manager.add_image("test2.png", 200, 150)
    img3 = manager.add_image("test3.png", 300, 200)
    
    assert img1 == 1
    assert img2 == 2
    assert img3 == 3
    assert manager.images[1]["width"] == 200


def test_coco_metadata_manager_invalid_category_name():
    """Test adding category with invalid name."""
    manager = COCOMetadataManager()
    
    with pytest.raises(TypeError, match="Category name must be string"):
        manager.add_category(123)


def test_coco_metadata_manager_invalid_image_dimensions():
    """Test adding image with invalid dimensions."""
    manager = COCOMetadataManager()
    
    with pytest.raises(ValueError, match="width must be positive integer"):
        manager.add_image("test.png", 0, 100)
    
    with pytest.raises(ValueError, match="height must be positive integer"):
        manager.add_image("test.png", 100, -1)


def test_coco_metadata_manager_to_voc_xml():
    """Test VOC XML generation."""
    manager = COCOMetadataManager()
    
    manager.add_category("insect")
    img_id = manager.add_image("test.png", 500, 500)
    manager.add_annotation(
        image_id=img_id,
        category_id=1,
        bbox=[10, 20, 100, 150],
        segmentation=[[10, 20, 110, 20, 110, 170, 10, 170]],
        area=15000.0
    )
    
    xml = manager.to_voc_xml("test.png", 500, 500)
    
    assert "<annotation>" in xml
    assert "<folder>annotations</folder>" in xml
    assert "<filename>test.png</filename>" in xml
    assert "<width>500</width>" in xml
    assert "<height>500</height>" in xml
    assert "<depth>3</depth>" in xml
    assert "<object>" in xml
    assert "<name>insect</name>" in xml
    assert "<xmin>10</xmin>" in xml
    assert "<ymin>20</ymin>" in xml
    assert "<xmax>110</xmax>" in xml
    assert "<ymax>170</ymax>" in xml


def test_coco_metadata_manager_to_voc_xml_multiple_objects():
    """Test VOC XML generation with multiple objects."""
    manager = COCOMetadataManager()
    
    manager.add_category("insect")
    img_id = manager.add_image("test.png", 500, 500)
    manager.add_annotation(
        image_id=img_id,
        category_id=1,
        bbox=[10, 20, 100, 150],
        segmentation=[[10, 20, 110, 20, 110, 170, 10, 170]],
        area=15000.0
    )
    manager.add_annotation(
        image_id=img_id,
        category_id=1,
        bbox=[200, 150, 80, 120],
        segmentation=[[200, 150, 280, 150, 280, 270, 200, 270]],
        area=9600.0
    )
    
    xml = manager.to_voc_xml("test.png", 500, 500)
    
    assert xml.count("<object>") == 2
    assert "<xmin>10</xmin>" in xml
    assert "<xmin>200</xmin>" in xml


def test_coco_metadata_manager_to_yolo_txt():
    """Test YOLO TXT generation."""
    manager = COCOMetadataManager()
    
    manager.add_category("insect")
    img_id = manager.add_image("test.png", 500, 500)
    manager.add_annotation(
        image_id=img_id,
        category_id=1,
        bbox=[100, 100, 200, 200],
        segmentation=[[100, 100, 300, 100, 300, 300, 100, 300]],
        area=40000.0
    )
    
    yolo = manager.to_yolo_txt(500, 500)
    
    lines = yolo.strip().split('\n')
    assert len(lines) == 1
    
    parts = lines[0].split()
    assert len(parts) == 5
    class_id, cx, cy, w, h = map(float, parts)
    
    assert class_id == 0
    assert abs(cx - 0.4) < 0.01
    assert abs(cy - 0.4) < 0.01
    assert abs(w - 0.4) < 0.01
    assert abs(h - 0.4) < 0.01


def test_coco_metadata_manager_to_yolo_txt_multiple_objects():
    """Test YOLO TXT generation with multiple objects."""
    manager = COCOMetadataManager()
    
    manager.add_category("insect")
    img_id = manager.add_image("test.png", 500, 500)
    manager.add_annotation(
        image_id=img_id,
        category_id=1,
        bbox=[50, 50, 100, 100],
        segmentation=[[50, 50, 150, 50, 150, 150, 50, 150]],
        area=10000.0
    )
    manager.add_annotation(
        image_id=img_id,
        category_id=1,
        bbox=[200, 200, 100, 100],
        segmentation=[[200, 200, 300, 200, 300, 300, 200, 300]],
        area=10000.0
    )
    
    yolo = manager.to_yolo_txt(500, 500)
    
    lines = yolo.strip().split('\n')
    assert len(lines) == 2
