# tests/test_annotation_formats.py
"""Tests for VOC and YOLO annotation format support."""

import pytest
import numpy as np
from pathlib import Path
from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon


def test_coco_metadata_manager_to_voc_xml_standard_format():
    """Test VOC XML is standard Pascal VOC format without polygon extension."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")

    bbox = [100, 50, 100, 100]
    polygon = [[100, 50, 200, 50, 200, 150, 100, 150]]

    manager.add_annotation(
        image_id=1, category_id=cat_id, bbox=bbox, segmentation=polygon, area=10000
    )

    xml = manager.to_voc_xml("test.png", 640, 480, segmentation=polygon)

    assert "<polygon>" not in xml
    assert "<point>" not in xml
    assert "<bndbox>" in xml
    assert "<xmin>100</xmin>" in xml
    assert "<ymin>50</ymin>" in xml
    assert "<xmax>200</xmax>" in xml
    assert "<ymax>150</ymax>" in xml


def test_coco_metadata_manager_to_yolo_txt_with_polygon():
    """Test YOLO TXT includes polygon when segmentation provided."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")

    bbox = [100, 50, 100, 100]
    polygon = [[100, 50, 200, 50, 200, 150, 100, 150]]

    manager.add_annotation(
        image_id=1, category_id=cat_id, bbox=bbox, segmentation=polygon, area=10000
    )

    yolo_txt = manager.to_yolo_txt(width=640, height=480, segmentation=polygon)

    lines = yolo_txt.strip().split("\n")
    assert len(lines) == 1
    parts = lines[0].split()
    assert len(parts) == 9


def test_coco_metadata_manager_to_yolo_txt_fallback_to_bbox():
    """Test YOLO TXT falls back to bbox format when no polygon."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")

    bbox = [100, 50, 100, 100]

    manager.add_annotation(
        image_id=1, category_id=cat_id, bbox=bbox, segmentation=None, area=10000
    )

    yolo_txt = manager.to_yolo_txt(width=640, height=480, segmentation=None)

    lines = yolo_txt.strip().split("\n")
    assert len(lines) == 1
    parts = lines[0].split()
    assert len(parts) == 5


def test_coco_metadata_manager_to_voc_xml_without_polygon():
    """Test VOC XML works without polygon (backward compatibility)."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")

    bbox = [100, 50, 100, 100]

    manager.add_annotation(
        image_id=1, category_id=cat_id, bbox=bbox, segmentation=None, area=10000
    )

    xml = manager.to_voc_xml("test.png", 640, 480, segmentation=None)

    assert "<polygon>" not in xml
    assert "<bndbox>" in xml
    assert "<xmin>100</xmin>" in xml


def test_mask_to_polygon_empty_mask():
    """Test mask_to_polygon returns empty list for empty mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    polygon = mask_to_polygon(mask)
    assert polygon == []


def test_mask_to_polygon_single_pixel():
    """Test mask_to_polygon handles single pixel mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 255
    polygon = mask_to_polygon(mask)
    assert len(polygon) == 1
    assert len(polygon[0]) >= 2


# ==============================================================================
# VOC/YOLO Multi-Object Tests
# ==============================================================================


def test_voc_annotation_single_object():
    """Test VOC format with single object creates one file."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import numpy as np
    from src.segmentation.processor import SegmentationProcessor

    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[10:30, 10:30] = 255
        mock_wrapper.predict_with_scores.return_value = {
            "masks": [mock_mask],
            "scores": [0.95],
        }
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt", device="cpu", segmentation_method="sam3", annotation_format="voc"
        )

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.process_image(image=img, output_dir=tmpdir, base_name="test")

            # VOC uses detcli-aligned Annotations/ directory
            annotations_dir = Path(tmpdir) / "Annotations"
            xml_files = list(annotations_dir.glob("*.xml"))
            assert len(xml_files) == 1, f"Expected 1 XML file, got {len(xml_files)}"
            content = xml_files[0].read_text()
            assert content.count("<object>") == 1


def test_voc_annotation_multiple_objects():
    """Test VOC format with multiple objects creates one file with all objects."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import numpy as np
    from src.segmentation.processor import SegmentationProcessor

    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_mask1 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask1[10:30, 10:30] = 255
        mock_mask2 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask2[50:70, 50:70] = 255
        mock_mask3 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask3[80:90, 80:90] = 255
        mock_wrapper.predict_with_scores.return_value = {
            "masks": [mock_mask1, mock_mask2, mock_mask3],
            "scores": [0.95, 0.85, 0.75],
        }
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt", device="cpu", segmentation_method="sam3", annotation_format="voc"
        )

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.process_image(image=img, output_dir=tmpdir, base_name="test")

            # VOC uses detcli-aligned Annotations/ directory
            annotations_dir = Path(tmpdir) / "Annotations"
            xml_files = list(annotations_dir.glob("*.xml"))
            assert len(xml_files) == 1, (
                f"Expected 1 XML file per input image, got {len(xml_files)}"
            )
            content = xml_files[0].read_text()
            assert content.count("<object>") == 3, (
                f"Expected 3 objects in single XML file, got {content.count('<object>')}"
            )


def test_yolo_annotation_single_object():
    """Test YOLO format with single object creates one file."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import numpy as np
    from src.segmentation.processor import SegmentationProcessor

    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[10:30, 10:30] = 255
        mock_wrapper.predict_with_scores.return_value = {
            "masks": [mock_mask],
            "scores": [0.95],
        }
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt",
            device="cpu",
            segmentation_method="sam3",
            annotation_format="yolo",
        )

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.process_image(image=img, output_dir=tmpdir, base_name="test")

            labels_dir = Path(tmpdir) / "labels"
            txt_files = list(labels_dir.glob("*.txt"))
            assert len(txt_files) == 1, f"Expected 1 TXT file, got {len(txt_files)}"
            content = txt_files[0].read_text()
            line_count = len(content.strip().split("\n"))
            assert line_count == 1, f"Expected 1 annotation, got {line_count}"


def test_yolo_annotation_multiple_objects():
    """Test YOLO format with multiple objects creates one file with all annotations."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import numpy as np
    from src.segmentation.processor import SegmentationProcessor

    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_mask1 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask1[10:30, 10:30] = 255
        mock_mask2 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask2[50:70, 50:70] = 255
        mock_mask3 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask3[80:90, 80:90] = 255
        mock_wrapper.predict_with_scores.return_value = {
            "masks": [mock_mask1, mock_mask2, mock_mask3],
            "scores": [0.95, 0.85, 0.75],
        }
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt",
            device="cpu",
            segmentation_method="sam3",
            annotation_format="yolo",
        )

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.process_image(image=img, output_dir=tmpdir, base_name="test")

            labels_dir = Path(tmpdir) / "labels"
            txt_files = list(labels_dir.glob("*.txt"))
            assert len(txt_files) == 1, (
                f"Expected 1 TXT file per input image, got {len(txt_files)}"
            )
            content = txt_files[0].read_text()
            line_count = len(content.strip().split("\n"))
            assert line_count == 3, (
                f"Expected 3 annotations in single TXT file, got {line_count}"
            )
