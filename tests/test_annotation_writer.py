"""Tests for src/common/annotation_writer.py"""

import json
import pytest
from pathlib import Path


def _make_dataset_and_write(
    tmp_path, fmt, coco_bbox_format="xywh", class_names=None
):
    """Helper: create minimal writer call with real supervision."""
    pytest.importorskip("cv2")
    sv = pytest.importorskip("supervision")
    pytest.importorskip("PIL")
    import numpy as np
    from src.common.annotation_writer import write_annotations
    from PIL import Image

    img_path = tmp_path / "images" / "test.jpg"
    img_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), (200, 100, 50)).save(img_path)

    dets = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        class_id=np.array([0]),
    )

    write_annotations(
        image_paths=[img_path],
        detections_per_image={str(img_path): dets},
        class_names=class_names or ["insect"],
        out_dir=tmp_path / "out",
        fmt=fmt,
        coco_bbox_format=coco_bbox_format,
    )
    return tmp_path / "out"


def test_coco_output_filename(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "coco")
    assert (out_dir / "annotations.coco.json").exists()


def test_coco_does_not_copy_source_images(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "coco")
    assert not (out_dir / "test.jpg").exists()


def test_coco_bbox_xywh_format(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "coco", coco_bbox_format="xywh")
    data = json.loads((out_dir / "annotations.coco.json").read_text())
    bbox = data["annotations"][0]["bbox"]
    # In xywh: [x, y, w, h] where w=h=40
    assert len(bbox) == 4
    assert bbox[2] == 40 and bbox[3] == 40  # w, h


def test_coco_bbox_xyxy_format(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "coco", coco_bbox_format="xyxy")
    data = json.loads((out_dir / "annotations.coco.json").read_text())
    bbox = data["annotations"][0]["bbox"]
    # In xyxy: [10, 10, 50, 50]
    assert bbox == [10, 10, 50, 50]


def test_yolo_layout(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "yolo")
    assert (out_dir / "images").is_dir()
    assert (out_dir / "labels").is_dir()
    assert (out_dir / "data.yaml").exists()


def test_yolo_yaml_quoted_names(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "yolo")
    content = (out_dir / "data.yaml").read_text()
    assert "train: images" in content
    assert 'names: ["insect"]' in content
    assert "nc: 1" in content


def test_yolo_yaml_written_as_utf8(tmp_path):
    from src.common.annotation_writer import _write_yolo_yaml

    yaml_path = tmp_path / "data.yaml"
    _write_yolo_yaml(yaml_path, ["昆虫"])
    raw = yaml_path.read_bytes()
    assert "昆虫".encode("utf-8") in raw


def test_coco_bbox_rewrite_writes_utf8(tmp_path):
    from src.common.annotation_writer import _rewrite_coco_bbox_to_xyxy

    json_path = tmp_path / "annotations.coco.json"
    json_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "昆虫"}],
                "annotations": [{"bbox": [10, 10, 40, 40]}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _rewrite_coco_bbox_to_xyxy(json_path)
    raw = json_path.read_bytes()
    assert "昆虫".encode("utf-8") in raw


def test_voc_layout(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "voc")
    assert (out_dir / "JPEGImages").is_dir()
    assert (out_dir / "Annotations").is_dir()
    assert (out_dir / "ImageSets" / "Main" / "default.txt").exists()


def test_invalid_format_raises(tmp_path):
    from src.common.annotation_writer import write_annotations

    with pytest.raises(ValueError, match="Unknown format"):
        write_annotations([], {}, [], tmp_path, fmt="csv")


def test_invalid_coco_bbox_format_raises(tmp_path):
    from src.common.annotation_writer import write_annotations

    with pytest.raises(ValueError, match="Unknown coco_bbox_format"):
        write_annotations([], {}, [], tmp_path, fmt="coco", coco_bbox_format="cwh")
