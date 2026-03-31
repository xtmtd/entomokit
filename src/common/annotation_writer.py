"""Unified annotation writer for COCO / YOLO / VOC formats.

Layout conventions match detcli:
  COCO  → flat dir, file named 'annotations.coco.json', bbox xywh by default
  YOLO  → images/ + labels/ + data.yaml (quoted names)
  VOC   → JPEGImages/ + Annotations/ + ImageSets/Main/default.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

SUPPORTED_FORMATS = {"coco", "yolo", "voc"}

# Valid bbox format choices for COCO output
COCO_BBOX_FORMATS = {"xywh", "xyxy"}


def write_annotations(
    image_paths: List[Path],
    detections_per_image: Dict[str, "sv.Detections"],
    class_names: List[str],
    out_dir: Path,
    fmt: str,
    coco_bbox_format: str = "xywh",
) -> None:
    """Write a detection dataset to *out_dir* in the requested *fmt*.

    Args:
        image_paths: Ordered list of source image paths.
        detections_per_image: Mapping from image path str → sv.Detections.
        class_names: Ordered list of class names (index == class_id).
        out_dir: Destination directory (will be created).
        fmt: One of 'coco', 'yolo', 'voc'.
        coco_bbox_format: 'xywh' or 'xyxy' (COCO only).
    """
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unknown format: {fmt!r}. Supported: {sorted(SUPPORTED_FORMATS)}"
        )
    if coco_bbox_format not in COCO_BBOX_FORMATS:
        raise ValueError(
            f"Unknown coco_bbox_format: {coco_bbox_format!r}. Use 'xywh' or 'xyxy'."
        )

    import cv2
    import supervision as sv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build images dict: str path → np.ndarray (required by supervision)
    images_dict: Dict[str, "np.ndarray"] = {}
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images_dict[str(p)] = img

    annotations_dict = {
        str(p): detections_per_image.get(str(p), sv.Detections.empty())
        for p in image_paths
    }

    dataset = sv.DetectionDataset(
        classes=class_names,
        images=images_dict,
        annotations=annotations_dict,
    )

    if fmt == "coco":
        _save_coco(dataset, out_dir, coco_bbox_format)
    elif fmt == "yolo":
        _save_yolo(dataset, out_dir, class_names)
    elif fmt == "voc":
        _save_voc(dataset, out_dir)


def _save_coco(
    dataset: "sv.DetectionDataset",
    out_dir: Path,
    coco_bbox_format: str,
) -> None:
    """Save as COCO JSON (flat layout, fixed filename)."""
    anno_path = out_dir / "annotations.coco.json"
    dataset.as_coco(
        annotations_path=str(anno_path),
    )
    if coco_bbox_format == "xyxy":
        _rewrite_coco_bbox_to_xyxy(anno_path)


def _save_yolo(
    dataset: "sv.DetectionDataset",
    out_dir: Path,
    class_names: List[str],
) -> None:
    """Save as YOLO (images/ + labels/ + data.yaml with quoted names)."""
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    dataset.as_yolo(
        images_directory_path=str(images_dir),
        annotations_directory_path=str(labels_dir),
    )
    _write_yolo_yaml(out_dir / "data.yaml", class_names)


def _save_voc(dataset: "sv.DetectionDataset", out_dir: Path) -> None:
    """Save as Pascal VOC (JPEGImages/ + Annotations/ + ImageSets/Main/default.txt)."""
    jpeg_dir = out_dir / "JPEGImages"
    ann_dir = out_dir / "Annotations"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    dataset.as_pascal_voc(
        images_directory_path=str(jpeg_dir),
        annotations_directory_path=str(ann_dir),
    )
    stems = [Path(p).stem for p in dataset.image_paths]
    _write_voc_imagesets(out_dir, stems)


def _write_yolo_yaml(
    yaml_path: Path, class_names: List[str], train_path: str = "images"
) -> None:
    """Write data.yaml with quoted class names (matches detcli convention)."""
    deduped = list(dict.fromkeys(class_names))  # preserve order, remove dups
    quoted = ", ".join(f'"{c}"' for c in deduped)
    yaml_path.write_text(
        f"train: {train_path}\nnc: {len(deduped)}\nnames: [{quoted}]\n",
        encoding="utf-8",
    )


def _write_voc_imagesets(out_dir: Path, stems: List[str]) -> None:
    """Write ImageSets/Main/default.txt."""
    imagesets_dir = out_dir / "ImageSets" / "Main"
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    (imagesets_dir / "default.txt").write_text(
        "\n".join(stems) + "\n", encoding="utf-8"
    )


def _rewrite_coco_bbox_to_xyxy(json_path: Path) -> None:
    """In-place convert bbox in COCO JSON from xywh → xyxy."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        ann["bbox"] = [x, y, x + w, y + h]
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
