"""Unified annotation writer for COCO / YOLO / VOC formats.

Layout conventions match detcli:
  COCO  → flat dir, file named 'annotations.coco.json', bbox xywh by default
  YOLO  → images/ + labels/ + data.yaml (quoted names)
  VOC   → JPEGImages/ + Annotations/ + ImageSets/Main/default.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

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
    # Record the convention so a later --resume can reject a format switch
    # instead of silently merging xywh and xyxy bboxes.
    record_coco_bbox_format(anno_path, coco_bbox_format)


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


def load_coco_json(path) -> Optional[dict]:
    """Read a COCO JSON file, returning None when missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def record_coco_bbox_format(path, bbox_format: str) -> None:
    """Stamp the bbox convention used by a COCO file into its ``info`` block.

    ``--resume`` merges a prior COCO file with this run's output; without a
    recorded convention a later run with a different ``--coco-bbox-format``
    would silently mix xywh and xyxy bboxes.
    """
    data = load_coco_json(path)
    if data is None:
        return
    info = data.get("info")
    if not isinstance(info, dict):
        info = {}
    info["bbox_format"] = bbox_format
    data["info"] = info
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def coco_bbox_format_of(path_or_payload) -> str:
    """Return the bbox convention recorded in a COCO file/payload.

    Files written before the convention was recorded (or produced by other
    writers) default to ``xywh``, matching the historical default.
    """
    if isinstance(path_or_payload, dict):
        data = path_or_payload
    else:
        data = load_coco_json(path_or_payload)
    if not isinstance(data, dict):
        return "xywh"
    info = data.get("info")
    if isinstance(info, dict) and info.get("bbox_format"):
        return str(info["bbox_format"])
    return "xywh"


def drop_coco_images(payload: dict, file_names: set) -> dict:
    """Return a COCO payload without the images whose ``file_name`` is listed.

    Used on ``--resume`` when a re-processed sample now yields no masks: its
    previous image and annotations must not survive the merge.
    """
    images = payload.get("images") or []
    dropped_ids = {
        image.get("id") for image in images if image.get("file_name") in file_names
    }
    result = dict(payload)
    result["images"] = [
        image for image in images if image.get("id") not in dropped_ids
    ]
    result["annotations"] = [
        annotation
        for annotation in (payload.get("annotations") or [])
        if annotation.get("image_id") not in dropped_ids
    ]
    return result


def merge_coco_json(prior: dict, target_path) -> None:
    """Merge prior COCO content into a freshly written unified COCO file.

    Used by ``--resume``: samples skipped this run keep their annotations, and a
    re-processed sample replaces its previous entry instead of duplicating it
    (entries are matched by ``file_name``). No-op when there is no prior data.
    """
    if not prior:
        return
    target_path = Path(target_path)
    current = load_coco_json(target_path)
    if current is None:
        return
    merged = _merge_coco_payloads(prior, current)
    target_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _merge_coco_payloads(existing: dict, new: dict) -> dict:
    """Union two COCO payloads; ``new`` wins on a ``file_name`` clash."""
    categories = list(existing.get("categories") or [])
    name_to_id = {
        category.get("name"): category.get("id")
        for category in categories
        if category.get("name") is not None and category.get("id") is not None
    }
    next_category_id = max((c.get("id", 0) for c in categories), default=0)
    new_category_ids: Dict = {}
    for category in new.get("categories") or []:
        name = category.get("name")
        if name in name_to_id:
            new_category_ids[category.get("id")] = name_to_id[name]
        else:
            next_category_id += 1
            merged_category = dict(category)
            merged_category["id"] = next_category_id
            categories.append(merged_category)
            if name is not None:
                name_to_id[name] = next_category_id
            new_category_ids[category.get("id")] = next_category_id

    images = list(existing.get("images") or [])
    image_by_name = {image.get("file_name"): image for image in images}
    next_image_id = max((image.get("id", 0) for image in images), default=0)
    replaced_image_ids = set()
    new_image_ids: Dict = {}
    for image in new.get("images") or []:
        file_name = image.get("file_name")
        if file_name in image_by_name:
            kept = image_by_name[file_name]
            replaced_image_ids.add(kept.get("id"))
            kept.update({k: v for k, v in image.items() if k != "id"})
            new_image_ids[image.get("id")] = kept.get("id")
        else:
            next_image_id += 1
            merged_image = dict(image)
            merged_image["id"] = next_image_id
            images.append(merged_image)
            image_by_name[file_name] = merged_image
            new_image_ids[image.get("id")] = next_image_id

    annotations = [
        annotation
        for annotation in (existing.get("annotations") or [])
        if annotation.get("image_id") not in replaced_image_ids
    ]
    next_annotation_id = max(
        (annotation.get("id", 0) for annotation in annotations), default=0
    )
    for annotation in new.get("annotations") or []:
        merged_annotation = dict(annotation)
        next_annotation_id += 1
        merged_annotation["id"] = next_annotation_id
        merged_annotation["image_id"] = new_image_ids.get(
            annotation.get("image_id"), annotation.get("image_id")
        )
        if merged_annotation.get("category_id") in new_category_ids:
            merged_annotation["category_id"] = new_category_ids[
                merged_annotation["category_id"]
            ]
        annotations.append(merged_annotation)

    merged = dict(new)
    merged["images"] = images
    merged["annotations"] = annotations
    merged["categories"] = categories
    return merged


def _rewrite_coco_bbox_to_xyxy(json_path: Path) -> None:
    """In-place convert bbox in COCO JSON from xywh → xyxy."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        ann["bbox"] = [x, y, x + w, y + h]
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
