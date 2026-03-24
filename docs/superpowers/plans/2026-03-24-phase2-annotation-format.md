# entomokit 重构 Phase 2 — 注释格式对齐 detcli

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `segment` 和 `synthesize` 的注释输出格式（COCO/YOLO/VOC 目录布局）对齐 detcli，使两个工具链产出的数据集可直接互用。

**Architecture:** 新建 `src/common/annotation_writer.py` 封装 supervision 库的 `DetectionDataset` 保存逻辑，`SegmentationProcessor` 和 `SynthesisProcessor` 统一调用此模块写出注释；原有内部格式转换逻辑替换为 supervision 标准调用。

**Tech Stack:** Python 3.8+, `supervision` (已在 detcli 中使用), `src/common/annotation_writer.py` (新建)

**Spec:** `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` — Section 5.1 (segment 注释格式变更)

**前置条件:** Phase 1 已完成（`entomokit segment` 命令可用）

---

## 格式规范（来自 detcli）

| 格式 | 目录布局 | 文件命名 | 备注 |
|------|----------|----------|------|
| **COCO** | 图像与 JSON 同级平铺 | `annotations.coco.json` | bbox 格式由 `--coco-bbox-format xywh/xyxy` 控制，默认 `xywh` |
| **YOLO** | `images/` + `labels/` 子目录 | `data.yaml`（含 `nc` + 带引号 `names`） | — |
| **VOC** | `JPEGImages/` + `Annotations/` + `ImageSets/Main/default.txt` | — | 标准 Pascal VOC |

---

## 文件结构

### 新建文件
```
src/common/annotation_writer.py    # supervision 注释写出封装
tests/test_annotation_writer.py    # 单元测试
```

### 修改文件
```
src/segmentation/processor.py      # 替换旧注释写出逻辑，调用 annotation_writer
src/synthesis/processor.py         # 同上
```

---

## Task 1: 新建 `src/common/annotation_writer.py`

**Files:**
- Create: `src/common/annotation_writer.py`

- [ ] **Step 1: 确认 supervision 已安装**

```bash
python -c "import supervision as sv; print(sv.__version__)"
```

如未安装：`pip install supervision`

- [ ] **Step 2: 写 `src/common/annotation_writer.py`**

```python
"""Unified annotation writer for COCO / YOLO / VOC formats.

Layout conventions match detcli:
  COCO  → flat dir, file named 'annotations.coco.json', bbox xywh by default
  YOLO  → images/ + labels/ + data.yaml (quoted names)
  VOC   → JPEGImages/ + Annotations/ + ImageSets/Main/default.txt
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    import supervision as sv

    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unknown format: {fmt!r}. Supported: {sorted(SUPPORTED_FORMATS)}")
    if coco_bbox_format not in COCO_BBOX_FORMATS:
        raise ValueError(f"Unknown coco_bbox_format: {coco_bbox_format!r}. Use 'xywh' or 'xyxy'.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build sv.DetectionDataset
    dataset = sv.DetectionDataset(
        classes=class_names,
        images={str(p): p for p in image_paths},
        annotations={str(p): detections_per_image.get(str(p), sv.Detections.empty()) for p in image_paths},
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
        images_directory_path=str(out_dir),
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
    stems = [Path(p).stem for p in dataset.images]
    _write_voc_imagesets(out_dir, stems)


def _write_yolo_yaml(yaml_path: Path, class_names: List[str]) -> None:
    """Write data.yaml with quoted class names (matches detcli convention)."""
    deduped = list(dict.fromkeys(class_names))  # preserve order, remove dups
    quoted = ", ".join(f'"{c}"' for c in deduped)
    yaml_path.write_text(
        f"nc: {len(deduped)}\n"
        f"names: [{quoted}]\n"
    )


def _write_voc_imagesets(out_dir: Path, stems: List[str]) -> None:
    """Write ImageSets/Main/default.txt."""
    imagesets_dir = out_dir / "ImageSets" / "Main"
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    (imagesets_dir / "default.txt").write_text("\n".join(stems) + "\n")


def _rewrite_coco_bbox_to_xyxy(json_path: Path) -> None:
    """In-place convert bbox in COCO JSON from xywh → xyxy."""
    data = json.loads(json_path.read_text())
    for ann in data.get("annotations", []):
        x, y, w, h = ann["bbox"]
        ann["bbox"] = [x, y, x + w, y + h]
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
```

- [ ] **Step 3: Commit**

```bash
git add src/common/annotation_writer.py
git commit -m "feat: add annotation_writer module (COCO/YOLO/VOC aligned with detcli)"
```

---

## Task 2: 单元测试 `annotation_writer`

**Files:**
- Create: `tests/test_annotation_writer.py`

- [ ] **Step 1: 写测试**

```python
"""Tests for src/common/annotation_writer.py"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_dataset_and_write(tmp_path, fmt, coco_bbox_format="xywh"):
    """Helper: create minimal writer call with mocked supervision."""
    import supervision as sv
    import numpy as np
    from src.common.annotation_writer import write_annotations

    # Create a minimal 10x10 PNG
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
        class_names=["insect"],
        out_dir=tmp_path / "out",
        fmt=fmt,
        coco_bbox_format=coco_bbox_format,
    )
    return tmp_path / "out"


def test_coco_output_filename(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "coco")
    assert (out_dir / "annotations.coco.json").exists()


def test_coco_bbox_xywh_format(tmp_path):
    out_dir = _make_dataset_and_write(tmp_path, "coco", coco_bbox_format="xywh")
    data = json.loads((out_dir / "annotations.coco.json").read_text())
    bbox = data["annotations"][0]["bbox"]
    # In xywh: w < x2, so check [x, y, w, h] where w=h=40
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
    assert 'names: ["insect"]' in content
    assert "nc: 1" in content


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
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/test_annotation_writer.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_annotation_writer.py
git commit -m "test: add annotation_writer unit tests for COCO/YOLO/VOC formats"
```

---

## Task 3: 更新 `SegmentationProcessor` 使用新 annotation_writer

**Files:**
- Modify: `src/segmentation/processor.py`

- [ ] **Step 1: 侦查现有注释写出逻辑**

先读懂现有代码，再决定如何替换：

```bash
# 找到注释写出的具体位置
grep -n "annotation\|coco\|yolo\|voc\|json\|xml\|write\|bbox\|sv\." src/segmentation/processor.py
```

然后阅读相关代码段，确认以下三点：
1. `SegmentationProcessor` 内部用什么数据结构存储检测结果（`sv.Detections`、原生 dict、还是自定义格式）？
2. 注释写出在哪个方法/位置触发（`process()`、`process_one()` 还是专门的保存方法）？
3. 现有注释文件的目录布局是什么（与 detcli 格式的差异在哪里）？

记录以上三点后再进行 Step 2。

- [ ] **Step 2: 编写格式转换适配代码**

根据 Step 1 的调研结果，在调用 `write_annotations()` 前，将内部格式转换为 `sv.Detections`：

```python
# 示例：若内部用 List[Dict] 存储 bbox（xyxy）和 class_id
import supervision as sv
import numpy as np

def _to_sv_detections(records: list) -> sv.Detections:
    """Convert processor internal records to sv.Detections."""
    if not records:
        return sv.Detections.empty()
    xyxy = np.array([r["bbox_xyxy"] for r in records], dtype=np.float32)
    class_ids = np.array([r["class_id"] for r in records], dtype=int)
    return sv.Detections(xyxy=xyxy, class_id=class_ids)
```

实际转换代码应基于 Step 1 调研到的真实数据结构写出。

- [ ] **Step 3: 替换注释写出调用**

用 `annotation_writer.write_annotations()` 替换现有写出逻辑：

```python
if self.annotation_format:
    from src.common.annotation_writer import write_annotations
    write_annotations(
        image_paths=processed_image_paths,
        detections_per_image={
            str(p): _to_sv_detections(detections_by_path[str(p)])
            for p in processed_image_paths
        },
        class_names=["insect"],
        out_dir=self.annotation_output_dir,
        fmt=self.annotation_format,
        coco_bbox_format=getattr(self, "coco_bbox_format", "xywh"),
    )
```

- [ ] **Step 4: 运行现有注释相关测试**

```bash
pytest tests/test_annotation_formats.py tests/test_segmentation.py -v
```

Expected: 全部 PASS（如有失败，按错误信息修复）。

- [ ] **Step 5: Commit**

```bash
git add src/segmentation/processor.py
git commit -m "feat: segment annotation output aligned to detcli COCO/YOLO/VOC layout"
```

---

## Task 4: 更新 `SynthesisProcessor` 使用新 annotation_writer

**Files:**
- Modify: `src/synthesis/processor.py`

- [ ] **Step 1: 侦查现有注释写出逻辑（同 Task 3 Step 1 模式）**

```bash
grep -n "annotation\|coco\|yolo\|voc\|json\|xml\|write\|bbox\|sv\." src/synthesis/processor.py
```

确认：内部数据结构、写出触发位置、现有目录布局与 detcli 的差异。

- [ ] **Step 2: 编写格式转换适配代码 + 替换调用**

参照 Task 3 Step 2-3 的模式，将合成结果的内部格式转换为 `sv.Detections`，再调用 `write_annotations()`：

```python
if self.annotation_format:
    from src.common.annotation_writer import write_annotations
    write_annotations(
        image_paths=synthesized_image_paths,
        detections_per_image=detections_dict,   # sv.Detections per image
        class_names=self.class_names,
        out_dir=self.annotation_output_dir,
        fmt=self.annotation_format,
        coco_bbox_format=getattr(self, "coco_bbox_format", "xywh"),
    )
```

- [ ] **Step 3: 验证 synthesize 端到端测试**

```bash
pytest tests/ -v -k "synth" --tb=short
```

Expected: 相关测试 PASS。

- [ ] **Step 4: Commit**

```bash
git add src/synthesis/processor.py
git commit -m "feat: synthesize annotation output aligned to detcli COCO/YOLO/VOC layout"
```

---

## Task 5: 端到端冒烟测试 segment + synthesize 注释输出

- [ ] **Step 1: 测试 segment COCO xywh 输出**

```bash
python -m entomokit.main segment \
    --input-dir data/insects \
    --out-dir /tmp/seg_test \
    --segmentation-method otsu \
    --annotation-format coco \
    --coco-bbox-format xywh
```

验证：
```bash
ls /tmp/seg_test/
# 应存在 annotations.coco.json（与图像同级）
python -c "
import json
d = json.load(open('/tmp/seg_test/annotations.coco.json'))
print('images:', len(d['images']))
print('annotations:', len(d['annotations']))
print('sample bbox:', d['annotations'][0]['bbox'] if d['annotations'] else 'no annotations')
"
```

- [ ] **Step 2: 测试 segment COCO xyxy 输出**

```bash
python -m entomokit.main segment \
    --input-dir data/insects \
    --out-dir /tmp/seg_xyxy \
    --segmentation-method otsu \
    --annotation-format coco \
    --coco-bbox-format xyxy
```

验证 bbox 第3、4个值 > 第1、2个值（xyxy 特征）。

- [ ] **Step 3: 测试 YOLO 输出布局**

```bash
python -m entomokit.main segment \
    --input-dir data/insects \
    --out-dir /tmp/seg_yolo \
    --segmentation-method otsu \
    --annotation-format yolo
ls /tmp/seg_yolo/images/ /tmp/seg_yolo/labels/ /tmp/seg_yolo/data.yaml
cat /tmp/seg_yolo/data.yaml
```

Expected: `images/`、`labels/`、`data.yaml` 均存在，`data.yaml` 含 `names: ["insect"]`。

- [ ] **Step 4: 测试 VOC 输出布局**

```bash
python -m entomokit.main segment \
    --input-dir data/insects \
    --out-dir /tmp/seg_voc \
    --segmentation-method otsu \
    --annotation-format voc
ls /tmp/seg_voc/JPEGImages/ /tmp/seg_voc/Annotations/ /tmp/seg_voc/ImageSets/Main/
```

Expected: 三个目录存在，`ImageSets/Main/default.txt` 含图像 stem 列表。

- [ ] **Step 5: 运行完整测试套件**

```bash
pytest tests/ -v --tb=short
```

Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test: Phase 2 smoke tests — annotation format alignment verified"
```

---

## Phase 2 完成标志

- [ ] `entomokit segment --annotation-format coco --coco-bbox-format xyxy` 产出 `annotations.coco.json`（平铺布局）
- [ ] `entomokit segment --annotation-format yolo` 产出 `images/` + `labels/` + 含引号 names 的 `data.yaml`
- [ ] `entomokit segment --annotation-format voc` 产出 `JPEGImages/` + `Annotations/` + `ImageSets/Main/default.txt`
- [ ] `entomokit synthesize` 注释格式相同规范
- [ ] `pytest tests/ -v` 全部通过
