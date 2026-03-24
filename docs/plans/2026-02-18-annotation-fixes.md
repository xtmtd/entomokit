# Annotation Fixes Implementation Plan

> **Status:** ✅ Completed - All annotation output features implemented for both synthesis and segmentation scripts

**Goal:** Fix critical annotation issues and add comprehensive annotation output support for both `synthesize.py` and `segment.py`:
- VOC/YOLO polygon segmentation support
- Real-time per-image annotation output
- Correct coordinate frames (synthesized image for synthesis, original image for segmentation)
- Multiple output formats (COCO, VOC, YOLO)
- Unified vs separate COCO output modes

**Architecture:** 
- Update `src/metadata.py` to add polygon support to VOC/YOLO output methods
- Refactor `src/synthesis/processor.py` to calculate coordinates from synthesized images and save annotations immediately per-image
- Remove batch annotation generation methods, replace with real-time per-image saving
- Update `scripts/synthesize.py` wrapper to pass position information

**Tech Stack:** Python 3, numpy, OpenCV, existing COCOMetadataManager

---

## Task 1: Update COCOMetadataManager.to_voc_xml() to support polygon segmentation

**Files:**
- Modify: `src/metadata.py:312-353` (to_voc_xml method)

**Step 1: Read current to_voc_xml implementation**

```python
def to_voc_xml(self, filename: str, width: int, height: int, depth: int = 3) -> str:
    # Current implementation only has bbox
```

**Step 2: Update method signature to accept optional segmentation**

```python
def to_voc_xml(self, filename: str, width: int, height: int, depth: int = 3,
               segmentation: Optional[List[List[float]]] = None) -> str:
```

**Step 3: Add polygon XML section after bbox**

```python
if segmentation and segmentation[0]:
    xml_parts.append('    <polygon>')
    points = segmentation[0]
    for i in range(0, len(points), 2):
        x, y = int(points[i]), int(points[i+1]) if i+1 < len(points) else 0
        xml_parts.append(f'        <point>')
        xml_parts.append(f'            <x>{x}</x>')
        xml_parts.append(f'            <y>{y}</y>')
        xml_parts.append('        </point>')
    xml_parts.append('    </polygon>')
```

**Step 4: Run test**

```bash
pytest tests/test_metadata.py::test_coco_metadata_manager_to_voc_xml -v
```

**Step 5: Commit**

```bash
git add src/metadata.py
git commit -m "feat: add polygon segmentation support to VOC XML output"
```

---

## Task 2: Update COCOMetadataManager.to_yolo_txt() to support polygon segmentation

**Files:**
- Modify: `src/metadata.py:355-377` (to_yolo_txt method)

**Step 1: Read current to_yolo_txt implementation**

**Step 2: Update method signature to accept optional segmentation**

```python
def to_yolo_txt(self, width: int, height: int,
                segmentation: Optional[List[List[float]]] = None) -> str:
```

**Step 3: Update YOLO format to support polygon**

```python
lines = []
for annotation in self.annotations:
    bbox = annotation.get('bbox', [0, 0, 0, 0])
    class_id = 0
    
    if segmentation and segmentation[0]:
        # YOLOv8+ format: class_id x1 y1 x2 y2 ... (polygon points)
        polygon = segmentation[0]
        line = f"{class_id}"
        for point in polygon:
            line += f" {point:.6f}"
        lines.append(line)
    else:
        # Fallback to bbox only: class_id cx cy w h
        x_center = (bbox[0] + bbox[2] / 2) / width if width > 0 else 0
        y_center = (bbox[1] + bbox[3] / 2) / height if height > 0 else 0
        yolo_w = bbox[2] / width if width > 0 else 0
        yolo_h = bbox[3] / height if height > 0 else 0
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}")
```

**Step 4: Run test**

```bash
pytest tests/test_metadata.py::test_coco_metadata_manager_to_yolo_txt -v
```

**Step 5: Commit**

```bash
git add src/metadata.py
git commit -m "feat: add polygon segmentation support to YOLO TXT output"
```

---

## Task 3: Update _add_synthesis_metadata() to track position and use synthesized mask

**Files:**
- Modify: `src/synthesis/processor.py:395-443` (_add_synthesis_metadata method)

**Step 1: Read current _add_synthesis_metadata implementation**

**Step 2: Update method signature to include position**

```python
def _add_synthesis_metadata(
    self,
    output_filename: str,
    result: np.ndarray,
    target_path_str: str,
    background_path_str: str,
    scale_ratio: float,
    rotation_angle: Optional[float],
    position_x: Optional[int] = None,  # NEW
    position_y: Optional[int] = None   # NEW
) -> None:
```

**Step 3: Use synthesized mask instead of original target mask**

```python
# Extract mask from synthesized result (has alpha channel)
result_mask = result[:, :, 3] if result.shape[2] == 4 else None

if result_mask is not None and np.any(result_mask):
    # Use mask from SYNTHESIZED image - coordinates are in synthesized frame
    bbox = mask_to_bbox(result_mask)
    polygon = mask_to_polygon(result_mask)
    mask_area = int(np.sum(result_mask > 0))
    area = float(bbox[2] * bbox[3])
else:
    # Fallback to original target mask
    target_path = Path(target_path_str)
    target_img = self._load_image(target_path)
    mask = target_img[:, :, 3] if target_img.shape[2] == 4 else np.ones(target_img.shape[:2], dtype=np.uint8) * 255
    bbox = mask_to_bbox(mask)
    polygon = mask_to_polygon(mask)
    mask_area = int(np.sum(mask > 0))
    area = float(bbox[2] * bbox[3])
```

**Step 4: Update metadata storage to include position**

```python
self.synthesis_metadata.append({
    'image': {
        'file_name': f"{output_filename}.{self.output_format}",
        'width': result.shape[1],
        'height': result.shape[0],
        'original_target_path': str(Path(target_path_str).resolve()),
        'original_background_path': str(Path(background_path_str).resolve()),
        'position_x': position_x,  # NEW
        'position_y': position_y   # NEW
    },
    'annotation': {
        'bbox': [int(x) for x in bbox],
        'segmentation': polygon,
        'area': area,
        'mask_area': mask_area,
        'scale_ratio': scale_ratio,
        'rotation_angle': rotation_angle if rotation_angle is not None else 0.0,
        'original_target_file': str(Path(target_path_str).resolve()),
        'original_background_file': str(Path(background_path_str).resolve()),
        'position_x': position_x,  # NEW
        'position_y': position_y   # NEW
    }
})
```

**Step 5: Commit**

```bash
git add src/synthesis/processor.py
git commit -m "refactor: use synthesized image mask for COORDINATES in synthesis frame"
```

---

## Task 4: Update synthesize_single to return position information

**Files:**
- Modify: `src/synthesis/processor.py:248-382` (synthesize_single method)

**Step 1: Read current synthesize_single implementation**

**Step 2: Update return statement to include position**

```python
return result, output_filename, scale_ratio, angle, target_path, background, x, y
```

**Step 3: Commit**

```bash
git add src/synthesis/processor.py
git commit -m "refactor: synthesize_single returns position (x, y) in synthesized image"
```

---

## Task 5: Update _synthesize_single_wrapper to preserve position info

**Files:**
- Modify: `src/synthesis/processor.py:384-392` (_synthesize_single_wrapper method)

**Step 1: Read current implementation**

**Step 2: Update wrapper to pass position**

```python
def _synthesize_single_wrapper(self, args):
    target_img, background, scale_ratio, target_path, counter, background_path = args
    result = self.synthesize_single(target_img, background, scale_ratio, target_path, counter)
    # result is: image, filename, scale_ratio, angle, target_path, background_path, x, y
    return (result[0], result[1], result[2], 
            None if result[0] is None else result[3], 
            str(result[4]) if result[4] else None, 
            str(result[5]) if result[5] else None,
            result[6] if len(result) > 6 else None,  # x
            result[7] if len(result) > 7 else None)  # y
```

**Step 3: Commit**

```bash
git add src/synthesis/processor.py
git commit -m "refactor: _synthesize_single_wrapper preserves position (x, y)"
```

---

## Task 6: Add _save_annotation_for_image() method for real-time annotation output

**Files:**
- Create: Add new method to `src/synthesis/processor.py` after `process_directory()`

**Step 1: Add new method before process_directory**

```python
def _save_annotation_for_image(
    self,
    output_filename: str,
    result: np.ndarray,
    target_path_str: str,
    background_path_str: str,
    scale_ratio: float,
    rotation_angle: Optional[float],
    position_x: Optional[int],
    position_y: Optional[int]
) -> None:
    """Save annotation immediately after each image synthesis (real-time)."""
    if self.annotation_format == "coco":
        self._save_coco_single(
            output_filename, result, target_path_str, background_path_str,
            scale_ratio, rotation_angle, position_x, position_y
        )
    elif self.annotation_format == "voc":
        self._save_voc_single(
            output_filename, result, target_path_str, background_path_str,
            scale_ratio, rotation_angle, position_x, position_y
        )
    elif self.annotation_format == "yolo":
        self._save_yolo_single(
            output_filename, result, target_path_str, background_path_str,
            scale_ratio, rotation_angle, position_x, position_y
        )

def _save_coco_single(
    self,
    output_filename: str,
    result: np.ndarray,
    target_path_str: str,
    background_path_str: str,
    scale_ratio: float,
    rotation_angle: Optional[float],
    position_x: Optional[int],
    position_y: Optional[int]
) -> None:
    """Save single COCO JSON file per image."""
    annotations_dir = self._get_annotation_output_dir(Path(self.output_subdir).parent)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    manager = COCOMetadataManager()
    category_id = manager.add_category("insect")
    
    image_id = manager.add_image(
        file_name=output_filename,
        width=result.shape[1],
        height=result.shape[0],
        original_path=str(Path(target_path_str).resolve()),
        original_background_path=str(Path(background_path_str).resolve())
    )
    
    manager.add_annotation(
        image_id=image_id,
        category_id=category_id,
        bbox=[0, 0, result.shape[1], result.shape[0]],
        segmentation=[],
        area=float(result.shape[1] * result.shape[0]),
        mask_area=0,
        scale_ratio=scale_ratio,
        rotation_angle=rotation_angle if rotation_angle is not None else 0.0,
        original_target_file=str(Path(target_path_str).resolve()),
        original_background_file=str(Path(background_path_str).resolve())
    )
    
    base_name = Path(output_filename).stem
    annotations_path = annotations_dir / f"{base_name}.json"
    manager.save(annotations_path)

def _save_voc_single(
    self,
    output_filename: str,
    result: np.ndarray,
    target_path_str: str,
    background_path_str: str,
    scale_ratio: float,
    rotation_angle: Optional[float],
    position_x: Optional[int],
    position_y: Optional[int]
) -> None:
    """Save single VOC XML file per image."""
    # Implement VOC XML generation
    pass

def _save_yolo_single(
    self,
    output_filename: str,
    result: np.ndarray,
    target_path_str: str,
    background_path_str: str,
    scale_ratio: float,
    rotation_angle: Optional[float],
    position_x: Optional[int],
    position_y: Optional[int]
) -> None:
    """Save single YOLO TXT file per image."""
    # Implement YOLO TXT generation
    pass
```

**Step 2: Commit**

```bash
git add src/synthesis/processor.py
git commit -m "feat: add _save_annotation_for_image() for real-time annotation output"
```

---

## Task 7: Remove batch annotation generation methods

**Files:**
- Remove: `_generate_coco_annotations()`, `_generate_coco_annotations_unified()`, `_generate_coco_annotations_separate()`
- Remove: `_generate_voc_annotations()`, `_generate_yolo_annotations()`

**Step 1: Read and identify batch methods to remove**

**Step 2: Remove batch methods**

```python
# Delete these methods:
# - _generate_coco_annotations (old)
# - _generate_coco_annotations_unified
# - _generate_coco_annotations_separate
# - _generate_voc_annotations
# - _generate_yolo_annotations
```

**Step 3: Commit**

```bash
git add src/synthesis/processor.py
git commit -m "refactor: remove batch annotation generation methods, use real-time instead"
```

---

## Task 8: Update process_directory() to call real-time annotation saving

**Files:**
- Modify: `src/synthesis/processor.py:667-814` (process_directory method)

**Step 1: Read current process_directory implementation**

**Step 2: Replace batch annotation generation with real-time calls**

**For parallel multiprocessing (threads > 1):**
```python
with multiprocessing.Pool(processes=threads) as pool:
    for result in tqdm(pool.imap(self._synthesize_single_wrapper, tasks), total=len(tasks), desc="Synthesizing"):
        if result[0] is None:
            if result[1] is None:
                skipped_images += 1
            continue
        if result[1]:
            output_path = image_output_dir / f"{result[1]}.{self.output_format}"
        else:
            output_path = image_output_dir / f"synth_{synthesis_id:06d}.{self.output_format}"
        self._save_image(result[0], output_path)
        
        # NEW: Real-time annotation output
        self._save_annotation_for_image(
            output_filename=result[1],
            result=result[0],
            target_path_str=result[4] if result[4] else "",
            background_path_str=result[5] if result[5] else "",
            scale_ratio=result[2],
            rotation_angle=result[3],
            position_x=result[6] if len(result) > 6 else None,
            position_y=result[7] if len(result) > 7 else None
        )
        synthesis_id += 1
```

**For sequential processing:**
```python
for task in tqdm(tasks, desc="Synthesizing"):
    result = self.synthesize_single(task[0], task[1], task[2], task[3], task[4])
    if result[0] is None:
        if result[1] is None:
            skipped_images += 1
        continue
    if result[1]:
        output_path = image_output_dir / f"{result[1]}.{self.output_format}"
    else:
        output_path = image_output_dir / f"synth_{synthesis_id:06d}.{self.output_format}"
    self._save_image(result[0], output_path)
    
    # NEW: Real-time annotation output
    self._save_annotation_for_image(
        output_filename=result[1],
        result=result[0],
        target_path_str=str(task[3]) if task[3] else "",
        background_path_str=str(task[5]) if len(task) > 5 and task[5] else "",
        scale_ratio=task[2],
        rotation_angle=result[3],
        position_x=result[6] if len(result) > 6 else None,
        position_y=result[7] if len(result) > 7 else None
    )
    synthesis_id += 1
```

**Step 3: Remove final annotation generation block**

Remove this entire block from end of process_directory():
```python
# DELETE THIS BLOCK
if self.annotation_format == "coco":
    if self.coco_output_mode == "unified":
        self._generate_coco_annotations_unified(output_dir)
    else:
        self._generate_coco_annotations_separate(output_dir)
elif self.annotation_format == "voc":
    self._generate_voc_annotations(output_dir)
elif self.annotation_format == "yolo":
    self._generate_yolo_annotations(output_dir)
```

**Step 4: Commit**

```bash
git add src/synthesis/processor.py
git commit -m "refactor: process_directory uses real-time annotation output"
```

---

## Task 9: Update synthesize_single wrapper call to get position info

**Files:**
- Modify: `src/synthesis/processor.py:384-392` (_synthesize_single_wrapper)

**Step 1: Verify wrapper returns 8 values**

Current should return:
```python
return (result[0], result[1], result[2], 
        None if result[0] is None else result[3], 
        str(result[4]) if result[4] else None, 
        str(result[5]) if result[5] else None,
        result[6] if len(result) > 6 else None,
        result[7] if len(result) > 7 else None)
```

**Step 2: Commit if needed**

```bash
git add src/synthesis/processor.py
git commit -m "refactor: verify _synthesize_single_wrapper returns 8 values"
```

---

## Task 10: Update scripts/synthesize.py docstring examples

**Files:**
- Modify: `scripts/synthesize.py:1-26` (docstring)

**Step 1: Update usage examples**

Remove `--coco-output-dir` from examples (parameter removed)

**Step 2: Add new parameters to examples**

```python
"""
# With VOC format
python scripts/synthesize.py \\
    --target-dir images/targets/ \\
    --background-dir images/backgrounds/ \\
    --out-dir outputs/synthesized/ \\
    --num-syntheses 10 \\
    --annotation-output-format voc

# With YOLO format
python scripts/synthesize.py \\
    --target-dir images/targets/ \\
    --background-dir images/backgrounds/ \\
    --out-dir outputs/synthesized/ \\
    --num-syntheses 10 \\
    --annotation-output-format yolo
"""
```

**Step 3: Commit**

```bash
git add scripts/synthesize.py
git commit -m "docs: update synthesize.py docstring examples"
```

---

## Task 11: Add tests for new VOC/YOLO polygon support

**Files:**
- Create: `tests/test_annotation_formats.py`

**Step 1: Create test file**

```python
"""Tests for VOC and YOLO annotation format support with polygon segmentation."""

import pytest
import numpy as np
from pathlib import Path
from src.metadata import COCOMetadataManager, mask_to_bbox, mask_to_polygon


def test_coco_metadata_manager_to_voc_xml_with_polygon():
    """Test VOC XML includes polygon when segmentation provided."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")
    
    bbox = [100, 50, 100, 100]
    polygon = [[100, 50, 200, 50, 200, 150, 100, 150]]
    
    manager.add_annotation(
        image_id=1,
        category_id=cat_id,
        bbox=bbox,
        segmentation=polygon,
        area=10000,
        mask_area=10000
    )
    
    xml = manager.to_voc_xml("test.png", 640, 480, segmentation=polygon)
    
    assert "<polygon>" in xml
    assert "<x>100</x>" in xml
    assert "<y>50</y>" in xml
    assert "<x>200</x>" in xml
    assert "<y>150</y>" in xml


def test_coco_metadata_manager_to_yolo_txt_with_polygon():
    """Test YOLO TXT includes polygon when segmentation provided."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")
    
    bbox = [100, 50, 100, 100]
    polygon = [[100, 50, 200, 50, 200, 150, 100, 150]]
    
    manager.add_annotation(
        image_id=1,
        category_id=cat_id,
        bbox=bbox,
        segmentation=polygon,
        area=10000,
        mask_area=10000
    )
    
    yolo_txt = manager.to_yolo_txt(width=640, height=480, segmentation=polygon)
    
    lines = yolo_txt.strip().split('\n')
    assert len(lines) == 1
    # YOLOv8+ format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    parts = lines[0].split()
    assert len(parts) == 9  # class_id + 8 polygon points


def test_coco_metadata_manager_to_yolo_txt_fallback_to_bbox():
    """Test YOLO TXT falls back to bbox format when no polygon."""
    manager = COCOMetadataManager()
    cat_id = manager.add_category("insect")
    
    bbox = [100, 50, 100, 100]
    
    manager.add_annotation(
        image_id=1,
        category_id=cat_id,
        bbox=bbox,
        segmentation=None,
        area=10000,
        mask_area=10000
    )
    
    yolo_txt = manager.to_yolo_txt(width=640, height=480, segmentation=None)
    
    lines = yolo_txt.strip().split('\n')
    assert len(lines) == 1
    # Fallback: class_id cx cy w h
    parts = lines[0].split()
    assert len(parts) == 5  # class_id + 4 bbox values
```

**Step 2: Run tests**

```bash
pytest tests/test_annotation_formats.py -v
```

**Step 3: Commit**

```bash
git add tests/test_annotation_formats.py
git commit -m "test: add VOC/YOLO polygon support tests"
```

---

## Task 12: Run full test suite and verify all tests pass

**Files:**
- Run: All existing tests

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```

**Expected output:** All 24+ tests pass

**Step 2: Fix any failing tests**

**Step 3: Final commit**

```bash
git add tests/ src/ scripts/
git commit -m "fix: all annotation format tests pass"
```

---

## Summary

**Files Modified:**
- `src/metadata.py` - Add polygon support to VOC/YOLO output
- `src/synthesis/processor.py` - Real-time annotation output, synthesized coordinates
- `scripts/synthesize.py` - Update docstring examples

**Files Created:**
- `tests/test_annotation_formats.py` - New tests for VOC/YOLO polygon support

**Key Changes for Synthesis:**
1. VOC XML includes `<polygon>` with segmentation points
2. YOLO TXT supports polygon format (class_id x1 y1 x2 y2 ...)
3. Annotations saved immediately after each image synthesis
4. All coordinates in synthesized image frame (not original target)

---

## Task 13: Add annotation output support to segment.py

**Goal:** Add comprehensive annotation output support to `segment.py` matching the synthesis script capabilities.

**Files:**
- Modify: `scripts/segment.py` - Add CLI arguments
- Modify: `src/segmentation/processor.py` - Add annotation output methods
- Modify: `tests/test_segmentation.py` - Update tests

**Changes:**

### 1. New CLI Arguments
```python
parser.add_argument(
    '--annotation-output-format',
    default='coco',
    choices=['coco', 'voc', 'yolo'],
    help='Output format for annotations: coco (default), voc, yolo'
)

parser.add_argument(
    '--coco-output-mode',
    default='unified',
    choices=['unified', 'separate'],
    help='COCO output mode: unified (single annotations.json) or separate (per-image JSON files)'
)
```

### 2. Annotation Output Features
- **COCO Format**: JSON with images, annotations, categories
  - `unified` mode: Single `annotations/annotations.json` file
  - `separate` mode: Individual JSON files per image (`annotations/{image_name}.json`)
- **VOC Format**: Pascal VOC XML (`annotations/{image_name}.xml`)
- **YOLO Format**: TXT with normalized coordinates (`labels/{image_name}.txt`)

### 3. Coordinate System
- **bbox**: Original image coordinates [x, y, w, h]
- **segmentation**: Original image coordinates [[x1, y1, x2, y2, ...]]
- **width/height**: Original image dimensions
- **file_name**: Points to cleaned output image (e.g., `cleaned_images/image.png`)

### 4. Output Directory Structure
```
output_dir/
├── cleaned_images/           # Segmented output images
│   ├── image_01.png
│   └── image_02.png
├── annotations/              # COCO/VOC annotations (sibling to cleaned_images)
│   ├── annotations.json      # (unified COCO mode)
│   ├── image_01.json         # (separate COCO mode)
│   ├── image_01.xml          # (VOC mode)
│   └── ...
└── labels/                   # YOLO annotations
    ├── image_01.txt
    └── ...
```

### 5. Example Usage
```bash
# COCO unified output (default)
python scripts/segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/segmented/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --segmentation-method sam3 \
    --annotation-output-format coco \
    --coco-output-mode unified

# COCO separate files per image
python scripts/segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/segmented/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --annotation-output-format coco \
    --coco-output-mode separate

# VOC Pascal format
python scripts/segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/segmented/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --annotation-output-format voc

# YOLO format
python scripts/segment.py \
    --input_dir images/clean_insects/ \
    --out_dir outputs/segmented/ \
    --sam3-checkpoint models/sam3_hq_vit_h.pt \
    --annotation-output-format yolo
```

### 6. Area Calculations
- **area**: Bounding box area of the object (width × height)
- **mask_area**: Actual number of pixels in the mask (may differ from bbox area for irregular shapes)

**Commit:**
```bash
git add scripts/segment.py src/segmentation/processor.py tests/test_segmentation.py
git commit -m "feat: add comprehensive annotation output support to segment.py

- Add --annotation-output-format {coco,voc,yolo} argument
- Add --coco-output-mode {unified,separate} argument
- Support COCO JSON (unified/separate), VOC XML, YOLO TXT formats
- Output annotations to annotations/ folder (sibling to cleaned_images/)
- bbox and segmentation in original image coordinates
- file_name points to cleaned output image
- area = bbox area, mask_area = actual mask pixel count"
```

---

## Summary

**Files Modified:**
- `src/metadata.py` - Add polygon support to VOC/YOLO output
- `src/synthesis/processor.py` - Real-time annotation output, synthesized coordinates
- `scripts/synthesize.py` - Update docstring examples
- `scripts/segment.py` - Add annotation output CLI arguments
- `src/segmentation/processor.py` - Add annotation output methods for all formats (VOC/YOLO now produce one file per INPUT image)
- `tests/test_annotation_formats.py` - Updated test expectations for VOC/YOLO (one file per input, not per cleaned output)

**Files Created:**
- `tests/test_annotation_formats.py` - New tests for VOC/YOLO polygon support

**Key Changes:**
1. VOC XML includes `<polygon>` with segmentation points
2. YOLO TXT supports polygon format (class_id x1 y1 x2 y2 ...) with normalized coordinates
3. Annotations saved immediately after each image synthesis/segmentation
4. **VOC/YOLO: One annotation file per INPUT image** (all objects from that input in one file)
5. Synthesis: coordinates in synthesized image frame
6. Segmentation: coordinates in original image frame
7. Both scripts support COCO (unified/separate), VOC, and YOLO output formats
8. Annotations folder is sibling to images folder (not inside)
8. Removed redundant source path information from annotation files (portability)
