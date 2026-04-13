# Segment Annotation Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不新增 CLI 参数的前提下，让 `segment` 的注释语义由 `--segmentation-method` 决定：`*-bbox` 仅 bbox，非 `*-bbox` 默认输出 bbox+segmentation。

**Architecture:** 在 `SegmentationProcessor` 内集中计算 `annotation_semantics`（`bbox_only`/`both`），并将该语义传递到 COCO/YOLO/VOC 导出分支。`annotation-format` 仅作为序列化目标。VOC 在 `both` 语义下新增分割 mask 导出目录 `SegmentationClass/`。

**Tech Stack:** Python 3.8+, OpenCV, NumPy, 现有 `src/metadata.py` 与 `src/segmentation/processor.py`

**Spec:** `docs/superpowers/specs/2026-04-13-segment-annotation-semantics-design.md`

---

## 文件结构

### 修改文件
- `src/segmentation/processor.py`
- `entomokit/segment.py`（帮助文案）
- `tests/test_annotation_formats.py`

---

### Task 1: 语义开关与导出分支对齐

**Files:**
- Modify: `src/segmentation/processor.py`

- [ ] **Step 1: 写失败测试（YOLO 在 sam3 下应输出 polygon）**

运行：`pytest tests/test_annotation_formats.py::test_yolo_annotation_single_object -q`  
预期：当前失败（或字段数仍是 5 列 bbox）。

- [ ] **Step 2: 在处理器内计算 `annotation_semantics`**

在 `process_image()` 内根据 `self.segmentation_method` 生成布尔开关：
- bbox 方法组 -> `bbox_only=True`
- 其余方法 -> `bbox_only=False`（表示 both）

- [ ] **Step 3: 按语义传递 segmentation 给 YOLO/VOC 导出**

修改导出处：
- `to_yolo_txt(..., segmentation=...)` 在 `both` 下传入真实 polygon 列表；`bbox_only` 下传 `None`
- `to_voc_xml(..., segmentation=...)` 保持 XML 仅 bbox，但在 `both` 下触发 mask 文件输出

- [ ] **Step 4: 运行目标测试验证修复**

运行：`pytest tests/test_annotation_formats.py -q`  
预期：相关测试通过。

---

### Task 2: VOC `both` 语义下导出分割 mask

**Files:**
- Modify: `src/segmentation/processor.py`
- Test: `tests/test_annotation_formats.py`

- [ ] **Step 1: 写失败测试（sam3+voc 需产生 SegmentationClass）**

新增测试断言：`SegmentationClass/*.png` 在非 `*-bbox` 方法下存在。

- [ ] **Step 2: 在 VOC 导出路径添加 mask 写出**

当 `annotation_format == "voc"` 且语义为 `both`：
- 创建 `SegmentationClass/`
- 以输入图基名写单通道 PNG（255=前景, 0=背景）

- [ ] **Step 3: 回归测试**

运行：`pytest tests/test_annotation_formats.py::test_voc_annotation_single_object -q`  
预期：通过，并且不破坏现有 XML 结构断言。

---

### Task 3: 帮助文案与行为说明

**Files:**
- Modify: `entomokit/segment.py`

- [ ] **Step 1: 更新 `--segmentation-method` 帮助文本**

明确 `*-bbox` 只输出 bbox 注释，其他方法输出 bbox+segmentation。

- [ ] **Step 2: 更新 `--annotation-format` 帮助文本**

明确该参数只决定格式，不决定语义粒度；VOC 分割信息落在 mask 文件。

- [ ] **Step 3: 检查 `--help` 输出**

运行：`python -m entomokit.main segment --help`（或项目等效入口）  
预期：文案可读、无矛盾。

---

### Task 4: 全量验证

**Files:**
- Test: `tests/test_annotation_formats.py`

- [ ] **Step 1: 增补/修订验收测试**

覆盖以下组合：
- `sam3 + yolo` -> polygon 行
- `sam3-bbox + yolo` -> 5 列 bbox 行
- `sam3 + voc` -> XML + SegmentationClass
- `sam3-bbox + voc` -> 仅 XML

- [ ] **Step 2: 运行测试**

运行：`pytest tests/test_annotation_formats.py -q`  
预期：全部通过。

- [ ] **Step 3: 运行相关回归（可选）**

运行：`pytest tests/test_segmentation.py -q`  
预期：不引入分割主流程回归。
