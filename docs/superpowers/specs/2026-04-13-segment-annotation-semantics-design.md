# Segment 注释语义设计（信息保真优先）

**日期**: 2026-04-13  
**状态**: 设计确认，进入实施

---

## 1. 背景

当前 `entomokit segment` 在不同 `--segmentation-method` 下，注释导出常表现为 bbox 主导，导致分割方法（如 `sam3`）的像素级信息未稳定体现在 `--annotation-format` 结果中。

这与用户预期不一致：
- `*-bbox` 方法应只输出检测框；
- 非 `*-bbox` 方法应保留分割信息，同时保留 bbox 兼容性。

---

## 2. 设计目标

1. 不新增 CLI 参数。
2. 将注释语义与 `--segmentation-method` 绑定，避免组合复杂度。
3. 默认信息保真优先：分割方法输出 `bbox + segmentation`。
4. 对 VOC/YOLO/COCO 给出一致且可预期的导出行为。

---

## 3. 核心语义

### 3.1 方法分组

- **BBox-only 组**：`sam3-bbox`、`otsu-bbox`、`grabcut-bbox`
- **Segmentation 组**：`sam3`、`otsu`、`grabcut`

### 3.2 语义映射

- 当方法属于 **BBox-only 组**：注释语义为 `bbox_only`
- 当方法属于 **Segmentation 组**：注释语义为 `both`（`bbox + segmentation`）

### 3.3 `annotation-format` 职责

`--annotation-format` 仅负责编码与落盘，不再决定是否丢弃分割信息。

---

## 4. 格式级行为定义

### 4.1 COCO

- `bbox_only`：写 `bbox`，`segmentation` 为空（`[]` 或 `null`）
- `both`：写 `bbox + segmentation`（当前以 polygon 为主）

### 4.2 YOLO

- `bbox_only`：检测格式（`class cx cy w h`）
- `both`：分割格式（`class x1 y1 x2 y2 ...`）

### 4.3 VOC

- `bbox_only`：标准 VOC XML，仅 `<bndbox>`
- `both`：
  - XML 仍写 `<bndbox>`（兼容检测生态）
  - 额外输出分割 mask 文件（`SegmentationClass/*.png`）

终端提示一次：VOC 的分割信息通过 mask 文件提供，不内嵌在 XML polygon 节点中。

---

## 5. 失败与降级策略

在 `both` 语义下，单实例若分割几何无法提取：
- 该实例降级为 bbox；
- 记录 warning（含文件名与实例索引）；
- 不中断整图处理。

---

## 6. 向后兼容

1. CLI 参数保持不变。
2. `*-bbox` 方法的历史行为保持不变（只 bbox）。
3. 非 `*-bbox` 方法行为升级为默认保留分割信息。

---

## 7. 验收标准

1. `sam3 + coco`：每个 annotation 含 bbox 与 segmentation。
2. `sam3-bbox + coco`：segmentation 为空。
3. `sam3 + yolo`：输出 polygon 行（每行字段数 > 5）。
4. `sam3-bbox + yolo`：输出 bbox 行（每行字段数 = 5）。
5. `sam3 + voc`：同时存在 `Annotations/*.xml` 与 `SegmentationClass/*.png`。
6. `sam3-bbox + voc`：仅有 XML，无分割 mask 文件。
