# SAM3 掩码形态测量设计（独立 measure 命令）

**日期**: 2026-04-14  
**状态**: 设计确认，待评审

---

## 1. 背景

当前工作流可通过 `entomokit segment --segmentation-method sam3` 产出虫体分割掩码，但缺少稳定、批量、可复现的命令行测量能力，无法直接输出体长、体宽、周长、面积及高级形态指标。

用户约束与目标：
- 不引入 GUI 工具；
- 第一版做独立命令，不改 `segment` 语义；
- 直接采用基于 `scikit-image` 的 v2 方案；
- 输入为 SAM3 生成的掩码图片目录；
- 比例尺按全批固定，单位优先使用微米（um/px）。

---

## 2. 设计目标

1. 新增独立 CLI：`entomokit measure`，支持目录批量测量。
2. 在 `scikit-image` 统一口径下交付核心+高级指标，避免跨库同名指标定义不一致。
3. 体长采用“去附肢近似后主干中心线长度”，支持弯曲虫体。
4. 指标定义尽量与 `scikit-image regionprops` 口径一致，并提供误差对齐验证。
5. 输出像素单位与物理单位并存；当提供比例尺时输出 `*_um`。
6. 通过质量标记暴露不稳定样本，避免静默误导。

---

## 3. 命令与输入输出

### 3.1 命令形态

- `entomokit measure --mask-dir <path> --output-dir <path> [options]`

### 3.2 输入

- `--mask-dir`：SAM3 掩码目录（支持 png/jpg/webp 等常见图像扩展名）。
- 默认约定前景为非零像素。
- 每张图默认仅保留最大连通域作为目标虫体（忽略碎片）。

### 3.3 输出

- `metrics.csv`：每张图一行，包含全部测量指标、质量标记与错误字段。
- `metrics_summary.csv`：样本计数、成功率、warn/fail 占比、关键指标分位统计。
- `log.txt`：参数、处理进度与警告详情（沿用现有日志习惯）。

---

## 4. 指标定义（首版即含高级层）

### 4.1 核心层

- `area_px`：虫体前景像素面积。
- `perimeter_px`：主轮廓周长（弧长）。
- `body_length_px`：去附肢近似后主干中心线测地长度。
- `body_width_px`：沿中心线法向多点采样宽度的中位数。

### 4.2 形状层（对齐 regionprops 思路）

- `major_axis_px`、`minor_axis_px`
- `eccentricity`
- `solidity`
- `extent`
- `circularity`（`4*pi*area/perimeter^2`）
- `convex_area_px`
- `equivalent_diameter_px`

### 4.3 高级层

- `max_feret_px`、`min_feret_px`
- `curvature_index`（中心线长度 / 两端点直线距离）
- `thickness_cv`（法向宽度序列变异系数）
- `symmetry_score`（基于主轴对称变换后的重叠度）

### 4.4 物理单位扩展

- 默认始终输出 `*_px`。
- 传入比例尺后，额外输出 `*_um`。
- 比例尺参数首版采用全批固定：`--pixel-size-um <float>`。

---

## 5. 体长与体宽算法（scikit-image 统一口径 v2）

### 5.1 预处理

1. 二值化与类型归一（0/1）。
2. 连通域筛选：保留最大连通域。
3. 轻量形态学平滑：闭运算抑制边缘毛刺。

### 5.2 去附肢近似

1. 细化得到中心线骨架（`skimage.morphology.skeletonize`）。
2. 端点/分支点检测并构建骨架图。
3. 基于相对长度阈值剪除短分支（阈值按主体尺度自适应）。
4. 保留主干连通子图，求最长端点路径。

### 5.3 指标计算

- `body_length_px`：主干最长路径长度（像素测地长度）。
- `body_width_px`：沿主干采样法向截线宽度，取中位数。
- 兜底策略：主干失败时退化为 `minAreaRect` 长/短边并标记 `warn`。

### 5.4 弯曲与倾斜鲁棒性

- 弯曲虫体：使用中心线测地长度，避免直线低估。
- 倾斜虫体：使用主轴/Feret/中心矩体系，旋转不敏感。

---

## 6. 质量控制与失败策略

### 6.1 质量标记

- `quality_flag`：`ok | warn | fail`
- `warn_reason`：可多值拼接，例如：
  - `touching_border`
  - `too_many_branches`
  - `skeleton_unstable`
  - `fallback_rect_used`

### 6.2 失败处理

- 单图失败不终止批任务；该图写 `fail` 与 `error_message`。
- 最终 summary 汇总失败原因计数。

---

## 7. 指标定义来源与一致性策略

本模块指标定义来源统一为 `scikit-image`（`regionprops` 及相关形态学算子），并在文档与导出定义文件中显式标注，避免跨库口径漂移。

在开发验证阶段建立对齐集：

1. 选取代表性掩码样本（直体、弯体、触角明显、边界噪声）。
2. 对同一输入同时计算：
   - 统一口径实现结果；
   - `regionprops` 参考结果。
3. 记录偏差并调参，确保核心指标误差在可接受阈值内。

建议阈值：
- `area/perimeter/convex_area/equivalent_diameter`：相对误差 < 1%
- `major/minor/eccentricity/solidity/extent`：相对误差 < 3%
- 骨架相关（`body_length/curvature/thickness_cv`）不做强制对齐，以定义一致性与稳定性优先。

---

## 8. 模块边界与后续联动

- 本期仅新增独立 `measure` 模块，不改 `segment` 命令行为。
- 预留后续软联动：`segment` 产物可直接作为 `measure` 输入。
- 不在本期引入部位分割/关键点；相关高精度方案在下一阶段评估。

---

## 9. 验收标准

1. 对 SAM3 掩码目录可稳定批处理，产出 `metrics.csv` 与 `metrics_summary.csv`。
2. 核心指标在正常样本上无空值（非失败样本）。
3. 弯曲样本中，`body_length_px >= major_axis_px`（允许少量异常并给 warn）。
4. 旋转同一掩码（0/45/90/135 度）后，`area/perimeter/body_length` 波动在容忍范围内。
5. 提供 `--pixel-size-um` 后，输出对应 `*_um` 列且单位换算正确。
6. 对边界接触、分支异常、骨架失败样本能正确写出 `warn/fail` 与原因。

---

## 10. 非目标（本期不做）

1. 不把测量逻辑并入 `segment`。
2. 不引入 GUI 工具链（Fiji/CellProfiler）。
3. 不要求每图不同倍率标定（当前假设全批固定比例尺）。
4. 不承诺在单掩码场景下完全排除触角/足/尾须影响，仅通过近似与质量标记降低风险。
