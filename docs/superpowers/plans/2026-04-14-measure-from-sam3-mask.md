# Measure From SAM3 Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `entomokit measure`，对 SAM3 分割掩码目录批量计算核心/高级形态指标并导出 `metrics.csv` 与 `metrics_summary.csv`。

**Architecture:** 新建 `entomokit/measure.py` 作为命令入口，核心计算放在 `src/measurement/`（掩码加载、骨架主干近似、指标计算、批处理导出）。采用 OpenCV+NumPy 实现，输出 `*_px`，在提供 `--pixel-size-um` 时追加 `*_um` 列。通过 `quality_flag/warn_reason` 暴露不稳定样本。

**Tech Stack:** Python 3.8+, OpenCV (`cv2`), NumPy, stdlib `csv/json/pathlib`, pytest

**Spec:** `docs/superpowers/specs/2026-04-14-measurement-from-sam3-mask-design.md`

---

## File Structure

### Create
- `entomokit/measure.py`
- `src/measurement/__init__.py`
- `src/measurement/io.py`
- `src/measurement/core.py`
- `src/measurement/skeleton.py`
- `src/measurement/service.py`
- `tests/test_measure_cli.py`
- `tests/test_measure_metrics.py`

### Modify
- `entomokit/main.py` (register `measure` command)
- `tests/test_main_cli.py` (command order/help coverage)
- `entomokit/cli_schema.py` (no logic change expected; only ensure tests cover new command)

---

### Task 1: CLI 命令骨架与主入口注册

**Files:**
- Create: `entomokit/measure.py`
- Modify: `entomokit/main.py`
- Test: `tests/test_main_cli.py`

- [ ] **Step 1: 写失败测试（主命令出现 measure）**

在 `tests/test_main_cli.py` 增加断言：

```python
assert commands[:8] == [
    "extract-frames",
    "segment",
    "synthesize",
    "clean",
    "augment",
    "split-csv",
    "measure",
    "classify",
]
```

运行：`pytest tests/test_main_cli.py::test_top_level_command_order_matches_dataset_workflow -q`  
预期：FAIL（尚未注册 `measure`）。

- [ ] **Step 2: 新建 `measure` 命令参数定义与 run 入口**

在 `entomokit/measure.py` 添加：

```python
def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("measure", ...)
    p.add_argument("--mask-dir", "-i", required=True, help="SAM3 mask directory")
    p.add_argument("--out-dir", "-o", required=True, help="Output directory")
    p.add_argument("--pixel-size-um", type=float, default=None)
    p.add_argument("--threads", "-n", type=int, default=1)
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=run)
```

- [ ] **Step 3: 在 `entomokit/main.py` 注册命令**

在 lazy import 区域加入并注册：

```python
from entomokit import measure as _measure
...
_measure.register(subparsers)
```

- [ ] **Step 4: 运行测试验证命令接线通过**

运行：
- `pytest tests/test_main_cli.py::test_top_level_command_order_matches_dataset_workflow -q`
- `python -m entomokit.main measure --help`

预期：测试通过，help 显示 `--mask-dir/--out-dir/--pixel-size-um`。

- [ ] **Step 5: Commit**

```bash
git add entomokit/main.py entomokit/measure.py tests/test_main_cli.py
git commit -m "feat(measure): add CLI command scaffold and register in main"
```

---

### Task 2: 掩码读取、连通域筛选与批处理输入校验

**Files:**
- Create: `src/measurement/io.py`
- Create: `src/measurement/__init__.py`
- Test: `tests/test_measure_metrics.py`

- [ ] **Step 1: 写失败测试（掩码归一和最大连通域）**

在 `tests/test_measure_metrics.py` 增加：

```python
def test_keep_largest_component_discards_small_islands() -> None:
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:20, 5:20] = 255
    mask[0:2, 0:2] = 255
    kept = keep_largest_component(mask > 0)
    assert kept.sum() == 15 * 15
```

运行：`pytest tests/test_measure_metrics.py::test_keep_largest_component_discards_small_islands -q`  
预期：FAIL（函数不存在）。

- [ ] **Step 2: 实现读取与归一函数**

在 `src/measurement/io.py` 实现：

```python
def load_binary_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (image > 0).astype(np.uint8)
```

- [ ] **Step 3: 实现最大连通域筛选**

在 `src/measurement/io.py` 增加：

```python
def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return (labels == idx).astype(np.uint8)
```

- [ ] **Step 4: 运行测试**

运行：`pytest tests/test_measure_metrics.py -q`  
预期：新增用例通过。

- [ ] **Step 5: Commit**

```bash
git add src/measurement/__init__.py src/measurement/io.py tests/test_measure_metrics.py
git commit -m "feat(measure): add mask loading and largest-component preprocessing"
```

---

### Task 3: 核心与形状层指标（regionprops 对齐口径）

**Files:**
- Create: `src/measurement/core.py`
- Test: `tests/test_measure_metrics.py`

- [ ] **Step 1: 写失败测试（面积/周长/主次轴/圆度）**

新增矩形掩码用例：

```python
def test_basic_metrics_for_axis_aligned_rectangle() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 1
    m = compute_core_metrics(mask)
    assert m["area_px"] == 60 * 40
    assert m["major_axis_px"] >= m["minor_axis_px"]
    assert 0 < m["circularity"] < 1
```

运行：`pytest tests/test_measure_metrics.py::test_basic_metrics_for_axis_aligned_rectangle -q`  
预期：FAIL。

- [ ] **Step 2: 实现核心形状指标**

在 `src/measurement/core.py` 实现：

```python
def compute_core_metrics(mask: np.ndarray) -> dict[str, float]:
    cnt = largest_contour(mask)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    convex_area = float(cv2.contourArea(hull))
    x, y, w, h = cv2.boundingRect(cnt)
    major, minor, ecc = ellipse_like_axes_from_moments(mask)
    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "major_axis_px": major,
        "minor_axis_px": minor,
        "eccentricity": ecc,
        "solidity": area / convex_area if convex_area > 0 else 0.0,
        "extent": area / float(w * h) if w > 0 and h > 0 else 0.0,
        "circularity": (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0,
        "convex_area_px": convex_area,
        "equivalent_diameter_px": np.sqrt(4.0 * area / np.pi) if area > 0 else 0.0,
    }
```

- [ ] **Step 3: 运行目标测试并补充旋转不敏感断言**

新增旋转后面积接近断言并运行：`pytest tests/test_measure_metrics.py -q`  
预期：通过。

- [ ] **Step 4: Commit**

```bash
git add src/measurement/core.py tests/test_measure_metrics.py
git commit -m "feat(measure): implement core shape metrics with regionprops-aligned definitions"
```

---

### Task 4: 主干体长、法向体宽与高级指标

**Files:**
- Create: `src/measurement/skeleton.py`
- Modify: `src/measurement/core.py`
- Test: `tests/test_measure_metrics.py`

- [ ] **Step 1: 写失败测试（弯曲体长 > 主轴长度）**

新增弯曲“幼虫”合成掩码测试：

```python
def test_curved_body_length_exceeds_major_axis() -> None:
    mask = synthetic_curved_larva_mask()
    m = compute_all_metrics(mask)
    assert m["body_length_px"] >= m["major_axis_px"]
    assert m["curvature_index"] >= 1.0
```

运行：`pytest tests/test_measure_metrics.py::test_curved_body_length_exceeds_major_axis -q`  
预期：FAIL。

- [ ] **Step 2: 实现骨架与主干近似**

在 `src/measurement/skeleton.py` 实现 Zhang-Suen 细化（或等价迭代细化）与分支剪枝：

```python
def extract_backbone(mask: np.ndarray, prune_ratio: float = 0.06) -> np.ndarray:
    skel = zhang_suen_thinning(mask)
    graph = build_pixel_graph(skel)
    graph = prune_short_branches(graph, min_len=max(3, int(prune_ratio * diagonal(mask))))
    return graph_to_mask(graph, mask.shape)
```

- [ ] **Step 3: 在 `core.py` 集成高级指标**

实现并回填：
- `body_length_px`（主干最长路径）
- `body_width_px`（法向截线中位数）
- `max_feret_px` / `min_feret_px`
- `curvature_index`
- `thickness_cv`
- `symmetry_score`

- [ ] **Step 4: 质量标记与兜底**

当骨架失败时回退：

```python
rect = cv2.minAreaRect(cnt)
length = max(rect[1])
width = min(rect[1])
warn_reason.append("fallback_rect_used")
```

并在输出中加入 `quality_flag` / `warn_reason`。

- [ ] **Step 5: 运行测试**

运行：`pytest tests/test_measure_metrics.py -q`  
预期：弯曲、旋转、高级指标测试通过。

- [ ] **Step 6: Commit**

```bash
git add src/measurement/skeleton.py src/measurement/core.py tests/test_measure_metrics.py
git commit -m "feat(measure): add backbone-based body length and advanced morphology metrics"
```

---

### Task 5: 批处理服务与 CSV 导出

**Files:**
- Create: `src/measurement/service.py`
- Modify: `entomokit/measure.py`
- Test: `tests/test_measure_cli.py`

- [ ] **Step 1: 写失败测试（生成两份 CSV）**

在 `tests/test_measure_cli.py` 增加端到端用例，运行 `run(args)` 后断言：

```python
assert (out_dir / "metrics.csv").exists()
assert (out_dir / "metrics_summary.csv").exists()
```

运行：`pytest tests/test_measure_cli.py::test_measure_writes_metrics_and_summary_csv -q`  
预期：FAIL。

- [ ] **Step 2: 实现批处理服务**

在 `src/measurement/service.py` 实现：

```python
def run_batch(mask_dir: Path, out_dir: Path, pixel_size_um: float | None) -> dict[str, int]:
    rows = []
    for path in iter_mask_files(mask_dir):
        rows.append(measure_one_mask(path, pixel_size_um=pixel_size_um))
    write_metrics_csv(out_dir / "metrics.csv", rows)
    write_summary_csv(out_dir / "metrics_summary.csv", rows)
    return summarize_counts(rows)
```

- [ ] **Step 3: 在 CLI run() 中接入日志与返回码**

`entomokit/measure.py` 复用 `src.common.cli`：
- `setup_logging(out_dir, verbose=args.verbose)`
- `save_log(out_dir, args)`
- 异常时 `sys.exit(1)`。

- [ ] **Step 4: 运行测试**

运行：
- `pytest tests/test_measure_cli.py -q`
- `python -m entomokit.main measure --mask-dir <tmp_masks> --out-dir <tmp_out>`

预期：CLI 成功，生成两份 CSV。

- [ ] **Step 5: Commit**

```bash
git add entomokit/measure.py src/measurement/service.py tests/test_measure_cli.py
git commit -m "feat(measure): batch processing and csv exports for metrics"
```

---

### Task 6: 比例尺（um/px）与回归验证

**Files:**
- Modify: `src/measurement/service.py`
- Modify: `tests/test_measure_cli.py`
- Modify: `tests/test_measure_metrics.py`

- [ ] **Step 1: 写失败测试（输出 `*_um`）**

新增断言：传入 `--pixel-size-um 2.5` 时 `metrics.csv` 包含 `area_um2`、`body_length_um`、`body_width_um` 列。

运行：`pytest tests/test_measure_cli.py::test_measure_adds_um_columns_when_scale_provided -q`  
预期：FAIL。

- [ ] **Step 2: 实现单位换算列**

在行组装处增加：

```python
if pixel_size_um is not None:
    row["body_length_um"] = row["body_length_px"] * pixel_size_um
    row["body_width_um"] = row["body_width_px"] * pixel_size_um
    row["area_um2"] = row["area_px"] * (pixel_size_um ** 2)
```

- [ ] **Step 3: 运行完整回归**

运行：
- `pytest tests/test_measure_metrics.py -q`
- `pytest tests/test_measure_cli.py -q`
- `pytest tests/test_main_cli.py -q`

预期：新增与相关回归全部通过。

- [ ] **Step 4: Commit**

```bash
git add src/measurement/service.py tests/test_measure_cli.py tests/test_measure_metrics.py
git commit -m "feat(measure): add um scale conversion and regression coverage"
```

---

## Final Verification Checklist

- [ ] `python -m entomokit.main measure --help` 可正常显示。
- [ ] 对 SAM3 掩码目录运行后生成：`metrics.csv`、`metrics_summary.csv`、`log.txt`。
- [ ] 非失败样本核心指标无空值。
- [ ] 弯曲样本 `body_length_px >= major_axis_px`（少量异常需 `warn_reason`）。
- [ ] 传入 `--pixel-size-um` 后出现 `*_um` 派生列。
