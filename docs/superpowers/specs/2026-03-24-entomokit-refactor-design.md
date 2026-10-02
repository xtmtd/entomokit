# entomokit 重构设计文档

**日期**: 2026-03-24\
**最后更新**: 2026-10-01（文档约定：命令参考迁至 `docs/commands/`、README 边界、help 链接与长期文档规则）\
**状态**: 已确认；重构已在 0.7.0 实现。本文档保留仍然有效的架构决策与跨模块约束，历史决策不再重复；当前用户参数与行为以 `docs/commands/` 参考为准。

---

## 1. 背景与目标

当前 entomokit 是五个相互独立的脚本（`scripts/segment.py` 等），各自拥有独立入口和 argparse，虽共享 `src/common/` 底层工具，但整体入口散乱、可扩展性差。

重构目标：
1. 采用主命令/子命令分层架构（参考 detcli），统一入口为 `entomokit`
2. 将 `add_functions/` 中的 AutoGluon 图片分类和 GradCAM 热力图功能整合进来
3. 共享底层框架（`src/common/`），减少重复代码
4. 为将来增加新功能组（如目标检测）预留扩展空间

---

## 2. 命令树

```text
entomokit
├── extract-frames   # 视频帧提取
├── segment          # 昆虫图像分割（SAM3/Otsu/GrabCut）
├── measure          # 形态学测量
├── synthesize       # 图像合成
├── clean            # 图像清洗与去重
├── augment          # 图像增强
├── split-csv        # CSV 数据集分割（原 split_dataset.py）
├── classify         # AutoGluon 图片分类组
│   ├── train        # 训练模型
│   ├── predict      # 推理预测（支持 AutoGluon / ONNX）
│   ├── evaluate     # 分类性能评估（支持 AutoGluon / ONNX）
│   ├── embed        # 嵌入提取 + 嵌入空间质量指标 + UMAP 可视化
│   ├── cam          # GradCAM 系列热力图（仅 PyTorch）
│   └── export-onnx  # 模型导出为 ONNX
├── doctor           # 环境与依赖诊断
├── update           # 检查并可选安装新版本
└── completion       # shell 补全脚本（子命令：bash/zsh/fish）
```

---

## 3. 目录结构

```text
entomokit/                       # CLI 入口包：只做参数解析与分发
├── main.py                      # 顶层 dispatcher
├── help_style.py                # 共享 help 格式化 + DOCS_BASE_URL/DOC_LINKS
├── cli_schema.py                # 运行时参数 schema（skill 与文档测试共用）
├── param_guard.py / execution_policy.py / workflow_gate.py   # 参数与执行门禁
├── <command>.py                 # 每个顶层命令一个注册模块
└── classify/                    # classify 组注册模块

src/                             # 领域逻辑：不含 argparse
├── common/                      # 共享工具（cli、logging、resume、annotation_writer、validators）
├── segmentation/                # 分割处理包（processor）
├── segmentation.py              # 同名的模块入口；两者并存，职责不同
├── framing/ cleaning/ augment/ splitting/ measurement/ synthesis/
├── classification/              # AutoMM / ONNX / 嵌入 / CAM / 导出
├── sam3/ lama/                  # 模型实现
└── doctor/                      # 环境诊断

tests/  data/  docs/  skills/    # 测试、示例数据、文档、AI skill
setup.py  requirements.txt
```

### 设计原则

- `entomokit/` 中的模块**只负责 CLI 参数解析和调用分发**，不含业务逻辑
- `src/` 中的模块**只含业务逻辑**，不含 argparse
- `src/common/` 被所有命令共享，新功能同样复用
- 命令的当前用户参数与行为以 `docs/commands/` 参考为准，本文档不复制参数表（见 §11）

---

## 4. 入口机制

### setup.py 入口点

旧的五个独立入口点全部移除，改为单一入口：

```python
entry_points={
    "console_scripts": [
        "entomokit=entomokit.main:main",
    ],
}
```

### 顶层 dispatcher（`entomokit/main.py`）

```python
import argparse

def main():
    parser = argparse.ArgumentParser(prog="entomokit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 注册各命令
    from entomokit import segment, extract_frames, clean, split_csv, synthesize
    from entomokit.classify import register as register_classify

    segment.register(subparsers)
    extract_frames.register(subparsers)
    clean.register(subparsers)
    split_csv.register(subparsers)
    synthesize.register(subparsers)
    register_classify(subparsers)

    args = parser.parse_args()
    args.func(args)
```

每个命令模块暴露 `register(subparsers)` 函数和对应的 `run(args)` 函数。

### classify 组 dispatcher（`entomokit/classify/__init__.py`）

```python
def register(subparsers):
    classify_parser = subparsers.add_parser("classify")
    classify_sub = classify_parser.add_subparsers(dest="subcommand", required=True)

    from entomokit.classify import train, predict, evaluate, embed, cam, export_onnx
    train.register(classify_sub)
    predict.register(classify_sub)
    evaluate.register(classify_sub)
    embed.register(classify_sub)
    cam.register(classify_sub)
    export_onnx.register(classify_sub)
```

---

## 5. 各命令参数规范

### 5.0 全局目录输入与输出策略（0.7.0 起）

```text
面向目录的命令默认递归扫描。
普通文件输出镜像每个输入文件相对于输入根目录的目录结构，但部分命令会重写文件名
（clean 规范化词干、保证目录内唯一并按 --out-image-format 改扩展名；augment 追加 _augN，
按 --multiply 的位数补零）。
segment 是例外：它不镜像输入目录，而是把所有图像平铺到 images/ 下，并写到各自的
注释目录，同时把输入相对路径编码为唯一的扁平样本 ID。完整的标准数据集布局
（例如 Pascal VOC 的 JPEGImages/）由后续转换/划分步骤生成。
普通目录处理命令不提供扁平化选项。
CSV 驱动的命令仍以 CSV 为准，不适用本目录策略。
clean --pad-color none|median|black|white 默认 none。
```

适用命令：`extract-frames`、`segment`、`measure`、`synthesize`、`clean`、
`augment`，以及 `classify embed` / `classify cam` / `classify predict` 的目录发现。

- `extract-frames`：视频目录递归扫描，帧写入
  `out-dir/<视频输入相对目录>/<视频词干>/`，同名视频互不覆盖。
- `measure`：掩码递归扫描，`metrics.csv` 的 `file_name` 为相对 `--mask-dir`
  的路径，并作为 `--resume` 的键。
- `segment`：递归扫描并把图像平铺到 `images/`（不镜像输入目录）；样本 ID = 可读词干 +
  `sha256(相对路径)[:12]`（如 `a__4cabcf2b3682`），用于图像、VOC XML、
  YOLO TXT、SegmentationClass 掩码、COCO 文件名与
  `ImageSets/Main/default.txt`；`--resume` 检查该映射产物。
- `synthesize`：目标与背景目录递归扫描，输出按目标相对路径镜像，背景采样
  有放回。
- `clean`：移除 `--recursive` / `--flatten`，始终递归、镜像父目录并重写文件名；新增
  `--pad-color`（`none`/`median`/`black`/`white`，默认 `none`）。

### 5.1 顶层独立命令

这五个命令的业务逻辑基本保留，只是入口从 `python scripts/xxx.py` 迁移至 `entomokit xxx`，部分命令有功能扩展。

| 新命令                      | 对应旧脚本               | 变化                              |
| --------------------------- | ------------------------ | --------------------------------- |
| `entomokit segment`         | `scripts/segment.py`     | 入口改变 + 注释输出格式对齐 detcli |
| `entomokit extract-frames`  | `scripts/extract_frames.py` | 入口改变 + `--input-dir` 支持单文件 |
| `entomokit clean`           | `scripts/clean_figs.py`  | 入口改变 + 默认递归 + 镜像输出 + `--pad-color` |
| `entomokit split-csv`       | `scripts/split_dataset.py` | 入口改变 + 命令改名 + 新增 val/copy-images |
| `entomokit synthesize`      | `scripts/synthesize.py`  | 入口改变 + 注释输出格式对齐 detcli |

当前每个命令的用户参数、输出契约与注意事项以 [命令参考](../../../docs/commands/) 为准（例如 [segment](../../../docs/commands/segment.md)、[clean](../../../docs/commands/clean.md)）；下面只保留跨命令的架构结论，不再重复参数表。

### `entomokit segment` / `entomokit synthesize` — 注释输出约定

`segment` 与 `synthesize` 都把图像写在 `images/` 下（`segment` 用编码后的扁平样本 ID，`synthesize` 按目标相对路径镜像）。注释写到各格式目录，但两个命令的 COCO 能力不同：`segment` 按 `--coco-output-mode` 写 `annotations.coco.json`（unified）或 `annotations/*.json`（separate），而 `synthesize` 接受该参数但当前只写 unified 的 `annotations.coco.json`（separate 尚未实现）。YOLO 都是 `labels/*.txt` + 输出根目录的 `data.yaml`，VOC 都是 `Annotations/*.xml`（`segment` 另写 `ImageSets/Main/default.txt`，mask 模式写 `SegmentationClass/*.png`）。标准数据集布局（例如 VOC `JPEGImages/`）由后续转换/划分步骤生成。

标注语义（bbox 与 mask、`area`、polygon）与格式级行为由专项设计拥有：[Segment 注释语义设计](2026-04-13-segment-annotation-semantics-design.md)；当前参数见 [segment 参考](../../../docs/commands/segment.md) 与 [synthesize 参考](../../../docs/commands/synthesize.md)。
### `entomokit clean` — 默认递归与填充

`clean` 始终递归扫描 `--input-dir` 并把输出写到 `out-dir/cleaned_images/` 下；不再提供 `--recursive` 或 `--flatten`。镜像的是输入文件的**父目录**，文件名会被重写（词干规范化、同目录内大小写不敏感的唯一后缀、扩展名跟 `--out-image-format`），因此不能按原名关联标签。`--pad-color` 等当前参数见 [clean 参考](../../../docs/commands/clean.md)。
### `entomokit segment` — 递归输入与样本 ID 编码

`segment` 递归扫描输入，把所有图像平铺到 `images/` 下（不镜像输入目录），把每个输入
相对路径编码为唯一的扁平样本 ID（可读词干 + `sha256(相对路径)[:12]`）。`--resume` 依据该样本 ID
检查映射后的输出产物，而非按基名 glob。

### `entomokit extract-frames` — `--input-dir` 增强

`--input-dir` 同时接受：
- 目录路径：扫描目录下所有支持的视频文件（原有行为）
- 单个视频文件路径：直接处理该文件，无需创建临时目录

### `entomokit split-csv` — 划分输出

`split-csv` 以 `--raw-image-csv` 为准，按 `ratio` 或 `count` 模式生成 `train.csv`、可选的 `val.csv`、`test.known.csv` 与可选的 `test.unknown.csv`，并写 `class_count/` 统计；`--copy-images` 时按划分把图像复制到 `images/{train,val,test_known,test_unknown}/`。完整参数与输出见 [split-csv 参考](../../../docs/commands/split-csv.md)。
### 5.2 `classify train`

当前契约：AutoGluon MultiModalPredictor + timm 骨干是唯一训练路径；训练集来自 `--train-csv`，验证由 AutoGluon 内部完成（因此 `split-csv` 的 `--val-ratio`/`--val-count` 保持 0）；产物写入 `out-dir/AutogluonModels/<base-model>`，供 predict/evaluate/embed/cam/export-onnx 复用；`--resume` 需在同一 `--base-model` 下延长轮数上限。

当前参数、默认值与增强预设语义见 [classify train 参考](../../../docs/commands/classify-train.md)；原始实现任务见 [Phase 3 plan](../plans/2026-03-24-phase3-classify.md)（历史）。
### 5.3 `classify predict`

当前契约：`--input-csv` 与 `--images-dir` 至少提供其一；`--images-dir` 递归发现并按相对路径记录（嵌套同名图像不冲突），显式 CSV 路径按原样使用；`--model-dir` 与 `--onnx-model` 二选一，ONNX 需要 `onnxruntime`，并在存在 `label_classes.json` 时输出类别名。

当前参数见 [classify predict 参考](../../../docs/commands/classify-predict.md)。
### 5.4 `classify evaluate`

当前契约：需要一个带标签的 `--test-csv`；输出 `evaluations.csv`（总体指标）、原始与行归一化混淆矩阵、`per_class_metrics.csv`，以及类别数可读时的 `confusion_matrix.pdf`；`--model-dir` 与 `--onnx-model` 二选一。

参数与指标清单见 [classify evaluate 参考](../../../docs/commands/classify-evaluate.md)。
### 5.5 `classify embed`

当前契约：`--model-dir` 复用微调骨干，否则使用 `--base-model` 的预训练 timm 骨干；`--label-csv` 的 `image` 必须唯一且与图像文件名有交集；质量指标（NMI/ARI/Recall@K/kNN/线性探针含 balanced/mAP@R/Purity/Silhouette）使用固定随机种子 42 的交叉验证，不可计算时写 `N/A` 而非 0；`--metrics-sample-size` 限制全部指标的样本量，是主要的运行时间与内存开关；`0.6.2` 起指标不可与更早运行数值比较，`0.7.0` 未改动该算法。

参数与指标定义见 [classify embed 参考](../../../docs/commands/classify-embed.md)。
### 5.6 `classify cam`

当前契约：CAM 依赖 PyTorch hook，因此不支持 ONNX；架构自动检测（Swin 按 ViT 风格取末 stage block，ConvNeXt 取末 stage block 而非 `mlp.fc2`）；`--save-npy raw` 保留未归一化幅值，只在同一模型、目标层与预处理配置内可比，`normalized` 为该图 min-max；`--eval-transform center-crop|whole-specimen-pad` 决定热力图视野。

当前参数见 [classify cam 参考](../../../docs/commands/classify-cam.md)。

**模型与预处理不变量**（无独立设计 owner，保留在本设计）：

- 使用 `--model-dir` 时，CAM 包装 AutoGluon 的 `backbone -> classification head`，解释最终训练类别的 logits；`pred_class` 写入 predictor 的真实类别标签，而不是 backbone 特征维度索引。
- CAM 复用 predictor 保存的 `ImageProcessor.val_processor`：保存的 `image_size` 是唯一输入尺寸来源（224、384 或自定义尺寸自动适配）；没有已构建 processor 时才由保存的 `val_transforms` 重建。
- `center-crop`（默认）保留该验证预处理并把热图逆映射回完整原图：模型视野外的像素被压暗去色而不是拉伸，绝不把裁剪区域铺满整幅图。`whole-specimen-pad` 先用图像四边逐通道中位色补成方形，再缩放到保存的输入尺寸。
- `--base-model` 且无保存 processor 时使用 timm 数据配置，热图映射保留实际的 resize 与 crop 尺寸（例如 `Resize(256) -> CenterCrop(224)`）。
- ViT 默认目标层 `blocks[-1].norm1`（去除 CLS token）；Swin 为 `layers[-1].blocks[-1].norm1`（channel-last 适配，`ablationcam` 使用对应适配器）；ConvNeXt 为最后 stage 的最后一个完整 block（例如 `stages.3.blocks.1`）而不是 pointwise `mlp.fc2`，其余 CNN 回退到最后一个 `Conv2d`，也可用 `--target-layer-name` 显式指定。
- 保存的数组是**模型输入空间**的 float32：`raw` 保留未归一化正值 CAM 幅值（跳过库内 `scale_cam_image()`，保留 ReLU 与 resize 到模型输入尺寸），overlay 始终使用独立的逐图 min-max 副本；`normalized` 在 float32 容差内与此前的归一化模型空间 mask 一致，但非逐位相同，也非跨模型/跨输入保证；overlay PNG 的像素差异不构成兼容性契约。
- `raw` 幅值只在同一模型、同一目标层、同一预处理配置内可比；`eigencam` 的 raw 值来自符号任意的 SVD 投影，不可用于响应强度统计。

输出目录与列定义见 [classify cam 参考](../../../docs/commands/classify-cam.md)。
### 5.7 `classify export-onnx`

当前契约：把 AutoGluon predictor 导出为 `model.onnx`，并同时写出 `label_classes.json`（`classify predict` 用它输出类别名）；`--opset` 默认 17；`--sample-image` 可选，缺省使用自动生成的临时图像做 trace。

当前参数见 [classify export-onnx 参考](../../../docs/commands/classify-export-onnx.md)。
## 6. ONNX 支持范围说明

| 命令 | AutoGluon | ONNX |
|------|-----------|------|
| `train` | 是 | — |
| `predict` | 是 | 是 |
| `evaluate` | 是 | 是 |
| `embed` | 是（fine-tuned backbone） | **否**（需 PyTorch hook，ONNX 不支持） |
| `cam` | 是 | **否**（技术限制）|
| `export-onnx` | 是（输入） | 是（输出） |

---

## 7. CPU/线程控制说明

除 `export-onnx` 外，classify 命令均支持 `--device`；支持批处理或 DataLoader 的命令还支持 `--num-workers`，CPU 计算或 ONNX 推理命令还支持 `--num-threads`。`classify cam` 逐图生成 CAM，不提供 `--num-workers`，其 `--cam-batch-size` 只控制 ScoreCAM/EigenCAM 的内部批量；默认 `--num-threads=0` 由框架决定。`export-onnx` 无需并发/设备参数。

`segment` 的 CPU 并发与 SAM3 串行边界、以及线程数的选择建议由专项设计拥有：[Segment CPU Parallelism Design](2026-07-10-segment-cpu-parallelism-design.md)。本设计不再重复其参数表。

每个命令的当前并发参数见对应命令参考。

---

## 8. 依赖管理

AutoGluon 和 grad-cam 是重型依赖，通过 `setup.py` 的 `extras_require` 管理，不作为默认安装依赖：

```python
extras_require={
    "segmentation": [...],
    "cleaning": [...],
    "video": [...],
    "data": [...],
    "classify": [
        "autogluon.multimodal",
        "timm",
        "umap-learn",
        "grad-cam",
        "onnxruntime",
        "scikit-learn",
    ],
    "dev": [...],
}
```

安装方式：
```bash
pip install -e ".[classify]"
```

---

## 9. 向后兼容说明

- `scripts/` 与 `add_functions/` 已随 0.7.0 移除，不再是入口（历史记录）
- 旧的 `setup.py` entry_points（`entomokit-segment` 等）已在迁移中移除
- 原有参数名（下划线风格如 `--input_dir`）迁移后统一改为连字符风格（`--input-dir`）；以下为主要改名对照：

| 旧参数（scripts/）        | 新参数（entomokit CLI）  |
| ------------------------- | ------------------------ |
| `--input_dir`             | `--input-dir`            |
| `--out_dir`               | `--out-dir`              |
| `--out_image_format`      | `--out-image-format`     |
| `--sam3-checkpoint`       | `--sam3-checkpoint`（不变） |
| `--segmentation-method`   | `--segmentation-method`（不变） |
| `--dedup_mode`            | `--dedup-mode`           |
| `--out_short_size`        | `--out-short-size`       |
| `--raw_image_csv`         | `--raw-image-csv`        |
| `--unknown_test_classes_ratio` | `--unknown-test-classes-ratio` |
| `--start_time`            | `--start-time`           |
| `--end_time`              | `--end-time`             |
| `--max_frames`            | `--max-frames`           |

---

## 10. 文档约定（长期规则）

文档分层与长期规则以 [文档约定设计](2026-10-01-entomokit-documentation-conventions-design.md) 为准，本节只记录要点：

- 分层归属：README 负责项目入口、安装、操作命令与共享行为；`docs/commands/*.md` 负责各功能命令的完整参数与输入输出契约；本设计负责架构决策与跨模块不变量；历史迁移事实由 plans 记录。
- 双语：每个命令参考都有同名的 `.cn.md` 中文镜像，两边章节顺序与语义一致。
- 当前参数：以 CLI 实现与运行时 schema 为准；命令参考是散文版完整参考，本设计不复制参数表。
- help 边界：每个 parser 的 description 末尾追加文档链接（`entomokit/help_style.py` 的 `DOCS_BASE_URL` 与 `DOC_LINKS` 唯一定义），逐项 option help 不改写。
- reference-first：细节已有专项设计时，本设计只保留结论与链接；plan 只作历史证据，不承载当前不变量。
- 新增命令：注册、双语参考、README 索引行、help 指针、Version Notes 与文档检查必须同批完成（详见设计 §6）。

---

## 11. 未来扩展预留

顶层命令组的设计允许未来增加新的功能组，例如：

```
entomokit detect    # 目标检测组（未来）
entomokit track     # 目标追踪组（未来）
```

每个新组只需在 `entomokit/main.py` 中注册，对现有代码零影响。
