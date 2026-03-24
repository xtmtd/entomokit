# entomokit 重构设计文档

**日期**: 2026-03-24  
**最后更新**: 2026-03-24（补充 segment 注释格式、clean/split-csv/extract-frames 功能扩展、classify CPU 线程控制）  
**状态**: 已确认，待实现  

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

```
entomokit
├── segment          # 昆虫图像分割（SAM3/Otsu/GrabCut）
├── extract-frames   # 视频帧提取
├── clean            # 图像清洗与去重
├── split-csv        # CSV 数据集分割（原 split_dataset.py）
├── synthesize       # 图像合成
└── classify         # AutoGluon 图片分类组
    ├── train        # 训练模型
    ├── predict      # 推理预测（支持 AutoGluon / ONNX）
    ├── evaluate     # 分类性能评估（支持 AutoGluon / ONNX）
    ├── embed        # 嵌入提取 + 嵌入空间质量指标 + UMAP 可视化
    ├── cam          # GradCAM 系列热力图（仅 PyTorch）
    └── export-onnx  # 模型导出为 ONNX
```

---

## 3. 目录结构

```
entomokit/
├── entomokit/                   # CLI 入口包（新增）
│   ├── __init__.py
│   ├── main.py                  # 顶层 dispatcher
│   ├── segment.py               # CLI 参数解析 → 调用 src/segmentation/
│   ├── extract_frames.py
│   ├── clean.py
│   ├── split_csv.py             # 原 split_dataset.py 改名
│   ├── synthesize.py
│   └── classify/
│       ├── __init__.py          # classify 组 dispatcher
│       ├── train.py
│       ├── predict.py
│       ├── evaluate.py
│       ├── embed.py
│       ├── cam.py
│       └── export_onnx.py
├── src/                         # 业务逻辑包（现有结构保持不变）
│   ├── common/                  # 共享工具（logging, validators, cli）
│   ├── segmentation/
│   ├── framing/
│   ├── cleaning/
│   ├── splitting/
│   ├── synthesis/
│   └── classification/          # 新增：classify 业务逻辑
│       ├── __init__.py
│       ├── trainer.py           # AutoGluon 训练逻辑
│       ├── predictor.py         # 推理（AutoGluon + ONNX）
│       ├── evaluator.py         # 评估（AutoGluon + ONNX）
│       ├── embedder.py          # 嵌入提取 + 质量指标 + UMAP
│       ├── cam.py               # GradCAM 热力图（仅 PyTorch）
│       └── exporter.py          # ONNX 导出
├── scripts/                     # 保留，过渡期不删除，不再作为主入口
├── add_functions/               # 保留原始脚本，作为参考，不再直接使用
├── tests/
├── data/
├── setup.py
└── requirements.txt
```

### 设计原则

- `entomokit/` 中的模块**只负责 CLI 参数解析和调用分发**，不含业务逻辑
- `src/` 中的模块**只含业务逻辑**，不含 argparse
- `src/common/` 被所有命令共享，新功能同样复用
- `scripts/` 旧目录保留但不再是主入口（过渡期后可移除）

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

### 5.1 顶层独立命令

这五个命令的业务逻辑基本保留，只是入口从 `python scripts/xxx.py` 迁移至 `entomokit xxx`，部分命令有功能扩展。

| 新命令                      | 对应旧脚本               | 变化                              |
| --------------------------- | ------------------------ | --------------------------------- |
| `entomokit segment`         | `scripts/segment.py`     | 入口改变 + 注释输出格式对齐 detcli |
| `entomokit extract-frames`  | `scripts/extract_frames.py` | 入口改变 + `--input-dir` 支持单文件 |
| `entomokit clean`           | `scripts/clean_figs.py`  | 入口改变 + 新增 `--recursive`      |
| `entomokit split-csv`       | `scripts/split_dataset.py` | 入口改变 + 命令改名 + 新增 val/copy-images |
| `entomokit synthesize`      | `scripts/synthesize.py`  | 入口改变 + 注释输出格式对齐 detcli |

### `entomokit segment` — 注释格式变更

`segment` 和 `synthesize` 生成的注释文件格式与目录布局需与 detcli 保持一致（使用 `supervision` 库），规范如下：

| 格式     | 目录布局                                                                                   | 说明                                       |
| -------- | ------------------------------------------------------------------------------------------ | ------------------------------------------ |
| **COCO** | 图像与 JSON 同级平铺，JSON 文件名固定为 `annotations.coco.json`                               | bbox 格式由 `--coco-bbox-format` 控制，默认 `xywh`；支持 `xywh`/`xyxy` |
| **YOLO** | `images/` + `labels/` 子目录，附带 `data.yaml`（含 `nc` 和带引号的 `names` 列表）              | 与 detcli 完全一致                         |
| **VOC**  | `JPEGImages/` + `Annotations/` + `ImageSets/Main/default.txt`                               | 标准 Pascal VOC 布局                       |

`segment` 新增参数：
- `--annotation-format`：`coco`/`yolo`/`voc`，选择注释输出格式（原脚本已有该参数，对齐格式规范）
- `--coco-bbox-format`：`xywh`/`xyxy`，COCO 格式时 bbox 的坐标约定，默认 `xywh`

### `entomokit clean` — 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--recursive` | flag | False | 递归扫描 `--input-dir` 下的所有子目录 |

### `entomokit extract-frames` — `--input-dir` 增强

`--input-dir` 同时接受：
- 目录路径：扫描目录下所有支持的视频文件（原有行为）
- 单个视频文件路径：直接处理该文件，无需创建临时目录

### `entomokit split-csv` — 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--val-ratio` | float | 0 | `ratio` 模式下验证集占 known 数据比例，0 表示不生成 val |
| `--val-count` | int | 0 | `count` 模式下验证集样本数，0 表示不生成 val |
| `--images-dir` | str | 可选 | 图像来源目录（`--copy-images` 时必填） |
| `--copy-images` | flag | False | 按 split 将图像复制到对应子目录 |

**输出结构**（含所有可选输出）：
```
out_dir/
├── train.csv
├── val.csv              # 仅 --val-ratio/--val-count > 0 时
├── test.known.csv
├── test.unknown.csv     # 仅 unknown > 0 时
├── images/              # 仅 --copy-images 时
│   ├── train/
│   ├── val/
│   ├── test_known/
│   └── test_unknown/
└── class_count/
    ├── class.count
    ├── class.train.count
    ├── class.val.count
    ├── class.test.known.count
    └── class.test.unknown.count
```

### 5.2 `classify train`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train-csv` | str | 必填 | CSV(image,label) |
| `--images-dir` | str | 必填 | 训练图像目录 |
| `--base-model` | str | `convnextv2_femto` | timm backbone 名称 |
| `--out-dir` | str | 必填 | 输出目录 |
| `--augment` | str | `medium` | 增强预设：`none`/`light`/`medium`/`heavy`，或 JSON 字符串自定义 |
| `--max-epochs` | int | 50 | 最大训练轮数 |
| `--time-limit` | float | 1.0 | 训练时限（小时） |
| `--focal-loss` | flag | False | 启用 focal loss |
| `--focal-loss-gamma` | float | 1.0 | focal loss gamma 值 |
| `--device` | str | `auto` | `auto`/`cpu`/`cuda`/`mps` |
| `--batch-size` | int | 32 | 训练 batch size |
| `--num-workers` | int | 4 | DataLoader worker 数 |
| `--num-threads` | int | 0 | CPU 计算线程数（0=框架自动）；`device=cpu` 时设置 `torch.set_num_threads()` |

**增强预设映射**（基于 AutoGluon `model.timm_image.train_transforms`）：

| 预设 | transforms |
|------|------------|
| `none` | `["resize_shorter_side", "center_crop"]` |
| `light` | `["resize_shorter_side", "center_crop", "random_horizontal_flip"]` |
| `medium` | `["resize_shorter_side", "center_crop", "random_horizontal_flip", "color_jitter", "trivial_augment"]` |
| `heavy` | `["random_resize_crop", "random_horizontal_flip", "random_vertical_flip", "color_jitter", "trivial_augment", "randaug"]` |

自定义方式：传入 JSON 数组字符串，如 `'["random_resize_crop","color_jitter"]'`

传入无效预设名称或非法 JSON 时，立即报错退出（不静默回退到 `medium`）。JSON 数组中包含未知 transform 名称时，同样在解析阶段报错退出，提示可用名称列表。

**输出结构**：
```
out_dir/
├── AutogluonModels/
│   └── {base_model}/          # AutoGluon predictor 目录
├── train.processed.csv
└── logs/
    └── log.txt
```

### 5.3 `classify predict`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-csv` | str | 与 `--images-dir` 二选一 | CSV(image) 或 CSV(image,label) |
| `--images-dir` | str | 与 `--input-csv` 二选一 | 直接扫描目录中的图像 |
| `--model-dir` | str | 与 `--onnx-model` 二选一 | AutoGluon predictor 目录 |
| `--onnx-model` | str | 与 `--model-dir` 二选一 | ONNX 模型路径 |
| `--out-dir` | str | 必填 | 输出目录 |
| `--batch-size` | int | 32 | 推理 batch size |
| `--num-workers` | int | 4 | DataLoader worker 数 |
| `--num-threads` | int | 0 | CPU 计算线程数（0=框架自动） |
| `--device` | str | `auto` | `auto`/`cpu`/`cuda`/`mps` |

**输出**：
```
out_dir/
└── predictions/
    └── predictions.csv        # 列：image, stem, [label（若输入含 label 列则保留）], prediction, proba_* 
```

### 5.4 `classify evaluate`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test-csv` | str | 必填 | CSV(image,label) |
| `--images-dir` | str | 必填 | 图像目录 |
| `--model-dir` | str | 与 `--onnx-model` 二选一 | AutoGluon predictor 目录 |
| `--onnx-model` | str | 与 `--model-dir` 二选一 | ONNX 模型路径 |
| `--out-dir` | str | 必填 | 输出目录 |
| `--batch-size` | int | 32 | 推理 batch size |
| `--num-workers` | int | 4 | DataLoader worker 数 |
| `--num-threads` | int | 0 | CPU 计算线程数（0=框架自动） |
| `--device` | str | `auto` | `auto`/`cpu`/`cuda`/`mps` |

**评估指标**：accuracy, precision_macro/micro, recall_macro/micro, f1_macro/micro, mcc, roc_auc_ovo

**输出**：
```
out_dir/
└── logs/
    └── evaluations.txt
```

### 5.5 `classify embed`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--images-dir` | str | 必填 | 图像目录 |
| `--out-dir` | str | 必填 | 输出目录 |
| `--base-model` | str | `convnextv2_femto` | timm backbone（无 `--model-dir`/`--onnx-model` 时使用） |
| `--model-dir` | str | 可选 | AutoGluon predictor，使用 fine-tuned backbone 提取 |
| `--label-csv` | str | 可选 | CSV(image,label)，提供后计算有监督指标 + UMAP 着色 |
| `--visualize` | flag | False | 生成 UMAP 可视化图；若未提供 `--label-csv` 则报错退出 |
| `--umap-n-neighbors` | int | 15 | UMAP n_neighbors |
| `--umap-min-dist` | float | 0.1 | UMAP min_dist |
| `--umap-metric` | str | `euclidean` | UMAP 距离度量 |
| `--umap-seed` | int | 42 | UMAP 随机种子 |
| `--batch-size` | int | 32 | 提取 batch size |
| `--num-workers` | int | 4 | DataLoader worker 数 |
| `--num-threads` | int | 0 | CPU 计算线程数（0=框架自动） |
| `--device` | str | `auto` | `auto`/`cpu`/`cuda`/`mps` |
| `--metrics-sample-size` | int | 10000 | 计算指标时最大样本数（≤0 禁用采样） |

**嵌入质量指标**（有 `--label-csv` 时计算）：NMI, ARI, Recall@1/5/10, kNN_Acc_k1/5/20, Linear_Probing_Acc, mAP, Purity, Silhouette_Score

**输出**：
```
out_dir/
├── embeddings.csv
├── umap.pdf                   # 仅 --visualize 时生成
└── logs/
    ├── metrics.csv
    └── log.txt
```

### 5.6 `classify cam`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--label-csv` | str | 必填 | CSV(image,label) |
| `--images-dir` | str | 必填 | 图像目录 |
| `--out-dir` | str | 必填 | 输出目录 |
| `--model-dir` | str | 与 `--base-model` 二选一 | AutoGluon predictor 目录 |
| `--base-model` | str | 与 `--model-dir` 二选一 | timm backbone 名称 |
| `--checkpoint-path` | str | 可选 | 自定义 .pth 权重（配合 `--base-model` 使用） |
| `--num-classes` | int | 可选 | 覆盖分类数（timm backbone 时） |
| `--no-pretrained` | flag | False | 不加载 timm 预训练权重 |
| `--cam-method` | str | `gradcam` | `gradcam`/`gradcampp`/`layercam`/`scorecam`/`eigencam`/`ablationcam` |
| `--arch` | str | 自动推断 | `cnn`/`vit` |
| `--target-layer-name` | str | 可选 | 指定 CAM 目标层（点分隔路径） |
| `--image-weight` | float | 0.5 | 原图与 CAM 叠加权重（0~1） |
| `--fig-format` | str | `png` | `png`/`jpg`/`pdf` |
| `--save-npy` | flag | False | 保存 CAM 数组为 .npy |
| `--max-images` | int | 可选 | 限制处理图像数量 |
| `--cam-batch-size` | int | 32 | CAM 内部 batch size（ScoreCAM/EigenCAM） |
| `--num-workers` | int | 4 | DataLoader worker 数 |
| `--num-threads` | int | 0 | CPU 计算线程数（0=框架自动） |
| `--device` | str | `auto` | `auto`/`cpu`/`cuda`/`mps` |

> **注意**：`cam` 命令仅支持 PyTorch 原生模型（AutoGluon checkpoint 或 timm backbone），**不支持 ONNX**。GradCAM 依赖 PyTorch hook 和反向传播机制，ONNX runtime 不具备此能力。

**输出**：
```
out_dir/
├── figures/
│   └── {image_stem}_cam.{format}   # 原图与 CAM 叠加的并排图
├── arrays/                          # 仅 --save-npy 时生成
│   └── {image_stem}.npy             # 归一化 CAM 数组，float32
└── cam_summary.csv                  # 列：image, label, pred_class, pred_prob, figure_path, cam_array_path
```

### 5.7 `classify export-onnx`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-dir` | str | 必填 | AutoGluon predictor 目录 |
| `--out-dir` | str | 必填 | ONNX 输出目录 |
| `--opset` | int | 17 | ONNX opset 版本 |
| `--input-size` | int | 224 | 模型输入尺寸（正方形） |

**输出**：
```
out_dir/
└── model.onnx
```

---

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

classify 各命令（train/predict/evaluate/embed/cam）统一支持以下三个并发控制参数：

| 参数 | 作用层次 | 底层实现 |
|------|----------|----------|
| `--num-workers` | DataLoader 图像加载并发 | `DataLoader(num_workers=N)` |
| `--num-threads` | PyTorch CPU 计算线程 / ONNX 推理线程 | `torch.set_num_threads(N)` / `InferenceSession(intra_op_num_threads=N)` |
| `--device` | 计算设备选择 | `auto` 时自动检测 cuda→mps→cpu 优先级 |

默认值：`--num-workers=4`，`--num-threads=0`（0 表示交由框架自动决定）。

`export-onnx` 无需这三个参数（纯模型格式转换，无推理运算）。

---

## 8. 依赖管理

AutoGluon 和 pytorch-grad-cam 是重型依赖，通过 `setup.py` 的 `extras_require` 管理，不作为默认安装依赖：

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
        "pytorch-grad-cam",
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

- `scripts/` 目录中的原始脚本**暂时保留**，不立即删除，但不再是主入口
- `add_functions/` 目录**保留作为参考**，不作为入口
- 旧的 `setup.py` entry_points（`entomokit-segment` 等）在迁移完成后移除
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

## 10. 未来扩展预留

顶层命令组的设计允许未来增加新的功能组，例如：

```
entomokit detect    # 目标检测组（未来）
entomokit track     # 目标追踪组（未来）
```

每个新组只需在 `entomokit/main.py` 中注册，对现有代码零影响。
