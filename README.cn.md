# 昆虫图像数据集工具包 (EntomoKit)

**中文** | [English](README.md)

一个基于 Python 的昆虫图像数据集构建工具包。提供统一的 `entomokit` 命令行工具，支持视频抽帧、图像分割、图像合成、图像清洗、图像增强、数据集划分、AutoMM 图像分类以及环境诊断等功能。

## 概述

所有功能通过单一入口访问：

```
entomokit <command> [options]
```

| 命令 | 描述 |
|---------|-------------|
| `extract-frames` | 从视频文件中提取帧 |
| `segment` | 从图像中分割昆虫（SAM3、Otsu、GrabCut） |
| `synthesize` | 将昆虫合成到背景图像上 |
| `clean` | 清洗和去重图像 |
| `augment` | 使用预设或自定义 albumentations 策略进行图像增强 |
| `split-csv` | 将数据集划分为 train/val/test CSV 文件 |
| `classify train` | 训练 AutoMM 图像分类器 |
| `classify predict` | 运行推理（AutoMM 或 ONNX） |
| `classify evaluate` | 评估模型性能 |
| `classify embed` | 提取嵌入向量 + UMAP + 质量指标 |
| `classify cam` | 生成 GradCAM 热力图 |
| `classify export-onnx` | 导出模型为 ONNX 格式 |
| `doctor` | 诊断环境与缺失依赖 |

## 功能特性

- **统一命令行接口**：单一 `entomokit` 入口，无需逐脚本调用
- **多种分割方法**：SAM3（带 alpha 通道）、SAM3-bbox（裁剪）、Otsu 阈值、GrabCut
- **灵活的修复策略**：OpenCV 形态学操作、基于 SAM3 或 LaMa 的孔洞填充
- **标注输出**：COCO JSON、VOC Pascal XML、YOLO TXT
- **视频抽帧**：多线程提取，支持时间范围设定
- **图像清洗**：调整大小、去重（MD5/Phash）、规范化命名；支持递归模式
- **图像增强**：基于 albumentations 的预设/自定义增强，支持确定性随机种子
- **数据集划分**：基于比例或数量的 train/val/test 划分，支持分层采样
- **图像合成**：高级合成功能，支持旋转、颜色匹配、黑区规避
- **AutoMM 分类**：训练、预测、评估、嵌入、GradCAM、ONNX 导出
- **环境诊断**：`doctor` 命令输出依赖状态并给出安装/升级建议
- **嵌入质量指标**：NMI、ARI、Recall@K、kNN 准确率、mAP@R、轮廓系数、UMAP 可视化
- **并行处理**：多线程图像处理，可配置工作线程数
- **完整日志**：详细日志记录，支持详细模式和日志文件输出

## 系统要求

- Python 3.8+
- 操作系统：Linux、macOS、Windows

## 安装

推荐使用隔离的 Python 环境，避免与系统/全局 site-packages 发生依赖冲突。

### 部署模式 A（推荐）：隔离环境

可任选其一：

**选项 1：conda**

```bash
conda create -n entomokit python=3.11 -y
conda activate entomokit
pip install -e .
```

**选项 2：uv + venv**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

**选项 3：标准库 venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 基础安装

```bash
pip install -e .
```

### 部署模式 B（不推荐）：直接全局 pip

也可以直接安装到当前 Python 环境，但可能导致与其他项目发生依赖冲突：

```bash
pip install -e .
```

### 安装分类功能支持

用于分类命令（AutoMM、timm、GradCAM、UMAP）：

```bash
pip install -e ".[classify]"
```

AutoMM 官方安装参考：
https://auto.gluon.ai/stable/install.html

### 安装分割功能支持

用于基于 SAM3 的分割：

```bash
pip install -e ".[segmentation]"
```

### 安装视频处理支持

用于视频抽帧：

```bash
pip install -e ".[video]"
```

### 安装图像清洗支持

用于感知哈希去重：

```bash
pip install -e ".[cleaning]"
```

### 安装图像增强支持

用于 `entomokit augment`：

```bash
pip install -e ".[augment]"
```

### 开发环境安装

```bash
pip install -e ".[dev,classify,segmentation,video,cleaning,augment]"
```

## 项目结构

```
.
├── entomokit/              # 统一命令行包
│   ├── main.py             # 入口点调度器
│   ├── extract_frames.py   # entomokit extract-frames
│   ├── segment.py          # entomokit segment
│   ├── synthesize.py       # entomokit synthesize
│   ├── clean.py            # entomokit clean
│   ├── augment.py          # entomokit augment
│   ├── split_csv.py        # entomokit split-csv
│   ├── doctor.py           # entomokit doctor
│   ├── help_style.py       # Rich 帮助格式化
│   └── classify/           # entomokit classify *
│       ├── train.py
│       ├── predict.py
│       ├── evaluate.py
│       ├── embed.py
│       ├── cam.py
│       └── export_onnx.py
├── src/
│   ├── common/             # 共享工具（CLI、annotation_writer、logging、validators）
│   ├── classification/     # AutoMM 分类逻辑
│   ├── segmentation.py     # 分割领域逻辑
│   ├── framing/            # 视频帧提取领域逻辑
│   ├── cleaning/           # 图像清洗领域逻辑
│   ├── augment/            # 图像增强领域逻辑
│   ├── splitting/          # 数据集划分领域逻辑
│   ├── synthesis/          # 图像合成领域逻辑
│   ├── doctor/             # 环境诊断
│   ├── sam3/               # SAM3 模型实现
│   └── lama/               # LaMa 修复实现
├── tests/                  # 测试文件
├── data/                   # 数据目录（大文件已忽略）
├── models/                 # 模型权重（大文件已忽略）
├── docs/                   # 计划、规格、变更摘要
├── requirements.txt        # Python 依赖
└── setup.py                # 包配置
```

## 模型要求

### SAM3 模型

对于基于 SAM3 的方法（`sam3`、`sam3-bbox`），需要从 Hugging Face 下载检查点并通过 `--sam3-checkpoint` 指定。

下载链接：https://huggingface.co/facebook/sam3

### LaMa 模型

对于 `--repair-strategy lama`，需要将 Big-LaMa 模型放置在：
```
models/big-lama/
├── config.yaml
└── models/best.ckpt
```

下载链接：https://github.com/advimman/lama

### AutoMM / timm（classify 命令）

安装 `classify` 扩展 — AutoMM 会在首次使用时自动下载骨干网络权重。

支持的 timm 骨干网络包括：
- `convnextv2_femto`（默认，轻量级）
- `convnextv2_tiny`、`convnextv2_small`、`convnextv2_base`
- `resnet18`、`resnet50`、`resnet101`
- `efficientnet_b0` 到 `efficientnet_b7`
- `vit_small_patch16_224`、`vit_base_patch16_224`
- 更多模型见 [timm models](https://huggingface.co/timm)

## 使用方法

推荐的工作流命令顺序：

1. `extract-frames`
2. `segment`
3. `synthesize`
4. `clean`
5. `augment`
6. `split-csv`
7. `classify`

### segment 命令

使用多种方法（SAM3、Otsu、GrabCut）从图像中分割昆虫。可选择生成 COCO、VOC 或 YOLO 格式的标注。

#### 基本用法

```bash
# SAM3 带 alpha 通道（透明背景）
entomokit segment \
    --input-dir images/clean_insects/ \
    --out-dir outputs/insects_clean/ \
    --sam3-checkpoint models/sam3.pt \
    --segmentation-method sam3 \
    --device auto

# 生成 COCO 标注
entomokit segment \
    --input-dir images/clean_insects/ \
    --out-dir outputs/insects_clean/ \
    --sam3-checkpoint models/sam3.pt \
    --segmentation-method sam3 \
    --annotation-format coco

# 生成 YOLO 标注和 xyxy 边界框格式
entomokit segment \
    --input-dir images/ --out-dir outputs/ \
    --segmentation-method otsu \
    --annotation-format yolo \
    --coco-bbox-format xyxy

# SAM3-bbox 模式（裁剪到边界框）
entomokit segment \
    --input-dir images/ --out-dir outputs/ \
    --sam3-checkpoint models/sam3.pt \
    --segmentation-method sam3-bbox \
    --padding-ratio 0.1

# 使用 LaMa 修复填充孔洞
entomokit segment \
    --input-dir images/ --out-dir outputs/ \
    --sam3-checkpoint models/sam3.pt \
    --repair-strategy lama \
    --lama-model models/big-lama/
```

#### 主要参数

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--input-dir` | 输入目录 | 必填 |
| `--out-dir` | 输出目录 | 必填 |
| `--segmentation-method` | `sam3`、`sam3-bbox`、`otsu`、`grabcut` | `sam3` |
| `--sam3-checkpoint` | SAM3 检查点路径 | sam3/sam3-bbox 必填 |
| `--hint` | SAM3 文本提示 | `insect` |
| `--device` | `auto`、`cpu`、`cuda`、`mps` | `auto` |
| `--confidence-threshold` | 掩码最小置信度 | `0.0` |
| `--padding-ratio` | 边界框填充比例 | `0.0` |
| `--repair-strategy` | `opencv`、`sam3-fill`、`black-mask`、`lama` | 无 |
| `--lama-model` | LaMa 模型目录 | 无 |
| `--annotation-format` | `coco`、`voc`、`yolo` | 无 |
| `--coco-bbox-format` | `xywh`、`xyxy` | `xywh` |
| `--threads` | 并行工作线程数 | 8 |

**输出结构（COCO 示例）：**
```
output_dir/
├── annotations.coco.json     # COCO 标注
├── cleaned_images/           # 分割后的图像
│   ├── image_01.png
│   └── ...
└── repaired_images/          # （启用 repair-strategy 时）
```

**YOLO/VOC 布局：**
```
output_dir/
├── images/
├── labels/                   # YOLO：每张图一个 .txt + data.yaml
└── Annotations/              # VOC：每张图一个 .xml + ImageSets/Main/
```

---

### extract-frames 命令

从视频文件中提取帧。支持目录或单个视频文件路径。

```bash
# 从目录提取，每秒一帧
entomokit extract-frames --input-dir videos/ --out-dir frames/

# 从单个视频提取，时间范围 5s–30s
entomokit extract-frames --input-dir video.mp4 --out-dir frames/ \
    --start-time 5.0 --end-time 30.0

# 自定义间隔和格式
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --interval 500 --out-image-format png

# 限制每个视频的最大帧数
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --max-frames 100
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--input-dir` | 视频目录或单个视频文件 | 必填 |
| `--out-dir` | 输出目录 | 必填 |
| `--interval` | 间隔（毫秒） | 1000 |
| `--start-time` | 开始时间（秒） | 0 |
| `--end-time` | 结束时间（秒） | 视频结束 |
| `--out-image-format` | jpg/png/tif | jpg |
| `--threads` | 并行线程数 | 8 |
| `--max-frames` | 每个视频最大帧数 | 全部 |

**支持的视频格式**：mp4、mov、avi、mkv、webm、flv、m4v、mpeg、mpg、wmv、3gp、ts

---

### clean 命令

清洗和去重图像，规范化命名。

```bash
# 基本用法（MD5 去重）
entomokit clean --input-dir images/raw/ --out-dir images/cleaned/

# 递归扫描 + 感知哈希去重
entomokit clean --input-dir images/ --out-dir cleaned/ \
    --recursive --dedup-mode phash --phash-threshold 5

# 调整短边为 512px
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size 512 --out-image-format png

# 保持原始尺寸和 EXIF 数据
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size -1 --keep-exif
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--input-dir` | 输入目录 | 必填 |
| `--out-dir` | 输出目录 | 必填 |
| `--recursive` | 扫描子目录 | 否 |
| `--out-short-size` | 短边大小（-1 = 原始） | 512 |
| `--dedup-mode` | `none`、`md5`、`phash` | md5 |
| `--phash-threshold` | Phash 相似度阈值 | 5 |
| `--out-image-format` | jpg/png/tif | jpg |
| `--keep-exif` | 保留 EXIF 元数据 | 否 |
| `--threads` | 并行线程数 | 12 |

---

### augment 命令

使用 albumentations 预设或自定义策略对图像进行增强。

```bash
# 默认 light 预设，每张输入图生成 1 张增强图
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/

# heavy 预设，每张输入图生成 3 张增强图
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/ \
    --preset heavy --multiply 3 --seed 123

# 使用自定义策略 JSON
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/ \
    --policy configs/augment_policy.json
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--input-dir` | 输入图像目录 | 必填 |
| `--out-dir` | 输出目录 | 必填 |
| `--preset` | `light`、`medium`、`heavy`、`safe-for-small-dataset` | `light` |
| `--policy` | 自定义策略 JSON 路径（与 `--preset` 互斥） | 无 |
| `--seed` | 随机种子 | 42 |
| `--multiply` | 每张输入图生成的增强副本数量 | 1 |

**输出：**
```
output_dir/
├── images/
└── augment_manifest.json
```

---

### split-csv 命令

将标注 CSV 划分为 train / val / test 文件。

```bash
# 比例划分（80/10/10）
entomokit split-csv --raw-image-csv data/images.csv \
    --known-test-classes-ratio 0.1 --val-ratio 0.1 --out-dir datasets/

# 数量划分并复制图像
entomokit split-csv --raw-image-csv data/images.csv --mode count \
    --known-test-classes-count 100 --val-count 50 \
    --copy-images --images-dir images/ --out-dir datasets/

# 带未知类别测试划分（用于开放集评估）
entomokit split-csv --raw-image-csv data/images.csv \
    --unknown-test-classes-ratio 0.1 \
    --known-test-classes-ratio 0.1 \
    --out-dir datasets/

# 过滤样本过少的类别
entomokit split-csv --raw-image-csv data/images.csv \
    --min-count-per-class 10 \
    --out-dir datasets/
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--raw-image-csv` | 输入 CSV（image、label 列） | 必填 |
| `--out-dir` | 输出目录 | 必填 |
| `--mode` | `ratio` 或 `count` | ratio |
| `--val-ratio` / `--val-count` | 验证集划分 | 无 |
| `--known-test-classes-ratio` | 已知类别测试比例 | 0.1 |
| `--unknown-test-classes-ratio` | 未知类别测试比例 | 0 |
| `--min-count-per-class` | 删除少于该数量的类别 | 0 |
| `--max-count-per-class` | 每个类别的最大图像数 | 无 |
| `--copy-images` | 复制图像到划分子目录 | 否 |
| `--images-dir` | 源图像目录（用于复制） | 无 |
| `--seed` | 随机种子 | 42 |

**输出：**
```
output_dir/
├── train.csv
├── val.csv          # 指定 --val-ratio / --val-count 时
├── test.known.csv
├── test.unknown.csv # 配置未知类别时
├── class_count/     # 各划分的类别统计
│   ├── class.train.count
│   ├── class.val.count
│   └── ...
└── images/          # 使用 --copy-images 时
    ├── train/
    ├── val/
    └── test_known/
```

---

### synthesize 命令

将目标对象合成到背景图像上，支持旋转、颜色匹配和智能定位。

```bash
# 基本合成
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --num-syntheses 10

# 带 COCO 标注和旋转
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --num-syntheses 10 \
    --annotation-output-format coco \
    --rotate 30

# 带 YOLO 标注
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --annotation-output-format yolo \
    --coco-bbox-format xyxy

# 避开背景中的黑色区域
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --avoid-black-regions \
    --color-match-strength 0.7
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--target-dir` | 目标图像（带 alpha 通道） | 必填 |
| `--background-dir` | 背景图像 | 必填 |
| `--out-dir` | 输出目录 | 必填 |
| `--num-syntheses` | 每个目标的合成次数 | 10 |
| `--annotation-output-format` | `coco`、`voc`、`yolo` | `coco` |
| `--coco-bbox-format` | `xywh`、`xyxy` | `xywh` |
| `--rotate` | 最大旋转角度 | 0 |
| `--avoid-black-regions` | 避开暗色背景区域 | 否 |
| `--color-match-strength` | 0–1 颜色匹配强度 | 0.5 |
| `--area-ratio-min` | 最小目标/背景面积比 | 0.05 |
| `--area-ratio-max` | 最大目标/背景面积比 | 0.20 |
| `--threads` | 并行工作线程数 | 4 |

**输出（COCO）：**
```
output_dir/
├── images/
│   ├── target_01.png
│   └── ...
└── annotations.coco.json
```

**输出（YOLO）：**
```
output_dir/
├── images/
├── labels/
└── data.yaml
```

---

### classify 命令组

所有分类命令需要安装 `classify` 扩展：

```bash
pip install -e ".[classify]"
```

#### `classify train`

使用 AutoMM MultiModalPredictor 训练图像分类器。

```bash
entomokit classify train \
    --train-csv data/train.csv \
    --images-dir data/images/ \
    --out-dir runs/exp1/ \
    --base-model convnextv2_femto \
    --augment medium \
    --max-epochs 50 \
    --learning-rate 3e-4 \
    --device auto
```

**恢复训练**（将 epoch 限制从 50 扩展到 100）：

```bash
entomokit classify train \
    --train-csv data/train.csv \
    --images-dir data/images/ \
    --out-dir runs/exp1/ \
    --base-model convnextv2_femto \
    --max-epochs 100 \
    --resume
```

**自定义数据增强**：

```bash
# 使用预设
entomokit classify train ... --augment heavy

# 使用自定义变换（JSON 数组）
entomokit classify train ... --augment '["random_resize_crop","color_jitter","randaug"]'
```

**使用 Focal Loss**（适用于类别不平衡）：

```bash
entomokit classify train \
    --train-csv data/train.csv \
    --images-dir data/images/ \
    --out-dir runs/exp1/ \
    --focal-loss \
    --focal-loss-gamma 2.0
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--train-csv` | 含 `image` 和 `label` 列的 CSV | 必填 |
| `--images-dir` | 训练图像目录 | 必填 |
| `--out-dir` | 输出目录 | 必填 |
| `--base-model` | timm 骨干网络名称 | `convnextv2_femto` |
| `--augment` | 预设或 JSON 数组 | `medium` |
| `--max-epochs` | 最大训练轮数 | 50 |
| `--time-limit` | 时间限制（小时） | 1.0 |
| `--resume` | 从检查点继续 | 否 |
| `--learning-rate` | AutoMM `optim.lr` | `1e-4` |
| `--weight-decay` | AutoMM `optim.weight_decay` | `1e-3` |
| `--warmup-steps` | AutoMM `optim.warmup_steps` | `0.1` |
| `--patience` | 早停耐心值 | 10 |
| `--top-k` | 检查点平均数量 | 3 |
| `--focal-loss` | 启用 focal loss | 否 |
| `--device` | `auto/cpu/cuda/mps` | `auto` |
| `--batch-size` | 批量大小 | 32 |
| `--num-workers` | DataLoader 工作线程数 | 4 |

**数据增强预设**：
| 预设 | 变换 |
|--------|-----------|
| `none` | resize_shorter_side, center_crop |
| `light` | none + random_horizontal_flip |
| `medium` | light + color_jitter + trivial_augment |
| `heavy` | random_resize_crop, random_horizontal_flip, random_vertical_flip, color_jitter, trivial_augment, randaug |

---

#### `classify predict`

使用 AutoMM 或 ONNX 模型对图像进行推理。

```bash
# AutoMM 模型
entomokit classify predict \
    --images-dir data/test/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/predict/

# ONNX 模型
entomokit classify predict \
    --input-csv test.csv \
    --onnx-model runs/onnx/model.onnx \
    --out-dir runs/predict/

# CSV 图像名称 + 图像根目录
entomokit classify predict \
    --input-csv out/split/test.known.csv \
    --images-dir data/Epidorcus/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/predict/
```

**输入解析规则**：
- 至少提供 `--input-csv` 或 `--images-dir` 之一
- 如果 CSV 的 `image` 值已经是可读路径，则直接使用 CSV
- 如果 CSV 的 `image` 值是文件名/相对路径，还需提供 `--images-dir`
- 如果只提供 `--images-dir`，则预测该目录下的所有图像

**ONNX 要求**：
```bash
pip install onnxruntime
# 或
pip install 'entomokit[classify]'
```

**ONNX 输出**：
- 当 ONNX 文件旁存在 `label_classes.json` 时，`prediction` 为类别名称
- `prediction_index` 始终存储数字类别索引

---

#### `classify evaluate`

在测试集上评估模型性能。

```bash
entomokit classify evaluate \
    --test-csv data/test.csv \
    --images-dir data/images/ \
    --onnx-model runs/onnx/model.onnx \
    --out-dir runs/eval/
```

**输出指标**（保存到 `evaluations.csv`）：
- Accuracy、Balanced Accuracy
- Precision/Recall/F1（macro、micro、weighted）
- Matthews 相关系数（MCC）
- Quadratic Kappa
- ROC-AUC（OVO、OVR）

---

#### `classify embed`

提取嵌入向量并计算质量指标。

```bash
# 预训练 timm 骨干网络（无需训练）
entomokit classify embed \
    --images-dir data/images/ \
    --base-model convnextv2_femto \
    --label-csv data/labels.csv \
    --visualize \
    --out-dir runs/embed/

# 微调后的 AutoMM 骨干网络
entomokit classify embed \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --label-csv data/labels.csv \
    --out-dir runs/embed/
```

**输出**：
- `embeddings.csv` — 特征向量（feat_0, feat_1, ...）
- `metrics.csv` — 质量指标
- `umap.pdf` — UMAP 可视化（使用 `--visualize`）

**质量指标**：
| 指标 | 描述 |
|--------|-------------|
| NMI | 标准化互信息 |
| ARI | 调整兰德指数 |
| Recall@1/5/10 | K 值检索召回率 |
| kNN_Acc_k1/5/20 | k-NN 分类准确率 |
| Linear_Probing_Acc | 线性分类器准确率 |
| mAP@R | R 处平均精度均值 |
| Purity | 聚类纯度 |
| Silhouette_Score | 聚类质量 |

---

#### `classify cam`

生成 GradCAM 热力图用于模型可解释性。

```bash
entomokit classify cam \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --cam-method gradcam \
    --out-dir runs/cam/ \
    --save-npy
```

**带真实标签**：
```bash
entomokit classify cam \
    --label-csv data/test.csv \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/cam/
```

**CAM 方法**：`gradcam`、`gradcampp`、`layercam`、`scorecam`、`eigencam`、`ablationcam`

**架构自动检测**：自动检测 CNN 或 ViT 架构。

**输出**：
- `figures/` — CAM 叠加图像
- `cam_summary.csv` — 元数据
- `arrays/` — 原始 CAM 数组（使用 `--save-npy`）

**查找目标层**：
```bash
entomokit classify cam \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --dump-model-structure \
    --out-dir runs/cam/
# 然后查看 runs/cam/model_layers.txt
```

**注意**：不支持 ONNX 模型（需要 PyTorch hooks）。

---

#### `classify export-onnx`

将 AutoMM 模型导出为 ONNX 格式用于部署。

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ \
    --opset 17
```

**使用示例图像进行追踪**：
```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ \
    --sample-image data/sample.jpg
```

**输出**：
- `model.onnx` — ONNX 模型文件
- `label_classes.json` — 类别标签映射

---

### doctor 命令

诊断环境与依赖是否满足当前功能需求。

```bash
entomokit doctor
```

报告内容包括：
- Python 版本与可用设备（`cpu`、`cuda`、`mps`）
- 关键依赖的版本与状态（ok/missing/outdated）
- 安装/升级建议（包含 `autogluon.multimodal>=1.4.0`）

---

## 通用行为

### 日志

所有命令会在输出目录保存 `log.txt`，包含：
- 完整命令行
- 时间戳
- 所有参数值
- 运行时输出

使用 `--verbose` 获取调试级别输出。

### 优雅退出

按 `Ctrl+C` — 当前图像处理完成后退出，保存部分结果。

### 设备选择

`--device auto` 自动选择：
1. CUDA（如果可用）
2. MPS / Apple Silicon（如果可用）
3. CPU（回退）

### Shell 补全

安装 entomokit 的 shell 补全：

```bash
entomokit --install-completion
```

支持的 shell：bash、zsh、fish

### 版本号

查看已安装版本：

```bash
entomokit --version
entomokit -v
```

---

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 联系方式

- 邮箱：`xtmtd.zf@gmail.com`

## 引用

如果您在研究中使用 EntomoKit，请引用：

```bibtex
@software{entomokit2026,
  author = {Zhang, Feng},
  title = {EntomoKit: A Python Toolkit for Insect Image Dataset Construction and Classification},
  year = {2026},
  url = {https://github.com/xtmtd/entomokit}
}
```
