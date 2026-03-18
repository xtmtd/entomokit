# 昆虫数据集工具包

基于 Python 的工具包，用于构建昆虫图像数据集，支持多种处理功能：图像分割、视频抽帧、图像清理、数据集划分和图像合成。

## 概述

本项目提供五个主要脚本：
- `segment.py` - 使用多种分割方法（SAM3、Otsu、GrabCut）从干净背景图像中分割昆虫
- `extract_frames.py` - 从视频文件中提取帧用于数据收集
- `clean_figs.py` - 清理和去重图像，统一命名格式
- `split_dataset.py` - 将数据集拆分为训练集/测试集/未知类（用于机器学习训练）
- `synthesize.py` - 将目标物体合成到背景图像上，支持高级定位

## 功能特性

- **多种分割方法**：SAM3（带透明通道）、SAM3-bbox（裁剪框）、Otsu 阈值分割、GrabCut
- **灵活的修复策略**：OpenCV 形态学操作或基于 SAM3 的孔洞填充
- **视频帧提取**：支持多种视频格式的多线程提取，带进度跟踪
- **图像清理**：调整大小、去重（MD5/感知哈希）、标准化图像命名
- **数据集划分**：基于比例或数量的训练集/测试集/未知类划分，支持分层抽样
- **图像合成**：先进的合成技术，支持旋转、颜色匹配、黑色区域避让
- **输入验证**：验证输入目录、图像文件和参数约束
- **优雅关闭**：处理 Ctrl+C 信号，完成当前任务后退出
- **并行处理**：多线程图像处理，可配置工作线程数
- **进度跟踪**：合成脚本支持 tqdm 进度条
- **全面日志**：详细日志记录，支持详细模式和日志文件输出

## 安装

```bash
pip install -r requirements.txt
```

或作为包安装：

```bash
pip install -e .
```

## 项目结构

```
.
├── scripts/              # CLI 脚本
│   ├── segment.py        # 分割脚本
│   ├── extract_frames.py # 视频帧提取
│   ├── clean_figs.py     # 图像清理和去重
│   ├── split_dataset.py  # 数据集划分
│   └── synthesize.py     # 图像合成脚本
├── src/
│   ├── common/          # 共用工具（CLI、日志、验证）
│   ├── segmentation/    # 分割领域逻辑
│   ├── framing/         # 视频帧领域逻辑
│   ├── cleaning/        # 图像清理领域逻辑
│   ├── splitting/       # 数据集划分领域逻辑
│   └── synthesis/       # 图像合成领域逻辑
│   └── lama/            # LaMa inpainting 实现
├── tests/               # 测试文件
├── data/                # 数据目录（大文件已忽略）
├── models/              # 模型权重（大文件已忽略）
├── outputs/             # 输出文件（已忽略）
├── requirements.txt     # Python 依赖
├── setup.py             # 包安装配置
├── CHANGES_SUMMARY.md   # 变更摘要
└── README.md            # 本文件
```

## 使用说明

### 1. 分割脚本 (segment.py)

使用多种方法（SAM3、Otsu、GrabCut）从图像中分割昆虫。可选生成 COCO、VOC 或 YOLO 格式的标注文件。

**标注输出**：指定 `--annotation-output-format` 时，会生成包含目标物体元数据（bbox、segmentation、area）的标注文件，坐标为原始图片坐标。

#### 基本用法

```bash
# SAM3 透明背景（带 alpha 通道）
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --device auto     --hint "insect"

# 带 COCO 标注输出（统一模式，默认）
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --annotation-output-format coco     --coco-output-mode unified

# 带 VOC Pascal 标注输出
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --annotation-output-format voc

# 带 YOLO 标注输出
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --annotation-output-format yolo
```

#### 全部参数

**必需参数：**
- `--input_dir`, `-i`：包含输入图像的目录
- `--out_dir`, `-o`：输出分割图像的目录

**可选参数：**
- `--hint`, `-t`：分割文本提示（默认："insect"）
- `--segmentation-method`：分割方法
  - `sam3` - SAM3 分割带 alpha 通道（透明背景）
  - `sam3-bbox` - SAM3 分割带裁剪边界框（无 alpha）
  - `otsu` - Otsu 阈值分割（不需要 SAM3 检查点）
  - `grabcut` - GrabCut 算法（不需要 SAM3 检查点）
- `--sam3-checkpoint`, `-c`：SAM3 检查点文件路径（sam3/sam3-bbox 方法必需）
- `--device`, `-d`：推理设备（`auto`、`cpu`、`cuda`、`mps`）
- `--confidence-threshold`：掩码置信度阈值（0.0-1.0，默认：0.0）
- `--repair-strategy`, `-r`：孔洞填充修复策略
  - `opencv` - OpenCV 形态学操作
  - `sam3-fill` - 基于 SAM3 的孔洞填充
  - `lama` - 基于 LaMa 的孔洞填充
- `--padding-ratio`：边界框填充比例（0.0-0.5，默认：0.0）
- `--out-image-format`, `-f`：输出图像格式（`png`、`jpg`）
- `--threads`, `-n`：并行工作线程数（默认：8）
- `--verbose`, `-v`：启用详细日志
- `--annotation-output-format`：标注输出格式（`coco`、`voc`、`yolo`）
- `--coco-output-mode`：COCO 输出模式（`unified`、`separate`）
  - `unified` - 单个 `annotations.json` 文件
  - `separate` - 每张图片单独的 JSON 文件

#### 使用示例

**1. SAM3 透明背景**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --device auto     --hint "insect"     --threads 12
```

**2. SAM3 边界框模式（无透明通道）**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_bbox/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3-bbox     --out-image-format jpg
```

**3. Otsu 阈值分割（不需要 SAM3 检查点）**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_otsu/     --segmentation-method otsu     --out-image-format jpg
```

**4. GrabCut 算法（不需要 SAM3 检查点）**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_grabcut/     --segmentation-method grabcut     --repair-strategy sam3-fill
```

**5. 带修复策略和填充参数**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_repaired/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --repair-strategy opencv     --padding-ratio 0.1     --confidence-threshold 0.5     --verbose
```

**6. 带 COCO 标注输出（统一模式）**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --annotation-output-format coco     --coco-output-mode unified
```

**带标注的输出目录结构：**
```
output_dir/
├── cleaned_images/           # 分割后的输出图像
│   ├── image_01.png
│   └── image_02.png
├── annotations/              # COCO/VOC 标注文件
│   └── annotations.json      # （统一 COCO 模式）
│   ├── image_01.json         # （分离 COCO 模式）
│   └── image_01.xml          # （VOC 模式）
└── labels/                   # YOLO 标注文件
    ├── image_01.txt
    └── ...
```

**标注格式说明：**
- **bbox**：原始图片坐标 `[x, y, w, h]`
- **segmentation**：原始图片坐标 `[[x1, y1, x2, y2, ...]]`
- **area**：边界框面积（宽 × 高）
- **mask_area**：实际掩码像素数
- **file_name**：指向清理后的输出图像（例如 `cleaned_images/image.png`）

### 2. 视频抽帧脚本 (extract_frames.py)

支持多线程从视频文件中提取帧。

#### 基本用法

```bash
# 每隔 1 秒提取一帧（默认）
python scripts/extract_frames.py     --input_dir videos/     --out_dir frames/

# 自定义间隔（每隔 500 毫秒）
python scripts/extract_frames.py     --input_dir videos/     --out_dir frames/     --interval 500

# 限制每视频的最大帧数
python scripts/extract_frames.py     --input_dir videos/     --out_dir frames/     --max_frames 100

# 自定义输出格式（PNG）
python scripts/extract_frames.py     --input_dir videos/     --out_dir frames/     --out_image_format png

# 提取指定时间范围的帧（5s 到 30s）
python scripts/extract_frames.py \
    --input_dir videos/ \
    --out_dir frames/ \
    --start_time 5.0 \
    --end_time 30.0

# 从 10 秒开始提取到视频结束
python scripts/extract_frames.py \
    --input_dir videos/ \
    --out_dir frames/ \
    --start_time 10.0

# 调整线程数
python scripts/extract_frames.py \
    --input_dir videos/ \
    --out_dir frames/ \
    --threads 4
```

**参数说明：**

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--input_dir` | 包含视频文件的输入目录 | 必需 |
| `--out_dir` | 提取图像的输出目录 | 必需 |
| `--interval` | 提取间隔（毫秒） | 1000 (1秒) |
| `--start_time` | 提取起始时间（秒） | 0 |
| `--end_time` | 提取结束时间（秒） | 视频结束 |
| `--out_image_format` | 输出格式 (jpg/png/tif/pdf) | jpg |
| `--threads` | 并行线程数 | 8 |
| `--max_frames` | 每视频最大帧数 | 全部 |

**输出结构：**

```
output_dir/
└── video_name/
    ├── video_name_01.jpg
    ├── video_name_02.jpg
    └── ...
```

### 3. 图像清理脚本 (clean_figs.py)

清理和去重图像，统一命名和格式。

#### 基本用法

```bash
# 基本清理，MD5 去重
python scripts/clean_figs.py     --input_dir images/raw/     --out_dir images/cleaned/

# 缩放到短边 512 像素
python scripts/clean_figs.py     --input_dir images/raw/     --out_dir images/cleaned/     --out_short_size 512

# 使用感知哈希（phash）进行去重
python scripts/clean_figs.py     --input_dir images/raw/     --out_dir images/cleaned/     --dedup_mode phash     --phash_threshold 5

# 转换为 PNG 格式
python scripts/clean_figs.py     --input_dir images/raw/     --out_dir images/cleaned/     --out_image_format png     --threads 16
```

**参数说明：**

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--input_dir` | 包含输入图像的目录 | 必需 |
| `--out_dir` | 清理后图像的输出目录 | 必需 |
| `--out_short_size` | 输出图像短边尺寸（-1 为原始尺寸） | 512 |
| `--out_image_format` | 输出格式 (jpg/png/tif/pdf) | jpg |
| `--threads` | 并行线程数 | 12 |
| `--keep_exif` | 保留 EXIF 数据 | 否 |
| `--dedup_mode` | 去重模式 (none/md5/phash) | md5 |
| `--phash_threshold` | Phash 相似度阈值 | 5 |

### 4. 数据集划分脚本 (split_dataset.py)

将数据集拆分为训练集/测试集/未知类，用于机器学习。

#### 基本用法

```bash
# 基于比例的划分（默认）
python scripts/split_dataset.py     --raw_image_csv data/images.csv     --mode ratio     --out_dir datasets/

# 包含未知测试类（开放集）
python scripts/split_dataset.py     --raw_image_csv data/images.csv     --mode ratio     --unknown_test_classes_ratio 0.1     --known_test_classes_ratio 0.1     --out_dir datasets/

# 基于数量的划分
python scripts/split_dataset.py     --raw_image_csv data/images.csv     --mode count     --unknown_test_classes_count 50     --known_test_classes_count 100     --min_count_per_class 10     --max_count_per_class 100     --out_dir datasets/

# 自定义随机种子
python scripts/split_dataset.py     --raw_image_csv data/images.csv     --mode ratio     --seed 42     --out_dir datasets/
```

**参数说明：**

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--raw_image_csv` | 包含图像和标签列的原始 CSV 文件路径 | 必需 |
| `--mode` | 划分模式：ratio 或 count | ratio |
| `--unknown_test_classes_ratio` | 未知测试类的比例 | 0 |
| `--known_test_classes_ratio` | 已知测试类的比例 | 0.1 |
| `--unknown_test_classes_count` | 未知测试类的数量 | 0 |
| `--known_test_classes_count` | 已知测试类的数量 | 0 |
| `--min_count_per_class` | 训练集每类的最小样本数 | 0 |
| `--max_count_per_class` | 训练集每类的最大样本数 | 无限制 |
| `--seed` | 随机种子（用于可重复性） | 42 |
| `--out_dir` | 输出目录 | datasets |

**输出结构：**

```
output_dir/
├── train.csv              # 训练样本
├── test.known.csv         # 已知类别测试样本
├── test.unknown.csv       # 未知类别测试样本（如果配置）
└── class_count/
    ├── class.count        # 总类别统计
    ├── class.train.count  # 训练集类别数量
    ├── class.test.known.count
    └── class.test.unknown.count
```

### 5. 图像合成脚本 (synthesize.py)

将目标物体合成到背景图像上，支持旋转、颜色匹配、智能定位和标注生成。可选生成 COCO、VOC 或 YOLO 格式的标注文件。

**标注输出**：指定 `--annotation-output-format` 时，会生成包含目标物体元数据（bbox、segmentation、area、scale_ratio、rotation_angle）的标注文件，坐标为合成图片坐标。

#### 基本用法

```bash
# 基本合成，每个目标生成 10 个变体
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10

# 带 COCO 标注输出（统一模式，默认）
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10     --annotation-output-format coco     --coco-output-mode unified

# 带 COCO 分离模式（每张图片单独文件）
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10     --annotation-output-format coco     --coco-output-mode separate

# 带 VOC Pascal 标注输出
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10     --annotation-output-format voc

# 带 YOLO 标注输出
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10     --annotation-output-format yolo

# 避免黑色区域
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --avoid-black-regions

# 带旋转和平行处理
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --rotate 30     --threads 4
```

**参数说明：**

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--target-dir`, `-t` | 目标对象图像目录（已清理，带 alpha 通道） | 必需 |
| `--background-dir`, `-b` | 背景图像目录 | 必需 |
| `--out-dir`, `-o` | 输出目录 | 必需 |
| `--num-syntheses`, `-n` | 每个目标图像的合成数量 | 10 |
| `--area-ratio-min`, `-a` | 最小面积比率（目标/背景面积） | 0.05 |
| `--area-ratio-max`, `-x` | 最大面积比率（目标/背景面积） | 0.20 |
| `--color-match-strength`, `-c` | 颜色匹配强度（0-1） | 0.5 |
| `--avoid-black-regions`, `-A` | 避免背景中的纯黑色区域 | 否 |
| `--rotate`, `-r` | 最大随机旋转角度 | 0 |
| `--out-image-format`, `-f` | 输出图像格式 (png/jpg) | png |
| `--threads`, `-d` | 并行工作线程数 | 4 |
| `--verbose`, `-v` | 启用详细日志 | 否 |
| `--annotation-output-format` | 标注输出格式（`coco`、`voc`、`yolo`） | coco |
| `--coco-output-mode` | COCO 输出模式（`unified`、`separate`） | unified |

**功能特性：**
- **旋转支持**：在指定角度内随机旋转
- **黑色区域避让**：自动将目标定位到暗色区域之外
- **颜色匹配**：使用 LAB 直方图匹配将目标颜色与背景匹配
- **自动缩放**：自动缩放目标使其适应背景
- **并行处理**：多线程合成，带进度条
- **多种标注格式**：COCO JSON、VOC XML、YOLO TXT

**输出结构：**

不带标注：
```
output_dir/
└── images/
    ├── target_name_01.png
    ├── target_name_02.png
    └── ...
```

带 COCO 标注（`--annotation-output-format coco`）：
```
output_dir/
├── images/
│   ├── target_name_01.png
│   ├── target_name_02.png
│   └── ...
├── annotations/              # COCO/VOC 标注文件
│   └── annotations.json      # （统一模式）
│   ├── target_name_01.json   # （分离模式）
│   └── target_name_01.xml    # （VOC 模式）
└── labels/                   # YOLO 标注文件
    ├── target_name_01.txt
    └── ...
```

**标注格式说明：**
- **bbox**：合成图片坐标 `[x, y, w, h]`
- **segmentation**：合成图片坐标 `[[x1, y1, x2, y2, ...]]`
- **area**：边界框面积（宽 × 高）
- **mask_area**：实际掩码像素数
- **scale_ratio**：使用的目标/背景面积比率
- **rotation_angle**：旋转角度（度）
- **file_name**：指向合成图像（例如 `images/target_name_01.png`）

## 输入验证和错误处理

所有脚本都包含全面的输入验证：

**通用验证：**
- **目录验证**：检查输入目录是否存在并包含预期文件
- **参数验证**：验证数值范围、格式选择和必需参数
- **文件格式验证**：支持标准图像和视频格式

**脚本特定验证：**
- **segment.py**：验证 SAM3 检查点是否存在（如需要），图像格式兼容性
- **clean_figs.py**：验证图像文件可打开，优雅处理损坏文件
- **extract_frames.py**：验证视频文件可被 OpenCV 打开
- **split_dataset.py**：验证 CSV 结构（存在图像和标签列）
- **synthesize.py**：验证目标图像（带 alpha 通道）、背景图像和输出目录；验证 COCO 输出目录创建

**错误处理：**
- 详细的错误消息和日志
- Ctrl+C 优雅关闭
- 中断时保存部分结果

## 优雅关闭

按任意 `Ctrl+C` 可触发优雅关闭：
- 完成当前图像处理
- 保存结果
- 清理退出并显示状态消息

## 进度跟踪

所有脚本支持进度跟踪：
- **合成脚本**：显示合成操作进度条（需要 `tqdm`）
- 进度条在无头环境中自动禁用
- 可使用特定脚本标志显式禁用

## 日志

所有脚本将日志保存到输出目录的 `log.txt`：
- 使用的命令和时间戳
- 所有参数值
- 处理进度和结果
- 错误和警告

使用 `--verbose` 启用详细模式以获取详细的调试信息。

## 模型要求

### SAM3 模型

对于 SAM3 相关方法（`sam3`、`sam3-bbox`），需要：
- SAM3 检查点文件（例如 `sam3_hq_vit_h.pt` 或 `sam3.pt`）
- 从 [Hugging Face](https://huggingface.co/facebook/sam3) 下载

### LaMa 模型

对于 LaMa 基于修复的合成（`--repair-strategy lama`），需要：
- LaMa Big-Lama 模型检查点（例如 `models/big-lama/models/best.ckpt`）
- 配置文件（例如 `models/big-lama/config.yaml`）

#### 下载 LaMa 模型

LaMa 模型可以从官方 Google Drive 文件夹下载：

**方法 1：从 Google Drive 下载（推荐）**

1. 访问 [LaMa Big-Lama 模型 - Google Drive](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=sharing)
2. 下载 **big-lama** 模型文件
3. 解压并放到正确位置：
   ```
   models/big-lama/
   ├── config.yaml          # LaMa 配置
   └── models/
       └── best.ckpt        # 模型权重（大文件，约 150MB）
   ```

**方法 2：使用 models 目录**

项目包含 `.gitkeep` 文件以确保 git 跟踪。实际模型文件较大，应单独下载。

**方法 3：验证下载**

下载后，验证模型结构：

```bash
ls -lh models/big-lama/config.yaml
ls -lh models/big-lama/models/best.ckpt
```

对于 `otsu` 和 `grabcut` 方法，不需要外部模型。

## 文档更新

### 2026-03-18 - 视频抽帧时间范围支持

**文档更新：**
- README.md - 添加 `--start_time` 和 `--end_time` 参数文档
- README.cn.md - 添加中文时间范围参数文档
- CHANGES_SUMMARY.md - 添加时间范围功能变更记录
- docs/plans/2026-03-18-extract-frames-time-range.md - 完整的实现计划

**时间范围功能：**
- `--start_time`：提取起始时间（秒），默认 0
- `--end_time`：提取结束时间（秒），默认视频结束
- 自动验证：起始时间 >= 0，结束时间 > 起始时间
- 自动处理：结束时间超过视频时长时自动截断
- 起始时间超过视频时长时返回空结果

### 2026-02-18 - 多格式标注输出支持

**文档更新：**
- README.md - 添加 segment.py 和 synthesize.py 的标注输出文档
- README.cn.md - 添加中文标注输出文档
- CHANGES_SUMMARY.md - 添加标注格式变更记录
- docs/plans/2026-02-18-annotation-fixes.md - 完整的实现计划，包含 segment.py 更新

**标注输出功能：**
- **多种格式**：COCO JSON、VOC Pascal XML、YOLO TXT
- **COCO 模式**：统一模式（单个文件）或分离模式（每张图片单独文件）
- **分割脚本**：标注使用原始图片坐标
- **合成脚本**：标注使用合成图片坐标
- **输出结构**：标注文件夹与图片文件夹同级
- **多边形支持**：VOC 和 YOLO 格式支持多边形分割

**segment.py：**
- 新参数：`--annotation-output-format`、`--coco-output-mode`
- bbox/segmentation 使用原始图片坐标
- file_name 指向清理后的输出图像

**synthesize.py：**
- 更新参数：`--annotation-output-format` 替代 `--coco-output`
- bbox/segmentation 使用合成图片坐标
- 移除冗余的源路径信息以提高可移植性

### 2026-02-16 - 统一脚本架构

**文档更新：**
- README.md - 添加所有 4 个脚本的完整使用示例
- requirements.txt - 验证所有依赖项
- setup.py - 验证入口点和额外功能
- docs/plans/2026-02-16-unified-scripts-architecture-design.md - 状态：已完成

**已文档化脚本：**
1. `segment.py` - 分割（SAM3、Otsu、GrabCut）
2. `extract_frames.py` - 视频帧提取
3. `clean_figs.py` - 图像清理和去重
4. `split_dataset.py` - 训练集/测试集/未知类划分
5. `synthesize.py` - 图像合成（旋转、颜色匹配、黑色区域避让、COCO 标注）

---

## 许可证

本项目采用 MIT 许可证 - 详细信息请参见 LICENSE 文件。
