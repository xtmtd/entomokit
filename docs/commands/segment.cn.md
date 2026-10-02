# entomokit segment
[English](segment.md) | [中文](segment.cn.md)

## 目的

从图像中分割昆虫，并写出边界框或分割标注（默认 COCO 格式）。提供六种分割方法：SAM3、Otsu、GrabCut，
每种都有对应的 bbox 裁剪变体。标注粒度跟随方法：以 `-bbox` 结尾的方法只输出边界框标注，
其余方法同时输出边界框与分割信息。

## 用法

使用默认 COCO 标注的最小 Otsu 调用：

```bash
entomokit segment --input-dir images/ --out-dir out/ --segmentation-method otsu
```

SAM3 需要检查点：

```bash
entomokit segment --input-dir images/ --out-dir out/ \
    --segmentation-method sam3 --sam3-checkpoint checkpoints/sam3.pt \
    --annotation-format coco
```

## 参数

### `--input-dir`, `-i`

必填。无默认值。输入图像目录，递归扫描。

### `--out-dir`, `-o`

必填。无默认值。输出目录。分割后的图像平铺到该目录下的 `images/`；标注文件写入「输出」
一节所述的各格式目录。

### `--segmentation-method`

可选。默认 `sam3`。可选值：`sam3`、`sam3-bbox`、`otsu`、`otsu-bbox`、`grabcut`、
`grabcut-bbox`。以 `-bbox` 结尾的方法只输出边界框标注；其余方法同时输出边界框与分割
标注。

### `--sam3-checkpoint`, `-c`

可选。无默认值。SAM3 检查点文件路径；当 `--segmentation-method` 为 `sam3` 或
`sam3-bbox` 时必填。

### `--hint`, `-t`

可选。默认 `insect`。用于 SAM3 grounding 的文本提示。

### `--device`, `-d`

可选。默认 `auto`。可选值：`auto`、`cpu`、`cuda`、`mps`。推理使用的设备；`auto`
自动选择可用设备。

### `--confidence-threshold`

可选。默认 `0.0`。接受掩码的最低置信度；`0.0` 表示不过滤。

### `--padding-ratio`

可选。默认 `0.0`。边界框填充比例；`0.0` 表示不填充。

### `--repair-strategy`, `-r`

可选。无默认值；不执行修复。可选值：`opencv`、`sam3-fill`、`black-mask`、`lama`。
作用于源图像：把检测到的前景掩码并集做 inpaint（`opencv`、`sam3-fill`、`lama`）或
填充为黑色（`black-mask`），结果按与分割图相同的编码样本 ID 与输出格式写入
`repaired_images/`。产出的掩码与标注文件不会被修改。参数帮助写作 "filling holes"，与实际行为不符。

### `--lama-model`

可选。无默认值。LaMa 模型检查点目录路径；使用 `--repair-strategy lama` 时必填。

### `--lama-mask-dilate`

可选。默认 `0`。对 LaMa 修复掩码执行的膨胀迭代次数。

### `--out-image-format`, `-f`

可选。默认 `png`。可选值：`png`、`jpg`。输出图像格式。

### `--threads`, `-n`

可选。默认 `8`。Otsu 与 GrabCut 方法使用的并发图像工作线程数；SAM3 与 SAM3-bbox 使用
单个有状态 predictor，在所有设备上均保持串行。

### `--annotation-format`

可选。无默认值，但 CLI 会替换为 `coco`，因此省略该参数仍会写出 COCO 标注；该参数无法
关闭标注输出。可选值：`coco`、`voc`、`yolo`。仅控制标注文件的编码方式；标注粒度跟随
`--segmentation-method`（VOC 分割以 mask PNG 文件存储）。参数帮助写作
`None = no annotations`，与实际行为不符。

### `--coco-output-mode`

可选。默认 `unified`。可选值：`unified`、`separate`。COCO 布局：`unified` 写出单个
`annotations.coco.json`，`separate` 每张图写一个 JSON 文件。

### `--coco-bbox-format`

可选。默认 `xywh`。可选值：`xywh`、`xyxy`。COCO 边界框坐标约定；仅在
`--annotation-format coco` 时使用。

### `--resume`

可选。默认关闭（不带取值的布尔开关）。继续上一次运行：精确的单 mask 输出已存在的输入会被
跳过，而多 mask 输入（`{sample_id}_01.png`、`_02` ...）始终重新处理，避免把写了一半的
结果当作完成。使用 unified COCO 输出时，会与上一次的 `annotations.coco.json` 合并，
被跳过的样本保留其标注。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新开始。

### `--verbose`, `-v`

可选。默认关闭（不带取值的布尔开关）。启用详细日志。

## 输入

- `--input-dir` 会被递归扫描。分割方法未返回掩码的图像会在日志中记录完整源路径，并写入
  `out-dir/no_mask_images.txt`。
- `sam3` 与 `sam3-bbox` 需要 `--sam3-checkpoint`；SAM3 还使用 `--hint` 作为 grounding
  提示。检查点不会自动下载：请从 https://huggingface.co/facebook/sam3 获取并传入文件路径。
- `lama` 修复需要 `--lama-model` 指向 Big-LaMa 检查点目录，且权重放在 `models/` 下：

  ```text
  models/big-lama/
  ├── config.yaml
  └── models/best.ckpt
  ```

  下载：https://github.com/advimman/lama。`opencv`、`sam3-fill`、`black-mask` 不需要
  模型文件。
- SAM3 在 `--device` 上运行；Otsu 与 GrabCut 在 CPU 工作线程上运行。

## 输出

分割后的图像总是平铺到 `images/`；输入相对路径会被编码为唯一、文件系统安全的样本 ID
（可读词干加路径摘要，例如 `a__4cabcf2b3682`）。该 ID 用于图像文件、VOC XML、
YOLO TXT、SegmentationClass 掩码、COCO 文件名和 `ImageSets/Main/default.txt`，因此
不同子目录下的同名图像会保持独立。

```text
out_dir/
├── images/                    # 分割后的图像
├── repaired_images/           # 仅在启用修复策略时生成
├── annotations.coco.json      # --annotation-format coco 且 --coco-output-mode unified
├── annotations/               # --annotation-format coco 且 --coco-output-mode separate
├── Annotations/               # --annotation-format voc：每张图一个 .xml
├── SegmentationClass/         # 非 -bbox 的 voc：mask PNG，前景=255，背景=0
├── ImageSets/Main/default.txt # voc：样本清单
├── labels/                    # --annotation-format yolo：每张图一个 .txt
├── data.yaml                  # --annotation-format yolo：类别列表（输出根目录）
└── no_mask_images.txt         # 所选方法未产生掩码的图像
```

完整的标准数据集布局（例如 Pascal VOC 的 `JPEGImages/`）由后续转换或划分步骤生成，
而不是 `segment` 直接产出。

### 标注字段说明

对于不带 `-bbox` 后缀的方法，`area` 为**掩码像素面积**（`np.sum(mask > 0)`），
`segmentation` 为 polygon 坐标数组（`[x1,y1,x2,y2,...]`）。对于 `-bbox` 方法，
`area` 为**边界框面积**（`w × h`），`segmentation` 为空。

## 示例

使用 GrabCut bbox 裁剪并输出 `xyxy` 约定的 COCO 标注：

```bash
entomokit segment --input-dir images/ --out-dir out/ \
    --segmentation-method grabcut-bbox --annotation-format coco \
    --coco-bbox-format xyxy
```

SAM3 分割 + LaMa 修复 + VOC 标注：

```bash
entomokit segment --input-dir images/ --out-dir out/ \
    --segmentation-method sam3 --sam3-checkpoint checkpoints/sam3.pt \
    --repair-strategy lama --lama-model models/lama \
    --annotation-format voc
```

继续上一次运行并保留其标注：

```bash
entomokit segment --input-dir images/ --out-dir out/ --resume
```

## 说明

- 目录扫描、扁平样本 ID 例外与输出目录安全属于共享规则：
  [目录策略](../../README.cn.md#directory-policy)。日志、设备选择与版本显示同样共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- 中断处理：关闭标志在图像之间检查。串行路径在完成当前图像后停止；并发 Otsu/GrabCut 路径会
  提前提交全部计算任务，因此已排队的图像仍会跑完才停止。
- `--resume` 只把精确的单 mask 输出视为完成；由于不记录 mask 数量，多 mask 输入始终重新
  处理。
- `--out-dir` 非空时必须显式传入 `--resume`（继续）或 `--overwrite`（重新开始），否则
  直接报错退出；`--overwrite` 会在运行前删除已有内容。
- `--annotation-format` 决定编码方式；是否写出边界框与分割数据由
  `--segmentation-method` 决定。

## 版本注记

- `0.7.0`：`--input-dir` 改为递归扫描，图像平铺到 `images/` 并使用编码后的样本 ID
  而非镜像路径；`--resume` 会合并上一次运行的 unified COCO 标注。
