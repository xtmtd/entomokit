# entomokit measure
[English](measure.md) | [中文](measure.cn.md)

## 目的

从分割掩码批量计算形态学指标并导出 CSV 报告。指标定义与 scikit-image `regionprops`
口径对齐，便于结果复现与跨工具对比。

## 用法

最小运行：先用 `segment` 生成掩码，再测量。

```bash
entomokit segment --input-dir images/ --out-dir segmented/ \
    --segmentation-method otsu --annotation-format voc
entomokit measure --mask-dir segmented/SegmentationClass --out-dir runs/measure
```

带比例尺：

```bash
entomokit measure --mask-dir segmented/SegmentationClass --out-dir runs/measure \
    --pixel-size-um 2.5
```

## 参数

### `--mask-dir`, `-i`

必填。无默认值。二值掩码图像目录，递归扫描。每张图按二值掩码读取：三维图像**只取第一个
通道**，以 `> 0` 二值化，并忽略 alpha 通道。

### `--out-dir`, `-o`

必填。无默认值。CSV 报告的输出目录。

### `--pixel-size-um`

可选。无默认值；测量结果保持像素单位，不推导物理单位列。像素尺寸（`um/px`，微米每像素）。

### `--resume`

可选。默认关闭（不带取值的布尔开关）。追加新掩码的测量结果，并跳过 `metrics.csv`
中已有的掩码。影响输出的参数（目前是 `--pixel-size-um`）会记录在
`out-dir/.entomokit/measure_params.json`，必须与上一次运行一致；不一致会直接报错退出，
而不是混用不同比例的测量结果。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新开始。

### `--verbose`, `-v`

可选。默认关闭（不带取值的布尔开关）。启用详细日志。

## 输入

- `--mask-dir` 会被递归扫描；其中找到的每个掩码图像都会被测量。
- 该目录必须存放二值掩码：`segment --annotation-format voc` 写出的
  `SegmentationClass/*.png`（前景 255、背景 0），或你自己的掩码图像。**不要**把
  `--mask-dir` 指向 `segment` 的 `images/` 目录：那里是 RGB/RGBA 裁剪图，而 `measure`
  只对第一个颜色通道做二值化，因此一个 alpha 通道正确但颜色很暗的标本可能被测成空掩码。
- 只测量每张掩码的最大连通分量：其余分量在计算指标前会被丢弃（连通性 2）。因此破碎的
  标本或含多个目标的掩码不会按合并前景测量——如有需要，请先拆分或合并这些掩码。
- 提供 `--pixel-size-um` 时启用物理单位指标；不提供时报告保持像素单位。

## 输出

```text
out_dir/
├── metrics.csv              # 每张图的指标与告警原因
├── metrics_summary.csv      # 汇总统计与按原因聚合的告警计数
└── metric_definitions.csv   # 指标说明（中英文字段 + 单位/公式）
```

`metrics.csv` 中的 `file_name` 是掩码相对于 `--mask-dir` 的路径（例如
`beetles/a.png`），既避免嵌套同名掩码冲突，也是 `--resume` 使用的键。

## 示例

带比例尺并启用详细日志的测量：

```bash
entomokit measure --mask-dir segmented/SegmentationClass --out-dir runs/measure \
    --pixel-size-um 2.5 --verbose
```

为上一次运行之后新增的掩码追加测量结果，并保持原比例尺：

```bash
entomokit measure --mask-dir segmented/SegmentationClass --out-dir runs/measure \
    --pixel-size-um 2.5 --resume
```

## 说明

- 递归扫描、镜像输出布局与输出目录安全属于共享规则：
  [目录策略](../../README.cn.md#directory-policy)。日志、中断与版本显示同样共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- 关于体长/体宽的谨慎说明：
  - `body_length_*` 与 `body_width_*` 是基于二值掩码几何形态的估计值，不等同于严格解剖学
    实测值。
  - 当掩码包含附肢（触角/足）、目标被图像边界截断、或虫体区域粘连/破碎时，体长与体宽可能
    产生偏差。
  - 下游分析前请结合 `quality_flag` 与 `warn_reason`（如 `touching_border`、
    `too_many_branches`）进行人工复核。
- `--out-dir` 非空时必须显式传入 `--resume`（追加到已有报告）或 `--overwrite`
  （重新开始），否则直接报错退出。
