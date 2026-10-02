# entomokit synthesize
[English](synthesize.md) | [中文](synthesize.cn.md)

## 目的

将 RGBA 目标抠图按旋转、颜色匹配与面积比例合成到背景图像上，并可选地为合成后的目标写出
边界框标注。

## 用法

每个目标合成 10 张的最小调用：

```bash
entomokit synthesize --target-dir images/targets/ --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ --num-syntheses 10
```

## 参数

### `--target-dir`, `-t`

必填。无默认值。目标对象图像目录；必须是带 alpha 通道的 RGBA 抠图。

### `--background-dir`, `-b`

必填。无默认值。背景图像目录。

### `--out-dir`, `-o`

必填。无默认值。输出目录。

### `--num-syntheses`, `-n`

可选。默认 `1`。正整数表示每个目标的合成次数，背景有放回采样且不受背景数量上限约束；
`0` 到 `1` 之间的小数表示采样该比例的目标，每个选中目标合成一张。

### `--seed`

可选。默认 `42`。目标采样与任务局部合成随机性的基础随机种子；固定种子使比例采样可复现。

### `--area-ratio-min`, `-a`

可选。默认 `0.05`。最小面积比（目标面积 / 背景面积）；取值范围 `0.01`-`0.50`。

### `--area-ratio-max`, `-x`

可选。默认 `0.2`。最大面积比（目标面积 / 背景面积）；取值范围 `0.01`-`0.50`。

### `--color-match-strength`, `-c`

可选。默认 `0.5`。目标与背景的颜色匹配强度，取值 `0` 到 `1`。

### `--avoid-black-regions`, `-A`

可选。默认关闭（不带取值的布尔开关）。避免合成到背景的纯黑区域。

### `--rotate`, `-r`

可选。默认 `0.0`。最大随机旋转角度；`0` 表示不旋转。

### `--out-image-format`, `-f`

可选。默认 `png`。可选值：`png`、`jpg`。输出图像格式。

### `--annotation-output-format`

可选。默认 `coco`。可选值：`coco`、`voc`、`yolo`。合成目标的标注输出格式。

### `--coco-output-mode`

可选。默认 `unified`。可选值：`unified`、`separate`。为兼容 CLI 而保留，但当前实现始终
按 unified 布局累积并只写出单个 `annotations.coco.json`；`separate` **尚未**产生每图 JSON
文件。

### `--coco-bbox-format`

可选。默认 `xywh`。可选值：`xywh`、`xyxy`。COCO 边界框坐标约定；仅在
`--annotation-output-format coco` 时使用。

### `--threads`, `-d`

可选。默认 `4`。并行工作进程数。

### `--resume`

可选。默认关闭（不带取值的布尔开关）。仅当某目标已有的输出编号集合与当前 `--num-syntheses`
集合相等时才跳过；唯一的额外校验是 `--coco-bbox-format` 与记录的 COCO 文件一致。不会比较
`--seed`、旋转或颜色参数，因此编号集合匹配时即使这些参数变了仍会跳过。集合不完整或
`--num-syntheses` 改变会重新合成该目标；对产生了新结果的目标，上一次更大数量留下的多余副本
及其标注会被删除。使用 unified COCO 输出时会与上一次的 `annotations.coco.json` 合并，被跳过
的目标保留其标注。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新开始。

### `--verbose`, `-v`

可选。默认关闭（不带取值的布尔开关）。启用详细日志。

## 输入

- `--target-dir` 必须是 RGBA 抠图，例如 mask 模式的 `segment` 输出
  （`--segmentation-method sam3`、`otsu` 或 `grabcut`）。bbox 模式裁剪
  （`sam3-bbox`、`otsu-bbox`、`grabcut-bbox`）、修复后的图像与原始照片均为 RGB，
  会被拒绝；当所有目标都不可用时，错误信息会列出观察到的模式（例如 `18 RGB`）。
- 两个目录都会被递归扫描。
- 背景按每个合成任务有放回采样，因此每个目标的合成次数不受背景数量上限约束。

## 输出

```text
out_dir/
├── images/                  # 镜像各目标的子目录路径
│   └── <subdir>/target_01.png
├── annotations.coco.json    # --annotation-output-format coco（当前始终 unified）
├── Annotations/             # --annotation-output-format voc：每张图一个 .xml
├── labels/                  # --annotation-output-format yolo：每张图一个 .txt
└── data.yaml                # --annotation-output-format yolo：类别列表
```

- 输出文件命名为 `{target_stem}_{NN}`。同一目录下仅扩展名不同的同名目标（例如 `a.png`
  与 `a.tif`）会在词干中加入短路径摘要，避免互相覆盖。
- 标注镜像与图像相同的目标相对路径。
- 已知差异：`--coco-output-mode separate` 被 CLI 接受且其 help 文本有描述，但 COCO 分发
  不区分该模式，始终调用 `_accumulate_coco_single` 累积到 unified 写入器，因此当前只产出
  `annotations.coco.json`。该选项为向前兼容保留；修改 CLI help 措辞需要单独评审。

## 示例

输出 COCO 标注并允许最多 30 度旋转：

```bash
entomokit synthesize --target-dir images/targets/ --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ --num-syntheses 10 \
    --annotation-output-format coco --rotate 30
```

输出 YOLO 标注，加强颜色匹配并避开黑色区域：

```bash
entomokit synthesize --target-dir images/targets/ --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ --annotation-output-format yolo \
    --avoid-black-regions --color-match-strength 0.7
```

## 说明

- 递归发现、镜像布局与输出目录安全属于共享规则：
  [目录策略](../../README.cn.md#directory-policy)。日志与版本显示同样共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- 中断处理：关闭标志只在准备任务时检查，合成循环本身不检查，因此已准备好的任务在 `Ctrl+C`
  后会全部执行完。
- `--out-dir` 非空时必须显式传入 `--resume` 或 `--overwrite`，否则直接报错退出。
- 使用小数 `--num-syntheses` 并固定 `--seed` 时，多次运行会复现相同的目标选择。

## 版本注记

- `0.7.0`：`--num-syntheses` 增加小数计数语义，并新增 `--seed` 以支持可复现的目标采样。
