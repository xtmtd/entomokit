# entomokit classify train
[English](classify-train.md) | [中文](classify-train.cn.md)

## 目的

使用 AutoGluon MultiModalPredictor 与 timm 骨干网络，在 CSV 标注的图像数据集上训练图像
分类器。需要 `classify` 附加依赖。

## 用法

```bash
entomokit classify train --train-csv data/train.csv --images-dir data/images/ \
    --out-dir runs/exp1/
```

## 参数

### `--train-csv`

必填。无默认值。包含 `image` 与 `label` 两列的 CSV。

### `--images-dir`

必填。无默认值。训练图像所在目录。

### `--out-dir`

必填。无默认值。输出目录；AutoGluon predictor 写入
`out-dir/AutogluonModels/<base-model>`。

### `--base-model`

可选。默认 `convnextv2_femto`。timm 骨干网络名称。

### `--augment`

可选。默认 `medium`。数据增强预设，或 AutoGluon 变换名的 JSON 数组。预设：
`none` = `resize_shorter_side`、`center_crop`；`light` = `none` 加
`random_horizontal_flip`；`medium` = `light` 加 `color_jitter`、`trivial_augment`；
`heavy` = `random_resize_crop`、`random_horizontal_flip`、`random_vertical_flip`、
`color_jitter`、`trivial_augment`、`randaug`。自定义示例：
`'["random_resize_crop","color_jitter","randaug"]'`。

### `--max-epochs`

可选。默认 `50`。最大训练轮数。

### `--time-limit`

可选。默认 `1.0`。训练时间上限（小时）。

### `--resume`

可选。默认关闭（不带取值的布尔开关）。从
`--out-dir/AutogluonModels/<base-model>` 的检查点继续已有的 AutoGluon 运行；配合更大的
`--max-epochs` 会继续训练到新的轮数上限。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并从头训练。

### `--learning-rate`

可选。默认 `0.0001`。优化学习率（`optim.lr`）。

### `--weight-decay`

可选。默认 `0.001`。优化器权重衰减（`optim.weight_decay`）。

### `--warmup-steps`

可选。默认 `0.1`。学习率 warmup 比例或步数（`optim.warmup_steps`）。

### `--patience`

可选。默认 `10`。早停耐心次数（`optim.patience`）。

### `--top-k`

可选。默认 `3`。用于模型平均的检查点数量（`optim.top_k`）。

### `--focal-loss`

可选。默认关闭（不带取值的布尔开关）。使用 focal loss 强化难样本，有助于类别不平衡。

### `--focal-loss-gamma`

可选。默认 `1.0`。focal loss 的 gamma 值。

### `--device`

可选。默认 `auto`。可选值：`auto`、`cpu`、`cuda`、`mps`。训练使用的计算设备。

### `--batch-size`

可选。默认 `32`。训练使用的小批量大小。

### `--num-workers`

可选。默认 `4`。dataloader 工作进程数。

### `--num-threads`

可选。默认 `0`。PyTorch 使用的 CPU 线程数；`0` 表示自动选择。

### `--seed`

可选。默认 `0`。传给 AutoMM 的随机种子，用于可复现训练。

## 输入

- `--train-csv` 需要 `image` 与 `label` 两列；图像路径相对 `--images-dir` 解析。
- 必须安装 `classify` 附加依赖（`pip install -e ".[classify]"`）。
- 模型准备：AutoMM 会在首次使用时自动下载骨干网络权重，无需手动准备检查点。`--base-model`
  接受 timm 骨干网络名；默认是 `convnextv2_femto`，其它常用选择包括 `convnextv2_tiny`、
  `convnextv2_small`、`convnextv2_base`、`resnet18`、`resnet50`、`resnet101`、
  `efficientnet_b0`-`efficientnet_b7`、`vit_small_patch16_224` 与
  `vit_base_patch16_224`；任何其它 timm 名称同样可用（https://huggingface.co/timm）。
- 训练只读取带标签的训练划分；若数据来自 `entomokit split-csv`，请把
  `--val-ratio`/`--val-count` 保持为 `0`，因为 AutoGluon 会自行执行内部训练/验证划分。

## 输出

- 训练好的 predictor 写入 `out-dir/AutogluonModels/<base-model>`，也就是
  `classify predict`、`classify evaluate`、`classify embed`、`classify cam` 与
  `classify export-onnx` 所需的 `--model-dir`。
- 运行日志写入 `out-dir/logs/log.txt`，解析后的训练表保存为 `out-dir/train.processed.csv`。

## 示例

使用更大的轮数预算与更低的学习率训练：

```bash
entomokit classify train --train-csv data/train.csv --images-dir data/images/ \
    --out-dir runs/exp1/ --max-epochs 100 --learning-rate 3e-4 --device auto
```

把已有运行从 50 轮延长到 100 轮：

```bash
entomokit classify train --train-csv data/train.csv --images-dir data/images/ \
    --out-dir runs/exp1/ --base-model convnextv2_femto --max-epochs 100 --resume
```

自定义增强并用 focal loss 处理类别不平衡：

```bash
entomokit classify train --train-csv data/train.csv --images-dir data/images/ \
    --out-dir runs/exp1/ --augment '["random_resize_crop","color_jitter","randaug"]' \
    --focal-loss --focal-loss-gamma 2.0
```

## 说明

- `classify train` 以 CSV 为准，因此不适用 README 的目录输入/输出策略：
  [目录策略](../../README.cn.md#directory-policy)。日志、设备选择与版本显示共享：
  [通用行为](../../README.cn.md#common-behaviours)；分类命令不安装 SIGINT 处理器。
- `--resume` 只在 `--base-model` 不变时有意义；更换骨干网络会切换检查点目录。
- `--time-limit` 与 `--max-epochs` 同时约束训练；先达到者即停止。
- 固定 `--seed` 时，相同数据与设置下的训练可复现。

## 版本注记

- `0.7.0`：新增 `--seed`，用于可复现的 AutoMM 训练。
