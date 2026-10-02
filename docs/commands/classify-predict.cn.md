# entomokit classify predict
[English](classify-predict.md) | [中文](classify-predict.cn.md)

## 目的

使用 AutoGluon predictor 或 ONNX 模型对图像做推理，并写出每张图的预测结果与类别概率。

## 用法

```bash
entomokit classify predict --images-dir data/test/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto --out-dir runs/predict/
```

## 参数

### `--input-csv`

可选。无默认值。包含 `image` 列的 CSV。`--input-csv` 与 `--images-dir` 至少提供一个。

### `--images-dir`

可选。无默认值。用于扫描图像的目录，或用于解析 `--input-csv` 中相对 `image` 值的根目录。
`--input-csv` 与 `--images-dir` 至少提供一个。

### `--model-dir`

可选。无默认值。AutoGluon predictor 目录，通常是 `classify train` 产出的
`out-dir/AutogluonModels/<base-model>`。

### `--onnx-model`

可选。无默认值。ONNX 模型文件路径；需要 `onnxruntime`。与 `--model-dir` 互斥。

### `--out-dir`

必填。无默认值。推理输出目录。

### `--batch-size`

可选。默认 `32`。模型推理的批大小。

### `--num-workers`

可选。默认 `4`。dataloader 工作进程数。

### `--num-threads`

可选。默认 `0`。PyTorch 或 ONNX runtime 使用的 CPU 线程数；`0` 表示自动选择。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新预测全部输入。

### `--device`

可选。默认 `auto`。可选值：`auto`、`cpu`、`cuda`、`mps`。推理使用的计算设备。

## 输入

- `--input-csv` 与 `--images-dir` 至少提供一个。
- 若 CSV 的 `image` 值本身已是可读路径，则直接使用 CSV。
- 若 CSV 的 `image` 值是文件名或相对路径，还需提供 `--images-dir`。
- 若只提供 `--images-dir`，则预测该目录下的所有图像。
- `--model-dir` 与 `--onnx-model` 必须提供一个；ONNX 需要 `onnxruntime`
  （`pip install onnxruntime`，或 `classify` 附加依赖）。
- `--images-dir` 的发现是递归的。发现的图像按相对 `--images-dir` 的路径记录（例如
  `beetles/a.jpg`），因此嵌套同名图像保持独立。通过 `--input-csv` 显式给出的路径按原样使用。

## 输出

- 预测结果写入 `out-dir/predictions/predictions.csv`，列与 AutoGluon 输出一致：`image`、
  `prediction`、`proba_<class_name>` 等。输入无法解析时，缺失图像列表写入
  `out-dir/logs/missing_images.txt`。
- 使用 ONNX 模型时，若 ONNX 文件旁存在 `label_classes.json`，`prediction` 为类别名，
  否则为数值类别索引。

## 示例

用 ONNX 模型预测：

```bash
entomokit classify predict --input-csv test.csv --onnx-model runs/onnx/model.onnx \
    --out-dir runs/predict/
```

把 CSV 中的相对路径解析到图像根目录：

```bash
entomokit classify predict --input-csv out/split/test.known.csv \
    --images-dir data/Epidorcus/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto --out-dir runs/predict/
```

## 说明

- `predict` 接受 CSV 输入，因此 CSV 形式不受 README 目录输入/输出策略约束：
  [目录策略](../../README.cn.md#directory-policy)。日志、设备选择与版本显示共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- `--out-dir` 非空时要求 `--overwrite`；没有 `--resume`。
- 传入 `--overwrite` 会删除 `--out-dir` 内容并重新预测全部输入。

## 版本注记

- `0.7.0`：`--images-dir` 递归发现后按相对 `--images-dir` 的路径记录图像，取代仅记文件名的
  方式。
