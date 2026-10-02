# entomokit classify evaluate
[English](classify-evaluate.md) | [中文](classify-evaluate.cn.md)

## 目的

在带标签的测试 CSV 上评估已训练的分类器，并导出总体指标、每类指标与混淆矩阵诊断。

## 用法

```bash
entomokit classify evaluate --test-csv data/test.csv --images-dir data/images/ \
    --onnx-model runs/onnx/model.onnx --out-dir runs/eval/
```

## 参数

### `--test-csv`

必填。无默认值。用于评估的、含 `image` 与 `label` 两列的 CSV。

### `--images-dir`

必填。无默认值。评估图像所在目录。

### `--model-dir`

可选。无默认值。用于评估的 AutoGluon predictor 目录。

### `--onnx-model`

可选。无默认值。ONNX 模型文件路径；需要 `onnxruntime`。与 `--model-dir` 互斥。

### `--out-dir`

必填。无默认值。评估日志与指标的输出目录。

### `--batch-size`

可选。默认 `32`。评估推理的批大小。

### `--num-workers`

可选。默认 `4`。dataloader 工作进程数。

### `--num-threads`

可选。默认 `0`。PyTorch 或 ONNX runtime 使用的 CPU 线程数；`0` 表示自动选择。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新评估。

### `--device`

可选。默认 `auto`。可选值：`auto`、`cpu`、`cuda`、`mps`。评估使用的计算设备。

## 输入

- `--test-csv` 必须包含 `image` 与 `label` 两列；图像路径相对 `--images-dir` 解析。
- `--model-dir` 与 `--onnx-model` 必须提供一个。
- 评估标签会与模型自身的类别列表比对，因此测试 CSV 应使用训练时的相同标签名。

## 输出

- `evaluations.csv`：总体指标，包括 accuracy、balanced accuracy、precision/recall/F1
  （macro、micro、weighted）、Matthews 相关系数、quadratic kappa 以及 ROC-AUC
  （OVO、OVR）。
- `confusion_matrix.csv` — 原始计数矩阵，行为真实标签、列为预测标签。
- `confusion_matrix_normalized.csv` — 行归一化矩阵，用于按类诊断召回率。
- `per_class_metrics.csv` — 每类的 precision、recall、F1 与 support。
- `confusion_matrix.pdf` — 当类别数保持可读时写出的热力图 PDF。

## 示例

评估训练产出的 AutoGluon predictor：

```bash
entomokit classify evaluate --test-csv data/test.csv --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto --out-dir runs/eval/
```

## 说明

- `evaluate` 由带标签的 CSV 驱动，因此该 CSV 不受 README 目录输入/输出策略约束：
  [目录策略](../../README.cn.md#directory-policy)。日志、设备选择与版本显示共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- `--out-dir` 非空时要求 `--overwrite`；没有 `--resume`。
- 在类别不平衡的数据上，请结合归一化混淆矩阵与 `per_class_metrics.csv` 一起判断，不要只看
  单个总体数字。

## 版本注记

- `0.7.0`：评估除总体指标外，还写出上述每类指标与混淆矩阵诊断。
