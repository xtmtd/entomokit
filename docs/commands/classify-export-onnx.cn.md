# entomokit classify export-onnx
[English](classify-export-onnx.md) | [中文](classify-export-onnx.cn.md)

## 目的

把已训练的 AutoGluon 图像分类器导出为 ONNX 以便部署，并同时写出
`classify predict` 与 `classify evaluate` 使用的类别标签映射。

## 用法

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ --opset 17
```

## 参数

### `--model-dir`

必填。无默认值。要导出的 AutoGluon predictor 目录。

### `--out-dir`

必填。无默认值。`model.onnx` 的输出目录。

### `--opset`

可选。默认 `17`。ONNX opset 版本。

### `--sample-image`

可选。无默认值；默认使用自动生成的临时图像做 trace。用作 ONNX trace 输入的图像路径。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新导出 ONNX 模型。

## 输入

- `--model-dir` 必须是 `classify train` 产出的 AutoGluon predictor 目录。
- 需要 `onnx` 与 `onnxruntime`；它们随 `classify` 附加依赖安装。
- `--sample-image` 应与模型训练时的输入尺寸与通道布局一致；不提供时会 trace 一张自动生成的
  临时图像。

## 输出

- `model.onnx` — 导出的 ONNX 模型。
- `label_classes.json` — 写在模型旁的类别标签映射；下游预测代码用它把数值输出还原为类别名。

## 示例

使用默认 opset 导出：

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto --out-dir runs/onnx/
```

导出并用示例图像做 trace：

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ --sample-image data/sample.jpg
```

## 说明

- 导出读取 predictor 目录，与 CSV 无关；共享目录规则见 README 的
  [目录策略](../../README.cn.md#directory-policy)。日志、设备选择与版本显示共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- `--out-dir` 非空时要求 `--overwrite`；没有 `--resume`。
- 请把 `label_classes.json` 与 `model.onnx` 放在一起：`classify predict` 会读取它把数值
  输出映射回类别名。

## 版本注记

- `0.7.0`：导出时在 `model.onnx` 旁写出 `label_classes.json`，使 ONNX 预测能输出类别名。
