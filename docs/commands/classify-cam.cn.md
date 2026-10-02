# entomokit classify cam
[English](classify-cam.md) | [中文](classify-cam.cn.md)

## 目的

为已训练的分类器生成 GradCAM 系列显著性热力图，便于直观检查预测依据。CAM 依赖 PyTorch
的前向/反向钩子，因此不支持 ONNX 模型。

## 用法

```bash
entomokit classify cam --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --cam-method gradcam --out-dir runs/cam/ --save-npy raw
```

## 参数

### `--images-dir`

必填。无默认值。包含图像的目录；提供 `--label-csv` 时也是其图像路径的根目录。

### `--out-dir`

必填。无默认值。CAM 可视化与产物的输出目录。

### `--label-csv`

可选。无默认值；省略时使用 `--images-dir` 中的全部图像。含 `image` 与 `label` 两列的 CSV。

### `--model-dir`

可选。无默认值。AutoGluon predictor 目录。

### `--base-model`

可选。无默认值。timm 骨干网络名称。

### `--checkpoint-path`

可选。无默认值。timm 骨干网络的自定义 `.pth` 权重。

### `--num-classes`

可选。无默认值。加载自定义 timm 检查点时的类别数。

### `--no-pretrained`

可选。默认关闭（不带取值的布尔开关）。使用 `--base-model` 时禁用预训练 timm 权重。

### `--cam-method`

可选。默认 `gradcam`。可选值：`gradcam`、`gradcampp`、`layercam`、`scorecam`、
`eigencam`、`ablationcam`。生成显著性图使用的 CAM 算法。

### `--arch`

可选。无默认值；自动检测架构。可选值：`cnn`、`vit`。强制指定架构类型。

### `--target-layer-name`

可选。无默认值；自动选择层。CAM 使用的具体模型层。

### `--image-weight`

可选。默认 `0.5`。CAM 叠加图中原始图像的混合权重，取值 `0` 到 `1`。

### `--fig-format`

可选。默认 `png`。可选值：`png`、`jpg`、`pdf`。CAM 图的输出格式。

### `--save-npy`

可选。默认 `none`。可选值：`none`、`raw`、`normalized`。把 CAM 数组保存到
`arrays/*.npy`：`none` 不写文件，`raw` 保留未归一化的正向 CAM 幅值，
`normalized` 写出每张图 min-max 到 `[0, 1]` 的掩码。

### `--dump-model-structure`

可选。默认关闭（不带取值的布尔开关）。把模型层名写入 `out-dir/model_layers.txt`，供
`--target-layer-name` 参考。

### `--max-images`

可选。无默认值；处理全部图像。最多处理的图像数量。

### `--cam-batch-size`

可选。默认 `32`。CAM 推理的批大小。

### `--eval-transform`

可选。默认 `center-crop`。可选值：`center-crop`、`whole-specimen-pad`。评估预处理：
使用模型保存的确定性中心裁剪，或使用保持长宽比的方形填充以覆盖完整标本。

### `--num-threads`

可选。默认 `0`。PyTorch 操作使用的 CPU 线程数；`0` 表示自动选择。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新生成 CAM 可视化。

### `--device`

可选。默认 `auto`。可选值：`auto`、`cpu`、`cuda`、`mps`。CAM 生成使用的计算设备。

## 输入

- 通过 `--model-dir`，或 `--base-model`（可配合 `--checkpoint-path` 与
  `--no-pretrained`）选择模型；ONNX 模型会被拒绝，因为 CAM 需要 PyTorch 钩子。
- 架构会自动检测：Swin 骨干网络按 ViT 风格处理，默认 CAM 目标为最后一个 stage block；
  ConvNeXt 使用最后一个 stage block 而不是其逐点 `mlp.fc2` 层，以避免退化热力图。
- 需要指定具体目标层时，先用 `--dump-model-structure` 查看层名。

## 输出

- `figures/` — CAM 叠加图，格式由 `--fig-format` 决定。
- `cam_summary.csv` — 生成热力图的元数据。
- `arrays/` — CAM 数组，仅在 `--save-npy raw` 或 `--save-npy normalized` 时写出。
- `model_layers.txt` — 层名，仅在 `--dump-model-structure` 时写出。

## 示例

带真实标签并导出归一化数组：

```bash
entomokit classify cam --label-csv data/test.csv --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/cam/ --save-npy normalized
```

使用完整标本预处理与另一种 CAM 方法：

```bash
entomokit classify cam --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --cam-method gradcampp --eval-transform whole-specimen-pad --out-dir runs/cam/
```

查找目标层：

```bash
entomokit classify cam --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --dump-model-structure --out-dir runs/cam/
```

## 说明

- CAM 读取目录，共享递归发现规则即 README 的
  [目录策略](../../README.cn.md#directory-policy)。日志、设备选择与版本显示共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- `--out-dir` 非空时要求 `--overwrite`；没有 `--resume`。
- raw 与 normalized 的区别：`normalized` 写出与叠加图相同的每图 min-max 副本，保持此前的
  归一化语义。`raw` 保留 CAM 幅值，只在同一模型、同一目标层与同一预处理配置内可比；
  `eigencam` 的 raw 值仅为完整性导出，不适合做响应幅值统计。两种模式下叠加图的显示语义
  相同，但不保证像素级一致。
- `--eval-transform center-crop`（默认）把热力图限制在模型的中心裁剪视野内；
  `whole-specimen-pad` 先用边界中位背景色把图像补成正方形再缩放，因此热力图覆盖完整标本
  且不产生长宽比畸变。

### CAM 约定

- 使用 `--model-dir` 时，CAM 包装 AutoGluon 的 `backbone -> classification head`，解释最终
  训练类别；`cam_summary.csv` 中的 `pred_class` 是 predictor 的真实类别标签，而不是 backbone
  特征维度索引。
- CAM 复用 predictor 保存的 `ImageProcessor.val_processor`：保存的 `image_size` 是唯一输入
  尺寸来源（224、384 或自定义尺寸自动适配）；没有已构建 processor 时才由保存的
  `val_transforms` 重建。
- `center-crop` 保留该验证预处理并把热图逆映射回完整原图：模型视野外的像素被压暗去色而不是
  拉伸，绝不把裁剪区域铺满整幅图。`whole-specimen-pad` 先用图像四边逐通道中位色补成方形，
  再缩放到保存的输入尺寸，覆盖完整标本。
- `--base-model` 且无保存 processor 时使用 timm 数据配置，热图映射保留实际的 resize 与 crop
  尺寸（例如 `Resize(256) -> CenterCrop(224)`）。
- 保存的数组是**模型输入空间**的 float32。`raw` 保留未归一化正值 CAM 幅值（跳过库内
  `scale_cam_image()`，保留 ReLU 与 resize 到模型输入尺寸），overlay 始终使用独立的逐图
  min-max 副本。`normalized` 在 float32 容差内与此前的归一化模型空间 mask 一致，但非逐位
  相同，也非跨模型/跨输入保证；overlay PNG 的像素差异不构成兼容性契约。
- `raw` 幅值只在同一模型、同一目标层、同一预处理配置内可比；`eigencam` 的 raw 值来自符号
  任意的 SVD 投影，不可用于响应强度统计。

## 版本注记

- `0.7.0`：`--save-npy` 改为显式取 `none`/`raw`/`normalized`，把未归一化的 CAM 数组与
  归一化数组分开导出。
