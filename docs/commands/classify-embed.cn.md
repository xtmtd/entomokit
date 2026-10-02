# entomokit classify embed
[English](classify-embed.md) | [中文](classify-embed.cn.md)

## 目的

为目录中的图像提取特征嵌入，并计算嵌入空间的质量指标，可选生成 UMAP 可视化。

## 用法

```bash
entomokit classify embed --images-dir data/images/ --out-dir runs/embed/
```

## 参数

### `--images-dir`

必填。无默认值。需要提取嵌入的图像目录；递归扫描。

### `--out-dir`

必填。无默认值。嵌入与可选可视化的输出目录。

### `--model-dir`

可选。无默认值。用于提取微调骨干网络的 AutoGluon predictor。

### `--base-model`

可选。默认 `convnextv2_femto`。timm 骨干网络；未提供 `--model-dir` 时使用。

### `--label-csv`

可选。无默认值。含 `image` 与 `label` 两列的 CSV，用于监督指标与 UMAP 着色。

### `--visualize`

可选。默认关闭（不带取值的布尔开关）。生成 UMAP 图；需要 `--label-csv`。

### `--umap-n-neighbors`

可选。默认 `15`。UMAP 流形构建的邻居数。

### `--umap-min-dist`

可选。默认 `0.1`。UMAP 嵌入点之间的最小距离。

### `--umap-metric`

可选。默认 `euclidean`。UMAP 使用的距离度量。

### `--umap-seed`

可选。默认 `42`。可复现 UMAP 布局的随机种子。

### `--metrics-sample-size`

可选。默认 `10000`。所有嵌入质量指标使用的最大样本数。

### `--batch-size`

可选。默认 `32`。嵌入提取使用的批大小。

### `--num-workers`

可选。默认 `4`。dataloader 工作进程数。

### `--num-threads`

可选。默认 `0`。PyTorch 操作使用的 CPU 线程数；`0` 表示自动选择。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新提取嵌入。

### `--device`

可选。默认 `auto`。可选值：`auto`、`cpu`、`cuda`、`mps`。嵌入提取使用的计算设备。

## 输入

- `--images-dir` 会被递归扫描。输出中的 `image` 列是相对 `--images-dir` 的路径（例如
  `beetles/a.jpg`），因此嵌套同名图像保持独立。
- `--label-csv` 的 `image` 值必须唯一，且至少有一个值与 `--images-dir` 中的图像**文件**名
  匹配。重复行或完全无交集会在提取前被拒绝；仅目录名看起来像图像不算匹配。
- 使用 `--model-dir` 获取微调的 AutoGluon 骨干网络，或使用 `--base-model` 获取预训练
  timm 骨干网络。

## 输出

- `embeddings.csv` — 特征向量（`feat_0`、`feat_1` ...），`image` 列保存相对
  `--images-dir` 的路径。
- `metrics.csv` — 质量指标，仅在提供 `--label-csv` 时写出。
- `umap.pdf` — UMAP 可视化，仅在 `--visualize` 时写出。

### 质量指标

| 指标 | 说明 |
|---|---|
| NMI | 归一化互信息（真实标签 vs KMeans） |
| ARI | 调整兰德指数 |
| Recall@1/5/10 | 检索召回率；所有查询都计入分母 |
| kNN_Acc_k1/5/20 | 交叉验证 k-NN 准确率 |
| Linear_Probing_Acc | 交叉验证线性探针准确率 |
| Linear_Probing_Balanced_Acc | 交叉验证线性探针平衡准确率 |
| mAP@R | R 处的平均精度均值；无（非自身）相关项的查询被排除 |
| Purity | 聚类纯度 |
| Silhouette_Score | 针对真实标签的余弦轮廓系数 |

## 示例

使用预训练骨干网络并生成 UMAP 图：

```bash
entomokit classify embed --images-dir data/images/ --base-model convnextv2_femto \
    --label-csv data/labels.csv --visualize --out-dir runs/embed/
```

使用微调的 AutoGluon 骨干网络：

```bash
entomokit classify embed --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --label-csv data/labels.csv --out-dir runs/embed/
```

在大数据集上降低指标采样规模：

```bash
entomokit classify embed --images-dir data/images/ --label-csv data/labels.csv \
    --metrics-sample-size 2000 --out-dir runs/embed/
```

## 说明

- `embed` 扫描目录，因此共享递归发现与镜像布局规则，即 README 的
  [目录策略](../../README.cn.md#directory-policy)。日志、设备选择与版本显示共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- `--out-dir` 非空时要求 `--overwrite`；没有 `--resume`。
- 指标约定：
  - k-NN 与线性探针使用
    `StratifiedKFold(n_splits=min(5, smallest class count), shuffle=True, random_state=42)`
    做交叉验证。
  - 聚类使用真实类别数作为 KMeans 的簇数。
  - 无法计算的指标会写成空 CSV 单元 / 打印为 `N/A`：分层交叉验证无法构建时（类别数少于
    两个，或任一类别样本少于两个）的 k-NN 与线性探针、类别数少于两个或不同嵌入数少于
    类别数时的聚类、样本数少于 `k + 1` 时的 `Recall@K`、以及未定义时的轮廓系数。单样本
    类别仍会产出 NMI、ARI 与纯度。请把 `N/A` 读作“不可计算”，绝不要当作 `0`。
  - `Recall@K` 与 `mAP@R` 保持上述定义，因此单类别但样本多于一个的标签集仍会得到 `1.0`
    而不是 `N/A`。
  - `--metrics-sample-size` 限制**所有**质量指标使用的行数（聚类、Recall@K、k-NN、
    mAP@R、轮廓系数与线性探针），是主要的运行时间开关。`mAP@R` 仍会把每个查询与所有其他
    行排序比较，其邻居索引矩阵随样本数的平方增长（默认 10000 时约 800 MB）；内存紧张时应
    降低该值。
- 可比性：与 `0.6.2` 之前的运行结果**不可数值比较**。CV 划分、轮廓系数距离度量、
  `Recall@K` 与 `mAP@R` 中基于索引的自我排除、以及不可用值处理都发生了变化。字段名未变，
  `Linear_Probing_Balanced_Acc` 是插入在 `Linear_Probing_Acc` 之后的新列，因此其后所有列
  右移一位——请按表头名而非位置取值。

## 版本注记

- `0.7.0`：不改变嵌入指标算法。
- `0.6.2`：kNN 与评估修正改变了 CV 划分、轮廓系数距离度量、`Recall@K`/`mAP@R` 中基于索引
  的自我排除与不可用值处理；新增 `Linear_Probing_Balanced_Acc` 列。
