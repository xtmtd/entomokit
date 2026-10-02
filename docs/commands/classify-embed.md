# entomokit classify embed
[English](classify-embed.md) | [中文](classify-embed.cn.md)

## Purpose

Extract feature embeddings for a directory of images and compute embedding-space
quality metrics, optionally with a UMAP visualisation.

## Usage

```bash
entomokit classify embed --images-dir data/images/ --out-dir runs/embed/
```

## Parameters

### `--images-dir`

Required. No default. Directory containing the images to embed; scanned
recursively.

### `--out-dir`

Required. No default. Directory to write embeddings and optional visualisations.

### `--model-dir`

Optional. No default. AutoGluon predictor used to extract a fine-tuned backbone.

### `--base-model`

Optional. Default `convnextv2_femto`. timm backbone; used when `--model-dir` is not
provided.

### `--label-csv`

Optional. No default. CSV with `image` and `label` columns, used for supervised
metrics and UMAP colouring.

### `--visualize`

Optional. Default off (boolean flag without a value). Generate a UMAP plot; requires
`--label-csv`.

### `--umap-n-neighbors`

Optional. Default `15`. UMAP neighbour count for manifold construction.

### `--umap-min-dist`

Optional. Default `0.1`. UMAP minimum distance between embedded points.

### `--umap-metric`

Optional. Default `euclidean`. Distance metric used by UMAP.

### `--umap-seed`

Optional. Default `42`. Random seed for reproducible UMAP layouts.

### `--metrics-sample-size`

Optional. Default `10000`. Maximum sample size used by all embedding quality
metrics.

### `--batch-size`

Optional. Default `32`. Batch size used during embedding extraction.

### `--num-workers`

Optional. Default `4`. Number of dataloader worker processes.

### `--num-threads`

Optional. Default `0`. CPU threads for PyTorch operations; `0` selects
automatically.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and re-extract embeddings.

### `--device`

Optional. Default `auto`. Choices: `auto`, `cpu`, `cuda`, `mps`. Compute device for
embedding extraction.

## Inputs

- `--images-dir` is scanned recursively. The `image` column of the outputs is the
  path relative to `--images-dir` (for example `beetles/a.jpg`), so nested
  same-named images stay distinct.
- `--label-csv` must have unique `image` values and at least one `image` value
  matching an image **file** name in `--images-dir`. Duplicate rows and an empty
  overlap are rejected before any extraction; a directory whose name looks like an
  image does not count.
- Use either `--model-dir` for a fine-tuned AutoGluon backbone or `--base-model`
  for a pretrained timm backbone.

## Outputs

- `embeddings.csv` — feature vectors (`feat_0`, `feat_1`, ...) with the `image`
  column holding the path relative to `--images-dir`.
- `metrics.csv` — quality metrics, written only with `--label-csv`.
- `umap.pdf` — UMAP visualisation, written with `--visualize`.

### Quality metrics

| Metric | Description |
|---|---|
| NMI | Normalized Mutual Information (true labels vs KMeans) |
| ARI | Adjusted Rand Index |
| Recall@1/5/10 | Retrieval recall at K; every query is in the denominator |
| kNN_Acc_k1/5/20 | Cross-validated k-NN accuracy |
| Linear_Probing_Acc | Cross-validated linear-probe accuracy |
| Linear_Probing_Balanced_Acc | Cross-validated linear-probe balanced accuracy |
| mAP@R | Mean Average Precision at R; queries with no non-self relevant item are excluded |
| Purity | Cluster purity |
| Silhouette_Score | Cosine silhouette against the true labels |

## Examples

Pretrained backbone with a UMAP plot:

```bash
entomokit classify embed --images-dir data/images/ --base-model convnextv2_femto \
    --label-csv data/labels.csv --visualize --out-dir runs/embed/
```

Fine-tuned AutoGluon backbone:

```bash
entomokit classify embed --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --label-csv data/labels.csv --out-dir runs/embed/
```

Reduce the metric sample size on a large dataset:

```bash
entomokit classify embed --images-dir data/images/ --label-csv data/labels.csv \
    --metrics-sample-size 2000 --out-dir runs/embed/
```

## Notes

- `embed` scans a directory; the shared recursive and mirrored-layout rules are the
  README's [directory policy](../../README.md#directory-policy). Logging, device
  selection and version display are shared:
  [common behaviours](../../README.md#common-behaviours).
- A non-empty `--out-dir` requires `--overwrite`; there is no `--resume` flag.
- Metric contract:
  - k-NN and linear probing are cross-validated with
    `StratifiedKFold(n_splits=min(5, smallest class count), shuffle=True, random_state=42)`.
  - Clustering uses KMeans with the true class count.
  - Metrics that cannot be computed are written as an empty CSV cell / printed as
    `N/A`: k-NN and linear probing when stratified CV is impossible (fewer than two
    classes, or any class with fewer than two samples), clustering when there are
    fewer than two classes or fewer distinct embeddings than classes, `Recall@K` when
    fewer than `k + 1` samples exist, and silhouette when it is undefined. Singleton
    classes still produce NMI, ARI and purity. Read `N/A` as "not computable", never
    as `0`.
  - `Recall@K` and `mAP@R` keep the definitions above, so a single-class label set
    with more than one sample still yields `1.0` rather than `N/A`.
  - `--metrics-sample-size` caps the rows used by **all** quality metrics
    (clustering, Recall@K, k-NN, mAP@R, silhouette and linear probing) and is the
    main runtime knob. `mAP@R` still ranks every query against every other row, so
    its neighbour-index matrix grows with the square of the sample size
    (~800 MB at the default 10000); lower this value when memory is tight.
- Comparability: values are **not** numerically comparable with runs produced
  before `0.6.2`. The CV split, the silhouette distance metric, index-based
  self-exclusion in `Recall@K` and `mAP@R`, and unavailable-value handling all
  changed. Field names are unchanged, and `Linear_Probing_Balanced_Acc` is a new
  column inserted after `Linear_Probing_Acc`, so later columns shift right by one —
  select by header name, not position.

## Version Notes

- `0.7.0`: does not change the embedding metrics algorithm.
- `0.6.2`: kNN and evaluation corrections changed the CV split, the silhouette
  distance metric, index-based self-exclusion in `Recall@K`/`mAP@R` and
  unavailable-value handling; `Linear_Probing_Balanced_Acc` was added.
