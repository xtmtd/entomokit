# entomokit split-csv
[English](split-csv.md) | [中文](split-csv.cn.md)

## 目的

把带标签的 CSV 划分为 train、validation 与 test 文件，可选地保留 unknown 类测试划分，
并把图像复制到各划分目录。

## 用法

带验证集与 known 类测试划分的比例划分：

```bash
entomokit split-csv --raw-image-csv data/images.csv \
    --known-test-sample-ratio 0.1 --val-ratio 0.1 --out-dir datasets/
```

## 参数

### `--raw-image-csv`

必填。无默认值。包含 `image` 与 `label` 两列的输入 CSV。

### `--mode`

可选。默认 `ratio`。可选值：`ratio`、`count`。划分策略：按样本比例或按显式样本数。

### `--known-test-sample-ratio`

可选。默认 `0.1`。`ratio` 模式下划入测试集的 known 样本比例。

### `--unknown-test-sample-ratio`

可选。默认 `0.0`。`ratio` 模式下保留给 unknown 测试划分的目标样本比例；`0` 表示不生成
unknown 划分。类别会被随机打乱并整类移入 unknown 划分，直到累计样本数达到该目标，因此
实际划分可能超出目标，且与 known 数据类别互斥。

### `--known-test-sample-count`

可选。默认 `0`。`count` 模式下划入测试集的 known 样本目标数量。

### `--unknown-test-sample-count`

可选。默认 `0`。`count` 模式下保留给 unknown 测试划分的目标数量。与 `ratio` 模式一样，
会整类移入直到累计数量达到目标，因此划分可能超出目标，且不会拆分任何类别。

### `--val-ratio`

可选。默认 `0.0`。从训练集切出的验证集比例；`0` 表示不生成验证划分。

### `--val-count`

可选。默认 `0`。从训练集切出的验证集样本数；`0` 表示不生成验证划分。

### `--min-count-per-class`

可选。默认 `0`。仅在 `--mode count` 下，丢弃剩余样本数少于该值的类别。

### `--max-count-per-class`

可选。无默认值；不设类别上限。仅在 `--mode count` 下，对剩余训练数据中每个类别最多保留
该数量。

### `--seed`

可选。默认 `42`。可复现划分的随机种子。

### `--out-dir`

可选。默认 `datasets`。划分 CSV 与复制图像的输出目录。

### `--images-dir`

可选。无默认值。仅由 `--copy-images` 使用的源图像目录；设置该开关时必填。它不会解析或
改写 CSV 的 `image` 值。

### `--copy-images`

可选。默认关闭（不带取值的布尔开关）。把图像复制到 `out_dir/images/{split}/` 子目录下。
只保留文件名——源父目录会被丢弃——而且没有冲突检查：落入同一划分的 `beetles/a.jpg` 与
`flies/a.jpg` 都指向 `images/<split>/a.jpg`，后复制者覆盖先前者。因此同一划分内 basename
必须唯一；增加重命名或冲突检查属于代码改动，需要单独授权。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新生成全部划分。

### `--verbose`, `-v`

可选。默认关闭（不带取值的布尔开关）。划分过程中启用详细输出。

## 输入

- `--raw-image-csv` 必须包含 `image` 与 `label` 列。`image` 值会按书写原样写入各划分
  CSV；`--images-dir` 只是 `--copy-images` 的复制来源，不会解析或改写这些值。
- `--mode ratio` 使用 `*-ratio` 参数；`--mode count` 使用 `*-count` 参数。
- `--copy-images` 需要 `--images-dir`。
- 若输出要交给 `entomokit classify train`，请把 `--val-ratio` 与 `--val-count` 保持为
  `0`；AutoGluon 会自行执行内部训练/验证划分。

## 输出

```text
out_dir/
├── train.csv
├── val.csv               # 仅当 --val-ratio 或 --val-count 大于 0
├── test.known.csv
├── test.unknown.csv      # 仅当配置了 unknown 划分
├── class_count/          # 各划分的类别计数
│   ├── class.train.count
│   ├── class.val.count
│   └── ...
└── images/               # 仅在 --copy-images 时
    ├── train/
    ├── val/
    ├── test_known/
    └── test_unknown/
```

## 示例

按数量划分并复制图像：

```bash
entomokit split-csv --raw-image-csv data/images.csv --mode count \
    --known-test-sample-count 100 --val-count 50 \
    --copy-images --images-dir images/ --out-dir datasets/
```

用于开放集评估的 unknown 类测试划分：

```bash
entomokit split-csv --raw-image-csv data/images.csv \
    --unknown-test-sample-ratio 0.1 --known-test-sample-ratio 0.1 \
    --out-dir datasets/
```

丢弃样本过少的类别（count 模式）：

```bash
entomokit split-csv --raw-image-csv data/images.csv --mode count \
    --min-count-per-class 10 --out-dir datasets/
```

## 说明

- `split-csv` 以 CSV 为准，因此不适用 README 的目录输入/输出策略：
  [目录策略](../../README.cn.md#directory-policy)。日志与版本显示共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- `--min-count-per-class` 与 `--max-count-per-class` 只在 `--mode count` 下生效；默认的
  `ratio` 分支会忽略它们。
- 使用 `--copy-images` 时副本会平铺为文件名（见参数说明）：同一划分内 basename 相同的两个
  输入会静默互相覆盖。
- 复制出的目录不是可直接重定位的自包含数据集：划分 CSV 仍保留原始 `image` 值，因此
  `beetles/a.jpg` 复制到 `images/train/a.jpg` 后，CSV 里仍写作 `beetles/a.jpg`。把后续
  命令的图像根目录指向 `out_dir/images/` 会去找 `images/train/beetles/a.jpg` 而失败；
  请改写 CSV，或继续使用原始 `--images-dir`。
- 固定 `--seed` 时，同一输入 CSV 会得到相同的划分。
- `--out-dir` 非空时必须显式传入 `--overwrite`，否则直接报错退出；`split-csv` 没有
  `--resume`，也从不复用已有的输出目录。
- 中断处理：已安装 SIGINT 处理器，但划分逻辑从不读取关闭标志，因此第一次 `Ctrl+C` 只设置
  标志并打印提示，需第二次 `Ctrl+C` 才退出。
