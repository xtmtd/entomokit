# entomokit augment
[English](augment.md) | [中文](augment.cn.md)

## 目的

使用 albumentations 预设或自定义策略文件对图像做增强。输入目录递归扫描，每个输出镜像输入的
相对路径。

## 用法

使用 light 预设，每张输入图生成一张增强图：

```bash
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/
```

## 参数

### `--input-dir`

必填。无默认值。包含图像的输入目录。支持格式为 `jpg`/`jpeg`、`png`、`bmp`、
`tif`/`tiff`、`webp`；输出格式与输入一致，不做转换。需要转换格式时使用
`entomokit clean --out-image-format`。

### `--out-dir`

必填。无默认值。增强图像与清单的输出目录。

### `--preset`

可选。无默认值；未提供且未指定 `--policy` 时使用 `light` 预设。命名预设：`light`、
`medium`、`heavy` 或 `safe-for-small-dataset`。

### `--policy`

可选。无默认值。自定义增强策略 JSON 文件路径；与 `--preset` 互斥。文件用 `json.loads`
读取，因此必须是含 `transforms` 数组的 JSON **对象**；数组每个元素是对象，其 `name` 为
一个 albumentations 类名，其余键为该类的构造参数，例如
`{"transforms": [{"name": "HorizontalFlip", "p": 0.5}]}`。

### `--seed`

可选。默认 `42`。可复现增强的随机种子。

### `--multiply`

可选。默认 `1`。每张输入图生成的增强副本数。

### `--resume`

可选。默认关闭（不带取值的布尔开关）。仅当某源图已有的 `_aug<数字>` 文件集合与当前
`--multiply` 集合完全相等时才跳过。集合不完整或数量不同会重新生成，上一次 `--multiply`
更大时留下的多余副本会被删除。没有 `--seed`/`--policy` 一致性校验，因此即使这些参数变了，
只要文件集合完全一致仍会跳过。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新开始。

## 输入

- `--input-dir` 会被递归扫描。
- `--preset` 与 `--policy` 互斥，最多只能提供一个。
- 自定义策略文件必须是形如
  `{"transforms": [{"name": "<albumentations 类名>", ...}]}` 的 JSON 对象。裸 JSON
  数组会被加载器拒绝；未知的变换名会在写出任何图像之前让运行失败。参数会原样传给对应的
  albumentations 类，因此必须与已安装版本的签名一致（例如 albumentations 2.x 的
  `RandomResizedCrop` 用 `size`，而不是 `height`/`width`）。

## 输出

```text
out_dir/
├── images/                 # 镜像各输入的子目录路径
│   └── <subdir>/source_aug1.png
└── augment_manifest.json
```

- 每个输出都保留 `_augN` 后缀（即使 `--multiply 1`），因此原始文件名永远不会被覆盖。
  序号按 `--multiply` 的位数补零：`1`–`9` 为 `_aug1`…`_aug9`，`10`–`99` 为
  `_aug01`…`_aug99`，`100` 为 `_aug001`。
- `augment_manifest.json` 记录 `preset`、`multiply`、`seed`、`images_processed` 与
  `augmented_images_created`。逐图的 `original`/`augmented` 路径记录只在内存中构建，当前
  **不会**写入该文件，因此不要把该清单当作路径映射使用。
- 输出格式始终与输入格式一致；`augment` 不做格式转换。

## 示例

heavy 预设、每张图 3 份副本并固定随机种子：

```bash
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/ \
    --preset heavy --multiply 3 --seed 123
```

自定义策略文件 `configs/augment_policy.json`：

```json
{"transforms": [
  {"name": "RandomResizedCrop", "size": [512, 512]},
  {"name": "HorizontalFlip", "p": 0.5}
]}
```

```bash
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/ \
    --policy configs/augment_policy.json
```

## 说明

- 递归发现、镜像布局与输出目录安全属于共享规则：
  [目录策略](../../README.cn.md#directory-policy)。日志、中断与版本显示同样共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- `--out-dir` 非空时必须显式传入 `--resume` 或 `--overwrite`，否则直接报错退出。
- `safe-for-small-dataset` 对极小训练集使用更保守的增强。
- 需要 `augment` 附加依赖；某些平台需先按 README 安装章节安装二进制 `stringzilla` wheel。

## 版本注记

- `0.7.0`：`augment` 递归扫描 `--input-dir` 并把各输入的相对路径镜像到 `images/` 下，
  不再平铺输出。
