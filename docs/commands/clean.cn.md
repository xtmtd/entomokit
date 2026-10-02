# entomokit clean
[English](clean.md) | [中文](clean.cn.md)

## 目的

以统一的命名清洗、缩放并去重图像。目录输入始终递归扫描，输出镜像各输入的父目录，并重写
文件名（规范化词干、唯一 `_N` 后缀、`--out-image-format` 扩展名）。

## 用法

最小的 MD5 去重调用：

```bash
entomokit clean --input-dir images/raw/ --out-dir images/cleaned/
```

## 参数

### `--input-dir`

必填。无默认值。输入图像目录。

### `--out-dir`

必填。无默认值。输出目录；清洗后的图像写入其下的 `cleaned_images/`。

### `--out-short-size`

可选。默认 `512`。短边目标尺寸；使用 `-1` 保持原始尺寸。

### `--out-image-format`

可选。默认 `jpg`。可选值：`jpg`、`png`、`tif`。清洗后文件的输出格式。

### `--dedup-mode`

可选。默认 `md5`。可选值：`none`、`md5`、`phash`、`md5+phash`。去重策略：
`none` 全部保留，`md5` 去精确重复，`phash` 去感知相似，`md5+phash` 先跑 `md5`
再对存活文件跑 `phash`。

### `--phash-threshold`

可选。默认 `5`。判定两张图为重复的最大感知哈希距离。

### `--pad-color`

可选。默认 `none`。可选值：`none`、`median`、`black`、`white`。用该填充色把非正方形
图像补成正方形；`none` 保持缩放后的尺寸。`median` 取边界像素 RGB 中位数。

### `--keep-exif`

可选。默认关闭（不带取值的布尔开关）。在输出图像中保留 EXIF 元数据。

### `--threads`

可选。默认 `12`。图像处理使用的工作线程数。

### `--resume`

可选。默认关闭（不带取值的布尔开关）。在非空输出目录中继续而不报错。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新开始。

### `--verbose`, `-v`

可选。默认关闭（不带取值的布尔开关）。启用详细进度输出。

## 输入

- `--input-dir` 始终递归扫描；不再提供 `--recursive` 或 `--flatten`。
- 支持的输入格式跟随成像库：`jpg`、`jpeg`、`png`、`bmp`、`tif`、`tiff`、`webp`。

## 输出

- 清洗后的图像写入 `out-dir/cleaned_images/` 下，镜像各输入的**父目录**；文件名本身会被重写：
  词干规范化（非法字符与空白变成 `_`，首尾的 `.`/`_` 被去除，空词干变为 `untitled`），
  同一目录内重名时追加大小写不敏感的 `_1`、`_2` … 后缀，扩展名跟随
  `--out-image-format`。例如默认格式下 `in/beetles/a.png` 变为
  `out/cleaned_images/beetles/a.jpg`，同目录的第二张 `in/beetles/a.tif` 变为
  `beetles/a_1.jpg`；请不要用原文件名关联标签。
- 先缩放、后填充；`--pad-color none` 时跳过填充步骤。
- 被判为重复的文件不会写出；不生成报告文件。

## 示例

使用距离阈值 5 的感知哈希去重：

```bash
entomokit clean --input-dir images/ --out-dir cleaned/ \
    --dedup-mode phash --phash-threshold 5
```

短边缩放到 512 像素、转为 PNG，并用边界中位色补成正方形：

```bash
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size 512 --out-image-format png --pad-color median
```

保持原始尺寸与 EXIF 数据：

```bash
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size -1 --keep-exif
```

## 说明

- 递归发现、镜像布局与输出目录安全属于共享规则：
  [目录策略](../../README.cn.md#directory-policy)。日志与版本显示同样共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- 中断处理：已安装 SIGINT 处理器，但清洗循环从不读取关闭标志，因此第一次 `Ctrl+C` 只设置
  标志并打印提示，需第二次 `Ctrl+C` 才退出；不承诺保留部分结果。
- `--out-dir` 非空时必须显式传入 `--resume` 或 `--overwrite`，否则直接报错退出。
- `--dedup-mode md5+phash` 最严格：感知重复只在通过精确哈希筛选后的图像中查找。

## 版本注记

- `0.7.0`：`clean` 默认递归扫描并把输出镜像到 `cleaned_images/`（移除了 `--recursive`
  与 `--flatten`），并新增 `--pad-color`。
