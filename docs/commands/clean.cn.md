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
`none` 全部保留，`md5` 去除解码像素完全相同的重复，`phash` 用连通分组去除感知相似，
`md5+phash` 先跑 `md5`，再对精确哈希组代表跑 `phash`。分组是传递的：`A~B` 与
`B~C` 会形成一组，即使 `A` 与 `C` 的距离超过阈值。

`md5` 是解码像素字节的 MD5，不是文件字节 MD5；不要拿它与 `md5sum` 比较。

### `--phash-threshold`

可选。默认 `5`。判定两张图为重复的最大感知哈希距离。

### `--pad-color`

可选。默认 `none`。可选值：`none`、`median`、`black`、`white`。用该填充色把非正方形
图像补成正方形；`none` 保持缩放后的尺寸。`median` 取边界像素 RGB 中位数。

### `--keep-exif`

可选。默认关闭（不带取值的布尔开关）。在输出图像中保留 EXIF 元数据。

### `--threads`

可选。默认 `12`。用于哈希输入和写出保留图像的工作线程数。

### `--resume`

可选。默认关闭（不带取值的布尔开关）。在非空输出目录中继续而不报错。匹配时优先使用
累积的 `clean_manifest.json["index"]`（源哈希与输入/输出路径），再对索引未匹配的输入
回退到对**所有**已有输出文件重新哈希。匹配到已有输出的输入报告为 `existing`，不重写。
子集 resume 会保留旧 index 行，后续输入仍可匹配。`dedup_mode="none"` 只按 index 的
`input_path` 匹配；没有 manifest 时不按镜像输出名猜测。没有可用 manifest 时，MD5
输出哈希可能漏掉 resize/转换后的输出，并会记录 warning。`--resume` 不重编码已有输出：
修改 `--out-short-size` 或 `--out-image-format` 需要 `--overwrite`。已 split 数据中
existing 在 train、重复输入在 test 时可能保留泄漏；需要重新平衡请用 `--overwrite` 或在
split 前 clean。resume 匹配按每个输入自己的 index/输出哈希进行：如果某个新输入只是通过
另一个新输入与已有输出传递相似，它会被保留并写出，而不会标记为 existing。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新开始。

### `--dry-run`

可选。默认关闭（不带取值的布尔开关）。只写决策报告和 `clean_manifest.json`，不创建
`cleaned_images/`、不写任何图片、不删除任何内容；与 `--overwrite` 互斥。manifest
原样复制上一次有效的累积 `index`，不添加“本应写出”的输出。dry-run 要求 `--out-dir` 为空或
传入 `--resume`；由于报告和日志会让目录非空，之后真正运行同一目录也需要 `--resume`。

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
- 被判为重复的文件不会写出；输入文件永不删除。
- 在 `--out-dir`（不是 `cleaned_images/` 内）写四个报告工件：
  - `filter_decisions.csv`：每个输入恰好一行，按输入相对路径排序。列为
    `input_path,status,reason,kept_path,kept_scope,matched_path,matched_scope,distance,group_id,group_size,md5,phash,output_path`。
  - `retained_images.csv`：`retained` 与 `existing` 行，列为
    `input_path,status,output_path`。它是 input-centric：多个输入可能映射一个
    existing 输出。per-class retention 必须按 `output_path`（实际保留的输出）统计，
    不能按 `input_path`；`n_existing` 统计输入匹配数，不是 unique output 数。
  - `dropped_images.csv`：`dropped` 行，列为
    `input_path,reason,matched_path,matched_scope,distance,group_id`。
  - `clean_manifest.json`：运行元数据、计数、报告路径、与
    `filter_decisions.csv` 行相同的 `files` 数组，以及累积 `index` 数组。
    `files` 只属于本次运行，不是 resume 索引。每个 `index` 行含
    `input_path`、`output_path`、`status`（`retained`/`existing`/`dropped`）、
    `md5`、`phash`；dropped 行映射到存活输出并保留自己的源哈希。`--dry-run`
    原样复制上一次有效 `index`，不添加“本应写出”的输出。
- 所有输入路径相对 `--input-dir`；输出路径与 existing 输出引用相对 `--out-dir`。
  `kept_scope`/`matched_scope` 标明路径是输入路径还是已有输出。
- 分组是确定性的。组是连通分量；代表按 held-out 优先级（`test` > `val` > `train`）
  选择，再取字典序最小的输入相对路径。没有可识别 split 段时只按路径顺序。
  `md5+phash` 先按 MD5 分组，再对 MD5 组代表按感知哈希分组。
- 报告中的 `md5` 是解码像素 MD5，不是文件字节 MD5。
- 最终 per-file `log.txt` 行在写出结束后由同一份最终 decision 行生成，因此不会与报告
  矛盾。
- 报告原子写入；`log.txt` 跨运行追加。

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
- `clean` 永不删除输入图片。`drop` 表示“不写入 `cleaned_images/`”；只有
  `--overwrite` 会删除输出目录内容。
- 报告表是结构化的事实来源；`log.txt` 保留可读的 per-file 行和汇总，但不是唯一记录。
- `--dry-run` 是只写报告预览。之后对同一 `--out-dir` 真正运行时要加 `--resume`。

## 版本注记

- `0.7.2`：去重改为确定性的连通分组，并写出 `filter_decisions.csv`、
  `retained_images.csv`、`dropped_images.csv`、带累积 resume `index` 的
  `clean_manifest.json`；新增 `--dry-run`；记录 per-file 路径与哈希；`--resume`
  优先匹配累积 index，并把匹配输入报告为 `existing`。传递分组可能比旧的成对行为
  合并更多图片。
- `0.7.0`：`clean` 默认递归扫描并把输出镜像到 `cleaned_images/`（移除了 `--recursive`
  与 `--flatten`），并新增 `--pad-color`。
