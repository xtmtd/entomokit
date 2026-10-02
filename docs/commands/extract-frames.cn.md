# entomokit extract-frames
[English](extract-frames.md) | [中文](extract-frames.cn.md)

## 目的

从视频文件中提取静态帧。`--input-dir` 既接受视频目录，也接受单个视频文件，因此处理单个
视频时无需再准备临时目录。

## 用法

使用视频目录的最小调用：

```bash
entomokit extract-frames --input-dir videos/ --out-dir frames/
```

单个视频文件使用相同的参数：

```bash
entomokit extract-frames --input-dir clip.mp4 --out-dir frames/
```

## 参数

### `--input-dir`, `-i`

必填。无默认值。视频目录或单个视频文件路径。目录会被递归扫描以查找支持的视频；传入文件
路径时只处理该视频。

### `--out-dir`, `-o`

必填。无默认值。提取帧的输出目录。

### `--out-image-format`

可选。默认 `jpg`。可选值：`jpg`、`png`、`tif`。写出帧的图像格式。

### `--interval`

可选。默认 `1000`。提取间隔（毫秒）；`1000` 表示每秒一帧。

### `--start-time`

可选。默认 `0.0`。采样开始时间（秒）。

### `--end-time`

可选。无默认值；采样持续到每个视频结束。采样结束时间（秒）。

### `--max-frames`

可选。无默认值；写入时间范围内采样的全部帧。每个视频最多提取的帧数。

### `--threads`

可选。默认 `8`。帧提取使用的工作线程数。

### `--resume`

可选。默认关闭（不带取值的布尔开关）。跳过 `--out-dir` 中已存在的帧，继续上一次运行。由于
帧编号按位置生成，影响输出的参数（`--interval`、`--out-image-format`、`--max-frames`、
`--start-time`、`--end-time`）会记录在 `out-dir/.entomokit/extract-frames_params.json`，
必须与上一次运行一致；不一致会直接报错退出，而不是用不同内容覆盖已有帧。

### `--overwrite`

可选。默认关闭（不带取值的布尔开关）。删除 `--out-dir` 内容并重新开始。

### `--verbose`, `-v`

可选。默认关闭（不带取值的布尔开关）。启用详细日志。

### `--quiet`, `-q`

可选。默认关闭（不带取值的布尔开关）。抑制非错误输出与进度条。

## 输入

- `--input-dir` 是目录或单个视频文件。目录输入会被递归扫描。
- 支持的视频扩展名：`mp4`、`mov`、`avi`、`mkv`、`webm`、`flv`、`m4v`、`mpeg`、
  `mpg`、`wmv`、`3gp`、`ts`。
- 采样按 `--interval` 毫秒在 `--start-time` 到视频结束或 `--end-time` 之间进行。
- 读取视频需要 video 附加依赖（OpenCV）；参见 README 的安装依赖映射。

## 输出

- 帧写入 `out-dir/<视频的输入相对目录>/<视频词干>/`，因此不同子目录下的同名视频会得到
  独立的帧目录树；`--resume` 检查的就是该映射后的帧目录。
- 每个采样帧写出一张图像文件，格式由 `--out-image-format` 选择。
- 不会在 `--out-dir` 之外写入任何内容。

## 示例

从单个视频提取 5 秒到 30 秒的片段：

```bash
entomokit extract-frames --input-dir video.mp4 --out-dir frames/ \
    --start-time 5.0 --end-time 30.0
```

每秒采样两次并输出 PNG，同时把每个视频限制在 100 帧：

```bash
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --interval 500 --out-image-format png --max-frames 100
```

继续被中断的运行，并保持原提取参数：

```bash
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --interval 500 --out-image-format png --max-frames 100 --resume
```

## 说明

- 递归发现、镜像输出布局与输出目录安全属于共享规则：
  [目录策略](../../README.cn.md#directory-policy)。日志与版本显示同样共享：
  [通用行为](../../README.cn.md#common-behaviours)。
- 中断处理：已安装 SIGINT 处理器，但提取循环从不读取关闭标志，因此第一次 `Ctrl+C` 只设置
  标志并打印提示，需第二次 `Ctrl+C` 才退出；不承诺保留部分结果。
- `--out-dir` 非空时必须显式传入 `--resume`（继续）或 `--overwrite`（重新开始），否则
  直接报错退出。
- `--verbose` 与 `--quiet` 只改变日志量，不改变写出哪些帧。

## 版本注记

- `0.7.0`：目录输入改为递归扫描，帧改为镜像各视频的输入相对目录，而不再平铺在
  `--out-dir` 下。
