# 昆虫图像数据集工具包 (EntomoKit)
[English](README.md) | **中文**

一个用于构建昆虫图像数据集的 Python 工具包。EntomoKit 提供统一的 `entomokit` 命令行，
覆盖视频抽帧、图像分割、形态测量、图像合成、图像清洗、数据增强、数据集划分、AutoMM 图像
分类与环境诊断，并附带面向 AI 助手的 `entomokit-workflow` skill。

## 工作流概览

整条流水线是可组合的，只需使用你的数据真正需要的步骤。

1. `extract-frames` — 把视频转为静态帧。
2. `segment` — 从图像中分割昆虫，可选输出标注。
3. `measure` — 可选：从分割掩码计算形态学指标。
4. `synthesize` — 可选：把 RGBA 抠图合成到新背景上。
5. `clean` — 缩放、填充并去重图像。
6. `augment` — 可选：扩充小样本训练集。
7. `split-csv` — 生成 train/val/test CSV。
8. `classify train` → `classify predict` / `classify evaluate` / `classify embed` /
   `classify cam` / `classify export-onnx`。

可以从任意一步开始：例如图像已标注时直接从 `clean` 进入 `classify train`，或只用
`segment` 提取掩码。AI 辅助运行遵循 skill 自身的引导策略，而不是这份清单。

## 系统要求与安装

- Python 3.9+
- Linux、macOS 或 Windows

```bash
git clone https://github.com/xtmtd/entomokit.git
cd entomokit
```

推荐安装到隔离环境：

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

等价方案：`python -m venv .venv && source .venv/bin/activate`，或
`conda create -n entomokit python=3.11 -y && conda activate entomokit`。直接装进全局环境
虽然可行，但可能与其它项目产生依赖冲突。

命令对应的附加依赖：

| 附加依赖 | 作用 |
|---|---|
| `.[video]` | `extract-frames`（OpenCV 视频解码） |
| `.[segmentation]` | SAM3 分割、SAM3 修复与合成标注输出 |
| `.[measurement]` | `measure`（形态学指标、骨架计算） |
| `.[synthesis]` | `synthesize`（多边形简化、COCO 标注） |
| `.[cleaning]` | `clean` 的感知哈希去重 |
| `.[augment]` | `augment`（albumentations） |
| `.[classify]` | `classify train/predict/evaluate/embed/cam/export-onnx`（AutoMM、timm、GradCAM、UMAP、ONNX） |
| `.[dev]` | pytest 与覆盖率 |

`measure` 与 `synthesize` 还会导入 `scikit-image`，而 `measurement` 与 `synthesis`
两个附加依赖都没有声明它；请同时安装声明了它的 `.[segmentation]`，或在这些附加依赖之外
显式加入 `scikit-image`。

可一次安装多项，例如完整的开发集合：

```bash
uv pip install --only-binary :all: stringzilla
pip install -e ".[dev,classify,segmentation,synthesis,measurement,video,cleaning,augment]"
```

在只提供二进制 wheel 的平台上，上面的 `stringzilla` 一行必须在安装 `.[augment]` 之前执行。
AutoMM 的官方安装说明见 https://auto.gluon.ai/stable/install.html。

## 快速开始

下面的例子需要 `video` 附加依赖，并把帧写入 `./frames`：

```bash
uv pip install -e ".[video]"
entomokit extract-frames --input-dir ./videos --out-dir ./frames
```

随后清洗帧并为分类构建划分：

```bash
entomokit clean --input-dir ./frames --out-dir ./cleaned
entomokit split-csv --raw-image-csv ./labels.csv --out-dir ./datasets
```

<a id="doctor-command"></a>
## doctor 命令

诊断环境与依赖就绪情况：

```bash
entomokit doctor
```

报告包含 Python 与可用设备（`cpu`、`cuda`、`mps`）、关键包版本与状态
（ok/missing/outdated），以及安装或升级建议。`classify` 附加依赖要求
`autogluon.multimodal>=1.5.0`，而 `doctor` 目前把 `1.4.0` 视为不过期，仅在低于该版本时
建议升级。`doctor` 没有用户可设置的参数。

<a id="update-command"></a>
## update 命令

检查 GitHub 上是否有新版本（读取 `main` 分支的 `version.txt`），并可选地安装。无论本地
git 历史如何都可用。

```bash
entomokit update           # 检查并询问
entomokit update --check   # 仅检查，不安装
entomokit update --yes     # 不再询问，直接安装
```

| 参数 | 描述 | 默认值 |
|---|---|---|
| `--check` | 只显示版本信息，不安装 | 否 |
| `--yes`, `-y` | 跳过确认提示 | 否 |

<a id="completion-command"></a>
## Shell 补全

为 bash、zsh 或 fish 生成静态补全脚本：

| Shell | 参数 | 安装路径 |
|---|---|---|
| `bash` | `--install` | `~/.local/share/bash-completion/completions/entomokit` |
| `zsh` | `--install` | `~/.zfunc/_entomokit` |
| `fish` | `--install` | `~/.config/fish/completions/entomokit.fish` |

不加 `--install` 时脚本直接打印到标准输出。使用 zsh 时，请确保 shell 配置包含
`fpath=(~/.zfunc $fpath)`，并用 `autoload -Uz compinit && compinit` 初始化补全。

<a id="directory-policy"></a>
## 目录输入与输出策略

本策略适用于所有面向目录的命令（`extract-frames`、`segment`、`measure`、`synthesize`、
`clean`、`augment`），以及 `classify embed`、`classify cam`、`classify predict` 的目录
发现：

- 面向目录的命令默认递归扫描；不再提供 `--recursive` 参数，也没有扁平化选项。
- 普通文件输出会镜像输入文件相对于输入根目录的**目录结构**，但部分命令还会重写文件名：
  `clean` 会规范化词干、保证同目录内唯一并套用 `--out-image-format`，`augment` 会追加
  `_augN`（按 `--multiply` 的位数补零：`1`–`9` 为 `_aug1`，`10`–`99` 为 `_aug01`，
  `100` 为 `_aug001`）。例如 `clean --input-dir in --out-dir out` 在默认格式下把 `in/beetles/a.png`
  输出为 `out/cleaned_images/beetles/a.jpg`，因此不同子目录下的同名文件保持独立，而同一目录
  下第二张 `a.tif` 会变成 `a_1.jpg`。
- `segment` 是刻意的例外：它不镜像输入路径，而是把所有图像平铺到 `images/` 下，并把每个
  输入相对路径编码为唯一的扁平样本 ID（可读词干加路径摘要，例如 `a__4cabcf2b3682`），
  注释写到各自的注释目录。完整的标准数据集布局（例如 Pascal VOC 的 `JPEGImages/`）由后续
  转换或划分步骤生成，而不是 `segment` 直接产出。
- CSV 驱动的命令（`split-csv`，以及显式传入 `--input-csv` 的分类命令）仍以 CSV 为准，
  不适用本策略。
- `--out-dir` 非空时会报错退出。提供 `--resume` 的命令可用它继续运行；没有
  `--resume` 的命令必须使用 `--overwrite` 才能重新开始。

<a id="common-behaviours"></a>
## 通用行为

- **日志**：有输出目录的命令会写入 `log.txt`：`extract-frames`、`segment`、`measure`、
  `synthesize`、`clean`、`augment`、`split-csv`、`classify predict`、
  `classify evaluate`、`classify cam` 与 `classify export-onnx` 写入 `out-dir/log.txt`；
  而 `classify train` 与 `classify embed` 写入 `out-dir/logs/log.txt`。`doctor`、
  `update` 与 `completion` 不写日志文件。文件头记录 `EntomoKit version:`（例如
  `0.7.1`）、`Commit:`、完整命令行、时间戳与所有参数值，随后是运行输出。`--verbose`
  启用 debug 级日志。
- **中断处理**：`measure` 与 `augment` 会在下一张图像的边界停止（逐图之前检查关闭标志）。
  `segment` 同样在图像之间检查，但只有串行路径会在完成当前图像后停止：并发 Otsu/GrabCut
  路径已提前提交全部任务，因此这些任务会先跑完。`synthesize` 只在准备任务时检查标志，
  因此已准备好的任务会全部执行完。`extract-frames`、`clean` 与 `split-csv` 安装了同一个
  处理器但从不读取该标志：第一次 `Ctrl+C` 只设置标志并打印提示，需第二次 `Ctrl+C` 才退出，
  且不承诺保留部分结果。分类命令与运维命令不安装该处理器。
- **设备选择**：`--device auto` 依次优先 CUDA、MPS、CPU。
- **版本号**：`entomokit --version`（或 `-v`）打印已安装版本。

<a id="assistant-integration"></a>
## AI 助手集成 (Skills)

`entomokit-workflow` skill 让 AI 助手（OpenCode、Claude Code、Codex 等）引导不熟悉命令行
的用户完成整条流水线：每次运行前用运行时 CLI schema 校验参数、逐步确认、给出错误恢复建议、
中断后恢复工作流，以及可选的教学演示模式。

```bash
mkdir -p ~/.config/opencode/skills && cp -r skills/entomokit-workflow ~/.config/opencode/skills/
mkdir -p ~/.claude/skills && cp -r skills/entomokit-workflow ~/.claude/skills/
mkdir -p ~/.codex/skills && cp -r skills/entomokit-workflow ~/.codex/skills/
```

详细的 skill 规则见 [SKILL.md](skills/entomokit-workflow/SKILL.md)，对话示例见
[teaching playbook](skills/entomokit-workflow/references/teaching-playbook.md#user-conversation-examples)。
skill 的引导式确认策略不是 CLI 的前置要求：下列每个命令在不使用 skill 时行为相同。

## 命令

功能命令的完整参考位于 `docs/commands/`；`doctor`、`update` 与 `completion` 已在上文说明。

| 命令 | 描述 | 参考 |
|---|---|---|
| `extract-frames` | 从视频文件中提取帧 | [参考](docs/commands/extract-frames.cn.md) |
| `segment` | 从图像中分割昆虫（SAM3、Otsu、GrabCut 与 bbox 裁剪模式） | [参考](docs/commands/segment.cn.md) |
| `measure` | 从分割掩码中计算形态学指标 | [参考](docs/commands/measure.cn.md) |
| `synthesize` | 将昆虫合成到背景图像上 | [参考](docs/commands/synthesize.cn.md) |
| `clean` | 清洗、缩放并去重图像 | [参考](docs/commands/clean.cn.md) |
| `augment` | 使用预设或自定义 albumentations 策略进行图像增强 | [参考](docs/commands/augment.cn.md) |
| `split-csv` | 将数据集划分为 train/val/test CSV 文件 | [参考](docs/commands/split-csv.cn.md) |
| `doctor` | 诊断环境与缺失依赖 | [参考](#doctor-command) |
| `update` | 检查更新并可选安装 GitHub 上的最新版本 | [参考](#update-command) |
| `completion` | 生成 shell 补全脚本 | [参考](#completion-command) |

<a id="classify-commands"></a>
### 分类命令

这些命令需要 `classify` 附加依赖。

| 命令 | 描述 | 参考 |
|---|---|---|
| `classify train` | 训练 AutoMM 图像分类器 | [参考](docs/commands/classify-train.cn.md) |
| `classify predict` | 运行推理（AutoGluon 或 ONNX） | [参考](docs/commands/classify-predict.cn.md) |
| `classify evaluate` | 评估模型性能并导出总体 + 类别级诊断结果 | [参考](docs/commands/classify-evaluate.cn.md) |
| `classify embed` | 提取嵌入向量 + UMAP + 质量指标 | [参考](docs/commands/classify-embed.cn.md) |
| `classify cam` | 生成 GradCAM 热力图 | [参考](docs/commands/classify-cam.cn.md) |
| `classify export-onnx` | 导出模型为 ONNX 格式 | [参考](docs/commands/classify-export-onnx.cn.md) |

## 项目结构

CLI 入口包位于 `entomokit/`，领域逻辑位于 `src/`；模块边界与架构约束见
[总设计文档](docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md)。

## 许可证

本项目基于 MIT 许可证发布 — 详见 LICENSE 文件。

## 联系方式

- 邮箱：`xtmtd.zf@gmail.com`

## 引用

如果 EntomoKit 对你的研究有帮助，请引用：

```bibtex
@software{entomokit2026,
  author = {Zhang, Feng},
  title = {EntomoKit: A Python Toolkit for Insect Image Dataset Construction and Classification},
  year = {2026},
  url = {https://github.com/xtmtd/entomokit}
}
```
