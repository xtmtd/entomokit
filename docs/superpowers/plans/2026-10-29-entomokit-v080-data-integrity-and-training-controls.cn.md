# EntomoKit v0.8.0 数据完整性与训练控制实现计划

> **给执行代理：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，按任务逐项执行。步骤使用复选框（`- [ ]`）跟踪。

**目标：** 发布 EntomoKit `0.8.0`，交付可审计且确定性的 `clean` 去重、真正生效的 SAM3 模型置信度阈值、显式的 AutoMM 验证集输入，以及同步的中英文命令文档。

**架构：** 保持命令适配层轻薄。`clean` 改为三阶段流水线：并行计算全部输入的哈希，按输入相对路径顺序串行决定 keep/drop（使用连通分组），再按预先分配的确定性文件名并行写出。所有报告、最终 per-file 日志行和日志汇总都来自同一份最终 decision 行。`segment` 通过现有的 `SAM3Wrapper -> Sam3Predictor -> Sam3Processor` 构造链传递两个独立阈值，不修改 vendored 模型源码。`classify train` 把可选的验证 CSV 传给 AutoMM 已有的 `fit(tuning_data=...)` 参数，并在日志记录验证集来源。文档遵循 `0.7.1` 的命令文档规范。

**技术栈：** Python 3.9+、argparse、pathlib、`csv`、`json`、`dataclasses`、Pillow、imagehash、NumPy、`concurrent.futures`、PyTorch/AutoGluon、pytest、Markdown。

**Spec：** `docs/superpowers/specs/2026-10-29-entomokit-v080-data-integrity-and-training-controls-design.md` 是配套设计文档。plan 与 spec 一起批准、一起修订；在操作者批准这一对文档前，二者都不是实现权威。

**状态：** `0.8.0` 实现完成并已提交；等待操作者批准推送、打标签或发布。

## 已确定决策

- 发布目标严格为 `0.8.0`。
- 四个 issue 用一份合并 plan 和一份合并 spec 覆盖；plan 与 spec 各有英文文件和同步中文文件。
- Release notes 写入每个受影响命令文档的 `Version Notes` / `版本注记`，最新在前。不新增 changelog。
- `clean` 采用 phyloAI 的报告文件命名约定：`retained_images.csv`、`dropped_images.csv`、`filter_decisions.csv`，以及命令级 manifest `clean_manifest.json`。这里只参考命名约定，不照搬 phyloAI 的列集合；phyloAI 写 `result.json`，而 `clean` 使用命令级 manifest。
- `clean_manifest.json` 含本次运行的 `files` 数组（等于 `filter_decisions.csv` 行）和供 `--resume` 使用的累积 `index` 数组。`index` 每个已知输入路径一行，包括映射到存活输出的 dropped 重复项，跨 invocation 保留，不随 per-run `files` 覆盖而丢失。dry-run 原样复制上一次的 `index`。
- `filter_decisions.csv` 每个输入文件恰好一行，按输入相对 POSIX 路径排序。`retained_images.csv` 和 `dropped_images.csv` 是它的过滤视图。
- `retained_images.csv` 同时包含 `status="retained"`（写出新输出）和 `status="existing"`（`--resume` 下已有输出）两类行，使 resume 运行仍能看到 retained 输入。它是 input-centric：多个输入可能映射到同一个 existing 输出。per-class retention 必须按 `output_path`（实际保留的输出）统计，而不是 `input_path`；`n_existing` 统计 input 匹配数，不是 unique output 数。`dropped_images.csv` 只含 `status="dropped"` 行。
- 报告记录决策所用的哈希值：`md5` 是 `compute_md5` 返回的像素内容 MD5，`phash` 是零填充的 64 位十六进制感知哈希。未计算哈希时两列均为空。
- `--dry-run` 写出同样的报告与 manifest，但不创建 `cleaned_images/`、不删除任何东西；与 `--overwrite` 互斥。dry-run 要求 `--out-dir` 为空或传入 `--resume`；dry-run 之后，真正写入同一目录也需要 `--resume`。
- 去重分组是连通分量。代表按 held-out 优先级 `test > val > train` 选择，再按输入相对路径字典序；没有可识别 split 段的分组只用路径顺序。
- `md5+phash` 先做精确 MD5 分组，再对存活 MD5 代表做 pHash 分组。非代表行若与任一组成员 MD5 相同，`reason=duplicate_md5`；否则为 `duplicate_phash`。
- 代表写盘失败时，先删除半成品目标文件，再按代表顺序提升下一个组员。所有提升按 `group_id` 顺序串行收尾，使输出名和 decision 行不依赖 future 完成顺序。
- `--sam3-confidence-threshold` 控制 SAM3 内部模型阈值；`--post-confidence-threshold` 控制现有后置过滤；`--confidence-threshold` 保留为后置过滤的弃用别名。两个后置过滤选项（`--post-confidence-threshold` 与 `--confidence-threshold`）默认都是 `0.0`。`0.0` 在别名解析中表示“未设置”：post 为 `0.0` 且 legacy 大于零时使用 legacy，并记录一条 info。两个都非零且不同时，`--post-confidence-threshold` 优先并 warning。
- 仅当解析后的 post 阈值实际生效且严格位于零与 SAM3 阈值之间（`0 < post < sam3`）时警告。默认 `post=0.0` 表示“不做后置过滤”，不得警告。
- no-mask 原因同时报告 SAM3 模型阈值和后置阈值，而不是只报告后置阈值。
- `--val-csv` 可选，使用与 `--train-csv` 相同的 `--images-dir`，并作为 `tuning_data` 传给 AutoMM。验证集重叠按解析后、大小写归一化的文件路径检查。验证 CSV 的文件级 SHA256 在 `fit()` 之前写入日志。
- 两个相邻的确定性问题纳入本次：输出文件名确定性分配，以及 MD5 注册泄漏。
- `--resume` 先通过累积的 `clean_manifest.json["index"]`（源哈希与输入/输出路径）匹配已有输出，再对索引未匹配的输入回退到对**所有**已有输出文件重新哈希。这是必需的，因为 `compute_md5` 是像素内容 MD5，resize 或格式转换会改变输出像素。
- 写出阶段失败时，error 行保留组信息。从未进入分组的哈希/解码错误使用空 `group_id` 和 `group_size=0`；写失败保留 `group_id` 和 `group_size`。
- 报告文件和 manifest 采用原子写入（临时文件后 `os.replace`），完成后再追加日志汇总。
- 不要在缺少 `from __future__ import annotations` 的模块里引入 PEP 604 的 `X | Y` 注解；`setup.py` 仍声明 Python 3.9。使用 `Optional[...]` 或添加 future import。
- 不需要修改 workflow 策略。只更新 `skills/entomokit-workflow/references/command-profiles.md` 中关于 `--val-csv` 和 clean 报告/dry-run 的事实性说明。

## 全局约束

- 把 `version.txt`、`entomokit/_version.py`、`entomokit/main.py` 的 fallback、`setup.py` 和版本测试统一设为 `0.8.0`。
- `clean` 永不删除输入图片。`drop` 表示“不写入 `cleaned_images/`”；只有 `--overwrite` 会删除输出目录内容。
- `clean` 的决策不得依赖线程调度、哈希完成顺序、文件系统遍历顺序或字典插入顺序。
- 报告中的所有输入路径都是相对 `--input-dir` 的 POSIX 路径。所有输出路径和 existing 输出引用都是相对 `--out-dir` 的 POSIX 路径。
- `filter_decisions.csv` 行、`clean_manifest.json["files"]` 和最终 per-file 日志行都按 `input_path` 排序。
- `clean` 报告与日志汇总必须来自同一份最终 decision 行列表；不得维护第二套计数逻辑。
- 在写出/提升阶段结束后，从最终行生成 per-file 日志行。不要保留旧的 worker 内散文行，因为提升会使其失真。
- `clean --dry-run` 不得创建 `cleaned_images/`、写图片或删除现有输出；拒绝 `--dry-run --overwrite`。非空 `--out-dir` 上的 dry-run 需要 `--resume`。
- `clean --resume` 保留已有输出文件，永不重写或删除它们。累积的 `clean_manifest.json["index"]` 是主要匹配来源；输出文件哈希是索引未匹配输入的回退方案。
- `clean_manifest.json["index"]` 是累积的，后续只覆盖部分输入时也不能丢失。`files` 只属于本次运行，绝不能当作 resume index。每次真实运行都要写 `index`，即使 `files` 为空。dropped 重复项的 index 行映射到存活代表输出，并保留自己的源哈希，使传递组成员仍可 resume。输出文件缺失的 index 行要剪掉并 warning。
- manifest 只有 schema 合法且未损坏时才可用。忽略不支持的 `schema_version`、损坏 JSON、非法行，并 warning。`output_path` resolve 后必须位于 `--out-dir` 内；缺失或不可读输出忽略并 warning。
- dry-run manifest 只可用于其复制过来的 `index`；绝不能把 dry-run 的 `files` 当成已有输出。
- `dedup_mode="none"` 没有可用 index/manifest 时，不得按镜像输出路径匹配；保留输入并 warning 说明精确 resume 需要 manifest。
- 图片写出失败时，必须在任何提升尝试前删除半成品目标。`cleaned_images/` 中不得留下半写文件。
- 报告文件和 `clean_manifest.json` 原子写入。只有在报告完成后才追加日志汇总。
- 不得修改 SAM3 vendored 文件 `src/sam3/model/sam3_image_processor.py`；新阈值通过现有构造函数传入。
- Otsu 与 GrabCut 的 score 仍为 `1.0`；`--sam3-confidence-threshold` 对它们无效，`--post-confidence-threshold` 在文档范围 `0..1` 内也没有实际效果。
- 未提供 `--val-csv` 时仍使用 AutoMM 内部随机切分；不实现新的切分算法。
- `--val-csv` 在 CSV 缺失、格式错误、为空、引用不存在的图片、验证集 image 重复、验证 label 不在训练集中、或与训练路径重叠时，必须在 `fit()` 之前失败。
- 不要把训练 CSV 的既有契约收紧到“列存在 + 路径解析”之外。完整的 existence/duplicate/label/overlap 检查只适用于验证 CSV。
- 命令文档必须同时更新 `docs/commands/<command>.md` 和 `docs/commands/<command>.cn.md`；`tests/test_docs.py` 会强制检查两种语言的 option 覆盖。
- 不编辑历史 plan 或已完成 spec；只更新当前 master design `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`，并在其中写架构级契约，不复制参数表。
- 不新增依赖、changelog、文档生成器或第二套 CLI 注册表。
- 不修改 `skills/entomokit-workflow` 的 workflow 策略；只修正 Task 5 列出的事实性命令说明。
- 未经操作者另行批准，不提交、不打标签、不推送、不发布。
- Python 3.9 兼容性：`src/cleaning/processor.py` 和 `src/segmentation/processor.py` 没有 `from __future__ import annotations`，必须使用 `Optional[...]`，不要把 `Path | None` 示例复制进这两个模块。

## 评审重点

- 同一输入集在 `--threads=1`、`8`、`32` 以及重复运行时，产生相同且同序的 `filter_decisions.csv` 行、group ID、代表、输出名和输出文件集合。
- pHash 链 `A~B`、`B~C`、`A!~C` 形成一组而不是两对；这个有意的行为变化写入 `clean` 的 Version Notes。
- 跨 `train/`、`val/`、`test/` 的重复组保留 `test/`，其次 `val/`，最后 `train/`；没有 split 段时保留字典序最小的相对路径。
- MD5 与 pHash 匹配引用都是相对路径而非 basename；不同类别目录下的同名文件保持可区分。
- 代表写盘失败会先删除半成品输出，再提升下一个组员；失败的 MD5 注册不能让后面真实存在的图片被误删。
- 提升后，最终 per-file 日志行、`clean_manifest.json` 计数和 `filter_decisions.csv` 三者一致。
- A+B → 只 A resume → 只 B resume，B 通过累积 index 保持 `existing`，不会写出 `_1`。
- 上一次是 md5-only manifest、当前是 phash 模式时，缺少可用 phash index 行的输出会走输出哈希回退。
- 传递组 `A~B~C` 只保留 C 时，后续只 resume B 仍能通过 B 自己的 index 行（status `dropped`）匹配到 C 的输出。
- `dedup_mode="none"` 且没有 manifest 时，同 stem、不同扩展名的输入不会被误标 existing。
- `retained_images.csv` 的 per-class retention 按 `output_path` 统计，不按 `input_path`；多个输入可能映射一个输出，`n_existing` 统计输入数而非 unique output 数。
- 提升后重算 `reason`、`distance`、`kept_scope`、`matched_scope`，不留下指向失败代表的引用或距离。
- `--dry-run` 写出与“无写失败的真实运行”相同的 decision 行和“本应写出”的 `output_path`；只有 manifest 的 `dry_run` 标志和 `output_files.cleaned_images=null` 不同，且不创建任何图片文件。
- 空输入目录仍然写出零行报告和 `n_input=0` 的合法 manifest。
- `--resume` 能通过累积 index 的源哈希匹配“输出已被 resize/转换”的输入。没有可用 index/manifest 时，回退逻辑在 MD5 输出哈希不可靠时警告。
- `--resume` 以 `matched_scope=existing_output` 和 output-relative 路径报告已有输出匹配；已有输出永不重写。
- 写出阶段失败的 error 行保留 `group_id`/`group_size`；只有哈希/解码错误的组字段为空。
- SAM3 模型阈值确实传到 `Sam3Predictor`；设置 `--sam3-confidence-threshold 0.2` 能保留 score 在 `(0.2, 0.5]` 的 mask，而旧硬编码 `0.5` 会丢掉它们。
- 默认 SAM3 运行（`sam3=0.5`、`post=0.0`）不产生 no-op 警告；显式 `0 < post < sam3` 会警告但不报错。
- no-mask 原因同时写出两个阈值；Otsu/GrabCut 输出不受 `--sam3-confidence-threshold` 影响。
- 仅在提供 `--val-csv` 时，AutoMM 收到 `tuning_data`；否则 fit kwargs 中不存在该键。
- 训练/验证重叠（经 `Path.resolve()` 和大小写归一化）、验证行重复、空验证集、图片缺失、未知验证 label 都在 `fit()` 前失败。
- `out-dir/logs/log.txt` 在调用 `fit()` 之前出现验证 CSV 路径、SHA256、图片数和 overlap 结果。
- 两种语言文档更新后 `tests/test_docs.py` 通过。
- 实现能在 Python 3.9 语法下导入和编译（没有 future import 的文件中不得出现 PEP 604 union）。

## 文件地图

| 区域 | 文件 | 责任 |
|---|---|---|
| clean 决策核心 | `src/cleaning/processor.py` | 哈希阶段、分组、代表选择、确定性命名、写出/提升、报告生成、resume 匹配 |
| clean CLI | `entomokit/clean.py` | `--dry-run`、输出目录处理、汇总打印 |
| CLI schema | `entomokit/cli_schema.py` | 识别自定义 argparse 类型上的 `_value_type`，使 `--threads` 在 runtime schema 中仍是 integer，新阈值选项仍是 float |
| clean 测试 | `tests/test_clean_dedup_manifest.py`（新）、`tests/test_clean_resume.py`（新）、`tests/test_clean_cli_manifest.py`（新）、`tests/test_clean_recursive.py`、`tests/test_clean_padding.py`、`tests/test_resume_flags.py` | 确定性、分组、报告、dry-run、resume、写失败提升、CLI namespace |
| augment manifest | `src/augment/service.py`、`tests/test_augment_manifest.py`（新） | 把已有 per-augmentation 记录写入 manifest |
| SAM3 wrapper | `src/sam3_wrapper.py`、`tests/test_sam3_wrapper.py` | 把模型阈值传给 `Sam3Predictor` |
| 分割 | `src/segmentation/processor.py`、`src/metadata.py`、`entomokit/segment.py`、`tests/test_segmentation.py`、`tests/test_metadata.py`、`tests/test_annotation_formats.py`、`tests/test_segment_cli_thresholds.py`（新）、`tests/test_resume_flags.py` | 两个阈值、范围校验、兼容别名、日志、COCO per-mask score、CLI namespace |
| 分类 CLI | `entomokit/classify/train.py` | 新增 `--val-csv` 并传递 |
| 分类 trainer | `src/classification/trainer.py` | 验证集加载、检查、`tuning_data`、来源日志、assets 元数据回填 |
| 分类测试 | `tests/test_classify_trainer.py`、`tests/test_resume_flags.py` | `tuning_data`、overlap、重复行、label 检查、日志哈希、CLI namespace |
| 命令文档 | `docs/commands/clean.md`、`clean.cn.md`、`segment.md`、`segment.cn.md`、`classify-train.md`、`classify-train.cn.md`、`augment.md`、`augment.cn.md` | 双语参数、输出、说明和 Version Notes |
| 补全 | `entomokit/completion.py`、`tests/test_main_cli.py` | 从生成的 shell 补全中省略 deprecated action，但保留在 help/CLI |
| Skill 说明 | `skills/entomokit-workflow/references/command-profiles.md` | `--val-csv` 与 clean 报告/dry-run 的事实性说明；不改 workflow 策略 |
| Master design | `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` | clean 报告、segment 阈值、train 验证集的当前架构级契约 |
| 版本 | `version.txt`、`entomokit/_version.py`、`entomokit/main.py`、`setup.py`、`tests/test_package_version.py`、`tests/test_main_cli.py`、`README.md`、`README.cn.md` | 设置并验证 `0.8.0` |

## 接口

### Clean decision 行

```python
from typing import Optional

@dataclass
class CleanDecision:
    input_path: str           # 相对 --input-dir 的 POSIX 路径
    status: str               # retained | dropped | existing | error
    reason: str               # "" | duplicate_md5 | duplicate_phash |
                              # duplicate_existing_output | unidentified_image |
                              # hash_error:<message> | write_error:<message> |
                              # group_write_failed
    kept_path: str            # 存活代表；只有不存在存活代表时为空
    kept_scope: str           # input | existing_output | ""
    matched_path: str         # 最近匹配成员；无匹配时为空
    matched_scope: str        # input | existing_output | ""
    distance: Optional[int]   # pHash Hamming 距离；MD5/none 为 None
    group_id: str             # g0001、g0002，…；只有哈希/解码错误为空
    group_size: int           # 组成员数，包含 existing outputs；只有哈希/解码错误为 0
    md5: str                  # 像素内容 MD5 十六进制；未计算时为空
    phash: str                # 零填充 64 位 pHash 十六进制；未计算时为空
    output_path: str          # 相对 --out-dir 的 POSIX 路径；未写出时为空
```

### Clean 报告文件

报告写到 `log_path` 的父目录（命令的 `--out-dir`），不写进 `cleaned_images/`。

`filter_decisions.csv` 列顺序如下，每个输入一行，按 `input_path` 排序：

```text
input_path,status,reason,kept_path,kept_scope,matched_path,matched_scope,distance,group_id,group_size,md5,phash,output_path
```

`retained_images.csv` 列，按 `input_path` 排序；行包含 `status="retained"` 和 `status="existing"`：

```text
input_path,status,output_path
```

`dropped_images.csv` 列，按 `input_path` 排序；行只包含 `status="dropped"`：

```text
input_path,reason,matched_path,matched_scope,distance,group_id
```

`clean_manifest.json` 键：

```json
{
  "schema_version": 1,
  "command": "clean",
  "dedup_mode": "md5",
  "phash_threshold": 5,
  "threads": 12,
  "dry_run": false,
  "n_input": 0,
  "n_retained": 0,
  "n_dropped": 0,
  "n_existing": 0,
  "n_errors": 0,
  "output_files": {
    "retained_images": "retained_images.csv",
    "dropped_images": "dropped_images.csv",
    "filter_decisions": "filter_decisions.csv",
    "cleaned_images": "cleaned_images/"
  },
  "files": [],
  "index": [
    {
      "input_path": "beetles/a.jpg",
      "output_path": "cleaned_images/beetles/a.jpg",
      "status": "retained",
      "md5": "pixel-content-md5-hex",
      "phash": "0000000000000000"
    }
  ]
}
```

`files` 是本次运行的 `filter_decisions.csv` 行，`distance` 为 `int | null`，`group_size` 为整数。`index` 是累积 resume 索引：每个已知输入路径一行，以 `input_path` 为键，指向规范输出路径并记录源哈希。`status` 可为 `retained`、`existing` 或 `dropped`；dropped 行保留自己的源哈希，但指向存活代表输出。真实运行会 merge `index`；`--dry-run` 原样复制上一次的 `index`。`--dry-run` 下 `output_files.cleaned_images` 为 `null`。

### Clean 状态与 reason

- `retained`：输入是组代表，会写出新输出（`--dry-run` 下为“本应写出”）。`kept_path` 是输入自身，`kept_scope="input"`，`output_path` 是新输出。
- `dropped`：普通输入组中的非代表重复项。与任一组成员 MD5 相同时 `reason=duplicate_md5`，否则 `duplicate_phash`。`kept_path` 是组代表，`kept_scope="input"`；`matched_path` 是最近成员，`matched_scope="input"`；`output_path` 为空。
- `existing`：`--resume` 下输入与已有输出匹配，不写新文件。`reason="duplicate_existing_output"`，`kept_path`/`matched_path`/`output_path` 都是规范 existing 输出，`kept_scope="existing_output"`，`matched_scope="existing_output"`。按 pHash 匹配时 `distance` 是到规范 existing 输出的 Hamming 距离；按 MD5/`none` 匹配时为空。`md5`/`phash` 描述当前输入，不是 existing 输出。`group_id` 按规范输出路径顺序分配；`group_size` 是映射到该规范输出的当前输入数再加一（existing 输出本身）。
- `error`，哈希/解码失败：`reason` 为 `unidentified_image` 或 `hash_error:<message>`；`group_id` 为空、`group_size=0`、所有路径字段为空。
- `error`，写出失败但组内仍有存活成员：`reason="write_error:<message>"`；保留 `group_id`/`group_size`；`kept_path`/`kept_scope` 来自存活代表；`matched_path`/`matched_scope` 为空；该失败成员的 `output_path` 为空。
- `error`，全组耗尽：`reason="group_write_failed"`；保留 `group_id`/`group_size`；`kept_path`/`kept_scope`/`matched_path`/`matched_scope`/`output_path` 全为空。
- `kept_path` 和 `matched_path` 除指向 existing 输出外都相对 `--input-dir`；指向 existing 输出时相对 `--out-dir`，scope 为 `existing_output`。`output_path` 始终相对 `--out-dir`。
- `retained_images.csv` 是 input-centric：每个输入一行，多个 `existing` 行可能指向同一个输出。per-class retention 按 `output_path` 统计；`n_existing` 统计输入数，不是 unique output 数。

### `ImageCleaner`

```python
class ImageCleaner:
    def __init__(..., dry_run: bool = False, resume: bool = False) -> None: ...

    def process_directory(
        self,
        input_dir: Optional[str] = None,
        log_path: Optional[str] = "log.txt",
    ) -> dict:
        """返回计数与报告路径。"""
```

`process_directory` 在提供 `log_path` 时派生 `report_dir = Path(log_path).parent`，否则使用 `self.output_dir.parent`；在 `resume=True` 时从 `report_dir/clean_manifest.json` 加载累积 `index`，并把报告写在那里。返回值保留现有键（`total`、`processed`、`skipped`、`errors`）以兼容旧调用，并新增 `retained`、`dropped`、`existing`、`reports`。`processed` 对应 `retained`；`skipped` 对应 `dropped`。

### `segment` CLI

```text
--sam3-confidence-threshold FLOAT   默认 0.5，范围 0..1，仅 SAM3/SAM3-bbox
--post-confidence-threshold FLOAT   默认 0.0，范围 0..1，后置过滤
--confidence-threshold FLOAT        默认 0.0，后置过滤的弃用别名
```

`run()` 中的解析：

```python
post = args.post_confidence_threshold
legacy = args.confidence_threshold
if post == 0.0 and legacy > 0.0:
    post = legacy
    logger.info(
        "--post-confidence-threshold is 0.0 (unset); using deprecated "
        "--confidence-threshold=%s",
        legacy,
    )
elif post > 0.0 and legacy > 0.0 and post != legacy:
    logger.warning("--post-confidence-threshold wins over deprecated --confidence-threshold")
```

SAM3 的有效过滤阈值为 `max(sam3_confidence_threshold, post_confidence_threshold)`；模型端是 `>`，后置端是 `>=`。post 阈值等于或低于模型阈值时是 no-op；仅在 `0 < post < sam3` 时警告。`0.0` 表示“未设置”；显式 `--post-confidence-threshold 0.0` 不能覆盖非零 legacy。

### `classify train` CLI

```text
--val-csv PATH   可选；含 image、label 列的 CSV；image 相对 --images-dir
```

提供时把 `tuning_data=val_df` 传给 `fit()`，不再依赖 AutoMM 内部切分。

## Task 1：构建确定性的 clean 决策流水线与报告

**文件：**
- 修改：`src/cleaning/processor.py`
- 新建：`tests/test_clean_dedup_manifest.py`
- 新建：`tests/test_clean_resume.py`
- 修改：`tests/test_clean_recursive.py`、`tests/test_clean_padding.py`，适配新构造/返回契约并更新已过时的提升测试。

**接口：**
- 消费：`iter_files`、`compute_md5`、`compute_phash`、`phash_to_int`、`hamming_distance`、`clean_filename`、`ensure_unique_prefix`。
- 产出：每个输入一条 `CleanDecision`、确定性 group ID、确定性输出名、`retained_images.csv`、`dropped_images.csv`、`filter_decisions.csv`、`clean_manifest.json`。

- [ ] **Step 1：写失败的决策契约测试**

新建 `tests/test_clean_dedup_manifest.py`，用嵌套目录中的小 RGB 图片作为 fixture，至少覆盖：

```python
def test_filter_decisions_has_one_row_per_input(tmp_path):
    # 三个输入，dedup_mode="none"
    # 断言 len(rows) == 3，status 全为 "retained"
    # 断言行按 input_path 排序
    # 断言 retained_images.csv 有 3 行，dropped_images.csv 无数据行


def test_md5_drop_uses_relative_input_and_match_paths(tmp_path):
    # train/beetles/same.png 与 test/beetles/same.png 像素内容完全相同
    # （fixture 在两个位置写入同一张图）
    # 断言 drop 行 input_path == "train/beetles/same.png"
    # 断言 matched_path == "test/beetles/same.png"（held-out-first）
    # 断言 matched_scope == "input"
    # 断言两行的 md5 列是同一个非空像素 MD5


def test_phash_group_is_transitive(tmp_path):
    # 用受控 pHash 值或 monkeypatch compute_phash 构造 A~B、B~C 但 A!~C
    # 断言一个 group_id 的 group_size == 3，一个 retained，两个 dropped
    # 断言每个 dropped 行记录 reason 和 distance


def test_group_representative_prefers_test_over_val_over_train(tmp_path):
    # 同一图片位于 train/、val/、test/
    # 断言 retained 的 input_path 以 "test/" 开头


def test_group_representative_falls_back_to_path_order(tmp_path):
    # 同一图片位于 b/ 与 a/，无 split 段
    # 断言 retained 的 input_path == "a/..."


def test_output_names_are_assigned_in_sorted_path_order(tmp_path):
    # 同目录 a.jpg 与 a.png，dedup_mode="none"
    # 断言字典序靠前的输入得到 "a.jpg"，另一个得到 "a_1.jpg"


def test_clean_is_deterministic_across_threads(tmp_path):
    # 同一 fixture 分别用 threads=1、8、32 输出到不同 out dir
    # 在某一线程数下再重复运行一次
    # 精确比较 filter_decisions.csv 行及其顺序
    # 忽略 run-level "threads" 字段后比较 manifest files/counts
    # 比较相对输出文件集合


def test_empty_input_writes_zero_row_reports(tmp_path):
    # 空输入目录
    # 断言四个报告工件都存在，n_input == 0，files == []，index == []


def test_report_counts_satisfy_the_status_invariant(tmp_path):
    # 混合 fixture：retained、dropped 和一张非法图片
    # 断言 n_input == n_retained + n_dropped + n_existing + n_errors
```

另外补这些契约断言：

- `test_manifest_contains_cumulative_index_with_dropped_rows`：manifest 含 `index`，传递组的 dropped 行指向存活输出。
- `test_existing_md5_match_has_empty_distance`：`md5+phash` resume 由 MD5 匹配时 `distance` 为空。
- `test_promotion_recomputes_reason_and_distance`：提升后 dropped 行从 `duplicate_md5` 改为 `duplicate_phash`（当存活成员只是 pHash 相似时）。
- `test_resume_falls_back_to_output_hash_for_indexed_output`：已索引输出仍参与 decoded-output 哈希回退。
- `test_resume_rejects_truncated_manifest_output`：无法完整解码的已索引输出被忽略，输入重新生成。

补旧 MD5 注册泄漏和写失败提升的回归测试：

```python
def test_write_failure_promotes_next_group_member(tmp_path, monkeypatch):
    # 两张相同图片；让第一个代表的 save 失败一次
    # 断言第二张被写出，第一张记为 error，
    # 且失败目标位置没有半成品文件


def test_md5_failure_does_not_drop_later_duplicate(tmp_path, monkeypatch):
    # 三张相同图片；让第一个 save 失败
    # 断言剩余图片中有一张被写出，不会因陈旧 MD5 注册被静默丢弃


def test_group_write_failure_preserves_group_membership(tmp_path, monkeypatch):
    # 三张相同图片；让所有 save 都失败
    # 断言失败代表是 write_error，其余成员是 group_write_failed；
    # 所有成员都保留 group_id 和 group_size
```

新建 `tests/test_clean_resume.py`：

```python
def test_resume_uses_previous_manifest_source_hash(tmp_path):
    # 第一次 clean 使用 resize 输出，然后对同一输入 resume
    # 断言输入为 status="existing"，不创建 _1 输出


def test_resume_reports_existing_output_relative_paths(tmp_path):
    # 断言 matched_scope/kept_scope 为 "existing_output"，路径相对 out-dir


def test_resume_ignores_missing_manifest_output_and_warns(tmp_path, caplog):
    # 删除上一次 manifest 引用的输出
    # 断言该行被忽略并记录 warning


def test_resume_none_mode_matches_previous_manifest_without_hashes(tmp_path):
    # dedup_mode="none"；第二次运行按 manifest index input_path 标记 existing


def test_resume_none_without_manifest_does_not_match_mirrored_path(tmp_path, caplog):
    # 第一次 a.jpg；第二次 a.png 且无 manifest
    # 断言 a.png 被 retained，不会按 a.jpg 标 existing，并记录 warning


def test_resume_without_manifest_falls_back_to_output_hashing(tmp_path, caplog):
    # 删除 manifest 后重跑；断言行回退行为，
    # 并在 resize 后 MD5 匹配可能不可靠时发出 warning


def test_resume_cumulative_index_survives_subset_runs(tmp_path):
    # 先 A+B，再只 resume A，再只 resume B
    # 断言 B 为 status="existing"，且不创建 B_1 输出


def test_resume_transitive_group_member_matches_by_own_hash(tmp_path):
    # A~B~C 传递组，只保留 C
    # 后续只 resume B；B 的 dropped index 行映射到 C 的输出
    # 断言 B 为 status="existing"，不被重写


def test_resume_fallback_hashes_output_without_current_mode_hash(tmp_path):
    # 第一次 md5-only，第二次 phash resume；manifest 无可用 phash 索引
    # 断言已有输出通过输出哈希回退被匹配


def test_resume_ignores_dry_run_or_unsupported_manifest(tmp_path, caplog):
    # dry_run=true manifest 与 schema_version=999 manifest 被忽略并 warning


def test_resume_rejects_manifest_output_path_outside_out_dir(tmp_path, caplog):
    # manifest output_path="../escape.jpg"、绝对路径，以及 cleaned_images/ 之外的
    # "other.jpg" 都被忽略并 warning
```

- [ ] **Step 2：运行新测试并确认失败**

```bash
python -m pytest tests/test_clean_dedup_manifest.py tests/test_clean_resume.py -q
```

预期：出现 import/attribute 失败或报告缺失断言，而不是无关的环境错误。

- [ ] **Step 3：把哈希与决策拆开**

把 `process_one_to` 重构为显式阶段：

```python
def _hash_images(files, input_dir, dedup_mode, threads) -> list[ImageRecord]:
    """按排序后的路径顺序读取每个输入，计算所需哈希。"""

def _decide_groups(records, dedup_mode, phash_threshold, existing_outputs) -> list[CleanDecision]:
    """分组、选代表，并生成 keep/drop/existing/error 行。"""

def _assign_output_paths(decisions, output_dir, out_image_format) -> None:
    """按输入路径顺序分配确定性输出名。"""

def _write_retained(decisions, output_dir, out_short_size, out_image_format, keep_exif, pad_color, threads) -> None:
    """写出 retained 图片；写失败时提升组员。"""
```

规则：

- 用 `ThreadPoolExecutor(max_workers=threads)` 哈希，但结果按输入路径存放；worker 内不做决策。
- 每个打开的图像都确定性关闭（`with Image.open(...)` 或 `img.close()`）。
- `dedup_mode="none"` 不需要哈希；所有输入 retained。
- `md5` 按 `compute_md5` 返回的像素内容 MD5 分组。代码和文档都要说明这不是文件字节 MD5。
- `phash` 对所有两两 `distance <= phash_threshold` 的边做 union-find。
- `md5+phash` 先按 MD5 分组，再对 MD5 组代表做 pHash union-find，并合并成员集合。
- 哈希或解码错误记为 `status="error"`，`group_id=""`、`group_size=0`，永不成为代表。
- 仅 `phash`：缺失或非法 pHash 记为 `status="error"`，`reason="hash_error:phash"`。
- `md5+phash`：MD5 成功但 pHash 失败时，保留为 MD5-only 候选；不要让该图片失败。
- `--resume` 的已有输出在 `process_directory` 中加载，而不是在 `__init__`。删除 `__init__` 中的 `_prepopulate_hashes` 调用；已有输出发现成为 `process_directory` 的一部分。
- 只有 `report_dir/clean_manifest.json` schema 合法且未损坏时才加载。忽略不支持的 `schema_version`、损坏 JSON、非法行，以及绝对路径、非 POSIX 相对路径或 resolve 后位于 `--out-dir` 或 `cleaned_images/` 输出目录之外的 `output_path`，并 warning。dry-run manifest 只贡献其 `index`，绝不把它的 `files` 当作输出。
- manifest 的累积 `index` 是 resume 主来源。`index` 行以 `input_path` 为键，携带 `output_path`、`status`、`md5`、`phash`。按当前 dedup 模式匹配：
  - `md5`：源 MD5 精确相等。
  - `phash`：`--phash-threshold` 内最近的源 pHash，按路径打破平局。
  - `md5+phash`：先精确源 MD5，否则阈值内最近源 pHash。
  - `none`：`input_path` 精确相等。
- 输出文件缺失、不可读或位于 `--out-dir`/`cleaned_images/` 之外的 index 行忽略并 warning。多个 index 行映射同一输出时，用字典序最小的 output-relative 路径作为规范 existing 输出。
- 对索引未匹配的输入，从**所有**已有输出图片构建输出哈希映射（不只是 manifest 中不存在的路径），按当前模式匹配输出哈希。这是为了覆盖“旧 manifest 有路径但缺少当前模式哈希”的情况（例如 md5-only manifest + 当前 phash）。
- 没有可用 index 时，如果 MD5 输出哈希匹配可能漏掉 resize/有损转换后的输出，记录 warning；需要精确 resume 时建议全新运行或 `--overwrite`。
- `--resume --dedup-mode none`：只按 `index`/`files` 的 `input_path` 匹配。没有可用 manifest 时，不得按镜像输出路径匹配；保留输入并 warn 说明精确 resume 需要 manifest。
- 真实运行后，把本次 `retained`/`existing` 行以及有存活代表的 `dropped` 行 merge 进 `index`，同一 `input_path` 的旧行被替换。dropped 行的 index 行映射到存活代表输出，并保留 dropped 输入自己的源哈希，使传递组成员仍可 resume；有存活代表的写失败行同样处理。只移除哈希/解码错误和全组耗尽错误的行，因为它们没有可用输出或哈希。当前运行未出现的输入，其旧行在输出仍存在时保留；输出缺失则剪掉并 warning。dry-run 原样复制上一次的 `index`，不添加“本应写出”的输出。
- `--resume` 下，当前输入按当前模式直接与累积 index/输出哈希匹配。匹配输入变成 `status="existing"`；未匹配输入进入普通分组。不得删除或重写已有输出。若一个输入匹配多个 existing 输出，按当前模式的 tie-break 选规范输出（MD5：路径；pHash：距离再路径）。
- pHash 分组在 `0.8.0` 接受 O(n²) 两两比较；不要引入 BK-tree 或新依赖。

- [ ] **Step 4：实现代表与匹配选择**

确定性规则：

```python
def split_rank(relative_path: str) -> int:
    """test=0，val=1，train=2，无可识别 split=3。"""
```

- split 识别大小写不敏感，并匹配精确路径段：`train`；`val`、`valid`、`validation`；`test`。如果一个路径含多个，使用优先级最高的段。
- 不含 existing 输出的分组：选 `(split_rank, relative_path)` 最小的输入成员为代表。
- 若一个输入匹配多个 existing 输出，按当前模式的 tie-break 选规范 `kept_path`/`output_path`（MD5：路径；pHash：距离再路径）；匹配输入变成 `existing`，该组不重写。
- 对每个非代表输入选最近组员：
  - `md5`：代表，`distance=None`。
  - `phash`：最小 Hamming 距离；同距按相对路径。
  - `md5+phash`：若成员与任一组成员 MD5 相同，使用 `duplicate_md5`、`distance=None`；代表同 MD5 时匹配代表，否则匹配同 MD5 成员中路径字典序最小者。否则使用 `duplicate_phash`，取最小 pHash 距离并按路径打破平局。
- `matched_path` 是最近成员，不一定是代表；这样即使传递闭包导致到代表距离超过阈值，也能记录真正把它拉进组的边。
- 代表确定后，按 `(split_rank, representative path)` 排序分配 `group_id`；existing 匹配使用 output-relative 规范路径。
- `retained` 行使用 `kept_path=input_path`、`kept_scope="input"`。
- `dropped` 行使用 `kept_scope="input"` 和 `matched_scope="input"`；它们只出现在普通输入组中。
- `existing` 行使用 `status="existing"`、`reason="duplicate_existing_output"`，`kept_path`/`matched_path`/`output_path` 指向规范 existing 输出，两个 scope 都是 `existing_output`。
- 没有写出任何东西的 `error` 行路径字段为空。有存活代表的写失败行保留 `kept_path`/`kept_scope`，`matched_path`/`matched_scope` 为空。
- `dedup_mode="none"` 只允许按 manifest/index 的 `input_path` 匹配；不得从镜像输出路径推断 existing。

- [ ] **Step 5：实现确定性输出命名**

- 按 `input_path` 排序 retained 输入决策。
- 按该顺序分配名字，复用 `clean_filename` 和同目录唯一后缀规则。
- `--resume` 的已有输出名也加入 used-name 集合。
- 不允许 worker 选名或按完成顺序追加后缀。
- 最终路径以相对 `--out-dir` 的形式存入 `output_path`（例如 `cleaned_images/beetles/a.jpg`）。

- [ ] **Step 6：实现写出与失败提升**

- 并行写 retained 图片到预先分配路径。
- 写失败时，尽力删除半成品目标（`dst.unlink(missing_ok=True)`），再把该成员标为 `error`，`reason=write_error:<message>`，保留 `group_id`/`group_size`。
- 收集所有并行写结果后再提升。按 `group_id` 顺序、组内代表顺序串行处理提升。
- 提升下一个组员，由确定性命名器分配输出名并写出。被提升的 `dropped` 行变成 `retained`，reason 清空，写入新的 `output_path`。
- 重复直到某成员写出成功或全组耗尽。
- 全组耗尽时，剩余成员都记为 `error`，`reason=group_write_failed`，保留 `group_id`/`group_size`，清空路径字段。
- 提升后更新组内所有成员的 `kept_path`/`kept_scope`/`matched_path`/`matched_scope`，指向存活代表；只要 `matched_path` 变化就重算 `reason` 和 `distance`。不得留下指向失败成员的引用或距离。
- 任何阶段都不写输入文件、不删除输入文件。

- [ ] **Step 7：从同一份行列表生成报告**

增加一个正常路径和 dry-run 共用的报告写入函数：

```python
def write_clean_reports(
    decisions: list[CleanDecision],
    report_dir: Path,
    log_path: Optional[Path],
    summary: dict,
    dry_run: bool,
) -> dict:
    """原子写入 retained_images.csv、dropped_images.csv、filter_decisions.csv、
    clean_manifest.json；log_path 非空时再追加 per-file 行和汇总。"""
```

要求：

- `filter_decisions.csv` 每个输入恰好一行，包含 error，按 `input_path` 排序。
- `retained_images.csv` 含 `status="retained"` 和 `status="existing"` 行，按 `input_path` 排序。
- `dropped_images.csv` 只含 `status="dropped"` 行，按 `input_path` 排序。
- `clean_manifest.json` 含 run-level 字段、与当前 CSV 行相同的 `files` 数组，以及供 resume 使用的累积 `index` 数组。
- `n_input`、`n_retained`、`n_dropped`、`n_existing`、`n_errors` 全部从同一 `decisions` 列表计算；测试断言 `n_input == n_retained + n_dropped + n_existing + n_errors`。
- `n_retained` 只计新输出（`status="retained"`）；`n_existing` 计 input 匹配（`status="existing"`），不是 unique existing output。因此 `retained_images.csv` 有 `n_retained + n_existing` 行，`output_path` 可能重复。
- 按 Step 3 使用提升结束后的最终行合并上一次有效 manifest 的 `index`；dry-run 原样复制上一次 `index`，不添加“本应写出”的输出。
- 每个 CSV/JSON 先写到 `report_dir` 中的临时文件，再 `os.replace` 到位。
- 报告完成后追加相同计数的日志 summary 和报告路径。
- 在提升结束后，从 `decisions` 按 `input_path` 顺序输出最终 per-file 日志行，例如：
  `retained: <input> -> <output>`、`existing: <input> -> <output>`、
  `dropped: <input> -> <matched> (<reason>, distance=<n|->)`、`error: <input> (<reason>)`。
- 删除旧的 worker 内重复/错误散文行，保证日志与报告不会互相矛盾。
- 如果 `log_path` 为 `None`，仍写报告，但跳过日志行和 summary。

- [ ] **Step 8：运行 clean 聚焦测试和现有测试**

```bash
python -m pytest tests/test_clean_dedup_manifest.py tests/test_clean_resume.py tests/test_clean_recursive.py tests/test_clean_padding.py -q
```

预期：PASS。报告、确定性与 resume 测试必须先通过，再进入 CLI。

完成注记：在操作者批准整个变更集前，工作区改动保持未提交。

## Task 2：新增 `clean --dry-run` 与 augment manifest 文件行

**文件：**
- 修改：`entomokit/clean.py`
- 修改：`src/augment/service.py`
- 新建：`tests/test_clean_cli_manifest.py`
- 新建：`tests/test_augment_manifest.py`
- 修改：`tests/test_resume_flags.py`

**接口：**
- 消费：`ImageCleaner(dry_run=..., resume=...)`、`write_clean_reports`。
- 产出：CLI 级 dry-run 契约，以及 `augment_manifest.json` 中每个增强副本的 `files` 数组。

- [ ] **Step 1：写失败的 CLI 与 manifest 测试**

```python
def test_clean_parser_accepts_dry_run_default_false():
    # parser.parse_args(["clean", "--input-dir", "in", "--out-dir", "out"]).dry_run is False

def test_clean_dry_run_writes_reports_without_images(tmp_path):
    # 运行 clean --dry-run
    # 断言 filter_decisions.csv、retained_images.csv、dropped_images.csv、
    #        clean_manifest.json 存在；断言没有 cleaned_images/ 目录
    # 断言 manifest["output_files"]["cleaned_images"] is None

def test_clean_dry_run_rejects_overwrite(tmp_path, capsys):
    # 断言 SystemExit code 1 和清晰错误信息

def test_clean_dry_run_requires_resume_on_nonempty_out_dir(tmp_path):
    # 创建 out/log.txt，不带 --resume 运行 --dry-run，断言 SystemExit

def test_clean_real_run_after_dry_run_uses_resume(tmp_path):
    # 先 dry-run 到 out/，再带 --resume 真正运行，断言图片写出

def test_clean_dry_run_preserves_previous_index(tmp_path):
    # 真实运行，再 --resume dry-run，再 --resume 真实运行
    # 断言之前的输出仍为 existing，不写重复文件

def test_augment_manifest_includes_per_file_rows(tmp_path, monkeypatch):
    # 对两张图运行 run_augment
    # 断言 manifest["files"] 有两个条目，含 original/augmented/copy_index
```

- [ ] **Step 2：实现 dry-run CLI 契约**

在 `entomokit/clean.py`：

- 增加 `--dry-run`，`action="store_true"`。
- 为 `--threads` 增加正整数类型（拒绝 `0` 和负数）。
- 设置 `_positive_int._value_type = "int"`，并更新 `entomokit/cli_schema.py` 的 `_infer_value_type`，让它识别自定义 argparse 类型上的 `_value_type`。这样 `--threads` 在 runtime schema 与 parameter guard 中仍是 integer。
- 在 `check_output_dir` 之前拒绝 `--dry-run --overwrite`；打印清晰错误并 `sys.exit(1)`。
- 保留 `check_output_dir(out_dir, args.resume, args.overwrite)`。
- 保留 `save_log(out_dir, args)`。
- 仅当 `not args.dry_run` 时创建 `out_dir / "cleaned_images"`。
- `ImageCleaner.__init__` 在 `dry_run=True` 时必须跳过 `self.output_dir.mkdir(...)`；`process_directory` 在 dry-run 下不得创建输出子目录。
- dry-run 下对每个“本应写出”的图片执行解码和内存中的 resize/pad，但跳过 `mkdir`/`save`；解码错误按真实写出的错误记录。
- 把 `dry_run=args.dry_run` 和 `resume=args.resume` 传给 `ImageCleaner`。
- 用新计数和报告位置打印 summary。dry-run 使用 `would write N new image(s)`，真实运行使用 `wrote N new image(s)`；单复数必须与计数一致。

`--dry-run` 本身不得删除或重写已有图片。`--resume` 下已有输出可参与匹配，但不写新图片。dry-run 原样复制上一次有效 `index` 到新 manifest，不添加“本应写出”的输出；之后真正写入同一目录需要 `--resume`，并可读取复制过来的 `index`。文档要说明这个流程。

- [ ] **Step 3：把 augment 文件行写入 manifest**

`run_augment` 已经构建 `augmented_images`，直接加入 manifest：

```python
manifest = {
    ...现有 run-level 字段...,
    "files": augmented_images,
}
```

不改变 `augmented_images` 的构建方式和现有计数字段。加测试断言该键存在、是列表，并保留 `original`、`augmented`、`copy_index`。文档说明 `original` 相对 `--input-dir`，`augmented` 相对 `out-dir/images/`。

- [ ] **Step 4：更新手工 CLI namespace**

在 `tests/test_resume_flags.py` 的 `_clean_args` 中加入：

```python
dry_run=False,
```

- [ ] **Step 5：运行聚焦测试**

```bash
python -m pytest tests/test_clean_cli_manifest.py tests/test_augment_manifest.py tests/test_resume_flags.py -q
```

预期：PASS。

## Task 3：拆分 SAM3 模型阈值与后置过滤

**文件：**
- 修改：`src/sam3_wrapper.py`
- 修改：`src/segmentation/processor.py`
- 修改：`entomokit/segment.py`
- 修改：`entomokit/completion.py`
- 修改：`tests/test_sam3_wrapper.py`
- 修改：`tests/test_segmentation.py`
- 新建：`tests/test_segment_cli_thresholds.py`
- 修改：`tests/test_resume_flags.py`
- 修改：`tests/test_main_cli.py`

**接口：**
- `SAM3Wrapper(..., confidence_threshold: float = 0.5)`。
- `SegmentationProcessor(..., sam3_confidence_threshold: float = 0.5, post_confidence_threshold: Optional[float] = None, confidence_threshold: float = 0.0)`。
- CLI 选项 `--sam3-confidence-threshold`、`--post-confidence-threshold`，以及弃用别名 `--confidence-threshold`。

说明：旧模块 `src/segmentation.py` 没有被包或 CLI 导入（`src/segmentation/__init__.py` 导出的是 `src.segmentation.processor`）。本次不要修改它。

- [ ] **Step 1：写失败的阈值测试**

```python
def test_sam3_wrapper_passes_confidence_threshold_to_predictor(monkeypatch):
    # mock src.sam3.build_sam.Sam3Predictor
    # 构造 SAM3Wrapper(..., confidence_threshold=0.2)
    # 断言 Sam3Predictor 收到 confidence_threshold=0.2


def test_processor_passes_sam3_threshold_to_wrapper(monkeypatch):
    # mock SAM3Wrapper，断言收到 confidence_threshold=0.2


def test_post_filter_uses_post_confidence_threshold():
    # scores [0.95, 0.45]；post 阈值 0.7；断言只保留一个 mask


def test_default_thresholds_do_not_warn(caplog):
    # sam3=0.5，post=0.0；断言没有 no-op 警告


def test_post_threshold_below_sam3_threshold_warns(caplog):
    # sam3=0.5，post=0.3；断言有 warning，无异常


def test_no_mask_reason_names_both_thresholds(caplog):
    # sam3=0.2，post=0.0，wrapper 返回空 mask；断言原因提到 0.2


def test_otsu_ignores_sam3_threshold():
    # 同一 Otsu fixture 在 sam3=0.5 与 sam3=0.2 下输出相同
```

还要覆盖 CLI 解析与别名解析：

```python
def test_segment_parser_threshold_defaults_and_ranges():
    # --sam3-confidence-threshold 默认 0.5
    # --post-confidence-threshold 默认 0.0
    # --confidence-threshold 默认 0.0
    # 三者都拒绝 -0.01 和 1.01

def test_legacy_confidence_threshold_is_used_when_post_is_zero(caplog):
    # post=0.0，legacy=0.7 -> 解析后的 post=0.7，并记录“0.0 未设置”的 info

def test_post_wins_when_both_are_nonzero():
    # post=0.6，legacy=0.7 -> 解析后的 post=0.6，并记录 warning
```

- [ ] **Step 2：让阈值穿过 wrapper**

在 `SAM3Wrapper.__init__`：

- 新增 `confidence_threshold: float = 0.5`。
- 校验 `0 <= confidence_threshold <= 1`。
- 保存该值。
- 传给 `Sam3Predictor(self.model, confidence_threshold=confidence_threshold)`。

不要编辑 `src/sam3/model/sam3_image_processor.py`；它的构造函数和 `_forward_grounding` 已经使用 `confidence_threshold`。

- [ ] **Step 3：给 `SegmentationProcessor` 加两个阈值**

- 保留现有 `confidence_threshold` 参数作为兼容的后置过滤 fallback。
- 为不破坏位置参数调用，把 `sam3_confidence_threshold` 和 `post_confidence_threshold` 加到构造函数末尾。
- `post_confidence_threshold` 使用 `Optional[float]`；本模块不要使用 `float | None`。
- 解析：

```python
self.sam3_confidence_threshold = sam3_confidence_threshold
self.post_confidence_threshold = (
    post_confidence_threshold
    if post_confidence_threshold is not None
    else confidence_threshold
)
self.confidence_threshold = self.post_confidence_threshold  # 兼容属性
```

- 用 `confidence_threshold=self.sam3_confidence_threshold` 构造 wrapper。
- 后置过滤比较改用 `self.post_confidence_threshold`。
- `_write_computation` 中 no-mask 原因同时写出两个阈值：

```python
reason = (
    "No masks passed the effective confidence threshold "
    f"(sam3={self.sam3_confidence_threshold}, post={self.post_confidence_threshold})"
)
```

  SAM3 方法且任一阈值大于零时使用该消息；两个阈值都为零时保留 `No masks returned by <method>`。
- 在 processor 内也校验两个阈值都在 `0..1`，不只依赖 CLI。
- SAM3 方法且 `0 < post_confidence_threshold < sam3_confidence_threshold` 时，每个 processor 只 `logger.warning(...)` 一次，不要每张图一次。`post == 0.0` 时不警告。
- Otsu/GrabCut 方法且 `sam3_confidence_threshold != 0.5` 时，记录“SAM3 阈值被忽略”。默认值时不记录。
- 给 `COCOMetadataManager.add_annotation` 增加可选 `score`，并在 `_write_computation` 的 COCO unified 与 separate 两条路径传入每个 mask 的 score。SAM3 写模型 score，Otsu/GrabCut 写 `1.0`；VOC/YOLO 不写 score。补 `tests/test_metadata.py` 与 `tests/test_annotation_formats.py` 测试。

- [ ] **Step 4：增加 CLI 选项与范围校验**

在 `entomokit/segment.py`：

- 增加 `_unit_float` argparse 类型，拒绝 `[0, 1]` 之外的值；设置 `_unit_float._value_type = "float"`，使 runtime schema 与 parameter guard 仍把该选项类型标为 float。
- 增加 `--sam3-confidence-threshold`，默认 `0.5`。
- 增加 `--post-confidence-threshold`，默认 `0.0`。
- 保留 `--confidence-threshold`，默认 `0.0`，help 标注为弃用别名。
- 给弃用 action 设置 `_deprecated = True`。`entomokit/completion.py` 跳过带该标记的 action，因此别名仍在 CLI/help 中可用，但不会出现在生成的 shell 补全里。
- 在 `run()` 中按 Interfaces 一节解析 post 值，包括 post 为 `0.0`（未设置）而使用 legacy 时记录 info。
- 把两个值传给 `SegmentationProcessor`。
- 日志记录 method、SAM3 阈值、post 阈值和任何 no-op 警告。
- 非 SAM3 方法且 SAM3 阈值不是默认值时，记录“ignored”。

- [ ] **Step 5：更新手工 CLI namespace**

在 `tests/test_resume_flags.py` 的 `_segment_args` 中加入：

```python
sam3_confidence_threshold=0.5,
post_confidence_threshold=0.0,
```

- [ ] **Step 6：运行阈值测试**

```bash
python -m pytest tests/test_sam3_wrapper.py tests/test_segmentation.py tests/test_segment_cli_thresholds.py tests/test_resume_flags.py tests/test_main_cli.py -q
```

预期：PASS。

## Task 4：新增 `classify train --val-csv`

**文件：**
- 修改：`entomokit/classify/train.py`
- 修改：`src/classification/trainer.py`
- 修改：`tests/test_classify_trainer.py`
- 修改：`tests/test_resume_flags.py`

**接口：**
- CLI：`--val-csv PATH`（可选）。
- `train(..., val_csv: Optional[Path] = None)`。
- 仅在提供验证 CSV 时设置 `fit_kwargs["tuning_data"] = val_df`。

- [ ] **Step 1：写失败的验证测试**

```python
def test_train_passes_tuning_data_when_val_csv_supplied(monkeypatch, tmp_path):
    # fake AutoGluon fit 捕获 kwargs
    # 断言 kwargs["tuning_data"] 是含已解析图片路径的 DataFrame

def test_train_omits_tuning_data_without_val_csv(monkeypatch, tmp_path):
    # 断言 fit kwargs 中没有 "tuning_data"

def test_train_rejects_train_val_path_overlap(monkeypatch, tmp_path):
    # train.csv 与 val.csv 含同一张图，包括 "./img.jpg" 与 "img.jpg"
    # 断言 fit 调用前抛 ValueError

def test_train_rejects_duplicate_validation_rows(...)
def test_train_rejects_missing_validation_images(...)
def test_train_rejects_empty_validation_csv(...)
def test_train_rejects_validation_labels_absent_from_training(...)
def test_train_logs_validation_csv_path_hash_and_count(...)
def test_train_logs_validation_before_fit(...)
def test_train_accepts_absolute_validation_image_paths(...)
def test_train_does_not_require_training_images_to_exist(...)
def test_train_maps_validation_labels_to_training_representation(...)
def test_train_rejects_blank_validation_image(...)
def test_train_rejects_blank_validation_label(...)
def test_classify_train_parser_accepts_val_csv(...)
```

- [ ] **Step 2：加 CLI 选项并传递**

在 `entomokit/classify/train.py`：

- 增加 `--val-csv`，无默认值。
- 把 `val_csv=Path(args.val_csv) if args.val_csv else None` 传给 `train()`。
- 不改变 `split-csv`，不新增切分逻辑。

- [ ] **Step 3：在 trainer 中加载并校验验证 CSV**

在 `src/classification/trainer.py` 增加小 helper：

```python
def _load_training_frame(
    csv_path: Path,
    images_dir: Path,
    role: str,
    *,
    validate_files: bool,
) -> pd.DataFrame:
    """读取 image,label CSV，解析图片路径并校验列。

    train 使用 validate_files=False：只要求列存在并解析路径。
    validation 使用 validate_files=True：还要求非空、图片存在、无重复行。
    label 子集与 train/val overlap 在 train() 拿到两个 DataFrame 后检查。
    """
```

规则：

- train 与 validation CSV 都必须有 `image`、`label` 列。
- 相对值按 `images_dir / image` 解析；绝对值直接接受。
- 用 `Path.resolve()` 和 `os.path.normcase(...)` 归一化后做 overlap 比较。
- 拒绝空 validation CSV。
- helper 负责拒绝 validation 内解析后重复的 `image` 行。
- `train()` 在拿到两个 DataFrame 后拒绝 validation 中出现训练集没有的 label。
- `train()` 在拿到两个 DataFrame 后拒绝同时出现在 train 与 validation 的解析路径；报告前几个重叠项。
- 本版本不拒绝训练集缺失图片或重复训练行；保持现有训练 CSV 行为。
- 增加 `import hashlib` 用于 validation CSV SHA256。
- 在 `predictor.fit()` 之前失败。

- [ ] **Step 4：传 `tuning_data` 并记录验证集来源**

- 提供 `val_csv` 时：
  - 设置 `fit_kwargs["tuning_data"] = val_df`。
  - 用 `hashlib.sha256(path.read_bytes()).hexdigest()` 计算验证 CSV 文件 SHA256。
  - 必要时创建 `out_dir / "logs"`，然后在 `predictor.fit()` **之前**追加到 `out_dir / "logs" / "log.txt"`：

```text
Validation CSV: <path>
Validation CSV SHA256: <sha256>
Validation images: <n>
Overlap check: passed
```

- 未提供 `val_csv` 时：
  - 不加 `tuning_data`。
  - 在 `fit()` 前追加：

```text
Validation: AutoMM internal random split (no --val-csv)
```

- 保持 `train.processed.csv` 不变；本次不新增 per-image validation manifest 或 per-image validation hash。
- `assets.json` 缺失、损坏或写入失败只 warning，不得让训练失败。
- 现有 NVML 重试路径使用同一个 `fit_kwargs`，包含 `tuning_data`。
- `predictor.fit()` 返回后调用 `_backfill_model_assets(model_dir, predictor, elapsed_seconds)`，用 `predictor._learner` 的 `best_score` 与 `total_train_time` 回填模型 `assets.json`；后者缺失时用实际耗时。AutoMM 在 top-k 保存时先写 assets，之后才设置这两个属性。不要删除 key，因为 `MultiModalPredictor.load()` 会直接索引它们；补一个 fake `_learner` 测试。

- [ ] **Step 5：更新手工 CLI namespace**

在 `tests/test_resume_flags.py` 的 `_train_args` 中加入 `val_csv=None`。

- [ ] **Step 6：运行分类测试**

```bash
python -m pytest tests/test_classify_trainer.py tests/test_resume_flags.py -q
```

预期：PASS，包括现有 seed、resume、NVML 和 optimizer 测试。

## Task 5：更新双语文档、版本与发布验证

**文件：**
- 修改：`docs/commands/clean.md`、`docs/commands/clean.cn.md`
- 修改：`docs/commands/segment.md`、`docs/commands/segment.cn.md`
- 修改：`docs/commands/classify-train.md`、`docs/commands/classify-train.cn.md`
- 修改：`docs/commands/augment.md`、`docs/commands/augment.cn.md`
- 修改：`skills/entomokit-workflow/references/command-profiles.md`
- 修改：`docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`
- 修改：`version.txt`、`entomokit/_version.py`、`entomokit/main.py`、`setup.py`
- 修改：`tests/test_package_version.py`、`tests/test_main_cli.py`
- 修改：`README.md`、`README.cn.md` 的版本示例与 `0.8.0` 参数变化表

**接口：**
- 消费：Task 1–4 已实现的 CLI 契约。
- 产出：同步的中英文命令文档、当前 master-design 契约、修正后的 skill 说明，以及一致的 `0.8.0` 版本。

- [ ] **Step 1：更新 clean 命令文档**

在 `docs/commands/clean.md` 和 `docs/commands/clean.cn.md`：

- 在 Parameters 下增加 `--dry-run` 的 H3 块。
- 把“被判为重复的文件不会写出；不生成报告文件”替换为四个报告工件。
- 记录 `retained_images.csv`、`dropped_images.csv`、`filter_decisions.csv`、`clean_manifest.json` 的准确列，并说明 `retained_images.csv` 同时包含 `retained` 与 `existing` 行。说明它是 input-centric：多个输入可能映射一个输出；per-class retention 必须按 `output_path` 统计；`n_existing` 统计输入数而非 unique output 数。
- 明确记录 `clean_manifest.json["index"]`：累积、每个已知输入一行、`status` 为 retained/existing/dropped、含 `input_path`/`output_path`/`md5`/`phash`；`files` 只属于本次运行，不是 resume index；dry-run 复制上一次 index。
- 说明报告中的 `md5` 是像素内容 MD5，不是文件字节 MD5；对 `existing` 行，`md5`/`phash` 描述当前输入，而不是 existing 输出。
- 记录确定性分组、连通分量、`test > val > train` 代表优先级，以及无 split 段时的路径顺序 fallback。
- 记录 `--resume` 行为：优先使用累积 `clean_manifest.json["index"]`，索引未匹配输入回退到输出哈希，`existing` 状态，output-relative 路径，不重写；说明没有 manifest 时 MD5 在 resize 后可能匹配不上。说明子集 resume 不能丢失旧 index，`dedup_mode="none"` 没有 manifest 时不按镜像名匹配。
- 说明 `--resume` 不会重编码已有输出；修改 `--out-short-size`/`--out-image-format` 需要 `--overwrite`。已 split 数据中 existing 在 train、重复输入在 test 时，resume 可能保留泄漏；需要重新平衡请用 `--overwrite` 或在 split 前 clean。
- 记录 `--dry-run` 流程：只写报告、不创建 `cleaned_images/`、非空 out-dir 需要 `--resume`、复制上一次 `index`，之后真正运行同一目录也需要 `--resume`。
- 说明 per-file 日志行在写出结束后由最终 decision 生成。
- 在 Version Notes 最上方加 `0.8.0` 条目，描述确定性传递分组去重、报告工件与哈希、`--dry-run`、累积 resume index，以及 `--resume` 的 existing 语义变化。

- [ ] **Step 2：更新 segment 命令文档**

两种语言：

- 为 `--sam3-confidence-threshold`、`--post-confidence-threshold` 和弃用别名 `--confidence-threshold` 增加 H3 块，默认值分别为 `0.5`、`0.0`、`0.0`。
- 记录 `0..1` 范围，以及有效阈值 `max(model, post)`。
- 记录 post 严格位于零与 model 之间时警告且不再额外过滤；默认 `0.0` 不警告。
- 记录别名优先级：两个都非零且不同时 `--post-confidence-threshold` 优先；post 为零视为未设置并回退到 legacy 值，同时记录 info。显式 `--post-confidence-threshold 0.0` 不能覆盖非零 legacy。
- 记录 Otsu/GrabCut 忽略 SAM3 阈值，且其固定 `1.0` score 使 post 过滤在 `0..1` 内无效。
- 记录 no-mask 原因同时写出两个阈值。
- 记录新的 COCO per-mask `score`：SAM3 写模型 score，Otsu/GrabCut 写 `1.0`，VOC/YOLO 无 score 字段；适用于 unified 与 separate COCO 输出。说明旧标注需要 `--overwrite` 才能补字段；`--resume` 会跳过已有图片。
- 在 Version Notes 最上方加 `0.8.0` 条目。

- [ ] **Step 3：更新 classify-train 命令文档**

两种语言：

- 为 `--val-csv` 增加 H3 块。
- 说明其 `image` 列相对同一个 `--images-dir`。
- 说明提供它会关闭 AutoMM 内部随机切分，并作为 `tuning_data` 使用。
- 记录路径重叠（解析后、大小写归一化）、重复行、空文件、图片缺失、未知 label 的检查。
- 说明训练侧校验保持不变（只要求列和路径解析）。
- 记录 `logs/log.txt` 中的验证 CSV SHA256 和图片数，且在 `fit()` 之前写入。
- 把无条件的“AutoGluon 自行进行内部 train/validation 切分”改为条件表述。
- 记录 `fit()` 后回填 `assets.json` 的 `best_score`/`total_train_time`，使重新加载后 `fit_summary()` 有值。
- 在 Version Notes 最上方加 `0.8.0` 条目。

- [ ] **Step 4：更新 augment manifest 文档**

两种语言：

- 记录 manifest 新增的 `files` 数组，含 `original`、`augmented`、`copy_index`。
- 说明 `original` 相对 `--input-dir`，`augmented` 相对 `out-dir/images/`。
- 在 Version Notes 最上方加 `0.8.0` 条目。

**README 参数表：** 在 `README.md` 与 `README.cn.md` 的 Commands 之后新增 `Parameter Changes in 0.8.0` / `0.8.0 参数变化` H2，用旧版 Key Parameters 风格（`Parameter | Description | Default`）列出：
- `clean --dry-run`；
- `segment --sam3-confidence-threshold`、`--post-confidence-threshold` 与弃用别名 `--confidence-threshold`；
- `classify train --val-csv`。

新 H2 必须放在 Commands H2 之外，避免 `test_readme_command_index` 把参数行当成命令行。命令参考仍是完整参数契约。

- [ ] **Step 5：更新当前 master design**

只更新 `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` 的相关章节：

- 5.1 clean：报告工件、确定性分组、split 优先级、dry-run、累积 index resume。
- 5.1 segment：模型阈值 vs 后置过滤契约与兼容别名。
- 5.2 classify train：条件式 `--val-csv`/`tuning_data` 契约与验证集来源。
- 第 10 节文档规范：规范本身不变；确保新命令文档遵守它。

写架构级契约并链接命令文档，不复制完整参数表。

不编辑带日期的历史 plan 或已完成 design spec。

- [ ] **Step 6：修正 workflow skill 的事实性说明**

在 `skills/entomokit-workflow/references/command-profiles.md`：

- clean 一节说明报告写到 `--out-dir`，并说明存在 `--dry-run`。
- classify/split-csv 一节说明 `--val-csv` 可提供 AutoMM 的 `tuning_data`，仅在其省略时使用内部切分。
- 不新增 workflow 步骤、门禁或策略规则。

- [ ] **Step 7：设置发布版本**

统一设置为 `0.8.0`：

- `version.txt`
- `entomokit/_version.py`
- `entomokit/main.py` 的 fallback literal
- `setup.py`
- `tests/test_package_version.py`
- `tests/test_main_cli.py`
- README 中若硬编码了发布版本示例，也一并更新

不新增 changelog。Release notes 就是 Step 1–4 添加的 `Version Notes` 条目。

- [ ] **Step 8：运行文档与版本门禁**

```bash
python -m pytest tests/test_docs.py -q
python -m pytest tests/test_package_version.py tests/test_main_cli.py -q
python -m entomokit.main clean --help
python -m entomokit.main segment --help
python -m entomokit.main classify train --help
```

预期：

- `tests/test_docs.py` 通过，包括两种语言的 `test_command_option_coverage`。
- 版本输出为 `entomokit 0.8.0`。
- 每个 help 都列出新选项和弃用别名。

- [ ] **Step 9：运行完整测试套件**

```bash
python -m pytest -q
```

预期：PASS。把准确总数记录到 Execution Evidence。如果可选后端不可用，记录阻塞项，不要用 skip 或 xfail 隐藏。

## 执行证据

- [ ] 基线：分支 `update-v0.7.2`，起始 ref `c071c37`，编辑前记录 `python -m pytest -q` 结果。
- [x] Task 1：clean 决策/报告/resume 测试通过，包括排序不变式、空输入 manifest、累积子集 resume index、缺少当前模式哈希时的输出哈希回退，以及 `none` 无 manifest 规则。
- [x] Task 2：dry-run、非空目录 resume、augment manifest 和更新后的 CLI namespace 测试通过。
- [x] Task 3：SAM3 阈值、默认不警告、no-mask 原因、别名优先级和 Otsu/GrabCut 测试通过。
- [x] Task 4：AutoMM 验证集测试通过，包括 fit 前写日志和训练侧宽松行为不变。
- [x] Task 5：`tests/test_docs.py`、版本测试、help 检查和完整套件通过。
- [x] Python 3.9 兼容性复核：没有 future import 的模块中不存在 PEP 604 union。
- [ ] 最终评审：不删除输入、失败后不留下半成品输出、无未批准提交/标签/推送，四个 issue 契约均有测试覆盖。既有 `--overwrite` 输入/输出重叠与符号链接目标重定向不在本 plan 范围内，需要单独批准处理。
- [x] 实现结果（2026-10-31）：未设置 SAM3 环境变量时 `522 passed, 2 skipped`；设置 `ENTOMOKIT_SAM3_CHECKPOINT=...` 时 `524 passed, 0 skipped`；这些运行包含 `tests/test_docs.py` 与版本测试。
