# EntomoKit v0.7.2 数据完整性与训练控制设计

## 目标

发布 EntomoKit `0.7.2`，完成四项相关修正：

1. `clean` 用结构化、可 diff 的报告记录每个 keep/drop 决策，而不是只有散文式日志。
2. `clean` 去重是确定性的、基于分组的，而不是 first-thread-wins 的成对判断。
3. `segment` 把 SAM3 自己的置信度阈值与现有后置过滤分开暴露。
4. `classify train` 把 AutoMM 已有的 `tuning_data` 参数暴露为显式验证 CSV。

本版本同时让中英文命令文档符合 `0.7.1` 的文档规范，并修正 `clean` 两个相邻的确定性问题（输出名分配和 MD5 注册泄漏）。

## 范围

- `clean`：带哈希的 per-input 决策报告、确定性分组去重、可选 dry-run、确定性输出命名、写失败提升与半成品清理、累积 index resume 匹配，以及 augment manifest 文件行。
- `segment`：`--sam3-confidence-threshold`、`--post-confidence-threshold`，以及弃用别名 `--confidence-threshold`；修正默认值不应警告的问题和 no-mask 原因。
- `classify train`：`--val-csv`、验证集检查、`tuning_data` 接线，以及在 `fit()` 之前写验证集来源日志。
- 文档：受影响的命令文档、当前 master design、Version Notes、修正后的 workflow skill 事实说明，以及 `0.7.2` 版本声明。

## 非目标

- 不实现新的切分算法。`classify train` 使用 AutoMM 已有的 `tuning_data`；显式切分仍由 `split-csv` 负责。
- 不在 `split-csv` 中实现 split-aware 去重。`clean` 的 `test > val > train` 优先级是已分目录数据的安全网，不替代 split 前去重。
- 本版本不引入 BK-tree 或其他近似最近邻索引来做 pHash 分组。
- 不做 per-image validation manifest 或 per-image validation hash；只记录验证 CSV 的文件级 SHA256。
- 不在训练集与验证集之间做内容哈希 overlap 检查。overlap 只按 `Path.resolve()` 和大小写归一化后的路径检查；内容重复（包括增强副本）不会被发现。
- 不修改 vendored SAM3 模型实现。
- 不新增 changelog、文档生成器、第二套 CLI 注册表或新依赖。
- 不修改 workflow 策略；只修正 workflow skill 中的事实性命令说明。
- 未经操作者另行批准，不提交、不打标签、不推送、不发布。

## 背景

### clean

当前实现在锁内“检查并注册”哈希，线程池里的图片谁先到锁谁保留，后来的图片被省略。决策因此依赖线程调度而不是数据本身。同目录两个清洗后同名的文件也会按完成顺序获得 `_1` 后缀，所以即使去重决策确定，输出名仍可能不确定。

该问题是非确定性的，且已观察到：同一输入集重复运行，可能一次保留 `test/` 副本，另一次保留 `train/` 副本。当重复对跨 train/test 边界时，泄漏方向因此是随机的。

当前日志把 reason、匹配 basename 和 distance 拼成散文行，保留文件不记录。不同类别目录下出现同名文件时，basename 有歧义。没有 per-file 表可用于 join 回 label、计算 per-class retention 或 diff 两次运行。

当前 `clean` 还会在后续 pHash 或保存之前注册 MD5。如果后续步骤失败，MD5 条目可能残留，导致后面真实存在的图片被误判为重复而丢掉，尽管输出里没有任何文件。三张相同图片加第一次保存失败时，当前实现可能以 `processed=0` 结束且没有任何输出。

`clean` 不删除输入图片，只是不把重复项写入 `cleaned_images/`。只有 `--overwrite` 会删除输出目录内容。报告和文档必须使用“drop/省略”，不能写“删除”。

### segment

`Sam3Processor` 已经接受 `confidence_threshold`，并在模型内部执行 `out_probs > threshold`。`Sam3Predictor` 暴露了该构造值，但 `SAM3Wrapper` 没有传递，`SegmentationProcessor` 也没有暴露。因此模型始终使用默认 `0.5`。CLI 的 `--confidence-threshold` 只是后置过滤，低于 `0.5` 的值无法让 SAM3 更宽松。

Otsu 与 GrabCut 的 score 恒为 `1.0`，所以后置过滤在文档范围 `0..1` 内没有实际效果。

### classify train

`train()` 只把 `train_data` 传给 `predictor.fit()`。AutoMM 因此每次运行都自己做随机 train/validation 切分。每个模型 arm 可能得到不同切片，增强副本可能落在两侧，选中的验证行也没有记录。

AutoMM 已经支持 `fit(tuning_data=...)`。提供它会关闭内部随机切分，不需要新的切分逻辑。本机 AutoGluon 1.5.0 的文档说明 `tuning_data` 非空时 `holdout_frac` 被忽略。

### resume

现有 `--resume` 路径对已有输出文件做哈希，再与输入文件比较。由于 `compute_md5` 哈希的是解码后的像素（`img.tobytes()`），resize 或有损格式转换后的输出与输入 MD5 可能不同。默认 512 短边 resize 加 JPEG 转换时，MD5 resume 匹配可能漏掉这些输出，并写出带 `_1` 的重复文件；pHash 可能仍能匹配，但不保证。新的累积 manifest index 记录源哈希，作为主要 resume 键；输出哈希保留为索引未匹配输入的回退方案。

## 设计

### 1. clean 决策流水线

#### 1.1 三个阶段

`clean` 拆成三个阶段：

1. **哈希阶段（并行）：** 读取每个输入，按 `--dedup-mode` 计算所需哈希，并关闭图像。结果按输入相对路径存放；worker 内不做 keep/drop 决策。
2. **决策阶段（串行）：** 按输入相对 POSIX 路径排序，构建连通分组，选择代表，并为每个输入分配一条 decision 行。
3. **写出阶段（并行）：** 按 worker 启动前分配好的输出名写出 retained 图片。写失败时可以提升其他组员并更新 decision 行。

决策阶段刻意串行。与图片 I/O 相比它很便宜，却能给分组、代表选择、group ID 和输出名提供唯一确定性顺序。提升结果按 `group_id` 顺序串行收尾，使输出名和行不依赖 future 完成顺序。

输入图片最多读两次：一次哈希，一次写出。“只读一次”只约束哈希阶段内部，不是整条流水线的保证。

#### 1.2 分组

- `md5`：按像素内容 MD5 精确分组。`compute_md5` 哈希 `img.tobytes()`，所以这不是文件字节 MD5。两个编码不同但解码像素相同的文件会分到一组；对文件执行 `md5sum` 不会得到报告中的值。
- `phash`：对所有 Hamming 距离不超过 `--phash-threshold` 的边做 union-find。这是连通分量（传递）定义，不是成对列表。
- `md5+phash`：先做精确 MD5 分组，再对 MD5 组代表做 pHash union-find；代表合并时合并成员集合。
- `none`：每个输入都是单例分组。

本版本 pHash 分组是 O(n²)。为了保持无依赖且严格传递，接受该成本。未来版本可以加索引而不改变报告语义。

如果 `phash` 无法为某个输入算出合法哈希，该输入记为 hash/decode error，组字段为空。`md5+phash` 中 MD5 成功但 pHash 失败时，保留为 MD5-only 候选，而不是让图片失败。

#### 1.3 代表选择

不含 existing 输出的分组，代表是以下元组最小的成员：

1. split rank：`test` = 0，`val`/`valid`/`validation` = 1，`train` = 2，无可识别 split = 3；
2. 输入相对 POSIX 路径字典序。

split 识别大小写不敏感，并匹配精确路径段。如果一个路径含多个可识别段，优先级最高的段生效。这样跨 train/val/test 的重复组不会保留训练副本而丢掉 held-out 副本。

没有可识别 split 段的分组只按路径顺序。

`--resume` 下，当前输入按当前模式直接与累积 index 和输出哈希匹配。匹配输入变成 `status="existing"`；未匹配输入进入普通分组。若一个输入匹配多个 existing 输出，按当前模式的 tie-break 选规范 `kept_path`（MD5：路径；pHash：距离再路径）；existing 输出永不删除或重写。

#### 1.4 匹配元数据

每个非代表输入记录最近的组员：

- MD5：代表，无 distance。
- pHash：Hamming 距离最小的成员；同距按相对路径。
- `md5+phash`：若成员与任一组成员 MD5 相同，`reason=duplicate_md5`、`distance=None`；代表同 MD5 时匹配代表，否则匹配同 MD5 成员中路径字典序最小者。否则 `reason=duplicate_phash`，取最小 pHash 距离并按路径打破平局。

`matched_path` 是最近成员，不一定是代表。这样即使传递闭包导致到代表的距离超过阈值，也能保留真正把它拉进组的边。`kept_path` 标识组代表。

`retained` 行使用 `kept_path=input_path`、`kept_scope="input"`。`dropped` 行使用 `kept_scope="input"`、`matched_scope="input"`；它们只出现在普通输入组中。`existing` 行的 `kept_path`、`matched_path` 和 `output_path` 指向规范 existing 输出，`kept_scope="existing_output"`、`matched_scope="existing_output"`。`existing` 行的 `md5`/`phash` 描述当前输入，不是 existing 输出。`group_id` 在 input 组与 existing 匹配的统一序列中按 `(split_rank, representative path)` 分配；existing 匹配使用 output-relative 规范路径。

#### 1.5 确定性输出名

分配名字前，先按 `input_path` 排序 retained 输入决策。命名复用现有规则：

- 对源 stem 执行 `clean_filename`；
- 扩展名来自 `--out-image-format`；
- 同目录重名时追加大小写不敏感的 `_1`、`_2` … 后缀；
- `--resume` 下已有输出名也加入 used-name 集合。

不允许 worker 选名或按完成顺序追加后缀。`output_path` 以相对 `--out-dir` 的形式保存，例如 `cleaned_images/beetles/a.jpg`。被提升的成员在初始名字全部固定后，由同一个确定性命名器分配名字。

#### 1.6 写失败与提升

代表并行写到预先分配的路径。所有写 future 完成后：

1. 按 `group_id` 顺序遍历分组；
2. 对失败代表，尽力删除半成品目标文件（`dst.unlink(missing_ok=True)`）；
3. 把该 decision 标为 `error`，`reason=write_error:<message>`，保留 `group_id` 和 `group_size`；
4. 按代表顺序提升下一个组员，分配新输出名并写出；
5. 若提升成员写出成功，标为 `retained`，reason 清空，写入新的 `output_path`；
6. 更新组内所有成员的 `kept_path`/`kept_scope`/`matched_path`/`matched_scope`，指向存活代表；只要 `matched_path` 变化就重算 `reason` 和 `distance`。

如果全组都无法写出，剩余成员都记为 `error`，`reason=group_write_failed`，保留 `group_id`/`group_size`，清空路径字段。`cleaned_images/` 中不得留下半写文件。

任何阶段都不写或删除输入文件。

#### 1.7 报告工件

报告写到 `log_path` 的父目录（命令的 `--out-dir`），不写进 `cleaned_images/`。所有报告行都按 `input_path` 排序。

`filter_decisions.csv` 每个输入恰好一行：

```text
input_path,status,reason,kept_path,kept_scope,matched_path,matched_scope,distance,group_id,group_size,md5,phash,output_path
```

`retained_images.csv` 包含 `retained` 与 `existing` 行：

```text
input_path,status,output_path
```

`dropped_images.csv` 只包含 `dropped` 行：

```text
input_path,reason,matched_path,matched_scope,distance,group_id
```

`clean_manifest.json` 含 run-level 字段、与 `filter_decisions.csv` 行相同的 `files` 数组，以及供 resume 使用的累积 `index` 数组：

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

`files` 是本次运行的行，绝不能当作 resume index。`index` 是累积 resume 索引：每个已知输入路径一行，以 `input_path` 为键，指向规范输出路径并记录源哈希。`status` 可为 `retained`、`existing` 或 `dropped`；dropped 行保留自己的源哈希，但指向存活代表输出。dry-run 原样复制上一次有效 `index`，不添加“本应写出”的输出。`--dry-run` 下 `output_files.cleaned_images` 为 `null`。`files` 使用带类型的值：`distance` 为 `int | null`，`group_size` 为整数，路径为字符串。

`md5` 列是 `compute_md5` 返回的像素内容 MD5 十六进制；`phash` 是零填充 64 位十六进制 pHash。本次运行未计算某个哈希时，对应列为空。

状态：

- `retained`：组代表；写出新输出（`--dry-run` 下为“本应写出”）。
- `dropped`：普通输入组中的重复项；reason 为 `duplicate_md5` 或 `duplicate_phash`。`kept_scope="input"`；`matched_scope="input"`。
- `existing`：`--resume` 下输入匹配已有输出；不写新文件；`kept_path`/`matched_path`/`output_path` 是规范 existing 输出。按 pHash 匹配时 `distance` 是到规范 existing 输出的 Hamming 距离；按 MD5/`none` 匹配时为空。`md5`/`phash` 描述当前输入，不是 existing 输出。`group_id` 按规范输出路径顺序分配；`group_size` 是映射到该规范输出的当前输入数再加一（existing 输出本身）。
- `error`：哈希/解码失败（`unidentified_image`、`hash_error:<message>`）时 `group_id`/`group_size` 为空、路径为空；写失败（`write_error:<message>`、`group_write_failed`）时保留 `group_id`/`group_size`。有存活成员的写失败保留 `kept_path`/`kept_scope`，`matched_path`/`matched_scope` 为空；全组耗尽时清空所有路径字段。

`n_input == n_retained + n_dropped + n_existing + n_errors`。`n_retained` 只计新输出；`n_existing` 计 input 匹配，不是 unique existing output。因此 `retained_images.csv` 有 `n_retained + n_existing` 行，`output_path` 可能重复；per-class retention 必须按 `output_path` 统计，而不是 `input_path`。

报告原子写入：每个 CSV/JSON 先写到 `--out-dir` 中的临时文件，再用 `os.replace` 到位。只有所有报告完成后才追加日志 summary。

输入目录为空时，仍写出零行报告和 `n_input=0` 的合法 manifest。

日志在写出/提升阶段结束后，从最终 decision 行按 `input_path` 顺序生成。per-file 行使用与报告相同的 status、路径、reason 和 distance；旧的 worker 内散文行删除。日志末尾追加由同一份行计算的 summary。

#### 1.8 Dry-run 与 resume

`clean --dry-run`：

- 写出与“无写失败的真实运行”相同的报告和 manifest；
- 对每个“本应写出”的图片执行解码和内存中的 resize/pad，但跳过 `mkdir`/`save`；
- 不创建 `cleaned_images/`；
- 不删除任何东西；
- 令 `output_files.cleaned_images` 为 `null`；
- 与 `--overwrite` 互斥；
- 要求 `--out-dir` 为空或传入 `--resume`。

dry-run 之后，真正写入同一目录也需要 `--resume`，因为报告和日志已使目录非空。文档必须给出这个流程。CLI summary 在 dry-run 使用 `would write N new image(s)`，真实运行使用 `wrote N new image(s)`，单复数与计数一致。

`clean --resume` 匹配优先级：

1. 只有 `clean_manifest.json` schema 合法且未损坏时才加载。忽略不支持的 `schema_version`、损坏 JSON、非法行，以及绝对路径、非 POSIX 相对路径或 resolve 后位于 `--out-dir` 或 `cleaned_images/` 输出目录之外的 `output_path`，并 warning。dry-run manifest 只贡献其 `index`；绝不把它的 `files` 当作输出。
2. 累积 `index` 是主来源，包括指向存活输出的 `dropped` 行。按当前模式匹配：`md5` 用源 MD5 精确相等；`phash` 用阈值内最近源 pHash；`md5+phash` 先精确 MD5 再最近 pHash；`none` 用 `input_path` 精确相等。多个 index 行映射同一输出时，用字典序最小的 output-relative 路径作为规范。
3. 输出缺失、不可读或位于 `--out-dir`/`cleaned_images/` 之外的 index 行忽略并 warning。
4. 对索引未匹配的输入，从**所有**已有输出图片构建输出哈希映射（不只是 manifest 中不存在的路径），按当前模式匹配输出哈希。这覆盖“旧 manifest 有路径但缺少当前模式哈希”的情况。记录 warning：MD5 输出哈希可能漏掉 resize/有损转换后的输出；需要精确匹配时建议全新运行或 `--overwrite`。
5. `dedup_mode="none"` 没有可用 index/manifest 时，不得按镜像输出路径匹配；保留输入并 warn 说明精确 resume 需要 manifest。

真实运行后，即使 `files` 为空也要写 `index`。把本次 `retained`/`existing` 行以及有存活代表的 `dropped` 行 merge 进 `index`，同一 `input_path` 的旧行被替换。dropped 行的 index 行映射到存活代表输出，并保留 dropped 输入自己的源哈希，使传递组成员仍可 resume；有存活代表的写失败行同样处理。只移除哈希/解码错误和全组耗尽错误的行，因为它们没有可用输出或哈希。当前运行未出现的输入，其旧行在输出仍存在时保留；输出缺失则剪掉并 warning。dry-run 原样复制上一次 `index`。

已有输出永不删除或重写。报告覆盖本次 invocation 发现的输入，并在每次 invocation 覆盖；`index` 不覆盖。日志继续 append。

### 2. segment 阈值

暴露两个独立阈值：

| 选项 | 默认 | 范围 | 作用 |
|---|---|---|---|
| `--sam3-confidence-threshold` | `0.5` | `0..1` | SAM3 与 SAM3-bbox 模型推理 |
| `--post-confidence-threshold` | `0.0` | `0..1` | mask 返回后的 EntomoKit 后置过滤 |
| `--confidence-threshold` | `0.0` | `0..1` | 后置过滤的弃用别名 |

SAM3 的有效过滤阈值为 `max(sam3_confidence_threshold, post_confidence_threshold)`，模型端是 `>`，后置端是 `>=`。post 阈值等于或低于模型阈值时是 no-op。仅在解析后的 post 阈值严格位于零与 SAM3 阈值之间（`0 < post < sam3`）时警告；默认 `post=0.0` 表示“不做后置过滤”，不得警告。

别名解析：`0.0` 视为“未设置”。`--post-confidence-threshold` 为 `0.0` 且 `--confidence-threshold` 大于零时，使用 legacy 值并记录 info。两个都非零且不同时，`--post-confidence-threshold` 优先并记录 warning。因此显式 `--post-confidence-threshold 0.0` 不能覆盖非零 legacy。

弃用别名仍在 `--help`/命令文档中可见并可用，但会标记 `_deprecated` 并从生成的 shell 补全中省略，引导用户使用 `--post-confidence-threshold`。它仍保留在 runtime schema 与 parameter guard 中，以保持向后兼容。

SAM3 阈值沿以下链路传递：

```text
SegmentationProcessor
  -> SAM3Wrapper(confidence_threshold=...)
  -> Sam3Predictor(confidence_threshold=...)
  -> Sam3Processor(confidence_threshold=...)
```

不编辑 vendored `src/sam3/model/sam3_image_processor.py`。它现有的构造函数和 `_forward_grounding` 已经应用该阈值。

Otsu 与 GrabCut 忽略 SAM3 阈值。它们恒为 `1.0` 的 score 使后置过滤在文档范围 `0..1` 内无效。“ignored” 日志只在 SAM3 阈值不是默认 `0.5` 时输出。

no-mask 原因同时写出两个阈值，而不是只写后置阈值：

```text
No masks passed the effective confidence threshold (sam3=<model>, post=<post>)
```

SAM3 方法且任一阈值大于零时使用该消息；两个阈值都为零时保留 `No masks returned by <method>`。这样可以区分“模型在自己的阈值之上没有返回任何东西”和“后置过滤把它们全部删掉了”。

两个阈值都在 CLI 和 `SegmentationProcessor` 内校验。`src/segmentation/processor.py` 没有 `from __future__ import annotations`，必须使用 `Optional[float]`，不能用 `float | None`。

COCO unified 与 separate 标注都会写入每个 mask 的 `score`：SAM3 写模型 score，Otsu/GrabCut 写 `1.0`。VOC XML 与 YOLO TXT 没有标准 score 字段，保持不变。score 由 `COCOMetadataManager.add_annotation(..., score=...)` 写入；省略该参数时保持原 schema。

### 3. classify train 验证集

`--val-csv` 可选。其 `image` 列相对与训练 CSV 相同的 `--images-dir` 解析；也接受绝对路径。label 子集与 overlap 检查在 `train()` 拿到两个 DataFrame 后执行；单文件 helper 只负责列、路径解析、空文件、validation 图片缺失和重复行。

提供时：

- 在 `fit()` 前加载并校验验证 CSV；
- 要求 `image`、`label` 列；
- 至少一行；
- 所有引用图片存在；
- 拒绝 validation 内解析后重复的 image 行；
- 拒绝 validation 中出现训练集没有的 label；
- 拒绝同时出现在 train 与 validation 的解析、大小写归一化路径；
- 把解析后的 DataFrame 作为 `fit(tuning_data=val_df)` 传入；
- 在 `fit()` 之前记录验证 CSV 路径、文件级 SHA256、图片数、overlap 结果。

训练侧加载保持现有宽松度：要求 `image`、`label` 列并解析路径，但本版本不为训练集增加缺图或重复行拒绝。

未提供时：

- 不传 `tuning_data`；
- 保留 AutoMM 内部随机切分；
- 日志记录 `Validation: AutoMM internal random split (no --val-csv)`。

不新增切分逻辑。`train.processed.csv` 仍是训练表。不创建 per-image validation manifest 或 per-image validation hash。`assets.json` 缺失、损坏或不可写只产生 warning，不让训练失败。现有 NVML 重试路径复用同一个 `fit_kwargs`，包含 `tuning_data`。

`fit()` 返回后，entomokit 会把 `predictor._learner` 的 `best_score` 与 `total_train_time` 回填到模型 `assets.json`（后者缺失时用实际耗时）。AutoMM 在 top-k averaging 时先写 `assets.json`，之后才设置这些属性，所以重新加载后 `fit_summary()` 原本会显示 null。只更新这两个 key，不删除，因为 `MultiModalPredictor.load()` 会直接索引它们。

### 4. 文档与发布

- Release notes 写入每个受影响命令文档的 `Version Notes` / `版本注记`，最新在前。
- 不新增 changelog。
- 每个改动命令都更新两种语言：`clean`、`segment`、`classify-train`，以及 `augment`（其 manifest 新增 `files`）。
- 更新当前 master design（`2026-03-24-entomokit-refactor-design.md`）的架构级契约；不复制完整参数表。
- 在 `README.md` 与 `README.cn.md` 新增简短的 `Parameter Changes in 0.7.2` / `0.7.2 参数变化` 段，用 Key Parameters 风格表格列出 `clean --dry-run`、segment 阈值与 `classify train --val-csv`。该 H2 必须位于 Commands H2 之外，保证 README 命令索引契约仍只看到命令行；命令参考仍是完整参数契约。
- 修正 `skills/entomokit-workflow/references/command-profiles.md` 中关于 `--val-csv` 和 clean 报告/dry-run 的事实性说明。不修改 workflow 策略。
- 不编辑历史 plan 或已完成 spec。
- 把 `version.txt`、`entomokit/_version.py`、`entomokit/main.py` fallback、`setup.py` 和版本测试设为 `0.7.2`。
- 本次不修改 workflow skill 的策略。

## 兼容与迁移

- 现有 `clean` 调用保留相同 CLI 参数和输出位置；新报告是增量，但报告/日志会让 `--out-dir` 在后续运行中非空，因此复用目录需要 `--resume` 或 `--overwrite`。
- `--confidence-threshold` 仍被接受，现在表示后置过滤；`0.0` 视为未设置。两个都非零且不同时 `--post-confidence-threshold` 优先；post 为 0.0 时 legacy 生效并记录 info。
- 默认 `0.5` 模型阈值保持现有 SAM3 行为。
- 现有 Otsu/GrabCut 行为保持不变。
- 未提供 `--val-csv` 时，现有 `classify train` 行为保持不变；训练侧 CSV 宽松度不变。
- `augment_manifest.json` 新增 `files` 数组；现有 run-level 字段保留。
- 0.7.2 之前生成的 COCO 标注在 `--resume` 下不会补 per-mask `score`，因为已有图片被跳过；需要重新生成时使用 `--overwrite`。
- `clean` 去重结果可能变化，因为分组是传递的而非成对；这是有意的，必须在 Version Notes 中说明。
- `--resume` 现在优先使用累积 manifest `index`，因此 resize/转换后的输出也能被识别，且子集运行不会丢失旧 index。没有 manifest 时，MD5 输出哈希匹配仍有限，并会警告。
- `--resume` 可能把匹配输入报告为 `existing` 而不写重复文件。
- `retained_images.csv` 包含 `existing` 行，但它是 input-centric：多个输入可能映射一个输出。per-class retention 必须按 `output_path` 统计，`n_existing` 统计输入数而非 unique output 数。
- `--resume` 不会重编码已有输出；修改 `--out-short-size`/`--out-image-format` 需要 `--overwrite`。已 split 数据中 existing 在 train、重复输入在 test 时，resume 可能保留泄漏；需要重新平衡请用 `--overwrite` 或在 split 前 clean。
- `dedup_mode="none"` 没有 manifest 时不按镜像名匹配；精确 resume 需要 manifest。
- 报告中的 MD5 是像素内容 MD5，不是文件字节 MD5；对 `existing` 行，`md5`/`phash` 描述当前输入，不是 existing 输出。用户不得拿它与 `md5sum` 输出比较。
- 旧 `log.txt` 仍可读；新报告是结构化的事实来源。

## 验证

- `clean` 在 `--threads=1`、`8`、`32` 以及重复运行时产生相同且同序的 decision 行和相同输出文件集合。
- pHash 链 `A~B`、`B~C`、`A!~C` 形成一组；`md5+phash` 中 MD5 相同的 dropped 行标为 `duplicate_md5`，其余标为 `duplicate_phash`。
- 跨 train/val/test 的重复组保留 test 成员，其次 val，最后 train。
- MD5 与 pHash 匹配路径是相对路径，跨类别目录无歧义；哈希列非空且在重复行之间一致。
- 代表写失败会删除半成品、提升下一个成员并更新组内引用。失败的 MD5 注册不能丢掉后面真实存在的图片。
- 全组耗尽时，每个成员都报告为 `group_write_failed`，并保留 `group_id`/`group_size`。
- `filter_decisions.csv` 计数与 `clean_manifest.json` 和日志 summary 一致；行按 `input_path` 排序。
- 空输入目录写出零行报告和 `n_input=0` 的合法 manifest；新建空运行 `index=[]`，空 resume 保留输出仍存在的旧 index 行。
- `clean_manifest.json` 同时含 `files`（本次运行）与累积 `index`；`files` 绝不作为 resume index。
- `md5+phash` existing 由 MD5 匹配时 `distance` 为空；只有 pHash 匹配才带 distance。
- `clean --dry-run` 写报告但不写图片，令 `output_files.cleaned_images=null`，并拒绝 `--overwrite`；非空 out-dir 需要 `--resume`。
- `clean --resume` 通过累积 manifest `index` 识别 resize/转换后的输出，并报告 `matched_scope=existing_output`；永不重写该输出。
- Resume 忽略绝对路径、位于 `--out-dir` 之外、位于 `cleaned_images/` 之外、缺失或非图片的 index 输出。
- A+B → 只 A resume → 只 B resume，B 保持 `existing`；子集运行不会丢失累积 index。
- 传递组 `A~B~C` 只保留 C 时，后续只 resume B 仍能通过 B 自己的 index 行（status `dropped`）匹配到 C 的输出。
- 上一次是 md5-only manifest、当前 phash 模式时，缺少可用 phash index 行的输出通过输出哈希回退匹配。
- 已索引的 existing 输出（不只是扫描到的输出）也参与 decoded-output 哈希回退。
- 已索引输出在接收前必须完整解码；截断输出被忽略并重新生成输入。
- validation label 在 `fit()` 前映射到 training label 的实际表示；validation 的空 image/label 单元格被拒绝。
- 没有 manifest 的 `clean --resume` 在 MD5 输出哈希可能不可靠时警告。
- `dedup_mode="none"` 且没有 manifest 时，同 stem、不同扩展名输入不会被误标 existing。
- 提升后重算 `reason`、`distance`、`kept_scope`、`matched_scope`。
- `retained_images.csv` 的 per-class retention 按 `output_path` 统计；`n_existing` 统计输入数而非 unique output 数。
- SAM3 阈值确实到达 `Sam3Predictor`；`--sam3-confidence-threshold 0.2` 能保留 `(0.2, 0.5]` 的 mask。
- 默认 SAM3 阈值不警告；显式 `0 < post < sam3` 警告而不报错。
- no-mask 原因同时写出两个阈值。
- COCO unified 与 separate 标注包含 mask `score`；VOC/YOLO 输出不变。
- Otsu/GrabCut 输出不受 SAM3 阈值影响。
- 仅在提供 `--val-csv` 时 AutoMM 收到 `tuning_data`。
- 解析/大小写归一化后的 overlap、validation 重复行、空 validation、缺失图片、未知 label 都在 `fit()` 前失败。
- 验证 CSV 路径、SHA256、图片数和 overlap 结果在 `fit()` 前写入日志。
- 训练完成后 `assets.json` 的 `best_score` 与 `total_train_time` 非空，重新加载后 `fit_summary()` 能报告它们。
- 两种语言文档更新后 `tests/test_docs.py` 通过。
- 生成的 bash/zsh/fish 补全省略 `--confidence-threshold`，而 `segment --help` 仍以弃用别名记录它。
- `README.md` 与 `README.cn.md` 含 `0.7.2` 参数变化表，且 `test_readme_command_index` 仍通过。
- Python 3.9 语法复核：没有 future import 的模块中不存在 PEP 604 union。
- 完整测试套件通过；不可用的可选后端作为阻塞项报告，不用 skip 或 xfail 隐藏。

## 风险与权衡

- **O(n²) pHash 分组：** 换取严格传递语义，代价是可扩展性。`0.7.2` 接受，并在文档中说明边界。
- **Split-aware 代表选择：** 基于精确路径段的启发式；使用其他 split 命名的数据集会回退到路径顺序。
- **Held-out-first 会移除训练侧重复：** 保留 test/val 副本可防泄漏，但训练集是整条样本消失，而不只是少一个重复。
- **`--resume` 下 existing 输出总是优先：** 避免重复写出，但可能不会重新平衡 split。需要全新布局时用 `--overwrite`。
- **Prior-manifest resume 依赖累积 index：** index 必须在子集运行中保留；陈旧或手工编辑的 index 会被信任。缺失/非法 index 行回退到输出哈希，MD5 可能漏掉 resize 后的输出，因此会警告。
- **累积 index 增长：** 每个已知输入路径一行，会随输入历史增长；剪掉缺失输出后仍可能随长期目录增长。
- **`none` 无 manifest：** 没有 manifest 无法精确 resume；实现选择保留并 warning，而不是按镜像路径猜测。
- **写失败提升：** 尽可能保留至少一份副本，但表格反映实际 I/O 结果；提升串行化以保持名字确定性。
- **post 阈值 no-op：** 用户可能期望 post 低于 model 时报错；warning 干扰更小，并保留 `0` 作为合法“不做后置过滤”值。默认值也警告的方案被明确拒绝，因为会产生噪声。由于 `0.0` 也用于别名解析中的“未设置”，显式 `--post 0.0` 不能覆盖非零 legacy；info 日志让该回退可见。
- **Validation overlap 只查路径：** 不同文件名但内容重复不会被发现。应在 split 前去重。
- **Resume 按输入匹配：** 如果某个输入只是通过另一个**新**输入与已有输出传递相似，它会被保留而不是标记 `existing`。上一次运行的输入由累积 index 覆盖（包括 dropped 行），所以只影响完全由新输入组成的链。这是避免通过不确定的链误匹配的刻意取舍。
- **像素 MD5 vs 文件 MD5：** 报告的 MD5 验证的是解码像素相等，不是文件字节相等；文档必须说明。
- **训练侧宽松度：** 本补丁仍不拒绝训练集缺失图片，以避免破坏性变更。

## 备选方案

- **保留成对 first-wins，只加 manifest：** 拒绝，因为决策仍依赖调度器，split 泄漏仍是随机的。
- **按 train > val > test 选代表：** 拒绝，因为保留的正是 issue 指出的泄漏方向。
- **现在增加 `--keep-priority`：** 推迟；固定的 held-out-first 规则更小，且覆盖已报告的失败。
- **用 BK-tree 做 pHash：** 推迟，以保持本版本无依赖且语义精确。
- **修改 vendored SAM3 processor：** 拒绝；构造函数已支持阈值，只需要接线。
- **在 `classify train` 中实现新的 validation 切分：** 拒绝；AutoMM 已支持 `tuning_data`。
- **新增 changelog：** 当前文档规范拒绝；Version Notes 是 release note 的归属。
- **只要 post < model 就警告：** 拒绝，因为默认 `post=0.0` 会让每次 SAM3 运行都警告。仅对 `0 < post < model` 警告。
- **用 `None` 作为 `--post-confidence-threshold` 的 CLI 默认值：** 拒绝，因为文档默认值是 `0.0`，runtime schema/参数卡会不一致。改用别名解析处理。
- **resume 只依赖输出文件哈希：** 拒绝，因为 resize/转换可能改变像素 MD5；累积 manifest `index` 才是主要来源。
- **单独增加 `clean_index.json`：** `0.7.2` 拒绝，以避免第五个报告工件；累积 `index` 放在 `clean_manifest.json` 内，`files` 保持 per-run。
- **`dedup_mode="none"` 无 manifest 时按镜像输出路径匹配：** 拒绝，因为同 stem、不同扩展名会混淆；精确 `none` resume 需要 manifest。

## 参考

- 实现计划：`docs/superpowers/plans/2026-10-29-entomokit-v072-data-integrity-and-training-controls.md`
- 中文计划：`docs/superpowers/plans/2026-10-29-entomokit-v072-data-integrity-and-training-controls.cn.md`
- 文档规范：`docs/superpowers/specs/2026-10-01-entomokit-documentation-conventions-design.md`
- 当前 master design：`docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`
- phyloAI filter 参考：`phyloAI/phyloai/pretree/filter.py`（`retained_loci.csv`、`dropped_loci.csv`、`filter_decisions.csv`、`result.json`）
