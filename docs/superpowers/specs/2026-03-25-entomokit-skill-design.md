# EntomoKit Skill 设计文档

**日期**: 2026-03-25（修订：2026-03-26 v2）
**状态**: 设计确认，待进入实施计划

---

## 1. 背景与目标

`entomokit` 已提供完整 CLI，但对无命令行经验的用户，核心痛点是：

1. 不清楚整条工作流的阶段顺序与每步的必要性
2. 数据格式不满足隐含约束（文件命名、图片规格、CSV 列名），导致执行失败
3. 失败后无法根据底层日志快速定位并自行恢复

本 Skill 的目标：通过自然语言引导，让用户逐步完成从原始数据到可部署分类模型的完整流程，同时把"数据规范准备"前置为最高优先级环节，确保每一步的前置条件都被满足。

---

## 2. 产品定位

### 目标用户

- **第一优先**：研究/业务用户，以"拿到可用分类结果"为目标，不关心 CLI 细节
- **第二优先**：数据分析初学者，需要有限配置能力与过程理解

### 交互原则

1. **阶段可见**：用户随时知道自己处于哪个阶段、还差哪些步骤
2. **必选先行**：必做步骤在前，可选步骤明确标注并说明使用场景
3. **先教学后执行**：关键数据格式先展示模板和约束说明，再让用户准备数据
4. **渐进推进**：每个阶段完成后主动询问是否继续下一步，用户可随时停止
5. **错误可恢复**：错误必须定位到具体问题，并给出明确修复动作
6. **参数展示分层**：执行前必须向用户展示关键参数摘要；高成本/不可逆操作需等待用户确认或修改，低风险操作展示后直接执行（详见第 5 节）
7. **用户数据优先**：默认始终处理用户提供的数据；仅在用户明确要求演示/排错/教学时，才切换或补充使用 `data/` 示例资产
8. **能力可见但不打扰**：在关键节点主动告知“可用 `data/` 做快速体验/教学/排错”，但默认不切换，需用户明确同意

---

## 3. 入口模式

Skill 支持两种入口模式，对话开始时根据用户表述自动判断：

### 模式 A：引导模式

**触发**：用户表述模糊或明确说"我想从头开始"、"我有一批图片要处理"。

**行为**：从 Phase 0 开始，按顺序逐步引导。

### 模式 B：直接模式

**触发**：用户明确说出具体命令意图，如"我想做 segment"、"帮我把 CSV 划分一下"、"我要训练模型"。

**行为**：识别目标步骤，直接跳转，**但 Phase 0（doctor）不跳过**。

**直接模式下 clean 的处理策略**：

clean 在引导模式下是强制通行证，但在直接模式下降级为**询问确认**：

> "你的图片是否已经过格式清理（统一格式/尺寸/规范命名）？如果没有，segment 可能因文件名或格式问题失败。是否先运行 clean？"

| 用户回答 | Skill 行为 |
|---|---|
| 是，需要先 clean | 进入 1b clean 流程，完成后继续目标步骤 |
| 不需要，图片已规范 | 直接进入目标步骤，跳过 clean |
| 不确定 | 展示 clean 的三项规范要求，由用户判断 |

**直接模式支持的步骤**（可直接跳入任意一步）：

| 用户意图关键词 | 跳转目标 |
|---|---|
| 视频提帧、extract frames | Phase 1a |
| 清理图片、clean、规范化 | Phase 1b |
| 分割、segment、抠图、bbox | Phase 1c |
| 合成、synthesize、背景增强 | Phase 1d |
| 增强、augment、数据扩充 | Phase 1e |
| 划分数据集、split、CSV | Phase 2 |
| 训练、train、classify | Phase 3-train |
| 推理、predict、预测 | Phase 3-predict |
| 评估、evaluate | Phase 3-eval |
| 嵌入、embed、UMAP、可视化 | Phase 3-embed |
| 导出、export、ONNX、部署 | Phase 3-onnx |

---

## 4. 参数展示分层规则

每个步骤执行前，Skill 必须向用户展示关键参数摘要，让用户有机会理解和修改。展示深度按执行成本分两档：

### 档位 A：高成本 / 不可逆操作

展示参数摘要后**等待用户明确确认**再执行。用户可回复"确认"继续，也可说"修改 xxx 参数"后重新展示。

适用步骤：`segment`（大批图片）、`synthesize`、`augment`、`classify train`

展示格式（示例 classify train）：
```
即将运行 classify train，参数如下：
  训练集：datasets/split/train.csv（992 张，12 类）
  骨干网络：convnextv2_femto
  最大 epoch：50 / batch size：32
  输出目录：runs/exp1/

确认执行？还是需要修改某个参数？
```

### 档位 B：快速 / 低风险操作

展示参数摘要后**直接执行**，无需等待确认。完成后展示结果摘要。

适用步骤：`extract-frames`、`clean`、`split-csv`、`classify predict`、`classify evaluate`、`classify embed`、`classify cam`、`classify export-onnx`

展示格式（示例 clean）：
```
即将运行 clean：
  输入：images/raw/（扫描到 1500 张图片）
  输出：images/cleaned/
  短边尺寸：512px / 格式：jpg / 去重：md5

开始执行...
```

**参数记录**：详细参数由 `log.txt` 自动保存，`entomokit_progress.json` 仅记录路径与状态，不重复记录参数。

---

## 5. 整体流程架构

```
Phase 0  环境确认          doctor
   │
Phase 1  数据集准备
   ├── 1a [可选]  extract-frames   视频 -> 图片
   ├── 1b [必做]  clean            图片规范化（通行证）
   ├── 1c [可选]  segment          分割/裁剪 bbox 图（背景复杂时）
   ├── 1d [可选]  synthesize       合成背景增强多样性
   └── 1e [可选]  augment          图片增强 / 改善长尾
   │
Phase 2  数据集划分
   ├── CSV 准备向导               格式说明 + 模板 + 验证
   └── split-csv                  生成 train / val / test CSV
   │
Phase 3  分类训练与评估（渐进拓展）
   ├── classify train
   ├── classify predict           （完成后询问是否继续）
   ├── classify evaluate          （完成后询问是否继续）
   ├── classify embed + cam       （完成后询问是否继续）
   └── classify export-onnx       （自然终止）
```

**关键约束**：
- `clean` 是所有后续步骤的**通行证**，未通过不得进入 1c 及之后的任何步骤
- CSV 格式验证通过后才允许执行 `split-csv`
- Phase 3 每步完成后询问是否继续，用户拒绝则 Skill 优雅终止

---

## 6. Phase 0：环境确认

### 触发时机

**每次新对话强制执行**，在理解用户意图后、进入任何实质性步骤前。

### 执行内容

```bash
entomokit doctor
```

### 输出处理

| doctor 报告状态 | Skill 行为 |
|---|---|
| 全部依赖 OK | 简短确认，立即进入用户任务 |
| 缺少大量基础依赖（entomokit 本身未安装） | **先询问是否创建虚拟环境**，再给出安装命令（见下方"首次安装引导"） |
| 缺少少量依赖（个别包缺失） | 显示缺失列表 + `pip install` 命令，询问用户安装后继续 |
| 缺少可选依赖（autogluon、SAM3 等） | 说明哪些功能不可用，询问是否继续（仅用可用功能） |
| Python 版本不满足 | 明确告知需要 Python 3.8+，终止并给出安装建议 |

### 首次安装引导（主要包未安装时）

当 doctor 报告缺少大量基础依赖，说明 entomokit 尚未在当前环境安装。此时 Skill **不应直接给出 pip 命令**，而应先询问环境隔离意愿：

> "检测到 entomokit 尚未安装。直接安装到全局 Python 环境可能与其他项目产生依赖冲突，建议先创建独立的虚拟环境。
>
> 你希望怎么做？
> 1. **用 conda 创建虚拟环境（推荐）** — 隔离性最好，适合长期使用
> 2. **用 venv 创建虚拟环境** — 无需 conda，Python 自带
> 3. **直接安装到当前环境** — 快速，但可能产生冲突"

根据用户选择给出对应命令：

**选项 1：conda（推荐）**
```bash
conda create -n entomokit python=3.11 -y
conda activate entomokit
pip install -e .
```

**选项 2：venv**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

**选项 3：直接安装**
```bash
pip install -e .
```

安装完成后提示用户重新运行 doctor 确认环境就绪：
> "安装完成后，请告诉我，我会再次检查环境是否就绪。"

### 面向用户的说明模板

> "我先检查一下你的工作环境是否就绪..."
> （运行 doctor）
> "环境检查完成。[状态摘要]。我们可以开始了。"

---

## 7. Phase 1：数据集准备

### 阶段说明（向用户展示）

> "数据集准备是整个流程的基础。核心目标是确保图片满足三项规范：**格式统一**（jpg/png）、**尺寸一致**（建议短边 ≥ 224px）、**文件命名规范**（无中文、无特殊字符）。这些是后续所有步骤的前提。"

可选步骤根据用户数据情况按需引入：

| 步骤 | 必选/可选 | 适用场景 |
|---|---|---|
| extract-frames | 可选 | 数据来源是视频文件 |
| clean | **必选** | 始终需要 |
| segment | 可选 | 背景复杂、需要提取目标区域或生成标注 |
| synthesize | 可选 | 目标图片数量少、需要增强背景多样性 |
| augment | 可选 | 数据量不足、类别分布不均（长尾问题） |

### 步骤 1a：extract-frames（可选）｜档位 B：展示后直接执行

**触发条件**：用户提及"视频"、"录像"、"mp4"等。

**引导问题**：
1. 视频文件在哪个目录？
2. 希望多久提取一帧？（默认每 1 秒）
3. 是否需要限定时间范围？

**关键参数说明**：

| 参数 | 说明 | 推荐值 |
|---|---|---|
| `--interval` | 提取间隔（毫秒） | 1000（每秒 1 帧） |
| `--out-image-format` | 输出格式 | jpg |
| `--max-frames` | 每个视频最多提取帧数 | 按需 |

**完成标志**：输出目录存在且含有图片文件。提示用户图片总数，询问是否继续 clean。

---

### 步骤 1b：clean（必选，通行证）｜档位 B：展示后直接执行

**这是整个流程的强制门槛。**

**执行前说明**：
> "clean 命令会将你的图片统一为标准格式、去除重复图片、重命名为规范文件名（如 `000001.jpg`）。这一步是后续所有操作的前提，必须通过。"

**引导问题**：
1. 图片在哪个目录？（支持递归扫描子目录）
2. 希望输出图片的短边尺寸是多少？（默认 512px，-1 保持原始）
3. 是否需要去除相似图片？（默认 MD5 精确去重，phash 模糊去重）

**关键参数说明**：

| 参数 | 说明 | 推荐值 |
|---|---|---|
| `--out-short-size` | 短边像素数 | 512（分类任务）；-1（保留原图） |
| `--out-image-format` | 输出格式 | jpg |
| `--dedup-mode` | 去重方式：none/md5/phash | md5 |
| `--phash-threshold` | phash 相似阈值 | 5 |
| `--recursive` | 扫描子目录 | 按需 |

**预检项**（执行前 Skill 主动检查）：
- 输入目录存在且非空
- 目录中确实有图片文件（jpg/jpeg/png/tif/bmp/webp）
- 输出目录路径可写

**完成判断**：
- 输出目录包含图片 -> 通过，展示数量摘要
- 输出目录为空 -> 报错，引导用户检查输入格式
- 发现大量被过滤图片 -> 警告并列出原因（格式不支持、损坏等）

**通行证颁发**：clean 通过后在对话状态中记录 `clean_output_dir`，后续步骤直接引用。

---

### 步骤 1c：segment（可选）｜档位 A：等待用户确认

**触发场景**：
- 用户说"背景太复杂"、"想把虫子从背景里抠出来"
- 用户需要生成 COCO/YOLO/VOC 格式标注
- 需要裁剪 bbox 图用于分类训练（`sam3-bbox` 模式）

**引导问题**：
1. 你想要什么样的输出？
   - a. 透明背景图（alpha 通道）
   - b. 裁剪到目标框（bbox crop，适合分类）
   - c. 生成标注文件（COCO/YOLO/VOC）
2. 有没有 SAM3 模型权重？（`sam3` 方法需要）
3. 无 SAM3 时可选 `otsu`（简单阈值）或 `grabcut`

**关键参数说明**：

| 参数 | 说明 | 推荐值 |
|---|---|---|
| `--segmentation-method` | sam3 / sam3-bbox / otsu / grabcut | sam3-bbox（分类任务） |
| `--sam3-checkpoint` | 模型权重路径 | 必填（sam3/sam3-bbox） |
| `--hint` | SAM3 文本提示 | insect |
| `--padding-ratio` | bbox 外扩比例 | 0.1 |
| `--annotation-format` | coco / voc / yolo | 按需 |

---

### 步骤 1d：synthesize（可选）｜档位 A：等待用户确认

**触发场景**：
- 用户说"目标图片太少"、"背景太单一"、"想合成训练数据"

**前提**：需要有透明背景的目标图（segment 产出的 PNG，或用户自备）。

**引导问题**：
1. 目标图片目录？（需含 alpha 通道的 PNG）
2. 背景图片目录？（自然场景图）
3. 每张目标图合成多少张？（默认 10）

**关键参数说明**：

| 参数 | 说明 | 推荐值 |
|---|---|---|
| `--num-syntheses` | 每目标合成数量 | 10 |
| `--rotate` | 最大旋转角度 | 30 |
| `--avoid-black-regions` | 避免粘贴到暗色区域 | 建议开启 |
| `--color-match-strength` | 色调匹配强度（0-1） | 0.5 |

---

### 步骤 1e：augment（可选）｜档位 A：等待用户确认

**触发场景**：
- 数据量不足
- 类别分布不均（长尾问题，某些类别图片极少）

**引导问题**：
1. 数据量少还是分布不均？（影响 preset 选择）
2. 每张图片想生成几张增强版本？（`--multiply`）

**Preset 选择引导**：

| 场景 | 推荐 Preset | 说明 |
|---|---|---|
| 小数据集，背景简单 | `safe-for-small-dataset` | 保守增强，不引入过多噪声 |
| 通用增强 | `light` / `medium` | 标准翻转+色彩变换 |
| 大幅增加多样性 | `heavy` | 激进变换，适合数据量充足时 |

---

## 8. Phase 2：数据集划分

### CSV 格式准备（教学前置）

**在用户进入 split-csv 前，Skill 必须先展示格式规范。**

#### CSV 格式标准说明

> "split-csv 需要一个 CSV 文件，必须包含两列：`image`（图片路径或文件名）和 `label`（类别名称）。格式如下："

```csv
image,label
images/beetle_001.jpg,Chrysomelidae
images/beetle_002.jpg,Chrysomelidae
images/moth_001.jpg,Sphingidae
images/moth_002.jpg,Sphingidae
```

**关键约束说明**（必须向用户明确）：

| 项目 | 约束 | 常见错误 |
|---|---|---|
| 列名 | 必须是 `image` 和 `label`（区分大小写） | `Image`、`filename`、`class` 均不可接受 |
| image 列 | 相对路径或文件名，需与 `--images-dir` 配合 | 使用绝对路径会导致跨机器不兼容 |
| label 列 | 字符串类别名，避免特殊字符和空格 | 用 `_` 替代空格，如 `Carabus_auratus` |
| 编码 | UTF-8，无 BOM | Excel 默认保存可能含 BOM |
| 文件扩展名 | `.csv` | `.xlsx` 不被接受 |

#### CSV 验证（执行 split-csv 前的预检）

Skill 在执行 split-csv 前自动验证：

1. **列名检查**：必须同时存在 `image` 和 `label` 列
2. **空值检查**：两列均不允许空值
3. **路径可达性**：如提供 `--images-dir`，抽样验证 image 路径可访问
4. **类分布报告**：展示每个类别的样本数，标记样本数 < 10 的类别

**验证失败处理**：

| 错误类型 | 提示内容 | 修复建议 |
|---|---|---|
| 缺少 `image` 列 | "未找到 `image` 列，发现列名：[实际列名]" | 重命名列或检查分隔符 |
| 缺少 `label` 列 | "未找到 `label` 列，发现列名：[实际列名]" | 重命名列 |
| 存在空值 | "第 [行号] 行存在空值" | 填充或删除对应行 |
| 类别样本过少 | "[类别名] 只有 [N] 张图，建议至少 10 张" | 警告（不阻断），用户确认继续 |

---

### split-csv 执行｜档位 B：展示后直接执行

**引导问题**：
1. CSV 文件路径？
2. 图片根目录？（如 image 列是相对路径）
3. 划分比例？（推荐 train 80% / val 10% / test 10%）
4. 是否有希望完全排除在训练集之外的"未知类"用于开放集评估？
5. 是否需要过滤样本数太少的类别？（`--min-count-per-class`）

**划分策略说明**：

| 参数 | 说明 | 推荐 |
|---|---|---|
| `--val-ratio` | 验证集比例 | 0.1 |
| `--known-test-classes-ratio` | 已知类测试比例 | 0.1 |
| `--unknown-test-classes-ratio` | 未知类测试比例（开放集） | 0（一般情况） |
| `--min-count-per-class` | 样本数低于此值的类别被过滤 | 10 |

**完成后展示**：

```
数据集划分完成：
- train.csv:       1200 张，12 个类别
- val.csv:          150 张，12 个类别
- test.known.csv:   150 张，12 个类别
输出目录：datasets/split/
```

询问用户是否继续进入 Phase 3 训练。

---

## 9. Phase 3：分类训练与评估（渐进拓展）

Phase 3 采用**渐进推进**模式：每个子步骤完成后，Skill 主动询问是否继续下一步。用户拒绝则优雅终止，不强制推进。

### classify train｜档位 A：等待用户确认

**前置检查**：
- `train.csv` 存在（由 split-csv 产生）
- 已安装 classify 依赖：`pip install -e ".[classify]"`

**引导问题**：
1. 使用哪个骨干网络？（默认 `convnextv2_femto`，轻量推荐）
2. 训练多少 epoch？（默认 50）
3. 是否使用 GPU？（Skill 根据 doctor 结果自动建议）
4. 数据类别是否严重不均衡？（是则建议开启 `--focal-loss`）

**骨干网络选择引导**：

| 场景 | 推荐模型 | 说明 |
|---|---|---|
| 快速验证/CPU 训练 | `convnextv2_femto` | 轻量，速度快 |
| 精度优先（GPU） | `convnextv2_tiny` / `efficientnet_b3` | 平衡精度与速度 |
| 高精度研究 | `vit_base_patch16_224` | 需要充足数据和算力 |

**完成后展示**：
```
训练完成：
- 最终验证集准确率：xx.x%
- 模型保存路径：runs/exp1/AutogluonModels/convnextv2_femto
- 训练耗时：xx 分钟
```

询问：**"是否继续用测试集评估模型效果？"**

---

### classify predict｜档位 B：展示后直接执行

**适用场景**：在新图片上运行推理。

**引导问题**：
1. 推理目标：目录（所有图片）还是 CSV（指定图片列表）？
2. 使用 AutoGluon 模型还是 ONNX 模型？

**完成后展示**：预测结果摘要（前 N 条 + 输出 CSV 路径）。

询问：**"是否继续对测试集做完整评估并生成评估报告？"**

---

### classify evaluate｜档位 B：展示后直接执行

**完成后展示**（面向用户的指标解读）：

| 指标 | 含义 | 参考阈值 |
|---|---|---|
| Accuracy | 整体准确率 | > 0.85 为良好 |
| Balanced Accuracy | 各类别平衡后的准确率（适合不均衡数据） | > 0.80 为良好 |
| F1 (macro) | 综合精确率与召回率 | > 0.80 为良好 |
| MCC | 综合指标（对不均衡数据更可靠） | > 0.75 为良好 |

询问：**"是否继续生成嵌入可视化和 GradCAM 热图，帮助理解模型行为？"**

---

### classify embed + cam（可选）｜档位 B：展示后直接执行

**embed**：生成特征嵌入 + UMAP 可视化，帮助发现类别混淆。

**cam**：生成 GradCAM 热图，可视化模型关注区域，用于判断模型是否学习到正确特征。

询问：**"是否将模型导出为 ONNX 格式，用于部署或无 GPU 环境推理？"**

---

### classify export-onnx（可选，部署终点）｜档位 B：展示后直接执行

**适用场景**：
- 部署到无 AutoGluon 环境
- 跨平台部署
- 推理加速

**完成后展示**：
```
ONNX 导出完成：
- 模型文件：runs/onnx/model.onnx
- 类别映射：runs/onnx/label_classes.json
- 推理方式：entomokit classify predict --onnx-model runs/onnx/model.onnx ...
```

**自然终止**：告知用户完整流程已完成，并展示整个 run 的阶段摘要。

---

## 10. 错误处理与恢复

### 错误码矩阵

| 错误码 | 含义 | 自动动作 | 用户动作 |
|---|---|---|---|
| E-ENV-MISSING | 依赖缺失 | 输出安装命令 | 安装后重试 |
| E-INPUT-EMPTY | 输入目录为空 | 列出目录内容 | 检查路径 |
| E-FILE-FORMAT | 非支持格式 | 列出不支持文件 | 替换或转换 |
| E-CSV-COLUMN | CSV 缺少必要列 | 显示实际列名 | 重命名列 |
| E-CSV-NULL | CSV 含空值 | 定位到行列 | 填充或删除 |
| E-PATH-NOT-FOUND | 路径不可达 | 建议标准化路径 | 重新输入 |
| E-CLASS-IMBALANCE | 类别严重不均衡 | 给出分布统计 | 确认继续或调整数据 |
| E-CLI-FAILED | CLI 执行失败 | 提取关键日志 | 按建议修复后重试 |
| E-OUTPUT-MISSING | 执行完成但无产物 | 生成诊断报告 | 调整参数重跑 |
| E-CHECKPOINT-MISSING | SAM3 权重缺失 | 输出下载指引 | 下载后重试 |

### 错误反馈原则

- 不暴露原始堆栈给用户
- 必须包含：**是什么问题 + 在哪里 + 怎么修复**
- 修复建议必须是可直接执行的具体操作（命令或文件修改）

---

## 11. 阶段状态跟踪与跨会话持久化

### 10.1 会话内状态

Skill 在对话过程中维护内存状态，记录每步的输入/输出路径，供后续步骤直接引用，避免用户重复输入：

```yaml
session_state:
  phase0_doctor_passed: bool
  phase1_clean_output_dir: str | null   # 通行证
  phase1_segment_output_dir: str | null
  phase2_csv_path: str | null
  phase2_split_output_dir: str | null
  phase3_model_dir: str | null
  phase3_onnx_path: str | null
  current_phase: "0" | "1a" | "1b" | "1c" | "1d" | "1e" | "2" | "3-train" | "3-predict" | "3-eval" | "3-embed" | "3-onnx"
```

### 10.2 跨会话持久化：entomokit_progress.json

**目的**：整个流程横跨数天是常态（今天清理图片，明天训练），新对话开始时 Skill 必须能恢复上次的进度，避免用户重新解释"我上次做到哪一步了"。

**文件位置**：用户项目工作目录下，`entomokit_progress.json`（不隐藏，方便用户直接查看和手动编辑）。

**读写时机**：

| 时机 | 动作 |
|---|---|
| 对话开始 | 检查工作目录是否存在该文件，存在则读取并展示进度摘要 |
| 每步成功完成后 | 立即更新对应步骤的状态和路径 |
| 用户中途停止 | 保留已完成步骤，`current_phase` 指向下一步 |
| 用户明确说"重新开始" | 覆盖文件，清空所有步骤状态 |

**文件 Schema**：

```json
{
  "version": "1",
  "working_dir": "/data/my_project",
  "last_updated": "2026-03-26T14:32:00",
  "current_phase": "3-train",
  "steps": {
    "doctor": {
      "status": "passed",
      "timestamp": "2026-03-26T10:00:00"
    },
    "clean": {
      "status": "passed",
      "input_dir": "images/raw/",
      "output_dir": "images/cleaned/",
      "image_count": 1240,
      "timestamp": "2026-03-26T10:15:00"
    },
    "segment": {
      "status": "skipped"
    },
    "synthesize": {
      "status": "skipped"
    },
    "augment": {
      "status": "skipped"
    },
    "split_csv": {
      "status": "passed",
      "csv_path": "data/labels.csv",
      "output_dir": "datasets/split/",
      "train_count": 992,
      "val_count": 124,
      "test_count": 124,
      "timestamp": "2026-03-26T11:00:00"
    },
    "classify_train": {
      "status": "in_progress",
      "model_dir": "runs/exp1/",
      "timestamp": "2026-03-26T14:00:00"
    },
    "classify_predict": { "status": "pending" },
    "classify_evaluate": { "status": "pending" },
    "classify_embed": { "status": "pending" },
    "classify_cam": { "status": "pending" },
    "classify_export_onnx": { "status": "pending" }
  }
}
```

**步骤状态值**：`pending` / `in_progress` / `passed` / `failed` / `skipped`

**对话开始时的恢复提示**：

> "发现上次的进度记录（entomokit_progress.json）：
> - ✓ 图片清理完成（1240 张，输出：images/cleaned/）
> - ✓ 数据集划分完成（train 992 / val 124 / test 124）
> - ⏳ 上次正在训练模型（runs/exp1/）
>
> 是否继续上次的进度？还是重新开始？"

**直接模式下的状态利用**：用户说"我要做 segment"，Skill 先查文件：
- 若 `clean.status == "passed"`：直接使用 `clean.output_dir` 作为 segment 的输入，无需再询问
- 若 `clean.status` 不存在或非 passed：提示 clean 尚未完成，按降级策略询问

---

## 12. 与现有仓库的对齐

| Skill 概念 | 仓库对应 |
|---|---|
| 环境检查 | `entomokit doctor` |
| 通行证（clean） | `entomokit clean` |
| CSV 模板 | `data/` 目录示例数据 |
| 字段约束说明 | `README.md` split-csv 参数说明 |
| 骨干网络选择 | `README.md` 支持的 timm 骨干列表 |
| 错误码 | `src/common/` 共享工具层 |

编排层与执行层保持解耦：对话逻辑不侵入 CLI 实现。

---

## 13. data/ 资产复用（按需展示 / 测试 / 教学）

`data/` 目录是 Skill 的**备用示例资产池**，用于演示、回归测试与教学。

**优先级规则（必须执行）**：

1. 默认使用用户当前任务数据（第一优先）。
2. 仅当用户提出"演示给我看"、"给个示例"、"帮我做教学"、"复现这个错误"等诉求时，才启用 `data/`。
3. 启用前先明确说明："下面示例使用仓库 `data/`，不会替换你的真实数据流程。"

### 触发式告知点（让用户知道有示例能力）

Skill 在以下节点给出一次轻提示，确保用户知道可以随时用 `data/` 快速体验：

1. Phase 0 完成后（环境通过时）：提示“如果你想先熟悉流程，我可以用仓库 `data/` 跑一遍演示”。
2. 用户在某一步卡住时（路径缺失、CSV 不规范、权重缺失）：提示“可先用 `data/` 做最小可运行示例，再切回你的数据”。
3. Phase 2 CSV 教学前：提示“可用 `data/Epidorcus/figs.csv` 先看标准格式与 split 结果”。
4. 用户明确询问“怎么做/能不能先示范”时：直接提供 `data/` 演示分支。

告知模板（短版）：

> "你也可以先用仓库 `data/` 跑一个最小示例，快速感受这一步（不影响你自己的数据流程）。如果需要我现在就用示例跑一遍。"

### 资产映射

| 资产 | 用途 | 对应阶段 |
|---|---|---|
| `data/video.mp4` | `extract-frames` 演示输入 | Phase 1a |
| `data/insects/` | 小规模图片目录（快速演示、故障复现） | Phase 1b / 1c / 3-predict |
| `data/segment/` | 分割产物示例（PNG、COCO、log） | Phase 1c 教学与验收说明 |
| `data/Epidorcus/images/` | 分类训练图片目录（中等规模） | Phase 2 / Phase 3 |
| `data/Epidorcus/figs.csv` | 标准 `image,label` CSV 示例 | Phase 2 CSV 规范教学 |

### 教学示例（默认顺序）

1. 用 `data/video.mp4` 演示视频提帧（让用户理解 Phase 1a 的输入输出）。
2. 用 `data/insects/` 演示 clean 与 segment 的差异（规范化 vs 目标提取）。
3. 用 `data/Epidorcus/figs.csv` + `data/Epidorcus/images/` 演示 split-csv 与 train 的最小闭环。
4. 用 `data/segment/annotations.coco.json` 展示标注结构，解释 COCO 字段含义。

### 回归测试用例（首批）

| 用例 ID | 输入 | 期望结果 |
|---|---|---|
| T1-clean-pass | `data/insects/` | clean 成功，输出目录非空，生成 `log.txt` |
| T2-csv-pass | `data/Epidorcus/figs.csv` | CSV 预检通过，可执行 split-csv |
| T3-segment-pass | `data/insects/` + sam3 权重 | 生成 `images/*.png` 与 `annotations.coco.json` |
| T4-train-smoke | `split-csv` 输出 + `convnextv2_femto` | 训练可启动并产出模型目录 |
| T5-predict-smoke | `data/insects/` + 已训练模型 | 生成预测结果 CSV |

### 规范要求

- `data/` 仅作示例与测试基线，不应在 Skill 运行中被原地覆盖；执行时统一写入用户指定 `--out-dir`。
- 教学场景优先使用小数据（`data/insects/`），降低首次体验耗时。
- 文档中的 CSV 示例优先引用 `data/Epidorcus/figs.csv`，避免示例与仓库真实数据不一致。
- 示例结束后，Skill 必须主动提示并切回用户数据上下文（复述用户输入目录/CSV 路径后再继续）。

---

## 14. Skill 打包结构建议（面向 skill-creator）

为保证后续实现符合 Skill 最佳实践，建议将本设计拆分为以下结构：

```text
entomokit-workflow-skill/
├── SKILL.md
├── references/
│   ├── workflow.md
│   ├── command-profiles.md
│   ├── csv-validation.md
│   ├── error-catalog.md
│   └── teaching-playbook.md
└── assets/
    └── examples/
        └── (软链接或复制 data/ 中最小可用样例)
```

拆分原则：

- `SKILL.md` 仅保留触发条件、入口模式决策、Phase 主流程与硬性门禁（clean/CSV/doctor）。
- 参数细节、错误码、教学话术放入 `references/`，按需加载，避免主 Skill 过长。
- 演示素材放入 `assets/examples/`（或明确映射到仓库 `data/`），确保教学与测试可复现。

### SKILL.md 骨架中的决策规则（用户数据优先）

在 `SKILL.md` 主体中固定加入以下 5 条规则，避免误把 Skill 做成纯教学模式：

1. 默认使用用户提供的输入目录、CSV、输出目录，不主动替换为 `data/`。
2. 仅当用户明确提出"演示 / 示例 / 教学 / 复现"时，才可启用 `data/` 示例资产。
3. 启用 `data/` 前先声明："以下为示例流程，不会替换你的真实数据流程。"
4. 示例执行结束后，立即复述用户真实路径并切回用户数据继续。
5. 任何会写文件的命令都写入用户项目 `--out-dir`，禁止覆盖 `data/` 原始样例。

建议在 SKILL.md 中以简短决策块呈现（可直接复用）：

```markdown
## Data Source Decision

- If user has provided task data, use user data.
- If user explicitly asks for demo/teaching/troubleshooting example, allow `data/` examples.
- Before using `data/`, state it is a demo path and will not replace user workflow.
- After demo step, switch back to user-provided paths immediately.
- Never write outputs into `data/`; always write to user `--out-dir`.
```

---

## 15. 非目标（当前版本不做）

1. 全图形化前端
2. 自动修复所有数据问题（首版以"可定位 + 可引导修复"为主）
3. 多任务并行执行
4. 用户偏好记忆（跨项目的全局配置）

---

## 16. 实施优先级

| 优先级 | 内容 |
|---|---|
| P0（必须） | Phase 0 doctor 集成；双模式入口识别；Phase 1b clean 通行证 + 直接模式降级；Phase 2 CSV 验证 + split-csv；`entomokit_progress.json` 读写 |
| P1（应该） | Phase 1a extract-frames；Phase 3 train + evaluate；错误码矩阵；进度恢复提示 |
| P2（可以） | Phase 1c segment；Phase 1d synthesize；Phase 1e augment；Phase 3 predict + embed + cam + export-onnx |
| P3（将来） | 用户偏好记忆（跨项目全局配置） |
