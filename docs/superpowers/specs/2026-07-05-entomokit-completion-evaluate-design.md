# entomokit completion 与 classify evaluate 增强设计

**日期**: 2026-07-05  
**状态**: 已确认，待实现

---

## 1. 背景

当前 `entomokit` 的自动补全通过 `argcomplete` 动态注入到 shell 启动脚本：

- `entomokit/main.py` 的 `--install-completion` 会把 `register-python-argcomplete entomokit` 写入 `~/.zshrc` 或 `~/.bashrc`
- 用户默认启动环境是 `base`，而 `register-python-argcomplete` 位于 `conda activate entomokit` 后的环境中
- 因此每次 `source ~/.zshrc` 都可能报错：`command not found: register-python-argcomplete`

这类问题不是补全本身失效，而是 shell 启动阶段被绑定到了某个特定 Python 环境。

同时，`classify evaluate` 当前只输出总体指标，缺少类别级诊断信息。昆虫分类中常见问题不是“整体很差”，而是“少数近缘种互相混淆明显”，因此需要最小增加能定位问题类别的输出。

---

## 2. 目标

本次只做两件事：

1. 去掉 `entomokit` 自动补全对 shell 启动时 Python 环境的依赖
2. 在保留总体指标的前提下，为 `classify evaluate` 增加类别级诊断产物

非目标：

- 不迁移 CLI 框架到 `Typer` 或 `Click`
- 不重写现有命令树、帮助文本或参数体系
- 不新增复杂交互式报告页面
- 不改变 `classify evaluate` 的核心指标定义

---

## 3. completion 方案对比

### 方案 A：保留 argparse，新增静态 completion 子命令（推荐）

做法：

- 保留现有 `argparse` CLI 架构
- 新增 `entomokit completion bash|zsh|fish`
- 命令输出静态 shell completion 脚本到 stdout
- 可选支持 `--install`，将静态脚本写到目标位置
- 移除当前把 `register-python-argcomplete ...` 直接写入 shell rc 的行为

优点：

- 不引入 `click` / `typer` 新依赖
- 可以彻底移除运行期 `argcomplete` 依赖
- 最小改动解决当前 `base` 环境报错
- 与 `phyloAI` 的“显式生成脚本”思路一致

缺点：

- 需要自己维护一份补全脚本生成逻辑

### 方案 B：保留 argparse，继续使用 argcomplete，但只在安装时使用

做法：

- CLI 仍用 `argparse`
- `entomokit completion ...` 或 `--install-completion` 在执行时调用 `argcomplete` 生成脚本
- shell rc 只 source 静态文件，不再直接执行 `register-python-argcomplete`

优点：

- 改动更小
- 复用 `argcomplete` 现成能力

缺点：

- 项目仍保留第三方依赖 `argcomplete`
- 只是把“启动时依赖”改成“安装时依赖”

### 方案 C：迁移到 Typer/Click

做法：

- 将顶层 CLI 与子命令改写为 `Typer` 或 `Click`
- 使用框架自带 completion 机制

优点：

- completion 与帮助系统更统一

缺点：

- 需要新增第三方依赖
- 会牵动命令注册、测试、schema 导出与帮助文本
- 对当前问题属于过度修复

### 结论

采用方案 A：**保留 `argparse`，实现静态 completion 子命令，不再依赖 `argcomplete` 注入 shell 启动脚本。**

---

## 4. completion 设计

### 4.1 命令接口

新增顶层命令组：

```bash
entomokit completion bash
entomokit completion zsh
entomokit completion fish
```

默认行为：

- 将对应 shell 的静态 completion 脚本打印到 stdout
- 用户可以自行重定向保存

示例：

```bash
entomokit completion zsh > ~/.zfunc/_entomokit
```

可选行为：

```bash
entomokit completion zsh --install
```

`--install` 为便捷选项，负责：

- `bash`: 写入用户 completion 目录或用户文件
- `zsh`: 写入 `~/.zfunc/_entomokit`
- `fish`: 写入 `~/.config/fish/completions/entomokit.fish`

本次实现允许 `--install` 仅覆盖最常见用户级安装路径，不做复杂系统级分支。

### 4.2 shell rc 策略

本次不再让 `entomokit` 自动向 `~/.zshrc` / `~/.bashrc` 写入如下动态命令：

```bash
eval "$(register-python-argcomplete entomokit)"
```

原因：

- 这会让 shell 启动绑定到当前 PATH 中某个 Python 环境
- 当命令不在当前环境时，启动脚本直接报错

新的策略：

- `entomokit` 只负责输出或安装静态脚本
- shell rc 只在需要时 source 静态文件
- 不在 shell 启动时执行任何 `entomokit` 专属 Python 辅助命令

### 4.3 与现有 `--install-completion` 的关系

保留 `--install-completion` 一个过渡版本，但行为改变：

- 内部不再写入 `register-python-argcomplete ...`
- 内部改为调用新的静态安装逻辑
- 终端提示用户后续优先使用 `entomokit completion <shell> --install`

这样做的原因：

- 减少已有文档或用户习惯的断裂
- 让迁移平滑

### 4.4 依赖策略

实现目标是：

- completion 功能仅依赖标准库 + 当前 CLI 命令树元数据
- 移除对 `argcomplete` 的运行时要求
- 若代码中不再需要 `argcomplete`，则从项目依赖与提示文案中移除

`argparse` 是 Python 标准库，保留即可；`click` / `typer` / `argcomplete` 都不是标准库，本次不为 completion 引入它们。

---

## 5. classify evaluate 增强设计

### 5.1 保留现有总体输出

当前行为保留：

- 继续计算总体指标
- 继续写出 `evaluations.csv`
- 继续在终端打印总体指标摘要

这保证已有使用者仍然能拿到整体质量判断。

### 5.2 新增输出文件

在 `out_dir/` 下最小增加以下产物：

```text
out_dir/
├── evaluations.csv
├── confusion_matrix.csv
├── confusion_matrix_normalized.csv
├── per_class_metrics.csv
└── confusion_matrix.pdf      # 仅类数不大时生成
```

定义如下：

- `confusion_matrix.csv`
  - 原始计数矩阵
  - 行表示真实类别，列表示预测类别
  - 这是最核心的诊断文件

- `confusion_matrix_normalized.csv`
  - 按真实类别行归一化
  - 每行和约为 1
  - 主要用于观察每类召回表现与易混淆去向

- `per_class_metrics.csv`
  - 每类 `precision` / `recall` / `f1-score` / `support`
  - 直接基于 `sklearn.metrics.classification_report(..., output_dict=True)` 展开

- `confusion_matrix.pdf`
  - 小到中等类别数时生成热图
  - 类别数过多时可读性很差，因此允许跳过

### 5.3 类别顺序

所有类别级输出应使用同一套类别顺序，保证：

- `confusion_matrix.csv`
- `confusion_matrix_normalized.csv`
- `per_class_metrics.csv`
- `confusion_matrix.pdf`

之间可直接对照。

顺序规则：

1. 若 ONNX sidecar 或模型元数据提供稳定类别顺序，则优先使用该顺序
2. 否则使用 `y_true ∪ y_pred` 的稳定排序结果

不按输入出现顺序拼凑，以避免不同运行间结果顺序漂移。

### 5.4 PDF 生成阈值

`confusion_matrix.pdf` 只在类别数不大时生成。

本次采用固定阈值：

- `num_classes <= 50`：生成 PDF
- `num_classes > 50`：跳过 PDF，仅保留 CSV

跳过时在终端打印提示，说明原因即可，不新增 CLI 参数。

这样做的原因：

- 先满足最常见场景
- 避免为一个简单阈值过早增加配置项

### 5.5 AutoGluon 与 ONNX 统一行为

无论使用：

- `--model-dir`（AutoGluon）
- `--onnx-model`（ONNX）

都应产出同样的四类新增文件。

实现上，评估流程统一整理出：

- `y_true`
- `y_pred`
- `proba`（若可得）
- `class_labels`

再交给共享的评估产物写出逻辑。

这样可以避免在 CLI 层分别复制两套 confusion matrix / per-class metrics 逻辑。

---

## 6. 代码组织建议

为保持最小改动，建议：

- `entomokit/main.py`
  - 移除现有动态 completion 安装逻辑
  - 注册新的 `completion` 顶层命令组或等价子模块

- `entomokit/completion.py`
  - 放置 `bash` / `zsh` / `fish` completion 输出与 `--install` 逻辑

- `src/classification/evaluator.py`
  - 扩展为同时返回总体指标与类别级评估中间结果
  - 增加 confusion matrix / per-class metrics / PDF 写出辅助函数

- `entomokit/classify/evaluate.py`
  - 继续负责 CLI 参数解析与调用
  - 调用共享评估逻辑写出全部产物

不新增额外抽象层，不拆成过多模块；只在当前边界上补一层最小能力。

---

## 7. 测试策略

### 7.1 completion

新增测试覆盖：

- `entomokit completion bash --help` / `zsh --help` / `fish --help`
- 各 shell 命令会输出非空脚本
- 输出内容包含正确程序名 `entomokit`
- `--install-completion` 走新的静态安装路径
- 不再断言 `argcomplete` 激活逻辑存在

### 7.2 classify evaluate

新增测试覆盖：

- `run()` 会写出 `evaluations.csv`
- `run()` 会写出 `confusion_matrix.csv`
- `run()` 会写出 `confusion_matrix_normalized.csv`
- `run()` 会写出 `per_class_metrics.csv`
- 类数小于等于阈值时会写出 `confusion_matrix.pdf`
- 类数大于阈值时不会写出 PDF，但不会报错
- ONNX 字符串标签映射后，类别级产物顺序与标签一致

---

## 8. 风险与控制

### 8.1 completion 脚本正确性风险

风险：自写静态 completion 脚本可能不如现成框架全面。

控制：

- 本次只覆盖当前 CLI 的主命令、子命令和显式枚举参数
- 先保证稳定可用，不追求动态路径补全等高级特性

### 8.2 类别太多导致 PDF 不可读

风险：类别数大时热图极度拥挤。

控制：

- 固定阈值 50 类
- 超阈值自动跳过，仅保留 CSV

### 8.3 结果顺序不稳定

风险：不同后端或不同运行导致类别顺序变化，影响比较。

控制：

- 明确统一类别顺序规则
- 所有类别级文件共用同一顺序

---

## 9. 最终决策

本次实现按以下决策推进：

1. 保留 `argparse`，不迁移到 `Typer` / `Click`
2. 移除基于 `register-python-argcomplete` 的 shell 启动注入方式
3. 新增 `entomokit completion bash|zsh|fish` 静态补全命令
4. 尽量移除 `argcomplete` 依赖，不为 completion 保留第三方包
5. `classify evaluate` 在保留总体指标基础上，新增四类诊断产物
6. `confusion_matrix.pdf` 仅在类别数不超过 50 时生成
