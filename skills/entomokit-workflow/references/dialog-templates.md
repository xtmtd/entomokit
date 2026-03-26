# Fixed Dialogue Templates

Use these templates for every step. Keep wording short and consistent.

## 1) Pre-Run Parameter Card (must approve before run)

```text
步骤：<step-name>
目的：<one-line intent>
输出目录：runs/runNNN/<step-output>/

参数卡：
（必须列出该命令全部可由用户设置的参数；可按“基础/高级”分组，但不能只给 3 个）
- <param-1>
  - 含义：<what it controls>
  - 可选：<allowed options/range>
  - 当前：<selected value>
- <param-2>
  - 含义：...
  - 可选：...
  - 当前：...

参数来源：runtime CLI schema | help fallback
校验状态：passed | blocked（blocked 时列出具体错误与可选修复值）

请确认本步：
1) 批准执行
2) 修改参数（请说参数名+新值）
3) 取消本步
```

## 2) Post-Run Result Card (must approve before next step)

```text
步骤结果：<step-name>
状态：成功 | 失败 | 部分成功

主要结果：
- <headline metric/result 1>
- <headline metric/result 2>

产物路径：
- <artifact-1 path>
- <artifact-2 path>

下一步请选择：
1) 继续到建议下一步（仅建议，不自动执行）
2) 重跑本步（调整参数）
3) 停在当前并结束
```

## 3) clean Special Template

```text
步骤：clean
目的：统一图片格式/尺寸/命名，并去重。
输出目录：runs/runNNN/clean-runNNN/

参数卡：
（以下仅为示例；实际展示时必须给出 clean 的完整可调参数）
- --recursive
  - 含义：是否扫描子目录
  - 可选：true | false
  - 当前：<true/false>
- --out-short-size
  - 含义：输出短边像素
  - 可选：-1 或 >=224
  - 当前：<value>
- --dedup-mode
  - 含义：去重策略
  - 可选：none | md5 | phash
  - 当前：<value>
- --<other-clean-param>
  - 含义：...
  - 可选：...
  - 当前：...

请确认本步：
1) 批准执行
2) 修改参数
3) 取消本步
```

## 4) classify train Special Template

```text
步骤：classify train
目的：训练分类模型。
输出目录：runs/runNNN/train-runNNN/

参数卡：
（以下仅为示例；实际展示时必须给出 classify train 的完整可调参数）
- --device
  - 含义：训练硬件后端
  - 可选：<doctor-detected options, e.g. mps|cpu>
  - 当前：<selected>
- --base-model
  - 含义：骨干网络
  - 可选：convnextv2_femto | convnextv2_base | eva02_tiny | eva02_base | ...
  - 当前：<selected>
- --max-epochs
  - 含义：训练轮数
  - 可选：>=1
  - 当前：<selected>
- --<other-train-param>
  - 含义：...
  - 可选：...
  - 当前：...

请确认本步：
1) 批准执行
2) 修改参数
3) 取消本步
```

After train completes, always show Post-Run Result Card and wait for explicit user approval before `predict` or `evaluate`.
