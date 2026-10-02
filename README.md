# Insect Dataset Toolkit (EntomoKit)
[中文文档](README.cn.md) | **English**

A Python-based toolkit for building insect image datasets. EntomoKit provides a
unified `entomokit` CLI for frame extraction, segmentation, morphology
measurement, synthesis, cleaning, augmentation, dataset splitting, AutoMM
classification and environment diagnostics, plus an `entomokit-workflow` skill for
AI assistants.

## Workflow Overview

The pipeline is composable; only the steps your data needs are required.

1. `extract-frames` — turn videos into still frames.
2. `segment` — cut insects out of images and optionally write annotations.
3. `measure` — optional: morphology metrics from segmentation masks.
4. `synthesize` — optional: composite RGBA cutouts onto new backgrounds.
5. `clean` — resize, pad and deduplicate images.
6. `augment` — optional: expand a small training set.
7. `split-csv` — build train/val/test CSVs.
8. `classify train` → `classify predict` / `classify evaluate` / `classify embed` /
   `classify cam` / `classify export-onnx`.

You can start at any step: for example go straight from `clean` to `classify train`
when the images are already labelled, or use `segment` alone for mask extraction.
AI-assisted runs follow the skill's own guided policy, not this list.

## Requirements and Installation

- Python 3.9+
- Linux, macOS or Windows

```bash
git clone https://github.com/xtmtd/entomokit.git
cd entomokit
```

Install into an isolated environment (recommended):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Equivalent alternatives: `python -m venv .venv && source .venv/bin/activate`, or
`conda create -n entomokit python=3.11 -y && conda activate entomokit`. Installing
directly into a global environment is possible but risks dependency conflicts.

Command extras:

| Extra | Adds |
|---|---|
| `.[video]` | `extract-frames` (OpenCV video decoding) |
| `.[segmentation]` | SAM3 segmentation, SAM3 repair and synthesis annotation output |
| `.[measurement]` | `measure` (morphology metrics, skeleton computation) |
| `.[synthesis]` | `synthesize` (polygon simplification, COCO annotations) |
| `.[cleaning]` | `clean` perceptual-hash dedup |
| `.[augment]` | `augment` (albumentations) |
| `.[classify]` | `classify train/predict/evaluate/embed/cam/export-onnx` (AutoMM, timm, GradCAM, UMAP, ONNX) |
| `.[dev]` | pytest and coverage |

`measure` and `synthesize` also import `scikit-image`, which neither the
`measurement` nor the `synthesis` extra declares; install `.[segmentation]` (which
declares it) or add `scikit-image` explicitly alongside those extras.

Install several at once, for example the full development set:

```bash
uv pip install --only-binary :all: stringzilla
pip install -e ".[dev,classify,segmentation,synthesis,measurement,video,cleaning,augment]"
```

The `stringzilla` line above is required before installing `.[augment]` on platforms
where only its binary wheel is available. AutoMM publishes its own install notes at
https://auto.gluon.ai/stable/install.html.

## Quick Start

The example below needs the `video` extra and writes frames into `./frames`:

```bash
uv pip install -e ".[video]"
entomokit extract-frames --input-dir ./videos --out-dir ./frames
```

Then clean the frames and build a split for classification:

```bash
entomokit clean --input-dir ./frames --out-dir ./cleaned
entomokit split-csv --raw-image-csv ./labels.csv --out-dir ./datasets
```

<a id="doctor-command"></a>
## Doctor Command

Diagnose environment and dependency readiness:

```bash
entomokit doctor
```

The report includes Python and available devices (`cpu`, `cuda`, `mps`), key
package versions and status (ok/missing/outdated), and install or upgrade
recommendations. The `classify` extra requires `autogluon.multimodal>=1.5.0`, while
`doctor` currently accepts `1.4.0` as up to date and only recommends an upgrade below
that. `doctor` has no user-settable options.

<a id="update-command"></a>
## Update Command

Check whether a newer version is available on GitHub (via `version.txt` on the
`main` branch) and optionally install it. Works for all users regardless of local
git history.

```bash
entomokit update           # check and prompt
entomokit update --check   # check only, no install
entomokit update --yes     # install without prompt
```

| Parameter | Description | Default |
|---|---|---|
| `--check` | Only show version info; do not install | No |
| `--yes`, `-y` | Skip the confirmation prompt | No |

<a id="completion-command"></a>
## Shell Completion

Generate static completion scripts for bash, zsh or fish:

| Shell | Option | Install path |
|---|---|---|
| `bash` | `--install` | `~/.local/share/bash-completion/completions/entomokit` |
| `zsh` | `--install` | `~/.zfunc/_entomokit` |
| `fish` | `--install` | `~/.config/fish/completions/entomokit.fish` |

Without `--install` the script is printed to stdout. For zsh, ensure your shell
config includes `fpath=(~/.zfunc $fpath)` and initialises completions with
`autoload -Uz compinit && compinit`.

<a id="directory-policy"></a>
## Directory Input and Output Policy

This policy applies to the directory-oriented commands (`extract-frames`, `segment`,
`measure`, `synthesize`, `clean`, `augment`) and to directory discovery in
`classify embed`, `classify cam` and `classify predict`:

- Directory-oriented commands scan recursively by default; there is no
  `--recursive` flag and no flatten option.
- Ordinary file outputs mirror each input's directory structure relative to the input
  root, but some commands also rewrite the file name: `clean` normalises the stem,
  guarantees a unique name inside a directory and applies `--out-image-format`, and
  `augment` appends `_augN`, zero-padded to the width of `--multiply` (`_aug1` for
  1–9, `_aug01` for 10–99, `_aug001` for 100). Example: `clean --input-dir in --out-dir out` turns
  `in/beetles/a.png` into `out/cleaned_images/beetles/a.jpg` with the default format,
  so same-named files in different subdirectories stay distinct while a second `a.tif`
  in the same directory becomes `a_1.jpg`.
- `segment` is the deliberate exception: it does not mirror input paths. It
  flattens every image into `images/` and encodes each input-relative path into a
  unique flat sample ID (readable stem plus a short path digest, for example
  `a__4cabcf2b3682`), with per-format annotation directories alongside. Standard
  dataset layouts (for example Pascal VOC `JPEGImages/`) are produced by a later
  conversion or split step, not directly by `segment`.
- CSV-driven commands (`split-csv`, and classification commands given
  `--input-csv`) stay CSV-driven and are outside this policy.
- A non-empty `--out-dir` stops with an error. Commands that expose `--resume`
  accept it to continue; commands without `--resume` require `--overwrite` to
  start fresh.

<a id="common-behaviours"></a>
## Common Behaviours

- **Logging**: commands with an output directory write `log.txt` there: `extract-frames`,
  `segment`, `measure`, `synthesize`, `clean`, `augment`, `split-csv`,
  `classify predict`, `classify evaluate`, `classify cam` and `classify export-onnx`
  write `out-dir/log.txt`, while `classify train` and `classify embed` write
  `out-dir/logs/log.txt`. `doctor`, `update` and `completion` write no log file.
  The header records `EntomoKit version:` (for example `0.7.1`), `Commit:`, the full
  command line, a timestamp and all parameter values, followed by the runtime
  output. `--verbose` enables debug-level logging.
- **Interruption**: `measure` and `augment` stop at the next image boundary — they
  check the shutdown flag before each image. `segment` also checks between images,
  but only its serial path stops after the current image: the parallel Otsu/GrabCut
  path has already queued its tasks, so those finish first. `synthesize` checks the
  flag only while preparing tasks, so the tasks already prepared run to completion.
  `extract-frames`, `clean` and `split-csv` install the handler but never read the
  flag: the first `Ctrl+C` only sets it and prints a notice, a second `Ctrl+C` exits,
  and no partial-result guarantee is documented. Classification and operational
  commands install no handler.
- **Device selection**: `--device auto` prefers CUDA, then MPS, then CPU.
- **Version**: `entomokit --version` (or `-v`) prints the installed version.

<a id="assistant-integration"></a>
## AI Assistant Integration (Skills)

The `entomokit-workflow` skill lets AI assistants (OpenCode, Claude Code, Codex and
similar tools) guide non-CLI users through the pipeline: parameter validation
against the runtime CLI schema before every run, step-by-step confirmation, error
recovery with suggested fixes, workflow resume after interruption, and an optional
teaching/demo mode.

```bash
mkdir -p ~/.config/opencode/skills && cp -r skills/entomokit-workflow ~/.config/opencode/skills/
mkdir -p ~/.claude/skills && cp -r skills/entomokit-workflow ~/.claude/skills/
mkdir -p ~/.codex/skills && cp -r skills/entomokit-workflow ~/.codex/skills/
```

Detailed skill rules live in [SKILL.md](skills/entomokit-workflow/SKILL.md), and
worked conversation examples are in the
[teaching playbook](skills/entomokit-workflow/references/teaching-playbook.md#user-conversation-examples).
The skill's guided approval policy is not a CLI requirement: every command below
works the same without the skill.

## Commands

Functional commands have a complete reference in `docs/commands/`; `doctor`,
`update` and `completion` are documented above.

| Command | Description | Reference |
|---|---|---|
| `extract-frames` | Extract frames from video files | [reference](docs/commands/extract-frames.md) |
| `segment` | Segment insects from images (SAM3, Otsu, GrabCut, bbox crop modes) | [reference](docs/commands/segment.md) |
| `measure` | Measure morphology metrics from segmentation masks | [reference](docs/commands/measure.md) |
| `synthesize` | Composite insects onto background images | [reference](docs/commands/synthesize.md) |
| `clean` | Clean, resize and deduplicate images | [reference](docs/commands/clean.md) |
| `augment` | Augment images with presets or a custom albumentations policy | [reference](docs/commands/augment.md) |
| `split-csv` | Split datasets into train/val/test CSVs | [reference](docs/commands/split-csv.md) |
| `doctor` | Diagnose environment and missing dependencies | [reference](#doctor-command) |
| `update` | Check for updates and optionally install the latest GitHub version | [reference](#update-command) |
| `completion` | Generate shell completion scripts | [reference](#completion-command) |

<a id="classify-commands"></a>
### Classification Commands

These require the `classify` extra.

| Command | Description | Reference |
|---|---|---|
| `classify train` | Train an AutoMM image classifier | [reference](docs/commands/classify-train.md) |
| `classify predict` | Run inference (AutoGluon or ONNX) | [reference](docs/commands/classify-predict.md) |
| `classify evaluate` | Evaluate model performance and export overall + per-class diagnostics | [reference](docs/commands/classify-evaluate.md) |
| `classify embed` | Extract embeddings + UMAP + quality metrics | [reference](docs/commands/classify-embed.md) |
| `classify cam` | Generate GradCAM heatmaps | [reference](docs/commands/classify-cam.md) |
| `classify export-onnx` | Export a model to ONNX format | [reference](docs/commands/classify-export-onnx.md) |

## Project Layout

The CLI package lives in `entomokit/`, the domain logic in `src/`; the module
boundaries and architecture constraints are described in the
[master design](docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md).

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Contact

- Email: `xtmtd.zf@gmail.com`

## Citation

If you use EntomoKit in your research, please cite:

```bibtex
@software{entomokit2026,
  author = {Zhang, Feng},
  title = {EntomoKit: A Python Toolkit for Insect Image Dataset Construction and Classification},
  year = {2026},
  url = {https://github.com/xtmtd/entomokit}
}
```
