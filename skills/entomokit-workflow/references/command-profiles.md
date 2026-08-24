# Command Profiles

## Presentation Template (All Commands)

- Parameter source must be runtime schema, not model memory:
  - `python skills/entomokit-workflow/scripts/export_cli_schema.py --command "<command>"`
  - fallback: `entomokit <command> --help` (only if export script fails)
- Before execution, present parameters in a compact table with four columns:
  - parameter (all user-settable parameters; do not truncate to top 3),
  - meaning,
  - options/range,
  - selected value.
- Apply pre-run validation against schema:
  - unknown parameter -> reject and correct,
  - missing required parameter -> reject and ask user,
  - value outside enum choices -> reject and provide valid values.
- If the command has many parameters, still show the full list in one card (group by basic/advanced if needed), then ask for approval.
- End with explicit approval question before run.
- After execution, present main results first and wait for approval before proposing next command.

## extract-frames

- Accepts a directory of videos OR a single video file path.
- Typical defaults: `--interval 1000`, `--out-image-format jpg`, `--threads 8`.
- If input has nested video folders, recommend `--recursive` (output mirrors the subdirectory structure) and wait for user confirmation.
- Non-empty `--out-dir` requires explicit `--resume` (continue) or `--overwrite` (fresh start); default exits with an error.

## clean

- Required in guided mode.
- Typical defaults: `--out-short-size 512`, `--out-image-format jpg`, `--dedup-mode md5`.
- If input has nested class folders, recommend `--recursive` and wait for user confirmation.
- Non-empty `--out-dir` requires explicit `--resume` (continue) or `--overwrite` (fresh start); default exits with an error.

## segment

- Supported `--segmentation-method` values:
  - `sam3`, `sam3-bbox`, `otsu`, `otsu-bbox`, `grabcut`, `grabcut-bbox`.
- `--sam3-checkpoint` is required only for `sam3` and `sam3-bbox`.
- For faster RGB crop output without alpha mask, recommend `otsu-bbox` or `grabcut-bbox`.
- `--threads` controls concurrent image workers for Otsu/GrabCut methods (default: 8). SAM3/SAM3-bbox use a single stateful predictor and remain serial on all devices (CPU, CUDA, MPS). For CPU methods, recommend starting near half the logical CPU count to avoid OpenCV oversubscription, then benchmarking.
- If user requests unsupported methods, mark unsupported and recommend nearest supported method.
- Non-empty `--out-dir` requires `--resume` (skip already-segmented images) or `--overwrite`; default exits with an error.

## measure

- Use for morphology metrics from segmentation masks.
- Required params: `--mask-dir`, `--out-dir`.
- Optional scale: `--pixel-size-um` with unit `um/px` (micrometers per pixel).
- Explicitly remind users that `body_length`/`body_width` are mask-geometry estimates and may be biased by appendages, border clipping, or merged/fragmented masks.
- After run, always summarize:
  - `metrics.csv` (per-image metrics + warn reasons),
  - `metrics_summary.csv` (aggregate stats + warn counters),
  - `metric_definitions.csv` (metric glossary, units, formulas).

## synthesize

- Required params: `--target-dir`, `--background-dir`, `--out-dir`.
- Targets must be RGBA PNG cutouts with an alpha channel.
- Typical defaults: `--num-syntheses 10`, `--annotation-output-format coco`.
- If targets are in nested class folders, recommend `--recursive` (recurses `--target-dir` only; output mirrors the subdirectory structure) and wait for user confirmation.
- Non-empty `--out-dir` requires explicit `--resume` (continue) or `--overwrite` (fresh start); default exits with an error.

## split-csv

- Input must contain `image,label` columns.
- Confirm label extraction strategy before split:
  - folder name,
  - filename first two words,
  - mapping table.
- AutoMM-oriented default recommendation: generate `train + test.known`; let AutoGluon create train/val split internally.
- If user requests explicit val set, confirm `--val-ratio`, `--known-test-classes-ratio`, and `--unknown-test-classes-ratio` before run.
- Always state ratios in plain language before execution.
- Non-empty `--out-dir` requires explicit `--overwrite` (fresh start); default exits with an error.

## classify train

- Typical defaults: `--base-model convnextv2_femto`, `--max-epochs 50`, `--batch-size 32`.
- Device selection is mandatory confirmation:
  - list doctor-detected options (for example `mps`, `cpu`),
  - recommend fastest available backend,
  - wait for explicit user choice,
  - do not silently choose CPU.
- Suggest `--focal-loss` for imbalanced classes.
- After train completes, do not auto-run evaluate. First show key train results and ask user whether to proceed to `predict` or `evaluate`.
- Non-empty `--out-dir` requires `--resume` (continue checkpoint training) or `--overwrite` (fresh training); default exits with an error.

## classify predict/evaluate/embed/cam/export-onnx

- Predict: accept `--images-dir` or `--input-csv`.
- Evaluate: explain key metrics (Accuracy, Balanced Accuracy, F1 macro, MCC).
- Embed: extract embeddings and quality metrics; optionally visualize with UMAP.
- CAM: generate GradCAM heatmaps for model interpretability.
- Export ONNX: generate `model.onnx` and `label_classes.json`.
- All five subcommands accept `--overwrite` to delete `--out-dir` contents and start fresh; non-empty `--out-dir` without `--overwrite` exits with an error.

## Retry and Rerun

- On failure, propose rerun with adjusted parameters.
- Run roots use `runs/runNNN/` naming (`run001`, `run002`, ...), not numeric-only names.
- Default rerun output must be a new sibling directory with `-runNNN` suffix (`train-run001`, `train-run002`, etc.).
- If user prefers cleanup, ask explicit approval before deleting failed output.
- `clean` retry outputs must remain under `runs/runNNN/...`, never next to raw image folders.
- CLI-level `--resume`/`--overwrite` flags apply within a single `--out-dir`. The skill-layer default (new sibling directory) is recommended for clean separation; `--resume` is appropriate when a run was interrupted and the partial output is valid.
