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

## clean

- Required in guided mode.
- Typical defaults: `--out-short-size 512`, `--out-image-format jpg`, `--dedup-mode md5`.
- If input has nested class folders, recommend `--recursive` and wait for user confirmation.

## segment

- Supported `--segmentation-method` values:
  - `sam3`, `sam3-bbox`, `otsu`, `otsu-bbox`, `grabcut`, `grabcut-bbox`.
- `--sam3-checkpoint` is required only for `sam3` and `sam3-bbox`.
- For faster RGB crop output without alpha mask, recommend `otsu-bbox` or `grabcut-bbox`.
- If user requests unsupported methods, mark unsupported and recommend nearest supported method.

## split-csv

- Input must contain `image,label` columns.
- Confirm label extraction strategy before split:
  - folder name,
  - filename first two words,
  - mapping table.
- AutoMM-oriented default recommendation: generate `train + test.known`; let AutoGluon create train/val split internally.
- If user requests explicit val set, confirm `--val-ratio`, `--known-test-classes-ratio`, and `--unknown-test-classes-ratio` before run.
- Always state ratios in plain language before execution.

## classify train

- Typical defaults: `--base-model convnextv2_femto`, `--max-epochs 50`, `--batch-size 32`.
- Device selection is mandatory confirmation:
  - list doctor-detected options (for example `mps`, `cpu`),
  - recommend fastest available backend,
  - wait for explicit user choice,
  - do not silently choose CPU.
- Suggest `--focal-loss` for imbalanced classes.
- After train completes, do not auto-run evaluate. First show key train results and ask user whether to proceed to `predict` or `evaluate`.

## classify predict/evaluate/export-onnx

- Predict: accept `--images-dir` or `--input-csv`.
- Evaluate: explain key metrics (Accuracy, Balanced Accuracy, F1 macro, MCC).
- Export ONNX: generate `model.onnx` and `label_classes.json`.

## Retry and Rerun

- On failure, propose rerun with adjusted parameters.
- Run roots use `runs/runNNN/` naming (`run001`, `run002`, ...), not numeric-only names.
- Default rerun output must be a new sibling directory with `-runNNN` suffix (`train-run001`, `train-run002`, etc.).
- If user prefers cleanup, ask explicit approval before deleting failed output.
- `clean` retry outputs must remain under `runs/runNNN/...`, never next to raw image folders.
