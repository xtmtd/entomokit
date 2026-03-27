---
name: entomokit-workflow
description: Conversational workflow orchestrator for EntomoKit from dataset preparation to model training/evaluation/export. Use when users ask to run, plan, troubleshoot, or learn entomokit commands (doctor, clean, segment, synthesize, augment, split-csv, classify), especially for step-by-step guidance, CSV validation, teaching demos, and safe parameter confirmation.
---

# EntomoKit Workflow

Run the workflow in phases. Keep user data as first priority.

## Core Rules

1. Run `entomokit doctor` before any substantive step in a new conversation.
2. Every command requires explicit approval before execution:
   - show one-line purpose,
   - show a full parameter card (all user-settable parameters for that command, not only key/top 3),
   - include meaning + options/range + selected value for each parameter,
   - parameter names/options must come from runtime CLI schema (not memory),
   - wait for user confirmation or parameter edits.
3. After each command result, summarize outputs first, then wait for user approval before any next-step command.
4. When presenting next-step suggestions, do not auto-run anything. Require explicit user selection and approval.
5. In guided mode, treat `clean` as the gate before segment/synthesize/augment/split/train.
6. Validate CSV (`image,label`) and label-source rules before `split-csv`.
7. Never write outputs directly into repository `data/` or mixed root paths.
8. Create a dedicated run root under the working directory with name `runNNN` (for example `runs/run001/`) and place all command outputs inside it.
9. On failed/retry runs, do not overwrite prior artifacts by default. Either:
   - create a new sibling output directory (for example `train-run001/`, `train-run002/`), or
   - delete failed output only after explicit user approval.
10. `clean` failures must still write to `runs/runNNN/...`; never place cleaned output next to raw input folders.
11. Workflow execution policy is `entomokit-only` by default:
   - prefer `entomokit <command>` for all workflow actions,
   - block custom Python scripts unless user explicitly approves fallback and reason is recorded.
   - when fallback is approved, pass explicit `fallback_reason` to execution gate.
12. Segment backend guardrail:
   - supported `--segmentation-method` values are `sam3`, `sam3-bbox`, `otsu`, `otsu-bbox`, `grabcut`, `grabcut-bbox`.
   - if user asks for unsupported values, mark unsupported and recommend nearest supported method.

## Parameter Source-of-Truth Protocol

- Before showing any parameter card, load schema from runtime CLI:
  - `python skills/entomokit-workflow/scripts/export_cli_schema.py --command "<command>"`
- Preferred execution bridge for conversational runs:
  - `python skills/entomokit-workflow/scripts/run_guarded_step.py ...`
- Never invent parameter names, aliases, or enum values from memory.
- If user asks for a parameter not present in schema, explicitly mark as unsupported and propose the nearest valid parameter.
- In parameter cards, add a short footer: `Schema source: runtime CLI export`.
- If schema export fails, fallback to `entomokit <command> --help`, and state fallback source explicitly.

## Parameter Validation Gate

- Before execution, validate approved params against schema:
  - unknown parameter -> block run,
  - missing required parameter -> block run,
  - enum value out of allowed set -> block run,
  - boolean flag value not true/false -> block run.
- When blocked, show a fix card with exact invalid field and valid alternatives.
- Only run command after validation status is `passed`.

## Data Source Decision

- If user provided task data, always use user data.
- Offer `data/` examples only when user asks for demo/teaching/troubleshooting.
- Resolve demo `data/` path dynamically from the installed entomokit location, not from hardcoded local paths.
- Before demo: clearly state demo run does not replace user workflow.
- After demo: explicitly switch back to user paths.
- Never write outputs into `data/`; always write to user output directories.

## Demo Visibility Prompts

At key checkpoints, remind user demo data is available without forcing a switch:

- after doctor passes,
- before CSV teaching,
- when blocked by data-format/path issues.

Short template:

"If helpful, I can run a quick demo with repository `data/` so you can preview this step, then we continue with your data."

## Entry Modes

- Guided mode: Phase 0 -> 1 -> 2 -> 3.
- Direct mode: jump to requested step after Phase 0.
- Direct mode clean policy: ask whether images are already cleaned; recommend clean when uncertain.

## Clean Precheck

- Before `clean`, detect whether input contains nested class folders.
- If nested folders are present, explain that first run may miss files without `--recursive`.
- Ask user to confirm `--recursive` strategy before execution and before any merge/clean follow-up.

## Label Strategy Confirmation

- Before CSV generation or `split-csv`, require explicit label-source confirmation.
- Offer at least these options:
  1. folder name as label,
  2. first two words of image filename as label,
  3. user-provided mapping table/rule.
- Do not generate CSV until user confirms one strategy.

## AutoMM Split Policy

- For AutoGluon AutoMM training, default suggestion is `train + test.known` only, because AutoGluon internally splits train/val.
- Ask user whether to keep this default or generate explicit `train + val + test` files.
- Always confirm and state split ratios before running `split-csv`.

## Device Selection Guardrail

- When `doctor` reports multiple compute backends (for example CPU + MPS), present options and recommendation.
- Never silently choose slow CPU when faster backend is available.
- Wait for explicit user confirmation of `--device` before `classify train`.
- After `classify train`, stop and present training results for discussion and approval before suggesting or running `predict`/`evaluate`.

## Phases

- Phase 0: `doctor`
- Phase 1: `extract-frames` (optional) -> `clean` (required gate) -> `segment`/`synthesize`/`augment` (optional)
- Phase 2: CSV teaching + CSV validation + `split-csv`
- Phase 3: `classify train` -> `predict` -> `evaluate` -> `embed`/`cam` -> `export-onnx`

## Session State

- Do not create or update `entomokit_progress.json`.
- Keep state in current conversation only; ask user for context when resuming in a new conversation.

## References

- See `references/workflow.md` for per-phase execution script.
- See `references/command-profiles.md` for parameter defaults.
- See `references/csv-validation.md` for strict CSV checks and fixes.
- See `references/error-catalog.md` for error mapping and repair actions.
- See `references/teaching-playbook.md` for opt-in demo flows using `data/`.
- See `references/path-resolution.md` and `scripts/resolve_data_dir.py` for dynamic example-data path discovery.
- See `scripts/export_cli_schema.py` for machine-readable command parameter schema.
- Runtime helpers in package:
  - `entomokit.execution_policy.validate_execution_command` for command gatekeeping.
- See `references/release-checklist.md` for packaging readiness checks.
- See `references/dialog-templates.md` for mandatory pre-run and post-run cards.
