# Workflow

## Phase 0 - Environment

1. Run `entomokit doctor`.
2. If core deps missing, offer conda/venv/global install options.
3. If optional deps missing, explain impacted commands.
4. If multiple compute backends are available, record options for later `classify train` device confirmation.

## Global Execution Gate (All Phases)

Template source: `references/dialog-templates.md` (mandatory).

Policy gate:

- Default execution policy is `entomokit-only`.
- Validate every shell command using `entomokit.execution_policy.validate_execution_command` before execution.
- If blocked, do not execute. Ask user whether to allow fallback script and collect explicit `fallback_reason`.
- Recommended unified entrypoint: `entomokit.workflow_gate.run_guarded_step(...)`.
- CLI wrapper for agent/tooling environments: `python skills/entomokit-workflow/scripts/run_guarded_step.py ...`.

Before every command:

1. Explain command intent in one short sentence.
2. Export runtime command schema first:
   - `python skills/entomokit-workflow/scripts/export_cli_schema.py --command "<command>"`
   - if export fails, fallback to `entomokit <command> --help` and mark fallback source.
3. Show a parameter card with meanings:
    - every user-settable parameter for this command,
    - what it controls,
    - allowed options/range,
    - chosen value for this run.
4. Validate parameters against exported schema:
   - reject unknown flags,
   - reject missing required flags,
   - reject values outside choices.
5. Show exact output path under `runs/runNNN/...`.
6. Wait for explicit user approval.

After every command:

1. Summarize key result and generated files.
2. Ask user to approve one action: continue, rerun with adjusted parameters, or stop.
3. Do not start the next command until user approves.

Recommended pre-run chain (must stay in order):

1. `validate_execution_command(...)`
2. `render_parameter_card(...)`
3. `validate_parameters(...)`
4. Execute command

## Output Directory Policy

1. Create one run root in working directory before first write, named `runs/runNNN/` (for example `runs/run001/`).
2. Store all step outputs under this root; avoid writing mixed artifacts directly under `data/`.
3. For retries after failure, create a new sibling folder using `-runNNN` suffix (`clean-run001`, `clean-run002`, `train-run001`, `train-run002`) unless user explicitly approves deleting failed output first.
4. `clean` output location must always stay inside the active run root, even on failure or rerun.

## Phase 1 - Dataset Preparation

1. Optional `extract-frames` when source is video.
2. Mandatory `clean` in guided mode.
3. Optional `segment`, `synthesize`, `augment` based on user goals.
4. Before `clean`, if nested class folders exist, confirm whether to use `--recursive`.

## Phase 2 - Split Preparation

1. Teach expected CSV format (`image,label`).
2. Validate headers, nulls, and path reachability.
3. Confirm label source rule (folder name, filename first two words, or mapping rule).
4. Confirm split policy for AutoMM:
   - default recommendation: `train + test.known` only,
   - optional: explicit `train + val + test`.
5. State and confirm split ratios before `split-csv`.
6. Run `split-csv` and show split summary.

## Phase 3 - Classification

1. Confirm `classify train` params, especially `--device`, using doctor-discovered backends.
2. Never auto-fallback to CPU without explicit user approval.
3. After train completes, present headline metrics and artifact paths; wait for approval before any next sub-step.
4. Optionally run `predict`, `evaluate`, `embed`, `cam`, `export-onnx` only after explicit approval.
5. Ask continue/rerun/stop between sub-steps.

## Demo Disclosure

Only use `data/` when user opts in. Resolve demo root with `scripts/resolve_data_dir.py`. After demo, restate user paths and continue with user data.

## Session Memory

- No JSON progress file is written.
- Resume across conversations relies on user-provided recap.
