# Release Checklist

Use this checklist before packaging the skill.

## Content Checks

- `SKILL.md` has only `name` and `description` in YAML frontmatter.
- Description starts with explicit trigger language and does not encode full workflow.
- References are one level deep and linked from `SKILL.md`.

## Behavior Checks

- User-data-first rule is explicit.
- Demo `data/` usage is opt-in and disclosed.
- Demo path resolution uses `scripts/resolve_data_dir.py` or `ENTOMOKIT_DATA_DIR`.
- Every command requires pre-run parameter card (parameter/meaning/options/selected value) and explicit approval.
- Every step requires post-run result summary and explicit approval before next step.
- Output policy enforces `runs/runNNN/` naming and forbids writing clean outputs next to raw folders.
- `classify train` explicitly blocks auto-transition to evaluate/predict without user approval.
- Memory policy enforces `entomokit_progress.json` in working directory root (YAML only on explicit user request).
- Fixed dialogue templates are referenced from `references/dialog-templates.md` and used in all phases.

## Quick Validation

```bash
python skills/entomokit-workflow/scripts/resolve_data_dir.py
```

Expected: prints a valid data directory.

## Packaging

If using superpowers packaging tooling:

```bash
scripts/package_skill.py skills/entomokit-workflow
```

If tooling is unavailable in this repo, keep the folder structure stable for manual distribution.
