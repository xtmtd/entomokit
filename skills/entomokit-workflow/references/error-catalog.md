# Error Catalog

## Error Style

Always respond with:

1. What failed
2. Where it failed
3. Exact next action

Avoid dumping raw stack traces unless user requests them.

## Common Errors

- `E-ENV-MISSING`: required dependency missing -> provide install command.
- `E-INPUT-EMPTY`: input folder empty -> verify path and file types.
- `E-CSV-COLUMN`: `image`/`label` missing -> show found headers and rename action.
- `E-CSV-NULL`: null cells present -> show row index and fix options.
- `E-CHECKPOINT-MISSING`: SAM3 checkpoint missing -> ask for valid checkpoint path.
- `E-OUTPUT-MISSING`: command finished but no artifacts -> verify output path and filtering settings.
- `E-CLI-FAILED`: runtime failure -> summarize key log lines and propose targeted rerun command.
