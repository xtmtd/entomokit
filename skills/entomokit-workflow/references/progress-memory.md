# Progress Memory

## Required File

- Primary memory file: `entomokit_progress.json`
- Location: working directory root (same level as `runs/`)
- Purpose: cross-session resume, audit of params/results, run linkage
- Must be created at session start if missing (before `doctor` and before any parameter discussion)
- On creation, explicitly show user the file path and initialized run id

## Optional YAML Mirror

- Only if user explicitly requests YAML: write `entomokit_progress.yaml`
- Keep fields equivalent to JSON file

## Minimum JSON Schema

```json
{
  "version": "1",
  "working_dir": "/abs/path/project",
  "active_run": "run001",
  "updated_at": "2026-03-26T10:20:30Z",
  "current_step": "clean",
  "steps": {
    "doctor": {
      "status": "passed",
      "approved": true,
      "params": {},
      "outputs": [],
      "timestamp": "2026-03-26T10:00:00Z"
    },
    "clean": {
      "status": "failed",
      "approved": true,
      "params": {
        "recursive": true,
        "out_short_size": 512,
        "dedup_mode": "md5"
      },
      "outputs": [
        "runs/run001/clean-run001/log.txt"
      ],
      "timestamp": "2026-03-26T10:20:00Z"
    }
  }
}
```

## Status Values

- `pending`
- `in_progress`
- `passed`
- `failed`
- `skipped`

## Update Timing

1. Initialize file at session start if missing, then announce creation path.
2. Before command execution, write `in_progress` with approved params.
3. After command completion, write final status + outputs.
4. On retry, keep previous attempt record and add new output path under the new `*-runNNN` folder.
5. Never overwrite history silently.
