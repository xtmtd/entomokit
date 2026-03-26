# Path Resolution

Do not hardcode `data/` absolute paths.

## Rule

- Resolve example-data root dynamically from the installed entomokit location.
- Allow explicit override by `ENTOMOKIT_DATA_DIR`.

## Resolver Script

Use:

```bash
python skills/entomokit-workflow/scripts/resolve_data_dir.py
```

If successful, the script prints a valid data root containing known markers.

## Shell Pattern

```bash
DATA_ROOT="$(python skills/entomokit-workflow/scripts/resolve_data_dir.py)"
```

Then reference examples via:

- `$DATA_ROOT/video.mp4`
- `$DATA_ROOT/insects/`
- `$DATA_ROOT/Epidorcus/figs.csv`
- `$DATA_ROOT/Epidorcus/images/`

## Failure Handling

If resolver fails:

1. Ask user to set `ENTOMOKIT_DATA_DIR`.
2. Confirm the directory includes at least one known marker file/folder.
3. Retry resolver before running demo commands.
