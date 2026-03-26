# Teaching Playbook (Opt-In)

Use this only when user asks for demonstration, onboarding, or troubleshooting by example.

Before any demo command, resolve `DATA_ROOT` dynamically:

```bash
DATA_ROOT="$(python skills/entomokit-workflow/scripts/resolve_data_dir.py)"
```

## Available Demo Assets

- `$DATA_ROOT/video.mp4` -> extract-frames demo
- `$DATA_ROOT/insects/` -> small clean/segment/predict demo
- `$DATA_ROOT/Epidorcus/figs.csv` + `$DATA_ROOT/Epidorcus/images/` -> split + train demo
- `$DATA_ROOT/segment/annotations.coco.json` -> segmentation output structure demo

## Demo Script Pattern

1. Announce demo isolation: this preview does not replace user data workflow.
2. Run smallest useful demo command.
3. Show expected output artifacts.
4. Ask whether to repeat same step on user data.
5. Restate user paths and continue with user data.

## Short Prompt for Offer

"If helpful, I can quickly demonstrate this step using repository `data/`, then apply the same pattern to your dataset."

## Demo Command Templates

```bash
# video -> frames
entomokit extract-frames --input-dir "$DATA_ROOT/video.mp4" --out-dir out/demo_frames/

# clean
entomokit clean --input-dir "$DATA_ROOT/insects/" --out-dir out/demo_clean/

# split-csv
entomokit split-csv --raw-image-csv "$DATA_ROOT/Epidorcus/figs.csv" --images-dir "$DATA_ROOT/Epidorcus/images/" --out-dir out/demo_split/
```
