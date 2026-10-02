# Teaching Playbook (Opt-In)

Use this only when user asks for demonstration, onboarding, or troubleshooting by example.

Before any demo command, resolve `DATA_ROOT` dynamically:

```bash
DATA_ROOT="$(python skills/entomokit-workflow/scripts/resolve_data_dir.py)"
```

## Available Demo Assets

- `$DATA_ROOT/video.mp4` -> extract-frames demo
- `$DATA_ROOT/insects/` -> small clean/segment/predict demo
- `out/demo_segment/SegmentationClass/` -> binary masks for the measure demo, written by the segment demo with `--annotation-format voc`
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

<a id="user-conversation-examples"></a>
## User Conversation Examples

Copy-pasteable openers for a guided session. The skill confirms parameters before
every run; these are prompts, not commands.

English:

- "I need to use the entomokit-workflow skill to clean images in data/Epidorcus and train a classification model."
- "Use the entomokit-workflow skill to process data/my_insects: clean images, split the dataset, and train a convnextv2_femto classifier."
- "I want to learn entomokit commands through the entomokit-workflow skill. Can you give me a teaching demo?"

Chinese:

- "我想用 entomokit-workflow skill 清洗 data/Epidorcus 里的图像，并训练一个分类模型。"
- "用 entomokit-workflow skill 处理 data/my_insects：清洗图像、划分数据集，并训练一个 convnextv2_femto 分类器。"
- "我想通过 entomokit-workflow skill 学习 entomokit 命令，可以给我做一个教学演示吗？"

## Demo Command Templates

```bash
# video -> frames
entomokit extract-frames --input-dir "$DATA_ROOT/video.mp4" --out-dir out/demo_frames/

# clean
entomokit clean --input-dir "$DATA_ROOT/insects/" --out-dir out/demo_clean/

# segment (writes the VOC binary masks the measure demo needs)
entomokit segment --input-dir "$DATA_ROOT/insects/" --out-dir out/demo_segment/ \
    --segmentation-method otsu --annotation-format voc

# measure (binary masks only; never the RGB/RGBA segment/images/ crops)
entomokit measure --mask-dir out/demo_segment/SegmentationClass/ --out-dir out/demo_measure/ --pixel-size-um 1.0

# split-csv
entomokit split-csv --raw-image-csv "$DATA_ROOT/Epidorcus/figs.csv" --images-dir "$DATA_ROOT/Epidorcus/images/" --out-dir out/demo_split/
```
