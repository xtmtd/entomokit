# entomokit 重构 Phase 1 — CLI 框架搭建与现有命令迁移

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 entomokit 从五个独立脚本重构为统一的 `entomokit <command>` CLI 入口，迁移所有现有命令，不改变业务逻辑。

**Architecture:** 新建 `entomokit/` 包作为 CLI 入口层，每个命令模块暴露 `register(subparsers)` + `run(args)` 接口；业务逻辑保留在 `src/` 不动；`setup.py` 改为单一 entry point。

**Tech Stack:** Python 3.8+, argparse, supervision (注释格式对齐 detcli)

**Spec:** `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` 第 2-5.1 节、第 8-9 节

---

## 文件结构

### 新建文件

```
entomokit/__init__.py
entomokit/main.py
entomokit/segment.py
entomokit/extract_frames.py
entomokit/clean.py
entomokit/split_csv.py
entomokit/synthesize.py
entomokit/classify/__init__.py      # classify 组 dispatcher（stub，Phase 4 填充）
```

### 修改文件

```
setup.py                            # 替换 entry_points，移除旧5个入口，加单一入口
```

### 参考文件（只读，不修改）

```
scripts/segment.py
scripts/extract_frames.py
scripts/clean_figs.py
scripts/split_dataset.py
scripts/synthesize.py
src/segmentation/processor.py
src/framing/extractor.py
src/cleaning/processor.py
src/splitting/splitter.py
src/synthesis/processor.py
src/common/cli.py
```

---

## Task 1: 初始化 `entomokit/` 包与顶层 dispatcher

**Files:**
- Create: `entomokit/__init__.py`
- Create: `entomokit/main.py`
- Create: `entomokit/classify/__init__.py`

- [ ] **Step 1: 创建包目录和 `__init__.py`**

```bash
mkdir -p entomokit/classify
touch entomokit/__init__.py
touch entomokit/classify/__init__.py
```

- [ ] **Step 2: 写 `entomokit/main.py`**

```python
"""entomokit — unified CLI entry point."""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="entomokit",
        description="A toolkit for building insect image datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # Lazy imports keep startup fast and avoid heavy optional deps at import time
    from entomokit import segment as _segment
    from entomokit import extract_frames as _extract_frames
    from entomokit import clean as _clean
    from entomokit import split_csv as _split_csv
    from entomokit import synthesize as _synthesize
    from entomokit.classify import register as _register_classify

    _segment.register(subparsers)
    _extract_frames.register(subparsers)
    _clean.register(subparsers)
    _split_csv.register(subparsers)
    _synthesize.register(subparsers)
    _register_classify(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 写 `entomokit/classify/__init__.py` (stub)**

```python
"""classify command group — implemented in Phase 4."""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the classify command group (stub)."""
    p = subparsers.add_parser(
        "classify",
        help="Image classification commands (AutoGluon). Coming in Phase 4.",
    )
    sub = p.add_subparsers(dest="subcommand", metavar="<subcommand>")
    sub.required = True
    # Subcommands registered in Phase 4
```

- [ ] **Step 4: 验证包可以导入（无报错）**

```bash
python -c "from entomokit.main import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add entomokit/
git commit -m "feat: init entomokit CLI package with top-level dispatcher and classify stub"
```

---

## Task 2: 迁移 `segment` 命令

**Files:**
- Create: `entomokit/segment.py`
- Reference: `scripts/segment.py` (copy arg definitions, adapt to register/run pattern)

- [ ] **Step 1: 写 `entomokit/segment.py`**

直接从 `scripts/segment.py` 提取所有 `add_argument` 调用，适配为 `register`/`run` 函数对。

```python
"""entomokit segment — insect image segmentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "segment",
        help="Segment insects from images using SAM3, Otsu, or GrabCut.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- I/O ---
    p.add_argument("--input-dir", required=True, help="Input images directory.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    # --- Segmentation method ---
    p.add_argument(
        "--segmentation-method",
        default="sam3",
        choices=["sam3", "sam3-bbox", "otsu", "grabcut"],
        help="Segmentation method.",
    )
    p.add_argument("--sam3-checkpoint", default=None, help="Path to SAM3 checkpoint.")
    p.add_argument("--hint", default="insect", help="Text hint for SAM3 grounding.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--threads", type=int, default=8)
    # --- Output format ---
    p.add_argument(
        "--out-image-format",
        default="png",
        choices=["png", "jpg", "jpeg"],
    )
    p.add_argument(
        "--repair-strategy",
        default=None,
        choices=["sam3-fill", "white-fill", "none"],
    )
    # --- Annotation output ---
    p.add_argument(
        "--annotation-format",
        default=None,
        choices=["coco", "yolo", "voc"],
        help="Annotation output format. None = no annotations.",
    )
    p.add_argument(
        "--coco-bbox-format",
        default="xywh",
        choices=["xywh", "xyxy"],
        help="COCO bbox coordinate convention (only used when --annotation-format=coco).",
    )
    # --- Misc ---
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import sys
    from pathlib import Path

    # Ensure project root is importable (handles editable installs and direct runs)
    _ensure_src_on_path()

    from src.common.cli import setup_shutdown_handler, get_shutdown_flag, setup_logging, save_log
    from src.segmentation.processor import SegmentationProcessor

    setup_shutdown_handler()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: --input-dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logging(out_dir, verbose=args.verbose)
    save_log(out_dir, args)

    processor = SegmentationProcessor(
        input_dir=str(input_dir),
        output_dir=str(out_dir),
        segmentation_method=args.segmentation_method,
        sam3_checkpoint=args.sam3_checkpoint,
        hint=args.hint,
        device=args.device,
        threads=args.threads,
        out_image_format=args.out_image_format,
        repair_strategy=args.repair_strategy,
        annotation_format=args.annotation_format,
        coco_bbox_format=args.coco_bbox_format,
        shutdown_flag=get_shutdown_flag(),
    )
    processor.process()


def _ensure_src_on_path() -> None:
    """Add project root to sys.path if needed (for non-editable installs)."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
```

- [ ] **Step 2: 冒烟测试（不实际运行分割，只验证 CLI 解析正确）**

```bash
python -m entomokit.main segment --help
```

Expected: segment 的 help 输出，包含 `--input-dir`, `--annotation-format` 等参数。

- [ ] **Step 3: Commit**

```bash
git add entomokit/segment.py
git commit -m "feat: add entomokit segment command (CLI wrapper)"
```

---

## Task 3: 迁移 `extract-frames` 命令

**Files:**
- Create: `entomokit/extract_frames.py`
- Reference: `scripts/extract_frames.py`

- [ ] **Step 1: 写 `entomokit/extract_frames.py`**

关键变化：`--input-dir` 同时接受目录路径和单个视频文件路径。

```python
"""entomokit extract-frames — extract frames from video files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "extract-frames",
        help="Extract frames from video files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", "-i", required=True,
        help="Input directory containing video files, OR a single video file path.",
    )
    p.add_argument("--out-dir", "-o", required=True, help="Output directory for frames.")
    p.add_argument(
        "--out-image-format",
        choices=["jpg", "png", "tif"],
        default="jpg",
    )
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Max frames to extract per video.")
    p.add_argument("--start-time", type=float, default=0.0,
                   help="Start time in seconds.")
    p.add_argument("--end-time", type=float, default=None,
                   help="End time in seconds.")
    p.add_argument("--interval", type=int, default=1000,
                   help="Extraction interval in milliseconds.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip frames that already exist (resume).")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import logging
    import sys
    import tempfile
    from pathlib import Path
    from src.common.cli import setup_shutdown_handler, get_shutdown_flag, save_log
    from src.framing.extractor import VideoFrameExtractor

    setup_shutdown_handler()

    input_path = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    # Accept single video file OR directory
    if input_path.is_file():
        # Wrap single file in a temp-like virtual dir by using its parent
        # but filter to only that file inside VideoFrameExtractor
        actual_input_dir = input_path.parent
        single_file = input_path
    elif input_path.is_dir():
        actual_input_dir = input_path
        single_file = None
    else:
        print(f"Error: --input-dir does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    extractor = VideoFrameExtractor(
        input_dir=str(actual_input_dir),
        output_dir=str(out_dir),
        interval_ms=args.interval,
        image_format=args.out_image_format,
        max_frames=args.max_frames,
        threads=args.threads,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    # If single file mode, filter video list
    if single_file is not None:
        extractor._single_file_filter = single_file.name

    stats = extractor.extract_all(show_progress=not args.quiet)

    if not args.quiet:
        print(f"Total videos processed: {stats['total_videos']}")
        print(f"Total frames extracted: {stats['total_frames']}")
        print(f"Errors: {stats['errors']}")
```

- [ ] **Step 2: 更新 `VideoFrameExtractor.get_video_files` 以支持单文件过滤**

在 `src/framing/extractor.py` 的 `get_video_files` 方法中检查 `_single_file_filter` 属性：

```python
def get_video_files(self) -> List[Path]:
    """Return video files to process."""
    single_filter = getattr(self, "_single_file_filter", None)
    files = []
    for ext in self.SUPPORTED_VIDEO_FORMATS:
        files.extend(self.input_dir.glob(f"*.{ext}"))
        files.extend(self.input_dir.glob(f"*.{ext.upper()}"))
    if single_filter:
        files = [f for f in files if f.name == single_filter]
    return sorted(set(files))
```

- [ ] **Step 3: 冒烟测试**

```bash
python -m entomokit.main extract-frames --help
```

Expected: help 输出含 `--input-dir (directory OR single video file)` 说明。

- [ ] **Step 4: 用真实视频文件测试单文件模式**

```bash
python -m entomokit.main extract-frames \
    --input-dir data/video.mp4 \
    --out-dir /tmp/frames_test \
    --max-frames 3 \
    --quiet
ls /tmp/frames_test/
```

Expected: 看到 3 个帧图像文件（如 `video_000000.jpg` 等）。

- [ ] **Step 5: Commit**

```bash
git add entomokit/extract_frames.py src/framing/extractor.py
git commit -m "feat: add entomokit extract-frames command; support single video file in --input-dir"
```

---

## Task 4: 迁移 `clean` 命令（含 `--recursive`）

**Files:**
- Create: `entomokit/clean.py`
- Modify: `src/cleaning/processor.py` — `process_directory` 新增 recursive 支持
- Reference: `scripts/clean_figs.py`

- [ ] **Step 1: 写 `entomokit/clean.py`**

```python
"""entomokit clean — image cleaning and deduplication."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "clean",
        help="Clean and deduplicate images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, help="Input images directory.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--out-short-size", type=int, default=512,
                   help="Resize shorter edge to this size. Use -1 to keep original.")
    p.add_argument("--out-image-format", default="jpg",
                   choices=["jpg", "png", "tif"],)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--keep-exif", action="store_true")
    p.add_argument("--dedup-mode", default="md5",
                   choices=["none", "md5", "phash"])
    p.add_argument("--phash-threshold", type=int, default=5)
    p.add_argument("--recursive", action="store_true",
                   help="Recursively scan subdirectories in --input-dir.")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.common.cli import setup_shutdown_handler, save_log
    from src.cleaning.processor import ImageCleaner

    setup_shutdown_handler()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: --input-dir does not exist or is not a directory: {input_dir}",
              file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_subdir = out_dir / "cleaned_images"
    images_subdir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    cleaner = ImageCleaner(
        input_dir=str(input_dir),
        output_dir=str(images_subdir),
        out_short_size=args.out_short_size,
        out_image_format=args.out_image_format,
        dedup_mode=args.dedup_mode,
        phash_threshold=args.phash_threshold,
        threads=args.threads,
        keep_exif=args.keep_exif,
    )

    log_path = str(out_dir / "log.txt")
    results = cleaner.process_directory(
        log_path=log_path,
        recursive=args.recursive,
    )

    print(f"Done. Processed {results['processed']} images, {results['errors']} errors.")
    print(f"Cleaned images saved to: {images_subdir}")
```

- [ ] **Step 2: 更新 `src/cleaning/processor.py` — 添加 recursive 支持**

在 `ImageCleaner.process_directory` 中增加 `recursive: bool = False` 参数，将文件收集从 `iterdir()` 改为支持 `rglob`：

```python
def process_directory(
    self,
    input_dir: Optional[str] = None,
    log_path: Optional[str] = "log.txt",
    recursive: bool = False,
) -> dict:
    input_dir = Path(input_dir or self.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if recursive:
        files = [p for p in input_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        files = [p for p in input_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    # ... rest unchanged
```

- [ ] **Step 3: 写单元测试**

在 `tests/test_clean_recursive.py`：

```python
"""Tests for clean --recursive flag."""
import pytest
from pathlib import Path
from src.cleaning.processor import ImageCleaner


def test_recursive_finds_images_in_subdirs(tmp_path):
    """Recursive mode collects images from nested directories."""
    sub = tmp_path / "input" / "subdir"
    sub.mkdir(parents=True)
    # Create minimal 1x1 JPEG in subdir
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(sub / "test.jpg")

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cleaner = ImageCleaner(
        input_dir=str(tmp_path / "input"),
        output_dir=str(out_dir),
        dedup_mode="none",
    )
    results = cleaner.process_directory(log_path=str(tmp_path / "log.txt"), recursive=True)
    assert results["processed"] == 1


def test_non_recursive_misses_subdir_images(tmp_path):
    """Non-recursive mode should NOT pick up images in subdirs."""
    sub = tmp_path / "input" / "subdir"
    sub.mkdir(parents=True)
    from PIL import Image
    img = Image.new("RGB", (10, 10))
    img.save(sub / "test.jpg")

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cleaner = ImageCleaner(
        input_dir=str(tmp_path / "input"),
        output_dir=str(out_dir),
        dedup_mode="none",
    )
    results = cleaner.process_directory(log_path=str(tmp_path / "log.txt"), recursive=False)
    assert results["processed"] == 0
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_clean_recursive.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: 冒烟测试**

```bash
python -m entomokit.main clean --help
```

Expected: help 含 `--recursive`。

- [ ] **Step 6: Commit**

```bash
git add entomokit/clean.py src/cleaning/processor.py tests/test_clean_recursive.py
git commit -m "feat: add entomokit clean command with --recursive support"
```

---

## Task 5: 迁移 `split-csv` 命令（含 val + copy-images）

**Files:**
- Create: `entomokit/split_csv.py`
- Modify: `src/splitting/splitter.py` — 新增 val 分割 + copy_images 功能
- Reference: `scripts/split_dataset.py`

- [ ] **Step 1: 更新 `src/splitting/splitter.py`**

在 `DatasetSplitter` 中：

1. `split_ratio_mode` 增加 `val_ratio: float = 0.0` 参数，在 train/test 分割后从 train 中再切出 val
2. `split_count_mode` 增加 `val_count: int = 0` 参数，逻辑同上
3. 新增 `copy_images(images_dir: Path) -> None` 方法，按 split CSV 将图像复制到对应子目录
4. `split()` 方法增加 `val_ratio`, `val_count`, `images_dir`, `copy_images` 参数

关键逻辑（val 切割示意）：
```python
# 在 split_ratio_mode 末尾，train_data 已确定后：
if val_ratio > 0 and len(train_data) > 0:
    val_idx = train_data.groupby('label', group_keys=False).sample(
        frac=val_ratio, random_state=self.seed
    ).index
    val_data = train_data.loc[val_idx].reset_index(drop=True)
    train_data = train_data.drop(val_idx).reset_index(drop=True)
    val_data.to_csv(self.out_dir / 'val.csv', index=False)
    val_data.label.value_counts().to_csv(
        self.class_count_dir / 'class.val.count', index=False
    )
```

copy_images 方法：
```python
def copy_images(self, images_dir: Path, splits: dict) -> None:
    """Copy images into out_dir/images/{split}/ subdirs.
    
    Args:
        images_dir: Source image directory.
        splits: Dict mapping split name -> DataFrame with 'image' column.
                e.g. {'train': train_df, 'val': val_df, ...}
    """
    import shutil
    images_root = self.out_dir / "images"
    for split_name, df in splits.items():
        if df is None or len(df) == 0:
            continue
        split_dir = images_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for img_path in df["image"]:
            src = images_dir / img_path
            dst = split_dir / Path(img_path).name
            if src.exists():
                shutil.copy2(src, dst)
```

- [ ] **Step 2: 写 `entomokit/split_csv.py`**

```python
"""entomokit split-csv — split a CSV dataset into train/val/test splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "split-csv",
        help="Split a CSV dataset (image, label) into train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-image-csv", required=True,
                   help="Input CSV with 'image' and 'label' columns.")
    p.add_argument("--mode", choices=["ratio", "count"], default="ratio")
    # --- known/unknown test ---
    p.add_argument("--unknown-test-classes-ratio", type=float, default=0.0)
    p.add_argument("--known-test-classes-ratio", type=float, default=0.1)
    p.add_argument("--unknown-test-classes-count", type=int, default=0)
    p.add_argument("--known-test-classes-count", type=int, default=0)
    # --- val split ---
    p.add_argument("--val-ratio", type=float, default=0.0,
                   help="Val split ratio (from train). 0 = no val split.")
    p.add_argument("--val-count", type=int, default=0,
                   help="Val split count (from train). 0 = no val split.")
    # --- train controls ---
    p.add_argument("--min-count-per-class", type=int, default=0)
    p.add_argument("--max-count-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="datasets")
    # --- copy images ---
    p.add_argument("--images-dir", default=None,
                   help="Source image directory. Required when --copy-images is set.")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy images into out_dir/images/{split}/ subdirectories.")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.common.cli import setup_shutdown_handler, save_log
    from src.splitting.splitter import DatasetSplitter

    setup_shutdown_handler()

    if args.copy_images and not args.images_dir:
        print("Error: --images-dir is required when --copy-images is set.", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.raw_image_csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    splitter = DatasetSplitter(
        raw_image_csv=str(csv_path),
        out_dir=str(out_dir),
        seed=args.seed,
    )

    results = splitter.split(
        mode=args.mode,
        unknown_test_ratio=args.unknown_test_classes_ratio,
        known_test_ratio=args.known_test_classes_ratio,
        unknown_test_count=args.unknown_test_classes_count,
        known_test_count=args.known_test_classes_count,
        min_count_per_class=args.min_count_per_class,
        max_count_per_class=args.max_count_per_class,
        val_ratio=args.val_ratio,
        val_count=args.val_count,
        copy_images=args.copy_images,
        images_dir=Path(args.images_dir) if args.images_dir else None,
    )

    print(f"All outputs saved in {out_dir}")
    print(f"Train: {results['train']}, Val: {results.get('val', 0)}, "
          f"Test known: {results['test_known']}, Test unknown: {results['test_unknown']}")
```

- [ ] **Step 3: 写单元测试**

在 `tests/test_split_csv.py`：

```python
"""Tests for split-csv val and copy-images features."""
import pytest
import pandas as pd
from pathlib import Path
from src.splitting.splitter import DatasetSplitter


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "image": [f"img_{i:03d}.jpg" for i in range(100)],
        "label": (["cat"] * 50 + ["dog"] * 50),
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


def test_val_ratio_creates_val_csv(tmp_path, sample_csv):
    csv_path, _ = sample_csv
    splitter = DatasetSplitter(str(csv_path), str(tmp_path / "out"), seed=42)
    results = splitter.split(mode="ratio", known_test_ratio=0.1, val_ratio=0.1)
    assert (tmp_path / "out" / "val.csv").exists()
    assert results["val"] > 0


def test_no_val_by_default(tmp_path, sample_csv):
    csv_path, _ = sample_csv
    splitter = DatasetSplitter(str(csv_path), str(tmp_path / "out"), seed=42)
    results = splitter.split(mode="ratio", known_test_ratio=0.1)
    assert not (tmp_path / "out" / "val.csv").exists()
    assert results.get("val", 0) == 0


def test_copy_images_creates_subdirs(tmp_path, sample_csv):
    csv_path, df = sample_csv
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # Create dummy image files
    for name in df["image"]:
        (images_dir / name).write_bytes(b"fake")

    splitter = DatasetSplitter(str(csv_path), str(tmp_path / "out"), seed=42)
    splitter.split(
        mode="ratio",
        known_test_ratio=0.1,
        copy_images=True,
        images_dir=images_dir,
    )
    assert (tmp_path / "out" / "images" / "train").is_dir()
    assert (tmp_path / "out" / "images" / "test_known").is_dir()
    # Should have files in train dir
    assert len(list((tmp_path / "out" / "images" / "train").iterdir())) > 0
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_split_csv.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: 冒烟测试**

```bash
python -m entomokit.main split-csv --help
```

Expected: help 含 `--val-ratio`, `--copy-images`, `--images-dir`。

- [ ] **Step 6: Commit**

```bash
git add entomokit/split_csv.py src/splitting/splitter.py tests/test_split_csv.py
git commit -m "feat: add entomokit split-csv command with val split and copy-images support"
```

---

## Task 6: 迁移 `synthesize` 命令

**Files:**
- Create: `entomokit/synthesize.py`
- Reference: `scripts/synthesize.py`

- [ ] **Step 1: 读取 `scripts/synthesize.py` 的完整参数列表**

```bash
python scripts/synthesize.py --help 2>/dev/null || python -c "
import sys; sys.path.insert(0, '.'); 
exec(open('scripts/synthesize.py').read().split('def main')[0])
print('done')
"
```

- [ ] **Step 2: 写 `entomokit/synthesize.py`**

按 `scripts/synthesize.py` 的参数，适配为 register/run 模式，参数名改为连字符风格。关键 COCO 注释参数：

```python
p.add_argument("--annotation-format", default=None,
               choices=["coco", "yolo", "voc"])
p.add_argument("--coco-bbox-format", default="xywh",
               choices=["xywh", "xyxy"],
               help="COCO bbox convention. Only used when --annotation-format=coco.")
```

其余参数按 `scripts/synthesize.py` 原样迁移（改下划线为连字符）。

- [ ] **Step 3: 冒烟测试**

```bash
python -m entomokit.main synthesize --help
```

Expected: help 含 `--target-dir`, `--background-dir`, `--annotation-format`, `--coco-bbox-format`。

- [ ] **Step 4: Commit**

```bash
git add entomokit/synthesize.py
git commit -m "feat: add entomokit synthesize command (CLI wrapper)"
```

---

## Task 7: 更新 `setup.py` — 单一入口点

**Files:**
- Modify: `setup.py`

- [ ] **Step 1: 更新 `setup.py`**

将旧的五个 entry_points 替换为单一入口，并新增 `classify` extras：

```python
entry_points={
    "console_scripts": [
        "entomokit=entomokit.main:main",
    ],
},
extras_require={
    "segmentation": [
        "torch>=2.0.0,<2.4.0",
        "torchvision>=0.15.0,<0.19.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
    ],
    "cleaning": [
        "imagehash",
    ],
    "video": [
        "opencv-python>=4.8.0",
    ],
    "data": [
        "pandas",
    ],
    "classify": [
        "autogluon.multimodal",
        "timm>=0.9.0",
        "umap-learn",
        "pytorch-grad-cam",
        "onnxruntime",
        "scikit-learn",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
},
```

- [ ] **Step 2: 重新安装**

```bash
pip install -e .
```

Expected: 安装成功，`entomokit` 命令可用。

- [ ] **Step 3: 全命令冒烟测试**

```bash
entomokit --help
entomokit segment --help
entomokit extract-frames --help
entomokit clean --help
entomokit split-csv --help
entomokit synthesize --help
entomokit classify --help
```

Expected: 所有命令都有正确的 help 输出，无 import error。

- [ ] **Step 4: Commit**

```bash
git add setup.py
git commit -m "feat: switch to single entomokit entry point, add classify extras_require"
```

---

## Task 8: 运行现有测试套件（验证无回归）

- [ ] **Step 1: 运行全套测试**

```bash
pytest tests/ -v --tb=short
```

Expected: 所有原有测试仍通过，无回归。若有失败，根据错误信息修复（通常是 import 路径问题）。

- [ ] **Step 2: 如有测试失败，修复后重新运行**

```bash
pytest tests/ -v --tb=short
```

Expected: 全部 PASS。

- [ ] **Step 3: 最终整体 commit**

```bash
git add -A
git commit -m "test: verify all existing tests pass after Phase 1 CLI refactor"
```

---

## Phase 1 完成标志

- [ ] `entomokit --help` 显示所有子命令
- [ ] 所有 5 个原有命令通过 `entomokit <cmd> --help` 可访问
- [ ] `entomokit clean --recursive` 参数存在
- [ ] `entomokit extract-frames --input-dir data/video.mp4` 可处理单文件
- [ ] `entomokit split-csv --val-ratio 0.1 --copy-images --images-dir ...` 参数存在
- [ ] `entomokit segment --coco-bbox-format xyxy` 参数存在
- [ ] `pytest tests/ -v` 全部通过
- [ ] 旧脚本 `scripts/` 目录保留未删除

---

## 下一步

Phase 1 完成后，继续：

- **Phase 2**: `segment`/`synthesize` 注释输出格式对齐 detcli（`supervision` 库集成，COCO/YOLO/VOC 布局规范）
- **Phase 3**: `classify` 组实现（train/predict/evaluate/embed/cam/export-onnx）
