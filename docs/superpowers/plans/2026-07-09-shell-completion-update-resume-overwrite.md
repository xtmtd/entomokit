# Shell Completion Fix / `update` Command / `--resume` & `--overwrite` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix shell file-path tab-completion; add `entomokit update`; add `--resume`/`--overwrite` to six processing commands plus `--overwrite` to seven additional output-producing commands; bump version to 0.4.0; update all affected documentation.

**Architecture:** Three independent feature areas plus cross-cutting doc/version updates. Shell completion fix is a pure string change in `completion.py`. `update` is a new module + registration. `--resume`/`--overwrite` adds one shared guard in `src/common/cli.py` and a `skip_existing` flag to three `src/` processor functions. `--overwrite`-only extension to `split-csv` and six `classify` subcommands adds the guard call without resume logic. Documentation updates touch README.md, README.cn.md, two skill files.

**Tech Stack:** Python stdlib only (argparse, pathlib, urllib.request, subprocess, csv, shutil). No new dependencies.

## Global Constraints

- Python ≥ 3.9
- No new pip dependencies
- All tests use `pytest`; no new test libraries
- `extract-frames --skip-existing` removed entirely (no alias)
- `--overwrite` default: non-empty `--out-dir` without either flag → print error, `sys.exit(1)`
- `--resume` for `clean`: suppresses guard only; no per-file skip logic
- Commands without `--resume` call `check_output_dir(out_dir, resume=False, overwrite=args.overwrite)`
- `classify train` maps its existing `--resume` (checkpoint resume) through the guard
- GitHub repo: `https://github.com/xtmtd/entomokit`
- Version bump: `0.3.0` → `0.4.0` in four files

---

## File Map

| Action | File |
|--------|------|
| Modify | `entomokit/completion.py` |
| Create | `entomokit/update.py` |
| Modify | `entomokit/main.py` |
| Modify | `entomokit/_version.py` |
| Modify | `setup.py` |
| Modify | `entomokit/extract_frames.py` |
| Modify | `entomokit/segment.py` |
| Modify | `entomokit/synthesize.py` |
| Modify | `entomokit/augment.py` |
| Modify | `entomokit/clean.py` |
| Modify | `entomokit/measure.py` |
| Modify | `entomokit/split_csv.py` |
| Modify | `entomokit/classify/train.py` |
| Modify | `entomokit/classify/predict.py` |
| Modify | `entomokit/classify/evaluate.py` |
| Modify | `entomokit/classify/embed.py` |
| Modify | `entomokit/classify/cam.py` |
| Modify | `entomokit/classify/export_onnx.py` |
| Modify | `src/segmentation/processor.py` |
| Modify | `src/synthesis/processor.py` |
| Modify | `src/augment/service.py` |
| Modify | `src/measurement/service.py` |
| Modify | `README.md` |
| Modify | `README.cn.md` |
| Modify | `skills/entomokit-workflow/SKILL.md` |
| Modify | `skills/entomokit-workflow/references/command-profiles.md` |
| Modify | `skills/entomokit-workflow/references/workflow.md` |
| Modify | `tests/test_main_cli.py` |
| Create | `tests/test_update.py` |
| Modify | `tests/test_cli_output_logging.py` |
| Create | `tests/test_resume_flags.py` |

---

## Task 1: Fix Shell File-Path Tab-Completion

**Files:**
- Modify: `entomokit/completion.py`
- Modify: `tests/test_main_cli.py`

**Interfaces:**
- Produces: updated `_render_bash_script()`, `_render_zsh_script()`, `_render_fish_script()` — no signature change, only output string changes

- [ ] **Step 1: Write failing tests**

Add to `tests/test_main_cli.py`:

```python
from entomokit.completion import render_completion_script

def test_bash_falls_back_to_files():
    script = render_completion_script("bash")
    assert "complete -o default -F _entomokit_completion entomokit" in script

def test_zsh_calls_files_fallback():
    script = render_completion_script("zsh")
    assert "_files" in script

def test_fish_no_global_no_files_flag():
    script = render_completion_script("fish")
    first_line = script.splitlines()[0]
    assert "-f" not in first_line
```

- [ ] **Step 2: Run to confirm failure**

```
cd /Users/zf/data/coding/entomokit
pytest tests/test_main_cli.py::test_bash_falls_back_to_files tests/test_main_cli.py::test_zsh_calls_files_fallback tests/test_main_cli.py::test_fish_no_global_no_files_flag -v
```

Expected: 3× FAILED

- [ ] **Step 3: Apply bash fix in `completion.py`**

In `_render_bash_script()`, change the final string line:
```python
# Before:
"complete -F _entomokit_completion entomokit\n"
# After:
"complete -o default -F _entomokit_completion entomokit\n"
```

- [ ] **Step 4: Apply zsh fix in `completion.py`**

In `_render_zsh_script()`, the return string ends with:
```python
        "  esac\n"
        "}\n"
        "compdef _entomokit entomokit\n"
```
Change to:
```python
        "  esac\n"
        "  _files\n"
        "}\n"
        "compdef _entomokit entomokit\n"
```

- [ ] **Step 5: Apply fish fix in `completion.py`**

In `_render_fish_script()`, the `lines` list begins with:
```python
lines = [
    "complete -c entomokit -f",
    ...
]
```
Change to:
```python
lines = [
    "complete -c entomokit",
    ...
]
```

- [ ] **Step 6: Run tests**

```
pytest tests/test_main_cli.py::test_bash_falls_back_to_files tests/test_main_cli.py::test_zsh_calls_files_fallback tests/test_main_cli.py::test_fish_no_global_no_files_flag -v
```

Expected: 3× PASSED

---

## Task 2: Version Bump to 0.4.0

**Files:**
- Modify: `entomokit/_version.py`
- Modify: `setup.py`
- Modify: `README.md` (line 7)
- Modify: `README.cn.md` (line 7)

**Interfaces:**
- Produces: `__version__ = "0.4.0"`, `__commit__ = "unknown"`, and `__commit_date__ = "unknown"` in `_version.py` (Task 3 consumes these values)

- [ ] **Step 1: Update `entomokit/_version.py`**

```python
"""Package version metadata."""

__version__ = "0.4.0"
__commit__ = "unknown"
__commit_date__ = "unknown"
```

- [ ] **Step 2: Update `setup.py`**

Change:
```python
version="0.3.0",
```
To:
```python
version="0.4.0",
```

Do not add a custom `cmdclass` for commit injection. It is easy to break normal setuptools install/develop behavior and `unknown` is acceptable for this release.

`# ponytail: release-time stamping can be added when release automation exists`

- [ ] **Step 3: Update `README.md` line 7**

Change:
```
Current release in this repository: `0.3.0`.
```
To:
```
Current release in this repository: `0.4.0`.
```

- [ ] **Step 4: Update `README.cn.md` line 7**

Change:
```
当前仓库版本：`0.3.0`。
```
To:
```
当前仓库版本：`0.4.0`。
```

---

## Task 3: Add `entomokit update` Command

**Files:**
- Create: `entomokit/update.py`
- Modify: `entomokit/main.py`
- Create: `tests/test_update.py`

**Interfaces:**
- Consumes: `entomokit._version.__version__`, `entomokit._version.__commit__`, `entomokit._version.__commit_date__`, GitHub Commits API (`https://api.github.com/repos/xtmtd/entomokit/commits/main`)
- Produces: `register(subparsers)` and `run(args)` — same convention as all other command modules

- [ ] **Step 1: Write tests**

Create `tests/test_update.py`:

```python
import json
from unittest.mock import patch, MagicMock
from entomokit.update import fetch_latest_commit, _status

def _make_response(sha: str, date: str, message: str) -> MagicMock:
    payload = json.dumps({
        "sha": sha,
        "commit": {
            "author": {"date": date},
            "message": message,
        }
    }).encode()
    mock = MagicMock()
    mock.read.return_value = payload
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock

def test_fetch_latest_commit_parses():
    with patch("urllib.request.urlopen", return_value=_make_response("abc1234def", "2026-07-10T00:00:00Z", "feat: X")):
        sha, date, msg = fetch_latest_commit()
    assert sha == "abc1234"   # first 7 chars
    assert "2026-07-10" in date
    assert "feat: X" in msg

def test_status_same_sha():
    assert _status("abc1234", "2026-07-10", "abc1234", "2026-07-11") == "same"

def test_status_remote_newer():
    assert _status("abc1234", "2026-07-09", "def5678", "2026-07-10") == "newer"

def test_status_unknown_local():
    assert _status("unknown", "unknown", "abc1234", "2026-07-10") == "unknown"
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_update.py -v
```

Expected: ImportError / FAILED

- [ ] **Step 3: Create `entomokit/update.py`**

```python
"""entomokit update — check for updates from GitHub and optionally install."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from typing import Tuple

from entomokit._version import __version__, __commit__, __commit_date__

_REPO = "xtmtd/entomokit"
_API_URL = f"https://api.github.com/repos/{_REPO}/commits/main"
_INSTALL_URL = f"git+https://github.com/{_REPO}.git"


def fetch_latest_commit(timeout: int = 10) -> Tuple[str, str, str]:
    """Return (short_sha, date_str, first_line_of_message)."""
    req = urllib.request.Request(
        _API_URL,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    entry = data
    sha = entry["sha"][:7]
    date = entry["commit"]["author"]["date"][:10]   # YYYY-MM-DD
    message = entry["commit"]["message"].splitlines()[0]
    return sha, date, message


def _status(local_commit: str, local_date: str, remote_sha: str, remote_date: str) -> str:
    """Return same, newer, or unknown."""
    if local_commit == remote_sha:
        return "same"
    if local_commit == "unknown" or local_date == "unknown":
        return "unknown"
    return "newer" if remote_date > local_date else "same"


def register(subparsers: argparse._SubParsersAction) -> None:
    from entomokit.help_style import RichHelpFormatter, style_parser

    p = subparsers.add_parser(
        "update",
        help="Check for updates and optionally install the latest version from GitHub.",
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument(
        "--check",
        action="store_true",
        help="Only show version information; do not install.",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt and install immediately.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    local_ver = __version__
    local_sha = __commit__
    local_date = __commit_date__
    print(f"Current version : {local_ver} ({local_sha})")
    print("Checking GitHub for updates...")

    try:
        remote_sha, remote_date, remote_msg = fetch_latest_commit()
    except Exception as exc:
        print(f"Error: could not reach GitHub — {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Latest commit   : {remote_sha}  ({remote_date})  {remote_msg}")

    status = _status(local_sha, local_date, remote_sha, remote_date)
    if status == "same":
        print("Already up to date.")
        return

    if status == "unknown" and not args.yes:
        print("Local commit/date is unknown. Re-run with --yes to install anyway.")
        return

    if args.check:
        print("(Run without --check to install the update.)")
        return

    if not args.yes:
        answer = input("Proceed with update? [y/N] ").strip().lower()
        if answer != "y":
            print("Update cancelled.")
            return

    print(f"Installing latest from GitHub ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", _INSTALL_URL],
        check=False,
    )
    if result.returncode == 0:
        print("Update complete. Restart your shell to use the new version.")
    else:
        print("Update failed. Check pip output above.", file=sys.stderr)
        sys.exit(result.returncode)
```

- [ ] **Step 4: Register in `entomokit/main.py`**

In `_build_parser()`, after the existing imports (around line 57), add:
```python
from entomokit import update as _update
```

After `_doctor.register(subparsers)` and before `_completion.register(subparsers)`, add:
```python
_update.register(subparsers)
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_update.py -v
```

Expected: 4× PASSED

- [ ] **Step 6: Smoke-test**

```
python -m entomokit.main update --check
```

Expected: shows version info, exits without prompting.

---

## Task 4: Shared Output-Dir Guard

**Files:**
- Modify: `src/common/cli.py`
- Modify: `tests/test_cli_output_logging.py`

This task adds the production guard used by Tasks 5-10. Do not copy-paste this guard into command modules.

- [ ] **Step 1: Write tests**

Add to `tests/test_cli_output_logging.py`:

```python
import pytest

from src.common.cli import check_output_dir

def test_guard_exits_on_nonempty_no_flags(tmp_path):
    p = tmp_path / "out"
    p.mkdir()
    (p / "x.txt").write_text("x")
    with pytest.raises(SystemExit):
        check_output_dir(p, resume=False, overwrite=False)

def test_guard_resume_allows_nonempty(tmp_path):
    p = tmp_path / "out"
    p.mkdir()
    (p / "x.txt").write_text("x")
    check_output_dir(p, resume=True, overwrite=False)
    assert (p / "x.txt").exists()

def test_guard_overwrite_clears_dir(tmp_path):
    p = tmp_path / "out"
    p.mkdir()
    (p / "x.txt").write_text("x")
    check_output_dir(p, resume=False, overwrite=True)
    assert not (p / "x.txt").exists()
    assert p.exists()

def test_guard_creates_missing_dir(tmp_path):
    p = tmp_path / "out" / "nested"
    check_output_dir(p, resume=False, overwrite=False)
    assert p.exists()

def test_guard_empty_dir_no_error(tmp_path):
    p = tmp_path / "out"
    p.mkdir()
    check_output_dir(p, resume=False, overwrite=False)
    assert p.exists()
```

- [ ] **Step 2: Add implementation to `src/common/cli.py`**

Add near the other CLI filesystem helpers:

```python
def check_output_dir(out_dir: Path, resume: bool, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if overwrite:
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
        elif not resume:
            print(
                f"Error: output directory '{out_dir}' is not empty.\n"
                "Use --resume to continue or --overwrite to start fresh.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
```

If `shutil` or `sys` is not already imported in `src/common/cli.py`, import it.

- [ ] **Step 3: Run**

```
pytest tests/test_cli_output_logging.py -v
```

Expected: 5× PASSED

---

## Task 5: `--resume` / `--overwrite` for `segment`

**Files:**
- Modify: `entomokit/segment.py`
- Modify: `src/segmentation/processor.py`
- Create: `tests/test_resume_flags.py` (first section)

**Interfaces:**
- `process_directory()` gains `skip_existing: bool = False`; when True, skips images whose stem already has output in `output_dir/images/`

- [ ] **Step 1: Add `skip_existing` parameter to `src/segmentation/processor.py`**

Locate `def process_directory(self, input_dir, output_dir, num_workers=1, disable_tqdm=False, output_format="png", shutdown_flag=None)` (around line 997).

Change signature to:
```python
def process_directory(
    self,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    num_workers: int = 1,
    disable_tqdm: bool = False,
    output_format: str = "png",
    shutdown_flag: Optional[Callable[[], bool]] = None,
    skip_existing: bool = False,
) -> Dict[str, Any]:
```

Inside the loop `for img_path in tqdm(image_paths, ...)`, add before `image = load_image(img_path)`:
```python
            if skip_existing:
                images_dir = output_dir / "images"
                if any(images_dir.glob(f"{img_path.stem}*")):
                    results.setdefault("skipped", 0)
                    results["skipped"] += 1
                    continue
```

- [ ] **Step 2: Add argparse flags to `entomokit/segment.py` `register()`**

After the last existing `add_argument` call and before `p.set_defaults(func=run)`, add:
```python
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in --out-dir and continue a previous run.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --out-dir contents and start fresh.",
    )
```

- [ ] **Step 3: Add guard and wire `skip_existing` in `entomokit/segment.py` `run()`**

At top of `run()`, add the import after the local imports block:
```python
    from src.common.cli import check_output_dir
```

Replace the bare `out_dir.mkdir(parents=True, exist_ok=True)` line with:
```python
    check_output_dir(out_dir, args.resume, args.overwrite)
```

In the `processor.process_directory(...)` call, add `skip_existing=args.resume`.

- [ ] **Step 4: Write integration test (first entry in `tests/test_resume_flags.py`)**

Create `tests/test_resume_flags.py`:

```python
"""Integration tests for --resume / --overwrite across processing commands."""
import argparse
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


# ── segment ──────────────────────────────────────────────────────────────────

def _segment_args(**kw):
    d = dict(
        input_dir="/nonexistent", out_dir="/tmp/seg_out",
        segmentation_method="otsu", confidence_threshold=0.3,
        min_area_ratio=0.01, max_area_ratio=0.9, sam3_checkpoint=None,
        lama_model=None, coco_output_mode="single", output_format="png",
        num_workers=1, verbose=False, resume=False, overwrite=False,
        annotation_output_format=None, iou_threshold=0.1, hint="insect",
        padding_ratio=0.0, repair_strategy=None, device="auto",
    )
    d.update(kw)
    return argparse.Namespace(**d)

def test_segment_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "dummy.png").write_bytes(b"")
    from entomokit import segment
    with pytest.raises(SystemExit):
        segment.run(_segment_args(out_dir=str(out), input_dir=str(tmp_path)))

def test_segment_overwrite_clears(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "dummy.png").write_bytes(b"")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    with patch("src.segmentation.processor.SegmentationProcessor") as MockP:
        MockP.return_value.process_directory.return_value = {
            "processed": 0, "failed": 0, "output_files": []
        }
        from entomokit import segment
        segment.run(_segment_args(out_dir=str(out), input_dir=str(in_dir), overwrite=True))
    assert not (out / "dummy.png").exists()
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_resume_flags.py::test_segment_exits_on_nonempty tests/test_resume_flags.py::test_segment_overwrite_clears -v
```

Expected: 2× PASSED

---

## Task 6: `--resume` / `--overwrite` for `synthesize`

**Files:**
- Modify: `entomokit/synthesize.py`
- Modify: `src/synthesis/processor.py`
- Modify: `tests/test_resume_flags.py`

**Interfaces:**
- `process_directory()` gains `skip_existing: bool = False`; when True, skips target images whose stem already has output matching `(output_dir / self.output_subdir).glob(f"{target_path.stem}_*")`

- [ ] **Step 1: Add `skip_existing` to `src/synthesis/processor.py` `process_directory()`**

Locate `def process_directory(self, target_dir, background_dir, output_dir, num_syntheses=10, disable_tqdm=False, threads=1)` (around line 985).

Add `skip_existing: bool = False` to the signature.

In the loop `for target_path in target_paths:`, after `target_img = self._load_image(target_path)`:
```python
            if skip_existing:
                img_out = output_dir / self.output_subdir
                if any(img_out.glob(f"{target_path.stem}_*")):
                    skipped_images += 1
                    continue
```

- [ ] **Step 2: Add argparse flags to `entomokit/synthesize.py` `register()` and guard in `run()`**

Add `--resume` and `--overwrite` arguments to `register()`. In `run()`, import `check_output_dir` from `src.common.cli`, replace the bare `out_dir.mkdir(...)` call with `check_output_dir(out_dir, args.resume, args.overwrite)`, and pass `skip_existing=args.resume` to `process_directory()`.

The `out_dir` variable in `synthesize.py`'s `run()` is assigned from `args.out_dir`; locate the `mkdir` call and apply the guard there.

- [ ] **Step 3: Add test to `tests/test_resume_flags.py`**

```python
# ── synthesize ───────────────────────────────────────────────────────────────

def _synthesize_args(**kw):
    d = dict(
        target_dir="/nonexistent", background_dir="/nonexistent",
        out_dir="/tmp/syn_out", num_syntheses=10,
        area_ratio_min=0.1, area_ratio_max=0.5,
        color_match_strength=0.3, avoid_black_regions=True,
        rotate=True, out_image_format="png",
        annotation_output_format="none", coco_output_mode="single",
        threads=1, verbose=False, resume=False, overwrite=False,
        coco_bbox_format="xywh",
    )
    d.update(kw)
    return argparse.Namespace(**d)

def test_synthesize_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.png").write_bytes(b"")
    from entomokit import synthesize
    with pytest.raises(SystemExit):
        synthesize.run(_synthesize_args(out_dir=str(out)))
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_resume_flags.py::test_synthesize_exits_on_nonempty -v
```

Expected: PASSED

---

## Task 7: `--resume` / `--overwrite` for `augment`

**Files:**
- Modify: `entomokit/augment.py`
- Modify: `src/augment/service.py`
- Modify: `tests/test_resume_flags.py`

**Interfaces:**
- `run_augment()` in `src/augment/service.py` gains `skip_existing: bool = False`; when True, skips source images whose stem already has output in the images output dir (`dst / "images"`)

- [ ] **Step 1: Add `skip_existing` to `run_augment()` in `src/augment/service.py`**

Locate `def run_augment(input_dir, ...) -> ...:` (around line 32). Add `skip_existing: bool = False` to the signature.

The variable `images_out = dst / "images"` is set around line 55. In the loop `for img_path in image_paths:` (around line 65), add before `img_array = cv2.imread(...)`:
```python
        if skip_existing and any(images_out.glob(f"{img_path.stem}*")):
            continue
```

- [ ] **Step 2: Add argparse flags to `entomokit/augment.py` and guard in `run()`**

Add `--resume` and `--overwrite` arguments to `register()`. In `run()`, import `check_output_dir` from `src.common.cli`, replace the bare `out_dir.mkdir(...)` call with `check_output_dir(out_dir, args.resume, args.overwrite)`, and pass `skip_existing=args.resume` to `run_augment(...)`.

- [ ] **Step 3: Add test to `tests/test_resume_flags.py`**

```python
# ── augment ──────────────────────────────────────────────────────────────────

def _augment_args(**kw):
    d = dict(
        input_dir="/nonexistent", out_dir="/tmp/aug_out",
        preset="light", policy=None, seed=42, multiply=2,
        verbose=False, resume=False, overwrite=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)

def test_augment_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.png").write_bytes(b"")
    from entomokit import augment
    with pytest.raises(SystemExit):
        augment.run(_augment_args(out_dir=str(out)))
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_resume_flags.py::test_augment_exits_on_nonempty -v
```

Expected: PASSED

---

## Task 8: `--resume` / `--overwrite` for `clean`

**Files:**
- Modify: `entomokit/clean.py`
- Modify: `tests/test_resume_flags.py`

**Note:** The cleaner's dedup logic already handles per-file skipping. `--resume` here only bypasses the non-empty dir guard. No `src/` change needed.

- [ ] **Step 1: Add argparse flags to `entomokit/clean.py` `register()`**

Add after the last existing `add_argument` and before `p.set_defaults(func=run)`:
```python
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue into a non-empty output directory without error.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --out-dir contents and start fresh.",
    )
```

- [ ] **Step 2: Add guard in `entomokit/clean.py` `run()`**

Replace the bare `out_dir.mkdir(parents=True, exist_ok=True)` with `check_output_dir(out_dir, args.resume, args.overwrite)`.

- [ ] **Step 3: Add test**

```python
# ── clean ────────────────────────────────────────────────────────────────────

def _clean_args(**kw):
    d = dict(
        input_dir="/nonexistent", out_dir="/tmp/clean_out",
        out_short_size=None, out_image_format="jpg", threads=4,
        keep_exif=False, dedup_mode="phash", phash_threshold=5,
        recursive=False, verbose=False, resume=False, overwrite=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)

def test_clean_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.jpg").write_bytes(b"")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    from entomokit import clean
    with pytest.raises(SystemExit):
        clean.run(_clean_args(out_dir=str(out), input_dir=str(in_dir)))
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_resume_flags.py::test_clean_exits_on_nonempty -v
```

Expected: PASSED

---

## Task 9: `--resume` / `--overwrite` for `measure`

**Files:**
- Modify: `entomokit/measure.py`
- Modify: `src/measurement/service.py`
- Modify: `tests/test_resume_flags.py`

**Interfaces:**
- `run_batch()` in `src/measurement/service.py` gains `existing_rows: list[dict[str, str]] | None = None`. When provided, masks whose `file_name` stem is already present are skipped; existing rows and new rows are combined, then all CSV outputs are rewritten from the combined rows.
- CSV key field: `file_name` (confirmed from source — `{"file_name": path.name}`)

- [ ] **Step 1: Modify `run_batch()` in `src/measurement/service.py`**

Current signature:
```python
def run_batch(mask_dir: Path, out_dir: Path, pixel_size_um: float | None) -> dict[str, int]:
```

New signature:
```python
def run_batch(
    mask_dir: Path,
    out_dir: Path,
    pixel_size_um: float | None,
    existing_rows: list[dict[str, str]] | None = None,
) -> dict[str, int]:
```

Inside the function, change:
```python
    files = iter_mask_files(mask_dir)
    rows = [measure_one_mask(path, pixel_size_um=pixel_size_um) for path in files]
```
To:
```python
    existing_rows = existing_rows or []
    skip_set = {Path(row["file_name"]).stem for row in existing_rows if row.get("file_name")}
    files = [p for p in iter_mask_files(mask_dir) if p.stem not in skip_set]
    rows = [measure_one_mask(path, pixel_size_um=pixel_size_um) for path in files]
```

Combine old and new rows before writing CSVs:
```python
    rows = [*existing_rows, *rows]
    _write_csv(out_dir / "metrics.csv", rows)
```

Summary and definitions CSVs are overwritten from the combined rows.

- [ ] **Step 2: Add argparse flags and resume logic to `entomokit/measure.py` `run()`**

Add flags to `register()`:
```python
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append measurements for new masks; skip masks already in metrics.csv.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --out-dir contents and start fresh.",
    )
```

In `run()`, replace `out_dir.mkdir(parents=True, exist_ok=True)` with the shared guard, then read existing rows:
```python
    check_output_dir(out_dir, args.resume, args.overwrite)

    existing_rows: list[dict[str, str]] = []
    if args.resume:
        import csv as _csv
        existing_csv = out_dir / "metrics.csv"
        if existing_csv.exists():
            with existing_csv.open(newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                existing_rows = list(reader)
            print(f"Resuming: skipping {len(existing_rows)} already-measured masks.")
```

Pass `existing_rows=existing_rows` to `run_batch(...)`.

- [ ] **Step 3: Add test**

```python
# ── measure ──────────────────────────────────────────────────────────────────

def _measure_args(**kw):
    d = dict(
        mask_dir="/nonexistent", out_dir="/tmp/meas_out",
        pixel_size_um=None, threads=1, verbose=False,
        resume=False, overwrite=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)

def test_measure_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "metrics.csv").write_text("file_name\na.png\n")
    from entomokit import measure
    with pytest.raises(SystemExit):
        measure.run(_measure_args(out_dir=str(out)))
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_resume_flags.py::test_measure_exits_on_nonempty -v
```

Expected: PASSED

---

## Task 10: Rename `extract-frames --skip-existing` → `--resume`, add `--overwrite`

**Files:**
- Modify: `entomokit/extract_frames.py`
- Modify: `tests/test_resume_flags.py`

**Note:** `--skip-existing` is removed entirely. Internally `args.skip_existing` becomes `args.resume`. The `src/framing/extractor.py` is not wired to `skip_existing` at CLI level (it was never connected); the extractor's `extract_all()` has its own logic. Check the actual call and wire `args.resume` correctly.

- [ ] **Step 1: Verify how `--skip-existing` was used in `extract_frames.py`**

```bash
rg -n "skip_existing|skip.existing" /Users/zf/data/coding/entomokit/entomokit/extract_frames.py /Users/zf/data/coding/entomokit/src/framing/extractor.py
```

If `args.skip_existing` is passed to the extractor, note the parameter name. If not (i.e., it was defined but never wired), it was a no-op — in that case the rename is safe but we must also wire it now.

- [ ] **Step 2: Replace `--skip-existing` with `--resume` and add `--overwrite` in `register()`**

Replace:
```python
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames that already exist (resume).",
    )
```
With:
```python
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip frames already present in --out-dir (continue a previous run).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --out-dir contents and start fresh.",
    )
```

- [ ] **Step 3: Update `run()` in `extract_frames.py`**

Replace the bare `out_dir.mkdir(parents=True, exist_ok=True)` with `check_output_dir(out_dir, args.resume, args.overwrite)`.

Change any `args.skip_existing` reference to `args.resume`. Also wire `args.resume` into the extractor if it supports a skip-existing parameter (check the grep result from Step 1).

- [ ] **Step 4: Add test**

```python
# ── extract-frames ───────────────────────────────────────────────────────────

def _ef_args(**kw):
    d = dict(
        input_dir="/nonexistent", out_dir="/tmp/ef_out",
        out_image_format="jpg", threads=1, max_frames=None,
        start_time=0.0, end_time=None, interval=1000,
        resume=False, overwrite=False, verbose=False, quiet=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)

def test_ef_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "frame_001.jpg").write_bytes(b"")
    from entomokit import extract_frames
    with pytest.raises(SystemExit):
        extract_frames.run(_ef_args(out_dir=str(out)))

def test_ef_no_skip_existing_in_help():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "entomokit.main", "extract-frames", "--help"],
        capture_output=True, text=True,
    )
    assert "--skip-existing" not in result.stdout
    assert "--resume" in result.stdout
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_resume_flags.py::test_ef_exits_on_nonempty tests/test_resume_flags.py::test_ef_no_skip_existing_in_help -v
```

Expected: 2× PASSED

---

## Task 10b: `--overwrite` for `split-csv` and `classify` Subcommands

**Files:**
- Modify: `entomokit/split_csv.py`
- Modify: `entomokit/classify/train.py`
- Modify: `entomokit/classify/predict.py`
- Modify: `entomokit/classify/evaluate.py`
- Modify: `entomokit/classify/embed.py`
- Modify: `entomokit/classify/cam.py`
- Modify: `entomokit/classify/export_onnx.py`

These commands produce output but have no per-item resume logic. Add `--overwrite` arg and `check_output_dir` guard for output-dir safety. `classify train` maps its existing `--resume` (checkpoint resume) through the guard.

### Interfaces

All seven `run()` functions gain `check_output_dir(out_dir, resume=<bool>, overwrite=args.overwrite)` before creating directories.

- `split-csv`: `resume=False`
- `classify train`: `resume=args.resume` (existing AutoGluon checkpoint resume flag aligns with the guard)
- `classify predict`: `resume=False`
- `classify evaluate`: `resume=False`
- `classify embed`: `resume=False`
- `classify cam`: `resume=False`
- `classify export-onnx`: `resume=False`

### Steps

- [ ] **Step 1: Add `--overwrite` arg and `check_output_dir` call to `split_csv.py`**

In `register()`:
```python
p.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete --out-dir contents and regenerate all splits.",
)
```

In `run()`, replace `out_dir.mkdir(...)` with:
```python
from src.common.cli import check_output_dir, setup_shutdown_handler, save_log
...
out_dir = Path(args.out_dir)
check_output_dir(out_dir, resume=False, overwrite=args.overwrite)
save_log(out_dir, args)
```

- [ ] **Step 2: Add `--overwrite` arg and `check_output_dir` call to `classify/train.py`**

In `register()`, after `--resume` arg:
```python
p.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete --out-dir contents and train from scratch.",
)
```

In `run()`, replace `out_dir.mkdir(...)` with:
```python
from src.common.cli import check_output_dir, save_log
...
out_dir = Path(args.out_dir)
check_output_dir(out_dir, resume=args.resume, overwrite=args.overwrite)
(out_dir / "logs").mkdir(exist_ok=True)
save_log(out_dir / "logs", args, log_filename="log.txt")
```

- [ ] **Step 3: Add `--overwrite` arg and `check_output_dir` call to `classify/predict.py`**

In `register()`, before `--device`:
```python
p.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete --out-dir contents and re-predict all inputs.",
)
```

In `run()`, before `pred_dir.mkdir(...)`:
```python
from src.common.cli import check_output_dir, save_log
...
out_dir = Path(args.out_dir)
check_output_dir(out_dir, resume=False, overwrite=args.overwrite)
pred_dir = out_dir / "predictions"
pred_dir.mkdir(parents=True, exist_ok=True)
save_log(out_dir, args)
```

- [ ] **Step 4: Add `--overwrite` arg and `check_output_dir` call to `classify/evaluate.py`**

Same pattern as Step 3; replace `out_dir.mkdir(...)` with `check_output_dir(out_dir, resume=False, overwrite=args.overwrite)`.

- [ ] **Step 5: Add `--overwrite` arg and `check_output_dir` call to `classify/embed.py`**

In `register()`, before `--metrics-sample-size`:
```python
p.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete --out-dir contents and re-extract embeddings.",
)
```

In `run()`, before `logs_dir.mkdir(...)`:
```python
from src.common.cli import check_output_dir, save_log
...
out_dir = Path(args.out_dir)
check_output_dir(out_dir, resume=False, overwrite=args.overwrite)
logs_dir = out_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
save_log(logs_dir, args)
```

- [ ] **Step 6: Add `--overwrite` arg and `check_output_dir` call to `classify/cam.py`**

In `register()`, before `--device`:
```python
p.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete --out-dir contents and regenerate CAM visualizations.",
)
```

In `run()`, replace `out_dir.mkdir(...)` with `check_output_dir(out_dir, resume=False, overwrite=args.overwrite)`.

- [ ] **Step 7: Add `--overwrite` arg and `check_output_dir` call to `classify/export_onnx.py`**

In `register()`, before `--sample-image`:
```python
p.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete --out-dir contents and re-export ONNX model.",
)
```

In `run()`, before `save_log`:
```python
from src.common.cli import check_output_dir, save_log
...
out_dir = Path(args.out_dir)
check_output_dir(out_dir, resume=False, overwrite=args.overwrite)
save_log(out_dir, args)
```

- [ ] **Step 8: Verify `--help` for all seven commands**

```bash
for cmd in "split-csv" "classify train" "classify predict" "classify evaluate" "classify embed" "classify cam" "classify export-onnx"; do
    python -m entomokit.main $cmd --help | grep -q "\-\-overwrite" || echo "MISSING: $cmd"
done
```

Expected: no output (all seven have `--overwrite` in `--help`).

- [ ] **Step 9: Run smoke tests**

```
pytest tests/test_resume_flags.py -v
```

---

## Task 11: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `README.cn.md`
- Modify: `skills/entomokit-workflow/SKILL.md`
- Modify: `skills/entomokit-workflow/references/command-profiles.md`
- Modify: `skills/entomokit-workflow/references/workflow.md`

This task is purely documentation; no code changes.

### README.md changes

- [ ] **Step 1: Add `update` row to commands table (around line 33)**

After the `| \`doctor\`` row, add:
```
| `update` | Check for updates and optionally install the latest version from GitHub |
```

- [ ] **Step 2: Add `### Update Command` section after Doctor Command section (around line 931)**

```markdown
### Update Command

Check whether a newer version is available on GitHub and optionally install it.

```bash
entomokit update           # check and prompt
entomokit update --check   # check only, no install
entomokit update --yes     # install without prompt
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--check` | Only show version info; do not install | No |
| `--yes`, `-y` | Skip confirmation prompt | No |
```

- [ ] **Step 3: Fix Shell Completion section in Common Behaviours (around line 956)**

Replace:
```markdown
```bash
entomokit --install-completion
```
```
With:
```markdown
```bash
entomokit completion bash --install
entomokit completion zsh --install
entomokit completion fish --install
```
```

- [ ] **Step 4: Add `--resume` / `--overwrite` to segment param table (lines 337–349)**

Add two rows after `--threads`:
```
| `--resume` | Skip images already present in `--out-dir` (continue previous run) | No |
| `--overwrite` | Delete `--out-dir` contents and start fresh | No |
```

- [ ] **Step 5: Add `--resume` / `--overwrite` to measure param table (lines 390–394)**

Add two rows after `--verbose, -v`:
```
| `--resume` | Append measurements for new masks; skip masks already in `metrics.csv` | No |
| `--overwrite` | Delete `--out-dir` contents and start fresh | No |
```

- [ ] **Step 6: Update extract-frames param table (lines 434–441)**

Remove any `--skip-existing` row if present. Add:
```
| `--resume` | Skip frames already present in `--out-dir` (continue previous run) | No |
| `--overwrite` | Delete `--out-dir` contents and start fresh | No |
```

- [ ] **Step 7: Add `--resume` / `--overwrite` to clean param table (lines 470–478)**

Add after `--threads`:
```
| `--resume` | Continue into a non-empty `--out-dir` without error | No |
| `--overwrite` | Delete `--out-dir` contents and start fresh | No |
```

- [ ] **Step 8: Add `--resume` / `--overwrite` to augment param table (lines 501–506)**

Add after `--multiply`:
```
| `--resume` | Skip source images already augmented in `--out-dir` | No |
| `--overwrite` | Delete `--out-dir` contents and start fresh | No |
```

- [ ] **Step 9: Add `--resume` / `--overwrite` to synthesize param table (lines 618–629)**

Add after `--threads`:
```
| `--resume` | Skip target images already synthesised in `--out-dir` | No |
| `--overwrite` | Delete `--out-dir` contents and start fresh | No |
```

- [ ] **Step 9b: Add `--overwrite` to split-csv param table (lines 587–598)**

Add a row after `--seed`:
```
| `--overwrite` | Delete `--out-dir` contents and regenerate all splits | No |
```

- [ ] **Step 9c: Add `--overwrite` to classify train param table (lines 757–768)**

Add after `--num-workers`:
```
| `--overwrite` | Delete `--out-dir` contents and train from scratch | No |
```

- [ ] **Step 9d: Add `--overwrite` mention to classify predict/evaluate/embed/cam/export-onnx sections**

These sections have no formal param tables. Add a note in each section's prose:

- `classify predict` (around line 869): "The `--overwrite` flag deletes `--out-dir` contents and re-predicts all inputs."
- `classify evaluate` (around line 908): "The `--overwrite` flag deletes `--out-dir` contents and re-evaluates."
- `classify embed` (around line 944): "The `--overwrite` flag deletes `--out-dir` contents and re-extracts embeddings."
- `classify cam` (around line 990): "The `--overwrite` flag deletes `--out-dir` contents and regenerates CAM visualizations."
- `classify export-onnx` (around line 1015): "The `--overwrite` flag deletes `--out-dir` contents and re-exports the ONNX model."

### README.cn.md changes

- [ ] **Step 10: Apply the same nine changes as above in Chinese**

Parallel positions in `README.cn.md`. Translations:
- `update` row description: `从 GitHub 检查更新并可选地安装最新版本`
- `--check`: `仅显示版本信息；不安装` / `--yes`: `跳过确认提示`
- Shell Completion fix: same code block substitution
- `--resume` (segment/augment/synthesize): `跳过 --out-dir 中已存在的文件，继续上次运行`
- `--resume` (clean): `允许进入非空输出目录，不报错`
- `--resume` (measure): `追加新掩码的测量结果；跳过 metrics.csv 中已有的记录`
- `--overwrite` (all): `删除 --out-dir 内容并重新开始`

### Skill file changes

- [ ] **Step 11: Update `skills/entomokit-workflow/SKILL.md`**

In the `description` field (line 3), add `update` to the commands list:
```
...entomokit commands (doctor, **update**, clean, segment, measure, synthesize, augment, split-csv, classify)...
```

In Core Rule 9 (around line 9), after "do not overwrite prior artifacts by default", add a parenthetical:
```
(CLI-level --resume/--overwrite flags provide per-run control within a single --out-dir; the skill-layer new-directory policy is a higher-level default)
```

- [ ] **Step 12: Update `skills/entomokit-workflow/references/command-profiles.md`**

In the `## clean` section, add:
```
- Non-empty `--out-dir` requires explicit `--resume` (continue) or `--overwrite` (fresh start); default exits with an error.
```

In the `## segment` section, add:
```
- Non-empty `--out-dir` requires `--resume` (skip already-segmented images) or `--overwrite`; default exits with an error.
```

In `## Retry and Rerun`, add a new bullet:
```
- CLI-level `--resume`/`--overwrite` flags apply within a single `--out-dir`. The skill-layer default (new sibling directory) is recommended for clean separation; `--resume` is appropriate when a run was interrupted and the partial output is valid.
```

- [ ] **Step 13: Update `skills/entomokit-workflow/references/workflow.md`**

In Phase 0, after step 1 (`Run entomokit doctor`), add step 1b:
```
1b. Optionally run `entomokit update --check` to confirm the installed version is current.
```

In Phase 1, after the `segment` method note, add:
```
Output directory note for Phase 1 commands (segment, measure, synthesize, augment, clean, extract-frames):
- Default: non-empty `--out-dir` causes an error — use a fresh path or pass `--resume`/`--overwrite`.
- `--resume`: continue an interrupted run by skipping already-processed items.
- `--overwrite`: wipe `--out-dir` and start fresh.
```

---

## Task 12: Full Test Suite + Completion Re-install

- [ ] **Step 1: Run full test suite**

```
cd /Users/zf/data/coding/entomokit
pytest -v
```

Expected: all new tests pass; no pre-existing tests broken.

- [ ] **Step 2: Verify all new flags appear in help**

```bash
python -m entomokit.main segment --help | grep -E "resume|overwrite"
python -m entomokit.main synthesize --help | grep -E "resume|overwrite"
python -m entomokit.main augment --help | grep -E "resume|overwrite"
python -m entomokit.main clean --help | grep -E "resume|overwrite"
python -m entomokit.main measure --help | grep -E "resume|overwrite"
    python -m entomokit.main extract-frames --help | grep -E "resume|overwrite|skip-existing"
    python -m entomokit.main split-csv --help | grep -q "\-\-overwrite" || echo "MISSING split-csv"
    python -m entomokit.main classify train --help | grep -q "\-\-overwrite" || echo "MISSING classify train"
    python -m entomokit.main classify predict --help | grep -q "\-\-overwrite" || echo "MISSING classify predict"
    python -m entomokit.main classify evaluate --help | grep -q "\-\-overwrite" || echo "MISSING classify evaluate"
    python -m entomokit.main classify embed --help | grep -q "\-\-overwrite" || echo "MISSING classify embed"
    python -m entomokit.main classify cam --help | grep -q "\-\-overwrite" || echo "MISSING classify cam"
    python -m entomokit.main classify export-onnx --help | grep -q "\-\-overwrite" || echo "MISSING classify export-onnx"
    python -m entomokit.main update --help
```

Expected for `extract-frames`: `--resume` and `--overwrite` present; `--skip-existing` absent.

- [ ] **Step 3: Reinstall completion**

```bash
python -m entomokit.main completion zsh --install
```

New flags are picked up automatically from argparse at generation time.

- [ ] **Step 4: Stop before committing**

Do not commit automatically. Report the changed files and test results; commit only when the user explicitly asks.

---

## Self-Review

**Spec coverage:**
- Shell path completion fix → Task 1 ✓
- Version bump 0.3.0→0.4.0 + `__commit__` variable → Task 2 ✓
- `entomokit update` with GitHub Commits API + latest commit message + `--check`/`--yes` → Task 3 ✓
- Guard policy tested standalone → Task 4 ✓
- `--resume`/`--overwrite` for segment → Task 5 ✓
- `--resume`/`--overwrite` for synthesize → Task 6 ✓
- `--resume`/`--overwrite` for augment → Task 7 ✓
- `--resume`/`--overwrite` for clean (guard only) → Task 8 ✓
- `--resume`/`--overwrite` for measure (combine existing rows + new rows, then rewrite CSVs) → Task 9 ✓
- `extract-frames --skip-existing` → `--resume`, add `--overwrite` → Task 10 ✓
- `--overwrite` for split-csv + classify train/predict/evaluate/embed/cam/export-onnx → Task 10b ✓
- README.md + README.cn.md + 3 skill files updated → Task 11 ✓
- Full suite + completion re-install → Task 12 ✓

**Placeholder scan:** None found.

**Type consistency:** `skip_existing: bool = False` added consistently to `process_directory()` in segmentation, synthesis, and `run_augment()` in augment. `existing_rows: list[dict[str, str]] | None = None` used in measurement. `check_output_dir(out_dir, resume, overwrite)` is shared from `src.common.cli` across all six CLI modules.
