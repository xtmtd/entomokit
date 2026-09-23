# EntomoKit v0.7.0 Recursive Layout and CLI Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release EntomoKit `0.7.0` with one consistent recursive directory-processing policy, mirrored image outputs, configurable cleaning padding, the remaining approved GitHub CLI fixes, versioned logs, and synchronized documentation.

**Architecture:** Add the smallest shared file-scanning/path-mapping helpers to `src/common/cli.py` (or a focused sibling module if the implementation would otherwise make that file domain-heavy). Directory-oriented commands consume the helper and pass each input's relative path through their existing processors; CSV-driven classification commands remain CSV-driven. Keep command adapters thin and preserve existing output manifests/annotation writers while changing only the path and parameter contracts required by this release.

**Tech Stack:** Python 3.9+, argparse, pathlib, Pillow, OpenCV, NumPy, PyTorch/AutoGluon, pytest, Markdown.

**Spec:** Approved decisions recorded in this plan and the existing architecture/design documents listed in Task 9. No separate design spec was requested; this plan is the implementation authority for the `0.7.0` change set.

## Global Constraints

- Release version is exactly `0.7.0` in `version.txt`, `entomokit/_version.py`, and the package fallback in `entomokit/main.py`.
- Directory-oriented image/video commands scan recursively by default and mirror the input-relative directory structure in output paths, except `segment`, which flattens samples into `images/` with encoded IDs and per-format annotation directories (standard VOC/COCO layouts are produced by a later conversion/split step, not by `segment`).
- `clean` removes both `--recursive` and `--flatten`; there is no flat-output mode for these directory-oriented commands.
- `measure` is included in the default recursive policy; CSV-driven commands are not changed into directory scans.
- `segment` recursively scans inputs but encodes each input-relative path into a unique dataset sample ID, so standard annotation consumers remain compatible.
- `clean --pad-color` choices are `none`, `median`, `black`, and `white`; default is `none`; every non-`none` choice pads to a square.
- Synthesis background selection remains with replacement; `synthesize --seed` defaults to `42`; independent per-task seeded randomness is required for both Python and NumPy random operations, but image-level duplicate-result detection is not.
- `synthesize --num-syntheses` uses positive integer-valued numbers for per-target synthesis count or values strictly between `0` and `1` for the fraction of targets to sample; fractional mode generates one synthesis per selected target.
- `classify embed` kNN behavior is already corrected in `0.6.2`; do not reimplement or regress it.
- Do not add a dependency for recursive scanning, padding, seed plumbing, logging metadata, or documentation.
- Source code, comments, tests, and English documentation use English; `README.cn.md` remains Chinese.
- Do not commit or execute implementation before the user separately approves plan execution.

## Review Focus

- Two same-named images in different input subdirectories must survive with distinct mirrored output paths; test this through at least `clean` and `augment`.
- A recursive video tree with same-named videos in different directories must produce distinct frame trees; test extraction path mapping and resume behavior.
- `clean` must preserve complete non-square content without padding by default, and each selected padding color must produce a square with the expected RGB values; test `none`, `median`, `black`, and `white`.
- Fractional synthesis counts must select a deterministic target subset, while integer counts allow unlimited with-replacement background draws; test invalid values and a seeded task plan.
- Existing callers that construct processor namespaces or call domain methods directly must receive an explicit migration path in tests; update stale `recursive`, `flatten`, and default-count fixtures rather than silently supporting two contracts.

---

## File Map

### Shared scanning and path layout

- Modify: `src/common/cli.py` or create `src/common/files.py` for `IMAGE_EXTENSIONS`, `VIDEO_EXTENSIONS`, recursive file iterators, and relative-output path helpers.
- Modify: `entomokit/extract_frames.py`, `src/framing/extractor.py`.
- Modify: `entomokit/segment.py`, `src/segmentation/processor.py` for recursive input scanning and standard dataset-safe sample IDs.
- Modify: `entomokit/measure.py`, `src/measurement/io.py`, `src/measurement/service.py`.
- Modify: `entomokit/clean.py`, `src/cleaning/processor.py`.
- Modify: `entomokit/augment.py`, `src/augment/service.py`.
- Modify: `entomokit/synthesize.py`, `src/synthesis/processor.py`.
- Modify: `entomokit/classify/embed.py`, `src/classification/embedder.py`.
- Modify: `entomokit/classify/cam.py`, `src/classification/cam.py`.
- Modify: `entomokit/classify/predict.py`, `src/classification/predictor.py`.

### Approved CLI fixes and metadata

- Modify: `entomokit/synthesize.py`, `src/synthesis/processor.py`.
- Modify: `entomokit/augment.py`, `src/augment/service.py`.
- Modify: `entomokit/classify/train.py`, `src/classification/trainer.py`.
- Modify: `src/common/cli.py`, `src/common/logging.py` if the legacy logging path is still used by an executable command.
- Modify: `version.txt`, `entomokit/_version.py`, `entomokit/main.py`, `setup.py` only if setup metadata contains a version copy.

### Tests and documentation

- Modify/create focused tests under `tests/` for every changed contract; update stale tests such as `tests/test_clean_recursive.py`, `tests/test_resume_flags.py`, and CLI parser tests.
- Modify: `README.md`, `README.cn.md`.
- Modify: `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` for global recursive/layout constraints.
- Modify: `docs/plans/2026-02-16-unified-scripts-architecture-design.md` and any still-current older design document that describes the affected command contracts; preserve historical implementation plans unless their text is explicitly a current design reference.
- Modify: `skills/entomokit-workflow/references/command-profiles.md`, `skills/entomokit-workflow/references/workflow.md`, `skills/entomokit-workflow/references/progress-memory.md`, and related schema/reference files after inspecting exact stale claims.

## Interfaces

The implementation should expose these minimum contracts; exact module placement may remain in the existing common utility module if tests and imports stay simple:

```python
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS: frozenset[str]
VIDEO_EXTENSIONS: frozenset[str]

def iter_files(root: Path, extensions: set[str]) -> list[Path]:
    """Return sorted files below root recursively, filtered case-insensitively."""

def relative_output_path(source: Path, input_root: Path, output_root: Path) -> Path:
    """Map source to output_root/source.relative_to(input_root)."""
```

The helper must not silently follow symlinked directories unless the existing command explicitly did so. It must reject a source outside `input_root` when computing a relative output path. Commands that need a file-level output stem must retain the relative parent directory rather than flattening it.

---

### Task 1: Define shared recursive scanners and path mapping

**Files:**
- Create or modify: `src/common/files.py` or `src/common/cli.py`.
- Test: `tests/test_common_files.py`.

**Interfaces:**
- Consumes: `Path` roots and extension sets.
- Produces: deterministic recursive file lists and safe input-relative output paths for later tasks.

- [ ] **Step 1: Write focused failing tests**

```python
from pathlib import Path

from src.common.files import iter_files, relative_output_path


def test_iter_files_is_recursive_sorted_and_case_insensitive(tmp_path: Path):
    root = tmp_path / "input"
    (root / "z").mkdir(parents=True)
    (root / "a").mkdir()
    (root / "z" / "b.JPG").write_bytes(b"")
    (root / "a" / "c.png").write_bytes(b"")
    (root / "ignore.txt").write_bytes(b"")
    assert [p.relative_to(root).as_posix() for p in iter_files(root, {".jpg", ".png"})] == [
        "a/c.png",
        "z/b.JPG",
    ]


def test_relative_output_path_preserves_parent_directories(tmp_path: Path):
    root = tmp_path / "input"
    source = root / "beetles" / "same.jpg"
    assert relative_output_path(source, root, tmp_path / "output") == (
        tmp_path / "output" / "beetles" / "same.jpg"
    )
```

Also test that a root file, a missing root, and a path outside the root fail with the project’s normal exception type/message rather than producing an unsafe output path.

- [ ] **Step 2: Run the focused tests and confirm the helper contract is absent**

Run: `pytest tests/test_common_files.py -q`
Expected: FAIL because the shared module/functions do not yet exist.

- [ ] **Step 3: Implement the minimal helpers**

Use `Path.rglob("*")`, filter `is_file()` and `suffix.lower()`, sort by relative POSIX path, and compute output paths with `source.relative_to(input_root)`. Keep extension constants in one place and have existing modules import them instead of defining slightly different sets.

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/test_common_files.py -q`
Expected: PASS.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

### Task 2: Apply recursive input and mirrored output policy to image/video commands

**Files:**
- Modify: `entomokit/extract_frames.py`, `src/framing/extractor.py`.
- Modify: `entomokit/segment.py`, `src/segmentation/processor.py` for recursive input scanning and standard dataset-safe sample IDs.
- Modify: `entomokit/measure.py`, `src/measurement/io.py`, `src/measurement/service.py`.
- Modify: `entomokit/clean.py`, `src/cleaning/processor.py`.
- Modify: `entomokit/augment.py`, `src/augment/service.py`.
- Create/modify tests: `tests/test_extract_frames_recursive.py`, `tests/test_segmentation.py`, `tests/test_measure_cli.py`, `tests/test_measure_metrics.py`, `tests/test_clean_recursive.py`, `tests/test_doctor_augment_cli.py`.

**Interfaces:**
- Consumes: Task 1 scanners and relative path helper.
- Produces: directory-oriented commands recurse without a command-specific recursive flag; ordinary image outputs mirror source-relative parents, while `segment` flattens images into `images/` with unique encoded sample IDs and per-format annotation directories.

- [ ] **Step 1: Add failing path-preservation tests before touching command code**

Cover at least these assertions:

```python
# clean and augment: input/a/same.jpg and input/b/same.jpg both produce outputs
# under images/a/ and images/b/, with no overwrite.
# segment: nested duplicate basenames receive unique encoded sample IDs while
# images stay under images/ and annotations go to the per-format directories.
# measure: nested masks are both included and their CSV image/mask identifiers remain unique.
# extract-frames: nested/beetles/movie.mp4 writes frames under frames/nested/beetles/movie/.
```

Use tiny generated images and mocked video extraction where OpenCV/video fixtures would add unrelated setup. Add parser assertions that `clean` no longer accepts `--recursive` or `--flatten`.

- [ ] **Step 2: Run the focused tests and record current failures**

Run: `pytest tests/test_clean_recursive.py tests/test_doctor_augment_cli.py tests/test_extract_frames_recursive.py tests/test_measure_cli.py tests/test_measure_metrics.py -q`
Expected: failures for flat output, top-level-only scans, stale clean flags, or missing nested path propagation.

- [ ] **Step 3: Change the command processors to consume relative paths**

For each ordinary file processor, replace local `iterdir()`/flat globbing with `iter_files()`. Pass either the full relative output path or the relative parent into existing writers. Preserve existing output subdirectories such as `cleaned_images`, `images`, `masks`, annotation directories, and `predictions`; only add the input-relative subdirectory below the existing output root. Resume checks must check the mapped destination for that source, not a flat stem glob.

For `segment`, update both existing `skip_existing` checks in `src/segmentation/processor.py` (the sequential and CPU-parallel paths) so they use the encoded sample ID from the input-relative path. Add a regression test with `input/beetles/a.jpg` and `input/moths/a.jpg`: after processing the first output, resuming must process or recognize the second independently rather than skipping it because `glob(f"a*")` found the first file.

For `segment`, recurse with the shared scanner and keep the flat `images/` plus per-format annotation directories (see the global policy in Task 9); standard VOC/COCO layouts are produced by a later conversion/split step. Build a filesystem-safe sample ID from a readable cleaned stem plus a stable digest of the input-relative POSIX path, for example `a__<sha256(relative_path)[:12]>`; do not rely only on replacing path separators, because source names may contain the chosen delimiter. Use that ID consistently for image files, VOC XML, YOLO TXT, segmentation masks, COCO file names where applicable, and `ImageSets/Main/default.txt`. Resume/`skip_existing` checks must derive this same encoded sample ID from the source-relative path and test the mapped output artifact; they must not use `glob(f"{img_path.stem}*")` or any other basename-only check. Add tests for two nested `a.jpg` inputs and delimiter-containing names proving there is no overwrite, the second nested `a.jpg` is not falsely skipped on resume, and standard consumers can resolve every annotation.

For `extract-frames`, keep single-file input support. For directory input, recursively enumerate videos and pass each video’s relative parent to output creation. Same-named videos in distinct directories must not share an output directory. Add `tests/test_extract_frames_recursive.py` with mocked `VideoFrameExtractor.extract_from_video()` assertions for both nested paths and single-file input.

For `measure`, recurse through masks while retaining the relative path in CSV identifiers and resume keys. Add nested-mask coverage to `tests/test_measure_cli.py` or `tests/test_measure_metrics.py`. Do not turn CSV-driven classification or `split-csv` into directory scans in this task.

- [ ] **Step 4: Remove clean recursive/flatten API and implement fixed recursion**

Delete `--recursive` and `--flatten` from `entomokit/clean.py`. Remove `recursive` and `flatten` from `ImageCleaner.process_directory()` and its call sites. `ImageCleaner` always calls the shared recursive scanner and always derives `dst_dir` from `src.parent.relative_to(input_dir)`.

- [ ] **Step 5: Run focused tests and the CLI schema/help tests**

Run:

```bash
pytest tests/test_clean_recursive.py tests/test_doctor_augment_cli.py tests/test_extract_frames_recursive.py tests/test_measure_cli.py tests/test_measure_metrics.py tests/test_cli_help_texts.py tests/test_cli_schema.py -q
```

Expected: PASS after stale test fixtures are updated to the new contract.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

### Task 3: Add `clean --pad-color`

**Files:**
- Modify: `entomokit/clean.py`, `src/cleaning/processor.py`.
- Test: `tests/test_clean_padding.py`, `tests/test_clean_recursive.py`.

**Interfaces:**
- Consumes: fixed recursive `ImageCleaner` contract from Task 2.
- Produces: `pad_color: Literal["none", "median", "black", "white"]` behavior in the cleaner and CLI.

- [ ] **Step 1: Write failing padding tests**

Create a non-square RGB image with distinct edge colors and assert:

```python
# pad_color="none" keeps the resized image dimensions unchanged.
# pad_color="black" and "white" produce square images with exact corner colors.
# pad_color="median" produces a square image whose added corner equals the
# median RGB value of the source border pixels.
# out_short_size=-1 preserves the original long edge before padding.
```

Add a transparent PNG case that verifies the cleaner converts it deterministically before saving JPEG, and a parser test that the default is `none` and choices are exactly the four approved values.

- [ ] **Step 2: Run the padding tests and confirm failure**

Run: `pytest tests/test_clean_padding.py -q`
Expected: FAIL because the CLI option and processor argument do not exist.

- [ ] **Step 3: Implement padding as one small Pillow helper**

Add a helper with a direct contract such as:

```python
def pad_to_square(image: Image.Image, color: str) -> Image.Image:
    """Return image unchanged for none, otherwise center it on a square canvas."""
```

For `median`, compute the per-channel median over pixels on the four outermost rows/columns. For `black` and `white`, use `(0, 0, 0)` and `(255, 255, 255)`. Apply resize first, then padding. Keep alpha handling explicit: composite RGBA input onto the selected fill color before padding when the output path requires RGB/JPEG; do not let Pillow silently discard alpha.

- [ ] **Step 4: Wire the argument and run tests**

Add `--pad-color`, default `none`, pass it to `ImageCleaner`, and run:

```bash
pytest tests/test_clean_padding.py tests/test_clean_recursive.py -q
```

Expected: PASS.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

### Task 4: Fix synthesis count semantics and seeded task planning

**Files:**
- Modify: `entomokit/synthesize.py`, `src/synthesis/processor.py`.
- Create: `tests/test_synthesis.py`, `tests/test_synthesize_cli.py`.
- Modify: `tests/test_resume_flags.py`.

**Interfaces:**
- Consumes: recursive target/background lists and target-relative output mapping from Task 2.
- Produces: `num_syntheses: int | float`, default `1`, where positive integer-valued numbers mean per-target synthesis count and values strictly between `0` and `1` mean the fraction of targets to sample, with one synthesis per selected target; also exposes `--seed INTEGER` with default `42` as the base seed for target sampling and task-local synthesis randomness.

- [ ] **Step 1: Add failing validation/count tests**

Test the parser default is `1`, invalid values `<= 0` and non-integral values above `1` are rejected, integer-valued `3` schedules three tasks per target without a background-count cap, and ratio `0.5` selects half of the targets with one task per selected target. Test that `1` and `1.0` both mean one synthesis per target, the selected target subset is deterministic for a fixed seed, and each task may reuse a background selected by another task while task-local random parameters remain independent.

- [ ] **Step 2: Run synthesis tests and confirm current behavior fails**

Run: `pytest tests/test_synthesis.py tests/test_synthesize_cli.py -q`
Expected: failures for default count, fractional parsing, count caps, nested target output paths, or global RNG behavior.

- [ ] **Step 3: Implement count resolution and task-local RNG**

Add a small pure resolver, for example:

```python
def resolve_num_syntheses(value: int | float, target_count: int) -> tuple[list[int], int]:
    """Return selected target indices and syntheses per selected target."""
```

For a positive integer-valued input, select all target indices and return that per-target count without comparing it to background count. For a fractional value in `(0, 1)`, sample `floor(target_count * value)` target indices without replacement, with a positive ratio selecting at least one target when targets exist, and return one synthesis per selected target. Add `--seed`, default `42`, and keep background sampling with replacement using a deterministic `random.Random(task_seed)` plus a task-local `numpy.random.Generator` per synthesis task. Derive target sampling and task seeds from the user seed plus stable target index and synthesis index; pass the chosen background and random parameters in the task tuple so multiprocessing does not consume shared global RNG state. Refactor existing synthesis helpers that currently call module-level `random.*` or `np.random.*` to accept the task-local RNG (or receive already-generated values), including position, scale, rotation, and black-region avoidance choices.

Preserve the existing synthesis random operations using only the task-local generators. Do not add output-image hash deduplication.

- [ ] **Step 4: Wire CLI and mirrored target paths**

Add `--seed INTEGER` with default `42`. Parse `--num-syntheses` numerically with these exact rules: values strictly between `0` and `1` mean the fraction of targets to select, with one synthesis per selected target; positive integer-valued numbers mean synthesis attempts per target; non-integral values above `1`, non-positive values, and non-finite values are invalid. Thus `1` and `1.0` both mean one synthesis per target. Document that background sampling is with replacement and has no background-count cap. Ensure output names remain unique under target-relative directories and annotation paths use the same mapped image path.

- [ ] **Step 5: Run synthesis tests and resume tests**

Run: `pytest tests/test_synthesis.py tests/test_synthesize_cli.py tests/test_resume_flags.py -q`
Expected: PASS.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

### Task 5: Fix augment naming and classification directory inputs

**Files:**
- Modify: `src/augment/service.py`.
- Modify: `entomokit/classify/embed.py`, `src/classification/embedder.py`.
- Modify: `entomokit/classify/cam.py`, `src/classification/cam.py`.
- Modify: `entomokit/classify/predict.py`, `src/classification/predictor.py`.
- Test: `tests/test_doctor_augment_cli.py`, `tests/test_classify_embed_cli.py`, `tests/test_classification_cam.py`, predictor tests.

**Interfaces:**
- Consumes: Task 1 scanner/path mapping.
- Produces: unique augmented names for `multiply=1`, relative image identifiers in classification outputs, mirrored CAM figures, and recursive directory input behavior.

- [ ] **Step 1: Add failing tests**

Assert `augment --multiply 1` creates `source_aug1.ext`, never overwrites/copies to the original basename, and nested files are written under matching nested output folders. Assert embed CSV image values are relative paths such as `beetles/a.jpg`, not basename-only values. Assert CAM figure paths mirror nested inputs. Assert predict directory discovery includes nested images and preserves their relative names in the prediction result.

- [ ] **Step 2: Implement the minimum path/name changes**

Use the source-relative path for output directories and manifest entries. Always use an augmentation suffix (`_aug01` or the existing width convention) regardless of multiply value. Keep CSV-driven predict behavior unchanged when `--input-csv` supplies explicit paths; only directory discovery becomes recursive.

- [ ] **Step 3: Run focused classification/augmentation tests**

Run:

```bash
pytest tests/test_doctor_augment_cli.py tests/test_classify_embed_cli.py tests/test_classification_cam.py tests/test_classification_predictor_onnx.py -q
```

Expected: PASS.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

### Task 6: Expose `classify train --seed`

**Files:**
- Modify: `entomokit/classify/train.py`, `src/classification/trainer.py`.
- Test: `tests/test_classify_trainer.py`, `tests/test_resume_flags.py`; create `tests/test_classify_train_cli.py` only if parser forwarding is not already covered by an existing test module.

**Interfaces:**
- Consumes: existing AutoMM hyperparameter construction.
- Produces: `--seed INTEGER`, default `0`, passed to AutoMM as the supported seed hyperparameter and recorded in the run log.

- [ ] **Step 1: Add failing parser and forwarding tests**

Parse `--seed 123` and assert `train(..., seed=123)` receives it. Mock `MultiModalPredictor` and assert the generated hyperparameters include the AutoMM seed key expected by the installed AutoGluon API. Assert the default remains `0` for reproducibility.

- [ ] **Step 2: Implement and verify**

Add the CLI option, add `seed: int` to the trainer signature, and include it in the AutoMM hyperparameters without changing unrelated defaults. Run:

```bash
pytest tests/test_classify_trainer.py tests/test_resume_flags.py -q
```

Expected: PASS.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

### Task 7: Remove stale CAM worker option and add version/commit to logs

**Files:**
- Modify: `entomokit/classify/cam.py`, `src/classification/cam.py` only if a stale worker parameter exists after current-tree verification.
- Modify: `src/common/cli.py`, `src/common/logging.py` as needed by active executable paths.
- Modify: `tests/test_cli_output_logging.py` and a focused CAM parser test.

**Interfaces:**
- Consumes: existing `save_log()` and version metadata.
- Produces: no accepted-but-ignored CAM `--num-workers` option; every command log header contains version and commit fields, with commit probing cached once per process.

- [ ] **Step 1: Add failing log metadata test**

Extend the existing `save_log` test to assert that the header contains a non-empty `EntomoKit version:` field and a `Commit:` field whose value is either a non-empty commit id or `unknown`. Patch the Git lookup, call `save_log()` twice in one process, and assert the subprocess lookup runs at most once. The exact `0.7.0` assertion belongs to Task 8 after the version sources are updated.

Add a parser test asserting `classify cam --help` does not expose `--num-workers` and that `--num-threads` remains available and is passed to `set_num_threads`. Current-tree verification already shows CAM exposes only `--num-threads`, so this is a regression assertion and documentation cleanup, not a planned CAM implementation change.

- [ ] **Step 2: Implement one shared metadata path**

Have `save_log()` import `__version__`/`__commit__` from `entomokit._version`, and use a short Git lookup only when commit metadata is `unknown` and Git is available. Cache the resolved commit at module scope so repeated `save_log()` calls in one process do not spawn repeated Git subprocesses. Never fail a command because Git metadata cannot be read; write `unknown`. Put the fields before command arguments so log headers are easy to scan. Keep `src/common/logging.py` synchronized only if its legacy `save_command_log()` is still used by an active entry point.

- [ ] **Step 3: Run logging and CLI tests**

Run: `pytest tests/test_cli_output_logging.py tests/test_cli_help_texts.py tests/test_main_cli.py -q`
Expected: PASS.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

### Task 8: Set version `0.7.0` and complete regression coverage

**Files:**
- Modify: `version.txt`, `entomokit/_version.py`, `entomokit/main.py`, `setup.py` if needed.
- Modify: package/version tests and all stale default-argument fixtures.
- Test: full affected suite and then full `pytest`.

**Interfaces:**
- Consumes: all prior task contracts.
- Produces: consistent runtime/package version and a regression suite for the breaking CLI changes.

- [ ] **Step 1: Add failing version assertions**

Assert `entomokit --version`, `entomokit._version.__version__`, `version.txt`, and package metadata/fallback all resolve to `0.7.0`.

- [ ] **Step 2: Update version and stale fixtures**

Change only the version sources and fixtures that encode removed arguments or old defaults. Do not rewrite historical release notes or old implementation plans merely to make them describe the new behavior; those are historical records unless Task 9 identifies a section explicitly marked as current design.

- [ ] **Step 3: Run the full verification set**

Run:

```bash
pytest tests/test_common_files.py tests/test_clean_padding.py tests/test_clean_recursive.py tests/test_synthesis.py tests/test_synthesize_cli.py tests/test_doctor_augment_cli.py tests/test_classify_trainer.py tests/test_classification_cam.py tests/test_classify_embed_cli.py tests/test_cli_output_logging.py tests/test_cli_help_texts.py tests/test_cli_schema.py tests/test_resume_flags.py -q
pytest -q
```

Expected: both commands pass. If the full suite exposes unrelated pre-existing failures, record exact failures and do not mark this task complete until changed-path failures are resolved.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

### Task 9: Update design, README, and workflow skill contracts

**Files:**
- Modify: `README.md`, `README.cn.md`.
- Modify: `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`.
- Modify: `docs/plans/2026-02-16-unified-scripts-architecture-design.md` and only other older design documents that still function as current design references.
- Modify: `skills/entomokit-workflow/references/command-profiles.md`, `skills/entomokit-workflow/references/workflow.md`, `skills/entomokit-workflow/references/progress-memory.md`, and generated/validated CLI schema references if tests identify them.
- Test: `tests/test_cli_help_texts.py`, `tests/test_cli_schema.py`, `tests/test_param_guard.py`, `tests/test_workflow_gate.py`, skill-specific tests.

**Interfaces:**
- Consumes: final argparse help/schema and behavior from Tasks 1-8.
- Produces: complete parameter documentation and a single documented global directory-processing policy.

- [ ] **Step 1: Inventory documentation drift**

Search for and classify every occurrence of `--recursive`, `--flatten`, old synthesize default `10`, old augment multiply naming, missing `--seed`, CAM `--num-workers`, old clean defaults, and version/log claims. Do not mechanically replace historical changelog entries; update current usage/design sections and clearly label historical records.

- [ ] **Step 2: Update the global design contract**

In the current refactor/design document add a dedicated “Global directory input and output policy” section stating:

```text
Directory-oriented commands recursively scan by default.
Ordinary file outputs mirror each input file's path relative to the input root.
segment is the exception: it flattens images into images/ with per-format
annotation directories and encodes input-relative paths as unique flat sample
IDs (standard VOC/COCO layouts come from a later conversion/split step).
No flatten option is provided for ordinary directory-processing commands.
CSV-driven commands remain CSV-driven and are outside this directory policy.
clean --pad-color none|median|black|white defaults to none.
```

Update the old `clean --recursive` table entry and any command-specific contradictions. Include `extract-frames` and `measure` explicitly because they were added to the global policy during this design discussion. Document `segment` as the one command that flattens images into `images/` instead of mirroring input paths, with per-format annotation directories and relative-path-to-sample-ID encoding.

- [ ] **Step 3: Document every changed parameter and output contract**

Update both READMEs with complete command parameter tables for the affected commands, including `clean --pad-color`, synthesis `--seed` and `--num-syntheses` target-sampling syntax, `classify train --seed`, `--num-threads`, recursive defaults, ordinary mirrored output examples, the `segment` standard-layout exception, and log version/commit headers. State that kNN corrections are already part of `0.6.2` and are not a new `0.7.0` algorithm change.

- [ ] **Step 4: Update workflow skill guidance**

Remove instructions that ask users to choose `--recursive` for `clean`. Change workflow output guidance to expect mirrored paths for ordinary processors and encoded flat sample IDs for `segment`. Add `--pad-color` selection guidance, explain when `median`/black/white is appropriate, document fractional target sampling and with-replacement background sampling for synthesis, and add `classify train --seed`. Keep operational safeguards such as run-root output isolation and resume/overwrite approval.

- [ ] **Step 5: Run documentation/schema/skill checks**

Run:

```bash
pytest tests/test_cli_help_texts.py tests/test_cli_schema.py tests/test_param_guard.py tests/test_workflow_gate.py -q
rg -n -- '--recursive|--flatten|num-syntheses|pad-color|--seed|num-workers|num-threads' README.md README.cn.md docs/superpowers/specs skills/entomokit-workflow
```

Expected: no current-contract reference recommends removed flags; all changed options appear in README and skill references; historical notes are explicitly historical where retained.

完成本任务后保留工作区变更，不提交；最终统一由用户批准后提交。

---

## Verification and Release Gate

After all tasks, run the complete suite again from a clean working tree:

```bash
pytest -q
entomokit --version
entomokit clean --help
entomokit synthesize --help
entomokit classify train --help
entomokit classify cam --help
```

Manually confirm:

- `clean --help` has no `--recursive` or `--flatten` and includes `--pad-color`.
- `synthesize --help` documents integer per-target counts and fractional target sampling, and shows default `1`.
- `classify train --help` includes `--seed` default `0`.
- `classify cam --help` includes `--num-threads` and does not include an ineffective `--num-workers`.
- A log begins with `EntomoKit version: 0.7.0` and a commit value.
- Nested input files with duplicate basenames remain separate in ordinary mirrored output directories, and `segment` encodes them as unique IDs in a flat `images/` tree with per-format annotation directories.

No implementation task is authorized until the user approves execution of this plan. No commit is authorized until the user separately approves the completed implementation and release verification.

---

## Addendum: post-review diagnostics wording (2026-09-22)

Scope added after the first end-to-end `v0.7.0` smoke runs. Wording and diagnostic
artifacts only; no algorithm, layout, or parameter-semantics changes.

- `segment`: replace the misleading `No masks passed confidence threshold (0.0)`
  warning with a method-aware message that names the full source path. The
  threshold is only quoted when it is greater than `0`
  (`No masks returned by <method>: <source path>` otherwise).
- `segment`: write every zero-mask input to `<out-dir>/no_mask_images.txt`
  (sorted, one path per line) and log a count plus that file path, so the
  affected images can be audited later without grepping `log.txt`.
- `synthesize`: when no synthesis task can be created, report the number of
  selected targets that failed and the observed PIL modes, e.g.
  `9 of 9 selected target image(s) failed to load (observed modes: 9 RGB)`,
  and point at mask-mode `segment` output as the RGBA source. Always log a
  per-mode summary when any target fails to load. This also fixes a stale
  message that reported the total target count while only the selected subset
  (for example the `0.5` fraction) had been attempted.
- Docs/skills: state explicitly that `synthesize --target-dir` requires RGBA
  cutouts, and that `*-bbox` crop output, `repaired_images`, and raw photos are
  RGB and rejected.

**Files:** `src/segmentation/processor.py`, `src/synthesis/processor.py`,
`README.md`, `README.cn.md`, `skills/entomokit-workflow/references/workflow.md`,
`skills/entomokit-workflow/references/command-profiles.md`.

**Tests:** `tests/test_segmentation.py::test_process_directory_records_images_with_no_masks`,
`tests/test_synthesis.py::test_all_rgb_targets_error_lists_observed_modes`.

---

## Addendum: audit follow-up (2026-09-22)

Second review pass. All five findings were reproduced before fixing; no
algorithm semantics changed.

1. **extract-frames single-file matching was name-based** (high).
   `VideoFrameExtractor.get_video_files()` compared `f.name` against a stored
   file name, so a same-named video elsewhere under the input root was also
   processed. The CLI now stores the full `Path` and the extractor compares
   resolved paths. Covered by
   `tests/test_extract_frames_recursive.py::test_single_file_filter_matches_only_the_exact_path`.

2. **`segment`/`synthesize --annotation-format voc` image directory** (withdrawn).
   An earlier pass moved VOC images from `images/` to `JPEGImages/` to match the
   design doc, the phase-2 annotation plan, and `src/common/annotation_writer.py`.
   That premise was wrong: the standard VOC/COCO layout is produced by a later
   conversion step (for example `split-csv`), not by `segment`/`synthesize`
   directly. entomokit's intermediate output keeps `images/` for every
   annotation format, so both changes were reverted. The discrepancy was in the
   docs, not the implementation, and was corrected on 2026-09-22:
   `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` now states
   that `segment`/`synthesize` write images under `images/` and that the
   standard VOC `JPEGImages/` layout is produced by a later conversion/split
   step (for example `split-csv`); `README.md`, `README.cn.md`, and the
   workflow skill references were updated to match, and a correction note was
   added to `docs/superpowers/plans/2026-03-24-phase2-annotation-format.md`.

3. **`synthesize --resume` matched too broadly** (medium). The old
   `glob(f"{stem}_*")` treated unrelated files such as `a_backup.png` as
   completion, and skipped a partially finished target entirely. Resume now
   checks the exact set of expected artifacts
   (`{stem}_{i:02d}.{format}` for `i in 1..num_syntheses`) and only skips when
   all of them exist. Covered by
   `tests/test_synthesis.py::test_resume_ignores_unrelated_same_prefix_files`,
   `::test_resume_does_not_skip_partially_completed_target`, and
   `::test_resume_skips_fully_completed_target`.

4. **`synthesize_single()` still consumed the global Python RNG** (medium).
   The direct-call fallbacks for `scale_ratio` and rotation angle used
   module-level `random.uniform()`. `synthesize_single()` now derives a
   task-local `random.Random(np_seed)` and `_rotate_image()` accepts that RNG,
   so no synthesis random operation reads global state.

5. **Stale wording and whitespace** (low). README/README.cn `clean` feature
   bullet now says "always recursive"; the 0.7.0 design-doc header line no
   longer carries trailing whitespace (`git diff --check` is clean).

---

## Addendum: audit follow-up 2 (2026-09-22)

Third review pass. All three findings were reproduced and fixed.

1. **`segment --resume` still used a prefix glob** (high). `_existing_output`
   matched `glob(f"{sample_id}*")`, so a file such as
   `{sample_id}_backup.png` counted as completion. New
   `collect_segment_artifacts()` scans `images/` once and records the exact
   single-mask stems; the resume check is a set lookup and unrelated prefixed
   files no longer match. (Follow-up 4 removed the multi-mask branch entirely -
   see below.) Covered by
   `tests/test_segmentation.py::test_resume_only_skips_exact_single_mask_artifact`.

2. **Synthesis overwrote same-stem targets with different extensions**
   (medium). `_get_target_filename()` used `Path.stem`, so `a.png` and `a.tif`
   in one directory both produced `a_01.png`. New `resolve_target_stems()`
   assigns each target a stable stem, appending a short path digest only when
   several targets share a directory *and* a stem; the resolved stem is threaded
   through the task tuple and used by both the writer and the `--resume`
   expected-artifact check. Covered by
   `tests/test_synthesis.py::test_same_stem_different_extension_does_not_overwrite`
   and `::test_same_stem_collision_resume_uses_resolved_names`.

3. **Synthesis `failed` could be negative after a full resume** (medium).
   `skipped_images` counted skipped *targets* but was subtracted from a
   *synthesis* total. The counter now tracks skipped syntheses
   (`skipped += per_target`) and failed syntheses are left to the
   `total - processed - skipped` formula, so a full resume reports
   `{"processed": 0, "failed": 0, "skipped": 3}` instead of a negative/incorrect
   `failed`. The `skipped` field returned by `process_directory` and printed by
   the CLI is now a synthesis count, consistent with `processed`/`failed`.

---

## Addendum: audit follow-up 3 (2026-09-22)

Fourth review pass. Two boundary issues in the previous fixes; both fixed.

1. **`segment --resume` accepted any digit suffix** (medium). Follow-up 2
   tightened the check to `{sample_id}_<digits>`, but that still accepted files
   such as `{sample_id}_2021.png`. This was superseded by follow-up 4, which
   removed the multi-mask branch entirely: only the exact single-mask artifact
   auto-skips, and any `{sample_id}_...` file is ignored for resume purposes.
   Covered by
   `tests/test_segmentation.py::test_resume_does_not_treat_unrelated_prefixed_files_as_completion`
   (`_backup`, `_2021`, `_1`, `_100` do not skip) and
   `::test_resume_only_skips_exact_single_mask_artifact`.

2. **Synthesis `failed` mixed in uncreated work** (low). The previous formula
   `total - processed - skipped` counted planned-but-never-created tasks
   (unreadable targets, failed background loads, shutdown during task
   construction) as failures. `process_directory()` now returns
   `failed = len(created_tasks) - processed` plus a separate `uncreated` count,
   so `failed` means "a synthesis task ran and produced no output" and aborted
   work is reported separately; the CLI prints a `Not attempted:` line when
   `uncreated > 0`. Covered by
   `tests/test_synthesis.py::test_unreadable_target_counts_as_uncreated_not_failed`
   and `::test_failed_background_counts_as_uncreated`.

---

## Addendum: audit follow-up 4 (2026-09-22)

Fifth review pass. One data-integrity issue plus stale plan wording.

1. **`segment --resume` could treat a partial multi-mask set as complete**
   (medium). Any single `{sample_id}_NN.{ext}` marked the sample as done, so an
   interrupted multi-mask input (`_01` written, `_02`/`_03` missing) was
   skipped. `collect_segment_artifacts()` now returns only exact single-mask
   stems and the resume check is `sample_id in artifacts`; multi-mask inputs are
   always re-processed because no mask count is recorded to verify a complete
   set. This deliberately trades re-running multi-mask inputs for never skipping
   partial data. Covered by
   `tests/test_segmentation.py::test_resume_only_skips_exact_single_mask_artifact`
   (partial `_01` and complete-looking `_01`+`_02` both re-run; only
   `{sample_id}.{ext}` auto-skips).

   A completion manifest (`.entomokit/segment_completed.txt` with sample ID,
   output format/extension and mask count) was considered and deliberately not
   added: the recommended fix for the data-integrity risk is the simple rule
   above, and the manifest is only worth adding if reliable incremental resume
   for multi-mask inputs is actually needed.

2. **Stale plan wording** (docs). Task 2 Step 3 still said "keep standard
   dataset output directories" and Task 9 still called `segment` a
   "standard-dataset-layout exception"; both now match the corrected policy
   (flat `images/` + per-format annotation dirs, standard layouts produced by a
   later step). `README.md`, `README.cn.md`, and the `segment --resume` CLI help
   now state that only an exact single-mask output auto-skips.

---

## Addendum: audit follow-up 5 (2026-09-22)

Sixth review pass. One high and one medium data-loss bug on `--resume`; both
fixed with a shared COCO merge.

1. **`segment --resume` erased an existing unified COCO file** (high). A pure
   resume skipped every input, so the per-run `COCOMetadataManager` was empty,
   yet `process_directory` still overwrote `annotations.coco.json` with zero
   images/annotations. The file is now snapshotted before the run and merged
   after it, so skipped samples keep their annotations. Covered by
   `tests/test_segmentation.py::test_resume_preserves_existing_unified_coco_annotations`
   (1 image/1 annotation before resume, unchanged after) and
   `::test_resume_merges_new_segment_samples_into_existing_coco`.

2. **`synthesize --resume` rebuilt the unified COCO file from this run only**
   (medium). `write_annotations()` always overwrote `annotations.coco.json`
   from `_ann_image_paths` (this run's images only), dropping annotations for
   skipped targets. The same snapshot-and-merge now applies. Covered by
   `tests/test_synthesis.py::test_resume_preserves_skipped_coco_annotations`
   (pure resume leaves the file untouched; adding a target yields
   `['a_01.png', 'c_01.png']` with two annotations).

**Implementation.** `src/common/annotation_writer.py` gains `load_coco_json()`
and `merge_coco_json()` (plus `_merge_coco_payloads()`): categories are unioned
by name, images by `file_name` (a re-processed sample replaces its previous
entry instead of duplicating it), and annotations are re-keyed with fresh ids.
The merge runs after any `--coco-bbox-format xyxy` rewrite; the file now records
its convention in `info.bbox_format`, and changing the format on `--resume` is
rejected instead of silently mixing conventions (see follow-up 6). Per-image
VOC/YOLO and COCO-separate outputs are already resume-safe and unchanged.

---

## Addendum: audit follow-up 6 (2026-09-22)

Seventh review pass. Two findings were reproduced and fixed; one half of the
first finding did not reproduce and was not "fixed".

1. **Stale multi-mask crops on re-run** (high, partially confirmed).
   `_remove_stale_crops()` now deletes a sample's previous
   `{sample_id}_{NN}.{ext}` crops (and the exact single-mask name) from
   `images/` before its new crops are written, in `_write_computation()`. This
   is required because follow-up 4 made multi-mask inputs always re-run. The
   audit's companion claim that unified COCO also kept a stale `_03` annotation
   did **not** reproduce: unified COCO stores one image entry per sample keyed
   by `file_name`, and `merge_coco_json()` drops the replaced sample's old
   annotations, so a 3-mask run followed by a 2-mask run leaves exactly two
   annotations. The separate-mode JSON is rewritten per sample as well. No COCO
   change was made. Covered by
   `tests/test_segmentation.py::test_resume_multi_mask_rerun_removes_orphan_crops`.

2. **`--resume` accepted a `--coco-bbox-format` switch** (medium, confirmed).
   A 3-mask `xywh` run followed by an `xyxy` resume on a new sample produced
   `a: [20,20,60,40]` (xywh) next to `b: [20,20,80,60]` (xyxy) in one file.
   `record_coco_bbox_format()` now stamps the convention into `info.bbox_format`
   after any rewrite, and `process_directory()` rejects a resume whose prior
   file recorded a different format before any output is touched. Files without
   the key default to `xywh` (the historical default). Covered by
   `tests/test_segmentation.py::test_resume_rejects_coco_bbox_format_switch`.

3. **VOC `ImageSets/Main/default.txt` duplicated rows on re-run** (medium,
   confirmed). The append now reads the existing stems and skips a sample that
   is already listed. Covered by
   `tests/test_segmentation.py::test_resume_multi_mask_rerun_does_not_duplicate_voc_imageset`.

---

## Addendum: audit follow-up 7 (2026-09-22)

Eighth review pass. The follow-up 6 bbox-format guard only covered `segment`.

1. **`synthesize --resume` could still mix COCO bbox formats** (medium,
   confirmed). `write_annotations()` now records the convention via
   `_save_coco()` → `record_coco_bbox_format()`, so any COCO file it writes
   carries `info.bbox_format`. `SynthesisProcessor.process_directory()` reads
   and validates the prior format when `skip_existing` is set and raises before
   loading targets/backgrounds or creating any synthesis, leaving the old JSON
   untouched. Covered by
   `tests/test_synthesis.py::test_synthesize_resume_rejects_coco_bbox_format_switch`.
   `segment` keeps its explicit record call because it writes unified COCO via
   `COCOMetadataManager.save()`, not `write_annotations()`.

---

## Addendum: audit follow-up 8 (2026-09-22)

Ninth review pass. One boundary case left by follow-up 6.

1. **Zero-mask re-run kept stale segment artifacts** (medium, confirmed). The
   follow-up 6 cleanup ran only when the sample produced at least one mask, so a
   2-mask sample that segmented to 0 masks on resume left its crops in
   `images/` while `no_mask_images.txt` and the run counters said it produced
   nothing. Cleanup now runs immediately after `_compute_image()` succeeds and
   before the zero-mask early return, so an exception still preserves the prior
   output. On zero masks, `_remove_sample_artifacts()` also drops the sample's
   VOC XML / YOLO txt / COCO-separate JSON, `SegmentationClass` mask,
   `repaired_images` file, and `ImageSets/Main/default.txt` row, and
   `drop_coco_images()` removes its previous unified-COCO image and annotations
   during the resume merge. Covered by
   `tests/test_segmentation.py::test_resume_zero_mask_rerun_clears_old_crops_and_coco`
   and `::test_resume_zero_mask_rerun_removes_voc_sample`.

---

## Addendum: audit follow-up 9 (2026-09-22)

Tenth review pass. Two resume-hygiene issues; both confirmed and fixed. The
side-file cleanup added in follow-up 8 (`SegmentationClass`, `repaired_images`)
was reviewed and kept.

1. **CPU-parallel `--resume` computed before skipping** (medium). The parallel
   branch submitted `_compute_image` for every input and only then checked
   `_existing_output`, so exact single-mask artifacts still ran inference (and
   could trigger model loading/errors). `pending_paths` is now filtered before
   `executor.submit`, the skipped count is recorded up front, and only pending
   inputs are computed. Covered by
   `tests/test_segmentation.py::test_parallel_resume_skips_before_computing`.

2. **`no_mask_images.txt` kept stale entries** (low). The ledger was only
   rewritten when this run had 0-mask sources, so an input that later produced
   a mask stayed listed. It is now refreshed each run: prior entries minus
   successfully handled sources, plus this run's 0-mask sources; the file is
   deleted when nothing remains. Covered by
   `tests/test_segmentation.py::test_no_mask_ledger_drops_reprocessed_source`.

---

## Addendum: full resume audit (2026-09-22)

Rather than another single-finding pass, every `--resume`/`skip_existing` path in
the tree was inventoried and classified. Findings and dispositions:

**Defects fixed**

1. **augment treated a partial copy set as complete and used an unstable seed**
   (`src/augment/service.py`). Skip now requires the exact expected
   `{stem}_aug{NN}{suffix}` set (and no leftovers); stale copies from a larger
   `--multiply` are deleted before rewriting; the per-copy seed uses the
   source's stable index in the sorted input list instead of the running
   processed count. The `_aug*` glob also matched unrelated names such as
   `_augment_backup`; the check is now the exact digit-suffixed pattern.
2. **augment seeding was not reproducible with albumentations >= 2**
   (`src/augment/runner.py`). Those versions keep the RNG on the `Compose`
   object, so seeding the global `random`/`numpy` modules did not make repeated
   runs identical. `run_pipeline` now calls `pipeline.set_random_seed(seed)`
   when available.
3. **classify embed accepted a bare basename during pre-check**
   (`entomokit/classify/embed.py`). The pre-check now only accepts
   input-relative POSIX paths (matching the embeddings contract), so a label
   like `a.jpg` for `beetles/a.jpg` fails fast. An empty merge after extraction
   now raises instead of writing an all-N/A metrics file (and never feeds empty
   arrays to UMAP); unmatched label rows are reported.
4. **synthesize kept stale copies when `--num-syntheses` decreased**
   (`src/synthesis/processor.py`). Skip requires the exact expected copy set;
   leftover `{stem}_{NN}` images and their VOC/YOLO/COCO-separate annotations
   are removed, and their unified-COCO image/annotations are dropped from the
   prior payload during the merge.
5. **measure resume could mix measurement scales** (`entomokit/measure.py`).
   `--resume` with a different `--pixel-size-um` silently kept old rows while
   adding new ones at another scale. A recorded parameter guard now rejects the
   change.
6. **extract-frames resume could mix frame intervals/content**
   (`entomokit/extract_frames.py`). Frame numbering is positional, so resuming
   with a different `--interval`, `--out-image-format`, `--max-frames`,
   `--start-time`, or `--end-time` would overwrite existing frames with
   different content. The recorded parameter guard rejects the change.

**Shared mechanism.** `src/common/resume.py::check_resume_params()` records the
output-affecting parameters under `<out-dir>/.entomokit/<command>_params.json`
and raises `ValueError` when a resume changes them. Bbox-format switches had
already been guarded in-file for segment/synthesize.

**Audited, no code change**

- **clean**: `--resume` is defined as "allow a non-empty out-dir" and always
  recomputes every image (dedup is global), so it cannot mix stale partial
  output; this matches the CLI help text.
- **classify train**: resume is delegated to AutoGluon `MultiModalPredictor`
  and operates on a model directory, not a per-item artifact set.
- **measure stale rows**: rows for masks no longer present are kept as a
  measurement ledger; `--pixel-size-um` (the only scale parameter) is now
  guarded, so the ledger cannot silently mix units.
- **segment / synthesis** count and format handling already covered by
  follow-ups 6-9.

**Non-blocking.** `.codegraph/` is now ignored by the project `.gitignore`.

### Resume audit follow-up (2026-09-22)

Read-only re-review of the audit fixes found three boundary gaps; all fixed.

1. **Parameter fingerprint was only written on resume.** `measure` and
   `extract-frames` called `check_resume_params()` only under `--resume`, so a
   normal first run recorded nothing and the first resume could change the
   parameter freely. Both now record unconditionally (a non-resume run only
   reaches the call with an empty out-dir, or after `--overwrite` cleared it).
   Covered by `tests/test_resume_audit.py::test_measure_rejects_pixel_size_change_after_normal_run`
   and `::test_extract_frames_rejects_interval_change_after_normal_run`.
2. **Synthesis deleted stale copies before the target was known readable.**
   The cleanup of `{stem}_{NN}` beyond the new `--num-syntheses` ran before
   `_load_target_image()`, so an unreadable target still lost its old copies.
   Cleanup now runs only after the target loads successfully. Covered by
   `tests/test_synthesis.py::test_resume_reduced_count_keeps_old_copies_when_target_unreadable`.
3. **Non-canonical indices counted as complete.** `_existing_synthesis_outputs()`
   accepted any digit suffix, so `a_1.png` satisfied `a_01.png` and a target was
   skipped. Candidates must now match the generator's exact `{index:02d}` name.
   Covered by
   `tests/test_synthesis.py::test_resume_ignores_non_canonical_index`.

4. **Synthesis cleanup ran before backgrounds/tasks were resolved.** Even after
   the target loaded, the stale-copy removal happened before background loading
   and task creation, so a run whose backgrounds all failed (`processed=0,
   uncreated=2`) still deleted the old copies, and unified COCO kept dangling
   `a_03.png` entries. Cleanup is now deferred: stale copies are recorded per
   target and removed only after that target produced at least one new result
   (`written_targets`), and the unified-COCO entries are dropped only for those
   targets. Covered by
   `tests/test_synthesis.py::test_resume_reduced_count_keeps_voc_copies_when_background_unreadable`
   and `::test_resume_reduced_count_keeps_coco_entries_when_background_unreadable`.

### Pillow `mode` deprecation cleanup (2026-09-22)

All 78 test warnings were the single Pillow deprecation
`Image.fromarray(..., mode=...)` (removed in Pillow 13, 2026-10-15). Production
sites `src/synthesis/processor.py::_save_image` (both branches) and
`src/utils.py::save_image_rgba`, plus the RGB fixtures in
`tests/test_synthesis.py` and `tests/test_doctor_augment_cli.py`, now omit the
argument and rely on Pillow's dtype/shape mode inference (uint8 HxWx3 → RGB,
HxWx4 → RGBA). The full suite now reports 0 warnings and also passes under
`-W error::DeprecationWarning`. Vendored `src/lama/**` uses the same pattern but
was intentionally left untouched.
