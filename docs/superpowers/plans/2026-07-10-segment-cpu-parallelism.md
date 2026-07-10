# Segment CPU Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `segment --threads` run concurrent Otsu/GrabCut image work, keep SAM3 serial, remove the unused measure worker option, and release version 0.4.1.

**Architecture:** CPU segmentation workers compute isolated per-image results only. The existing `SegmentationProcessor` main thread consumes results in sorted input order and remains the sole writer of images, metadata, repaired images, and COCO/VOC/YOLO annotations. SAM3 retains its single stateful predictor and current serial flow.

**Tech Stack:** Python 3.9+, `concurrent.futures.ThreadPoolExecutor`, argparse, pytest, OpenCV, Pillow.

## Global Constraints

- Apply `ThreadPoolExecutor` only to `otsu`, `otsu-bbox`, `grabcut`, and `grabcut-bbox`.
- Keep `sam3` and `sam3-bbox` serial on CPU, CUDA, and MPS regardless of `--threads`.
- Workers must not mutate processor-owned metadata or annotation state, or write output files.
- Preserve sorted input order for output and annotation generation.
- Keep all repair strategies, including `sam3-fill`, on the parent thread.
- `--threads` must be a strictly positive integer; the CLI default `8` is real CPU-method concurrency.
- OpenCV calls run only on independent image arrays; benchmark worker counts rather than assuming a GIL or OpenCV-threading speedup.
- Do not add dependencies, SAM3 batching, device auto-tuning, augmentation parallelism, or measurement parallelism.
- Remove `measure --threads` and `measure -n`; no replacement option is added.
- Set all live package/CLI versions to `0.4.1`.
- Update live docs and skills only; leave dated historical plans/specifications unchanged.

---

## File Map

- Modify `src/segmentation/processor.py` to select the CPU worker path and keep shared output handling on the parent thread.
- Modify `entomokit/segment.py` to document and log method-specific `--threads` behavior.
- Modify `entomokit/measure.py` to delete the unused CLI options.
- Modify `entomokit/_version.py`, `entomokit/main.py`, and `setup.py` for version `0.4.1`.
- Modify `tests/test_segmentation.py` for concurrent CPU and serial SAM3 coverage.
- Modify `tests/test_measure_cli.py`, `tests/test_cli_schema.py`, and `tests/test_resume_flags.py` for removed measure options.
- Modify `tests/test_main_cli.py` and `tests/test_package_version.py` for the release version.
- Modify `README.md`, `README.cn.md`, and `skills/entomokit-workflow/references/command-profiles.md` for current parallelism semantics.

### Task 1: Test And Implement CPU-Only Segment Worker Dispatch

**Files:**
- Modify: `tests/test_segmentation.py`
- Modify: `src/segmentation/processor.py`

**Interfaces:**
- Consumes: `SegmentationProcessor.process_directory(input_dir, output_dir, ..., num_workers: int)`.
- Produces: CPU Otsu/GrabCut methods submit image compute work to `ThreadPoolExecutor(max_workers=num_workers)` when `num_workers > 1`; SAM3 methods never create that executor; outputs match at one and multiple workers.

- [ ] **Step 1: Write failing CPU dispatch tests**

Add a fixture set with at least two valid images. Patch the executor class at the
module import site and assert its `max_workers` argument and that it receives
one work item per sorted CPU-method image. Exercise both `otsu` and `grabcut`.

```python
class ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


@pytest.mark.parametrize("method", ["otsu", "grabcut"])
def test_cpu_methods_use_image_workers(monkeypatch, image_dir, tmp_path, method):
    submitted = []

    class RecordingExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, fn, *args):
            submitted.append(args[0])
            return ImmediateFuture(fn(*args))

    monkeypatch.setattr("src.segmentation.processor.ThreadPoolExecutor", RecordingExecutor)
    processor = make_processor(method)
    processor.process_directory(image_dir, tmp_path, num_workers=2)

    assert submitted == sorted(image_dir.glob("*.png"))
```

Add a SAM3 test that patches `src.segmentation.processor.ThreadPoolExecutor`
to raise if constructed.

```python
def test_sam3_never_constructs_cpu_worker_executor(monkeypatch, image_dir, tmp_path):
    monkeypatch.setattr(
        "src.segmentation.processor.ThreadPoolExecutor",
        lambda **_kwargs: pytest.fail("SAM3 must remain serial"),
    )
    make_sam3_processor().process_directory(image_dir, tmp_path, num_workers=8)
```

- [ ] **Step 2: Run the new tests to verify failure**

Run: `pytest tests/test_segmentation.py -k "cpu_methods_use_image_workers or sam3_never_constructs_cpu_worker_executor" -v`

Expected: FAIL because directory processing is serial and no executor is constructed for CPU methods.

- [ ] **Step 3: Add a compute-only worker boundary**

In `src/segmentation/processor.py`, extract the read-and-segment portion of
the existing per-image loop into a private method returning the same data the
existing parent path needs for writing. Do not move output, repair, metadata,
or annotation calls into this method.

```python
@dataclass
class ImageComputation:
    image_path: Path
    image: np.ndarray
    masks: list[np.ndarray]
    scores: list[float]

def _compute_image(self, image_path: Path) -> ImageComputation:
    image = load_image(image_path)
    masks, scores = self._segment_and_filter(image)
    return ImageComputation(image_path, image, masks, scores)

def _write_computation(self, computation, output_dir, output_format) -> dict:
    """Parent-only former process_image output, repair, metadata, and annotation path."""
```

Move the existing `process_image` path after segmentation and confidence
filtering into `_write_computation`. It performs every output, metadata,
annotation, and repair operation. Make `process_image` a serial compatibility
wrapper around the same two stages for its existing in-memory-image callers.

- [ ] **Step 4: Dispatch only CPU methods, then write in the parent**

Import `ThreadPoolExecutor` and select the worker path by method name.
Submit every sorted input image, retain futures in input order, then call
`future.result()` in that same order and pass each result to the existing
parent-side write/annotation logic. Keep the original serial loop for SAM3.

```python
image_paths = sorted(image_paths)
cpu_parallel_methods = {"otsu", "otsu-bbox", "grabcut", "grabcut-bbox"}
if self.method in cpu_parallel_methods and num_workers > 1:
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(self._compute_image, path) for path in image_paths]
        for future in futures:
            self._write_computation(future.result())
else:
    for path in image_paths:
        self._write_computation(self._compute_image(path))
```

Preserve the current per-image exception logging/skipping behavior by catching
exceptions around each `future.result()` exactly where the serial path catches
the corresponding image failure.

- [ ] **Step 5: Add output-equivalence and worker-failure tests**

Run Otsu and mocked GrabCut on the same three sorted fixture images into two
separate output directories. Compare relative output file lists and file bytes
for `num_workers=1` versus `num_workers=2`, including `annotations.coco.json`.
Patch `_compute_image` to raise on the middle path and assert the result has
one failure, later ordered work writes successfully, and COCO output parses.

```python
def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }

def run_cpu_segment(image_dir: Path, output_dir: Path, workers: int, method: str) -> dict:
    processor = make_processor(method, annotation_format="coco")
    return processor.process_directory(image_dir, output_dir, num_workers=workers)

def test_otsu_outputs_match_with_one_and_two_workers(image_dir, tmp_path):
    run_cpu_segment(image_dir, tmp_path / "one", workers=1, method="otsu")
    run_cpu_segment(image_dir, tmp_path / "two", workers=2, method="otsu")
    assert tree_bytes(tmp_path / "one") == tree_bytes(tmp_path / "two")

def test_cpu_worker_failure_is_counted_and_later_images_write(monkeypatch, image_dir, tmp_path):
    processor = make_processor("otsu")
    original = processor._compute_image
    monkeypatch.setattr(
        processor,
        "_compute_image",
        lambda path: (_ for _ in ()).throw(ValueError("bad image"))
        if path.stem == "02" else original(path),
    )
    result = processor.process_directory(image_dir, tmp_path, num_workers=2)
    assert (result["processed"], result["failed"]) == (2, 1)
    json.loads((tmp_path / "annotations.coco.json").read_text())
```

- [ ] **Step 6: Run focused segment tests**

Run: `pytest tests/test_segmentation.py -v`

Expected: PASS, including existing Otsu, GrabCut, bbox, annotation, and new
worker-dispatch tests.

### Task 2: Clarify Segment Thread Semantics In The CLI

**Files:**
- Modify: `entomokit/segment.py`
- Modify: `tests/test_cli_help_texts.py`

**Interfaces:**
- Consumes: argparse `--threads` option and selected segmentation method.
- Produces: help text says threads apply only to Otsu/GrabCut, and runtime logs explain that SAM3 remains serial when supplied more than one worker.

- [ ] **Step 1: Write failing help and logging tests**

Add a help assertion for the exact user-facing phrase and a CLI invocation
test that captures logging when a SAM3 command receives `--threads 2`.

```python
def test_segment_threads_help_is_method_specific(segment_parser):
    assert "Otsu/GrabCut" in segment_parser.format_help()
    assert "SAM3 remains serial" in segment_parser.format_help()

def test_segment_logs_serial_sam3_threads(caplog, sam3_args):
    sam3_args.threads = 2
    run_segment(sam3_args)
    assert "SAM3 remains serial" in caplog.text

@pytest.mark.parametrize("value", ["0", "-1"])
def test_segment_rejects_nonpositive_threads(segment_parser, value):
    with pytest.raises(SystemExit):
        segment_parser.parse_args(["--input-dir", "in", "--out-dir", "out", "--threads", value])
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/test_cli_help_texts.py -k segment tests/test_segmentation.py -k serial_sam3_threads -v`

Expected: FAIL because current help calls all threads generic parallel workers
and runtime does not emit the SAM3 message.

- [ ] **Step 3: Update help and one-time runtime notice**

Replace the `--threads` help text with an explicit method scope. Before
processing, log an informational notice only when the selected method is
`sam3` or `sam3-bbox` and the requested count is greater than one.

```python
"Concurrent image workers for Otsu/GrabCut methods (default: 8); SAM3 remains serial."

if args.segmentation_method in {"sam3", "sam3-bbox"} and args.threads > 1:
    logger.info("--threads=%s ignored for SAM3: single-model inference remains serial.", args.threads)
```

Use an argparse positive-integer type rather than `max(1, value)` so invalid
user input fails visibly instead of silently changing it.

- [ ] **Step 4: Run CLI/help tests**

Run: `pytest tests/test_cli_help_texts.py tests/test_segmentation.py -v`

Expected: PASS.

### Task 3: Remove The Nonfunctional Measure Worker Option

**Files:**
- Modify: `entomokit/measure.py`
- Modify: `tests/test_measure_cli.py`
- Modify: `tests/test_cli_schema.py`
- Modify: `tests/test_resume_flags.py`

**Interfaces:**
- Consumes: `entomokit measure` argparse parser and generated CLI schema.
- Produces: `measure` accepts no `--threads` or `-n`; runtime schema excludes both options.

- [ ] **Step 1: Write failing removal tests**

Replace the default-value test with parser rejection tests and add a schema
assertion. Remove `threads` from any synthetic `Namespace` representing a
measure command.

```python
@pytest.mark.parametrize("option", ["--threads", "-n"])
def test_measure_rejects_removed_threads_option(measure_parser, option):
    with pytest.raises(SystemExit):
        measure_parser.parse_args(["--mask-dir", "masks", "--out-dir", "out", option, "2"])

def test_measure_schema_excludes_threads(cli_schema):
    names = {option["name"] for option in cli_schema["measure"]["options"]}
    assert "--threads" not in names
    assert "-n" not in names
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_measure_cli.py tests/test_cli_schema.py tests/test_resume_flags.py -v`

Expected: FAIL because measure currently registers both options and its schema
contains them.

- [ ] **Step 3: Delete the argparse options and stale test fixture field**

Remove the `parser.add_argument("--threads", "-n", ...)` declaration from
`entomokit/measure.py`. Do not add a deprecation shim. Delete the `threads`
field only from measure-specific test namespaces; keep segment fixture fields.

- [ ] **Step 4: Run measure and schema tests**

Run: `pytest tests/test_measure_cli.py tests/test_cli_schema.py tests/test_resume_flags.py -v`

Expected: PASS.

### Task 4: Release Version 0.4.1

**Files:**
- Modify: `entomokit/_version.py`
- Modify: `entomokit/main.py`
- Modify: `setup.py`
- Modify: `tests/test_main_cli.py`
- Modify: `tests/test_package_version.py`

**Interfaces:**
- Consumes: package version declaration and source-tree fallback.
- Produces: installed and source-tree CLI report `0.4.1`.

- [ ] **Step 1: Write failing version assertions**

Update existing expectations so every live version check expects `0.4.1`.
Rename `test_setup_version_is_0_3_0` and its docstring to `0.4.1`; it is
currently stale even though `setup.py` already declares `0.4.0`.

```python
def test_package_version():
    assert entomokit.__version__ == "0.4.1"

def test_main_version(capsys):
    assert run_main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.4.1"
```

- [ ] **Step 2: Run version tests to verify failure**

Run: `pytest tests/test_package_version.py tests/test_main_cli.py -v`

Expected: FAIL because the declared and fallback values are not `0.4.1`.

- [ ] **Step 3: Change each live version source**

Set exactly `0.4.1` in the package version module, `setup.py`, and
`entomokit/main.py` source-tree fallback. Do not modify historical documents.

```python
__version__ = "0.4.1"
```

- [ ] **Step 4: Run version tests**

Run: `pytest tests/test_package_version.py tests/test_main_cli.py -v`

Expected: PASS.

### Task 5: Update Live Documentation And Workflow Guidance

**Files:**
- Modify: `README.md`
- Modify: `README.cn.md`
- Modify: `skills/entomokit-workflow/references/command-profiles.md`

**Interfaces:**
- Consumes: current CLI behavior from Tasks 1-4.
- Produces: user documentation and AI workflow instructions match the release behavior.

- [ ] **Step 1: Add documentation assertions or review checks**

Add focused textual assertions to the existing documentation test location if
the repository has one; otherwise use this exact check before editing and
again after editing:

```bash
rg -n --glob 'README*.md' --glob 'command-profiles.md' '0\.4\.0|Reserved worker count|measure.*threads|Parallel workers' .
```

Expected before editing: matches current release and stale worker wording.

- [ ] **Step 2: Update README files**

Set the release number to `0.4.1`. Replace the generic parallel-processing
claim and segment table row with explicit text: Otsu/GrabCut variants process
images concurrently using `--threads`; SAM3/SAM3-bbox use one model and remain
serial on CPU/CUDA/MPS. State the `--threads` default is 8 and recommend
starting near half the logical CPU count for CPU methods. Remove the measure
`--threads` table row in both languages.

- [ ] **Step 3: Update workflow command profile**

Remove all measure worker guidance. State that SAM3 uses serial single-model
inference and that `--threads` is meaningful only for Otsu/GrabCut variants;
recommend starting at about half of logical CPU cores for those CPU methods to
avoid OpenCV oversubscription, then benchmarking the user's machine.

- [ ] **Step 4: Verify documentation has no stale live claims**

Run: `rg -n --glob 'README*.md' --glob 'command-profiles.md' '0\.4\.0|Reserved worker count|measure.*threads' .`

Expected: no matches in live README or command-profile files. Manually verify
both README files contain the method-specific segment explanation.

- [ ] **Step 5: Run all targeted tests**

Run: `pytest tests/test_segmentation.py tests/test_measure_cli.py tests/test_cli_schema.py tests/test_resume_flags.py tests/test_main_cli.py tests/test_package_version.py -v`

Expected: PASS.

### Task 6: Final Repository Verification

**Files:**
- Modify: none expected

**Interfaces:**
- Consumes: completed implementation, tests, docs, and package metadata.
- Produces: verified release-ready 0.4.1 working tree.

- [ ] **Step 1: Search all live references**

Run:

```bash
rg -n '0\.4\.0|Reserved worker count|--threads, -n' entomokit src tests README.md README.cn.md setup.py skills
```

Expected: no live stale version/measure-thread references. If a match is in a
test fixture or current help string, correct it; do not touch dated history.

- [ ] **Step 2: Run the full suite**

Run: `pytest -v`

Expected: PASS with no collection errors.

- [ ] **Step 3: Inspect changes before release commit**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only the files listed in this plan plus the new spec/plan documents,
no whitespace errors.
