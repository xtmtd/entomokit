# LaMa Mask & Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure LaMa repair uses clean binary masks with optional dilation, leverages structured logging, and fails loudly when models are missing.

**Architecture:** Extend `SegmentationProcessor` to normalize+dilate masks before delegating to a cached `LaMaInpainter`; wire CLI parameters and logging configuration so verbose mode controls LamA diagnostics, and abort when checkpoints can't load.

**Removed:** `lama_refine` repair strategy - only `lama` remains for LaMa-based inpainting.

**Tech Stack:** Python 3, OpenCV, PyTorch, pytest, logging.

---

### Task 1: Wire CLI parameter for mask dilation and verbose logging

**Files:**
- Modify: `scripts/segment.py`
- Modify: `src/segmentation/processor.py`
- Test: `tests/test_repairs.py`

**Step 1: Write the failing test**

```python
def test_cli_passes_lama_mask_dilate(tmp_path, monkeypatch):
    # Arrange a fake processor capturing lama_mask_dilate
    captured = {}

    class FakeProcessor:
        def __init__(self, *_, **kwargs):
            captured['lama_mask_dilate'] = kwargs['lama_mask_dilate']
        def process_directory(self, *a, **k):
            return {'processed': 0, 'failed': 0, 'output_files': []}

    monkeypatch.setattr('scripts.segment.SegmentationProcessor', FakeProcessor)

    # Act: run main with --lama-mask-dilate 2 and --verbose
    argv = ['--input_dir', str(tmp_path), '--out_dir', str(tmp_path/'out'), '--lama-mask-dilate', '2', '--repair-strategy', 'lama', '--verbose']
    with pytest.raises(SystemExit) as excinfo:
        scripts.segment.main(argv)

    assert excinfo.value.code == 0
    assert captured['lama_mask_dilate'] == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_repairs.py::test_cli_passes_lama_mask_dilate -q`
Expected: FAIL complaining `lama_mask_dilate` missing.

**Step 3: Write minimal implementation**

```python
# scripts/segment.py
parser.add_argument('--lama-mask-dilate', type=int, default=0, help='Iterations for mask dilation before LaMa (default: 0)')

processor = SegmentationProcessor(..., lama_mask_dilate=args.lama_mask_dilate)

if args.verbose:
    logging.getLogger('imagekit.lama').setLevel(logging.DEBUG)

# src/segmentation/processor.py __init__ signature
def __init__(..., lama_mask_dilate: int = 0):
    self.lama_mask_dilate = max(0, lama_mask_dilate)

# Pass to repair method when needed
```

**Step 4: Re-run the test**

Run: `pytest tests/test_repairs.py::test_cli_passes_lama_mask_dilate -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/segment.py src/segmentation/processor.py tests/test_repairs.py
git commit -m "feat: add LaMa mask dilation CLI wiring"
```

### Task 2: Normalize/dilate mask before LaMa and cache inpainter

**Files:**
- Modify: `src/segmentation/processor.py`
- Modify: `src/lama/lama_inpainter.py`
- Test: `test_lama.py`

**Step 1: Write the failing test**

```python
def test_lama_mask_dilation_changes_pixels(monkeypatch):
    calls = {}

    class DummyInpainter:
        def __call__(self, image, mask):
            calls['mask_sum'] = int(mask.sum())
            return image

    monkeypatch.setattr('src.segmentation.processor.LaMaInpainterFactory.get', lambda *a, **k: DummyInpainter())

    proc = SegmentationProcessor(..., repair_strategy='lama', lama_mask_dilate=1)
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1
    proc._apply_lama_repair(image, [mask])

    assert calls['mask_sum'] > mask.sum()
```

**Step 2: Run test to confirm failure**

Run: `pytest test_lama.py::test_lama_mask_dilation_changes_pixels -q`
Expected: FAIL (mask_sum unchanged / attribute missing).

**Step 3: Write minimal implementation**

```python
# processor.py
class SegmentationProcessor:
    _lama_cache: dict[str, LaMaInpainter] = {}

    def _get_lama_inpainter(self) -> LaMaInpainter:
        key = f"{self.device}:{self.lama_model}:{self.refine}"
        if key not in self._lama_cache:
            self._lama_cache[key] = LaMaInpainter(...)
        return self._lama_cache[key]

    def _prepare_lama_mask(self, combined_mask):
        mask = (combined_mask > 0).astype(np.uint8) * 255
        if self.lama_mask_dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=self.lama_mask_dilate)
        return mask

    def _repair_with_lama(...):
        mask = self._prepare_lama_mask(combined_mask)
        logger.debug("LaMa mask pixels=%d", int(mask.sum() / 255))
        inpainter = self._get_lama_inpainter()
        return inpainter(image, mask)

# lama_inpainter.py
logger = logging.getLogger("imagekit.lama")
logger.setLevel(logging.INFO)

class LaMaInpainter:
    def __init__(...):
        logger.info("Loaded LaMa checkpoint %s on %s", checkpoint_path, self.device)

    def __call__(...):
        logger.debug("masked pixels=%d", int(mask.sum()))
```

**Step 4: Re-run tests**

Run: `pytest test_lama.py::test_lama_mask_dilation_changes_pixels -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/segmentation/processor.py src/lama/lama_inpainter.py test_lama.py
git commit -m "feat: normalize and dilate LaMa masks"
```

### Task 3: Enforce fatal failure when LaMa cannot load

**Files:**
- Modify: `src/segmentation/processor.py`
- Modify: `scripts/segment.py`
- Test: `tests/test_repairs.py`

**Step 1: Write the failing test**

```python
def test_lama_missing_checkpoint_aborts(monkeypatch, tmp_path):
    class ExplodingInpainter:
        def __init__(*a, **k):
            raise FileNotFoundError('missing ckpt')

    monkeypatch.setattr('src.segmentation.processor.LaMaInpainter', ExplodingInpainter)

    proc = SegmentationProcessor(..., repair_strategy='lama')
    with pytest.raises(RuntimeError, match='LaMa model failed'):
        proc._repair_with_lama(np.zeros((4,4,3), np.uint8), np.zeros((4,4), np.uint8))
```

**Step 2: Run test to see failure**

Run: `pytest tests/test_repairs.py::test_lama_missing_checkpoint_aborts -q`
Expected: FAIL (currently falls back to OpenCV).

**Step 3: Write minimal implementation**

```python
# processor.py
def _get_lama_inpainter(...):
    try:
        ...
    except Exception as exc:
        logger.error("LaMa initialization failed: %s", exc)
        raise RuntimeError("LaMa model failed to initialize") from exc

# scripts/segment.py main
try:
    processor.process_directory(...)
except RuntimeError as exc:
    logger.error("Segmentation failed: %s", exc)
    sys.exit(1)
```

**Step 4: Re-run test**

Run: `pytest tests/test_repairs.py::test_lama_missing_checkpoint_aborts -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/segmentation/processor.py scripts/segment.py tests/test_repairs.py
git commit -m "fix: abort when LaMa checkpoints missing"
```

### Task 4: Integrate logging control with verbose flag

**Files:**
- Modify: `scripts/segment.py`
- Modify: `src/lama/lama_inpainter.py`
- Test: `tests/test_repairs.py`

**Step 1: Write the failing test**

```python
def test_lama_logs_debug_only_in_verbose(monkeypatch, caplog, tmp_path):
    class DummyProcessor:
        def __init__(self, *a, **kw):
            pass
        def process_directory(self, *a, **k):
            logger = logging.getLogger('imagekit.lama')
            logger.debug('masked area pixels: 42')
            return {'processed': 0, 'failed': 0, 'output_files': []}

    monkeypatch.setattr('scripts.segment.SegmentationProcessor', DummyProcessor)

    argv = ['--input_dir', str(tmp_path), '--out_dir', str(tmp_path/'o')]
    scripts.segment.main(argv)
    assert 'masked area pixels' not in caplog.text

    caplog.clear()
    scripts.segment.main(argv + ['--verbose'])
    assert 'masked area pixels' in caplog.text
```

**Step 2: Run test to confirm failure**

Run: `pytest tests/test_repairs.py::test_lama_logs_debug_only_in_verbose -q`
Expected: FAIL due to uncontrolled logging.

**Step 3: Write minimal implementation**

```python
# scripts/segment.py
if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger('imagekit.lama').setLevel(logging.WARNING)

# lama_inpainter.py
logger = logging.getLogger('imagekit.lama')
logger.propagate = True
```

**Step 4: Re-run test**

Run: `pytest tests/test_repairs.py::test_lama_logs_debug_only_in_verbose -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/segment.py src/lama/lama_inpainter.py tests/test_repairs.py
git commit -m "chore: align LaMa logging with verbose flag"
```

---
