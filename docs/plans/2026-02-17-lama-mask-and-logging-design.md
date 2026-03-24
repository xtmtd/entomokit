## Title

LaMa mask handling and logging updates

## Background

- Current repair strategy `-r lama` feeds the original RGB image with a combined binary mask into `LaMaInpainter`, but the mask pipeline does not guarantee single-channel 0/255 values or edge dilation. Some inputs behave as if LaMa “recreates” the removed object instead of completing the background.
- CLI emits noisy `[LaMa] masked area pixels: …` prints even in non-verbose runs, cluttering logs. When LaMa weights are missing, the code silently falls back to OpenCV repairs, hiding the problem.

## Requirements

1. Keep official LaMa expectations: original RGB image + single-channel binary mask where 1 represents the hole. Allow optional morphological dilation so users can slightly expand the hole edges.
2. Expose the dilation amount via CLI, but default to official behaviour (no dilation) unless requested. Implementation must be deterministic and work for CPU-only runs.
3. Route LaMa logs through the project logging system; only show masked-area counts or similar diagnostics when `--verbose` is set. Provide clear warnings and abort processing if LaMa cannot load.
4. Add regression coverage verifying mask preparation, CLI wiring, logging levels, and fatal failure when checkpoints are missing.

## Proposed Changes

### Mask preparation pipeline

- After all object masks are collected for a frame, compute a single union mask as `uint8` with values {0, 255}.
- Introduce a `lama_mask_dilate` integer parameter (default 0) on `SegmentationProcessor` and surface it through `segment.py` via `--lama-mask-dilate`. When >0, apply `cv2.dilate` with a 3x3 elliptical kernel and `iterations=lama_mask_dilate` before handing the mask to LaMa.
- Pass the normalized (and optionally dilated) mask into `LaMaInpainter`; update unit tests so mask stats reflect the dilation toggle.

### LaMa inpainter lifecycle

- Lazily instantiate `LaMaInpainter` once per processor/device/checkpoint combo and cache it, reducing repeated checkpoint loads.
- If initialization fails (missing weights, config, incompatible device), raise `RuntimeError`. The CLI should catch this and stop the run with a clear error message.

### Logging behaviour

- Replace plain `print` statements with a module logger namespace `logging.getLogger("imagekit.lama")`.
- Respect the existing `--verbose` flag: DEBUG logs (masked pixels, dilation info, device selection) only appear when verbose is enabled; INFO logs cover one-time “LaMa model loaded” messages; WARN/ERROR remain visible in all modes.

### Testing

- Extend `test_lama.py` (or add a new test) to mock `LaMaInpainter` and assert that mask dilation changes the number of masked pixels passed downstream.
- Add a regression test within `tests/test_repairs.py` (or equivalent) that runs `SegmentationProcessor` with `--lama-mask-dilate=1` and checks logging behaviour under verbose vs non-verbose modes.
- Add a negative test ensuring that an invalid `--lama-model` path causes a failure instead of silently falling back to OpenCV.

## Open Questions

- Do we need to persist the dilated mask preview for debugging? (Out of scope unless requested later.)
