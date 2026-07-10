# Segment CPU Parallelism Design

## Goal

Make `entomokit segment --threads` perform actual image-level concurrency for
the CPU-only Otsu and GrabCut methods, while preserving serial SAM3 inference
and deterministic output handling. Remove the unused `measure --threads`
option and release the change as version 0.4.1.

## Scope

- Apply a `ThreadPoolExecutor` only to `otsu`, `otsu-bbox`, `grabcut`, and
  `grabcut-bbox`.
- Keep `sam3` and `sam3-bbox` on their existing single-predictor, serial path.
- Keep all output, metadata, and annotation writes on the main thread.
- Remove `measure --threads` and its short alias from the public CLI.
- Update user documentation, workflow command guidance, version declarations,
  and affected tests.

## Non-goals

- No SAM3 batching or multi-model inference.
- No device-specific worker tuning or automatic OpenCV thread controls.
- No augmentation or measurement parallelization.
- No change to existing historical design documents or plans.

## Design

`SegmentationProcessor.process_directory()` sorts and filters input images
before processing. For the four CPU methods, it submits one pure per-image
compute task to `ThreadPoolExecutor(max_workers=num_workers)`. A task may read
its source image and calculate the segmentation result, but it does not mutate
processor-owned metadata or write annotations.

### Process Boundary

`SegmentationProcessor.process_image()` currently combines segmentation,
repair, output, metadata, and COCO/VOC/YOLO annotation writes. Split it into
two private stages:

- `_compute_image(image_path) -> ImageComputation`: worker-side image loading,
  segmentation, confidence filtering, and per-image data preparation. It has
  no output paths, writer calls, or processor-state mutation.
- `_write_computation(computation, output_dir, output_format) -> dict`:
  parent-side crop/alpha generation, repair, image output, metadata, and
  annotation writes. `process_image()` becomes the serial compatibility
  wrapper that computes one in-memory image and calls this write stage.

This boundary keeps all repair strategies on the parent thread. In particular,
`sam3-fill` must never call the shared SAM3 wrapper from a CPU worker.

The main thread consumes each future in the sorted input order and performs
the existing image output, repair output, metadata, and COCO/VOC/YOLO writes.
This preserves output ordering and avoids races in the processor's annotation
accumulators and metadata manager. Worker exceptions are reported through the
existing per-image failure behavior without corrupting shared output state.

SAM3 methods retain the existing serial loop regardless of `--threads`.
`--threads` remains accepted for command compatibility, but its CLI help,
runtime logging, README files, and workflow command profile state that it
controls concurrent image workers only for Otsu and GrabCut. SAM3 uses one
stateful predictor and processes images serially on CPU, CUDA, and MPS.

The CLI default for `--threads` is 8 and becomes real concurrency for CPU
methods. Both the CLI and `process_directory()` must reject zero or negative
values rather than silently clamping them. Documentation recommends starting
near half the logical CPU count because OpenCV may use internal native threads.
The implementation relies only on independent image arrays and no shared
OpenCV state; it does not promise that OpenCV releases the GIL, and throughput
must be benchmarked on each target platform.

`measure --threads` and `-n` are deleted because they have no effect. No
replacement option is introduced.

## Version And Documentation

Set package, CLI, and source-tree fallback version references to `0.4.1`.
Update `README.md`, `README.cn.md`, and the workflow command profile. Update
the segmentation, measure CLI, CLI schema, resume fixture, main CLI, and
package-version tests. At implementation time, search the repository for stale
version literals, `measure --threads`, and segment thread descriptions so
generated/help/test references are not missed.

CLI schema export and shell completion are runtime-derived from argparse, so
removing the measure option updates them without editing generated artifacts.
The project has no changelog, release workflow, static completion file, or CI
release metadata to update. Existing dated plans and design documents remain
historical records and are not edited.

## Verification

- Add tests proving Otsu/GrabCut submit image work through the executor when
  `threads > 1`, and SAM3 does not.
- Compare CPU method outputs from one worker and multiple workers on the same
  sorted fixture set, including annotations when supported by the fixture.
- Test a worker exception: it increments the normal per-image failure count,
  later ordered work still completes, and shared annotation output remains
  valid.
- Test that zero and negative `--threads` values are rejected by argparse.
- Update CLI tests so `measure --threads` is rejected and the version is
  `0.4.1`.
- Run the relevant segmentation, measure CLI, main CLI, package version, and
  full test suites after source is available.
