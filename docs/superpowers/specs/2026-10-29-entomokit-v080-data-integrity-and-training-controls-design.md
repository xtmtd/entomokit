# EntomoKit v0.8.0 Data Integrity and Training Controls Design

## Goal

Release EntomoKit `0.8.0` with four related corrections:

1. `clean` records every keep/drop decision in structured, diffable reports instead of prose-only logs.
2. `clean` deduplication is deterministic and group-based rather than first-thread-wins and pair-based.
3. `segment` exposes SAM3's own confidence threshold separately from the existing post-filter.
4. `classify train` exposes AutoMM's existing `tuning_data` parameter as an explicit validation CSV.

The release also aligns English and Chinese command documentation with the `0.7.1` documentation conventions, and corrects two adjacent determinism defects in `clean` (output-name assignment and the MD5-registration leak).

## Scope

- `clean`: per-input decision reports with hashes, deterministic group deduplication, optional dry-run, deterministic output naming, write-failure promotion with partial-file cleanup, cumulative-index resume matching, and augment manifest file rows.
- `segment`: `--sam3-confidence-threshold`, `--post-confidence-threshold`, and a deprecated `--confidence-threshold` alias, with a corrected no-warning default and a corrected no-mask reason.
- `classify train`: `--val-csv`, validation checks, `tuning_data` wiring, and validation provenance logging before `fit()`.
- Documentation: affected command documents, current master design, Version Notes, corrected workflow-skill factual notes, and version declarations for `0.8.0`.

## Non-goals

- No new splitting algorithm. `classify train` uses AutoMM's existing `tuning_data`; `split-csv` remains the tool that creates explicit splits.
- No split-aware deduplication in `split-csv`. The `test > val > train` priority in `clean` is a safety net for already-split trees, not a replacement for deduplicating before splitting.
- No BK-tree or other approximate nearest-neighbor index for pHash grouping in this release.
- No per-image validation manifest or per-image validation hash. The validation CSV's file-level SHA256 is logged.
- No content-hash overlap check between training and validation images. Overlap is path-level after `Path.resolve()` and case normalization; content duplicates (including augmented copies) are not detected.
- No changes to the vendored SAM3 model implementation.
- No changelog, documentation generator, second CLI registry, or new dependency.
- No workflow-policy changes. Only factual command notes in the workflow skill are corrected.
- No commit, tag, push, or publication without separate operator approval.

## Background

### clean

The current implementation checks and registers hashes inside a lock while images are processed by a thread pool. Whichever image reaches the lock first becomes the retained copy; the later image is omitted. The decision therefore depends on thread scheduling, not on the data. Two same-named files can also receive `_1` suffixes in completion order, so even a deterministic dedup decision can be paired with nondeterministic output names.

This is nondeterministic and has been observed: repeated runs of the same input set can keep a `test/` copy on one run and a `train/` copy on another. When a duplicate pair spans the train/test boundary, the leak direction is therefore random.

The current log writes prose lines that combine reason, matched basename, and distance. Retained files are not recorded. Basenames are ambiguous when the same name appears in multiple class directories. There is no per-file table for joining back to labels, computing per-class retention, or diffing two runs.

The current `clean` also registers an MD5 before later pHash or save work. If that later work fails, the MD5 entry can remain, causing later real images to be dropped even though no output exists. With three identical images and a first-save failure, the current implementation can finish with `processed=0` and no output at all.

`clean` does not delete input images; it omits duplicates from `cleaned_images/`. Only `--overwrite` deletes output-directory contents. Reports and documentation must use "drop/omitted", not "deleted".

### segment

`Sam3Processor` already accepts a `confidence_threshold` and applies `out_probs > threshold` inside the model. `Sam3Predictor` exposes that constructor value, but `SAM3Wrapper` does not pass it and `SegmentationProcessor` does not expose it. The model therefore always uses its default `0.5`. The CLI's `--confidence-threshold` is only a post-filter, so values below `0.5` cannot make SAM3 more permissive.

Otsu and GrabCut assign a constant score of `1.0`, so the post-filter has no practical effect in the documented `0..1` range.

### classify train

`train()` passes only `train_data` to `predictor.fit()`. AutoMM therefore performs its own random train/validation split on every run. Each model arm can receive a different slice, augmented copies can land on both sides, and the selected validation rows are not logged.

AutoMM already supports `fit(tuning_data=...)`. Supplying it disables the internal random split. No new splitting logic is required. Local AutoGluon 1.5.0 documents that `holdout_frac` is ignored when `tuning_data` is supplied.

### resume

The existing `--resume` path hashes existing output files and compares those hashes with input files. Because `compute_md5` hashes decoded pixels (`img.tobytes()`), a resized or lossy-converted output can have a different MD5 from its input. With the default 512-pixel resize and JPEG conversion, MD5 resume matching can miss those outputs and re-write duplicates with `_1` suffixes; pHash may still match, but is not guaranteed. The new cumulative manifest index records the source hashes and provides the primary resume key; output hashing remains a fallback for inputs the index does not match.

## Design

### 1. clean decision pipeline

#### 1.1 Three phases

`clean` is split into three phases:

1. **Hash phase (parallel):** read every input, compute the hashes required by `--dedup-mode`, and close each image. Results are stored by input-relative path. No keep/drop decision is made in a worker.
2. **Decision phase (serial):** sort records by input-relative POSIX path, build connected groups, choose representatives, and assign one decision row per input.
3. **Write phase (parallel):** write retained images to output names assigned before workers start. A write failure may promote another group member and update the decision rows.

The decision phase is deliberately serial. It is cheap compared with image I/O and gives a single deterministic ordering for grouping, representative selection, group IDs, and output names. Promotion results are finalized serially in `group_id` order so names and rows do not depend on future completion order.

The input image is read twice at most: once for hashing and once for writing. The "read once" rule applies within the hash phase; it is not a whole-pipeline guarantee.

#### 1.2 Grouping

- `md5`: exact pixel-content MD5 groups. `compute_md5` hashes `img.tobytes()`, so this is not a file-byte MD5. Two different file encodings with identical decoded pixels group together; an `md5sum` of the files will not reproduce the reported value.
- `phash`: union-find over every pair whose Hamming distance is at most `--phash-threshold`. This is a connected-component (transitive) definition, not a pair list.
- `md5+phash`: exact MD5 groups first, then pHash union-find over the MD5-group representatives. Member sets are merged when representatives join.
- `none`: every input is a singleton group.

pHash grouping is O(n²) in this release. This is accepted to keep the implementation dependency-free and exactly transitive. A future release may add an index without changing report semantics.

If `phash` cannot compute a valid hash for an input, that input becomes a hash/decode error with empty group fields. In `md5+phash`, a failed pHash after a successful MD5 keeps the record as an MD5-only candidate instead of failing the image.

#### 1.3 Representative selection

For groups without existing outputs, the representative is the member with the smallest tuple:

1. split rank: `test` = 0, `val`/`valid`/`validation` = 1, `train` = 2, no recognized split = 3;
2. input-relative POSIX path, lexicographically.

Split detection is case-insensitive and matches exact path components. If a path contains multiple recognized segments, the highest-priority segment wins. This prevents a duplicate group that spans train/val/test from keeping the training copy and dropping the held-out copy.

Groups without recognized split segments fall back to path order only.

Under `--resume`, current inputs are matched directly against the cumulative index and output hashes using the current mode. Matching inputs become `status="existing"`; unmatched inputs form normal groups. If multiple existing outputs match one input, the current-mode tie-break selects the canonical `kept_path` (MD5: path; pHash: distance then path). Existing outputs are never deleted or rewritten.

#### 1.4 Matching metadata

For every non-representative input, the report records the nearest group member:

- MD5: the representative, with no distance.
- pHash: the member with the smallest Hamming distance; ties are broken by relative path.
- `md5+phash`: if the member shares an MD5 with any group member, `reason=duplicate_md5`, `distance=None`, matching the representative when it shares the MD5, otherwise the lexicographically smallest same-MD5 member. Otherwise `reason=duplicate_phash` with the smallest pHash distance and a path tie-break.

`matched_path` is the nearest member, not necessarily the representative. This preserves the edge that actually pulled the record into a transitive group even when its distance to the representative exceeds the threshold. `kept_path` identifies the group representative.

`retained` rows use `kept_path=input_path` and `kept_scope="input"`. `dropped` rows use `kept_scope="input"` and `matched_scope="input"`; they only occur in normal input groups. `existing` rows point `kept_path`, `matched_path`, and `output_path` at the canonical existing output with `kept_scope="existing_output"` and `matched_scope="existing_output"`. For `existing` rows, `md5`/`phash` describe the current input, not the existing output. Group IDs are assigned in one sequence across input groups and existing matches, sorted by `(split_rank, representative path)`; existing matches use their output-relative canonical path.

#### 1.5 Deterministic output names

Retained input decisions are sorted by `input_path` before names are assigned. Names use the existing rules:

- `clean_filename` on the source stem;
- the output extension from `--out-image-format`;
- a case-insensitive `_1`, `_2`, ... suffix when the name is already used in that output directory;
- existing output names under `--resume` are included in the used-name set.

No worker chooses a name or appends a suffix based on completion order. `output_path` is stored relative to `--out-dir`, for example `cleaned_images/beetles/a.jpg`. Promoted members receive names from the same deterministic allocator after all initial names are fixed.

#### 1.6 Write failure and promotion

Representatives are written in parallel to preassigned paths. After all write futures complete:

1. iterate groups in `group_id` order;
2. for a failed representative, best-effort remove the partial target file (`dst.unlink(missing_ok=True)`);
3. mark that decision as `error` with `reason=write_error:<message>`, preserving `group_id` and `group_size`;
4. promote the next group member in representative order, assign it a new output name, and write it;
5. if the promoted member succeeds, mark it `retained` with an empty reason and the new `output_path`;
6. update every group member's `kept_path`/`kept_scope`/`matched_path`/`matched_scope` to reference the surviving representative, and recompute `reason` and `distance` whenever `matched_path` changes.

If no member can be written, all remaining members of the group are reported as errors with `reason=group_write_failed`, preserving `group_id`/`group_size` and clearing path fields. No half-written file may remain in `cleaned_images/`.

No phase writes or deletes input files.

#### 1.7 Report artifacts

Reports are written to the parent directory of `log_path` (the command's `--out-dir`), not inside `cleaned_images/`. Every report row list is sorted by `input_path`.

`filter_decisions.csv` has exactly one row per input:

```text
input_path,status,reason,kept_path,kept_scope,matched_path,matched_scope,distance,group_id,group_size,md5,phash,output_path
```

`retained_images.csv` contains `retained` and `existing` rows:

```text
input_path,status,output_path
```

`dropped_images.csv` contains only `dropped` rows:

```text
input_path,reason,matched_path,matched_scope,distance,group_id
```

`clean_manifest.json` contains run-level fields, a `files` array equal to the `filter_decisions.csv` rows, and the cumulative `index` array used by resume:

```json
{
  "schema_version": 1,
  "command": "clean",
  "dedup_mode": "md5",
  "phash_threshold": 5,
  "threads": 12,
  "dry_run": false,
  "n_input": 0,
  "n_retained": 0,
  "n_dropped": 0,
  "n_existing": 0,
  "n_errors": 0,
  "output_files": {
    "retained_images": "retained_images.csv",
    "dropped_images": "dropped_images.csv",
    "filter_decisions": "filter_decisions.csv",
    "cleaned_images": "cleaned_images/"
  },
  "files": [],
  "index": [
    {
      "input_path": "beetles/a.jpg",
      "output_path": "cleaned_images/beetles/a.jpg",
      "status": "retained",
      "md5": "pixel-content-md5-hex",
      "phash": "0000000000000000"
    }
  ]
}
```

`files` is the current run's rows and MUST NOT be used as a resume index. `index` is the cumulative resume index: one row per known input path, keyed by `input_path`, pointing at the canonical output path with the source hashes used for matching. Its `status` may be `retained`, `existing`, or `dropped`; dropped rows preserve their own source hashes but point at the surviving representative output. A dry-run copies the previous valid `index` unchanged and adds no would-be outputs. In `--dry-run`, `output_files.cleaned_images` is `null`. `files` uses typed values: `distance` is `int | null`, `group_size` is an integer, and all paths are strings.

The `md5` column is the pixel-content MD5 hex returned by `compute_md5`; `phash` is the zero-padded 64-bit hexadecimal pHash. Either column is empty when that hash was not computed for the run.

Statuses:

- `retained`: group representative; a new output is written (or would be written under `--dry-run`).
- `dropped`: duplicate in a normal input group; reason `duplicate_md5` or `duplicate_phash`. `kept_scope="input"`; `matched_scope="input"`.
- `existing`: input matches an existing output under `--resume`; no new file; `kept_path`/`matched_path`/`output_path` are the canonical existing output. When matching used pHash, `distance` is the Hamming distance to the canonical existing output; for MD5/`none` matching it is empty. `md5`/`phash` describe the current input, not the existing output. `group_id` is assigned by canonical output path order; `group_size` is the number of current inputs mapping to that canonical output plus one for the existing output.
- `error`: hash/decode failure (`unidentified_image`, `hash_error:<message>`) with empty `group_id`/`group_size` and empty paths, or write failure (`write_error:<message>`, `group_write_failed`) with preserved `group_id`/`group_size`. A write error with a surviving member keeps `kept_path`/`kept_scope` from the survivor and leaves `matched_path`/`matched_scope` empty; an exhausted group clears all path fields.

`n_input == n_retained + n_dropped + n_existing + n_errors`. `n_retained` counts new outputs only; `n_existing` counts input matches, not unique existing outputs. `retained_images.csv` therefore has `n_retained + n_existing` rows, with possible repeated `output_path` values; per-class retention must be computed from `output_path`, not `input_path`.

Reports are written atomically: each CSV/JSON is written to a temporary file in `--out-dir` and moved into place with `os.replace`. The log summary is appended only after all reports are complete.

If the input directory is empty, the command still writes zero-row reports and a valid manifest with `n_input=0`.

The log is emitted from the finalized decision rows after the write/promotion phase, in `input_path` order. The per-file lines use the same statuses, paths, reasons, and distances as the reports; the old worker-time prose lines are removed. The log ends with a summary computed from the same rows.

#### 1.8 Dry-run and resume

`clean --dry-run`:

- writes the same reports and manifest as a real run with no write failures;
- decodes each would-be retained image and applies resize/pad in memory, but skips `mkdir`/`save`;
- creates no `cleaned_images/` directory;
- deletes nothing;
- sets `output_files.cleaned_images` to `null`;
- is mutually exclusive with `--overwrite`;
- requires an empty `--out-dir` or `--resume`.

After a dry run, a real run into the same directory also requires `--resume`, because the reports and log make the directory non-empty. Documentation must show this flow. The CLI summary uses `would write N new image(s)` for dry-run and `wrote N new image(s)` for a real run, with singular/plural matching the count.

`clean --resume` matching precedence:

1. Load `clean_manifest.json` from `--out-dir` only when it is schema-valid and not corrupt. Ignore unsupported `schema_version`, malformed JSON, invalid rows, and `output_path` values that are absolute, non-POSIX-relative, or resolve outside `--out-dir` or outside the `cleaned_images/` output directory, each with a warning. A dry-run manifest may contribute its `index` only; never treat its `files` as outputs.
2. The cumulative `index` is the primary source, including `dropped` rows that point at a surviving output. Match current inputs by the current mode: exact source MD5 for `md5`; nearest source pHash within threshold for `phash`; exact MD5 then nearest pHash for `md5+phash`; exact `input_path` for `none`. If multiple index rows map to one output, use the lexicographically smallest output-relative path as canonical.
3. Ignore index rows whose output is missing, unreadable, or outside `--out-dir`/`cleaned_images/`, with a warning.
4. For inputs not matched by the index, build an output hash map from **all** existing output image files, not only paths absent from the manifest, and match by output hash using the current mode. This covers an older manifest that has a path but no usable hash for the current mode. Log a warning that MD5 output-hash matching may miss resized/lossy-converted outputs; recommend a fresh run or `--overwrite` when exact matching is required.
5. `dedup_mode="none"` without a usable index/manifest must not match by mirrored output path; retain the input and warn that exact resume matching requires a manifest.

After a real run, write `index` even when `files` is empty. Merge current `retained`/`existing` rows and `dropped` rows that have a surviving representative into `index`, replacing any previous row for the same `input_path`. A dropped row's index row maps to the surviving representative output and keeps the dropped input's own source hashes, so transitive group members remain resumable; a write-error row with a surviving representative is treated the same way. Remove only rows for hash/decode errors and exhausted-group errors, which have no usable output or hash. Keep prior rows for inputs not present in the current run while their output exists; prune missing outputs with a warning. A dry-run copies the previous `index` unchanged.

Existing outputs are never deleted or rewritten. Reports cover inputs discovered in the current invocation and are overwritten per invocation; `index` is not. The log continues to append.

### 2. segment thresholds

Two independent thresholds are exposed:

| Option | Default | Range | Applies to |
|---|---|---|---|
| `--sam3-confidence-threshold` | `0.5` | `0..1` | SAM3 and SAM3-bbox model inference |
| `--post-confidence-threshold` | `0.0` | `0..1` | EntomoKit post-filter after masks are returned |
| `--confidence-threshold` | `0.0` | `0..1` | Deprecated alias for the post-filter |

The effective SAM3 filter is `max(sam3_confidence_threshold, post_confidence_threshold)` with model-side `>` and post-side `>=`. A post threshold at or below the model threshold is a no-op. Warn only when the resolved post threshold is strictly between zero and the SAM3 threshold (`0 < post < sam3`); do not warn for the default `post=0.0`, which means "no post-filter".

Alias resolution: `0.0` is treated as "unset". When `--post-confidence-threshold` is `0.0` and `--confidence-threshold` is greater than zero, the legacy value is used and an info line is logged. When both are non-zero and different, `--post-confidence-threshold` wins and a warning is logged. An explicit `--post-confidence-threshold 0.0` therefore does not override a non-zero legacy value.

The deprecated alias remains accepted and visible in `--help`/command docs, but is marked `_deprecated` and omitted from generated shell completion so users are steered to `--post-confidence-threshold`. It remains in the runtime schema and parameter guard for backward compatibility.

The SAM3 threshold is passed through:

```text
SegmentationProcessor
  -> SAM3Wrapper(confidence_threshold=...)
  -> Sam3Predictor(confidence_threshold=...)
  -> Sam3Processor(confidence_threshold=...)
```

The vendored `src/sam3/model/sam3_image_processor.py` is not edited. Its existing constructor and `_forward_grounding` implementation already apply the threshold.

For Otsu and GrabCut, the SAM3 threshold is ignored. Their constant `1.0` score makes the post-filter ineffective in the documented `0..1` range. The "ignored" log line is emitted only when the SAM3 threshold differs from its default `0.5`.

The no-mask reason names both thresholds, not just the post threshold:

```text
No masks passed the effective confidence threshold (sam3=<model>, post=<post>)
```

Use that message for SAM3 when either threshold is greater than zero; keep `No masks returned by <method>` when both are zero. This distinguishes "the model returned nothing above its own threshold" from "the post-filter removed everything".

Both thresholds are validated in the CLI and in `SegmentationProcessor`. `src/segmentation/processor.py` does not import `from __future__ import annotations`, so it must use `Optional[float]`, not `float | None`.

COCO annotations in both unified and separate modes include a per-mask `score`: SAM3 writes the model score, while Otsu/GrabCut write `1.0`. VOC XML and YOLO TXT have no standard score field and remain unchanged. The score is written by `COCOMetadataManager.add_annotation(..., score=...)`; omitting the argument preserves the previous schema.

### 3. classify train validation

`--val-csv` is optional. Its `image` column is resolved against the same `--images-dir` as the training CSV. Absolute image paths are accepted. Cross-frame label-subset and overlap checks run in `train()` after both frames are loaded; the per-file helper only handles columns, path resolution, empty files, missing validation images, and duplicate validation rows.

When supplied:

- load and validate the validation CSV before `fit()`;
- require `image` and `label` columns;
- require at least one row;
- require every referenced image to exist;
- reject duplicate resolved image rows within validation;
- reject validation labels absent from the training labels;
- reject any resolved, case-normalized image path present in both training and validation;
- pass the resolved DataFrame as `fit(tuning_data=val_df)`;
- log the validation CSV path, file-level SHA256, image count, and overlap result before `fit()`.

Training-side loading keeps its existing leniency: require the `image` and `label` columns and resolve paths, but do not add missing-file or duplicate-row rejection for training in this release.

When omitted:

- do not pass `tuning_data`;
- keep AutoMM's internal random split;
- log `Validation: AutoMM internal random split (no --val-csv)`.

No new split logic is added. `train.processed.csv` remains the training table. No per-image validation manifest or per-image validation hash is created. The existing NVML retry path reuses the same `fit_kwargs`, including `tuning_data`. Missing, corrupt, or unwritable `assets.json` produces a warning only and does not fail training.

After `fit()` returns, entomokit backfills the model's `assets.json` with `best_score` from `predictor._learner` and `total_train_time` from the learner (falling back to measured elapsed time). AutoMM writes `assets.json` during top-k averaging before those attributes are populated, so a reloaded model otherwise reports null `fit_summary()` metadata. The keys are updated, never deleted, because `MultiModalPredictor.load()` indexes them.

### 4. Documentation and release

- Release notes live in each affected command document's `Version Notes` / `版本注记`, newest first.
- Do not add a changelog.
- Update both languages for every changed command: `clean`, `segment`, `classify-train`, and `augment` (because its manifest gains `files`).
- Update the current master design (`2026-03-24-entomokit-refactor-design.md`) with architecture-level contracts; do not duplicate full parameter tables.
- Add a short `Parameter Changes in 0.8.0` section to `README.md` and `README.cn.md` with Key-Parameters-style tables for `clean --dry-run`, the segment thresholds, and `classify train --val-csv`. Keep it outside the Commands H2 so the README command-index contract still sees only command rows; the command docs remain the full parameter reference.
- Correct the factual command notes in `skills/entomokit-workflow/references/command-profiles.md` for `--val-csv` and clean reports/dry-run. Do not change workflow policy.
- Do not edit historical plans or completed specs.
- Set the release version to `0.8.0` in `version.txt`, `entomokit/_version.py`, the `entomokit/main.py` fallback, `setup.py`, and version tests.
- Do not change workflow-skill policy for this patch.

## Compatibility and Migration

- Existing `clean` invocations keep the same CLI flags and output location. The new reports are additive, but report/log files make `--out-dir` non-empty on later runs, so reuse requires `--resume` or `--overwrite`.
- `--confidence-threshold` remains accepted. It now means the post-filter; `--post-confidence-threshold` takes precedence when both are non-zero and different.
- Existing SAM3 behavior is preserved by the default `0.5` model threshold.
- Existing Otsu/GrabCut behavior is preserved.
- Existing `classify train` behavior is preserved when `--val-csv` is omitted; training-side CSV leniency is unchanged.
- `augment_manifest.json` gains a `files` array; existing run-level fields remain.
- COCO annotations written before `0.8.0` do not gain per-mask `score` under `--resume`, because existing images are skipped; rerun with `--overwrite` to regenerate them.
- `clean` deduplication results can change because groups are transitive rather than pairwise. This is intentional and must be called out in Version Notes.
- `--resume` now prefers the cumulative manifest `index`, so resized/converted outputs can be recognized and subset runs do not lose prior index rows. Without a manifest, MD5 output-hash matching remains limited and warns.
- `--resume` may report matching inputs as `existing` instead of writing duplicates.
- `retained_images.csv` includes `existing` rows but is input-centric; multiple inputs may map to one output. Per-class retention must be computed from `output_path`, and `n_existing` counts inputs rather than unique outputs.
- `--resume` keeps existing outputs as-is; changing `--out-short-size` or `--out-image-format` requires `--overwrite`. On an already-split tree, an existing train output can keep a duplicate out of test; use `--overwrite` or clean before splitting to rebalance.
- `dedup_mode="none"` without a manifest does not match by mirrored name; exact resume for `none` requires a manifest.
- Reports name the pixel-content MD5, not a file-byte MD5. For `existing` rows, `md5`/`phash` describe the current input, not the existing output.
- Old `log.txt` files remain readable; the new reports are the structured source of truth.

## Verification

- `clean` at `--threads=1`, `8`, and `32`, and repeated runs, produce identical decision rows in identical order and identical output file sets.
- A pHash chain `A~B`, `B~C`, `A!~C` forms one group; `md5+phash` labels MD5-equal drops `duplicate_md5` and the rest `duplicate_phash`.
- A duplicate group spanning train/val/test keeps the test member, then val, then train.
- MD5 and pHash match paths are relative and unambiguous across class directories; hash columns are populated and agree across duplicate rows.
- Representative write failure removes the partial file, promotes the next member, and updates group references. A failed MD5 registration cannot drop a later real image.
- An exhausted group reports every member as `group_write_failed` with preserved `group_id`/`group_size`.
- `filter_decisions.csv` counts match `clean_manifest.json` and the log summary; row order is sorted by `input_path`.
- An empty input directory writes zero-row reports and a valid manifest with `n_input=0`; a fresh empty run has `index=[]`, while an empty resume keeps the prior index rows whose outputs still exist.
- `clean_manifest.json` contains both `files` (current run) and cumulative `index`; `files` is never used as the resume index.
- A `md5+phash` existing match by MD5 reports an empty `distance`; only pHash matches carry a distance.
- `clean --dry-run` writes reports without images, sets `output_files.cleaned_images=null`, and rejects `--overwrite`; a non-empty out-dir requires `--resume`.
- `clean --resume` recognizes a resized/converted output through the cumulative manifest `index` and reports `matched_scope=existing_output`; it never rewrites the output.
- Resume ignores index outputs that are absolute, outside `--out-dir`, outside `cleaned_images/`, missing, or not images.
- A+B → A resume → B resume keeps B `existing`; the cumulative index is not lost when a later run covers a subset of inputs.
- A transitive group `A~B~C` where only C was retained still matches B on a later resume through B's own index row (status `dropped`) pointing at C's output.
- A previous md5-only manifest with current phash mode falls back to output hashing for outputs lacking a usable phash index row.
- Indexed existing outputs, not only scanned outputs, are eligible for decoded-output hash fallback.
- Indexed outputs are fully decoded before acceptance; truncated outputs are ignored and the input is regenerated.
- Validation labels are mapped to the training label representation before `fit()`; blank validation image/label cells are rejected.
- `clean --resume` without a manifest warns when MD5 output-hash matching may be unreliable.
- `dedup_mode="none"` without a manifest does not mark a same-stem, different-extension input as existing.
- Promotion recomputes `reason`, `distance`, `kept_scope`, and `matched_scope` after the representative changes.
- `retained_images.csv` per-class retention is computed from `output_path`; `n_existing` counts inputs, not unique outputs.
- SAM3's threshold reaches `Sam3Predictor`; `--sam3-confidence-threshold 0.2` can retain masks in `(0.2, 0.5]`.
- Default SAM3 thresholds do not warn; an explicit `0 < post < sam3` warns and does not error.
- The no-mask reason names both thresholds.
- COCO unified and separate annotations carry the mask `score`; VOC/YOLO output is unchanged.
- Otsu/GrabCut output is unchanged by the SAM3 threshold.
- AutoMM receives `tuning_data` only when `--val-csv` is supplied.
- Overlap after resolution/case normalization, duplicate validation rows, empty validation, missing images, and unknown labels fail before `fit()`.
- The validation CSV path, SHA256, image count, and overlap result are logged before `fit()`.
- A completed training run writes non-null `best_score` and `total_train_time` to `assets.json`, and a reloaded `fit_summary()` reports them.
- `tests/test_docs.py` passes with both languages updated.
- Generated bash/zsh/fish completion omits `--confidence-threshold` while `segment --help` still documents it as the deprecated alias.
- README.md and README.cn.md contain the `0.8.0` parameter-change tables, and `test_readme_command_index` still passes.
- Python 3.9 syntax review: no PEP 604 unions in modules without `from __future__ import annotations`.
- Full test suite passes; unavailable optional backends are reported as blockers, not hidden with skips or xfail.

## Risks and Trade-offs

- **O(n²) pHash grouping:** exact transitive semantics at the cost of scaling. Acceptable for `0.8.0`; document the boundary.
- **Split-aware representative selection:** a heuristic based on exact path components. Datasets with different split names fall back to path order.
- **Held-out-first removes training-side duplicates:** keeping the test/val copy prevents leakage, but the training set loses that sample entirely rather than just its duplicate.
- **Existing outputs always win under `--resume`:** this avoids duplicate writes but may not rebalance splits. Users who need a fresh layout should use `--overwrite`.
- **Prior-manifest resume depends on the cumulative index:** the index must survive subset runs; a stale or manually edited index is trusted. Missing/invalid index rows fall back to output hashing, and MD5 may miss resized outputs, so a warning is logged.
- **Cumulative index growth:** the index keeps one row per known input path; it grows with the input history. Pruning missing outputs keeps it bounded to existing outputs, but long-lived directories can still grow.
- **`none` without a manifest:** exact resume is impossible without a manifest; the implementation retains and warns rather than guessing from a mirrored path.
- **Write-failure promotion:** keeps at least one copy when possible, but the table reflects actual I/O success; promotion is serialized to keep names deterministic.
- **Post threshold no-op:** users may expect an error when post is below model. A warning is less disruptive and keeps `0` as a valid "no post-filter" value; warning on the default was explicitly rejected as noise. Because `0.0` also means "unset" for the deprecated alias, an explicit `--post 0.0` cannot override a non-zero legacy value; an info line makes that fallback visible.
- **Validation overlap is path-level:** different filenames pointing to duplicate content are not detected. Deduplicate before splitting.
- **Resume matching is per input:** an input that is only transitively similar to an existing output through another *new* input is retained rather than marked `existing`. Prior-run inputs are covered by the cumulative index (including dropped rows), so this only affects chains made entirely of new inputs. This is the deliberate trade-off that avoids matching an input through a chain it may not actually belong to.
- **Pixel MD5 vs file MD5:** the reported MD5 verifies decoded-pixel equality, not file-byte equality; documentation must say so.
- **Training-side leniency:** missing training images are still not rejected, to avoid a breaking change in this patch.

## Alternatives Considered

- **Keep pairwise first-wins behavior and only add a manifest:** rejected because the decision would remain scheduler-dependent and split leakage would remain random.
- **Choose the representative by train > val > test:** rejected because it preserves the leak direction called out in the issue.
- **Add `--keep-priority` now:** deferred; a fixed held-out-first rule is smaller and covers the reported failure.
- **Use a BK-tree for pHash:** deferred to keep the release dependency-free and behavior-exact.
- **Edit the vendored SAM3 processor:** rejected; the constructor already supports the threshold, so only wiring is needed.
- **Implement a new validation split in `classify train`:** rejected; AutoMM already supports `tuning_data`.
- **Add a changelog:** rejected by the current documentation conventions; Version Notes are the release-note home.
- **Warn whenever post < model:** rejected because the default `post=0.0` would warn on every SAM3 run. Warn only for `0 < post < model`.
- **Use `None` as the CLI default for `--post-confidence-threshold`:** rejected because the documented default is `0.0` and the runtime schema/parameter card would disagree. Resolution handles the deprecated alias instead.
- **Rely only on output-file hashing for resume:** rejected because resizing/conversion can change pixel MD5; the cumulative manifest `index` is the primary source.
- **Add a separate `clean_index.json`:** rejected for `0.8.0` to avoid a fifth report artifact; the cumulative `index` lives in `clean_manifest.json` while `files` stays per-run.
- **Match `dedup_mode="none"` by mirrored output path without a manifest:** rejected because same-stem, different-extension inputs would be conflated; require a manifest for exact `none` resume.

## References

- Implementation plan: `docs/superpowers/plans/2026-10-29-entomokit-v080-data-integrity-and-training-controls.md`
- Chinese plan: `docs/superpowers/plans/2026-10-29-entomokit-v080-data-integrity-and-training-controls.cn.md`
- Documentation conventions: `docs/superpowers/specs/2026-10-01-entomokit-documentation-conventions-design.md`
- Current master design: `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`
- phyloAI filter reference: `phyloAI/phyloai/pretree/filter.py` (`retained_loci.csv`, `dropped_loci.csv`, `filter_decisions.csv`, `result.json`)
