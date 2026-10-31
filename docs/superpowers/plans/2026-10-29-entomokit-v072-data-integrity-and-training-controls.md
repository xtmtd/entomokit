# EntomoKit v0.7.2 Data Integrity and Training Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release EntomoKit `0.7.2` with auditable deterministic `clean` deduplication, a real SAM3 model confidence threshold, explicit AutoMM validation input, and synchronized bilingual command documentation.

**Architecture:** Keep command adapters thin. `clean` becomes a three-stage pipeline: hash every input in parallel, decide keep/drop serially in input-relative path order using connected groups, then write in parallel to preassigned deterministic names. All reports, the final per-file log lines, and the log summary are emitted from the same finalized decision rows. `segment` passes two independent thresholds through the existing `SAM3Wrapper -> Sam3Predictor -> Sam3Processor` constructor chain without editing vendored model source. `classify train` passes an optional validation CSV to AutoMM's existing `fit(tuning_data=...)` parameter and records validation provenance in the log. Documentation follows the `0.7.1` command-document conventions.

**Tech Stack:** Python 3.9+, argparse, pathlib, `csv`, `json`, `dataclasses`, Pillow, imagehash, NumPy, `concurrent.futures`, PyTorch/AutoGluon, pytest, Markdown.

**Spec:** `docs/superpowers/specs/2026-10-29-entomokit-v072-data-integrity-and-training-controls-design.md` is the companion design document. The plan and spec are approved and revised together; neither is implementation authority until the operator approves the pair.

**Status:** Implementation complete for `0.7.2`; awaiting operator approval to commit, tag, push, or release.

## Resolved Decisions

- Release target is exactly `0.7.2`.
- One combined plan and one combined spec cover all four issues; each plan and spec has an English file and a synchronized Chinese file.
- Release notes live in each affected command document's `Version Notes` / `版本注记` section, newest first. Do not add a changelog.
- `clean` adopts the phyloAI report-file naming convention: `retained_images.csv`, `dropped_images.csv`, `filter_decisions.csv`, and a command manifest named `clean_manifest.json`. The naming convention is the reference, not the exact phyloAI column set; phyloAI writes `result.json`, while `clean` uses a command-specific manifest.
- `clean_manifest.json` has a per-run `files` array equal to `filter_decisions.csv` and a cumulative `index` array used by `--resume`. `index` has one row per known input path, including dropped duplicates that map to a surviving output, and survives per-run `files` overwrites. A dry-run copies the previous `index` unchanged.
- `filter_decisions.csv` has exactly one row per input file, sorted by input-relative POSIX path. `retained_images.csv` and `dropped_images.csv` are filtered views.
- `retained_images.csv` contains both `status="retained"` rows (new output written) and `status="existing"` rows (already-present output under `--resume`), so a resume run still shows retained inputs. It is input-centric: multiple inputs may map to one existing output. Compute per-class retention from `output_path` (the actual retained image), not from `input_path`; `n_existing` counts input matches, not unique outputs. `dropped_images.csv` contains only `status="dropped"` rows.
- Reports record the hash values used for the decision: `md5` is the pixel-content MD5 returned by `compute_md5`, and `phash` is a zero-padded 64-bit hexadecimal perceptual hash. Both columns are empty when the hash was not computed.
- `--dry-run` writes the same reports and manifest but creates no `cleaned_images/` directory and deletes nothing. It is mutually exclusive with `--overwrite`. A dry run requires an empty `--out-dir` or `--resume`; after a dry run, a real run into the same directory also requires `--resume`.
- Deduplication groups are connected components. The representative is chosen by held-out priority `test > val > train`, then by input-relative path lexicographic order. Groups with no recognized split segment use path order only.
- `md5+phash` runs exact MD5 grouping first, then pHash grouping on the surviving MD5 representatives. A non-representative row is `duplicate_md5` when it shares an MD5 with any group member, otherwise `duplicate_phash`.
- A representative write failure removes the partial target file and promotes the next group member in representative order. Promotions are finalized serially in `group_id` order so output names and decision rows do not depend on future completion order.
- `--sam3-confidence-threshold` controls SAM3's internal model threshold. `--post-confidence-threshold` controls the existing post-filter. `--confidence-threshold` remains as a deprecated alias for the post-filter. The two post-filter options (`--post-confidence-threshold` and `--confidence-threshold`) both default to `0.0`. `0.0` is treated as "unset" for alias resolution: when post is `0.0` and the legacy value is greater than zero, the legacy value is used and an info line is logged. `--post-confidence-threshold` wins when both are non-zero and different, with a warning.
- Warn only when the resolved post threshold is explicitly in effect and strictly between zero and the SAM3 threshold (`0 < post < sam3`). Do not warn for the default `post=0.0`, which means "no post-filter".
- The no-mask reason reports both the SAM3 model threshold and the post threshold, not just the post threshold.
- `--val-csv` is optional, uses the same `--images-dir` as `--train-csv`, and is passed to AutoMM as `tuning_data`. Validation overlap is checked at the resolved, case-normalized file-path level. The validation CSV's file-level SHA256 is logged before `fit()`.
- The two adjacent determinism fixes are in scope: deterministic output-name assignment and the MD5-registration leak.
- `--resume` matches inputs against existing outputs using the cumulative `clean_manifest.json["index"]` first (source hashes and input/output paths), then falls back to hashing every existing output file for inputs the index did not match. This is required because `compute_md5` is pixel-content MD5 and resizing/format conversion changes the output pixels.
- Error rows preserve group membership when the failure happens during writing. Hash/decode errors that never joined a group have empty `group_id` and `group_size=0`; write errors keep `group_id` and `group_size`.
- Report files and the manifest are written atomically (temporary file then `os.replace`) before the log summary is appended.
- Do not introduce PEP 604 `X | Y` annotations in modules that lack `from __future__ import annotations`; `setup.py` still declares Python 3.9. Use `Optional[...]` or add the future import.
- No workflow-policy changes are required. Update only the factual command notes in `skills/entomokit-workflow/references/command-profiles.md` for the new `--val-csv` and clean report/dry-run behavior.

## Global Constraints

- Set `version.txt`, `entomokit/_version.py`, the `entomokit/main.py` fallback, `setup.py`, and the version tests to `0.7.2`.
- `clean` never deletes input images. `drop` means "not written to `cleaned_images/`"; `--overwrite` is the only operation that deletes output-directory contents.
- `clean` decisions must not depend on thread scheduling, hash-completion order, filesystem iteration order, or dictionary insertion order.
- All input paths in reports are POSIX paths relative to `--input-dir`. All output paths and existing-output references are POSIX paths relative to `--out-dir`.
- `filter_decisions.csv` rows, `clean_manifest.json["files"]`, and the final per-file log lines are sorted by `input_path`.
- `clean` reports and the log summary must be derived from the same finalized decision-row list. Do not maintain a second counting path.
- Emit per-file log lines after the write/promotion phase from the finalized rows. Do not keep the old worker-time prose lines, because a promotion can invalidate them.
- `clean --dry-run` must not create `cleaned_images/`, write images, or delete existing output. Reject `--dry-run --overwrite`. A dry run on a non-empty `--out-dir` requires `--resume`.
- `clean --resume` keeps existing output files and never rewrites or deletes them. The cumulative `clean_manifest.json["index"]` is the primary matching source; output-file hashing is the fallback for inputs the index did not match.
- `clean_manifest.json["index"]` is cumulative and must not be lost when a later run covers a subset of inputs. `files` is current-run only and MUST NOT be used as a resume index. `index` is written on every real run, even when `files` is empty. Index rows for dropped duplicates map to the surviving representative output and keep their own source hashes, so transitive group members remain resumable. Prune index rows whose output file is missing, with a warning.
- A manifest is usable only when it is schema-valid and not corrupt. Ignore unsupported `schema_version`, malformed JSON, and invalid rows with a warning. `output_path` must resolve inside `--out-dir`; missing or unreadable outputs are ignored with a warning.
- A dry-run manifest is usable for its copied `index` only; its `files` are not outputs. Never treat dry-run `files` as existing outputs.
- `dedup_mode="none"` without a usable index/manifest must not match by mirrored output path; retain the input and warn that exact resume matching requires a manifest.
- A failed image write must remove the partial target before any promotion attempt. No half-written file may remain in `cleaned_images/`.
- Report files and `clean_manifest.json` are written atomically. The log summary is appended only after the reports are complete.
- SAM3's vendored `src/sam3/model/sam3_image_processor.py` must not be edited. The new threshold is passed through the existing constructor.
- Otsu and GrabCut scores remain `1.0`; `--sam3-confidence-threshold` is ignored for them and `--post-confidence-threshold` has no practical effect in the documented `0..1` range.
- AutoMM's internal random split must still be used when `--val-csv` is omitted. Do not implement a new splitting algorithm.
- `--val-csv` validation must fail before `fit()` when the CSV is missing, malformed, empty, references missing images, contains duplicate image rows, contains labels absent from training, or overlaps the training paths.
- Do not tighten the existing training-CSV contract beyond column presence and path resolution. Full existence/duplicate/label/overlap checks apply to the validation CSV only.
- Command documentation must update both `docs/commands/<command>.md` and `docs/commands/<command>.cn.md`; `tests/test_docs.py` enforces option coverage in both languages.
- Do not edit historical plans or completed specifications. Update only the current master design at `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`, and state architecture-level contracts there rather than duplicating parameter tables.
- Do not add dependencies, a changelog, a documentation generator, or a second CLI registry.
- Do not modify workflow policy in `skills/entomokit-workflow`; only correct the factual command notes listed in Task 5.
- Do not commit, tag, push, or publish without separate operator approval.
- Python 3.9 compatibility: use `Optional[...]` in `src/cleaning/processor.py` and `src/segmentation/processor.py`, which do not import `from __future__ import annotations`. Do not copy `Path | None` examples into those modules.

## Review Focus

- The same input set produces the same `filter_decisions.csv` rows in the same sorted order, group IDs, representatives, output names, and output file set at `--threads=1`, `8`, and `32`, and across repeated runs.
- A pHash chain `A~B`, `B~C`, `A!~C` forms one group, not two pairs; this intentional behavior change is recorded in `clean`'s Version Notes.
- A duplicate group spanning `train/`, `val/`, and `test/` keeps the `test/` member, then `val/`, then `train/`; without split segments it keeps the lexicographically smallest relative path.
- MD5 and pHash match references are relative paths, never basenames; same-named files in different class directories remain distinguishable.
- A representative write failure removes the partial output, promotes the next member, and a failed MD5 registration cannot cause a later real image to be dropped.
- The final per-file log lines, `clean_manifest.json` counts, and `filter_decisions.csv` agree after promotions.
- A+B → A resume → B resume keeps B `existing` through the cumulative index; B is not re-written with a `_1` suffix.
- A previous md5-only manifest with current phash mode falls back to output hashing for outputs that lack a usable phash index row.
- A transitive group `A~B~C` where only C was retained still matches B on a later resume through B's own index row (status `dropped`) pointing at C's output.
- `dedup_mode="none"` without a manifest does not mark a same-stem, different-extension input as existing.
- `retained_images.csv` per-class retention is computed from `output_path`, not `input_path`; multiple inputs may map to one output, and `n_existing` counts inputs rather than unique outputs.
- Promotion recomputes `reason`, `distance`, `kept_scope`, and `matched_scope`; it does not leave references or distances tied to a failed representative.
- `--dry-run` writes the same decision rows and would-be `output_path` values as a real run with no write failures; only the manifest's `dry_run` flag and `output_files.cleaned_images=null` differ. No image files are created.
- An empty input directory still writes zero-row reports and a valid manifest with `n_input=0`.
- `--resume` matches an input whose output was resized/converted using the cumulative index's source hash. Without a usable index/manifest, the fallback warns when MD5 output-hash matching is unreliable.
- `--resume` reports existing-output matches with `matched_scope=existing_output` and output-relative paths; existing outputs are never rewritten.
- Error rows preserve `group_id`/`group_size` when the failure occurred during writing; only hash/decode errors have empty group fields.
- SAM3's model threshold actually reaches `Sam3Predictor`; setting `--sam3-confidence-threshold 0.2` produces masks with scores in `(0.2, 0.5]` that the old hard-coded `0.5` would have dropped.
- Default SAM3 runs (`sam3=0.5`, `post=0.0`) do not emit a no-op warning. An explicit `0 < post < sam3` does warn and does not error.
- The no-mask reason names both thresholds, and Otsu/GrabCut output is unchanged by `--sam3-confidence-threshold`.
- AutoMM receives `tuning_data` exactly when `--val-csv` is supplied; no `tuning_data` key is present otherwise.
- Train/validation overlap after `Path.resolve()` and case normalization, duplicate validation rows, empty validation, missing images, and unknown validation labels fail before `fit()`.
- The validation CSV path, SHA256, image count, and overlap result appear in `out-dir/logs/log.txt` before `fit()` is called.
- `tests/test_docs.py` passes with both language documents updated.
- The implementation imports and compiles under Python 3.9 syntax (no PEP 604 unions in files without the future import).

## File Map

| Area | Files | Responsibility |
|---|---|---|
| Clean decision core | `src/cleaning/processor.py` | Hash phase, grouping, representative selection, deterministic naming, write/promotion, report generation, resume matching |
| Clean CLI | `entomokit/clean.py` | `--dry-run`, output-directory handling, summary printing |
| CLI schema | `entomokit/cli_schema.py` | Honor `_value_type` on custom argparse types so `--threads` stays integer and the new threshold options stay float in the runtime schema |
| Clean tests | `tests/test_clean_dedup_manifest.py` (new), `tests/test_clean_resume.py` (new), `tests/test_clean_cli_manifest.py` (new), `tests/test_clean_recursive.py`, `tests/test_clean_padding.py`, `tests/test_resume_flags.py` | Determinism, groups, reports, dry-run, resume, write-failure promotion, CLI namespaces |
| Augment manifest | `src/augment/service.py`, `tests/test_augment_manifest.py` (new) | Add existing per-augmentation records to the manifest |
| SAM3 wrapper | `src/sam3_wrapper.py`, `tests/test_sam3_wrapper.py` | Pass the model threshold to `Sam3Predictor` |
| Segmentation | `src/segmentation/processor.py`, `src/metadata.py`, `entomokit/segment.py`, `tests/test_segmentation.py`, `tests/test_metadata.py`, `tests/test_annotation_formats.py`, `tests/test_segment_cli_thresholds.py` (new), `tests/test_resume_flags.py` | Two thresholds, range validation, compatibility alias, logging, COCO per-mask score, CLI namespaces |
| Classification CLI | `entomokit/classify/train.py` | Add `--val-csv` and pass it through |
| Classification trainer | `src/classification/trainer.py` | Validation loading, checks, `tuning_data`, provenance logging, assets metadata backfill |
| Classification tests | `tests/test_classify_trainer.py`, `tests/test_resume_flags.py` | `tuning_data`, overlap, duplicate rows, label checks, log hash, CLI namespaces |
| Command docs | `docs/commands/clean.md`, `clean.cn.md`, `segment.md`, `segment.cn.md`, `classify-train.md`, `classify-train.cn.md`, `augment.md`, `augment.cn.md` | Bilingual parameter, output, note, and Version Notes updates |
| Completion | `entomokit/completion.py`, `tests/test_main_cli.py` | Omit deprecated actions from generated shell completion while keeping them in help/CLI |
| Skill notes | `skills/entomokit-workflow/references/command-profiles.md` | Factual notes for `--val-csv` and clean reports/dry-run; no workflow policy change |
| Master design | `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` | Current architecture-level contracts for clean reports, segment thresholds, and train validation |
| Version | `version.txt`, `entomokit/_version.py`, `entomokit/main.py`, `setup.py`, `tests/test_package_version.py`, `tests/test_main_cli.py`, `README.md`, `README.cn.md` | Set and verify `0.7.2` |

## Interfaces

### Clean decision row

```python
from typing import Optional

@dataclass
class CleanDecision:
    input_path: str           # POSIX path relative to --input-dir
    status: str               # retained | dropped | existing | error
    reason: str               # "" | duplicate_md5 | duplicate_phash |
                              # duplicate_existing_output | unidentified_image |
                              # hash_error:<message> | write_error:<message> |
                              # group_write_failed
    kept_path: str            # surviving representative; empty only when no survivor exists
    kept_scope: str           # input | existing_output | ""
    matched_path: str         # nearest matching member; empty when there is no match
    matched_scope: str        # input | existing_output | ""
    distance: Optional[int]   # pHash Hamming distance; None for MD5/none
    group_id: str             # g0001, g0002, ...; empty only for hash/decode errors
    group_size: int           # group members, including existing outputs; 0 only for hash/decode errors
    md5: str                  # pixel-content MD5 hex; empty when not computed
    phash: str                # zero-padded 64-bit pHash hex; empty when not computed
    output_path: str          # POSIX path relative to --out-dir; empty when not written
```

### Clean report files

Reports are written to the parent directory of `log_path` (the command's `--out-dir`), never inside `cleaned_images/`.

`filter_decisions.csv` columns, in order, one row per input sorted by `input_path`:

```text
input_path,status,reason,kept_path,kept_scope,matched_path,matched_scope,distance,group_id,group_size,md5,phash,output_path
```

`retained_images.csv` columns, sorted by `input_path`; rows are `status="retained"` and `status="existing"`:

```text
input_path,status,output_path
```

`dropped_images.csv` columns, sorted by `input_path`; rows are `status="dropped"` only:

```text
input_path,reason,matched_path,matched_scope,distance,group_id
```

`clean_manifest.json` keys:

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

`files` is the current run's sequence of `filter_decisions.csv` rows, with `distance` as `int | null` and `group_size` as an integer. `index` is the cumulative resume index: one row per known input path, keyed by `input_path`, pointing at the canonical output path with the source hashes used for matching. Its `status` may be `retained`, `existing`, or `dropped`; dropped rows preserve their own source hashes but point at the surviving representative output. `index` is updated on real runs and copied unchanged on `--dry-run`. In `--dry-run`, `output_files.cleaned_images` is `null`.

### Clean statuses and reasons

- `retained`: the input is the group representative and a new output is written (or would be written under `--dry-run`). `kept_path` is the input itself, `kept_scope="input"`, `output_path` is the new output.
- `dropped`: a non-representative duplicate in a normal (non-existing) group. `reason` is `duplicate_md5` when the row shares an MD5 with a group member, otherwise `duplicate_phash`. `kept_path` is the group representative with `kept_scope="input"`; `matched_path` is the nearest member with `matched_scope="input"`; `output_path` is empty.
- `existing`: the input matches an existing output under `--resume`; no new file is written. `reason="duplicate_existing_output"`, `kept_path` and `matched_path` point to the canonical existing output, `kept_scope="existing_output"`, `matched_scope="existing_output"`, and `output_path` is the canonical existing output path. `distance` is the pHash distance to the canonical existing output when matching used pHash, and empty for MD5/`none` matching. `md5`/`phash` describe the current input, not the existing output. `group_id` is assigned by canonical output path order; `group_size` is the number of current inputs mapping to that canonical output plus one for the existing output.
- `error`, hash/decode failure: `reason` is `unidentified_image` or `hash_error:<message>`; `group_id` is empty, `group_size=0`, and all path fields are empty.
- `error`, write failure with a surviving group member: `reason="write_error:<message>"`; `group_id`/`group_size` are preserved; `kept_path` points to the surviving representative and `kept_scope` is that representative's scope; `matched_path`/`matched_scope` are empty; the failed member's `output_path` is empty.
- `error`, group exhausted: `reason="group_write_failed"`; `group_id`/`group_size` are preserved; `kept_path`, `kept_scope`, `matched_path`, `matched_scope`, and `output_path` are empty.
- `kept_path` and `matched_path` are relative to `--input-dir` unless they point to an existing output, in which case they are relative to `--out-dir` and their scope is `existing_output`. `output_path` is always relative to `--out-dir`.
- `retained_images.csv` is input-centric: one row per rendered input, so multiple `existing` rows may point to the same output. Per-class retention counts by `output_path`; `n_existing` counts inputs, not unique outputs.

### `ImageCleaner`

```python
class ImageCleaner:
    def __init__(..., dry_run: bool = False, resume: bool = False) -> None: ...

    def process_directory(
        self,
        input_dir: Optional[str] = None,
        log_path: Optional[str] = "log.txt",
    ) -> dict:
        """Return counts and report paths."""
```

`process_directory` derives `report_dir = Path(log_path).parent` when `log_path` is provided, otherwise `self.output_dir.parent`; it loads the cumulative `index` from `report_dir/clean_manifest.json` when `resume=True` and writes reports there. The return value keeps existing keys (`total`, `processed`, `skipped`, `errors`) for compatibility and adds `retained`, `dropped`, `existing`, and `reports`. `processed` mirrors `retained`; `skipped` mirrors `dropped`.

### `segment` CLI

```text
--sam3-confidence-threshold FLOAT   default 0.5, range 0..1, SAM3/SAM3-bbox only
--post-confidence-threshold FLOAT   default 0.0, range 0..1, post-filter
--confidence-threshold FLOAT        default 0.0, deprecated alias for the post-filter
```

Resolution in `run()`:

```python
post = args.post_confidence_threshold
legacy = args.confidence_threshold
if post == 0.0 and legacy > 0.0:
    post = legacy
    logger.info(
        "--post-confidence-threshold is 0.0 (unset); using deprecated "
        "--confidence-threshold=%s",
        legacy,
    )
elif post > 0.0 and legacy > 0.0 and post != legacy:
    logger.warning("--post-confidence-threshold wins over deprecated --confidence-threshold")
```

The effective SAM3 filter is `max(sam3_confidence_threshold, post_confidence_threshold)` with model-side `>` and post-side `>=`. The post threshold is a no-op when it is at or below the model threshold; warn only when `0 < post < sam3`. `0.0` means "unset" for alias resolution; an explicit `--post-confidence-threshold 0.0` does not override a non-zero legacy value.

### `classify train` CLI

```text
--val-csv PATH   optional; CSV with image and label columns; image paths relative to --images-dir
```

When supplied, pass `tuning_data=val_df` to `fit()` and do not rely on AutoMM's internal split.

## Task 1: Build the deterministic clean decision pipeline and reports

**Files:**
- Modify: `src/cleaning/processor.py`
- Create: `tests/test_clean_dedup_manifest.py`
- Create: `tests/test_clean_resume.py`
- Modify: `tests/test_clean_recursive.py`, `tests/test_clean_padding.py` for the new constructor/return contract and the obsolete promotion test.

**Interfaces:**
- Consumes: `iter_files`, `compute_md5`, `compute_phash`, `phash_to_int`, `hamming_distance`, `clean_filename`, `ensure_unique_prefix`.
- Produces: one `CleanDecision` per input, deterministic group IDs, deterministic output names, `retained_images.csv`, `dropped_images.csv`, `filter_decisions.csv`, and `clean_manifest.json`.

- [ ] **Step 1: Write the failing decision-contract tests**

Create `tests/test_clean_dedup_manifest.py` with fixtures that write small RGB images into nested directories. Cover at least:

```python
def test_filter_decisions_has_one_row_per_input(tmp_path):
    # three inputs, dedup_mode="none"
    # assert len(rows) == 3, all statuses == "retained"
    # assert rows are sorted by input_path
    # assert retained_images.csv has 3 rows and dropped_images.csv has 0 data rows


def test_md5_drop_uses_relative_input_and_match_paths(tmp_path):
    # train/beetles/same.png and test/beetles/same.png have identical pixel content
    # (the fixture writes the same image in both locations)
    # assert drop row input_path == "train/beetles/same.png"
    # assert matched_path == "test/beetles/same.png" (held-out-first)
    # assert matched_scope == "input"
    # assert the md5 column is the same non-empty pixel MD5 on both rows


def test_phash_group_is_transitive(tmp_path):
    # Build A~B and B~C but A!~C using controlled pHash values or monkeypatched compute_phash.
    # assert one group_id with group_size == 3, one retained, two dropped
    # assert each dropped row records reason and distance


def test_group_representative_prefers_test_over_val_over_train(tmp_path):
    # same image in train/, val/, test/
    # assert retained input_path startswith "test/"


def test_group_representative_falls_back_to_path_order(tmp_path):
    # same image in b/ and a/, no split segments
    # assert retained input_path == "a/..."


def test_output_names_are_assigned_in_sorted_path_order(tmp_path):
    # a.jpg and a.png in the same directory, dedup_mode="none"
    # assert the lexicographically first input gets "a.jpg" and the other gets "a_1.jpg"


def test_clean_is_deterministic_across_threads(tmp_path):
    # run the same fixture with threads=1, 8, 32 into separate out dirs
    # rerun the same command a second time at one thread count
    # compare filter_decisions.csv rows exactly, including their order
    # compare manifest files/counts after ignoring the run-level "threads" field
    # compare relative output file sets


def test_empty_input_writes_zero_row_reports(tmp_path):
    # empty input directory
    # assert all four report artifacts exist, n_input == 0, files == [], index == []


def test_report_counts_satisfy_the_status_invariant(tmp_path):
    # mixed fixture with retained, dropped, and an invalid image
    # assert n_input == n_retained + n_dropped + n_existing + n_errors
```

Also add these focused contract assertions:

- `test_manifest_contains_cumulative_index_with_dropped_rows`: the manifest has `index`, and a transitive group's dropped rows point at the surviving output.
- `test_existing_md5_match_has_empty_distance`: `md5+phash` resume matched by MD5 reports an empty `distance`.
- `test_promotion_recomputes_reason_and_distance`: promotion changes a dropped row from `duplicate_md5` to `duplicate_phash` when the surviving member is only pHash-similar.
- `test_resume_falls_back_to_output_hash_for_indexed_output`: an indexed output is still eligible for decoded-output hash fallback.
- `test_resume_rejects_truncated_manifest_output`: an indexed output that cannot be fully decoded is ignored and the input is regenerated.

Add regression tests for the old MD5-registration leak and write-failure promotion:

```python
def test_write_failure_promotes_next_group_member(tmp_path, monkeypatch):
    # two identical images; make the first representative's save fail once
    # assert the second image is written, the first is reported as an error,
    # and no partial file remains at the failed target


def test_md5_failure_does_not_drop_later_duplicate(tmp_path, monkeypatch):
    # three identical images; make the first save fail
    # assert one of the remaining images is written and is not silently dropped
    # because of a stale MD5 registration


def test_group_write_failure_preserves_group_membership(tmp_path, monkeypatch):
    # three identical images; make every save fail
    # assert the failed representative is write_error, the other members are
    # group_write_failed, and all preserve group_id and group_size
```

Create `tests/test_clean_resume.py`:

```python
def test_resume_uses_previous_manifest_source_hash(tmp_path):
    # run clean once with a resized output, then resume the same input
    # assert the input is status="existing", no _1 output is created


def test_resume_reports_existing_output_relative_paths(tmp_path):
    # assert matched_scope/kept_scope are "existing_output" and paths are out-dir relative


def test_resume_ignores_missing_manifest_output_and_warns(tmp_path, caplog):
    # delete an output referenced by the previous manifest
    # assert it is ignored and a warning is logged


def test_resume_none_mode_matches_previous_manifest_without_hashes(tmp_path):
    # dedup_mode="none"; second run marks existing by manifest index input_path


def test_resume_none_without_manifest_does_not_match_mirrored_path(tmp_path, caplog):
    # a.jpg first run; a.png second run with no manifest
    # assert a.png is retained, not marked existing against a.jpg, with a warning


def test_resume_without_manifest_falls_back_to_output_hashing(tmp_path, caplog):
    # delete the manifest, rerun; assert fallback behavior and a warning when
    # MD5 matching may be unreliable after resizing


def test_resume_cumulative_index_survives_subset_runs(tmp_path):
    # run A+B, then resume A only, then resume B only
    # assert B is status="existing" and no B_1 output is created


def test_resume_transitive_group_member_matches_by_own_hash(tmp_path):
    # A~B~C transitive group, only C retained
    # later resume with B only; B's dropped-status index row maps to C's output
    # assert B is status="existing", not re-written


def test_resume_fallback_hashes_output_without_current_mode_hash(tmp_path):
    # first run md5-only, then resume phash; manifest has no usable phash index
    # assert the existing output is hashed as fallback and matched


def test_resume_ignores_dry_run_or_unsupported_manifest(tmp_path, caplog):
    # dry_run=true manifest and schema_version=999 manifest are ignored with warnings


def test_resume_rejects_manifest_output_path_outside_out_dir(tmp_path, caplog):
    # manifest output_path="../escape.jpg", an absolute path, and "other.jpg"
    # outside cleaned_images/ are ignored with warnings
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_clean_dedup_manifest.py tests/test_clean_resume.py -q
```

Expected: import/attribute failures or missing-report assertions, not an unrelated environment error.

- [ ] **Step 3: Split hashing from decision-making**

Refactor `process_one_to` into explicit phases:

```python
def _hash_images(files, input_dir, dedup_mode, threads) -> list[ImageRecord]:
    """Read every input, compute requested hashes, and return records in sorted path order."""

def _decide_groups(records, dedup_mode, phash_threshold, existing_outputs) -> list[CleanDecision]:
    """Group records, choose representatives, and assign keep/drop/existing/error rows."""

def _assign_output_paths(decisions, output_dir, out_image_format) -> None:
    """Assign deterministic output names in input-path order."""

def _write_retained(decisions, output_dir, out_short_size, out_image_format, keep_exif, pad_color, threads) -> None:
    """Write retained images and promote group members on write failure."""
```

Rules:

- Hash with `ThreadPoolExecutor(max_workers=threads)`, but store results by input path. Do not decide inside the worker.
- Close every opened image deterministically (`with Image.open(...)` or `img.close()`).
- `dedup_mode="none"` does not need hashes; every input is retained.
- `md5` groups by the pixel-content MD5 returned by `compute_md5`. Document in code and docs that this is not a file-byte MD5.
- `phash` groups with union-find over every pair `distance <= phash_threshold`.
- `md5+phash` first groups by MD5, then runs pHash union-find over the MD5-group representatives and merges the member sets.
- Errors from hashing or decoding become `status="error"` with `group_id=""` and `group_size=0`, and never become representatives.
- `phash` only: a missing or invalid pHash becomes `status="error"` with `reason="hash_error:phash"`.
- `md5+phash`: if MD5 succeeds but pHash fails, keep the record as an MD5-only candidate; do not fail the image.
- `--resume` existing outputs are loaded in `process_directory`, not `__init__`. Remove the `__init__` call to `_prepopulate_hashes`; existing-output discovery becomes part of `process_directory`.
- Load `report_dir/clean_manifest.json` only when it is schema-valid and not corrupt. Ignore unsupported `schema_version`, malformed JSON, invalid rows, and `output_path` values that are absolute, non-POSIX-relative, or resolve outside `--out-dir` or outside the `cleaned_images/` output directory, each with a warning. A dry-run manifest may contribute its `index` only; never treat its `files` as outputs.
- The manifest's cumulative `index` is the primary resume source. `index` rows are keyed by `input_path` and carry `output_path`, `status`, `md5`, and `phash`. Match current inputs using the current dedup mode:
  - `md5`: exact source MD5.
  - `phash`: nearest source pHash within `--phash-threshold`, tie-broken by path.
  - `md5+phash`: exact source MD5 first; otherwise nearest source pHash within threshold.
  - `none`: exact `input_path`.
- Ignore index rows whose output file is missing, unreadable, or outside `--out-dir`/`cleaned_images/`, with a warning. If multiple index rows map to one output, use the lexicographically smallest output-relative path as the canonical existing output.
- For inputs not matched by the index, build an output hash map from **all** existing output image files, not only paths absent from the manifest, and match by output hash using the current mode. This is required when an older manifest has a path but no usable hash for the current mode (for example md5-only manifest, current phash mode).
- If no usable index exists, log a warning when MD5 output-hash matching may miss resized/converted outputs. Recommend a fresh run or `--overwrite` when exact resume matching is required.
- `--resume --dedup-mode none`: match by `index`/`files` `input_path` only. Without a usable manifest, do not match by mirrored output path; retain the input and warn that exact resume matching requires a manifest.
- After a real run, merge the current `retained`, `existing`, and `dropped` rows that have a surviving representative into `index`, replacing any previous row for the same `input_path`. A dropped row's index row maps to the surviving representative output and keeps the dropped input's own source hashes, so transitive group members remain resumable. A write-error row with a surviving representative is treated the same way. Remove only rows for hash/decode errors and exhausted-group errors, which have no usable output or hash. Keep prior rows for inputs not present in the current run when their output still exists; prune missing outputs with a warning. A dry-run copies the previous `index` unchanged and adds no would-be outputs.
- Under `--resume`, current inputs are matched directly against the cumulative index/output hashes using the current mode. Matched inputs become `status="existing"`; unmatched inputs form normal groups. Existing outputs are never deleted or rewritten. If multiple existing outputs match one input, the current-mode tie-break selects the canonical output (MD5: path; pHash: distance then path).
- pHash grouping is an accepted O(n²) pairwise pass for `0.7.2`; do not add a BK-tree or a new dependency.

- [ ] **Step 4: Implement representative and match selection**

Use these deterministic rules:

```python
def split_rank(relative_path: str) -> int:
    """test=0, val=1, train=2, no recognized split=3."""
```

- Split detection is case-insensitive and matches exact path components: `train`; `val`, `valid`, `validation`; `test`. If a path contains more than one, use the highest-priority segment.
- For groups without existing outputs, choose the minimum `(split_rank, relative_path)` input member as representative.
- For inputs matched to existing outputs, the existing output is the canonical representative; every matching input becomes `existing` and is not rewritten. If multiple existing outputs match one input, use the current-mode tie-break (MD5: path; pHash: distance then path) to choose the canonical `kept_path`/`output_path`.
- For each non-representative input, choose the nearest group member:
  - `md5`: the representative, `distance=None`.
  - `phash`: minimum Hamming distance; tie-break by relative path.
  - `md5+phash`: if the member shares an MD5 with a group member, use `duplicate_md5` and `distance=None`, matching the representative when it shares the MD5, otherwise the lexicographically smallest same-MD5 member. Otherwise use `duplicate_phash` with the minimum pHash distance and path tie-break.
- `matched_path` is the nearest member, not necessarily the representative. This preserves the actual edge that pulled the record into the group even when transitive distance exceeds the threshold.
- Assign `group_id` after representatives are fixed, sorted by `(split_rank, representative path)`; existing matches use their output-relative canonical path.
- `retained` rows use `kept_path=input_path`, `kept_scope="input"`.
- `dropped` rows use `kept_scope="input"` and `matched_scope="input"`; they only occur in normal input groups.
- `existing` rows use `status="existing"`, `reason="duplicate_existing_output"`, `kept_path`/`matched_path`/`output_path` set to the canonical existing output, and both scopes set to `existing_output`.
- `error` rows that wrote nothing have empty path fields. A write-error row with a surviving representative keeps `kept_path`/`kept_scope` from that survivor and leaves `matched_path`/`matched_scope` empty.
- For `dedup_mode="none"`, only manifest/index `input_path` matching is allowed; do not infer existing status from a mirrored output path.

- [ ] **Step 5: Implement deterministic output naming**

- Sort retained input decisions by `input_path`.
- Assign names in that order, using the existing `clean_filename` and per-directory uniqueness rules.
- Include existing output names from `--resume` in the used-name set.
- Never let a worker choose a name or append a suffix based on completion order.
- Store the final path relative to `--out-dir` in `output_path` (for example `cleaned_images/beetles/a.jpg`).

- [ ] **Step 6: Implement write and promotion**

- Write retained images in parallel to their preassigned paths.
- On a write failure, best-effort remove the partial target (`dst.unlink(missing_ok=True)`), then mark the member `error` with `reason=write_error:<message>`, preserving `group_id`/`group_size`.
- Collect all parallel write results before promoting. Process promotions serially in `group_id` order, then representative order within each group.
- Promote the next group member, assign its output name from the deterministic name allocator, and write it. A promoted `dropped` row becomes `retained` with an empty reason and the new `output_path`.
- Repeat until a member writes successfully or the group is exhausted.
- If the group is exhausted, mark the remaining members as `error` with `reason=group_write_failed`, preserving `group_id`/`group_size` and clearing path fields.
- After promotion, update every group member's `kept_path`/`kept_scope`/`matched_path`/`matched_scope` to reference the surviving representative; recompute `reason` and `distance` whenever `matched_path` changes. Do not leave references or distances tied to a failed member.
- Do not write input files or delete input files at any point.

- [ ] **Step 7: Write the reports from one row list**

Add a single report writer used by both normal and dry-run paths:

```python
def write_clean_reports(
    decisions: list[CleanDecision],
    report_dir: Path,
    log_path: Optional[Path],
    summary: dict,
    dry_run: bool,
) -> dict:
    """Write retained_images.csv, dropped_images.csv, filter_decisions.csv,
    clean_manifest.json atomically, then append the per-file lines and summary
    to log_path when it is not None."""
```

Requirements:

- `filter_decisions.csv` has exactly one row per input, including errors, sorted by `input_path`.
- `retained_images.csv` contains `status="retained"` and `status="existing"` rows, sorted by `input_path`.
- `dropped_images.csv` contains only `status="dropped"` rows, sorted by `input_path`.
- `clean_manifest.json` contains the run-level fields, a `files` array equal to the current CSV rows, and the cumulative `index` array used by resume.
- Compute `n_input`, `n_retained`, `n_dropped`, `n_existing`, and `n_errors` from the same `decisions` list; tests assert `n_input == n_retained + n_dropped + n_existing + n_errors`.
- `n_retained` counts only new outputs (`status="retained"`); `n_existing` counts input matches (`status="existing"`), not unique existing outputs. `retained_images.csv` therefore has `n_retained + n_existing` rows, with possible repeated `output_path` values.
- Merge `index` from the previous valid manifest as specified in Step 3, using the finalized rows after promotion; a dry-run copies the previous `index` unchanged and adds no would-be outputs.
- Write each CSV/JSON to a temporary file in `report_dir` and `os.replace` it into place.
- Append the log summary with the same counts and the report paths after the reports are complete.
- Emit the final per-file log lines from `decisions` in `input_path` order after promotion, for example:
  `retained: <input> -> <output>`, `existing: <input> -> <output>`,
  `dropped: <input> -> <matched> (<reason>, distance=<n|->)`, `error: <input> (<reason>)`.
- Remove the old worker-time duplicate/error prose lines so the log and reports cannot disagree.
- If `log_path` is `None`, write reports but skip the log lines and summary.

- [ ] **Step 8: Run the focused and existing clean tests**

Run:

```bash
python -m pytest tests/test_clean_dedup_manifest.py tests/test_clean_resume.py tests/test_clean_recursive.py tests/test_clean_padding.py -q
```

Expected: PASS. The report, determinism, and resume tests must pass before moving to the CLI.

Completion note: keep workspace changes uncommitted until the operator approves the full change set.

## Task 2: Add `clean --dry-run` and augment manifest file rows

**Files:**
- Modify: `entomokit/clean.py`
- Modify: `src/augment/service.py`
- Create: `tests/test_clean_cli_manifest.py`
- Create: `tests/test_augment_manifest.py`
- Modify: `tests/test_resume_flags.py`

**Interfaces:**
- Consumes: `ImageCleaner(dry_run=..., resume=...)`, `write_clean_reports`.
- Produces: a CLI-level dry-run contract and a per-augmentation `files` array in `augment_manifest.json`.

- [ ] **Step 1: Write the failing CLI and manifest tests**

```python
def test_clean_parser_accepts_dry_run_default_false():
    # parser.parse_args(["clean", "--input-dir", "in", "--out-dir", "out"]).dry_run is False

def test_clean_dry_run_writes_reports_without_images(tmp_path):
    # run clean --dry-run
    # assert filter_decisions.csv, retained_images.csv, dropped_images.csv,
    #        clean_manifest.json exist; assert no cleaned_images/ directory
    # assert manifest["output_files"]["cleaned_images"] is None

def test_clean_dry_run_rejects_overwrite(tmp_path, capsys):
    # assert SystemExit code 1 and a clear error message

def test_clean_dry_run_requires_resume_on_nonempty_out_dir(tmp_path):
    # create out/log.txt, run --dry-run without --resume, assert SystemExit

def test_clean_real_run_after_dry_run_uses_resume(tmp_path):
    # dry-run into out/, then real run with --resume, assert images are written

def test_clean_dry_run_preserves_previous_index(tmp_path):
    # real run, then dry-run with --resume, then real run with --resume
    # assert prior outputs remain existing and no duplicates are written

def test_augment_manifest_includes_per_file_rows(tmp_path, monkeypatch):
    # run run_augment on two images
    # assert manifest["files"] has 2 entries with original/augmented/copy_index
```

- [ ] **Step 2: Implement the dry-run CLI contract**

In `entomokit/clean.py`:

- Add `--dry-run` with `action="store_true"`.
- Add a positive-int type for `--threads` (reject `0` and negative values).
- Set `_positive_int._value_type = "int"`, and update `entomokit/cli_schema.py` `_infer_value_type` to honor `_value_type` on custom argparse types. This keeps `--threads` typed as integer in the runtime schema and parameter guard.
- Reject `--dry-run --overwrite` before `check_output_dir`; print a clear error and `sys.exit(1)`.
- Keep `check_output_dir(out_dir, args.resume, args.overwrite)`.
- Call `save_log(out_dir, args)` as today.
- Create `out_dir / "cleaned_images"` only when `not args.dry_run`.
- `ImageCleaner.__init__` must skip `self.output_dir.mkdir(...)` when `dry_run=True`; `process_directory` must not create output subdirectories in dry-run.
- In dry-run, decode each would-be retained image and apply resize/pad in memory, but skip `mkdir`/`save`; record decode errors exactly as a real write would.
- Pass `dry_run=args.dry_run` and `resume=args.resume` to `ImageCleaner`.
- Print a summary using the new counts and the report locations. Use `would write N new image(s)` for dry-run and `wrote N new image(s)` for a real run; singular/plural must match the count.

Do not make `--dry-run` itself delete or rewrite existing images. With `--resume`, existing outputs may participate in matching, but no new images are written. A dry-run copies the previous valid `index` unchanged into the new manifest and adds no would-be outputs; a later real run into the same directory requires `--resume` and may use that copied `index`. Document this flow.

- [ ] **Step 3: Add augment file rows to the manifest**

`run_augment` already builds `augmented_images`. Add it to the manifest:

```python
manifest = {
    ...existing run-level fields...,
    "files": augmented_images,
}
```

Do not change the `augmented_images` construction or the existing count fields. Add a test asserting the key exists, is a list, and preserves `original`, `augmented`, and `copy_index`. Document that `original` is relative to `--input-dir` and `augmented` is relative to `out-dir/images/`.

- [ ] **Step 4: Update the manual CLI namespaces**

Add the new attributes to `_clean_args` in `tests/test_resume_flags.py`:

```python
dry_run=False,
```

- [ ] **Step 5: Run the focused tests**

```bash
python -m pytest tests/test_clean_cli_manifest.py tests/test_augment_manifest.py tests/test_resume_flags.py -q
```

Expected: PASS.

## Task 3: Split SAM3's model threshold from the post-filter

**Files:**
- Modify: `src/sam3_wrapper.py`
- Modify: `src/segmentation/processor.py`
- Modify: `entomokit/segment.py`
- Modify: `entomokit/completion.py`
- Modify: `tests/test_sam3_wrapper.py`
- Modify: `tests/test_segmentation.py`
- Create: `tests/test_segment_cli_thresholds.py`
- Modify: `tests/test_resume_flags.py`
- Modify: `tests/test_main_cli.py`

**Interfaces:**
- `SAM3Wrapper(..., confidence_threshold: float = 0.5)`.
- `SegmentationProcessor(..., sam3_confidence_threshold: float = 0.5, post_confidence_threshold: Optional[float] = None, confidence_threshold: float = 0.0)`.
- CLI options `--sam3-confidence-threshold`, `--post-confidence-threshold`, and deprecated `--confidence-threshold`.

Note: the legacy `src/segmentation.py` module is not imported by the package or CLI (`src/segmentation/__init__.py` exports `src.segmentation.processor`). Do not modify it in this patch.

- [ ] **Step 1: Write the failing threshold tests**

```python
def test_sam3_wrapper_passes_confidence_threshold_to_predictor(monkeypatch):
    # mock src.sam3.build_sam.Sam3Predictor
    # construct SAM3Wrapper(..., confidence_threshold=0.2)
    # assert Sam3Predictor was called with confidence_threshold=0.2


def test_processor_passes_sam3_threshold_to_wrapper(monkeypatch):
    # mock SAM3Wrapper and assert it receives confidence_threshold=0.2


def test_post_filter_uses_post_confidence_threshold():
    # scores [0.95, 0.45]; post threshold 0.7; assert one mask survives


def test_default_thresholds_do_not_warn(caplog):
    # sam3=0.5, post=0.0; assert no no-op warning


def test_post_threshold_below_sam3_threshold_warns(caplog):
    # sam3=0.5, post=0.3; assert a warning, no exception


def test_no_mask_reason_names_both_thresholds(caplog):
    # sam3=0.2, post=0.0, wrapper returns no masks; assert the reason mentions 0.2


def test_otsu_ignores_sam3_threshold():
    # same Otsu fixture with sam3=0.5 and sam3=0.2 produces identical outputs
```

Also cover CLI parsing and alias resolution:

```python
def test_segment_parser_threshold_defaults_and_ranges():
    # --sam3-confidence-threshold default 0.5
    # --post-confidence-threshold default 0.0
    # --confidence-threshold default 0.0
    # all three reject -0.01 and 1.01

def test_legacy_confidence_threshold_is_used_when_post_is_zero(caplog):
    # post=0.0, legacy=0.7 -> resolved post=0.7 and an info line saying 0.0 is unset

def test_post_wins_when_both_are_nonzero():
    # post=0.6, legacy=0.7 -> resolved post=0.6 and a warning
```

- [ ] **Step 2: Pass the threshold through the wrapper**

In `SAM3Wrapper.__init__`:

- Add `confidence_threshold: float = 0.5`.
- Validate `0 <= confidence_threshold <= 1`.
- Store it.
- Pass it to `Sam3Predictor(self.model, confidence_threshold=confidence_threshold)`.

Do not edit `src/sam3/model/sam3_image_processor.py`; its constructor and `_forward_grounding` already use `confidence_threshold`.

- [ ] **Step 3: Add the two thresholds to `SegmentationProcessor`**

- Keep the existing `confidence_threshold` parameter for compatibility; treat it as the post-filter fallback.
- Add `sam3_confidence_threshold` and `post_confidence_threshold` at the end of the constructor signature to avoid breaking positional callers.
- Use `Optional[float]` for `post_confidence_threshold`; do not use `float | None` in this module.
- Resolve:

```python
self.sam3_confidence_threshold = sam3_confidence_threshold
self.post_confidence_threshold = (
    post_confidence_threshold
    if post_confidence_threshold is not None
    else confidence_threshold
)
self.confidence_threshold = self.post_confidence_threshold  # compatibility attribute
```

- Construct the wrapper with `confidence_threshold=self.sam3_confidence_threshold`.
- Replace the post-filter comparison with `self.post_confidence_threshold`.
- Update the no-mask reason in `_write_computation` to name both thresholds:

```python
reason = (
    "No masks passed the effective confidence threshold "
    f"(sam3={self.sam3_confidence_threshold}, post={self.post_confidence_threshold})"
)
```

  Use that message when the method is SAM3 and either threshold is greater than zero; keep `No masks returned by <method>` when both thresholds are zero.
- Validate both thresholds in `0..1` inside the processor as well as the CLI.
- When the method is SAM3 and `0 < post_confidence_threshold < sam3_confidence_threshold`, `logger.warning(...)` once per processor, not once per image. Do not warn when `post == 0.0`.
- When the method is Otsu/GrabCut and `sam3_confidence_threshold != 0.5`, log that the SAM3 threshold is ignored. Do not log this for the default value.
- Add an optional `score` argument to `COCOMetadataManager.add_annotation` and pass each mask's `score` from `_write_computation` in both COCO unified and separate output. SAM3 writes the model score; Otsu/GrabCut write `1.0`. VOC/YOLO remain score-free. Add tests in `tests/test_metadata.py` and `tests/test_annotation_formats.py`.

- [ ] **Step 4: Add CLI flags and range validation**

In `entomokit/segment.py`:

- Add a `_unit_float` argparse type that rejects values outside `[0, 1]`; set `_unit_float._value_type = "float"` so the runtime schema and parameter guard keep the option typed as float.
- Add `--sam3-confidence-threshold` default `0.5`.
- Add `--post-confidence-threshold` default `0.0`.
- Keep `--confidence-threshold` default `0.0` with help marking it as a deprecated alias.
- Mark the deprecated action with `_deprecated = True`. `entomokit/completion.py` skips actions with this marker, so the alias stays accepted and visible in `--help` but is omitted from generated shell completion.
- Resolve the post value in `run()` as shown in the Interfaces section, including the info log when the legacy alias is used because post is `0.0` (unset).
- Pass both values to `SegmentationProcessor`.
- Log the method, SAM3 threshold, post threshold, and any no-op warning.
- If the method is not SAM3 and the SAM3 threshold differs from its default, log that it is ignored.

- [ ] **Step 5: Update the manual CLI namespace**

Add the new attributes to `_segment_args` in `tests/test_resume_flags.py`:

```python
sam3_confidence_threshold=0.5,
post_confidence_threshold=0.0,
```

- [ ] **Step 6: Run the threshold tests**

```bash
python -m pytest tests/test_sam3_wrapper.py tests/test_segmentation.py tests/test_segment_cli_thresholds.py tests/test_resume_flags.py tests/test_main_cli.py -q
```

Expected: PASS.

## Task 4: Add `classify train --val-csv`

**Files:**
- Modify: `entomokit/classify/train.py`
- Modify: `src/classification/trainer.py`
- Modify: `tests/test_classify_trainer.py`
- Modify: `tests/test_resume_flags.py`

**Interfaces:**
- CLI: `--val-csv PATH` (optional).
- `train(..., val_csv: Optional[Path] = None)`.
- `fit_kwargs["tuning_data"] = val_df` only when a validation CSV is supplied.

- [ ] **Step 1: Write the failing validation tests**

```python
def test_train_passes_tuning_data_when_val_csv_supplied(monkeypatch, tmp_path):
    # fake AutoGluon fit captures kwargs
    # assert kwargs["tuning_data"] is a DataFrame with resolved image paths

def test_train_omits_tuning_data_without_val_csv(monkeypatch, tmp_path):
    # assert "tuning_data" not in fit kwargs

def test_train_rejects_train_val_path_overlap(monkeypatch, tmp_path):
    # same image in train.csv and val.csv, including "./img.jpg" vs "img.jpg"
    # assert ValueError before fit is called

def test_train_rejects_duplicate_validation_rows(...)
def test_train_rejects_missing_validation_images(...)
def test_train_rejects_empty_validation_csv(...)
def test_train_rejects_validation_labels_absent_from_training(...)
def test_train_logs_validation_csv_path_hash_and_count(...)
def test_train_logs_validation_before_fit(...)
def test_train_accepts_absolute_validation_image_paths(...)
def test_train_does_not_require_training_images_to_exist(...)
def test_train_maps_validation_labels_to_training_representation(...)
def test_train_rejects_blank_validation_image(...)
def test_train_rejects_blank_validation_label(...)
def test_classify_train_parser_accepts_val_csv(...)
```

- [ ] **Step 2: Add the CLI option and pass it through**

In `entomokit/classify/train.py`:

- Add `--val-csv` with no default.
- Pass `val_csv=Path(args.val_csv) if args.val_csv else None` to `train()`.
- Do not change `split-csv` or add new splitting logic.

- [ ] **Step 3: Load and validate the validation CSV in the trainer**

Add a small helper in `src/classification/trainer.py`:

```python
def _load_training_frame(
    csv_path: Path,
    images_dir: Path,
    role: str,
    *,
    validate_files: bool,
) -> pd.DataFrame:
    """Read an image,label CSV, resolve image paths, and validate columns.

    Training uses validate_files=False: require columns and resolve paths only.
    Validation uses validate_files=True: also require a non-empty file, existing
    images, and no duplicate resolved image rows. Cross-frame label-subset and
    overlap checks happen in train() after both frames are loaded.
    """
```

Rules:

- Require `image` and `label` columns in both train and validation CSVs.
- Resolve `images_dir / image` for relative values; accept absolute paths.
- Normalize with `Path.resolve()` and `os.path.normcase(...)` for overlap comparison.
- Reject an empty validation CSV.
- Reject duplicate resolved `image` rows within validation in the helper.
- After both frames are loaded in `train()`, reject validation labels not present in the training labels.
- After both frames are loaded in `train()`, reject any resolved path present in both training and validation; report the first few overlaps.
- Do not reject missing training images or duplicate training rows in this release; preserve the existing training-CSV behavior.
- Add `import hashlib` for the validation CSV SHA256.
- Fail before `predictor.fit()`.

- [ ] **Step 4: Pass `tuning_data` and log validation provenance**

- If `val_csv` is supplied:
  - Add `fit_kwargs["tuning_data"] = val_df`.
  - Compute the val CSV file SHA256 with `hashlib.sha256(path.read_bytes()).hexdigest()`.
  - Create `out_dir / "logs"` if needed, then append to `out_dir / "logs" / "log.txt"` **before** `predictor.fit()`:

```text
Validation CSV: <path>
Validation CSV SHA256: <sha256>
Validation images: <n>
Overlap check: passed
```

- If `val_csv` is omitted:
  - Do not add `tuning_data`.
  - Append before `fit()`:

```text
Validation: AutoMM internal random split (no --val-csv)
```

- Keep `train.processed.csv` unchanged. Do not add a per-image validation manifest or per-image validation hashes in this release.
- Missing, corrupt, or unwritable `assets.json` produces a warning only; it must not fail training.
- Keep the existing NVML retry path using the same `fit_kwargs`, including `tuning_data`.
- After `predictor.fit()` returns, call `_backfill_model_assets(model_dir, predictor, elapsed_seconds)` to rewrite the standard `best_score` and `total_train_time` keys in the model's `assets.json` from `predictor._learner` (falling back to measured elapsed time for the latter). This is needed because AutoMM writes assets before those attributes are populated. Add a focused test with a fake `_learner`; do not delete the keys, because `MultiModalPredictor.load()` indexes them.

- [ ] **Step 5: Update the manual CLI namespace**

Add `val_csv=None` to `_train_args` in `tests/test_resume_flags.py`.

- [ ] **Step 6: Run the classification tests**

```bash
python -m pytest tests/test_classify_trainer.py tests/test_resume_flags.py -q
```

Expected: PASS, including the existing seed, resume, NVML, and optimizer tests.

## Task 5: Update bilingual documentation, version, and release verification

**Files:**
- Modify: `docs/commands/clean.md`, `docs/commands/clean.cn.md`
- Modify: `docs/commands/segment.md`, `docs/commands/segment.cn.md`
- Modify: `docs/commands/classify-train.md`, `docs/commands/classify-train.cn.md`
- Modify: `docs/commands/augment.md`, `docs/commands/augment.cn.md`
- Modify: `skills/entomokit-workflow/references/command-profiles.md`
- Modify: `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`
- Modify: `version.txt`, `entomokit/_version.py`, `entomokit/main.py`, `setup.py`
- Modify: `tests/test_package_version.py`, `tests/test_main_cli.py`
- Modify: `README.md`, `README.cn.md` for version examples and the `0.7.2` parameter-change tables.

**Interfaces:**
- Consumes: the implemented CLI contracts from Tasks 1–4.
- Produces: synchronized English/Chinese command docs, current master-design contracts, corrected skill notes, and a consistent `0.7.2` version.

- [ ] **Step 1: Update the clean command documents**

In both `docs/commands/clean.md` and `docs/commands/clean.cn.md`:

- Add `--dry-run` under Parameters with an H3 block.
- Replace "Deduplicated files are omitted from the output; no report file is written" with the four report artifacts.
- Document `retained_images.csv`, `dropped_images.csv`, `filter_decisions.csv`, and `clean_manifest.json`, including their exact columns and that `retained_images.csv` includes both `retained` and `existing` rows. State that it is input-centric: multiple inputs may map to one output, per-class retention must count by `output_path`, and `n_existing` counts inputs rather than unique outputs.
- Document `clean_manifest.json["index"]` explicitly: cumulative, one row per known input, `status` retained/existing/dropped, `input_path`/`output_path`/`md5`/`phash`; `files` is current-run only and is not the resume index; dry-run copies the previous index.
- Document that `md5` in the report is pixel-content MD5, not a file-byte MD5; for `existing` rows, `md5`/`phash` describe the current input, not the existing output.
- Document deterministic grouping, connected components, the `test > val > train` representative priority, and the fallback to path order.
- Document `--resume` behavior: cumulative `clean_manifest.json["index"]` first, output-hash fallback for unmatched inputs, `existing` status, output-relative paths, no rewrites; note the MD5 resize limitation when no manifest exists. State that a subset resume must not lose prior index rows, and that `dedup_mode="none"` without a manifest does not match by mirrored name.
- Document that `--resume` keeps existing outputs as-is; changing `--out-short-size` or `--out-image-format` requires `--overwrite`. State that an existing output in train can keep a duplicate out of test when resuming an already-split tree; use `--overwrite` or clean before splitting to rebalance.
- Document `--dry-run` flow: reports only, no `cleaned_images/`, requires `--resume` on a non-empty out-dir, copies the previous `index` unchanged, and a later real run into the same directory needs `--resume`.
- Document that per-file log lines are emitted from the finalized decisions after writes.
- Add a `0.7.2` Version Notes entry first, describing deterministic transitive group deduplication, the report artifacts and hashes, `--dry-run`, the cumulative resume index, and the `--resume` existing-output change.

- [ ] **Step 2: Update the segment command documents**

In both languages:

- Add H3 blocks for `--sam3-confidence-threshold`, `--post-confidence-threshold`, and the deprecated `--confidence-threshold` alias. Use the actual defaults (`0.5`, `0.0`, `0.0`).
- Document the `0..1` ranges and the effective `max(model, post)` behavior.
- Document that a post threshold strictly between zero and the model threshold warns and has no additional effect; default `0.0` does not warn.
- Document the legacy alias precedence: `--post-confidence-threshold` wins when both are non-zero and different; a zero post value is treated as unset and falls back to the legacy value, with an info log. State that an explicit `--post-confidence-threshold 0.0` does not override a non-zero legacy value.
- Document that Otsu/GrabCut ignore the SAM3 threshold and that their constant `1.0` scores make the post-filter ineffective in the `0..1` range.
- Document that the no-mask reason names both thresholds.
- Document the new COCO per-mask `score`: SAM3 writes its model score, Otsu/GrabCut write `1.0`, and VOC/YOLO have no score field. It applies to unified and separate COCO output. Note that older annotations need `--overwrite` to gain the field; `--resume` skips existing images.
- Add a `0.7.2` Version Notes entry first.

- [ ] **Step 3: Update the classify-train command documents**

In both languages:

- Add an H3 block for `--val-csv`.
- State that its `image` column is relative to the same `--images-dir`.
- State that supplying it disables AutoMM's internal random split and uses it as `tuning_data`.
- Document the path-overlap (resolved and case-normalized), duplicate-row, empty-file, missing-image, and unknown-label checks.
- Document that training-side validation remains unchanged (columns and path resolution only).
- Document the validation CSV SHA256 and image count in `logs/log.txt`, written before `fit()`.
- Replace the unconditional "AutoGluon performs its own internal train/validation split" guidance with a conditional statement.
- Document that `assets.json` is refreshed after `fit()` with `best_score` and `total_train_time`, so `fit_summary()` is populated after reload.
- Add a `0.7.2` Version Notes entry first.

- [ ] **Step 4: Update the augment manifest documentation**

In both languages:

- Document the manifest's new `files` array with `original`, `augmented`, and `copy_index`.
- State that `original` is relative to `--input-dir` and `augmented` is relative to `out-dir/images/`.
- Add a `0.7.2` Version Notes entry first.

**README parameter tables:** In both `README.md` and `README.cn.md`, add a `Parameter Changes in 0.7.2` / `0.7.2 参数变化` H2 after the Commands section with Key-Parameters-style tables (`Parameter | Description | Default`) for:
- `clean --dry-run`;
- `segment --sam3-confidence-threshold`, `--post-confidence-threshold`, and the deprecated `--confidence-threshold`;
- `classify train --val-csv`.

Keep the new H2 outside the Commands H2 so `test_readme_command_index` still sees only command rows. The command docs remain the full parameter reference.

- [ ] **Step 5: Update the current master design**

Update only the relevant sections of `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`:

- Section 5.1 clean: report artifacts, deterministic group semantics, split priority, dry-run, cumulative-index resume.
- Section 5.1 segment: model threshold vs post-filter contract and compatibility alias.
- Section 5.2 classify train: conditional `--val-csv`/`tuning_data` contract and validation provenance.
- Section 10 documentation conventions: no change to the convention itself; ensure new command docs follow it.

State architecture-level contracts and link to the command docs; do not duplicate the full parameter tables.

Do not edit dated historical plans or completed design specs.

- [ ] **Step 6: Correct the workflow skill's factual notes**

In `skills/entomokit-workflow/references/command-profiles.md`:

- In the clean section, note that reports are written to `--out-dir` and that `--dry-run` exists.
- In the classify/split-csv section, note that `--val-csv` can supply AutoMM's `tuning_data` and that internal splitting is used only when it is omitted.
- Do not add new workflow steps, gates, or policy rules.

- [ ] **Step 7: Set the release version**

Set `0.7.2` consistently in:

- `version.txt`
- `entomokit/_version.py`
- the `entomokit/main.py` fallback literal
- `setup.py`
- `tests/test_package_version.py`
- `tests/test_main_cli.py`
- README version examples, if they hard-code a release string

Do not add a changelog. The release notes are the `Version Notes` entries added in Steps 1–4.

- [ ] **Step 8: Run the documentation and version gates**

```bash
python -m pytest tests/test_docs.py -q
python -m pytest tests/test_package_version.py tests/test_main_cli.py -q
python -m entomokit.main clean --help
python -m entomokit.main segment --help
python -m entomokit.main classify train --help
```

Expected:

- `tests/test_docs.py` passes, including `test_command_option_coverage` for both languages.
- Version output is `entomokit 0.7.2`.
- Each help output lists the new options and the deprecated alias.

- [ ] **Step 9: Run the full suite**

```bash
python -m pytest -q
```

Expected: PASS. Record the exact totals in Execution Evidence. If an optional backend is unavailable, record the blocker rather than hiding it with skips or xfail.

## Execution Evidence

- [ ] Baseline: branch `update-v0.7.2`, starting ref `c071c37`, `python -m pytest -q` result recorded before edits.
- [x] Task 1: clean decision/report/resume tests pass, including the sorted-row invariant, empty-input manifest, cumulative subset-resume index, output-hash fallback for a missing current-mode hash, and the `none` no-manifest rule.
- [x] Task 2: dry-run, resume-on-nonempty, augment manifest, and updated CLI namespace tests pass.
- [x] Task 3: SAM3 threshold, default-no-warning, no-mask reason, alias precedence, and Otsu/GrabCut tests pass.
- [x] Task 4: AutoMM validation tests pass, including log-before-fit and unchanged training-side leniency.
- [x] Task 5: `tests/test_docs.py`, version tests, help checks, and full suite pass.
- [x] Python 3.9 compatibility review: no PEP 604 unions in modules without `from __future__ import annotations`.
- [ ] Final review: no input deletion, no half-written outputs left after failures, no unapproved commit/tag/push, and the four issue contracts are covered by tests. Pre-existing `--overwrite` input/output overlap and symlink destination redirection remain outside this plan and need separate approval.
- [x] Implementation result (2026-10-31): without SAM3 env, `522 passed, 2 skipped`; with `ENTOMOKIT_SAM3_CHECKPOINT=...`, `524 passed, 0 skipped`; `tests/test_docs.py` and version tests are included in those runs.
