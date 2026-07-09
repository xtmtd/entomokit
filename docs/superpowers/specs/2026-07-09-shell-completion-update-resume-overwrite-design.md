# Shell Completion Fix / `update` Command / `--resume` & `--overwrite` Design

## Goal

Three independent but co-released improvements to entomokit 0.4.0:

1. **Shell file-path completion fix** — `--input-dir` / `--out-dir` and similar path arguments currently cannot use the shell's native filesystem tab-completion because the generated scripts suppress it. Fix this for bash, zsh, fish.
2. **`entomokit update` command** — check whether a newer commit exists on GitHub and offer to install it via `pip install git+<url>`.
3. **`--resume` / `--overwrite` for processing commands** — protect non-empty output directories from accidental overwrite and allow interrupted long runs to be continued.

---

## Version Bump

`0.3.0` → `0.4.0`

Files to update: `entomokit/_version.py`, `setup.py`, `README.md` (line 7), `README.cn.md` (line 7).

---

## Feature 1: Shell File-Path Tab-Completion Fix

### Root cause

`entomokit/completion.py` generates static completion scripts. The scripts handle enum-valued options (e.g. `--segmentation-method`, `--device`) explicitly, but for all other options (path parameters like `--input-dir`, `--out-dir`) they produce no completion at all. The shell therefore offers nothing when the user presses Tab after a path argument.

### Fix per shell

| Shell | Problem | Fix |
|-------|---------|-----|
| bash  | `complete -F _entomokit_completion entomokit` — no `-o default`, so bash never falls back to filename completion when `COMPREPLY` is empty | Add `-o default` flag: `complete -o default -F _entomokit_completion entomokit` |
| zsh   | `_entomokit()` function ends without calling `_files`, so no filename expansion | Append `_files` call at the end of the function body, after all `case` blocks |
| fish  | First line `complete -c entomokit -f` sets the `-f` (no-files) flag globally, suppressing filesystem completion everywhere | Remove `-f` from the first global `complete` line |

**Scope:** only `entomokit/completion.py`. No changes to argparse definitions. No behaviour change for enum-valued options — those still show their choices as before.

**Re-installation:** users need to re-run `entomokit completion <shell> --install` after upgrading to pick up the fix. Mention this in the release notes.

---

## Feature 2: `entomokit update` Command

### Behaviour

```
entomokit update             # check + prompt
entomokit update --check     # check only, no install
entomokit update --yes       # check + install without prompt
```

**Check step:**
1. Fetch the latest commit on `main` from the GitHub Commits API (`https://api.github.com/repos/xtmtd/entomokit/commits/main`, timeout 10 s).
2. Extract the commit `sha` (short: first 7 chars) and `commit.author.date`.
3. Compare with the local commit hash in `entomokit/_version.py` (add `__commit__ = "unknown"` and `__commit_date__ = "unknown"` alongside `__version__`; packaged installs can remain unknown until release automation writes those values).
4. Display:
   - Current: `0.4.0 (abc1234, 2026-07-09)`
   - Latest: `def5678, 2026-07-10`
   - If sha matches: "Already up to date."
   - If `__commit__` is `"unknown"`: "Development install — cannot compare commits." Do not prompt unless `--yes` is explicitly passed.
   - If date is newer than the local commit date when known: show the latest commit message and date, then prompt.
   - If local date is unknown and sha differs: show the latest commit message and ask the user to confirm manually.

**Install step:**
```
pip install --upgrade git+https://github.com/xtmtd/entomokit.git
```
Run via `subprocess` with the current `sys.executable`. Show stdout/stderr. Report success or failure.

**`--yes` flag:** skip the `Proceed? [y/N]` prompt.
**`--check` flag:** print check result only; never call pip.

### `__commit__` / `__commit_date__`

Add `__commit__ = "unknown"` and `__commit_date__ = "unknown"` to `entomokit/_version.py`. Do not override `setup.py` install/develop commands for this release; that can break normal setuptools behavior for little gain.

`# ponytail: unknown commit is safer than custom install hooks; add release-time stamping only when release automation exists`

### Files

- Create: `entomokit/update.py`
- Modify: `entomokit/main.py` (register the subcommand)
- Modify: `entomokit/_version.py` (add `__commit__` and `__commit_date__` variables)
- Modify: `setup.py` (version bump only)

---

## Feature 3: `--resume` / `--overwrite` for Processing Commands

### Commands receiving `--resume` + `--overwrite`

| Command | Has slow/interruptible processing? | Resume skip unit | Notes |
|---|---|---|---|
| `segment` | Yes — SAM3/LaMa per image can be slow | per input image (by stem) | Skip if any file matching `<stem>*` exists in `out_dir/images/` |
| `synthesize` | Yes — many syntheses per target | per target image (by stem) | Skip if any file matching `<target_stem>_*` exists in `out_dir/images/` |
| `augment` | Moderate | per source image (by stem) | Skip if any file matching `<stem>*` exists in `out_dir/images/` |
| `clean` | Moderate | N/A — existing dedup logic already handles | `--resume` just suppresses the non-empty dir error |
| `measure` | Moderate for large mask sets | per mask file | Read existing `metrics.csv`, collect `file_name` column, skip already-measured masks; combine old + new rows and rewrite CSVs |
| `extract-frames` | Depends on video count/length | per frame file (already implemented via `--skip-existing`) | Rename `--skip-existing` → `--resume`; add `--overwrite` |

### Commands receiving `--overwrite` only (no `--resume`)

| Command | `--overwrite` behaviour |
|---|---|
| `split-csv` | Delete `--out-dir` and regenerate all splits |
| `classify train` | Delete `--out-dir`, train from scratch; existing `--resume` also passes through guard |
| `classify predict` | Delete `--out-dir` and re-predict all inputs |
| `classify evaluate` | Delete `--out-dir` and re-evaluate |
| `classify embed` | Delete `--out-dir` and re-extract embeddings |
| `classify cam` | Delete `--out-dir` and regenerate CAM visualizations |
| `classify export-onnx` | Delete `--out-dir` and re-export ONNX model |

### `--overwrite` guard policy

Applied consistently at the CLI layer (`run()` function) before any processing starts:

```
if out_dir is non-empty AND --overwrite:   delete out_dir entirely, recreate
if out_dir is non-empty AND --resume:      proceed (skip logic per command)
if out_dir is non-empty AND neither flag:  print error and exit 1
if out_dir does not exist:                 mkdir -p (no flag needed)
```

**Implementation:** add a small shared `check_output_dir(out_dir, resume, overwrite)` helper in `src/common/cli.py`, next to the existing CLI helpers. Each command calls it before logging or processing.
`# ponytail: one shared guard is less code and makes tests cover production behavior`

### `--resume` skip logic detail

**segment:** `process_directory()` in `src/segmentation/processor.py` gains `skip_existing: bool = False`. Inside the per-image loop, before loading the image: check `list((output_dir / "images").glob(f"{img_path.stem}*"))`; if non-empty, skip.

**synthesize:** `process_directory()` in `src/synthesis/processor.py` gains `skip_existing: bool = False`. Inside the per-target loop: check `list((output_dir / output_subdir).glob(f"{target_path.stem}_*"))`; if non-empty, skip that target.

**augment:** `run_augment()` in `src/augment/service.py` gains `skip_existing: bool = False`. Inside the per-image loop: check `list(images_out.glob(f"{img_path.stem}*"))`; if non-empty, skip.

**clean:** No `src/` change. The cleaner already pre-populates its dedup hashes from existing output files. `--resume` in `clean` only serves to bypass the non-empty dir guard.

**measure:** CLI layer reads the existing `metrics.csv` (column `file_name`) to build a `skip_set` of already-measured stems and pass existing rows to the service. The service measures only new masks, combines existing rows + new rows, then rewrites `metrics.csv`, `metrics_summary.csv`, and `metric_definitions.csv` from the combined rows. This avoids duplicate headers, schema drift, and stale summaries.

**extract-frames:** `--skip-existing` → `--resume` (rename only in argparse; internal attribute name `args.skip_existing` changes to `args.resume`). Add `--overwrite` with the standard guard.

### `extract-frames` flag rename

Old flag `--skip-existing` is removed entirely (not aliased). Users who have the old flag in scripts will get an "unrecognized argument" error, which is acceptable since the flag was undocumented in normal usage flow.

---

## Documentation Updates

All documentation updates are in scope for this release. Nothing lands in code without the docs being updated in the same change set.

### Files to update

| File | What changes |
|---|---|
| `entomokit/_version.py` | Add `__commit__ = "unknown"` and `__commit_date__ = "unknown"` |
| `setup.py` | Version bump to `0.4.0` |
| `README.md` | Version bump (line 7); add `update` command section; add `--resume`/`--overwrite` to 6 Phase 1 command param tables + `--overwrite` to split-csv + classify train param tables + prose notes for predict/evaluate/embed/cam/export-onnx; fix shell completion section (remove old `--install-completion` reference at line ~958); update `extract-frames` to show `--resume` not `--skip-existing` |
| `README.cn.md` | Same as above, Chinese version |
| `skills/entomokit-workflow/SKILL.md` | Add `update` to command list in description; add note about `--resume`/`--overwrite` behaviour vs skill-layer directory policy |
| `skills/entomokit-workflow/references/command-profiles.md` | Add `--resume`/`--overwrite` note to `clean`, `segment` profiles; add `--overwrite` note to `split-csv`, `classify train`, `classify predict/evaluate/embed/cam/export-onnx` profiles; update Retry/Rerun section |
| `skills/entomokit-workflow/references/workflow.md` | Add `--resume`/`--overwrite` hints in Phase 1 command notes; add output-dir notes in Phase 2 and Phase 3 |

### Files confirmed not needing update

`error-catalog.md`, `csv-validation.md`, `dialog-templates.md`, `path-resolution.md`, `teaching-playbook.md`, `release-checklist.md` — none contain command-parameter documentation or version references.

---

## Testing Approach

- Unit tests for `src.common.cli.check_output_dir` guard logic.
- Unit tests for `update.py` (`is_newer`, `fetch_latest_commit` with mocked urllib).
- Per-command smoke tests: non-empty dir without flags → `SystemExit`; with `--overwrite` → dir cleared; with `--resume` → no error.
- Completion script tests: assert `-o default` in bash output; `_files` in zsh output; no global `-f` in fish output.
- All tests in `tests/` using `pytest`. No new test dependencies.

---

## What Is Out of Scope

- `--resume` for `split-csv`, `classify predict/evaluate/embed/cam/export-onnx` — these commands are fast or single-pass; per-item skip logic is unnecessary. `classify train` already has its own checkpoint `--resume`.
- `doctor` — one-shot diagnostic, no output dir.
- Deprecation alias for `--skip-existing` — removed cleanly.
- PyPI publishing — install via `git+` URL only.
- Windows shell completion — not supported, not changed.

### `--overwrite` extension

`--overwrite` (the output-dir safety guard only, no `--resume`) is also added to:

- `split-csv`
- `classify train` (aligns existing `--resume` with the guard: `--resume` passes through non-empty dir; `--overwrite` deletes and starts fresh)
- `classify predict`, `classify evaluate`, `classify embed`, `classify cam`, `classify export-onnx`

These commands all have `--out-dir` but previously had no guard at all (`mkdir(exist_ok=True)` allowed silent overwrite). The guard makes accidental overwrite explicit.
