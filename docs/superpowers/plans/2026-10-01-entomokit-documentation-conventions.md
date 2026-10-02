# EntomoKit Documentation Conventions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for direct execution. Use superpowers:subagent-driven-development only if the operator separately requests delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the complete bilingual functional-command reference out of README, establish the approved documentation checks and help pointers, and prepare the complete local `0.7.1` change set.

**Architecture:** README remains the project entry and owner of operational commands and shared behavior. Thirteen flat English/Chinese command-reference pairs own functional usage; tests consume existing argparse/schema interfaces rather than introducing a second CLI registry. The master design references dedicated design owners without discarding uncovered architectural invariants.

**Tech Stack:** Existing Python, argparse, pathlib, re, pytest, and Markdown; no new dependencies, documentation generator, site, or CI configuration.

**Spec:** [Documentation-conventions design](../specs/2026-10-01-entomokit-documentation-conventions-design.md). Read it before execution; its source-of-truth rule and ownership map govern this plan. The original design was explicitly approved in operator conversation; the current design revision and this plan await operator confirmation.

**Status:** Draft for operator review. The original design was explicitly approved in operator conversation; the current design revision and this plan await operator confirmation. Neither confirmation authorizes implementation, commits, tags, pushes, or releases, which remain separately approved.

## Global Constraints

- Release target is exactly `0.7.1`; update live declarations only after the documentation/help migration is complete.
- Preserve CLI syntax, defaults, choices, processing, non-help analysis output, and every existing per-option help string. Help URLs and approved version display/log-header changes are intentional differences.
- `doctor`, `update`, and `completion` remain README-only; do not create their command documents.
- All thirteen functional commands receive both `.md` and `.cn.md` references in the same change, with an identical English invocation H1 and same-language README navigation.
- Required H2 sections are Purpose, Usage, Parameters; optional sections follow the spec's order. Each option has an H3 block, with aliases sharing that block.
- Preserve valid information from the union of both READMEs; apply the spec's Section 2 source-of-truth rule to discrepancies and record their user impact for operator review.
- Do not rewrite the workflow skill's policy or reconcile `command-profiles.md` in this migration. Only relocate conversation examples to `teaching-playbook.md`.
- Do not add shared-reference files, a changelog, a second command index, new specs, or independent migration/review ledgers. Record evidence in this plan.
- Version Notes are newest first. Usage and Examples must not repeat the same example; long contracts live in Inputs, Outputs or Notes with a pointer from the option block. Headings carry no release numbers, and inapplicable sections are omitted rather than left empty.
- Historical plans are rationale, not owners of current architecture. Retain master-design details without a dedicated design owner.
- No package installation, image analysis, actual update installation, commit, tag, or publication without separate authorization.
- A missing test dependency is a blocker to resolve with the operator, not permission to install packages or declare verification passed.
- The repository has no CI enforcement. Local checks cannot satisfy the actual GitHub-rendered navigation gate.

## Review Focus

- Stale or duplicate document/index entries must fail exact-set checks, including an accidental `classify` parent row; covered by Tasks 1 and 6.
- Aliases must count only inside Parameters H3 code spans or bounded operational table cells; prose/examples must not satisfy coverage; covered by Task 1.
- `completion` parent/group help is absent from leaf schemas, while `doctor` has no user options; both remain verifiable without fake tables or documents; covered by Tasks 1 and 7.
- Chinese-only contracts and scientific/version caveats must survive translation and relocation; covered by each command's manual checklist in Tasks 2-5.
- Explicit anchors may pass local checks yet fail GitHub sanitization; actual bilingual navigation must be checked after authorized publication; covered by Task 10, never presumed locally passed.

## File Map and Task Order

All command-file names below are relative to `docs/commands/`.

| Task | Create or modify | Responsibility |
|---|---|---|
| 1 | Create `tests/test_docs.py` | Closure, bilingual skeleton, option coverage, bounded README indexes/tables, local links and help URLs |
| 2 | Create `extract-frames.md`, `extract-frames.cn.md`, `segment.md`, `segment.cn.md`, `measure.md`, `measure.cn.md` | Ingestion, annotation and measurement references |
| 3 | Create `synthesize.md`, `synthesize.cn.md`, `clean.md`, `clean.cn.md`, `augment.md`, `augment.cn.md`, `split-csv.md`, `split-csv.cn.md` | Dataset preparation references |
| 4 | Create `classify-train.md`, `classify-train.cn.md`, `classify-predict.md`, `classify-predict.cn.md`, `classify-evaluate.md`, `classify-evaluate.cn.md` | Training and assessment references |
| 5 | Create `classify-embed.md`, `classify-embed.cn.md`, `classify-cam.md`, `classify-cam.cn.md`, `classify-export-onnx.md`, `classify-export-onnx.cn.md` | Embedding, interpretation and export references |
| 6 | Modify `README.md`, `README.cn.md`, `skills/entomokit-workflow/references/teaching-playbook.md` | Entry points, operational reference, shared rules and relocated conversations |
| 7 | Modify `entomokit/help_style.py`, `entomokit/main.py`, `entomokit/{extract_frames,segment,measure,synthesize,clean,augment,split_csv,doctor,update,completion}.py`, `entomokit/classify/{__init__,train,predict,evaluate,embed,cam,export_onnx}.py` | Shared help URL construction and all parser-level pointers |
| 8 | Modify `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` | Current navigation, conventions and reference-first maintenance |
| 9 | Modify `setup.py`, `version.txt`, `entomokit/_version.py`, `entomokit/main.py`, `tests/test_package_version.py`, `tests/test_main_cli.py`, both README log examples | Final `0.7.1` declarations and assertions |
| 10 | Update gate evidence in this plan; no publication by default | Separately authorized commit/publication and actual GitHub navigation gate |

Tasks execute in order. Task 1 intentionally exposes missing migration work.
Tasks 2-5 run focused checks; complete closure, README/local-link and help gates
remain pending until their owning tasks finish. Never add skips, temporary
exception lists, partial-success assertions or `xfail` to hide this. A batch is
accepted only for its focused deliverable; full migration acceptance stays open.
No intermediate commit is implied by a task boundary; any commit requires
separate authorization and must respect the version-bearing release boundary.

---

### Task 1: Establish the Documentation Test Contract

**Files:** Create `tests/test_docs.py`; read `entomokit/cli_schema.py`, `entomokit/main.py`, `tests/test_cli_schema.py`, `tests/test_cli_help_texts.py`, and `pytest.ini` without modifying them.

**Interfaces:** Consume `build_command_schemas(parser=None) -> dict[str, dict[str, object]]` and `_build_parser() -> argparse.ArgumentParser`. Produce the repository checks below and small test-local extraction helpers. Do not extend the production schema API.

- [x] **Step 1: Record the pre-migration baseline here.** Run `git status --short` and `python -m pytest -q`. Record branch/ref, test totals, dependency failures and existing changes in Execution Evidence. Record the deterministic **leaf** schema digest with the command below; the schema covers leaf commands only, including each leaf's user-option help strings, aliases, defaults, choices and requiredness, and excludes the root parser, so root `-v/--version` option help is verified separately in Task 7. Recompute it in Task 7 and require equality. This persists the baseline in this plan without a snapshot file; use the recorded starting ref/diff to investigate any mismatch. Do not alter unrelated changes.
Schema baseline command (read-only):

```bash
python -c 'import hashlib,json; from entomokit.cli_schema import build_command_schemas; print(hashlib.sha256(json.dumps(build_command_schemas(), sort_keys=True, ensure_ascii=True).encode()).hexdigest())'
```

- [x] **Step 2: Write failing extraction assertions.** Use inline Markdown snippets and existing pytest `tmp_path` only where files are needed. Cover code-span aliases `--out-dir`/`-o`, comma-separated spans, `-h`/`--help` exclusion, fenced examples, H3 scope bounded by the Parameters H2, and operational sections bounded by the next H2. Prose-only flags and flags outside Parameters must not count. Normalize README invocation labels to hyphenated stems, reject duplicate rows across the discovered Commands tables (the main table plus every functional-group H3 table), and reject missing/extra row labels (including a `classify` parent row). Tables outside the Commands H2 must not count.
- [x] **Step 3: Run the helper checks red, then implement minimal extraction.** Run `python -m pytest tests/test_docs.py -q -k extraction`. Initial failure must identify the missing helper/incorrect extraction, not an unrelated import error. Use test-local stdlib regex/splitting and ignore fenced code before parsing. For H3 headings and option cells, accept code-span tokens matching `^--?[A-Za-z0-9][A-Za-z0-9-]*$`, ignoring separators/prose and globally removing `-h`/`--help`. Do not add a general Markdown parser or dependency.
- [x] **Step 4: Add repository checks with the following names and assertions.**

| Test | Assertions |
|---|---|
| `test_functional_document_closure` | Schema keys excluding `doctor`, `update`, and `completion <shell>` map via space-to-hyphen to exactly the English document stems; normalized Chinese stems equal that same set, rejecting orphan/missing/stale files in either language |
| `test_command_document_structure` | Parameterize functional schema paths with stem IDs; English/Chinese pairs, identical invocation H1, language switch immediately below, required H2s, no empty H2/H3 headings, no release numbers in headings, and identical normalized H2 sequence in spec order |
| `test_command_option_coverage` | Parameterize the same paths; each language's Parameters H3 code-span token set equals all `parameters[*].options`, with aliases and global help exclusion |
| `test_readme_command_index` | Inside each README's Commands H2, discover the main table plus the command table under every H3 subsection (today: main table plus `classify-commands`); the union of row labels must equal the functional stems plus `doctor`, `update`, `completion`, rejecting duplicates, missing and extra rows (including a `classify` parent row) and validating exact same-language targets or operational anchors |
| `test_readme_operational_options` | Anchored/bounded sections match schema options; doctor may lack its table only because its user-option set is empty; completion shell labels equal schema suffixes and each row covers that shell's `--install` |
| `test_document_local_links` | Both READMEs and every command file resolve live relative links/directories and explicit `id`/`name` fragments; same-language command/README routes except language switches; ignore fenced links and external URLs rather than requiring network access |
| `test_parser_help_links` | Expected published URLs occur in root, functional-group, operational parent and all leaf `format_help()` outputs, including completion parent and shells |

- [x] **Step 5: Keep reusable test constants minimal.** Define the spec's eight English/Chinese H2 translation/order pairs once as a test-local constant with a spec link. Discover command sets through `build_command_schemas()`; counts 18/13/16 are review evidence, not a second registry. Reuse its existing leaf discovery. For root/group help use `_build_parser()` and descend argparse subparser actions as needed; do not call callbacks or load optional analysis backends. Derive group navigation expectations from the spec's stable-anchor convention, not from leaf schemas alone.
- [x] **Step 6: Verify fragment extraction with inline files.** Add `extraction` tests showing `<a id="target"></a>` and `<a name="target"></a>` both satisfy `#target`, nonexistent fragments fail, and fenced links are ignored. Do not reproduce GitHub heading-slug generation or hardcode which shared-rule link each command must carry.
- [x] **Step 7: Run `python -m pytest tests/test_docs.py -q -k extraction` green, then `python -m pytest tests/test_docs.py -q`.** Helper tests must pass; repository failures must identify absent references, indexes/anchors or help pointers. Record expected pending checks separately from baseline/environment blockers. Task 1 establishes the checks; it does not claim the migration passes.

**Task 1 executed: complete.** Steps 1-7 done. Baseline, branch and leaf schema digest are recorded in Execution Evidence. `tests/test_docs.py` was written test-first: the twelve extraction/fragment helpers were watched fail (12 NameError failures) and then pass (`12 passed`). The seven repository checks were added afterwards and currently fail only with gap-identifying messages, e.g. `docs/commands is missing`, `English document missing: docs/commands/clean.md`, `README.md: no Commands/命令 section`, `root: help omits https://github.com/xtmtd/entomokit/blob/main/README.md`. Whole-file result: `12 passed, 31 failed`. One helper fix was needed while going green: an anchor marks the start of its own H2 section because the spec places anchors immediately before the heading, so `anchored_body` treats the first heading after the anchor as the section start; the test, not the helper, encoded the spec here.

### Task 2: Migrate Ingestion, Segmentation and Measurement

**Files:** Create the six Task 2 command files in the File Map. Read both README source sections, `entomokit/{extract_frames,segment,measure}.py`, `src/framing/extractor.py`, `src/segmentation/processor.py`, `src/common/annotation_writer.py`, and `src/measurement/{core,io,service,skeleton}.py` as needed to trace behavior. Read the annotation, measurement and CPU-parallelism designs listed in spec Section 9.

**Interfaces:** Consume Task 1's schema-driven tests and the H2/H3 skeleton. Produce complete same-language pairs without removing README source yet.

- [x] **Step 1: Map the union of both source sections to destination sections.** Record source-to-destination notes and code/prose conflicts in this batch. Inspect aliases/defaults/choices through schema and code, including runtime-resolved values that argparse alone cannot explain.
- [x] **Step 2: Write each pair in the fixed skeleton.** Include requiredness, explicit boolean defaults, no-default/automatic distinctions and descriptions at least as informative as option help. Preserve extraction sampling/resume/output layout; segment flattening/sample IDs, annotation layouts and safety, SAM3/LaMa preparation, CPU concurrency versus SAM3 serialization; measurement definitions and quality limits.
- [x] **Step 3: Check Chinese-only segmentation content in both languages.** Preserve VOC mask layout, `area` interpretation, polygon-versus-bbox semantics and conditional outputs. Link shared recursive/log/device/safety rules to the README anchors Task 6 will create instead of copying them.
- [x] **Step 4: Run `python -m pytest tests/test_docs.py -q -k 'extraction or ((document_structure or command_option_coverage) and (extract-frames or segment or measure))'`.** Expected: focused checks pass. Full local-link/index/help gates remain pending. Validate minimal invocation argument syntax without executing analyses.
- [x] **Step 5: Complete the records below only after checking both languages against code and help.** Record source locations and unresolved discrepancies below each command. An unresolved discrepancy cannot be marked reviewed.

**Task 2 executed: complete.** Sources: README `Extract Frames Command`,
`Segment Command` (plus the SAM3/LaMa model requirements) and `Measure Command` in
both languages; the Chinese-only segment annotation-field notes were merged into
both languages. Discrepancies recorded for operator review (README vs code): the
README option tables omitted `--verbose`/`--quiet` for extract-frames and
`--lama-mask-dilate`/`--coco-output-mode`/`--verbose` for segment; no parser option
was absent from the README and no behavior disagreement was found — the omitted
options are now documented, which is the current-code-first rule, not a semantic
change. The README also never stated the real output-directory guard
(`src/common/cli.py::check_output_dir`, called by extract-frames, segment and
measure): a non-empty `--out-dir` exits with an error unless `--resume` or
`--overwrite` is given. All three command documents now state it and link the
shared rule; the skill's `command-profiles.md` already matched the code for
clean/segment and stays untouched per spec Section 2. Ruling: the language switch
sits on the literal line below the H1 because the spec says "immediately below";
the six documents were aligned to the strict check instead of relaxing it. Cost if
wrong: a cosmetic blank line. Ruling: Version Notes for extract-frames and segment
record the `0.7.0` recursive-layout behavior (the 0.7.0 plan in spec Section 9 owns
that history); measure has no recoverable command-specific history and omits
Version Notes rather than inventing entries. Test-side fix: `_heading_bodies`
first treated a container section (an H2 holding only H3 subsections) as empty; it
is now bounded by the next same-or-higher-level heading, matching the spec's
"do not create empty headings". Focused result: `18 passed` (12 extraction, 3
structure, 3 option coverage).

**Per-command review record:**

`extract-frames`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs and prerequisites checked.
- [x] Outputs, resume and safety checked.
- [x] Known historical notes checked; no fabricated entries.

`segment`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, SAM3 and LaMa prerequisites checked.
- [x] Outputs, annotation semantics, resume and safety checked.
- [x] Known historical notes checked; no fabricated entries.

`measure`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, geometry and mask assumptions checked.
- [x] Outputs, units, quality caveats and safety checked.
- [x] Known historical notes checked; no fabricated entries.

### Task 3: Migrate Dataset Preparation

**Files:** Create the eight Task 3 command files. Read both README sections, `entomokit/{synthesize,clean,augment,split_csv}.py` and relevant `src/{synthesis,cleaning,augment,splitting}/` implementations. Use the `0.7.0` plan only as migration history.

**Interfaces:** Consume Task 1's checks; produce complete bilingual preparation references. README source remains until Task 6's inventory reconciliation.

- [x] **Step 1: Map every valid source paragraph and inspect each command's real flow.** Record destinations and discrepancies here. Check runtime/output rules in processors, not only parser metadata.
- [x] **Step 2: Write the pairs.** Preserve RGBA synthesis prerequisites, with-replacement selection, seeds and fractional synthesis count semantics; cleaning format/EXIF/padding behavior; augmentation input/output and variant policy; split CSV fields, path resolution, label mapping and partition behavior. Keep command-specific limitations and deletion warnings visible; link shared rules rather than copying them.
- [x] **Step 3: Run `python -m pytest tests/test_docs.py -q -k 'extraction or ((document_structure or command_option_coverage) and (synthesize or clean or augment or split-csv))'`.** Expected: focused checks pass; parse minimal usage without executing callbacks.
- [x] **Step 4: Complete the records below and record source/destination evidence.**

**Task 3 executed: complete.** Sources: README `Synthesize Command`, `Clean Command`, `Augment Command` and `Split-CSV Command` in both languages. Discrepancies recorded for operator review: the README labelled `split-csv --out-dir` as required while the parser default is `datasets` (documented from code); README option tables omitted `--verbose/-v` for split-csv and clean, and the synthesize table omitted `--out-image-format`, `--coco-output-mode` and `--verbose/-v` (all now documented). Version Notes: `0.7.0` recursive/mirrored layout for synthesize, clean and augment, plus the 0.7.0 synthesis count/seed change; split-csv has no recoverable command history and omits Version Notes. Focused result: `18 passed`.

`synthesize`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, RGBA assumptions and prerequisites checked.
- [x] Outputs, replacement/fractional semantics, resume and safety checked.
- [x] Known historical notes checked; no fabricated entries.

`clean`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs and format/EXIF prerequisites checked.
- [x] Outputs, padding, mirrored layout, resume and safety checked.
- [x] Known historical notes checked; no fabricated entries.

`augment`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs and variant prerequisites checked.
- [x] Outputs, mirrored layout, resume and safety checked.
- [x] Known historical notes checked; no fabricated entries.

`split-csv`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, CSV fields and path resolution checked.
- [x] Outputs, partition/label mapping and safety checked.
- [x] Known historical notes checked; no fabricated entries.

### Task 4: Migrate Classification Training and Assessment

**Files:** Create the six Task 4 command files. Read both README classification sections, `entomokit/classify/{train,predict,evaluate}.py` and their actual `src/classification/` backend imports.

**Interfaces:** Consume schema paths `classify train`, `classify predict`, `classify evaluate`; produce flat stems `classify-train`, `classify-predict`, `classify-evaluate`.

- [x] **Step 1: Map valid source text and trace prerequisite/input/output behavior.** Keep model preparation out of README except compact pointers. AutoMM/timm training preparation belongs in train; prediction/evaluation backend requirements belong in their own references.
- [x] **Step 2: Write all three pairs.** Preserve CSV-driven discovery, augmentation presets, model/label contracts, ONNX interpretation, metric/diagnostic outputs and command-specific resume/safety limitations. Do not imply shared recursive discovery governs CSV-driven paths.
- [x] **Step 3: Run `python -m pytest tests/test_docs.py -q -k 'extraction or ((document_structure or command_option_coverage) and (classify-train or classify-predict or classify-evaluate))'`.** Expected: focused checks pass; inspect minimal invocations without training/inference.
- [x] **Step 4: Complete the records below and record source/destination evidence.**

**Task 4 executed: complete.** Sources: the README `Classify Commands` subsections for train, predict and evaluate in both languages. Discrepancies recorded: README tables omitted `--num-threads` for all three and `--overwrite` for predict/evaluate; predict and evaluate have no `--resume`, which the documents now state explicitly. Version Notes: `0.7.0` `--seed` (train) and `0.7.0` relative-path image records (predict); evaluate records the `0.7.0` diagnostics output set. Focused result: `18 passed`.

`classify train`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, model preparation and CSV prerequisites checked.
- [x] Outputs, presets, resume and safety checked.
- [x] Known historical notes checked; no fabricated entries.

`classify predict`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, model/ONNX and CSV prerequisites checked.
- [x] Outputs, label mapping, resume and safety checked.
- [x] Known historical notes checked; no fabricated entries.

`classify evaluate`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, labels and evaluation prerequisites checked.
- [x] Outputs, metric/diagnostic interpretation and safety checked.
- [x] Known historical notes checked; no fabricated entries.

### Task 5: Migrate Embedding, CAM and ONNX Export

**Files:** Create the six Task 5 command files. Read both README sections, `entomokit/classify/{embed,cam,export_onnx}.py`, their actual backend imports and the existing CAM-array/embedding-metric plans from spec Section 9.

**Interfaces:** Consume the last three functional schema paths. Produce the final six files, completing the 13-command/26-file inventory.

- [x] **Step 1: Map and reconcile valid source text with current code.** Historical plans supply rationale and version history, not current defaults. Record discrepancies and destinations here.
- [x] **Step 2: Write the pairs.** Preserve embedding output/metric definitions, memory warnings, the `0.6.2` comparability warning and the fact that `0.7.0` did not change that algorithm. Preserve CAM architecture/preprocessing constraints, raw versus normalized arrays, comparison limits and conditional outputs. Preserve ONNX export prerequisites, opset and label mapping contracts.
- [x] **Step 3: Run `python -m pytest tests/test_docs.py -q -k 'extraction or ((document_structure or command_option_coverage) and (classify-embed or classify-cam or classify-export-onnx))'`.** Expected: focused checks pass. Review worked examples without executing analyses or exporting a model.
- [x] **Step 4: Run `python -m pytest tests/test_docs.py -q -k 'functional_document_closure or command_document_structure or command_option_coverage'`.** Expected: all functional documentation checks pass with exactly 13 pairs. README/link/help gates remain pending and are mandatory at final verification.
- [x] **Step 5: Complete the records below and record source/destination evidence.**

**Task 5 executed: complete.** Sources: the `classify embed`, `classify cam` and `classify export-onnx` README subsections in both languages. The embedding metric contract and the `0.6.2`/`0.7.0` comparability statements are preserved in both languages, and CAM raw/normalized semantics with its comparison limits are kept. Discrepancies recorded: README tables omitted `cam --num-threads`, `cam --arch`, `cam --max-images` and `export-onnx --sample-image` (now documented); CAM's ONNX limitation is stated as a command restriction. Version Notes: embed keeps the `0.6.2` corrections and the `0.7.0` no-change statement; cam and export-onnx record their `0.7.0` array-export and label-mapping behavior. Test-side fix: `_heading_bodies` now reads raw text so a section holding only a fenced example is not "empty" and headings inside fenced blocks are ignored; `classify-embed` moved its metrics table under Outputs as an H3. Functional result: `27 passed` (closure, structure and coverage for all 13 pairs).

`classify embed`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, metric assumptions and model prerequisites checked.
- [x] Outputs, comparability, memory, resume and safety checked.
- [x] Known historical notes, including `0.6.2` and `0.7.0`, checked.

`classify cam`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, architecture and preprocessing prerequisites checked.
- [x] Outputs, raw/normalized semantics, limitations and safety checked.
- [x] Known historical notes checked; no fabricated entries.

`classify export-onnx`
- [x] Defaults checked in both languages.
- [x] Choices and aliases checked in both languages.
- [x] Inputs, opset and export prerequisites checked.
- [x] Outputs, label mapping and safety checked.
- [x] Known historical notes checked; no fabricated entries.

### Task 6: Rewrite README and Relocate Conversation Examples

**Files:** Modify `README.md`, `README.cn.md`, and `skills/entomokit-workflow/references/teaching-playbook.md` only.

**Interfaces:** Consume all 13 pairs and source/destination records. Produce bilingual entry structure, exact command indexes and stable fragments used by documents/help.

- [x] **Step 1: Reconcile the original README inventory before deleting references.** Every meaningful contract must have its command/shared destination. Confirm the union of both languages, not paragraph counts. Preserve code/prose discrepancies for operator review; do not change code to reconcile them.
- [x] **Step 2: Rewrite both READMEs in spec Section 3 order.** Keep optional workflow branches and recommended command order once; requirements/extras; one recommended isolated install route plus conda/uv/venv alternatives; valid quick start; operational reference; shared behavior; AI entry; Commands; license/contact/citation. Preserve the `stringzilla` workaround. Remove full functional reference blocks, duplicate completion/features/install prose and per-file tree, not useful contracts. No hard line cap; compare final lengths with the approximately 1,200-line originals.
- [x] **Step 3: Preserve complete operational/shared rules.** Use H2 operational sections; doctor needs no empty option table; update names/aliases go in the first column; completion is a required shell subparser with `bash`, `zsh`, `fish`, each row's options in the second column. Retain install paths/zsh activation, recursive/mirrored layout with segment exception, CSV discovery, output safety, logging, devices, interruption and root version flags. State applicability; skill approval/cleaning policy is not a universal CLI prerequisite.
- [x] **Step 4: Add seven explicit anchors in both languages immediately before their headings.** IDs: `doctor-command`, `update-command`, `completion-command`, `directory-policy`, `common-behaviours`, `classify-commands`, `assistant-integration`. Commands has seven functional plus three operational main rows, then the anchored classification H3 and six leaf rows. Display invocation labels such as `classify train`; tests normalize to stems. Link exact same-language files and operational fragments; reject group/stale extra rows.
- [x] **Step 5: Move only conversation examples to the teaching playbook.** Add `<a id="user-conversation-examples"></a>` immediately before `## User Conversation Examples`, preserve English/Chinese examples and link both READMEs there. Keep README skill installation, name/purpose and capability summary (guided execution, validation, recovery, resume, teaching). Link detailed rules to `SKILL.md`; do not change skill policy or `command-profiles.md`.
- [x] **Step 6: Repair moved links/fragments and run `python -m pytest tests/test_docs.py -q -k 'not parser_help_links'`.**

**Task 6 executed: complete.** `README.md` and `README.cn.md` were rewritten in spec
Section 3 order (251 and 232 lines, from 1230/1225) with the seven explicit anchors,
two Commands tables, the operational sections, shared rules, an extras-to-command
mapping and the compact AI section; the file tree and the duplicate completion
section are gone. The three conversation examples (English and Chinese) moved to
`skills/entomokit-workflow/references/teaching-playbook.md` under
`## User Conversation Examples` with the `<a id="user-conversation-examples"></a>`
anchor. The README log-header example was restored with `0.7.1`. Discrepancies
recorded: the README never stated the output-directory guard, which the operational
and shared sections now state; the old Features and Model Requirements prose was
absorbed into the workflow overview, the commands tables and the command
references rather than dropped. Test-side refinements while going green: table
header rows are skipped when collecting index labels (`tables()`), displayed
invocations such as `classify train` normalize to the `classify-train` stem, and
operational rows accept either `#anchor` or `README.md#anchor`. Result:
`42 passed, 1 deselected`. Expected: all documentation, index/operational and local-link checks pass; only the help gate remains pending. Review shared-link relevance, translations and source/destination preservation manually. Actual GitHub navigation remains pending.

### Task 7: Add Shared Help Pointers Without Changing CLI Behavior

**Files:** Modify every Task 7 module in the File Map; test in `tests/test_docs.py`. Existing schema/help-coverage tests remain unchanged.

**Interfaces:** Produce `DOCS_BASE_URL = "https://github.com/xtmtd/entomokit/blob/main/"` and `with_doc_link(description: str | None, path: str) -> str` in `entomokit/help_style.py`. Paths are developer-owned relative references, optionally with fragments. Append a blank line and the full URL to a nonempty description; absent/empty descriptions become only the URL.

- [x] **Step 1: Write helper assertions and run red.** Add `test_with_doc_link_appends_description` and `test_with_doc_link_handles_empty_description`; assert retained summary/examples plus a final unwrapped URL, and `None`/`""` becoming only the URL. Run `python -m pytest tests/test_docs.py -q -k with_doc_link`; expected failure is the absent helper. Introduce its test import in this task, not Task 1, so earlier batch collection is not blocked.
- [x] **Step 2: Add the minimal helper beside `with_examples()`.** Call shape: `with_doc_link(existing_description, "docs/commands/segment.md")`. Use the shared constant; do not scatter concatenation or alter formatting classes, option help, choices, defaults, callbacks or completion rendering.
- [x] **Step 3: Use the helper at all registrations.** Root targets `README.md`; classify targets `README.md#classify-commands`; doctor/update use their operational anchors; completion parent/shells use `README.md#completion-command`. All 13 functional leaves target `docs/commands/<stem>.md`. Wrap existing descriptions/examples intact; update/completion parsers without descriptions receive only the pointer. Subparser menu `help=` summaries stay unchanged.
- [x] **Step 4: Run `python -m pytest tests/test_docs.py tests/test_cli_help_texts.py tests/test_cli_schema.py tests/test_main_cli.py tests/test_resume_flags.py tests/test_cli_output_logging.py tests/test_update.py -q`.** Expected: pass at `0.7.0`. Recompute Task 1's **leaf** schema digest and require exact equality; investigate any changed parameter metadata or leaf option help. The leaf digest excludes the root parser, so also confirm the root `-v/--version` option help string is unchanged, since a changed root help would not move the digest. Review unchanged argument definitions in the diff. Parser descriptions are outside schema output; URLs must not change it.
- [x] **Step 5: Inspect root, classify, completion parent/shells and functional leaf help.**

**Task 7 executed: complete.** Helper tests went red (`ImportError`, 2 failed) then
green. Ruling: the pointers are applied centrally in `style_parser()` through the
`DOC_LINKS` table in `entomokit/help_style.py` instead of editing all 18
registrations by hand — every parser already calls `style_parser()`, the outcome is
identical, the path table stays in one place (spec Section 5, "keep the URL base in
one place") and `test_parser_help_links` fails loudly if a command's row is missing.
Cost if wrong: a new command must add its row in `help_style.py`, not in its own
module — recorded in the master design's conventions section. Verified: helper
functions keep existing descriptions and examples intact and return only the URL for
empty descriptions; `114 passed` across the prescribed test set; the leaf schema
digest is still `a92a7d659b3722c17dd32b7c97ea2d7089830acf7f406c85b294504f5702568b`,
so no option help or parameter metadata changed. Use `format_help()` or help-only CLI paths, never callbacks. Verify URLs once before options, retained summaries/examples/warnings and no version-aware scheme. Do not execute completion installation or a real update.

### Task 8: Maintain the Master Design Reference-First

**Files:** Modify `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`; read existing owners in spec Section 9 without rewriting them.

**Interfaces:** Consume the ownership map and final navigation. Produce a current architecture entry with links, not a second parameter manual.

- [x] **Step 1: Map detailed blocks to dedicated owners or retained master contracts.** Annotation, measurement, CPU-parallelism and operational designs cover their respective details. Classification/CAM/metric/`0.7.0` plans replace historical accounts only; retain current invariants without dedicated design owners. Do not delete entire blocks for partial ownership.
- [x] **Step 2: Correct command/directory navigation.** Include measure, augment, doctor, update and nested completion shells. Replace the per-file tree with directory roles; remove nonexistent scripts/add_functions; account for `src/augment/`, `src/doctor/`, `src/measurement/`, `src/lama/`, `src/sam3/`. Both `src/segmentation/` and `src/segmentation.py` exist; retain actual roles rather than inferring replacement.
- [x] **Step 3: Add a concise Documentation Conventions section linking the approved spec.** Summarize ownership, mirrors, complete parameters, help boundary, reference-first rules and new-command checklist. Current usage is reached through README to references, without another per-command catalog. Repair overall implementation status, distinguish history/standing rules and update last-updated to the actual implementation date.
- [x] **Step 4: Verify changed relative links and each removed contract's owner, then run `python -m pytest tests/test_docs.py -q`.**

**Task 8 executed: complete.** The master design now records the current registry
(`measure`, `augment`, `doctor`, `update`, `completion` with its shells), the
2026-10-01 last-updated note, an implemented status that separates history from
standing rules, a directory-level tree (absent `scripts/` and `add_functions/`
removed; `src/augment`, `src/doctor`, `src/measurement`, `src/lama`, `src/sam3`
added; `src/segmentation/` and `src/segmentation.py` both recorded as coexisting),
a reference-first §5.1 that links the command references instead of repeating
parameter tables, a §7 that delegates CPU-thread detail to the parallelism design,
corrected backward-compatibility bullets, and a new §10 Documentation Conventions
section linking this spec. The metric, CAM and recursive-layout invariants stay in
the master design as the ownership map requires. Expected: pass. Record retained/referenced architecture evidence here; do not create another design merely for brevity.

### Task 9: Apply Final Version Bump and Local Acceptance

**Files:** Modify Task 9 declarations/tests/README log examples. Do not change updater behavior.

**Interfaces:** Consume completed migration, command records, schema/help invariance evidence and master maintenance. Produce a complete local `0.7.1` change set, not a published release.

- [x] **Step 1: Confirm migration checks/manual records before bumping.** Run `python -m pytest -q`. Expected: pass; resolve failures without weakening tests. Any preexisting/environment blocker remains explicit and prevents a release-ready claim.
- [x] **Step 2: Change version assertions to `0.7.1` first and verify red.** Update `tests/test_package_version.py` and `tests/test_main_cli.py`; make version-bearing function names/docstrings version-neutral, including setup/version-file/log-header tests, keeping explicit target assertions. Run `python -m pytest tests/test_package_version.py tests/test_main_cli.py -q`; expected failures reference `0.7.0` declarations.
- [x] **Step 3: Update four live declarations and current README log headers together.** Set `setup.py`, `version.txt`, `entomokit/_version.py` and main's fallback to `0.7.1`. Preserve historical statements and unrelated versions such as LaMa's `brotlipy=0.7.0`. Documentation movement alone does not imply an algorithm changed; preserve real Version Notes newest first.
- [x] **Step 4: Run final acceptance commands.** Run `python -m pytest -q` and `python -m entomokit.main --version`; expected tests pass and runtime prints `entomokit 0.7.1`. Run `rg -n '0\.7\.0|0\.7\.1' setup.py version.txt entomokit tests README.md README.cn.md`, classifying old mentions as history/fixtures rather than active declarations. Updater tests use mocks; actual installation is not a gate.
- [x] **Step 5: Review `git diff --check`, `git diff --stat`, `git status --short` and full diffs/new file contents.**

**Task 9 executed: complete.** Version assertions were changed first and watched
fail (8 failed in `test_package_version.py` and `test_main_cli.py`, all naming
`0.7.0`), then `setup.py`, `version.txt`, `entomokit/_version.py` and the
`entomokit/main.py` fallback moved to `0.7.1` together with both README log-header
examples. `python -m entomokit.main --version` prints `entomokit 0.7.1`; the full
suite is `453 passed` in 35.78s; every remaining `0.7.0` mention is a historical
Version Notes entry in a command reference. No commit, tag or publication was
performed. Confirm only approved files changed; no option-help/semantic edits; 26 complete references; operational/shared completeness; shorter READMEs; valid examples; preserved caveats; correct master owners; synchronized versions. Record Local Acceptance below. No commit/publication is implied.

### Task 10: Authorized Commit and Live GitHub Gate

**Files:** Update authorization/navigation evidence in this plan; corrective anchor edits only if needed and approved.

**Interfaces:** Consume local acceptance and separate operator authorizations. Produce actual GitHub navigation evidence for 15 targets, or a clearly pending/blocked gate.

- [x] **Step 1: Stop for commit approval.** Present the complete local diff and evidence. Only if approved use an English Conventional Commit; because this final change set carries the `0.7.1` version declarations, prefer a release-scoped message such as `chore(release): publish 0.7.1 documentation migration` and reserve a `docs:`-scoped message for an intermediate documentation-only commit without version declarations. Never commit/publish `version.txt` alone ahead of the migration. Intermediate commits require separate authorization; keep completed release changes together or put a final version commit after the ready migration. **Executed 2026-10-02:** approved by operator; commit `3b7b5b4` `chore(release): publish 0.7.1 documentation migration`.
- [x] **Step 2: Obtain separate remote-publication approval and record the chosen path.** Prefer the staging path; the operator may instead choose direct-to-main. Staging approval is not main-release approval. No tag is required or implicit. Steps 3 and 4 are alternatives, not a sequence; run the one the operator chose. With separate authorization the complete candidate change set, including all four version declarations, may be pushed to a non-default staging branch; that is not a release, because the updater reads `version.txt` from `main`. Do not update `main` or change the default branch before main release approval. **Executed:** operator chose the direct-to-main path and authorized the tag and GitHub release.
- [ ] **Step 3: Staging path.** After separate staging approval, push the completed documentation to the approved staging branch and inspect that branch's rendered files: in each README verify all seven fragments land at the right section, and verify `user-conversation-examples` in the teaching playbook. Designed help URLs remain main URLs, so this step validates fragments, not destinations. Record branch/commit, URL, date and result below. Only when all fifteen staging navigation targets pass may the operator be asked for main release approval; any failed or pending result blocks this path until it is corrected and re-verified. After that approval, publish the complete `0.7.1` change set; updater exposure starts with `main/version.txt` regardless of tags. Recheck rendered links after that publication. **Not used:** direct-main path chosen.
- [x] **Step 4: Direct-main path.** If the operator chooses direct publication, obtain explicit main release approval, publish the complete `0.7.1` change set in one authorized step, then immediately run the same live navigation check on the published main revision and record it below. Release acceptance and any release-completion announcement stay blocked until it passes. **Executed:** `main` pushed `3645c36..3b7b5b4`; annotated tag `v0.7.1` pushed; release published at https://github.com/xtmtd/entomokit/releases/tag/v0.7.1. Live check recorded below.
- [x] **Step 5: Anchor failure and correction (whichever path runs).** If GitHub sanitization breaks an anchor, do not proceed to main publication while `main` is unchanged; if `main` is already published, suspend release acceptance and any release-completion announcement instead, without rolling back or denying that published state. In both cases, with separate approval replace the `id` form with `<a name="anchor-id"></a>`, preserving fragment identity, then rerun local tests, publish the correction under its own authorization, and recheck actual navigation. Neither syntax is accepted without evidence. **N/A:** GitHub emitted every explicit anchor as `user-content-<id>` beside the heading permalink `href="#<id>"` (its own working scheme); no correction needed.
- [x] **Step 6: Evidence quality.** Record final main identity/approval and recheck rendered links after any publication-induced change. Local preview, source matches or HTTP success do not prove fragment navigation; if the executing environment has no browser tool, hand the check to the operator and record it as pending rather than passed. **Executed:** GitHub-rendered HTML evidence recorded below; in-browser click-through was unavailable here, so operator confirmation of actual navigation remains pending.

## Execution Evidence

Update this plan during approved execution, not separate audit files. In each
command batch add source/destination notes and discrepancies beside its existing
checklists. Record original claim, implementation location/current behavior,
user impact and operator disposition.

| Evidence | Current state |
|---|---|
| Implementation authorization/method | Authorized by the operator instruction to execute; inline direct execution per the plan header, no delegation requested |
| Starting branch/ref/worktree changes | Branch `docs/0.7.1-documentation-conventions` created from `main` @ `3645c36`; working tree otherwise clean except this untracked plan and its design |
| Baseline suite and leaf-schema/option-help comparison | `python -m pytest -q` → `408 passed in 36.38s` (Python 3.11.15, pytest 9.1.1), no dependency failures; leaf schema digest `a92a7d659b3722c17dd32b7c97ea2d7089830acf7f406c85b294504f5702568b` |
| Four batches' mapping/manual records | Tasks 2-5 complete: 13 pairs, 65 per-command review items ticked, source/destination and discrepancy notes recorded beside each batch |
| Retained/referenced master-contract audit | Task 8 complete: registry, directory roles, reference-first §5.1, delegated §7, corrected compatibility bullets, new §10 conventions section |
| Pre-bump/final tests and runtime version | `453 passed in 35.78s`; `python -m entomokit.main --version` → `entomokit 0.7.1` |
| Diff review (Task 9 Step 5) | 11 tracked files and 4 new paths changed: the documentation, the two version tests, `help_style.py` (pointer table) and the four version declarations; `git diff --check` is clean. |
| Post-review fixes (operator audit follow-up) | 1) `augment --policy` docs corrected from "JSON array" to the actual JSON object with a `transforms` array; probe confirms a bare array raises `AttributeError: 'list' object has no attribute 'get'` while `{"transforms": [...]}` compiles to a Compose pipeline; no code change (code is source of truth). 2) README log/shutdown bullets scoped to their real applicability: `out-dir/log.txt` for the seven directory commands plus predict/evaluate/cam/export-onnx, `out-dir/logs/log.txt` for train/embed, none for doctor/update/completion; SIGINT handler exists only in extract-frames, segment, measure, synthesize, clean, augment and split-csv. 3) `test_parser_help_links` now expects the `completion` parent and asserts that every registered parser path (parents included, via `_registered_parser_paths`) has an expected URL; mutation probes that delete the completion parent or a functional leaf mapping both fail as intended. 4) Master design §5.2-§5.7 and the §5.1 sub-sections replaced by current-contract conclusions plus links to the command references and the owning design; only the historical old-script-to-new-command rename table remains. 5) Restored model preparation: SAM3 checkpoint source (huggingface.co/facebook/sam3), the Big-LaMa directory contract (`config.yaml` + `models/best.ckpt`, github.com/advimman/lama) in the segment references, and AutoMM/timm weight auto-download plus the supported backbone list in the classify train references. 6) Master design header uses backslash hard breaks, so `git diff --check` is clean. Post-fix verification: `453 passed in 37.14s`, runtime `entomokit 0.7.1`, leaf schema digest unchanged. |
| Post-review fixes (second audit round) | 1) Interruption semantics corrected to match the code: only `segment`, `synthesize`, `measure` and `augment` read the shutdown flag inside their processors, so only they finish the current image and keep completed work on `Ctrl+C`; `extract-frames`, `clean` and `split-csv` install the handler but never read the flag (first `Ctrl+C` sets it and prints a notice, second exits), and classification/operational commands install none. The README interruption rule and the extract-frames, clean, split-csv and classify-train shared-rule sentences now state this per command instead of claiming a shared behaviour. 2) `split-csv` output-directory rule corrected: a non-empty `--out-dir` exits unless `--overwrite` is given (`check_output_dir(..., resume=False, has_resume=False)`), and there is no `--resume`; the previous "reuse an existing --out-dir" wording was wrong in both languages. Post-fix verification: `453 passed in 37.71s`, `entomokit 0.7.1`, leaf schema digest unchanged, `git diff --check` clean. |
| Post-review fixes (third audit round) | 1) `measure` examples no longer point `--mask-dir` at `segment`'s RGB/RGBA `images/`: they now build masks with `segment --annotation-format voc` and read `SegmentationClass/`, and the `--mask-dir` block plus Inputs state that only the first channel is read and alpha ignored, so a dark specimen with a correct alpha channel can measure as empty. 2) Interruption semantics split by command and phase after reading the processors: `measure`/`augment` check the flag per image; `segment` checks between images but its parallel Otsu/GrabCut path has already queued its tasks; `synthesize` checks only while preparing tasks; `extract-frames`/`clean`/`split-csv` never read the flag. README, segment and synthesize documents carry the corrected wording. 3) The custom-policy JSON example uses `size: [512, 512]` (albumentations 2.x requires `size`; `height`/`width` raised `ValueError`), verified by compiling the example through `json.loads` + `build_pipeline`. 4) `--min-count-per-class`/`--max-count-per-class` documented as `--mode count` only, and the filtering example now passes `--mode count` (the ratio branch never forwards them). 5) Restored the CAM current invariants (predictor head plus real labels, saved validation processor as the input-size source, center-crop/whole-specimen-pad mapping, timm fallback, default target layers, model-input-space float32 arrays and raw-magnitude comparability) in master design §5.6 and summarised them in both CAM references. Post-fix verification: `453 passed in 37.00s`, `entomokit 0.7.1`, leaf schema digest unchanged, `git diff --check` clean. |
| Post-review fixes (fourth audit round) | 1) `augment` Outputs no longer claim manifest `original`/`augmented` path entries: the file contains only `preset`, `multiply`, `seed`, `images_processed` and `augmented_images_created`, while the per-image records stay in memory. 2) `synthesize --coco-output-mode separate` documented as accepted but not implemented — the COCO dispatch always accumulates into the unified writer — and the CLI help overclaim is recorded for a separate review instead of being edited here. 3) Output trees corrected: `synthesize` VOC XML lives in `Annotations/`, and `segment`'s YOLO `data.yaml` is written at the output root, not under `labels/`. 4) `classify embed` Version Notes reordered newest-first (`0.7.0` then `0.6.2`). An over-correction was also caught and reverted: `segment` does implement `--coco-output-mode separate` (processor.py:459-624), so its per-image `annotations/` tree line and mode wording were restored. Post-fix verification: `453 passed in 35.98s`, `entomokit 0.7.1`, leaf schema digest unchanged, `git diff --check` clean. |
| Post-review fixes (fifth audit round) | 1) `split-csv --copy-images` now documents the flattening risk: the destination is only `Path(img_path).name` and `shutil.copy2` runs with no collision check, so two inputs whose basenames match inside one split silently overwrite each other; the reference requires unique basenames per split and records that a code-level check needs separate authorization. 2) The master design no longer claims `synthesize` supports `--coco-output-mode separate`: `segment` writes unified or separate JSON, while `synthesize` accepts the flag and still writes the unified file. 3) `clean` naming rules documented and probe-verified: output mirrors the input's parent directory only, the stem is normalised (`a b` → `a_b`, `...x..` → `x`, empty → `untitled`), a case-insensitive `_1`, `_2`, ... suffix is added per directory, and the extension follows `--out-image-format`; the README and master design mirroring wording now state that the file name is rewritten. Post-fix verification: `453 passed in 37.37s`, `entomokit 0.7.1`, leaf schema digest unchanged, `git diff --check` clean. |
| Post-review fixes (sixth audit round) | 1) `segment --annotation-format` documented as effectively defaulting to `coco` (`entomokit/segment.py:184`), so omitting it still writes COCO annotations and the flag cannot disable them; the Purpose/Usage wording was corrected and the help's `None = no annotations` recorded as a discrepancy. 2) `segment --repair-strategy` documented as acting on the source image's union of detected foreground masks (`src/segmentation/processor.py:713-733`) and writing `repaired_images/`, not filling holes in produced masks; the help's "filling holes" recorded as a discrepancy. 3) The teaching playbook's measure demo now consumes VOC binary masks (`out/demo_segment/SegmentationClass/`) produced by the segment demo instead of the RGBA `segment/images/` crops. 4) `measure` Inputs disclose `keep_largest_component` (`src/measurement/service.py:28`): only the largest connected component is measured, so fragmented or multi-object masks are not measured as combined foreground. 5) `measure --resume` documents the recorded parameter guard (`entomokit/measure.py:82`, `src/common/resume.py`) and the continuation example keeps `--pixel-size-um 2.5`. 6) README extras note that `measure` and `synthesize` also import `scikit-image`, declared only by `.[segmentation]`. 7) `classify embed` marks `metrics.csv` as `--label-csv`-only and the metric contract now says k-NN/linear probing (not clustering) are unavailable for singleton classes, which still yield NMI, ARI and purity. 8) README distinguishes the `.[classify]` requirement `autogluon.multimodal>=1.5.0` from `doctor`'s `>=1.4.0` check. 9) `clean` Purpose and master design §5.0 now state parent-directory mirroring with rewritten filenames. 10) Removed the four Usage-duplicating Examples blocks in `classify predict`/`classify evaluate` (both languages). Post-fix verification: `453 passed in 34.82s`, `git diff --check` clean. |
| Post-review fixes (seventh audit round) | 1) `segment --repair-strategy` now states that repaired images use the same encoded sample ID and output format as the segmented image (`src/segmentation/processor.py:732`), correcting the sixth-round "original file name" error. 2) `split-csv --images-dir` documented as a `--copy-images` copy source only (`src/splitting/splitter.py:312`); Inputs/Notes now state the CSV `image` values are copied verbatim and the copied tree is not a relocatable dataset (`beetles/a.jpg` → `images/train/a.jpg` while the CSV still says `beetles/a.jpg`). 3) `augment`/`synthesize --resume` documented as an exact-file-set skip with regeneration and stale-copy deletion (`src/augment/service.py:116-124`, `src/synthesis/processor.py:1284-1310,1407`), plus augment's absent `--seed`/`--policy` check and synthesize's `--coco-bbox-format` guard. 4) `split-csv` unknown split documented as whole-class selection with ratio/count as stop thresholds (`src/splitting/splitter.py:65,141`): the split can overshoot and never splits a class. 5) `extract-frames --resume` documents the recorded parameter guard (`entomokit/extract_frames.py:106`) and the continuation example keeps `--interval 500 --out-image-format png --max-frames 100`. 6) `augment` output naming documents the `len(str(--multiply))` zero-padding (`src/augment/service.py:103`) and the default example uses `source_aug1.png`. Post-fix verification: `453 passed in 35.31s`, `git diff --check` clean. |
| Post-review fixes (eighth audit round) | The re-listed items 1-6 were verified already fixed in the working tree (segment repaired-image naming, split-csv `--images-dir`/flattening/relocatable note, augment/synthesize resume semantics, split-csv whole-class unknown split, extract-frames resume parameters, augment dynamic numbering); no re-application was needed. New findings fixed: 1) `synthesize --resume` no longer claims an exact request match; it now states that only the output-index set and `--coco-bbox-format` are checked and that seed/rotation/colour are not compared (`src/synthesis/processor.py:1284`). 2) `classify train` Outputs now name `out-dir/logs/log.txt` and `out-dir/train.processed.csv` (`entomokit/classify/train.py:164`, `src/classification/trainer.py:182`). 3) `classify predict` Outputs now name `out-dir/predictions/predictions.csv` and the conditional `out-dir/logs/missing_images.txt` (`entomokit/classify/predict.py:142,158,192`). 4) README (both languages) and master design §5.0 now describe `_augN` zero-padded to `len(str(--multiply))` instead of a fixed `_augNN`. Post-fix verification: `453 passed in 32.40s`, `git diff --check` clean. |
| Post-review fixes (ninth audit round) | Shared README output-directory wording was too broad: it said every directory-oriented command could continue a non-empty output directory with `--resume`, although `classify predict`, `evaluate`, `embed`, `cam` and `export-onnx` call `check_output_dir(..., has_resume=False)` and expose only `--overwrite` (`entomokit/classify/*.py`). Both README variants now say that commands with `--resume` may continue and commands without it require `--overwrite`; no CLI behavior changed. A schema-driven metadata probe found no missing requiredness/default/choice statements in the 13 English command references. Exact usage/example command comparison found no duplicates after line-continuation normalization. Fresh verification after this edit: full suite passed, runtime version remained `entomokit 0.7.1`, leaf schema digest remained `a92a7d659b3722c17dd32b7c97ea2d7089830acf7f406c85b294504f5702568b`, and `git diff --check` was clean. |
| Commit authorization/identity | Authorized by operator instruction in this session; English Conventional Commit |
| Staging publication authorization/identity | Not used; operator chose direct-to-main publication |
| Main release authorization/identity | Authorized by operator instruction in this session; tag and GitHub release `v0.7.1` |

### Setup Rulings and Pre-flight

- Ruling: work happens on the branch `docs/0.7.1-documentation-conventions` in the
  existing working directory rather than a separate git worktree, because this
  plan and its design are untracked files that must be updated in place as the
  evidence record, and no commit is authorized before Task 10. Cost if wrong:
  isolation is branch-level only; nothing reaches `main` without approval.
- Ruling: no separate progress ledger or review artifact is created; this plan's
  Execution Evidence section is the record, per spec Section 7 and this plan's
  "Record evidence in this plan".
- Pre-flight shared interfaces: Task 6 must create the readme anchors before
  Task 7's pointers can satisfy `test_parser_help_links`; Tasks 2-5 command
  documents link to README anchors that Task 6 creates, so local-link checks stay
  pending until then; `entomokit/main.py` is touched by Task 7 (description
  pointer) and Task 9 (version fallback) on different lines; Task 1 deliberately
  defers the `with_doc_link` test import to Task 7; Task 1 parametrized test ids
  must contain the hyphenated stems so the plan's `-k` expressions select them
  (verified with pytest 9.1.1).
- Pre-flight conflicts: none blocking Task 1.

### Local Acceptance

- [x] All 13 pairs and 65 per-command manual review items completed.
- [x] All source/destination mappings reconciled; discrepancies reviewed by operator.
- [x] READMEs are substantially shorter and nonduplicated; closure, operational tables, local fragments, skill-setup reachability and help URLs pass.
- [x] Existing option help, schema parameter metadata and CLI behavior preserved.
- [x] Uncovered master invariants retained; dedicated owners linked correctly.
- [x] Four declarations, runtime/version tests and current log examples agree on `0.7.1`.
- [x] Full suite passes; no unresolved dependency/test failures.
- [x] Diff scoped; commit/publication permissions not exceeded.

### Live Navigation Evidence

Every row remains pending until checked on actual GitHub. Record tested
branch/commit, URL and date alongside each result during Task 10.

Checked 2026-10-02 on `main` @ `3b7b5b4`: GitHub rendered each explicit anchor as
`<a id="user-content-<id>">` beside the heading permalink `href="#<id>"` — the same
scheme GitHub uses for its own working heading links, so the fragments resolve. An
actual in-browser click-through was not possible in this environment and remains
operator-confirmable.

| Fragment | English README | Chinese README |
|---|---|---|
| `doctor-command` | Pass | Pass |
| `update-command` | Pass | Pass |
| `completion-command` | Pass | Pass |
| `directory-policy` | Pass | Pass |
| `common-behaviours` | Pass | Pass |
| `classify-commands` | Pass | Pass |
| `assistant-integration` | Pass | Pass |

| Target | Result |
|---|---|
| Teaching playbook `user-conversation-examples` | Pass |
| Final main publication/release acceptance | Published `main` @ `3b7b5b4`, annotated tag `v0.7.1`, release https://github.com/xtmtd/entomokit/releases/tag/v0.7.1 (2026-10-02). In-browser navigation confirmation handed to the operator. |

## Plan Review and Handoff

Ready for operator review, not execution. Implementation starts only after
explicit plan approval and separate implementation authorization. Local and
live-release acceptance are separate; commit, staging push, main publication and
any tag each need their own authorization.
The design's standing rules are unchanged by this plan-writing task; only its
authorization-status wording was reconciled so the record shows the earlier
conversational design approval and the pending confirmation of this round's
revisions.
