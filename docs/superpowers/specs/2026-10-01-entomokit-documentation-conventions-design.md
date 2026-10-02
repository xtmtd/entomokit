# EntomoKit Documentation Conventions Design

Date: 2026-10-01
Status: The original design was explicitly approved in operator conversation, and
drafting this implementation plan was operator-authorized. Revisions made after
that approval await operator confirmation. Plan approval, implementation, commit,
and remote release publication each remain separate gates.
Target release: 0.7.1, a documentation patch with no CLI behavior change beyond
help documentation links and version display/log-header updates.

## 1. Purpose and Scope

The two READMEs contain approximately 1,230 English and 1,225 Chinese lines;
about 66% is usage and command-reference material. Installation examples,
feature lists, shell-completion instructions, and file-level directory trees
repeat information. The command tables omit `completion`, and the Chinese
`segment` section contains annotation details missing from the English version.

The goal is a readable project entry point, complete bilingual command
references, and standing rules that prevent new modules from rebuilding the
same oversized README. Readers should be able to identify a suitable command,
install its dependencies, run a minimal example, and find its complete reference.

Sections 2-7 are standing conventions. Sections 8-10 describe the one-time
0.7.1 migration and acceptance criteria. This is a design, not an execution plan.

In scope:

- Move functional command references into `docs/commands/`.
- Rewrite both READMEs for navigation and setup, preserving useful information.
- Keep `doctor`, `update`, `completion`, and shared behavior in the READMEs.
- Establish bilingual naming, command-document structure, and new-command rules.
- Add documentation links to CLI help without changing parameter descriptions.
- Correct the master design's documentation boundaries and obvious drift.
- Add lightweight documentation tests and update version metadata to 0.7.1
  during approved implementation.

Out of scope:

- Changing CLI syntax, defaults, dependencies, processing, or output semantics.
- Compressing or rewriting existing per-option help strings.
- Creating shared user-reference documents, a documentation site, generated
  manuals, a new changelog, or a second command-index page.
- Renaming `README.cn.md`, restructuring historical specs/plans, or redesigning
  the workflow skill or project architecture.
- Installing packages, running image analyses, publishing, tagging, or committing
  without separate operator approval.

## 2. Documentation Ownership

| Layer | Owns | Does not own |
|---|---|---|
| `README.md`, `README.cn.md` | Project entry, requirements, installation/extras, minimal start, command index, operational commands, shared behavior, AI-assistant entry | Full functional-command parameter reference and worked examples |
| `docs/commands/*.md` | Functional command purpose, complete parameters, input/output contracts, examples, caveats, command-specific history | Copies of shared README rules |
| `docs/superpowers/specs/*.md` | Design intent, architectural decisions, invariants, approved scope; uncovered master-design details remain here | A second complete current CLI reference |
| `docs/superpowers/plans/*.md` | Implementation tasks, verification steps, historical migration facts, and recorded release decisions | A second user manual or permanent source of current contracts/defaults |
| `skills/entomokit-workflow/` | Guided orchestration, approval flow, presentation, validation and teaching | Current CLI parameter truth; runtime schema and command references own it |

The CLI implementation is the executable source of truth. Command documents
are the complete prose reference and must match that implementation. Specs may
name flags when needed to state a decision or invariant. Do not add a second
complete current option surface; retain existing master-design detail without
a dedicated design owner under the reference-first rule below.

The master design follows a reference-first rule. It keeps system purpose,
command and directory navigation, module boundaries, stable architecture
constraints, and details that have no dedicated design owner. When an existing
design already covers a detailed contract, algorithm, or output schema, the
master design keeps the necessary conclusion and a relative link rather than
copying the detail. Existing plans may replace repeated historical implementation
or migration accounts, but they do not become owners of current invariants.
If a plan is the only record of a still-current architectural contract, retain
that contract in the master design and cite the plan as historical rationale.
Current user-facing behavior is also documented in the command reference.

The source-of-truth rule is: current command behavior and parameter descriptions
follow the CLI implementation and runtime schema. When an older README, plan,
skill profile, or other document disagrees, document the current behavior and
record the discrepancy and user impact for operator review. Do not silently
preserve stale wording or silently change executable behavior in a documentation
patch. The skill's `references/command-profiles.md` may repeat parameter names,
defaults, and behavior for presentation, but it is a known duplicate surface
outside this migration's document-coverage checks. Its existing runtime-schema
rule remains in force; a later task may reconcile or remove those copies.

Do not create another design just to shorten the master design; retain uncovered
detail until a separate design is deliberately requested and approved.

README links to same-language command documents. Command documents link back
to the README sections that own shared behavior. The master design links to
this standing design and relevant feature designs; it states once that readers
find current usage through README -> command reference. It does not repeat a
per-command link catalog.

Internal modules without a CLI surface need no README row or command document.
Record an architectural contract in a design spec when one is introduced.

## 3. README Boundary and Structure

Both READMEs follow the same section order:

1. Title, language switch, and one paragraph identifying EntomoKit's purpose.
2. Workflow overview, with optional branches rather than a mandatory full pipeline.
   The existing recommended command order belongs here, once; it is not repeated
   under Directory Input and Output Policy.
3. Requirements and installation, including the extras-to-command mapping.
4. Quick start: a valid invocation with explicit prerequisites and output path.
5. `doctor`, `update`, and `completion`: concise but complete operational reference.
6. Directory input/output policy and common behavior.
7. AI-assistant integration: concise installation instructions, a short
   capability summary, and links to the existing skill and relocated
   conversation examples.
8. Commands: the complete functional-command index and operational-command links,
   split into a main table and a classification table.
9. License, contact, and citation.

There is no hard line-count limit. The migrated READMEs must be substantially
shorter than their current approximately 1,200 lines. Brevity comes from removing
duplication and moving references, not from dropping useful contracts. Absorb
Features into the workflow overview and command-index descriptions; retain
command-specific constraints in the corresponding references rather than keeping
a second feature catalog.

Installation presents one recommended isolated-environment path. Existing
conda, uv, and stdlib-venv alternatives remain available without repeating the
same basic install under multiple deployment headings. Extras use a compact
mapping rather than one heading per package group. Preserve existing special
installation guidance, such as the binary `stringzilla` workaround, unless
verified evidence and operator approval justify changing it.

Shared rules stay here: recursive discovery and output layout, the `segment`
exception, CSV-driven discovery, output-directory safety, logging, device
selection, interruption, and version display. Do not create
`directory-layout.md` or `common-behaviours.md`. State applicability explicitly;
do not turn a behavior of some commands into a claim about all commands.
Command-specific resume and output exceptions remain in their command documents.

The workflow overview describes CLI possibilities, not the AI skill's guided
execution policy. The skill's mandatory cleaning gate and operator-confirmation
rules remain owned by the skill; they are not universal CLI prerequisites.

Keep one `completion` section, not two. `completion` is a README-only parent
command with a required subcommand selection: `bash`, `zsh`, or `fish`. Each
shell has its own leaf parser and `--install` option; this is not a positional
argument with choices. Document these subcommands, their options, install paths,
and zsh activation requirement. `doctor` and `update` similarly retain their
complete user-facing option descriptions.

Each operational command has an anchored level-two README section. Put option
names and aliases in the first column of its option table. `doctor` currently
has no user-settable options; omit its table rather than adding an empty one.
For `completion`, use a subcommand table with shell names in the first column
and each shell's option names in the second. These bounded tables, not mentions
in prose or examples, are the coverage surface. Exclude built-in help.

Detailed model preparation belongs with the commands that need it: SAM3 and
LaMa in `segment`, AutoMM/timm training prerequisites in `classify-train`, and
specific inference/embedding/CAM prerequisites in the respective documents.
README may carry brief pointers, not a growing model catalog.

Keep the current AI-assistant installation instructions in README in compact
form, along with the skill name, one-line purpose, and a link to
`skills/entomokit-workflow/SKILL.md`. Retain a short capability summary covering
guided pipeline execution, runtime parameter validation, error recovery, resume
support, and teaching/demo mode; link the detailed skill rules to `SKILL.md`.
Move only the user conversation examples to
`skills/entomokit-workflow/references/teaching-playbook.md`, in a new
`## User Conversation Examples` section preceded by
`<a id="user-conversation-examples"></a>`. Preserve English and Chinese examples
there and link both READMEs to this explicit anchor. Do not move setup into an
agent-only demo section or change the skill's execution or approval policy.

Remove the detailed README file tree. A short project-structure sentence may
link to `entomokit/`, `src/`, and the master design; no maintained per-file tree
is required.

Add stable explicit HTML anchors using `<a id="anchor-id"></a>` immediately
before the relevant README heading: `doctor-command`, `update-command`,
`completion-command`, `directory-policy`, `common-behaviours`, `classify-commands`,
and `assistant-integration`. Use the same IDs in both languages. Under
`## Commands` / `## 命令`, place a main table for seven functional commands and
three operational entries, followed by the `classify-commands` anchor immediately
before `### Classification Commands` / `### 分类命令` and a second table for the
six classification leaves. Do not insert a standalone HTML anchor inside a table.
The subsection is navigation, not a separate group manual. Avoid relying on
translated heading slugs. Source checks or local previews do not prove GitHub
navigation; the Section 10 release gate requires actual GitHub-rendered checks.

## 4. Naming, Language, and Command Skeleton

Every command reference has an English file and a `.cn.md` Chinese twin in the
same change. Keep the existing `.cn.md` convention; do not rename either README.
Filenames are flat and follow the full command path joined by hyphens:

```text
docs/commands/segment.md
docs/commands/segment.cn.md
docs/commands/classify-train.md
docs/commands/classify-train.cn.md
```

Both files of a pair use the same literal invocation as their H1, for example
`# entomokit classify train`, including in `classify-train.cn.md`. Immediately
below it place a language switch:

```markdown
[English](classify-train.md) | [中文](classify-train.cn.md)
```

English and Chinese files share section order and semantic content. Keep flags,
paths, filenames, CSV fields, identifiers, and literal choices unchanged in
Chinese text. Each README links to its own language's document.

| Order | English | Chinese | Requirement |
|---|---|---|---|
| 1 | Purpose | 目的 | Required; role and scope in one or two short paragraphs |
| 2 | Usage | 用法 | Required; minimal valid invocation |
| 3 | Parameters | 参数 | Required; every user-settable command option |
| 4 | Inputs | 输入 | When applicable; schemas, path resolution, prerequisites |
| 5 | Outputs | 输出 | When applicable; files, directories, fields, conditional outputs |
| 6 | Examples | 示例 | When needed; distinct complete worked examples |
| 7 | Notes | 说明 | When needed; interactions, exceptions, scientific caveats |
| 8 | Version Notes | 版本注记 | When recoverable command history exists, or a new command is introduced |

The skeleton sections are H2 (`##`). Omit inapplicable sections; do not create
empty headings. Repeated examples in Usage and Examples are unnecessary.
Headings do not carry release numbers. Under Parameters use an H3 (`###`) per
option; aliases share a block:

```markdown
### `--out-dir`, `-o`

Required. No default. Output directory for this run.
```

Each block states requiredness, default (or no default), enumerated choices where
defined, and the option's complete meaning and constraints. Boolean defaults
are explicit; distinguish a missing value from an automatically resolved value.
Do not combine unrelated options merely because their names are similar.
The description must be at least as informative as current option help. Long
contracts live in Inputs, Outputs, or Notes with a pointer from the option block.

The built-in `--help` action need not have a block. Root version flags belong
in README. If a future functional command genuinely introduces positional
arguments, document them explicitly and extend the coverage test deliberately;
do not mislabel them as options. Today's `completion` shell subcommands stay
in README and are checked as a nested parser tree, not as positional arguments.

Version Notes preserves known behavior changes, newest first. Do not invent
release dates or historical entries. From 0.7.1 onward, user-visible behavior
changes update every affected command document's Version Notes in the same
change. A documentation-only reorganization does not imply that every command
changed behavior in 0.7.1. Compatibility warnings remain visible near the
relevant current behavior as well as the history entry when needed for safe use.

## 5. CLI Help Boundary

Existing per-option descriptions remain unchanged in this migration. Do not
shorten them, move their warnings out of the CLI, or insert URLs into their text.
Any later proposal to rewrite a parameter description requires its own review.

Command-level help is an index: purpose, concise examples, and a document link.
Long non-parameter explanations belong in the corresponding command document.
Warnings necessary to choose safe inputs or interpret results stay visible at
the point of use; do not hide them merely to shorten help.

Current functional-command descriptions already generally contain a short
summary and examples. 0.7.1 retains these blocks and appends the documentation
URL as the final line of the description, so it appears before the option
listing; preserve an empty existing description by adding only the pointer.
It must not alter option help. The current `RawTextHelpFormatter` preserves
these URL lines without wrapping. No speculative long-block removal is needed.
If implementation discovers a genuinely long non-option block, apply the source-
of-truth rule in Section 2 and present any proposed migration for operator
confirmation before removing it.

Help links use the published GitHub URL, not a filesystem-relative `docs/` path:

```text
https://github.com/xtmtd/entomokit/blob/main/docs/commands/classify-train.md
```

Root help links to
`https://github.com/xtmtd/entomokit/blob/main/README.md`. Every functional group
has a stable README group anchor, an H3 command subsection/table, and a group-level
help pointer to that anchor; the current `classify` group uses
`README.md#classify-commands`. `doctor` and `update` link to their explicit
README anchors. The `completion` parent and all its shell leaf parsers link to
`README.md#completion-command` using the same published URL base. Functional
leaf commands link to their own English reference, whose language switch
provides the Chinese route.

The `main` branch may be newer than an installed release. This is an accepted
limitation; version-aware links are not part of this patch. Add one
`DOCS_BASE_URL` constant and a `with_doc_link()` helper to
`entomokit/help_style.py`, alongside `with_examples()`. Its shape is
`with_doc_link(description, "docs/commands/segment.md")`: it appends the
published URL formed from the base and path. Use the helper from parser
registration for root, group, operational-anchor, and leaf pointers instead of
scattering URL construction across command modules. Do not change `argparse`
parsing or completion behavior to implement documentation links.

## 6. New-Command Checklist

A new functional CLI command is complete only when the same change includes:

1. CLI registration and both command-reference language files.
2. The required skeleton, complete parameter blocks, and valid minimal usage.
3. Input/output descriptions and applicable limitations, including safe output use.
4. Same-language README index rows pointing to the command's own document.
5. A command-level help pointer and retained point-of-use warnings. For a new
   functional group, add a stable group anchor, an H3 subsection and command
   table in README, and a group-level help pointer to that anchor; the group
   itself need not get a leaf command document.
6. Version Notes for the introduced or changed behavior.
7. A design record when a new architectural contract or invariant is introduced,
   without a duplicated exhaustive option table.
8. Passing documentation checks and manual verification of defaults/choices.

The README-only exception list is explicit: `doctor`, `update`, `completion`.
Do not put a future functional command in README to avoid writing its reference.
A new operational exception requires an explicit revision to these conventions.

## 7. Lightweight Documentation Checks

Add focused checks in `tests/test_docs.py`, using the existing local pytest
flow and stdlib only. This repository has no CI configuration; these checks are
not remotely enforced until a future CI task adds them. Use the existing public
`entomokit.cli_schema.build_command_schemas()` for executable leaf paths and
options, choices, defaults and help; it already filters built-in help and has
its own covered `_leaf_commands()` traversal. Use
`entomokit.main._build_parser()` separately for root help, functional-group help
(such as `classify`), and the README-only `completion` parent and shell leaves.
The `skills/entomokit-workflow/scripts/export_cli_schema.py` wrapper is the
existing runtime-schema entry point for the skill; it is not a second source of
truth. Do not execute command callbacks or import optional inference/training
backends.

Required checks:

- **Command closure:** call `build_command_schemas()` and treat its keys as
  executable leaf paths. For each path, replace spaces with hyphens to derive
  the document stem. Exclude `doctor`, `update`, and all `completion <shell>`
  leaves from the functional-document set; compare the remaining 13 stems with
  discovered English documents by exact set equality in both directions (no
  missing document and no stale document). Inside that H2 section, discover the
  main index table and the command table under every H3 subsection; the current
  layout has the main table plus the `classify-commands` H3 table, and the number
  of tables is not fixed. The union of all discovered row labels must be an exact
  set equal to the current functional document stems plus `doctor`, `update`, and
  `completion`; functional rows point to exact same-language document targets and
  operational rows point to their anchors. Do not read tables outside the Commands
  H2. Reject duplicate rows across the discovered tables. The current schema has
  18 leaf keys and 16 README index rows in two tables; derive sets from the schema
  rather than these snapshot counts, and extend discovery when a new functional
  group adds its own H3 table.
- **Bilingual structure:** one-to-one English/Chinese pairs, a valid language
  switch after the identical invocation H1, required H2 sections, and matching
  normalized H2 sequences. Define the Section 4 translation/order mapping once
  as a test-local constant, reused by all structure checks, with a comment linking
  this spec. Any skeleton change must update this constant and both document
  languages in the same change. Do not build a parser for this design's table.
  File existence alone is insufficient.
- **Option coverage:** for each functional schema key, compare the option names
  extracted from H3 headings inside Parameters with that schema entry's
  `parameters[*].options`, bidirectionally. In both command documents and README
  operational tables, extract code spans and accept tokens matching
  `^--?[A-Za-z0-9][A-Za-z0-9-]*$`, ignoring commas, pipes, and surrounding prose;
  remove `-h` and `--help` globally. Include aliases and exclude the already-
  filtered help action. A mention in an example cannot satisfy coverage.
  Check both languages. In each README, locate the operational section by its
  explicit anchor and bound it by the next level-two heading. Extract complete
  option tokens from the option-name table column for `doctor` and `update`; a
  missing table is valid only when the schema entry has no user-settable options.
  For `completion`, compare the subcommand table's shell names with the schema
  keys' `completion <shell>` suffixes, then compare each row's option column with
  that shell's schema options. Thus `--install` coverage is checked per shell;
  a mention elsewhere cannot satisfy it. Compare sets, not order.
- **Local links:** check Markdown links in both READMEs and all command documents
  against the source document's directory; directories are valid targets too.
  Resolve fragments to explicit `id` or `name` anchors in their targets.
  Check same-language command/README targets, except explicit language switches.
  Do not hardcode which shared-rule links each command must carry; review their
  relevance manually. Do not reimplement GitHub's multilingual heading slugger.
  Ignore fenced code blocks, since example Markdown links are not live links.
- **Help links:** perform a simple expected-URL substring check on root, group,
  and leaf parser help, including the `completion` parent and shell leaves.
  Existing description rendering preserves the URL line; no wrapping or
  whitespace-normalization framework is needed. This small check keeps a new
  command from silently omitting its pointer.

Defaults, allowed values, scientific interpretation, translation fidelity, and
which shared-rule sections a command should link are reviewed manually against
code and rendered help under the source-of-truth rule in Section 2. Rendered
fragment navigation is reviewed manually. Do not introduce a natural-language
parser or generated documentation pipeline. The implementation plan may add
evidence-only checks, such as a leaf-schema and option-help digest comparison that
must stay unchanged, provided they do not alter CLI behavior or widen this
scope. Preserve existing behavioral tests;
new documentation tests complement them. The implementation plan must deliver
these documents in command batches. Each batch's plan subsection contains a
checked Markdown checklist for each command covering defaults, choices, inputs,
outputs, and known historical notes; do not create a separate review artifact.

## 8. 0.7.1 Migration Inventory

Thirteen functional command documents, twenty-six files:

| Document stem | CLI command | Current README source |
|---|---|---|
| `extract-frames` | `extract-frames` | Extract Frames Command |
| `segment` | `segment` | Segment Command, SAM3/LaMa Model Requirements |
| `measure` | `measure` | Measure Command |
| `synthesize` | `synthesize` | Synthesize Command |
| `clean` | `clean` | Clean Command |
| `augment` | `augment` | Augment Command |
| `split-csv` | `split-csv` | Split-CSV Command |
| `classify-train` | `classify train` | Classification prerequisites, train reference, augmentation presets |
| `classify-predict` | `classify predict` | predict reference and ONNX interpretation |
| `classify-evaluate` | `classify evaluate` | evaluate metrics and diagnostics |
| `classify-embed` | `classify embed` | embed outputs, metric definitions, compatibility and memory warnings |
| `classify-cam` | `classify cam` | CAM architecture, preprocessing, array semantics and limitations |
| `classify-export-onnx` | `classify export-onnx` | export-onnx reference and label mapping |

Migrate the union of valid information from both READMEs, not just a translation
of the shorter side. In particular, retain Chinese `segment`'s VOC mask layout,
`area` interpretation, and polygon-versus-bbox semantics in both documents.
Retain the 0.6.2 embedding-metric comparability warning and the statement that
0.7.0 did not change that algorithm. Preserve CAM raw/normalized semantics,
RGBA synthesis prerequisites, resume limitations, and output deletion warnings.

Before replacing content, map every meaningful paragraph to its destination and
compare documented flags/defaults/choices to the real CLI schema. Apply the
source-of-truth rule in Section 2. Ordinary de-duplication and clear rewriting
of consistent information are allowed. Repair moved relative links.

README command tables gain `completion` and exact same-language links. Shared
content and operational references are consolidated rather than duplicated.
Detailed model instructions and AI conversation examples move as specified in
Section 3; compact AI installation instructions stay in README. Useful content
is not discarded to meet a length target.

## 9. Master Design and Version Updates

In [the master design](2026-03-24-entomokit-refactor-design.md):

- Correct the command tree to the current registry, including `measure`,
  `augment`, `doctor`, `update`, and `completion`.
- Replace the stale per-file tree with directory-level responsibilities matching
  the repository. Remove absent `scripts/` and `add_functions/`; account for
  `src/augment/`, `src/doctor/`, `src/measurement/`, `src/lama/`, and `src/sam3/`.
  Both `src/segmentation/` and `src/segmentation.py` exist; do not infer that one
  replaces the other or change their ownership as part of this documentation patch.
- Apply the reference-first rule: keep purpose, dispatcher shape, module
  boundaries, stable cross-module constraints, and detail not covered by an
  existing dedicated design. Replace duplicated design detail with a conclusion
  and its design link. Plans may replace repeated historical implementation and
  migration accounts only; they must not displace a current invariant from the
  master design. Command documents own the current user-facing parameter and
  behavior reference, not architectural decision history.
- Do not create or propose a new design merely to make the master design shorter.
  If no existing dedicated design covers a current architectural detail, retain
  it in the master design, even if a plan records its original implementation.
  A later request may deliberately split it, subject to normal approval.
- Repair the misleading overall implementation status and explicitly distinguish
  historical refactor decisions from standing conventions. Update the last-updated
  note during approved implementation, not merely because this draft exists.
- Add a short Documentation Conventions section: ownership, `.cn.md` mirroring,
  complete command parameters, help boundary, reference-first master-design
  rule, new-command checklist, and a link to this design. Do not copy this entire
  document into the master design.

Keep this a targeted maintenance pass, not an architecture rewrite. The result
may be shorter than the current design; shortening is a success when every
removed decision remains reachable through an existing owner document.

Use this map to distinguish dedicated design ownership from historical evidence.
Do not delete a whole block when its source covers only part of it. Historical
syntax is not a replacement for current CLI definitions.

| Detail | Existing source and permitted use |
|---|---|
| Segment bbox/mask annotation semantics | [Annotation design](2026-04-13-segment-annotation-semantics-design.md): dedicated design; reference covered details |
| Measure definitions, geometry, and quality caveats | [Measurement design](2026-04-14-measurement-from-sam3-mask-design.md): dedicated design; reference covered details |
| CPU concurrency and SAM3 serial boundary | [Parallelism design](2026-07-10-segment-cpu-parallelism-design.md): dedicated design; reference covered details |
| Original classification implementation | [Phase 3 plan](../plans/2026-03-24-phase3-classify.md): historical tasks; keep current architectural contracts in master design |
| CAM array migration and rationale | [CAM plan](../plans/2026-09-15-classify-cam-unnormalized-arrays.md): historical decisions; retain current raw/normalized and comparison invariants in master design |
| Embedding-metric corrections | [Metric plan](../plans/2026-09-22-classify-embed-metrics-corrections.md): migration history; retain current metric/comparability invariants in master design |
| Recursive layout, seeds, and 0.7.0 migration | [0.7.0 plan](../plans/2026-09-22-entomokit-v070-recursive-layout-and-cli-updates.md): release history; retain current cross-module directory contracts in master design |
| Operational commands and output-directory safety | [Completion/evaluation design](2026-07-05-entomokit-completion-evaluate-design.md), [operational safety design](2026-07-09-shell-completion-update-resume-overwrite-design.md): reference covered design details; retain later changes not covered there |

The corresponding command references still describe current behavior, including
CAM limits, metric comparability, and directory exceptions. A historical plan
link is additional rationale, never their sole current reference.

During approved implementation update all live version declarations from 0.7.0
to 0.7.1: `setup.py`, `version.txt`, `entomokit/_version.py`, and the fallback in
`entomokit/main.py`. Update expectations in `tests/test_package_version.py` and
`tests/test_main_cli.py`. Rename version-bearing test function names and docstrings
in the same change; prefer version-neutral names (for example
`test_setup_version_matches_target` and `test_save_log_header_contains_version`)
while keeping the expected release explicit in assertions. README log-header
examples use 0.7.1; historical statements retain their original version. Search
for active declarations rather than replacing every historical mention.

A local version edit does not authorize commit or publication, but publishing
`main/version.txt` is an effective release signal: `entomokit update` reads it
and installs the repository's default-branch HEAD through `pip install git+...`,
not a pinned tag. Do not commit or publish `version.txt` alone ahead of the
completed documentation migration. Keep all version declarations, final READMEs,
command references, and help pointers in the same approved final release change
set, or publish a final version commit only after all those changes are ready.
Apply the bump after migration is complete, then run the final checks. Publishing
that change to main requires separate explicit release approval; a tag is not
required for users to see 0.7.1. Do not change updater behavior in this patch.

## 10. Acceptance and Approval Gates

Before calling implementation complete:

1. Both READMEs use the agreed entry-point structure, are substantially shorter,
   and contain no full functional-command reference blocks.
2. All thirteen command references have complete same-language mirrors, valid
   minimal examples, complete options, and correct input/output descriptions.
3. Every migrated contract has a destination. Per the source-of-truth rule in
   Section 2, current command prose matches the CLI schema and code; any
   README/code discrepancy is recorded with its user impact, corrected in the
   current documentation, and separately reviewed by the operator rather than
   silently preserved or silently changing code.
4. README operational sections and shared behavior remain complete, nonduplicated,
   and explicit about applicability. Skill setup information remains reachable.
5. New documentation checks and the existing test suite pass. Manually review
   help and both languages for defaults, choices, constraints, and relevant
   shared-rule links. Local rendering is useful but cannot satisfy the separate
   live GitHub release gate below; no new rendering dependency is required.
6. Help has working intended pointers, and existing per-option help strings are
   unchanged. Parsing, defaults, and non-help analysis output remain unchanged.
   The added help links and the approved 0.7.1 version display/log-header changes
   are intentional output differences, not violations of this boundary.
7. The master design states the conventions, references this document, uses
   reference-first links for covered design detail and historical accounts,
   and no longer maintains a stale file tree. Current architectural details
   without a dedicated design stay in master design; plans alone do not replace
   them. Current usage remains available in command references.
8. Version declarations and runtime `--version` output agree on 0.7.1. Update
   tests use a mocked remote; a real update installation is not a test gate.
9. Review the final diff for unrelated edits and confirm no commit/tag/publication
   was performed without separate authorization.

### Live GitHub Release Gate

After the README changes are first pushed with operator approval, check actual
GitHub-rendered navigation for seven anchor IDs in each README and
`user-conversation-examples` in the skill reference (15 targets, eight distinct
IDs). Each must survive GitHub sanitization and land at its intended section.
A local/GitHub-compatible preview is not a substitute. Prefer verification on
an approved staging branch before publishing the version-bearing changes to main.

If first publication is directly to main, perform this check immediately after
publication; release acceptance and any release-completion announcement remain blocked
until it passes. That staging choice does not remove the immediate updater
exposure described in Section 9. If an `id` anchor fails on GitHub, replace it
with `<a name="anchor-id"></a>` retaining the same fragment and recheck actual
navigation; accept neither syntax without evidence. Source fragment checks must
recognize the chosen explicit `id` or `name` form. Any corrective publication
still requires operator authorization. Record this gate as pending, not passed,
while the changes remain local.

The original design was explicitly approved in operator conversation before the
implementation plan was drafted; the revisions made since then await operator
confirmation. Plan approval, implementation, commit, and remote release
publication each remain separate explicit gates, and drafting the plan implies
none of them.

## 11. References

- [EntomoKit master refactor design](2026-03-24-entomokit-refactor-design.md)
- [EntomoKit skill design](2026-03-25-entomokit-skill-design.md)
- [Completion and evaluation design](2026-07-05-entomokit-completion-evaluate-design.md)
- [Completion, update, resume, and overwrite design](2026-07-09-shell-completion-update-resume-overwrite-design.md)
- [Embedding-metric corrections plan](../plans/2026-09-22-classify-embed-metrics-corrections.md)
- [0.7.0 recursive-layout and CLI updates plan](../plans/2026-09-22-entomokit-v070-recursive-layout-and-cli-updates.md)
- Operator-provided reference: OTU-Former documentation-conventions design,
  dated 2026-09-29. Its separation of entry points and references is adapted,
  not copied: EntomoKit retains shared behavior and all three operational
  commands in README and handles nested argparse commands rather than Typer apps.
