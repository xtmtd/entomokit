# Classify CAM Unnormalized Array Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `classify cam`'s boolean `--save-npy` flag with an explicit three-valued contract — `--save-npy {none,raw,normalized}` — where `raw` writes the unnormalized positive CAM (magnitude preserved), `normalized` reproduces the previous per-image min-max values, `none` (default) saves nothing, the overlay figure always uses a separate min-max copy, and version 0.6.1 is released.

**Architecture:** A local `UnnormalizedCAMMixin` subclasses each pytorch-grad-cam method and overrides `compute_cam_per_layer` and `aggregate_multi_layers`, so both internal `scale_cam_image` calls are skipped while ReLU, multi-layer averaging, and resize-to-input-size stay identical to upstream. `process_image` then owns two arrays: the raw CAM that is saved, and a min-max copy used for `map_cam_to_original` and the overlay. There is exactly one CAM code path; only the saved array depends on `--save-npy`'s value.

**Tech Stack:** Python 3.9+, pytorch-grad-cam 1.5.5, PyTorch, NumPy, OpenCV, argparse, pytest.

## Approved Scope Decisions

Recorded from review before implementation started:

- No separate `-design.md` spec. The design rationale lives in this plan's `Design Decisions` section instead of being duplicated into `docs/superpowers/specs/`.
- `--save-npy` takes a required value from `{none,raw,normalized}`; the bare flag is removed. The command examples (`README.md:898`, `README.cn.md:909`) therefore become `--save-npy raw`, the mode an example that wants arrays should use.
- `grad-cam` is **not** pinned in `setup.py`. The mixin's drift protection is a test, not a version constraint. See `D1`.
- The internal save-mode value is a string, never a bool: `"none" | "raw" | "normalized"`. See `D3`.

## Measurements Taken While Writing This Plan

All numbers below were produced against the environment snapshot (`grad-cam 1.5.5`, the repo's conda env) before implementation, using the single `Task 2` fixture (`_tiny_conv_model` + `_fixed_cam_input`, 96×96 gradcam). They are **reference only, not acceptance criteria**, and **not** test assertions — see `D1`. An earlier revision quoted `2.100544516e-04` from a different throwaway fixture; that row is deleted so every number here belongs to one fixture.

| Measurement | Value |
| --- | --- |
| Library output for the fixed tiny model (`official.max`) | `0.99999988` |
| Same input, raw mixin output (`raw.max`, gradcam) | `2.324e-04` |
| `max\|min-max(lib) - lib\|` (entomokit's current second pass) | `1.192e-07` |
| End-to-end figure, official vs raw extractor, `process_image` → PNG (128×96 input, fov dimming active) | `scorecam`: `44` of `73728` pixels differ, max `2` LSB. `gradcam`, `gradcampp`, `layercam`, `eigencam`, `ablationcam`: `0` |
| Current unmodified `CAM_METHODS["gradcam"]` | `max == 0.99999988`, so the raw-CAM test fails before implementation, as intended |

### Order-of-operations check (all six CAM methods)

The proposed change moves min-max from *before* the resize to *after* it. This table measures whether that changes the result. `official` is the untouched pytorch-grad-cam class; `raw` is the mixin; `display` is `min-max(raw)`.

| Method | `official.max` | `raw.max` | `max\|display - official\|` |
| --- | --- | --- | --- |
| gradcam | `0.99999988` | `2.324e-04` | `1.788e-07` |
| gradcampp | `0.99999988` | `1.340e-02` | `2.384e-07` |
| layercam | `0.99999988` | `4.648e-04` | `2.384e-07` |
| scorecam | `0.99999988` | `5.918e-03` | `1.788e-07` |
| eigencam | `0.99999988` | `1.833e-01` | `1.788e-07` |
| ablationcam | `0.99999988` | `2.658e-02` | `2.384e-07` |

Every method agrees to float32 rounding (~1e-7), so `min-max` and `resize` are interchangeable here: `min-max` is affine-invariant, and `cv2.INTER_LINEAR` resize is an affine map (kernel weights sum to 1), so `resize(min-max(x)) == min-max(resize(x))` up to floating point. The `1e-7` term inside upstream's `scale_cam_image` divisor is the only source of drift.

### CLI parsing behaviour

`choices=["none", "raw", "normalized"]`, `default="none"`, value required, measured on Python 3.11:

| argv | `save_npy` |
| --- | --- |
| (absent) | `"none"` |
| `--save-npy none` | `"none"` |
| `--save-npy raw` | `"raw"` |
| `--save-npy normalized` | `"normalized"` |
| `--save-npy` (no value) | `SystemExit`: `expected one argument` |
| `--save-npy bogus` | `SystemExit`: `invalid choice` |

## Design Decisions

### D1. Skip upstream normalization with a mixin, and detect upstream drift with a test, not a version pin

Upstream pytorch-grad-cam 1.5.5 normalizes a CAM twice:

- `pytorch_grad_cam/base_cam.py:166` — `scaled = scale_cam_image(cam, target_size)`
- `pytorch_grad_cam/base_cam.py:175` — `return scale_cam_image(result)`
- `pytorch_grad_cam/utils/image.py:162` — `scale_cam_image` applies `(x - min) / (1e-7 + max)` per image

Both call sites live inside `BaseCAM.compute_cam_per_layer` and `BaseCAM.aggregate_multi_layers`. Patching `base_cam.scale_cam_image` reaches only the min-max step. Any future signed (positive/negative) CAM must also drop the rectification at `base_cam.py:165` (`cam = np.maximum(cam, 0)`) and `base_cam.py:173` (`cam_per_target_layer = np.maximum(cam_per_target_layer, 0)`), which no module-level monkeypatch can express without patching NumPy globally.

The mixin is therefore the seam where the rectification policy, the per-layer normalization policy, and the aggregation policy live together. It is chosen now rather than deferred because signed CAM is a stated direction, and a `scale_cam_image` patch would be thrown away on that change. A monkeypatch of `scale_cam_image` was measured to produce identical raw values, so choosing the mixin costs nothing numerically today.

Do NOT add a `rectify`/`scale` policy option in this change. The mixin is the extension point; the option is added with the signed-CAM feature.

**Drift protection (the important part).** `setup.py:56` declares `"grad-cam",` with no version constraint, while the mixin mirrors 1.5.5 internals. That combination must be closed by a test, not by a pin:

- **Structural proof that normalization is bypassed** (the primary evidence): `test_raw_cam_does_not_call_scale_cam_image` replaces `pytorch_grad_cam.base_cam.scale_cam_image` with a function that raises, then asserts the raw class still returns a map while the official class raises under the same patch. The official half is the control that keeps this from passing vacuously. It depends on no numeric value at all.
- Record the mirrored version and both method signatures in the mixin docstring.
- **Display-equivalence relationship** (secondary): for the same input, `max|min-max(raw_mixin) - official_gradcam| < 1e-6`, plus `official.max() == approx(1.0)`. Both sides run in the same process, so this is stable across PyTorch/BLAS/OpenCV versions while still failing on a changed resize, a changed ReLU policy, a changed aggregation, or a new non-affine upstream step. The `official.max() ≈ 1.0` half additionally catches an upstream that stops normalizing.
- **What this does not cover, by design:** a change that scales raw CAM magnitudes (for example upstream multiplying raw values by a constant) is invisible to a min-max comparison and would not be caught. Raw-magnitude semantics across arbitrary upstream upgrades are therefore *not* guaranteed by the suite; the docstring records the mirrored version, and a dependency upgrade requires re-reading these two upstream methods.
- Do **not** assert `raw.max() == 2.1005445e-04`. That value depends on PyTorch, torchvision, OpenCV, BLAS, and CPU/GPU, and would produce false failures on unrelated upgrades. It no longer appears in this plan or in any test.
- Do not pin `grad-cam==1.5.5`: it would freeze users of the `classify` extra and can conflict with `autogluon.multimodal`'s own constraints. A runtime version warning is also rejected: it is unactionable for users and the structural test already covers the case that matters.

### D2. `--save-npy` takes an explicit value from `none | raw | normalized`

```python
p.add_argument(
    "--save-npy",
    choices=["none", "raw", "normalized"],
    default="none",
    help="Save CAM arrays to arrays/*.npy: 'none' (default), 'raw' "
         "(unnormalized positive CAM, magnitude preserved), or 'normalized' "
         "(per-image min-max in [0, 1]).",
)
```

`argparse` rejects an unknown value, and because the value is required, `--save-npy` on its own fails at startup with `expected one argument` instead of silently selecting a mode. The CLI surface, the internal state (`D3`), the schema, and the completion menus all carry the same three strings, so no consumer needs a special case for a value-less flag.

**This is a deliberate breaking change to a released flag.** `README.md:898`, `README.cn.md:909`, and the option row in `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md:353` use the bare form and are updated in `Task 4`. `runs/cam/log.txt:1` also contains the bare form, but `runs/` is gitignored (`.gitignore:53`) and `git ls-files runs` is empty: that file is an untracked record of a past local run. It is therefore left exactly as it is — it is not a live document, it is not covered by any consistency check in `Task 4` Step 6, and rewriting it would falsify the record of what was actually executed. `Task 7` Step 3 uses the new syntax instead of replaying that line. The break is loud and immediate (`expected one argument`), and it lands in the same release that already changes what the flag writes (`D4`), so users absorb one migration instead of two.

Rejected: `nargs="?"` with `const="raw"`, which would keep the bare flag as an alias. It gives the flag two meanings, and keeping the alias machine-readable would require an extra `const` field in `entomokit/cli_schema.py`. The compatibility it buys is worth little when the values the flag stores are changing anyway.

Because the value is required, `entomokit/param_guard.py` and `entomokit/workflow_gate.py` need no optional-flag awareness: a workflow passes the explicit string, for example `--save-npy raw`. An optional-value flag would push the burden of reconstructing the alias onto those layers.

No `cli_schema.py` change is needed: the action is a plain `store` with `choices`, so `_value_hint` already returns `"none | raw | normalized"` and the existing `test_cli_schema.py` tests keep passing. Slot completion needs no edit either: `entomokit/completion.py:32` builds the parser with `_build_parser()` and reads `action.choices` at `:26-29`, so the three values appear in the bash/zsh/fish menus automatically.

### D3. The internal save-mode value is a string, never a bool

`save_npy` keeps its name but becomes `Literal["none", "raw", "normalized"]` with default `"none"`. The `argparse` `dest` is already `save_npy`, so the CLI and the internal parameters match and there is no `None`/`False`/string mixture:

- `prepare_output_dirs(out_dir, save_npy)` creates `arrays/` when `save_npy != "none"`.
- `process_image(..., save_npy)` saves the raw array when `save_npy == "raw"`, the min-max array when `save_npy == "normalized"`, and nothing when `save_npy == "none"`.
- `run_cam` forwards the same string.

Existing test call sites passing `save_npy=False` must be updated to `save_npy="none"` so that `False` stops being accepted internal state: `tests/test_classification_cam.py:18,76,470,508,569` and `tests/test_resume_flags.py:403`.

### D4. Intentionally changed behavior

`--save-npy raw` now writes unnormalized values instead of per-image min-max values, and `--save-npy` requires an explicit value.

In-repo documents disagree today, which is what the upstream issue reported:

| Location | Current text | Accurate? |
| --- | --- | --- |
| `README.md:919` | `Raw CAM arrays (with --save-npy)` | no |
| `README.cn.md:930` | `原始 CAM 数组（使用 --save-npy）` | no |
| `entomokit/classify/cam.py:94` | `Save raw CAM heatmaps as NumPy arrays.` | no |
| `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md:378` | `模型输入空间的归一化 CAM 数组，float32` | yes |

The change resolves the conflict in the direction the user-facing docs already promised.

**Array and figure compatibility, stated precisely.** The evidence has two levels and they support different claims.

*Model-space mask.* `--save-npy normalized` writes `min-max(raw)`, which equals the previous implementation's array to float32 precision — not bit-for-bit. For the recorded fixture, `max|min-max(raw) - official|` is `1.8e-7`–`2.4e-7` across all six methods (see the order-of-operations table). That is a fixture measurement, not a universal bound. The mechanism behind it is general:

- `min-max` is affine-invariant: `min-max(a·x + b) == min-max(x)`.
- `cv2.INTER_LINEAR` resize is affine: `resize(a·x + b) == a·resize(x) + b`.
- Therefore `min-max(resize(x))` and `resize(min-max(x))` agree up to floating point, for a single target layer, which is the only configuration `prepare_cam` produces (`target_layers` always has exactly one entry).

*Rendered figure — mask equality does NOT imply pixel equality.* `show_cam_on_image` applies a global division (`cam = cam / np.max(cam)`) and `np.uint8` quantization after `map_cam_to_original`, so the overlay was measured end to end (`process_image` → `map_cam_to_original` → `show_cam_on_image` → fov dimming → `combined.save`) with a 128×96 input so the fov-dimming branch is active:

| Method | Differing pixels | Max integer diff |
| --- | --- | --- |
| gradcam | `0` of `73728` | `0` |
| gradcampp | `0` of `73728` | `0` |
| layercam | `0` of `73728` | `0` |
| eigencam | `0` of `73728` | `0` |
| ablationcam | `0` of `73728` | `0` |
| scorecam | `44` of `73728` | `2` LSB |

The `scorecam` result is reproducible (identical on two runs). Five of six methods are pixel-identical; the sixth is not. Display semantics are unchanged, but **overlay pixel identity is not a contract and must not be claimed.**

Wording rules for the docs:

- Do not write "restores the previous array exactly" or "figures are unaffected".
- Prefer: "`--save-npy normalized` preserves the previous normalized semantics and is numerically equivalent within the measured float32 tolerance; the overlay keeps the same display semantics."
- Do not put the `1.8e-7`–`2.4e-7` numbers or the `44`-pixel scorecam result in the READMEs. They are fixture measurements, not guarantees for every model and input; the README states the semantics and the non-guarantee only.

Users who need a renderable array from raw values add one line: `x = (x - x.min()) / max(x.max(), 1e-32)`.

### D5. Documented correctness limits

- Raw magnitudes are comparable only within one model, one target layer, and one preprocessing configuration. They are not comparable across backbones, layers, or CAM methods.
- `eigencam` raw values come from `get_2d_projection` (`pytorch_grad_cam/utils/svd_on_activations.py`), an SVD projection without `abs()`, and the mixin still applies `np.maximum(cam, 0)`. The projection's sign is arbitrary, so an overall sign flip changes the result qualitatively: one run can keep a large positive response while the next is almost entirely zeroed by the ReLU. Export remains available for completeness, but these values must not be interpreted as response magnitude or used for magnitude statistics.

Documented wording for the READMEs and the skill: raw magnitude statistics are intended for gradient-based CAM methods under fixed settings; `eigencam` raw values are exported for completeness but are not suitable for response-magnitude statistics.

### Not in scope

- Signed/negative CAM values (the mixin is the seam; a follow-up change adds the option).
- `fov_mask` or geometry metadata alongside the array.
- PNG/TIFF/CSV export, `--array-format`, or any change away from `.npy`.
- Adding a `--no-normalize` flag that would affect the overlay figure.
- Pinning `grad-cam`, or a runtime version warning.
- Two CAM code paths (official classes for `none`/`normalized`, raw classes for `raw`). See `D4`: the mask difference is float32-level and five of six methods render pixel-identical figures, so a second path would add a branch, a doubled test matrix, and its own figure differences for no measured benefit.

## Global Constraints

- The overlay figure always uses a `[0, 1]` copy; `pytorch_grad_cam.utils.image.show_cam_on_image` computes `cv2.applyColorMap(np.uint8(255 * mask), ...)`.
- Saved arrays stay `float32` at model input resolution. Do not resize or remap the saved array.
- ReLU semantics, resize-to-input-size, per-layer mean, and `ablationcam`/`eigencam`/`scorecam` behavior stay identical to upstream.
- All six `CAM_METHODS` entries use the raw classes. One code path, no per-method special cases.
- `save_npy` is a string in `{"none","raw","normalized"}` at every layer; `False`/`None` must not remain as internal states.
- `--save-npy` requires an explicit value from `{none,raw,normalized}`: no `nargs="?"` alias and no `const` export. Live docs and examples use an explicit value.
- No new dependencies, and no version pin on `grad-cam`.
- Core acceptance must not *rely on* environment-specific magnitudes, and no test gates on one. The only magnitude-related assertion left is positivity (`raw.max() > 0.0`) for the fixed fixture, documented as a non-authoritative diagnostic. The binding evidence is structural: `test_cam_methods_all_use_raw_classes` (per-method mapping), `test_raw_cam_does_not_call_scale_cam_image` (a raising `scale_cam_image` sentinel), and `test_process_image_saves_raw_array_by_default`, which asserts `allclose(saved, FAKE_CAM)` against a test-owned array whose maximum is `2.0`, with no model, gradient, or environment involvement. Drift is detected by comparing against the official class in the same process.
- Overlay pixel identity is not a contract. The end-to-end figure test uses a small LSB tolerance, not equality.
- `cam_summary.csv` columns do not change.
- Update live documentation and skills only. The only dated document that may change is the live classify-cam contract section of `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` (it was edited for cam in commit `7b6c459`).
- Set every live version reference to `0.6.1`.

## File Map

- Modify `entomokit/classify/cam.py` — `--save-npy` three-value parsing and help text.
- Modify `src/classification/cam.py` — mixin, raw CAM classes, two-array `process_image`, string save mode.
- Modify `tests/test_classification_cam.py` — mixin/drift tests, six-method coverage, save-mode tests, `--save-npy` parsing tests, `save_npy="none"` updates.
- Modify `tests/test_resume_flags.py` — `save_npy="none"`.
- Modify `tests/test_cli_schema.py` — `choices`/`default` assertions for `classify cam`.
- Modify `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` — cam option row, output tree, numeric semantics.
- Modify `README.md`, `README.cn.md` — cam outputs and array semantics.
- Modify `skills/entomokit-workflow/references/command-profiles.md` — CAM guidance.
- Modify `version.txt`, `entomokit/_version.py`, `entomokit/main.py`, `setup.py` — `0.6.1`.
- Modify `tests/test_package_version.py`, `tests/test_main_cli.py` — `0.6.1`.
- No change needed: `entomokit/completion.py` (parser-derived, verified) and `scripts/export_cli_schema.py` (generates on demand; no committed schema artifact).

## Task 1: Fix The Save-Mode Contract And Internal State

**Files:**
- Modify: `entomokit/classify/cam.py`
- Modify: `src/classification/cam.py`
- Modify: `tests/test_classification_cam.py`
- Modify: `tests/test_cli_schema.py`
- Modify: `tests/test_resume_flags.py`

**Interfaces:**
- Produces: `args.save_npy` is always `"none" | "raw" | "normalized"`, the value is required, and the internal parameters carry the same three strings. `prepare_output_dirs` creates `arrays/` only when the mode is not `"none"`; no internal parameter accepts a bool. `get_command_schema("classify cam")` reports `choices == ["none","raw","normalized"]` and `default == "none"` for `--save-npy`.

This task runs first because it fixes both the CLI contract and the internal state that `Task 3`'s save path branches on. Steps 6 and 7 belong to this task, not to `Task 3`: without the truthiness change, `--save-npy none` still writes arrays because the string `"none"` is truthy under the current `if save_npy:` check, and without the call-site update the six `save_npy=False` sites would be treated as "save".

- [x] **Step 1: Write the failing parser tests**

Add to `tests/test_classification_cam.py` (and add `import pytest` after `import pandas as pd`, line 9 — that module does not import it today):

```python
def _cam_args(extra: list[str]) -> argparse.Namespace:
    from entomokit.classify import cam as cam_cli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cam_cli.register(sub)
    return parser.parse_args(
        ["cam", "--images-dir", "images", "--out-dir", "out",
         "--model-dir", "model", *extra]
    )


def test_cam_save_npy_defaults_to_none() -> None:
    assert _cam_args([]).save_npy == "none"


def test_cam_save_npy_accepts_explicit_values() -> None:
    assert _cam_args(["--save-npy", "none"]).save_npy == "none"
    assert _cam_args(["--save-npy", "raw"]).save_npy == "raw"
    assert _cam_args(["--save-npy", "normalized"]).save_npy == "normalized"


def test_cam_save_npy_requires_a_value() -> None:
    with pytest.raises(SystemExit):
        _cam_args(["--save-npy"])


def test_cam_save_npy_rejects_unknown_value() -> None:
    with pytest.raises(SystemExit):
        _cam_args(["--save-npy", "bogus"])
```

Add to `tests/test_cli_schema.py`:

```python
def test_classify_cam_schema_exposes_save_npy_values() -> None:
    from entomokit.cli_schema import get_command_schema

    schema = get_command_schema("classify cam")
    assert schema is not None
    params = {item["name"]: item for item in schema["parameters"]}

    save_npy = params["--save-npy"]
    assert save_npy["choices"] == ["none", "raw", "normalized"]
    assert save_npy["default"] == "none"
    assert save_npy["value_hint"] == "none | raw | normalized"
```

- [x] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_classification_cam.py tests/test_cli_schema.py -k save_npy -v
```

One `-k` expression for both files. The new schema test name contains `save_npy`, so it is selected and no other `test_cli_schema.py` test is; do not write two `-k` options separated by a path, which is parsed but ambiguous to read.

Expected: FAIL. `action="store_true"` yields `True`/`False`, so `test_cam_save_npy_defaults_to_none` fails on `"none"`, `test_cam_save_npy_accepts_explicit_values` raises `SystemExit` because a value is not accepted, and `test_cam_save_npy_requires_a_value` reports no exception. The schema test fails because `choices` is `None`.

- [x] **Step 3: Replace the argument definition**

In `entomokit/classify/cam.py`, replace the current `--save-npy` block (lines 93-97) with the definition from `D2`. `run` already forwards `save_npy=args.save_npy` (line 168), so no other CLI change is needed.

- [x] **Step 4: Run the tests**

Run:

```bash
pytest tests/test_classification_cam.py -k save_npy -v
pytest tests/test_cli_schema.py -v
```

Expected: PASS. The four new parser tests and the schema test pass; the four pre-existing `test_cli_schema.py` tests are unaffected because no `cli_schema.py` source change is made (that file goes from four tests to five).

Three of the four parser tests fail before this step (`test_cam_save_npy_defaults_to_none`, `test_cam_save_npy_accepts_explicit_values`, `test_cam_save_npy_requires_a_value`); `test_cam_save_npy_rejects_unknown_value` already passes, because `argparse` also rejects the stray token under `action="store_true"`. It is kept as a contract lock on the error path, not as a TDD failure.

- [x] **Step 5: Verify the completion menu follows automatically**

Run: `python -c "from entomokit.completion import _parser_completion_tree; print(_parser_completion_tree()['classify cam']['choice_map']['--save-npy'])"`

Expected: `['none', 'raw', 'normalized']`, confirming no completion edit is required (`_parser_completion_tree` is defined at `entomokit/completion.py:32` and reads `action.choices` at `:26-29`).

- [x] **Step 6: Unify the internal save-mode state**

```python
SaveMode = Literal["none", "raw", "normalized"]
```
(import `Literal` from `typing` alongside the existing imports)

```python
def prepare_output_dirs(out_dir: Path, save_npy: SaveMode) -> Dict[str, Optional[Path]]:
    ...
    if save_npy != "none":
        array_dir = out_dir / "arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
```
```python
    save_npy: SaveMode,
```
in both `process_image` and `run_cam`.

`process_image` still saves the current min-max array for both non-`none` modes at this point, so `raw` and `normalized` behave identically until `Task 3` gives `raw` its own values. Nothing asserts raw semantics yet, so the suite stays green either way.

- [x] **Step 7: Update the existing call sites that pass a bool**

Replace `save_npy=False` with `save_npy="none"` at `tests/test_classification_cam.py:18,76,470,508,569` and `tests/test_resume_flags.py:403`. Without this, `False != "none"` is true and those tests would start writing arrays.

## Task 2: Add The Raw CAM Mixin And Pin Its Behaviour

**Files:**
- Modify: `src/classification/cam.py`
- Modify: `tests/test_classification_cam.py`

**Interfaces:**
- Consumes: `CAM_METHODS[name]`, the `BaseCAM` constructor contract already used by `prepare_cam`.
- Produces: `UnnormalizedCAMMixin`, plus `RawGradCAM`, `RawGradCAMPlusPlus`, `RawLayerCAM`, `RawScoreCAM`, `RawEigenCAM`, `RawAblationCAM`. `cam_extractor(input_tensor=..., targets=...)[0]` returns a `float32` `(size, size)` array with ReLU applied, unnormalized, at model input size.

- [x] **Step 1: Write the failing tests**

```python
CAM_METHOD_NAMES = (
    "ablationcam",
    "eigencam",
    "gradcam",
    "gradcampp",
    "layercam",
    "scorecam",
)


def _tiny_conv_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 3),
    )


def _fixed_cam_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 3, 96, 96) * 2 - 1


def test_cam_methods_all_use_raw_classes() -> None:
    from src.classification.cam import CAM_METHODS, UnnormalizedCAMMixin

    assert sorted(CAM_METHODS) == list(CAM_METHOD_NAMES)
    for name, cls in CAM_METHODS.items():
        assert issubclass(cls, UnnormalizedCAMMixin), name


@pytest.mark.parametrize("name", CAM_METHOD_NAMES)
def test_every_cam_method_returns_unnormalized_map(name: str) -> None:
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.classification.cam import CAM_METHODS

    torch.manual_seed(0)
    model = _tiny_conv_model().eval()
    cam = CAM_METHODS[name](model=model, target_layers=[model[4]])

    raw = cam(input_tensor=_fixed_cam_input(), targets=[ClassifierOutputTarget(1)])[0]

    # Structural invariants: no value here is an acceptance criterion.
    assert raw.shape == (96, 96)
    assert raw.dtype == np.float32
    assert np.isfinite(raw).all()
    assert raw.min() >= 0.0
    # Positivity holds for this fixture, not for CAM methods in general: a fully
    # ReLU'd-zero map is legal, and eigencam is additionally exposed to the SVD sign
    # (D5). This is the only magnitude-related assertion in the suite and it is a
    # fixture-scoped diagnostic, not acceptance evidence. The binding evidence is:
    # test_cam_methods_all_use_raw_classes (per-method mapping),
    # test_raw_cam_does_not_call_scale_cam_image (structural), and
    # test_process_image_saves_raw_array_by_default (save path, test-owned array).
    assert raw.max() > 0.0


def test_raw_cam_does_not_call_scale_cam_image(monkeypatch) -> None:
    """Structural proof: the raw path never reaches upstream's normalization.

    The official class must raise under the same patch, otherwise this test
    would pass vacuously.
    """
    import pytorch_grad_cam.base_cam as base_cam_module
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.classification.cam import CAM_METHODS

    def _boom(*_args, **_kwargs):
        raise AssertionError("scale_cam_image must not be called")

    monkeypatch.setattr(base_cam_module, "scale_cam_image", _boom)
    tensor = _fixed_cam_input()
    targets = [ClassifierOutputTarget(1)]

    torch.manual_seed(0)
    official_model = _tiny_conv_model().eval()
    with pytest.raises(AssertionError):
        GradCAM(model=official_model, target_layers=[official_model[4]])(
            input_tensor=tensor, targets=targets
        )

    torch.manual_seed(0)
    raw_model = _tiny_conv_model().eval()
    raw = CAM_METHODS["gradcam"](model=raw_model, target_layers=[raw_model[4]])(
        input_tensor=tensor, targets=targets
    )[0]

    assert raw.shape == (96, 96)


def test_raw_cam_agrees_with_official_cam_after_min_max() -> None:
    """Protect display equivalence; does not pin any CAM magnitude.

    Catches: changed resize, changed ReLU policy, changed aggregation, a new
    non-affine upstream normalization, and an upstream that stops normalizing
    (via the official.max() check). Does NOT catch a pure magnitude scaling of
    raw CAMs, which min-max would hide.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.classification.cam import CAM_METHODS

    torch.manual_seed(0)
    model = _tiny_conv_model().eval()
    tensor = _fixed_cam_input()
    targets = [ClassifierOutputTarget(1)]

    official = GradCAM(model=model, target_layers=[model[4]])(
        input_tensor=tensor, targets=targets
    )[0]
    raw = CAM_METHODS["gradcam"](model=model, target_layers=[model[4]])(
        input_tensor=tensor, targets=targets
    )[0]

    display = raw - raw.min()
    display = display / display.max()

    # min-max is affine-invariant and cv2.INTER_LINEAR is affine, so both agree to
    # float32 precision. A new non-affine step upstream, a changed resize, a changed
    # ReLU policy, or a different aggregation breaks this.
    assert official.max() == pytest.approx(1.0, abs=1e-6)
    assert np.abs(display - official).max() < 1e-6


@pytest.mark.parametrize("name", ["gradcam", "scorecam"])
def test_overlay_figure_matches_between_official_and_raw_cam(
    name: str,
    tmp_path: Path,
) -> None:
    """The rendered overlay must not change beyond uint8 quantization.

    Coverage is gradcam and scorecam. `scorecam` is included deliberately: it is
    the only method whose overlay measurably differs (44 of 73728 pixels by at
    most 2 LSB on this fixture), so a gradcam-only check could not detect the
    regression this test exists for. The other four methods keep their
    development-time measurements only.
    """
    from pytorch_grad_cam import GradCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.classification import cam as cam_mod
    from src.classification.cam import CAM_METHODS

    official_cls = {"gradcam": GradCAM, "scorecam": ScoreCAM}[name]
    tensor = _fixed_cam_input()
    targets = [ClassifierOutputTarget(1)]

    def render(cls, stem: str) -> np.ndarray:
        image_path = tmp_path / f"{stem}.png"
        Image.new("RGB", (128, 96), (200, 180, 160)).save(image_path)
        torch.manual_seed(0)
        model = _tiny_conv_model().eval()
        extractor = cls(model=model, target_layers=[model[4]])
        record = cam_mod.process_image(
            img_path=image_path,
            label="x",
            model=model,
            preprocess=lambda _image: tensor[0],
            cam_extractor=lambda **kwargs: extractor(**kwargs),
            device=torch.device("cpu"),
            fig_dir=tmp_path,
            array_dir=None,
            image_weight=0.5,
            fig_format="png",
            save_npy="none",
            model_size=96,
            resize_size=96,
        )
        return np.asarray(Image.open(record["figure_path"])).astype(np.int16)

    difference = np.abs(
        render(official_cls, f"{name}-official") - render(CAM_METHODS[name], f"{name}-raw")
    )

    assert difference.max() <= 4
    assert (difference > 0).mean() < 0.01


```

Reference only; **not acceptance criteria**. For the recorded fixture the six methods return maxima of gradcam `2.324e-04`, gradcampp `1.340e-02`, layercam `4.648e-04`, scorecam `5.918e-03`, eigencam `1.833e-01`, ablationcam `2.658e-02`. Those numbers are not asserted. `max > 0` holds for this fixture, not for CAM methods in general: a fully ReLU'd-zero map is legal, and `eigencam` is additionally exposed to the SVD sign (`D5`).

- [x] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_classification_cam.py -k "raw_cam or cam_method" -v`

Expected, measured against the unmodified code today (`CAM_METHODS["gradcam"] is GradCAM` is `True`):

- **FAIL** `test_cam_methods_all_use_raw_classes` — `UnnormalizedCAMMixin` does not exist.
- **FAIL** `test_raw_cam_does_not_call_scale_cam_image` — the official control half passes (GradCAM raises), then the raw half raises as well, because `CAM_METHODS["gradcam"]` is still the official class.
- **PASS** `test_every_cam_method_returns_unnormalized_map` — the current `CAM_METHODS` returns library-normalized output, but this test asserts only shape, dtype, finiteness, non-negativity, and positivity, so all six cases already pass. It is an invariant lock, not a TDD failure.
- **PASS** `test_raw_cam_agrees_with_official_cam_after_min_max` — before implementation both sides are the official class, so `official.max() ≈ 1.0` and `max|display - official| == 1.192e-07 < 1e-6`.
- **PASS** `test_overlay_figure_matches_between_official_and_raw_cam` — both sides run the official class, so the difference is exactly `0`.

Counting: at the **logical test-group level**, two groups fail and three pass. At the **pytest item level** this is **2 failed and 9 passed** (11 items), because `test_every_cam_method_returns_unnormalized_map` is parametrized over six methods and the overlay test over two.

Only the first two are TDD failures. The other three are guards that must keep passing once the mixin lands; do not "fix" them to fail first.

- [x] **Step 3: Implement the mixin**

Add above `CAM_METHODS` in `src/classification/cam.py`:

```python
class UnnormalizedCAMMixin:
    """Keep CAM magnitude by skipping BaseCAM's scale_cam_image calls.

    Mirrors pytorch-grad-cam 1.5.5 ``BaseCAM.compute_cam_per_layer`` and
    ``BaseCAM.aggregate_multi_layers``. ReLU, resize-to-input-size and the
    per-layer mean are identical to upstream; only the two min-max
    normalizations are removed.

    ``test_raw_cam_agrees_with_official_cam_after_min_max`` detects upstream
    drift: if grad-cam adds a non-affine step, changes its resize, or changes
    its ReLU or aggregation policy, that test fails.
    """

    @staticmethod
    def _resize_batch(cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        return np.stack(
            [
                cv2.resize(
                    np.float32(image), target_size, interpolation=cv2.INTER_LINEAR
                )
                for image in cam
            ]
        )

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets, eigen_smooth: bool
    ) -> list:
        if self.detach:
            activations_list = [
                activation.cpu().data.numpy()
                for activation in self.activations_and_grads.activations
            ]
            grads_list = [
                gradient.cpu().data.numpy()
                for gradient in self.activations_and_grads.gradients
            ]
        else:
            activations_list = list(self.activations_and_grads.activations)
            grads_list = list(self.activations_and_grads.gradients)
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for index, target_layer in enumerate(self.target_layers):
            activations = (
                activations_list[index] if index < len(activations_list) else None
            )
            gradients = grads_list[index] if index < len(grads_list) else None
            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                activations,
                gradients,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            cam_per_target_layer.append(self._resize_batch(cam, target_size)[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: list) -> np.ndarray:
        stacked = np.concatenate(cam_per_target_layer, axis=1)
        stacked = np.maximum(stacked, 0)
        return np.mean(stacked, axis=1)
```

- [x] **Step 4: Wire the raw classes into `CAM_METHODS`**

```python
class RawGradCAM(UnnormalizedCAMMixin, GradCAM):
    pass


class RawGradCAMPlusPlus(UnnormalizedCAMMixin, GradCAMPlusPlus):
    pass


class RawLayerCAM(UnnormalizedCAMMixin, LayerCAM):
    pass


class RawScoreCAM(UnnormalizedCAMMixin, ScoreCAM):
    pass


class RawEigenCAM(UnnormalizedCAMMixin, EigenCAM):
    pass


class RawAblationCAM(UnnormalizedCAMMixin, AblationCAM):
    pass


CAM_METHODS = {
    "gradcam": RawGradCAM,
    "gradcampp": RawGradCAMPlusPlus,
    "layercam": RawLayerCAM,
    "ablationcam": RawAblationCAM,
    "scorecam": RawScoreCAM,
    "eigencam": RawEigenCAM,
}
```

Keep the classes explicit rather than generated with `type()`.

- [x] **Step 5: Run the tests**

Run: `pytest tests/test_classification_cam.py -k "raw_cam or cam_method" -v`

Expected: PASS, including all six parametrized cases, the raising-`scale_cam_image` sentinel with its official-class control, the display-equivalence test, and the end-to-end figure check for gradcam and scorecam. Runtime varies by backend; `scorecam` and `ablationcam` are slowest because they run many internal forward passes, so treat no particular duration as an expectation.

- [x] **Step 6: Run the CAM suite for regressions**

Run: `pytest tests/test_classification_cam.py -v`

Expected: PASS, with no known failures. The `save_npy=False` call sites were already updated to `save_npy="none"` in `Task 1` Step 7, so this task leaves the suite green on its own and does not depend on `Task 3`. Any CAM failure here is a real regression from the mixin, not a pending cross-task edit.

## Task 3: Split The Raw And Display Arrays

**Files:**
- Modify: `src/classification/cam.py`
- Modify: `tests/test_classification_cam.py`

**Interfaces:**
- Consumes: `grayscale_cam` from whichever classes `CAM_METHODS` currently holds, plus the `"none" | "raw" | "normalized"` state from `Task 1`. The fake-extractor tests below do not need the mixin.
- Produces: `arrays/{stem}.npy` containing `grayscale_cam` when `save_npy == "raw"`, the min-max copy when `save_npy == "normalized"`, and no array when `save_npy == "none"`. The overlay always receives the min-max copy, never the raw array.

- [x] **Step 1: Separate the display copy from the raw CAM**

Replace the normalization block at `src/classification/cam.py:492-506`:

```python
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

    # Display copy: show_cam_on_image needs [0, 1]. The saved array keeps the
    # unnormalized CAM unless save_npy == "normalized".
    cam_display = grayscale_cam - grayscale_cam.min()
    if cam_display.max() > 0:
        cam_display = cam_display / cam_display.max()
    else:
        cam_display = np.zeros_like(cam_display)

    cam_on_full, fov_mask = map_cam_to_original(
        cam_display,
        original_img.size,
        model_size=model_size,
        resize_size=resize_size,
        eval_transform=eval_transform,
    )
```

Rename the `map_cam_to_original` parameter `cam_norm` to `cam_display` (line 351) and its internal uses. Existing call sites and tests pass it positionally.

- [x] **Step 2: Select the saved array**

```python
    cam_array_path = ""
    if save_npy != "none" and array_dir is not None:
        npy_path = array_dir / f"{stem}.npy"
        cam_array = cam_display if save_npy == "normalized" else grayscale_cam
        np.save(npy_path, cam_array.astype(np.float32))
        cam_array_path = str(npy_path)
```

- [x] **Step 3: Add the save-mode tests with concrete data**

```python
FAKE_CAM = np.array([[0.0, 2.0], [1.0, 0.5]], dtype=np.float32)


def _run_process_image(tmp_path: Path, monkeypatch, save_npy: str):
    from src.classification import cam

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)
    array_dir = tmp_path / "arrays"
    array_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(
        cam,
        "show_cam_on_image",
        lambda *_a, **_k: np.full((32, 32, 3), 240, dtype=np.uint8),
    )

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 1.0]], device=x.device).repeat(x.shape[0], 1)

    return cam.process_image(
        img_path=image_path,
        label="x",
        model=FakeModel(),
        preprocess=lambda _image: torch.zeros(3, 32, 32),
        cam_extractor=lambda **_k: [FAKE_CAM.copy()],
        device=torch.device("cpu"),
        fig_dir=tmp_path,
        array_dir=array_dir,
        image_weight=0.5,
        fig_format="png",
        save_npy=save_npy,
    )


def test_process_image_saves_raw_array_by_default(tmp_path: Path, monkeypatch) -> None:
    record = _run_process_image(tmp_path, monkeypatch, "raw")

    saved = np.load(tmp_path / "arrays" / "image.npy")
    assert saved.dtype == np.float32
    np.testing.assert_allclose(saved, FAKE_CAM)
    assert saved.max() != pytest.approx(1.0)
    assert record["cam_array_path"].endswith("image.npy")


def test_process_image_saves_normalized_array_on_request(
    tmp_path: Path, monkeypatch
) -> None:
    _run_process_image(tmp_path, monkeypatch, "normalized")

    saved = np.load(tmp_path / "arrays" / "image.npy")
    np.testing.assert_allclose(saved, np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32))


def test_process_image_writes_no_array_when_disabled(tmp_path: Path, monkeypatch) -> None:
    record = _run_process_image(tmp_path, monkeypatch, "none")

    assert record["cam_array_path"] == ""
    assert list((tmp_path / "arrays").iterdir()) == []


def test_overlay_receives_normalized_mask_not_raw(
    tmp_path: Path, monkeypatch
) -> None:
    """The overlay must always receive the normalized, mapped copy.

    Measured on this fixture: the captured mask is `(32, 32)` with min/max
    `0.0 / 1.0`, and it matches `map_cam_to_original(min-max(FAKE_CAM), ...)`
    exactly. `FAKE_CAM` peaks at `2.0`, so if the raw array leaked into the
    overlay the mapped mask would peak at `2.0` and both the bound and the
    reference comparison below would fail. Note the mask is at the original
    image size, not `FAKE_CAM`'s `2x2`, so it must be compared against a
    same-shaped reference.
    """
    from src.classification import cam

    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(image_path)
    array_dir = tmp_path / "arrays"
    array_dir.mkdir()

    captured: dict[str, np.ndarray] = {}

    def _capture(_rgb, mask, **_kwargs):
        captured["mask"] = np.asarray(mask).copy()
        return np.full((32, 32, 3), 240, dtype=np.uint8)

    monkeypatch.setattr(cam, "show_cam_on_image", _capture)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 1.0]], device=x.device).repeat(x.shape[0], 1)

    cam.process_image(
        img_path=image_path,
        label="x",
        model=FakeModel(),
        preprocess=lambda _image: torch.zeros(3, 32, 32),
        cam_extractor=lambda **_kwargs: [FAKE_CAM.copy()],
        device=torch.device("cpu"),
        fig_dir=tmp_path,
        array_dir=array_dir,
        image_weight=0.5,
        fig_format="png",
        save_npy="raw",
        model_size=32,
        resize_size=32,
    )

    display = FAKE_CAM - FAKE_CAM.min()
    display = display / display.max()
    expected, _fov = cam.map_cam_to_original(
        display, (32, 32), model_size=32, resize_size=32
    )

    mask = captured["mask"]
    assert mask.shape == (32, 32)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    np.testing.assert_allclose(mask, expected)
```

This drives `process_image` with fixed data, so it depends on no model, gradient, or display detail.

- [x] **Step 4: Run the tests**

Run: `pytest tests/test_classification_cam.py tests/test_resume_flags.py -v`

Expected: PASS, including the four new save-mode tests and the `save_npy="none"` call sites updated in `Task 1`.

For reference, if these tests are run before this task but after `Task 1`: only `test_process_image_saves_raw_array_by_default` fails, because the current code saves the min-max array for every non-`none` mode. `test_process_image_writes_no_array_when_disabled` already passes (`Task 1` made `"none"` disable saving), `test_process_image_saves_normalized_array_on_request` passes because it pins the current behavior, and `test_overlay_receives_normalized_mask_not_raw` passes because the current code always normalizes — it becomes a real guard only once `raw` starts carrying unnormalized values.

## Task 4: Update The Live Cam Contract And READMEs

**Files:**
- Modify: `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md`
- Modify: `README.md`
- Modify: `README.cn.md`

**Interfaces:**
- Produces: no live document claims that `.npy` output is unnormalized by accident, and no live document claims it is always normalized.

- [x] **Step 1: Update the cam option row**

At `2026-03-24-entomokit-refactor-design.md:353`, replace `| --save-npy | flag | False | 保存 CAM 数组为 .npy |` with a row describing the three values: `none`（默认，不保存）/`raw`（未归一化正值 CAM，保留幅值）/`normalized`（逐图 min-max，落在 `[0, 1]`）；值为必填，默认 `none`。

- [x] **Step 2: Update the output tree**

At the `arrays/` entry (around line 378), replace `模型输入空间的归一化 CAM 数组，float32` so it states: `arrays/` 仅在 `--save-npy raw` 或 `--save-npy normalized` 时生成（默认 `none` 不生成）；`raw` 保存未归一化正值 CAM，`normalized` 保存逐图 min-max CAM；两者都是模型输入空间的 `float32` 数组。Do not write "raw by default": the CLI default is `none`.

- [x] **Step 3: Document the numeric semantics**

Append to the 模型与预处理契约 bullet list:

- entomokit 跳过 pytorch-grad-cam 内部两次 `scale_cam_image()`，保留 ReLU 与 resize 到模型输入尺寸，因此保存的数组保留幅值；overlay 图始终使用独立的逐图 min-max 副本。
- `min-max` 具有仿射不变性，`cv2.INTER_LINEAR` 缩放是仿射的，因此 `--save-npy normalized` 与旧版本的模型空间 normalized CAM mask 在测量范围内保持 float32 精度内的一致性（当前 fixture 上六种方法最大偏差 `1.8e-7`~`2.4e-7`），但不是逐位相同，也不是跨模型/跨输入的普遍保证。
- overlay 图继续使用 normalized display mask，显示语义保持不变；但最终 PNG 已经过 `np.uint8` 量化，其像素差异不是兼容性契约。当前 fixture 的端到端测量显示五种方法像素一致，`scorecam` 有 44 个像素、最大 2 LSB 的差异；该结果仅作为测试基线，不作为普遍保证。
- `raw` 幅值仅在同一模型、同一目标层、同一预处理配置内可比，不适用于跨 backbone/跨层/跨 CAM 方法比较。
- `eigencam` 的数值来自 SVD 投影，符号任意且之后仍经过 ReLU；符号翻转时可能整幅图被清零，其 `raw` 幅值不可用于响应强度统计。

- [x] **Step 4: Update the English README**

At `README.md:919`, replace `- `arrays/` — Raw CAM arrays (with `--save-npy`)` with wording matching the new contract:

```markdown
- `arrays/` — CAM arrays (`--save-npy raw` keeps the unnormalized CAM; `--save-npy normalized` writes per-image min-max values; no arrays are written by default)
```

Also update the command example at `README.md:898` from the bare `--save-npy` to `--save-npy raw`, since the example wants arrays.

```markdown
`--save-npy` takes `none` (default), `raw`, or `normalized`. `raw` keeps the CAM magnitude; `normalized` writes the per-image min-max `[0, 1]` mask, which is the same normalized copy the overlay uses. Raw magnitude statistics are intended for gradient-based CAM methods under fixed settings: they are only comparable within one model, target layer, and preprocessing configuration, and `eigencam` raw values are exported for completeness but are not suitable for response-magnitude statistics. `--save-npy normalized` preserves the previous normalized semantics and is numerically equivalent within the tolerance measured during development; the overlay keeps the same display semantics, and pixel-identical output is not guaranteed.
```

- [x] **Step 5: Update the Chinese README**

Mirror all three edits: replace `- `arrays/` — 原始 CAM 数组（使用 `--save-npy`）` at `README.cn.md:930`, change the example at `README.cn.md:909` to `--save-npy raw`, and add the equivalent Chinese paragraph.

- [x] **Step 6: Verify no stale wording remains**

Run: `rg -n "Raw CAM arrays|原始 CAM 数组|Save raw CAM heatmaps|归一化 CAM 数组" README.md README.cn.md entomokit docs/superpowers/specs skills`

Expected: no matches in live documents.

## Task 5: Update Skill Guidance

**Files:**
- Modify: `skills/entomokit-workflow/references/command-profiles.md`

**Interfaces:**
- Produces: the skill's CAM profile explains the three save modes and their comparability limits.

- [x] **Step 1: Extend the CAM profile**

At `command-profiles.md:78`, append to the existing CAM paragraph: `--save-npy` takes a required value: `none` (default), `raw` (unnormalized positive CAM, magnitude preserved), or `normalized` (per-image min-max `[0, 1]`). Note that raw magnitudes are comparable only within one model, target layer, and preprocessing setting, and that `eigencam` raw values must not be used for response-magnitude statistics.

- [x] **Step 2: Check for other CAM mentions**

Run: `rg -n "save-npy|CAM" skills/entomokit-workflow/SKILL.md skills/entomokit-workflow/references/workflow.md`

Expected: any additional mention is consistent with the new semantics; update it if it describes `.npy` contents.

## Task 6: Release Version 0.6.1

**Files:**
- Modify: `version.txt`, `entomokit/_version.py`, `entomokit/main.py`, `setup.py`
- Modify: `tests/test_package_version.py`, `tests/test_main_cli.py`

**Interfaces:**
- Produces: `entomokit --version` prints `entomokit 0.6.1`, and `entomokit._version.__version__` equals the `setup.py` version.

This task is deliberately last and is not a mechanical string replacement: the release carries a user-visible change to what `--save-npy` writes.

- [x] **Step 1: Bump the live version references**

Set `0.6.1` in `version.txt:1`, `entomokit/_version.py:3`, `entomokit/main.py:87` (the `PackageNotFoundError` fallback), and `setup.py:5`.

- [x] **Step 2: Update version assertions**

Update `tests/test_package_version.py` lines 7 and 11, and rename the test function on line 6 from `test_setup_version_is_0_6_0` to `test_setup_version_is_0_6_1` — the version is embedded in the identifier as well as in the docstring and assertion, so renaming only lines 7 and 11 would leave a test named for the old version. Then update `tests/test_main_cli.py:261,275,280`.

- [x] **Step 3: Confirm what the release does and does not need**

- No `CHANGELOG` file exists in this repository; the release note is the commit message and must follow Conventional Commits. This change alters what a released flag writes *and* makes its value required, so mark it breaking, and do **not** say "by default": the CLI default is `none`, and only `raw` writes unnormalized values. Use `feat(cam)!: add --save-npy {none,raw,normalized} with unnormalized raw arrays`. The commit itself is `Task 7` Step 5 and needs the owner's approval.
- `entomokit/completion.py` is parser-derived, so the new `{none,raw,normalized}` menu needs no edit.
- No committed CLI schema artifact exists; `skills/entomokit-workflow/scripts/export_cli_schema.py` regenerates on demand.

- [x] **Step 4: Verify no stale live version remains**

Run: `rg -n '0\.6\.0' entomokit src tests setup.py version.txt README.md README.cn.md skills`

Expected: matches only where `0.6.0` is a remote-version fixture rather than the local version (`tests/test_update.py:25,26` and the `v0.5.0` tag payload). Those exercise parsing and never compare against the local version; leave them unchanged. Confirm with `rg -n "Update available" tests` returning no match.

- [x] **Step 5: Run the version tests**

Run: `pytest tests/test_package_version.py tests/test_main_cli.py tests/test_update.py -v`

Expected: PASS.

## Task 7: Final Verification

**Files:**
- No modifications.

**Interfaces:**
- Consumes: the completed tasks above.
- Produces: evidence that the feature, help text, schema, completion, and version are consistent.

- [x] **Step 1: Run the full suite**

Run: `pytest -v`

Expected: PASS with no collection errors.

- [x] **Step 2: Check the CLI surface**

Run: `python -m entomokit.main classify cam --help`

Expected: the usage line shows `--save-npy {none,raw,normalized}` (no inner brackets, because the value is required) with the new help text, and no other option changed.

- [x] **Step 3: Verify the end-to-end array semantics**

Re-use the arguments from the run recorded in `runs/cam/log.txt:1`, but write them with the new syntax and a fresh `--out-dir` each time. Do **not** replay that log line verbatim: it contains the removed bare `--save-npy`.

```bash
entomokit classify cam --images-dir data/Epidorcus/images \
  --model-dir runs/train/AutogluonModels/convnextv2_femto \
  --cam-method gradcam --save-npy raw --out-dir runs/cam_raw --overwrite

entomokit classify cam --images-dir data/Epidorcus/images \
  --model-dir runs/train/AutogluonModels/convnextv2_femto \
  --cam-method gradcam --save-npy normalized --out-dir runs/cam_norm --overwrite

entomokit classify cam --images-dir data/Epidorcus/images \
  --model-dir runs/train/AutogluonModels/convnextv2_femto \
  --cam-method gradcam --out-dir runs/cam_none --overwrite
```

`--overwrite` is required on every rerun: `check_output_dir` (`src/common/cli.py:218-232`) calls `sys.exit(1)` when `--out-dir` exists and is not empty, and `classify cam` passes `has_resume=False`, so `--resume` is not an option here. Then:

```bash
python -c "
import numpy as np, glob
for out in ['runs/cam_raw', 'runs/cam_norm']:
    f = sorted(glob.glob(out + '/arrays/*.npy'))[0]
    a = np.load(f)
    print(out, a.dtype, a.shape, float(a.min()), float(a.max()))
import os
print('no-flag arrays dir exists:', os.path.isdir('runs/cam_none/arrays'))
"
```

Expected: the `raw` run prints a `float32` array whose max is not pinned to `1.0`; the `normalized` run prints min `0.0` and max `1.0`; the no-flag run creates no `arrays/` directory.

- [x] **Step 4: Inspect changes before the release commit**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only the files listed in this plan plus the plan document, no whitespace errors.

- [ ] **Step 5: Commit the release**

**This step requires explicit approval from the repository owner. Do not run it otherwise** — recording the command here is not authorization to commit.

```bash
git add -A
git commit -m "feat(cam)!: add --save-npy {none,raw,normalized} with unnormalized raw arrays"
git log --oneline -1
git status --short
```

Expected: a single commit containing the code, tests, live docs, skill text, and version bump, followed by a clean working tree. The repository convention is one release commit rather than a commit per task, matching `docs/superpowers/plans/2026-07-10-segment-cpu-parallelism.md`. The message must not say "by default": the CLI default is `none`.

## Risk Register

| Risk | Mitigation |
| --- | --- |
| Upstream grad-cam changes its internals and the mixin silently diverges, while `setup.py` does not pin the version | `test_raw_cam_agrees_with_official_cam_after_min_max` compares the mixin against the official class in the same process, so it is stable across PyTorch/BLAS/OpenCV versions but fails on any non-affine upstream change; the mixin docstring records the mirrored version and signatures |
| A magnitude assertion is mistaken for proof that normalization was bypassed | No test gates on a fixture magnitude. The proof is structural — `test_cam_methods_all_use_raw_classes` (per-method mapping) and the raising `scale_cam_image` sentinel — and the save path is pinned against a test-owned `FAKE_CAM` array; the sole remaining magnitude check is a documented positivity diagnostic |
| `--save-npy normalized` or the overlay figures are claimed to be byte-identical to the old release | Wording fixed in `D4` and `Task 4`: the mask matches to `1.8e-7`–`2.4e-7` on the fixture, and the overlay was measured end to end — 5 of 6 methods pixel-identical, `scorecam` differing in `44` of `73728` pixels by `2` LSB. Pixel identity is explicitly not a contract, and `test_overlay_figure_matches_between_official_and_raw_cam` asserts a small LSB tolerance instead |
| Users treat raw magnitudes as cross-image statistics, especially for `eigencam` | Documented in `D5`, `Task 4`, and `Task 5`, including the sign-flip/ReLU interaction |
| Existing scripts that use the bare `--save-npy` fail after upgrading | The failure is loud and immediate (`expected one argument`); the same release already changes what the flag writes; `Task 4` updates every live example, and `Task 6` marks the commit breaking |
| Only one CAM method is covered while all six are claimed | `test_cam_methods_all_use_raw_classes` locks the mapping and `test_every_cam_method_returns_unnormalized_map` runs all six |

## Post-Plan Addition (owner-requested, after Task 7 verification)

The owner reported that `classify cam` showed almost nothing on screen. Root cause: the command never configured logging, so every `logging.info` in `src/classification/cam.py` was silently dropped by the unconfigured root logger (default level `WARNING`); only third-party `print()` output survived. Two changes, both in files already listed in the File Map, so no new file is introduced:

- `entomokit/classify/cam.py` — `run()` now calls `setup_logging(out_dir)` before `save_log(out_dir, args)`, matching `entomokit/segment.py` and `entomokit/synthesize.py`. The order matters: `setup_logging` first binds the console handler to the real `sys.stdout`, so each log line reaches the screen once and `log.txt` once, with no Tee duplication.
- `src/classification/cam.py` — `run_cam` now reports where the outputs went: `CAM heatmaps written to: <figures dir>`, `Summary: <out_dir>/cam_summary.csv`, and, only when `--save-npy` is not `none`, `CAM arrays written to: <arrays dir> (<mode>)`.

Covered by `test_run_cam_logs_where_outputs_were_written`, parametrized over `raw` and `none` so both branches of the conditional arrays line are asserted.
