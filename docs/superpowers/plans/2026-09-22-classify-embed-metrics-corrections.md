# Classify Embed Metrics Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `entomokit classify embed` embedding-quality metrics semantically correct and reproducible by porting the OTU-Former v0.7.1 evaluator corrections — cross-validated kNN and linear probing with an explicit shuffled `StratifiedKFold`, a new balanced-accuracy linear-probe field, `None`/empty-CSV semantics for metrics that cannot be computed, explicit retrieval and clustering bounds, true-label cosine silhouette, and label-CSV input integrity — then release version `0.6.2`.

**Architecture:** Keep `src/classification/embedder.py:compute_embedding_metrics()` as the single implementation and keep its public signature and returned keys (one new key added). Add two module-level private helpers — `_METRIC_KEYS` (the ordered key contract, which also fixes the `metrics.csv` column order) and `_make_stratified_cv()` (the single source of CV configuration for kNN and linear probing). `entomokit/classify/embed.py` stays a thin adapter: it validates the label CSV before doing any work, converts `None` to an empty CSV cell, and prints `N/A` instead of formatting `None` as a float. No new dependency, no new CLI option.

**Tech Stack:** Python 3.9+, argparse, NumPy, pandas, scikit-learn (already in the `classify` extra), pytest.

**Source of the corrections:** `OTU-Former/docs/superpowers/plans/2026-09-21-otuformer-v071-cli-metrics-corrections.md` (v0.7.1) and `OTU-Former/src/otuformer/embedding/evaluator.py`.

## Approved Scope Decisions

Recorded before implementation started:

- No separate `-design.md` spec. The design rationale lives in this plan's `Design Decisions` section, matching `docs/superpowers/plans/2026-09-15-classify-cam-unnormalized-arrays.md`.
- Port only what maps onto entomokit. The applicability matrix below is the authority on scope; items marked **Not applicable** must not be implemented.
- `compute_embedding_metrics()` keeps its name, its positional signature `(embeddings, labels, sample_size=10000)`, and all 12 existing returned keys. Exactly one key is added: `Linear_Probing_Balanced_Acc`. See `D1`.
- Unavailable metrics are `None` at the function boundary, an empty cell in `metrics.csv`, and `N/A` on stdout. See `D3`.
- No new dependency. `scikit-learn` is already declared in the `classify` extra of `setup.py`.
- CLI source, comments, docstrings, plan text, and tests are English. `README.cn.md` stays Chinese.

## Applicability Matrix (OTU-Former v0.7.1 → entomokit)

| v0.7.1 task | entomokit state before this plan | Decision |
| --- | --- | --- |
| T2/T3 cross-validated kNN with explicit `StratifiedKFold` | `KNeighborsClassifier.fit(X_norm, y).score(X_norm, y)` — resubstitution | **Port**. Measured `kNN_Acc_k1 == 1.0000` on uninformative embeddings, i.e. a constant |
| T2 cross-validated linear probing + `Linear_Probing_Balanced_Acc` | in-sample `LogisticRegression.fit(...).score(...)` | **Port**. Also `1.0` on separable/overfit data; balanced accuracy is the only field that exposes minority-class collapse |
| T2/T3 unsupported metrics become unavailable instead of `0.0`/crash | no `0.0` fallback, but hard crashes: singleton class raises in `LogisticRegression`/`silhouette_score`; `n < k + 1` raises in `NearestNeighbors` | **Port the principle**, different symptom: add bounds guards and return `None` |
| T3 mAP must not rank the query itself | `indices[:, 1:]` drops the first neighbour *positionally*, which assumes the query is always its own first neighbour. Under exact-duplicate embeddings it is not (measured: 2000/2000 duplicate trials), so the query survives into the candidate list and can satisfy its own relevance test | **Port**: exclude the query by index, keep the AP formula, and return `None` instead of `0.0` when nothing is rankable |
| T3 explicit Recall@K bounds and index-based self-exclusion | `n_neighbors=k+1` raises when `k >= n`; self is dropped positionally (`[:, 1:]`), which is ambiguous under exact-duplicate embeddings | **Port**: per-k bound returning `None`, plus explicit per-row index exclusion |
| T3 reject degenerate clustering instead of fabricating clusters | no `np.zeros` fallback, but scikit-learn 1.7 silently returns fewer clusters plus a `ConvergenceWarning` when distinct rows `< n_clusters` | **Port**: distinct-rows guard returning `None` for NMI/ARI/Purity |
| T3 silhouette keeps true-label semantics, cosine distance, `None` when undefined | true labels used, but default `metric="euclidean"`; undefined cases raise | **Port**: `metric="cosine"`, guard, `None` |
| T4/T5 trainer CSV schema migration, atomic resume append, metric plotting | `classify embed` writes a single-row wide `metrics.csv` with `resume=False`; there is no metrics logger, no append path, no training plot | **Not applicable** — do not implement |
| T4/T5 sorted subsample indices so trainer and extract agree | embed is the only consumer, and the correction existed to make *two consumers* agree on sample identity | **Downgrade to one line** (`np.sort`) for a canonical sampled-row order only. It does **not** make metrics invariant to input row order: `rng.choice` selects positions, so permuting the input selects different samples. See `D9` |
| T5 extract rejects duplicate `image` keys before label alignment | `merge(..., how="inner")` silently duplicates embedding rows, and an empty overlap silently yields zero metric rows | **Port** as local, embed-specific input-integrity checks |
| T6 Typer/Click `ParameterSource` compatibility | entomokit uses argparse; no `click`/`typer` anywhere | **Not applicable** |
| AMI field, removal of `compute_linear_probing()`, leave-one-out kNN | — | **Not applicable / YAGNI** |

## Measurements Taken While Writing This Plan

Produced against the current working tree before any edit, on a synthetic fixture (`RandomState`, L2-normalized rows), for reference only — **not** test assertions. Fixture A: 20 classes, 3000 samples, 128 dims, random labels (no class structure). Fixture B: 3 classes, sizes 30/10/5, 16 dims.

| Metric | Current implementation | Proposed CV implementation |
| --- | --- | --- |
| `kNN_Acc_k1` (A) | `1.0000` | `0.0463` |
| `kNN_Acc_k5` (A) | `0.2830` | `0.0557` |
| `kNN_Acc_k20` (A) | `0.1760` | `0.0513` |
| `Linear_Probing_Acc` (A) | `0.2123` | `0.0547` |
| `Linear_Probing_Balanced_Acc` (A) | — | `0.0547` |
| `Linear_Probing_Acc` (B) | `0.8889` | `0.8889` |
| `Linear_Probing_Balanced_Acc` (B) | — | `0.6667` |

Runtime on 5000×768, 20 classes, 5 folds (`StratifiedKFold(shuffle=True, random_state=42)`):

| Operation | Wall time |
| --- | --- |
| Current in-sample `LogisticRegression(liblinear, max_iter=500)`: fit + score | `0.97 s` |
| Proposed 5-fold `LogisticRegression(lbfgs, max_iter=1000)` (`accuracy` + `balanced_accuracy`) | `0.07 s` |
| 5-fold `LogisticRegression(liblinear, max_iter=1000)` (rejected alternative) | `3.87 s` |
| 100 classes, 5000×768, 5-fold `lbfgs` | `0.22 s`, `0` `ConvergenceWarning` |
| Current in-sample kNN for `k=1,5,20` (fit + score each) | `0.14 s` |
| Proposed 5-fold kNN for `k=1,5,20` | `0.26 s` |

Reproduced failure modes of the current implementation:

| Fixture | Current behavior |
| --- | --- |
| single-class labels | `ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class` |
| `n=10`, 2 classes | `ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 11, n_samples_fit = 10` |
| 2 distinct normalized rows, 3 requested clusters | `KMeans` returns `[0 0 1]` plus `ConvergenceWarning` — a silent degenerate clustering |
| all-singleton labels (`a,b,c,d,e,f`) | `mAP@R = 0.0` (should be unavailable; no query has a non-self relevant item) |
| hand-written ranking `[[1, 0, 2], [0, 1, 2], [2, 0, 1]]` with `y = [a, b, a]` (query 0's own index at rank 1, as exact duplicates produce) | `Recall@1 = 1.0` and `mAP@R = 1.0` by counting each query's own index as a relevant hit; excluding the query by index gives `Recall@1 = 1/3` and `mAP@R = 0.5` |
| 2000 random fixtures with exact-duplicate rows (n = 4-30, `NearestNeighbors(metric="euclidean")`, scikit-learn 1.7.2) | the query is its own **first** neighbour in `0 / 2000` cases — `idx[:, 0] == arange(n)` never held, so `[:, 1:]` removes a non-query sample and keeps the query |

## Design Decisions

**D1 — The field set is name-stable, not append-only.** The returned keys are fixed by a module-level `_METRIC_KEYS` tuple. No existing key is renamed or removed; `Linear_Probing_Balanced_Acc` is inserted immediately after `Linear_Probing_Acc` for readability, which shifts every later column one position to the right. The guaranteed compatibility is **by header name** (all existing `runs/**/metrics.csv` readers in this repository use `pd.read_csv` and select by name — there is no positional consumer), not by column position. Downstream analysis must know the values changed (see `D7`).

**D2 — `StratifiedKFold(shuffle=True, random_state=42)` with the fold count bounded by the smallest class.** Bounding by `min(5, min_class_count)` rather than by `n_classes` is the substantive v0.7.1 fix: a two-class dataset with many samples per class still gets five folds. A singleton class makes stratified CV impossible, so kNN and linear probing return unavailable values rather than a forced two-fold split. kNN reuses the same splitter object for all three `k` values.

**D3 — Unavailable is `None`, an empty CSV cell, and `N/A`.** `None` is the single internal representation. pandas serializes `None` as an empty field (read back as `NaN`), so no `NaN` literal and no `"unavailable"` enum is introduced. The CLI print loop is the only place that must stop assuming a float. Rationale: this is the existing `InstantMetricsLogger`-style empty-field convention in the sibling project and it is the shortest correct representation.

**D4 — `LogisticRegression(max_iter=1000, random_state=42)` (default `lbfgs`), not `liblinear`.** Multinomial lbfgs measures 14× faster than the current single `liblinear` fit while doing five folds (see measurements), because `liblinear` trains one binary problem per class. `liblinear` is dropped rather than kept behind a flag: there is no second strategy to select at runtime. The existing `warnings.catch_warnings()` block also ignores `sklearn.exceptions.ConvergenceWarning` so a non-converged probe degrades to a warning-free run rather than stderr noise.

**D5 — Recall@K and mAP keep different singleton-query denominators on purpose.** Recall divides by every query (a singleton query can never hit, so it counts as a miss); mAP excludes queries with no non-self relevant item. This matches the current behavior and v0.7.1's documented `Silhouette_Score`/Recall/mAP semantics. The difference is documented in the docstring rather than unified.

**D6 — KMeans input degeneracy returns unavailable clustering metrics.** The guard is `n_classes >= 2` and `len(unique_rows(X_norm)) >= n_classes`. No `np.zeros` fallback and no per-item relabeling is added; the metrics are simply unavailable. Rationale: `KMeans` already refuses to do anything meaningful here and only emits a warning.

**D7 — Metrics are not numerically comparable with pre-0.6.2 runs.** CV fold selection, Recall/mAP self-exclusion, the silhouette distance metric, and unavailable-value handling all change the numbers. Both READMEs get an explicit warning; the field names stay stable. Version stays at the requested `0.6.2` (patch), and the commit is marked breaking with `!` because the numbers change meaning.

**D8 — Label-CSV integrity is validated locally in `embed`, not in `load_image_csv`, and before any work.** `load_image_csv` is also used by `classify predict`, where duplicate rows are harmless (duplicate predictions), so the checks stay at the one call site where duplicates silently desynchronize embeddings and labels. Both checks — duplicate `image` values and empty overlap with the image files — run *before* `check_output_dir()` and before extraction, by intersecting `label_df["image"]` with the image file names present in `--images-dir`. Only an *empty* overlap is an error: partially labelled image directories are the normal case, and unlabelled images are intentionally excluded by the inner merge. The post-merge `merged.empty` check is therefore not needed and must not be added.

**D9 — The subsample is position-based and stays that way.** `np.random.RandomState(42).choice(n, sample_size, replace=False)` selects row positions, so the sampled *set* depends on input row order; `np.sort` only canonicalises the order of the selected rows. Making metrics truly row-order invariant would require stable sample IDs threaded through the evaluator, which no consumer needs here (see Deferred).

**D10 — The ranking rule lives in two pure helpers that take a precomputed neighbour matrix.** `_recall_at_k(neighbors, y, k)` and `_mean_ap_at_r(neighbors, y)` implement the index exclusion; `compute_embedding_metrics()` builds the matrices. Rationale: on real embeddings the exclusion bug is only observable through exact-duplicate ties, and scikit-learn's tie order is a backend implementation detail — a fixture asserting a hand-computed value would either be fragile or fail to discriminate. Injecting the ranking makes the rule exactly and deterministically testable. Neighbour counts stay exactly as they are today (recall: `k + 1` per k; mAP: `len(y)`, because a truncated mAP window is not tie-equivalent to the full ranking), and the tie order among equidistant samples remains scikit-learn's. Do not add more helpers than these three (`_make_stratified_cv`, `_recall_at_k`, `_mean_ap_at_r`) plus `_METRIC_KEYS`.

## File Map

| File | Change |
| --- | --- |
| `src/classification/embedder.py` | module-level `IMAGE_EXTS`, `_METRIC_KEYS`, `_make_stratified_cv()`, `_recall_at_k()`, `_mean_ap_at_r()`, rewrite of `compute_embedding_metrics()` (lines ~105-225) |
| `entomokit/classify/embed.py` | early label-CSV validation before `check_output_dir()` (lines ~118-130, ~176-195), `N/A` printing |
| `tests/test_classification_embed_metrics.py` | new: evaluator unit tests |
| `tests/test_classify_embed_cli.py` | adapter tests: duplicate/empty label CSV, `N/A` output |
| `README.md` | metric contract section + warning, English |
| `README.cn.md` | same content, Chinese |
| `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` | one-line evaluator-vintage note on line 324 only |
| `version.txt`, `entomokit/_version.py`, `entomokit/main.py`, `setup.py` | `0.6.2` |
| `tests/test_package_version.py`, `tests/test_main_cli.py` | `0.6.2` assertions |

Do not modify: `docs/superpowers/plans/2026-03-24-phase3-classify.md` or any other historical plan/spec body, `src/classification/evaluator.py` (classifier metrics are a different code path), `tests/test_resume_flags.py:371` (its embed case exits before metrics).

---

### Task 1: Add failing evaluator and adapter regression tests

**Files:**
- Create: `tests/test_classification_embed_metrics.py`
- Modify: `tests/test_classify_embed_cli.py`
- Test target: both files

**Interfaces:**
- Consumes: the current `compute_embedding_metrics()` and `embed.run()`.
- Produces: failing tests that define the 0.6.2 behavior before implementation.

- [ ] **Step 1: Create the evaluator test module**

Create `tests/test_classification_embed_metrics.py` with a shared fixture builder:

```python
"""Tests for classify embed embedding-quality metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest


def _blobs(n_classes: int = 3, per_class: int = 12, dim: int = 16, seed: int = 0):
    """Synthetic embeddings with well-separated class means."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, dim) * 3.0
    x = np.vstack([centers[c] + rng.randn(per_class, dim) for c in range(n_classes)])
    y = np.repeat([f"class_{c}" for c in range(n_classes)], per_class)
    return x.astype(np.float64), y
```

Add these tests, importing `from src.classification.embedder import compute_embedding_metrics, _recall_at_k, _mean_ap_at_r`:

- `test_knn_and_linear_probing_use_shuffled_five_fold_stratified_cv`: monkeypatch `sklearn.model_selection.cross_val_score` **and** `sklearn.model_selection.cross_validate` to capture the `cv` object and return finite scores. Assert for both that `cv.n_splits == 5`, `cv.shuffle is True`, `cv.random_state == 42`. Use `per_class=10` so a classes-based fold cap could not accidentally produce `5`.
- `test_uninformative_embeddings_do_not_give_perfect_knn_k1`: 20 classes × 15 samples of pure noise, `dim=8`. Assert `metrics["kNN_Acc_k1"] < 0.5`. (Before the fix this is exactly `1.0`.)
- `test_linear_probing_field_mapping`: monkeypatch `sklearn.model_selection.cross_validate` to return `{"test_accuracy": [0.8, 0.8], "test_balanced_accuracy": [0.3, 0.3]}` and assert `Linear_Probing_Acc == pytest.approx(0.8)` and `Linear_Probing_Balanced_Acc == pytest.approx(0.3)`. This tests the field mapping deterministically; do **not** assert an ordering like `balanced < accuracy` on real data, because a separable fixture makes both `1.0`.
- `test_linear_probing_reports_both_finite_scores`: integration assertion on `_blobs(3, 12)` only — both `Linear_Probing_Acc` and `Linear_Probing_Balanced_Acc` are finite floats.
- `test_smallest_class_bounds_fold_count`: 3 classes of 3 samples each. Patch `cross_val_score`/`cross_validate` and assert `cv.n_splits == 3`.
- `test_unsupported_k_is_none_and_supported_k_stays_numeric`: 2 classes × 3 samples (`n=6`). Assert `metrics["kNN_Acc_k20"] is None` and `metrics["kNN_Acc_k1"]` is a float.
- `test_single_class_returns_unavailable_instead_of_raising`: one label, 8 samples; `with warnings.catch_warnings(): warnings.simplefilter("error")` — assert no exception and that `NMI`, `ARI`, `Purity`, `Silhouette_Score`, `kNN_Acc_k1`, `Linear_Probing_Acc`, `Linear_Probing_Balanced_Acc` are all `None`.
- `test_recall_helper_bounds_are_per_k`: `n=10`; `_recall_at_k(neighbors, y, 10) is None` while `_recall_at_k(neighbors, y, 1)` returns a float. Also assert `_recall_at_k(neighbors, y, 0) is None`.
- `test_recall_helper_excludes_the_query_by_index`: hand-written ranking `np.array([[1, 0, 2], [0, 1, 2], [2, 0, 1]])` with `y = np.array(["a", "b", "a"])` — query 0's own index sits at rank 1, so a positional `[:, 1:]` slice would count it as a hit. Assert `_recall_at_k(ranking, y, 1) == pytest.approx(1 / 3)`: query 0 is miss (nearest non-self is the class-`b` sample), query 1 is miss, query 2 is a hit. A positional implementation returns `1.0` on this fixture.
- `test_map_helper_excludes_the_query_by_index`: same ranking and labels. Assert `_mean_ap_at_r(ranking, y) == pytest.approx(0.5)`: query 0 has `r = 1` and its nearest non-self is the class-`b` sample, giving `AP = 0.0`; query 2 gives `AP = 1.0`; the class-`b` query is skipped. A positional implementation returns `1.0` on this fixture.
- `test_map_helper_is_none_when_no_query_has_a_relevant_pair`: ranking over six distinct rows with six singleton labels; assert `_mean_ap_at_r(ranking, y) is None`.
- `test_map_matches_an_independent_brute_force_reference`: on `_blobs(3, 12, dim=8, seed=5)` (continuous data, no exact distance ties) compare `metrics["mAP@R"]` against a local `_brute_force_map(x, y)` helper that ranks by an L2-normalized `x @ x.T` similarity matrix, drops `j == i`, takes the first `r` non-self candidates, and applies the same `AP@R` formula. Assert `pytest.approx` equality. This validates the wiring and the AP@R formula against an implementation that does not use `NearestNeighbors`.
- `test_recall_bounds_are_per_k_on_real_data`: 2 classes × 5 samples (`n=10`); assert `metrics["Recall@10"] is None`, while `Recall@1` and `Recall@5` are floats.
- `test_map_is_none_when_no_query_has_a_relevant_pair`: labels `["a", "b", "c", "d", "e", "f"]`, distinct rows. Assert `metrics["mAP@R"] is None`.
- `test_clustering_is_none_when_distinct_rows_fewer_than_classes`: 3 classes, embeddings with only 2 distinct rows. Assert `NMI`, `ARI`, `Purity` are all `None`.
- `test_silhouette_uses_cosine_and_is_undefined_for_one_class`: assert `Silhouette_Score is None` for a single class, and that a two-class fixture returns a finite value equal to a direct `sklearn.metrics.silhouette_score(x_norm, y_codes, metric="cosine")` call on the same 2000-sample draw (import `_METRIC_KEYS` only if needed).
- `test_metrics_are_deterministic_for_identical_input`: call `compute_embedding_metrics(x, y, sample_size=120)` on `_blobs(6, 40, seed=3)` twice and assert both dicts are equal, `None` values included. This is a determinism guard (seeded subsample, seeded `KMeans`, no unseeded randomness); it must **not** be written as an input-row-permutation invariance test, which would fail because sampling is position-based (`D9`).

- [ ] **Step 2: Add the adapter tests**

In `tests/test_classify_embed_cli.py`, first fix the existing fixture. `test_classify_embed_run_writes_metrics_to_out_dir_and_reports_paths` currently points `--images-dir` at the non-existent `tmp_path / "images"` and relies on the monkeypatched extractor, so the new pre-extraction directory and overlap checks would fail it. Add to that test (and to every new adapter test below):

```python
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (images_dir / name).write_bytes(b"")
```

and pass `images_dir=str(images_dir)`. The placeholder files can be empty bytes because the extractor stays monkeypatched; only the names matter to the overlap scan. Then reuse the existing monkeypatch pattern (fake `extract_embeddings_timm` that also records its calls, `select_device`, `set_num_threads`, `save_log`) and add:

- `test_classify_embed_rejects_duplicate_label_images`: `labels.csv` with `a.jpg` twice and `b.jpg` once; `pytest.raises(ValueError, match="--label-csv")`; assert `not (out_dir / "embeddings.csv").exists()` (validation precedes extraction).
- `test_classify_embed_rejects_label_csv_without_matching_images`: labels only for `zzz.jpg`; `pytest.raises(ValueError, match="matching")`; assert `not out_dir.exists()` **and** that the fake extractor recorded zero calls (validation precedes `check_output_dir()` and extraction).
- `test_classify_embed_prints_na_for_unavailable_metrics`: patch `compute_embedding_metrics` to return `{"NMI": 0.5, "Linear_Probing_Balanced_Acc": None}`; assert `"Linear_Probing_Balanced_Acc: N/A"` in stdout and that the written `metrics.csv` `Linear_Probing_Balanced_Acc` cell is `NaN`.

- [ ] **Step 3: Run the tests and confirm they fail**

```bash
pytest tests/test_classification_embed_metrics.py tests/test_classify_embed_cli.py -q
```

Expected: the new tests fail against the current implementation (perfect `kNN_Acc_k1`, missing `Linear_Probing_Balanced_Acc`, raised `ValueError`s from `Recall@10`/singleton labels, `TypeError` on `None` formatting, `mAP@R == 0.0` on the all-singleton fixture, and the hand-written-ranking helper assertions failing because the helpers do not exist yet). No production code is modified in this task.

---

### Task 2: Cross-validate kNN and linear probing

**Files:**
- Modify: `src/classification/embedder.py`
- Test: `tests/test_classification_embed_metrics.py`

**Interfaces:**
- Consumes: the CV tests from Task 1.
- Produces:
  - `_METRIC_KEYS: tuple[str, ...]` — ordered, module level, single source of the returned keys and the `metrics.csv` column order.
  - `_make_stratified_cv(labels: np.ndarray, max_splits: int = 5)` — module-level private helper.
  - `compute_embedding_metrics(...) -> Dict[str, float | None]` with real CV for kNN and linear probing.

- [ ] **Step 1: Add the ordered key contract**

Above `compute_embedding_metrics` in `src/classification/embedder.py`:

```python
_METRIC_KEYS: tuple[str, ...] = (
    "NMI",
    "ARI",
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "kNN_Acc_k1",
    "kNN_Acc_k5",
    "kNN_Acc_k20",
    "Linear_Probing_Acc",
    "Linear_Probing_Balanced_Acc",
    "mAP@R",
    "Purity",
    "Silhouette_Score",
)
```

- [ ] **Step 2: Add the shared splitter helper**

```python
def _make_stratified_cv(labels: np.ndarray, max_splits: int = 5):
    """Return a shuffled StratifiedKFold, or None when stratified CV is impossible.

    The fold count is bounded by the smallest class count, not by the number of
    classes, so a two-class dataset with enough samples per class still gets
    ``max_splits`` folds. A singleton class cannot be stratified.
    """
    from sklearn.model_selection import StratifiedKFold

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or int(counts.min()) < 2:
        return None
    return StratifiedKFold(
        n_splits=min(max_splits, int(counts.min())),
        shuffle=True,
        random_state=42,
    )
```

- [ ] **Step 3: Rewrite the kNN block as cross-validation**

Replace the inner `def knn_acc(k)` resubstitution helper (currently `src/classification/embedder.py:164-168`) with a single CV block. It must build `cv = _make_stratified_cv(y)` once, compute the smallest training-fold size as `max_train = len(y) - int(np.ceil(len(y) / cv.n_splits))`, skip (leaving `None`) any `k < 1 or k > max_train`, otherwise run:

```python
KNeighborsClassifier(n_neighbors=k, metric="euclidean")
```

through `cross_val_score(..., cv=cv, error_score="raise")` inside `try/except Exception`, and store `float(np.mean(scores))` only when `np.all(np.isfinite(scores))`.

- [ ] **Step 4: Rewrite the linear-probe block as cross-validation**

Replace the `LogisticRegression` fit/score block (currently `src/classification/embedder.py:181-183`) with one `cross_validate` call using the same `cv` object:

```python
cross_validate(
    LogisticRegression(max_iter=1000, random_state=42),
    X_norm,
    y,
    cv=cv,
    scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
    error_score="raise",
)
```

Store `Linear_Probing_Acc` from `test_accuracy` and `Linear_Probing_Balanced_Acc` from `test_balanced_accuracy`, both as finite means or `None`. Add `category=ConvergenceWarning` (imported from `sklearn.exceptions` inside the function) to the existing `warnings.filterwarnings("ignore", ...)` block. On any exception both keys stay `None`.

- [ ] **Step 5: Initialize and return the ordered dict**

Replace the inline return literal with `metrics: Dict[str, float | None] = dict.fromkeys(_METRIC_KEYS)` initialised at the top of the `with` block, filled in place, and returned. Update the function signature's return annotation to `Dict[str, float | None]` and extend the docstring to state: shuffled `StratifiedKFold(shuffle=True, random_state=42)` with the fold count bounded by the smallest class, `None` for metrics that cannot be computed, and that Recall keeps every query in its denominator while mAP excludes queries with no non-self relevant item (`D5`).

- [ ] **Step 6: Run the task's tests**

```bash
pytest tests/test_classification_embed_metrics.py -q
```

Expected: the CV, `kNN_Acc_k1`, balanced-accuracy, fold-count, and unsupported-`k` tests pass. The single-class, Recall-bound, mAP, clustering, and silhouette tests still fail — Task 3 owns them.

---

### Task 3: Correct retrieval, clustering, and unavailable-value semantics

**Files:**
- Modify: `src/classification/embedder.py`
- Test: `tests/test_classification_embed_metrics.py`

**Interfaces:**
- Consumes: `_METRIC_KEYS` and the `dict.fromkeys` initialisation from Task 2.
- Produces:
  - `_recall_at_k(neighbors: np.ndarray, y: np.ndarray, k: int) -> float | None`
  - `_mean_ap_at_r(neighbors: np.ndarray, y: np.ndarray) -> float | None`

  Both are module-level private helpers that take a **precomputed neighbour-index matrix** and exclude the query by index. The matrix is built by the caller, so the exclusion rule is testable with a hand-written ranking: on real data the rule only becomes observable through distance ties whose order is scikit-learn–backend dependent, which would make any hand-computed expected value fragile (`D10`).

- [ ] **Step 1: Guard the empty and single-sample inputs**

At the start of the metric section (after label encoding), return the all-`None` dict when `len(y) == 0`. Every downstream helper then only needs to handle `n >= 1`.

- [ ] **Step 2: Add the two ranking helpers**

```python
def _recall_at_k(neighbors: np.ndarray, y: np.ndarray, k: int) -> float | None:
    """Recall@k over a precomputed ranking, with the query excluded by index.

    The query index is removed by value, not by position: under exact-duplicate
    embeddings the query is not reliably the first entry of its own ranking.
    Every query stays in the denominator, so a singleton query counts as a miss.
    """
    if k < 1 or k > len(y) - 1:
        return None
    hits = 0
    for i in range(len(y)):
        candidates = [int(j) for j in neighbors[i][: k + 1] if j != i][:k]
        if any(y[j] == y[i] for j in candidates):
            hits += 1
    return float(hits / len(y))


def _mean_ap_at_r(neighbors: np.ndarray, y: np.ndarray) -> float | None:
    """mAP@R over a precomputed ranking, with the query excluded by index.

    Unlike recall, queries with no non-self relevant item are excluded from the
    mean instead of counting as a miss.
    """
    aps = []
    for i in range(len(y)):
        r = int((y == y[i]).sum()) - 1
        if r == 0:
            continue
        candidates = [int(j) for j in neighbors[i] if j != i][:r]
        hits = y[candidates] == y[i]
        precisions = hits.cumsum() / (np.arange(len(hits)) + 1)
        aps.append(float((precisions * hits).sum() / r))
    return float(np.mean(aps)) if aps else None
```

- [ ] **Step 3: Build the rankings and call the helpers from `compute_embedding_metrics`**

Recall keeps the current per-k neighbour count: for each requested k in `1, 5, 10`, build `NearestNeighbors(n_neighbors=min(len(y), k + 1), metric="euclidean")`, rank with `kneighbors(X_norm, return_distance=False)`, and call `_recall_at_k(neighbors, y, k)`. The helper returns `None` for `k < 1` or `k > len(y) - 1`, which is the only new bound. Skip the block when `len(y) < 2`.

mAP keeps the **full** ranking, exactly as today: one `NearestNeighbors(n_neighbors=len(y), metric="euclidean")` fit, `kneighbors(X_norm, return_distance=False)`, then `_mean_ap_at_r(neighbors, y)`. Skip it when `len(y) < 2` or no class has a non-self member, leaving `mAP@R` at `None`.

Do **not** shrink the mAP request to `max_relevant + 1` (or any other window): with the exact distance ties produced by duplicate embeddings — the very case this task fixes — a truncated window is not guaranteed to contain the full ranking's first `r` non-query candidates, so the truncation is not value-preserving. The `n × n` index matrix that the full request allocates is pre-existing behaviour and stays recorded in Deferred; it is not part of this correction.

Distance-tie contract (`D10`): the order among exactly equidistant samples is scikit-learn's, unchanged from the pre-0.6.2 implementation. This release changes *which* candidates a query may match (its own index is removed by value) and what happens when a metric cannot be computed — never the tie ordering itself. State this in the helper docstrings rather than introducing a tie-resolution layer.

- [ ] **Step 4: Guard the clustering block**

Wrap the existing `KMeans`/NMI/ARI/Purity block in a single condition:

Wrap the existing `KMeans`/NMI/ARI/Purity block in a single condition:

```python
unique_labels = np.unique(y)
n_clusters = len(unique_labels)
if (
    n_clusters >= 2
    and len(X_norm) >= 2
    and np.unique(X_norm, axis=0).shape[0] >= n_clusters
):
    ...
```

Leave `NMI`, `ARI`, and `Purity` at `None` otherwise. Keep the existing `KMeans(n_clusters=n_clusters, random_state=42, n_init=10)` call parameters and the existing `Counter`-based purity formula.

- [ ] **Step 5: Fix silhouette semantics**

Replace the unconditional call with a guarded one that uses the true-label codes and cosine distance:

```python
sil_idx = np.random.RandomState(42).choice(len(y), min(2000, len(y)), replace=False)
sil_labels = y[sil_idx]
if len(sil_labels) > len(np.unique(sil_labels)) >= 2:
    try:
        metrics["Silhouette_Score"] = float(
            silhouette_score(X_norm[sil_idx], sil_labels, metric="cosine")
        )
    except Exception:
        pass
```

- [ ] **Step 6: Sort the sampled row indices**

Change the subsample draw to `idx = np.sort(np.random.RandomState(42).choice(n, sample_size, replace=False))` and comment that this only canonicalises the order of the selected rows. Do **not** claim row-order invariance: the sampled set is position-based (`D9`).

- [ ] **Step 7: Run the evaluator tests**

```bash
pytest tests/test_classification_embed_metrics.py -q
```

Expected: all evaluator tests pass, including the single-class, Recall-bound helpers, the hand-written-ranking index-exclusion cases, the independent brute-force mAP reference, degenerate-clustering, and cosine-silhouette cases.

---

### Task 4: Validate the label CSV and format unavailable metrics in the CLI

**Files:**
- Modify: `entomokit/classify/embed.py`
- Test: `tests/test_classify_embed_cli.py`

**Interfaces:**
- Consumes: `compute_embedding_metrics()` returning `float | None` values.
- Produces: fail-fast label validation before any output directory or embedding work, plus `N/A` console output and empty CSV cells for unavailable metrics.

- [ ] **Step 1: Load and validate the label CSV before any output or extraction work**

In `run()`, move label loading to just after the existing `--visualize requires --label-csv` check and before `check_output_dir(...)`. Both the duplicate check and the overlap check happen here:

```python
    label_df = None
    if args.label_csv:
        from src.classification.embedder import IMAGE_EXTS
        from src.classification.utils import load_image_csv

        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            raise ValueError(f"--images-dir is not a directory: {images_dir}")

        label_df = load_image_csv(Path(args.label_csv), require_label=True)
        duplicated = label_df["image"][label_df["image"].duplicated()].unique()
        if len(duplicated):
            raise ValueError(
                f"--label-csv contains duplicate image rows: "
                f"{', '.join(map(str, duplicated[:5]))}"
            )

        present = {
            p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        }
        if not (set(label_df["image"].astype(str)) & present):
            raise ValueError(
                f"--label-csv has no 'image' values matching the image file names "
                f"in {images_dir}"
            )
```

Remove the old late `load_image_csv` call and reuse the pre-loaded `label_df` in the metrics block. Do not add validation inside `load_image_csv` (`D8`), and do not add a post-merge `merged.empty` check — the early overlap scan already covers it.

- [ ] **Step 2: Hoist the shared image-extension set**

`IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}` is currently duplicated inside `extract_embeddings_timm` (`src/classification/embedder.py:47`) and `extract_embeddings_ag` (`src/classification/embedder.py:76`). Move it to module level in `src/classification/embedder.py` and use it from both extractors and from the new pre-check, so the pre-check can never drift from what the extractors load. Do not touch `entomokit/classify/predict.py:12` (its own copy serves an unrelated command) or `src/classification/cam.py:133` (a deliberately different set that also accepts `.tif`).

- [ ] **Step 3: Print `N/A` for unavailable metrics**

Replace `print(f"  {k}: {v:.4f}")` with:

```python
        for k, v in metrics.items():
            print(f"  {k}: N/A" if v is None else f"  {k}: {float(v):.4f}")
```

The existing `pd.DataFrame([metrics]).to_csv(...)` call already writes `None` as an empty cell; do not add an explicit `na_rep` or fill step.

- [ ] **Step 4: Run the adapter tests**

```bash
pytest tests/test_classify_embed_cli.py tests/test_resume_flags.py -q
```

Expected: duplicate labels, non-matching labels, and `N/A` formatting are covered; the pre-existing embed CLI test passes with its updated image-directory fixture; and the existing resume-flag tests still pass.

---

### Task 5: Document the metric contract

**Files:**
- Modify: `README.md` (`classify embed` section, lines ~846-886, and the feature bullet on line 48)
- Modify: `README.cn.md` (lines ~857-897, and the bullet on line 48)
- Modify: `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md:324` — one vintage note only
- Do not modify: `docs/superpowers/plans/2026-03-24-phase3-classify.md`, `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` line 324's existing metric list, or any recorded metric value

**Interfaces:**
- Consumes: the final metric contract from Tasks 2-4.
- Produces: matching English/Chinese user documentation of the metric contract and a comparability warning.

- [ ] **Step 1: Update the `classify embed` quality-metric table**

Add the `Linear_Probing_Balanced_Acc` row and mark which fields can be empty:

| Metric | Description |
| --- | --- |
| NMI / ARI | Clustering agreement with the true labels |
| Recall@1/5/10 | Retrieval recall at K; every query is in the denominator |
| kNN_Acc_k1/5/20 | Cross-validated k-NN accuracy |
| Linear_Probing_Acc | Cross-validated linear-probe accuracy |
| Linear_Probing_Balanced_Acc | Cross-validated linear-probe balanced accuracy |
| mAP@R | Mean Average Precision at R (queries with no non-self relevant item are excluded) |
| Purity | Cluster purity |
| Silhouette_Score | Cosine silhouette against the true labels |

- [ ] **Step 2: Add the metric contract and comparability warning**

Directly under the table, state: kNN and linear probing use `StratifiedKFold(n_splits=min(5, smallest class count), shuffle=True, random_state=42)`; clustering uses KMeans with the true class count; metrics that cannot be computed (singleton class, too few samples, degenerate clustering) are written as empty cells and printed as `N/A`; `--label-csv` must have unique `image` values and at least one match in `--images-dir`; and **values are not numerically comparable with runs produced before 0.6.2** because CV, silhouette distance, self-exclusion, and unavailable-value handling changed. Also state that `--metrics-sample-size` caps the number of rows used by *all* embedding-quality metrics (clustering, Recall@K, kNN, mAP@R, silhouette, and linear probing) — it is not specific to the linear probe — and that it is the primary runtime and memory knob, including for mAP@R's neighbourhood search.

Mirror the same content in `README.cn.md` in Chinese.

- [ ] **Step 3: Mark the historical spec**

Append a single italic line to `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` at the end of line 324's paragraph, stating that its metric list predates the 0.6.2 evaluator corrections described in this plan and that its field names remain current while its values are not comparable. Do not alter the list or any recorded value.

- [ ] **Step 4: Verify the documentation**

```bash
rg -n "Linear_Probing_Balanced_Acc|StratifiedKFold|0\.6\.2" README.md README.cn.md docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md
```

Expected: both READMEs list the new field and the CV/splitter contract; the historical spec carries the vintage note.

---

### Task 6: Release version 0.6.2

**Files:**
- Modify: `version.txt:1`, `entomokit/_version.py:3`, `entomokit/main.py:87`, `setup.py:5`
- Modify: `tests/test_package_version.py`, `tests/test_main_cli.py` (three assertions: lines 261, 275, 280)

**Interfaces:**
- Consumes: the implementation and documentation from Tasks 2-5.
- Produces: `entomokit --version` prints `entomokit 0.6.2`, and every live version declaration matches.

- [ ] **Step 1: Set the version in all four declarations**

Set `0.6.2` in `version.txt`, `entomokit/_version.py` (`__version__`), the `PackageNotFoundError` fallback in `entomokit/main.py`, and `version="0.6.2"` in `setup.py`.

- [ ] **Step 2: Update the current-version tests**

Update `test_setup_version_is_0_6_1` (rename to `test_setup_version_is_0_6_2`, fix the docstring and the `version="0.6.2"` assertion) and all three `0.6.1` assertions in `tests/test_main_cli.py`.

- [ ] **Step 3: Run the version tests**

```bash
pytest tests/test_package_version.py tests/test_main_cli.py -q
```

Expected: all pass and no live code path still reports `0.6.1`.

---

### Task 7: Verify the whole suite and inspect the diff

**Files:**
- Modify: none unless verification exposes a task-owned failure
- Test: full repository suite

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: a verified 0.6.2 working tree with no unrelated changes.

- [ ] **Step 1: Run the full suite**

```bash
pytest -q
```

Expected: previously 291 collected tests, all green plus the new ones. Any failure must be fixed in its owning task before declaring the plan complete.

- [ ] **Step 2: Run the static regression searches**

```bash
rg -n "liblinear" src/classification/embedder.py
rg -n "knn\.fit\(X_norm" src/classification/embedder.py
rg -n "lr\.fit\(X_norm" src/classification/embedder.py
rg -n "return float\(np\.mean\(aps\)\) if aps else 0\.0" src/classification/embedder.py
rg -n "\[:, 1:\]|max_relevant" src/classification/embedder.py
rg -n "_METRIC_KEYS|_make_stratified_cv|_recall_at_k|_mean_ap_at_r" src/classification/embedder.py
rg -n "0\.6\.1" setup.py version.txt entomokit/_version.py entomokit/main.py tests/test_main_cli.py tests/test_package_version.py
```

Expected: the first five searches have no matches — in particular there must be no positional `[:, 1:]` self-slice and no truncated `max_relevant`-based mAP window; `_METRIC_KEYS`, `_make_stratified_cv`, `_recall_at_k`, and `_mean_ap_at_r` are each defined once and referenced by the metric body; the last search has no matches.

- [ ] **Step 3: Re-measure the acceptance numbers**

```bash
python - <<'PY'
import numpy as np
from src.classification.embedder import compute_embedding_metrics
rng = np.random.RandomState(1)
n, d, c = 3000, 128, 20
x = rng.randn(n, d)
y = np.repeat([f"c{i}" for i in range(c)], n // c)
m = compute_embedding_metrics(x, y)
print({k: (None if v is None else round(float(v), 4)) for k, v in m.items()})
assert m["kNN_Acc_k1"] < 0.5, m["kNN_Acc_k1"]
assert set(m) == set(__import__("src.classification.embedder", fromlist=["x"])._METRIC_KEYS)
print("OK")
PY
```

Expected: `kNN_Acc_k1` is near chance, not `1.0`; the key set equals `_METRIC_KEYS`.

- [ ] **Step 4: Inspect the diff**

```bash
git diff --check
git diff --stat
git status --short
```

Expected: only `src/classification/embedder.py`, `entomokit/classify/embed.py`, the two test files, both READMEs, the historical spec's one-line note, the four version declarations, and both version tests are changed (plus this plan document). No unrelated edits, no whitespace errors.

- [ ] **Step 5: Commit the release**

**This step requires explicit approval from the repository owner. Do not run it otherwise** — recording the command here is not authorization to commit.

```bash
git add -A
git commit -m "fix(classify)!: cross-validated embed metrics with unavailable-value semantics"
git log --oneline -1
git status --short
```

Expected: one release commit (repository convention is one commit per release, matching `docs/superpowers/plans/2026-09-15-classify-cam-unnormalized-arrays.md`), clean working tree. The `!` is required because previously published metric values change meaning; the message must not claim the numbers are comparable with earlier runs.

- [ ] **Step 6: Record the compatibility note**

The release note or final change summary must state that pre-0.6.2 `metrics.csv` files are not numerically comparable with 0.6.2 output because of four changes (explicit shuffled 5-fold CV for kNN and linear probing, cosine silhouette, index-based self-exclusion in Recall and mAP@R, and `None`/empty semantics for unsupported metrics), that all previous field names remain available (`--metrics-sample-size` also no longer feeds an `n × n` mAP neighbourhood search), and that `Linear_Probing_Balanced_Acc` is a new column inserted after `Linear_Probing_Acc`, which shifts all later columns right by one — select columns by header name — and is empty in old rows.

---

## Deferred Deliberate Simplifications

- No `AMI` field: it is a v0.7.1 feature addition, not a correction, and entomokit's clustering block is already the largest part of the function.
- No optional-mechanics removal: OTU-Former's `compute_linear_probing()` wrapper does not exist in entomokit, and `compute_embedding_metrics()` stays the only public entry point. Its three private helpers (`_make_stratified_cv`, `_recall_at_k`, `_mean_ap_at_r`) exist for a testability reason, not for reuse (`D10`); do not add a fourth.
- No train/test ID canonicalisation pipeline and no shared sampling module. `np.sort` on one line canonicalises the selected-row order; metrics stay position-based (`D9`), and stable-ID sampling is the upgrade path if cross-order comparability is ever required.
- No chunked mAP ranking. The mAP path keeps today's full `n_neighbors=len(y)` request, which allocates an `n × n` index matrix (~800 MB at the default `sample_size=10000`). Bounding it would change values under distance ties, and a memory-bounded rewrite (query-row chunks or distance-based tie expansion with a stable secondary sort) is a separate change with its own tie-contract decision. Revisit only if mAP memory becomes a reported problem.
- Recall and mAP keep different singleton-query denominators (`D5`); unifying them is a metric-contract revision, not a correction.
- No `--metrics-solver` flag or any other new CLI option; the solver choice is fixed at `lbfgs` (`D4`) and `--metrics-sample-size` is the only cost knob.

## Risk Register

| Risk | Mitigation |
| --- | --- |
| Existing `runs/**/metrics.csv` from earlier versions is silently compared against new values | README (both languages) and the commit message state the incomparability explicitly (`D7`); field names are unchanged so tooling keeps loading |
| The new `--images-dir`/`--label-csv` validation breaks the existing fake-extractor CLI test | `Task 1` Step 2 creates the image directory and empty `a.jpg`/`b.jpg` placeholders in that fixture and reuses it for the new tests; the fake extractor is only needed to avoid loading a real model, not to bypass the directory check |
| The mAP `n × n` neighbour-index matrix (`~800 MB` at the default `--metrics-sample-size 10000`) is mistaken for something this release fixes | It is pre-existing and explicitly out of scope (`Task 3` Step 3, Deferred); the correction must not introduce a truncated mAP window, because truncation is not tie-equivalent |
| CV makes large-class-count runs slow enough to look hung | Measured 0.22 s for 100 classes × 5000 samples × 5 folds; `--metrics-sample-size` remains the documented knob; no timeout is added |
| `LogisticRegression` hits `max_iter` on real embeddings and prints warnings | `max_iter=1000` plus `ConvergenceWarning` suppression inside the existing `catch_warnings` block; a non-finite or failed fold yields `None` rather than a wrong number |
| A single-class or too-small label CSV now produces a row of `N/A` instead of a crash, and someone reads it as a successful evaluation | The CLI prints `N/A` per field and the README documents that empty cells mean "not computable"; the empty-overlap case still raises (`Task 4`) instead of reporting all-`N/A` |
| The duplicate-`image` check is added to the shared `load_image_csv` and breaks `classify predict` | Validation is local to `embed.run()` (`D8`); `tests/test_classify_predict_cli.py` must stay green |
| Silhouette's `metric="cosine"` change is mistaken for a bug fix that preserves values | Documented as a semantic change (`D7`); the test asserts equality against a direct `silhouette_score(..., metric="cosine")` call rather than against a historical number |
| Only one side of the Recall/mAP rule is verified, and a future edit reintroduces a positional `[:, 1:]` | `D10`'s helpers take the ranking matrix, so the hand-written fixture with the query at rank 1 fails immediately in both helpers, independently of the installed scikit-learn version |
## Post-Implementation Corrections (review follow-up, after Task 7 verification)

Two review findings were fixed after the tasks above were executed. The task text above is left unchanged so the plan stays a record of the reviewed intent.

- **The single-class documentation was wrong.** The README bullet listed `singleton class` as making a metric unavailable, but only clustering, the CV-based kNN/linear probing, and silhouette become `None`; `Recall@K` and `mAP@R` keep their table definitions and return `1.0` when a single class has more than one sample (measured: `Recall@1 = Recall@5 = mAP@R = 1.0`, `Recall@10 = None` because `k > n - 1`). Both READMEs now say "some metrics" and state the Recall/mAP exception explicitly, and `test_single_class_keeps_recall_and_map_numeric` locks the `D5` behaviour so a future edit cannot blank those fields by mistake.
- **Image-name matching ignored the file type.** The pre-extraction overlap check in `entomokit/classify/embed.py` and both extractor scans in `src/classification/embedder.py` matched on suffix alone, so a *directory* named `x.jpg` satisfied the check and then failed inside `Image.open()` after the output directory had already been created. All three scans now require `p.is_file()`, matching the existing `entomokit/classify/predict.py:82` and `src/classification/cam.py:167` convention, and `test_classify_embed_ignores_directories_named_like_images` covers it.

Task 3's full-ranking requirement (and Task 1's independent brute-force mAP reference, which replaced the earlier bounded-ranking equivalence test) was already in effect before execution, so no plan text needed changing there: `n_neighbors=len(y)` is the implemented behaviour and the `n × n` allocation stays recorded in Deferred. The README wording was tightened so it cannot be read as an improvement: `mAP@R`'s neighbour-index matrix grows with the square of `--metrics-sample-size` (~800 MB at the default 10000).
