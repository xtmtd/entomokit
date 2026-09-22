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


def _capture_cv(monkeypatch, validate_payload=None):
    """Patch the sklearn CV entry points and return the captured ``cv`` objects."""
    import sklearn.model_selection as model_selection

    captured: dict[str, list] = {}

    def fake_cross_val_score(estimator, X, y, **kwargs):
        captured.setdefault("cross_val_score", []).append(kwargs.get("cv"))
        return np.array([0.9, 0.9])

    def fake_cross_validate(estimator, X, y, **kwargs):
        captured.setdefault("cross_validate", []).append(kwargs.get("cv"))
        if validate_payload is not None:
            return validate_payload
        return {
            "test_accuracy": np.array([0.9, 0.9]),
            "test_balanced_accuracy": np.array([0.8, 0.8]),
        }

    monkeypatch.setattr(model_selection, "cross_val_score", fake_cross_val_score)
    monkeypatch.setattr(model_selection, "cross_validate", fake_cross_validate)
    return captured


def _hand_ranking():
    """Ranking where query 0's own index sits at rank 1 (duplicate-vector case)."""
    return np.array([[1, 0, 2], [0, 1, 2], [2, 0, 1]])


def _brute_force_map(x: np.ndarray, y: np.ndarray) -> float | None:
    """AP@R with the query excluded by index, ranked by cosine similarity."""
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    similarity = x_norm @ x_norm.T
    aps = []
    for i in range(len(y)):
        r = int((y == y[i]).sum()) - 1
        if r == 0:
            continue
        order = np.argsort(-similarity[i])
        candidates = [int(j) for j in order if j != i][:r]
        hits = y[candidates] == y[i]
        precisions = hits.cumsum() / (np.arange(len(hits)) + 1)
        aps.append(float((precisions * hits).sum() / r))
    return float(np.mean(aps)) if aps else None


# ── cross-validation contract ────────────────────────────────────────────────


def test_knn_and_linear_probing_use_shuffled_five_fold_stratified_cv(monkeypatch):
    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(3, 10)
    captured = _capture_cv(monkeypatch)

    compute_embedding_metrics(x, y)

    assert captured["cross_val_score"], "kNN must use cross_val_score"
    assert captured["cross_validate"], "linear probing must use cross_validate"
    for cv in captured["cross_val_score"] + captured["cross_validate"]:
        assert cv.n_splits == 5
        assert cv.shuffle is True
        assert cv.random_state == 42


def test_smallest_class_bounds_fold_count(monkeypatch):
    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(3, 3)
    captured = _capture_cv(monkeypatch)

    compute_embedding_metrics(x, y)

    for cv in captured["cross_val_score"] + captured["cross_validate"]:
        assert cv.n_splits == 3


def test_uninformative_embeddings_do_not_give_perfect_knn_k1():
    from src.classification.embedder import compute_embedding_metrics

    rng = np.random.RandomState(3)
    x = rng.randn(20 * 15, 8)
    y = np.repeat([f"class_{c}" for c in range(20)], 15)

    metrics = compute_embedding_metrics(x, y)

    assert metrics["kNN_Acc_k1"] < 0.5


def test_linear_probing_field_mapping(monkeypatch):
    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(3, 10)
    _capture_cv(
        monkeypatch,
        validate_payload={
            "test_accuracy": np.array([0.8, 0.8]),
            "test_balanced_accuracy": np.array([0.3, 0.3]),
        },
    )

    metrics = compute_embedding_metrics(x, y)

    assert metrics["Linear_Probing_Acc"] == pytest.approx(0.8)
    assert metrics["Linear_Probing_Balanced_Acc"] == pytest.approx(0.3)


def test_linear_probing_reports_both_finite_scores():
    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(3, 12)

    metrics = compute_embedding_metrics(x, y)

    assert np.isfinite(metrics["Linear_Probing_Acc"])
    assert np.isfinite(metrics["Linear_Probing_Balanced_Acc"])


def test_unsupported_k_is_none_and_supported_k_stays_numeric():
    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(2, 3, dim=4)

    metrics = compute_embedding_metrics(x, y)

    assert metrics["kNN_Acc_k20"] is None
    assert isinstance(metrics["kNN_Acc_k1"], float)


def test_single_class_returns_unavailable_instead_of_raising():
    from src.classification.embedder import compute_embedding_metrics

    x, _ = _blobs(1, 8, dim=4)
    y = np.array(["only"] * 8)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = compute_embedding_metrics(x, y)

    for key in (
        "NMI",
        "ARI",
        "Purity",
        "Silhouette_Score",
        "kNN_Acc_k1",
        "Linear_Probing_Acc",
        "Linear_Probing_Balanced_Acc",
    ):
        assert metrics[key] is None, key


def test_single_class_keeps_recall_and_map_numeric():
    """D5: Recall@K and mAP@R keep their own denominator rules, so a single
    class with more than one sample is 1.0 rather than unavailable."""
    from src.classification.embedder import compute_embedding_metrics

    x, _ = _blobs(1, 8, dim=4)

    metrics = compute_embedding_metrics(x, np.array(["only"] * 8))

    assert metrics["Recall@1"] == pytest.approx(1.0)
    assert metrics["Recall@5"] == pytest.approx(1.0)
    assert metrics["Recall@10"] is None  # k > n - 1
    assert metrics["mAP@R"] == pytest.approx(1.0)


# ── retrieval ranking rule ───────────────────────────────────────────────────


def test_recall_helper_bounds_are_per_k():
    from src.classification.embedder import _recall_at_k

    neighbors = np.tile(np.arange(10), (10, 1))
    y = np.array(["a"] * 5 + ["b"] * 5)

    assert _recall_at_k(neighbors, y, 10) is None
    assert _recall_at_k(neighbors, y, 0) is None
    assert isinstance(_recall_at_k(neighbors, y, 1), float)


def test_recall_helper_excludes_the_query_by_index():
    from src.classification.embedder import _recall_at_k

    ranking = _hand_ranking()
    y = np.array(["a", "b", "a"])

    assert _recall_at_k(ranking, y, 1) == pytest.approx(1 / 3)


def test_map_helper_excludes_the_query_by_index():
    from src.classification.embedder import _mean_ap_at_r

    ranking = _hand_ranking()
    y = np.array(["a", "b", "a"])

    assert _mean_ap_at_r(ranking, y) == pytest.approx(0.5)


def test_map_helper_is_none_when_no_query_has_a_relevant_pair():
    from src.classification.embedder import _mean_ap_at_r

    neighbors = np.tile(np.arange(6), (6, 1))
    y = np.array(["a", "b", "c", "d", "e", "f"])

    assert _mean_ap_at_r(neighbors, y) is None


def test_map_matches_an_independent_brute_force_reference():
    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(3, 12, dim=8, seed=5)

    metrics = compute_embedding_metrics(x, y)

    assert metrics["mAP@R"] == pytest.approx(_brute_force_map(x, y))


# ── bounds, degeneracy, unavailable values ───────────────────────────────────


def test_recall_bounds_are_per_k_on_real_data():
    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(2, 5, dim=4)

    metrics = compute_embedding_metrics(x, y)

    assert metrics["Recall@10"] is None
    assert isinstance(metrics["Recall@1"], float)
    assert isinstance(metrics["Recall@5"], float)


def test_map_is_none_when_no_query_has_a_relevant_pair():
    from src.classification.embedder import compute_embedding_metrics

    x = np.eye(6)
    y = np.array(["a", "b", "c", "d", "e", "f"])

    metrics = compute_embedding_metrics(x, y)

    assert metrics["mAP@R"] is None


def test_clustering_is_none_when_distinct_rows_fewer_than_classes():
    from src.classification.embedder import compute_embedding_metrics

    x = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    y = np.array(["a", "a", "b", "b", "c", "c"])

    metrics = compute_embedding_metrics(x, y)

    assert metrics["NMI"] is None
    assert metrics["ARI"] is None
    assert metrics["Purity"] is None


def test_silhouette_uses_cosine_and_is_undefined_for_one_class():
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(2, 8, dim=4, seed=1)
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    codes = LabelEncoder().fit_transform(y)
    sil_idx = np.random.RandomState(42).choice(
        len(y), min(2000, len(y)), replace=False
    )
    expected = silhouette_score(x_norm[sil_idx], codes[sil_idx], metric="cosine")

    metrics = compute_embedding_metrics(x, y)

    assert metrics["Silhouette_Score"] == pytest.approx(expected)

    single_x, _ = _blobs(1, 8, dim=4)
    single = compute_embedding_metrics(single_x, np.array(["only"] * 8))
    assert single["Silhouette_Score"] is None


def test_metrics_are_deterministic_for_identical_input():
    from src.classification.embedder import compute_embedding_metrics

    x, y = _blobs(6, 40, seed=3)

    first = compute_embedding_metrics(x, y, sample_size=120)
    second = compute_embedding_metrics(x, y, sample_size=120)

    assert first == second
