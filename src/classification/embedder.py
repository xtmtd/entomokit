"""Embedding extraction, quality metrics, and UMAP visualization."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Image suffixes both embedders load; the CLI reuses this to pre-validate
# --label-csv against --images-dir.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class _ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


def extract_embeddings_timm(
    images_dir: Path,
    base_model: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Extract embeddings using a pretrained timm backbone (no fine-tuning)."""
    import timm
    from timm.data import resolve_model_data_config
    from timm.data.transforms_factory import create_transform

    model = timm.create_model(base_model, pretrained=True, num_classes=0)
    model.eval().to(device)

    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )

    dataset = _ImageDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    all_embeddings, all_paths = [], []
    with torch.no_grad():
        for batch_tensors, batch_paths in loader:
            feats = model(batch_tensors.to(device)).cpu().numpy()
            all_embeddings.append(feats)
            all_paths.extend(batch_paths)

    embeddings = np.vstack(all_embeddings)
    df = pd.DataFrame(
        embeddings, columns=[f"feat_{i}" for i in range(embeddings.shape[1])]
    )
    df.insert(0, "image", [Path(p).name for p in all_paths])
    return df


def extract_embeddings_ag(
    images_dir: Path,
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Extract embeddings using a fine-tuned AutoGluon model."""
    paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )
    df_in = pd.DataFrame({"image": [str(p) for p in paths]})

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.",
            category=UserWarning,
        )
        from autogluon.multimodal import MultiModalPredictor

        predictor = MultiModalPredictor.load(str(model_dir))
        embeddings = np.asarray(predictor.extract_embedding(df_in), dtype=np.float32)

    embed_df = pd.DataFrame(
        embeddings, columns=[f"feat_{i}" for i in range(embeddings.shape[1])]
    )
    embed_df.insert(0, "image", [p.name for p in paths])
    return embed_df


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


def compute_embedding_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10000,
) -> Dict[str, float | None]:
    """Compute embedding space quality metrics.

    Returns the keys in ``_METRIC_KEYS``: NMI, ARI, Recall@1/5/10,
    kNN_Acc_k1/5/20, Linear_Probing_Acc, Linear_Probing_Balanced_Acc, mAP@R,
    Purity, Silhouette_Score.

    Contract:
      * kNN and linear probing are cross-validated with
        ``StratifiedKFold(shuffle=True, random_state=42)`` whose fold count is
        bounded by the smallest class count.
      * Metrics that cannot be computed (singleton class, too few samples,
        degenerate clustering, no rankable query) are ``None``; the CLI writes
        them as empty CSV cells and prints ``N/A``.
      * Recall@K keeps every query in its denominator, while mAP@R excludes
        queries with no non-self relevant item.
      * Only a query's own index is excluded from its ranking; the order of
        exactly equidistant neighbours stays whatever scikit-learn returns.
      * Silhouette_Score is a cosine silhouette against the true labels.
    """
    from collections import Counter

    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
    )
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearn.preprocessing import LabelEncoder

    metrics: Dict[str, float | None] = dict.fromkeys(_METRIC_KEYS)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        y = LabelEncoder().fit_transform(labels)
        if len(y) == 0:
            return metrics

        # Subsample if needed; sorted so the selected rows keep a canonical order.
        n = len(y)
        if 0 < sample_size < n:
            idx = np.sort(np.random.RandomState(42).choice(n, sample_size, replace=False))
            X, y = embeddings[idx], y[idx]
        else:
            X = embeddings

        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / np.where(norms > 0, norms, 1)

        cv = _make_stratified_cv(y)

        # Clustering metrics: KMeans says nothing when there are fewer distinct
        # points than clusters, and would silently return a degenerate labelling
        # (plus a ConvergenceWarning) instead of failing.
        unique_labels = np.unique(y)
        n_clusters = len(unique_labels)
        if (
            n_clusters >= 2
            and len(X_norm) >= 2
            and np.unique(X_norm, axis=0).shape[0] >= n_clusters
        ):
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(X_norm)

            metrics["NMI"] = float(normalized_mutual_info_score(y, cluster_labels))
            metrics["ARI"] = float(adjusted_rand_score(y, cluster_labels))

            purity_sum = sum(
                Counter(y[cluster_labels == c]).most_common(1)[0][1]
                for c in np.unique(cluster_labels)
            )
            metrics["Purity"] = float(purity_sum / len(y))

        # Recall@K: the query is excluded by index, every query stays in the
        # denominator.
        if len(y) >= 2:
            for k in (1, 5, 10):
                if k < 1 or k > len(y) - 1:
                    continue
                neighbors = (
                    NearestNeighbors(n_neighbors=min(len(y), k + 1), metric="euclidean")
                    .fit(X_norm)
                    .kneighbors(X_norm, return_distance=False)
                )
                metrics[f"Recall@{k}"] = _recall_at_k(neighbors, y, k)

        # kNN accuracy: cross-validated, so k must fit the smallest training fold.
        if cv is not None:
            max_train_size = len(y) - int(np.ceil(len(y) / cv.n_splits))
            for k in (1, 5, 20):
                if k < 1 or k > max_train_size:
                    continue
                try:
                    scores = cross_val_score(
                        KNeighborsClassifier(n_neighbors=k, metric="euclidean"),
                        X_norm,
                        y,
                        cv=cv,
                        error_score="raise",
                    )
                except Exception:
                    continue
                if np.all(np.isfinite(scores)):
                    metrics[f"kNN_Acc_k{k}"] = float(np.mean(scores))

        # Linear probing: one shared CV pass, ordinary and balanced accuracy.
        if cv is not None:
            try:
                scores = cross_validate(
                    LogisticRegression(max_iter=1000, random_state=42),
                    X_norm,
                    y,
                    cv=cv,
                    scoring={
                        "accuracy": "accuracy",
                        "balanced_accuracy": "balanced_accuracy",
                    },
                    error_score="raise",
                )
            except Exception:
                pass
            else:
                if np.all(np.isfinite(scores["test_accuracy"])):
                    metrics["Linear_Probing_Acc"] = float(
                        np.mean(scores["test_accuracy"])
                    )
                if np.all(np.isfinite(scores["test_balanced_accuracy"])):
                    metrics["Linear_Probing_Balanced_Acc"] = float(
                        np.mean(scores["test_balanced_accuracy"])
                    )

        # mAP@R: mean Average Precision at R (R = number of same-class samples
        # excluding the query). The full ranking is kept: a truncated window is
        # not tie-equivalent to it when duplicate embeddings are present.
        if len(y) >= 2:
            neighbors = (
                NearestNeighbors(n_neighbors=len(y), metric="euclidean")
                .fit(X_norm)
                .kneighbors(X_norm, return_distance=False)
            )
            metrics["mAP@R"] = _mean_ap_at_r(neighbors, y)

        # Silhouette: true-label cosine silhouette on a capped sample.
        sil_idx = np.random.RandomState(42).choice(
            len(y), min(2000, len(y)), replace=False
        )
        sil_labels = y[sil_idx]
        if len(sil_labels) > len(np.unique(sil_labels)) >= 2:
            try:
                metrics["Silhouette_Score"] = float(
                    silhouette_score(X_norm[sil_idx], sil_labels, metric="cosine")
                )
            except Exception:
                pass

    return metrics


def visualize_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
) -> None:
    """Generate and save a UMAP scatter plot coloured by label."""
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    embedding_2d = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    colors = [color_map[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[color_map[lbl]],
            label=lbl,
            s=5,
            alpha=0.7,
        )
    ax.legend(markerscale=3, bbox_to_anchor=(1, 1), loc="upper left", fontsize=8)
    ax.set_title("UMAP Embedding Visualization")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
