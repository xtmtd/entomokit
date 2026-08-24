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
    recursive: bool = False,
) -> pd.DataFrame:
    """Extract embeddings using a pretrained timm backbone (no fine-tuning)."""
    import timm
    from timm.data import resolve_model_data_config
    from timm.data.transforms_factory import create_transform

    model = timm.create_model(base_model, pretrained=True, num_classes=0)
    model.eval().to(device)

    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if recursive:
        paths = sorted(
            [
                p
                for p in images_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
        )
    else:
        paths = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
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
    recursive: bool = False,
) -> pd.DataFrame:
    """Extract embeddings using a fine-tuned AutoGluon model."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if recursive:
        paths = sorted(
            [
                p
                for p in images_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
        )
    else:
        paths = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
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


def compute_embedding_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10000,
) -> Dict[str, float]:
    """Compute embedding space quality metrics.

    Returns dict with: NMI, ARI, Recall@1/5/10, kNN_Acc_k1/5/20,
    Linear_Probing_Acc, Purity, Silhouette_Score.
    """
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )

        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Subsample if needed
        n = len(y)
        if 0 < sample_size < n:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, sample_size, replace=False)
            X, y = embeddings[idx], y[idx]
        else:
            X = embeddings

        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / np.where(norms > 0, norms, 1)

        # Clustering metrics
        from sklearn.cluster import KMeans

        n_clusters = len(np.unique(y))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_norm)

        nmi = normalized_mutual_info_score(y, cluster_labels)
        ari = adjusted_rand_score(y, cluster_labels)

        # Purity
        from collections import Counter

        purity_sum = sum(
            Counter(y[cluster_labels == c]).most_common(1)[0][1]
            for c in np.unique(cluster_labels)
        )
        purity = purity_sum / len(y)

        # kNN accuracy
        def knn_acc(k):
            knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            knn.fit(X_norm, y)
            return knn.score(X_norm, y)

        # Recall@K
        def recall_at_k(k):
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
            nbrs.fit(X_norm)
            indices = nbrs.kneighbors(X_norm, return_distance=False)[:, 1:]
            hits = sum(any(y[j] == y[i] for j in indices[i]) for i in range(len(y)))
            return hits / len(y)

        # Linear probing
        lr = LogisticRegression(max_iter=500, random_state=42, solver="liblinear")
        lr.fit(X_norm, y)
        lp_acc = lr.score(X_norm, y)

        # Silhouette (sample 2000 for speed)
        sil_idx = np.random.RandomState(42).choice(
            len(y), min(2000, len(y)), replace=False
        )
        sil = silhouette_score(X_norm[sil_idx], y[sil_idx])

        # mAP@R: mean Average Precision at R (R = number of same-class samples)
        def mean_ap_at_r() -> float:
            from sklearn.neighbors import NearestNeighbors

            n = len(y)
            nbrs = NearestNeighbors(n_neighbors=n, metric="euclidean")
            nbrs.fit(X_norm)
            indices = nbrs.kneighbors(X_norm, return_distance=False)[:, 1:]
            aps = []
            for i in range(n):
                r = (
                    int((y == y[i]).sum()) - 1
                )  # number of same-class samples excluding self
                if r == 0:
                    continue
                retrieved = indices[i, :r]
                hits = y[retrieved] == y[i]
                precisions = hits.cumsum() / (np.arange(len(hits)) + 1)
                aps.append((precisions * hits).sum() / r)
            return float(np.mean(aps)) if aps else 0.0

        return {
            "NMI": nmi,
            "ARI": ari,
            "Recall@1": recall_at_k(1),
            "Recall@5": recall_at_k(5),
            "Recall@10": recall_at_k(10),
            "kNN_Acc_k1": knn_acc(1),
            "kNN_Acc_k5": knn_acc(5),
            "kNN_Acc_k20": knn_acc(20),
            "Linear_Probing_Acc": lp_acc,
            "mAP@R": mean_ap_at_r(),
            "Purity": purity,
            "Silhouette_Score": sil,
        }


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
