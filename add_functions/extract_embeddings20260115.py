#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract image embeddings with/without fine-tuning.
Usage:
    python extract_embeddings.py \
        --input_images_dir data/to_embed \
        --base_model convnextv2_femto \
        --out_dir results \
        --batch_size 16

require fine-tuning using new data:
    python extract_embeddings.py \
        --input_images_dir data/to_embed \
        --train_data train.csv \
        --train_images_dir data/train_images \
        --base_model convnextv2_femto \
        --Focal_Loss --focal_loss_gamma 1.0 \
        --max_epochs 50 --time_limit 1 \
        --out_dir results

directly load trained AutoGluon models:
    python extract_embeddings.py \
        --input_images_dir data/to_embed \
        --load_AutogluonModel results/AutogluonModels/convnextv2_femto \
        --out_dir results

save both train and test embeddings:
    python extract_embeddings.py \
        --input_images_dir data/to_embed \
        --train_data train.csv \
        --train_images_dir data/train_images \
        --base_model convnextv2_femto \
        --save_train_embeddings \
        --out_dir results

evaluate and predict classification performance:
    python extract_embeddings.py \
        --input_images_dir data/to_embed \
        --train_data train.csv \
        --train_images_dir data/train_images \
        --base_model convnextv2_femto \
        --evaluate_classification \
        --predict_classification \
        --out_dir results
"""

import argparse
import os
import atexit
import sys
import warnings
import csv
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# ---- block NVML before importing AutoGluon ----
os.environ.setdefault("AG_AUTOMM_DISABLE_NVML", "1")
os.environ.setdefault("AUTOMM_DISABLE_NVML", "1")
def _install_dummy_pynvml():
    try:
        import pynvml
    except ImportError:
        return
    class _DummyHandle:
        pass
    class _DummyMemInfo:
        def __init__(self):
            self.total = 0
            self.free = 0
            self.used = 0
    def _noop(*args, **kwargs):
        return None
    pynvml.nvmlInit = _noop
    pynvml.nvmlShutdown = _noop
    pynvml.nvmlDeviceGetCount = lambda: 0
    pynvml.nvmlDeviceGetHandleByIndex = lambda *_: _DummyHandle()
    pynvml.nvmlDeviceGetName = lambda *_: "Fake GPU"
    pynvml.nvmlDeviceGetMemoryInfo = lambda *_: _DummyMemInfo()
    pynvml.nvmlSystemGetDriverVersion = lambda: "0.0"
_install_dummy_pynvml()

import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from PIL import Image
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import umap.umap_ as umap
import math
from torch.utils.data import DataLoader, Dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

warnings.filterwarnings("ignore", category=UserWarning)



try:
    from autogluon.multimodal import MultiModalPredictor
except ImportError:
    MultiModalPredictor = None
    print("Warning: AutoGluon is not installed. Only direct TIMM extraction will be available.", file=sys.stderr)


# --------------------------- Dataset helpers --------------------------- #
class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["image"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx]


class TimmEmbedder:
    def __init__(self, model_name: str, device: torch.device):
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(device).eval()
        self.device = device
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config, is_training=False)

    def extract(self, df: pd.DataFrame, batch_size: int, num_workers: int):
        dataset = ImageDataset(df, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        feats = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                feats.append(self.model(images).cpu())
        return torch.cat(feats, dim=0).numpy()


# --------------------------- Utility functions --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings via timm backbone or AutoGluon fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_model", type=str, default="convnextv2_femto",
                        help="Backbone name registered in timm.")
    parser.add_argument("--input_images_dir", type=str, required=True,
                        help="Directory of images to embed (filenames without suffix must be unique).")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Folder to store embeddings, processed CSVs, UMAP plots, etc.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None,
                        help="Force a specific device for TIMM extraction and (optionally) AutoGluon training.")

    # Fine-tuning
    parser.add_argument("--train_data", type=str, default=None,
                        help="CSV (image,label) for fine-tuning.")
    parser.add_argument("--train_images_dir", type=str, default=None,
                        help="Directory containing training images referenced by --train_data.")
    parser.add_argument("--load_AutogluonModel", type=str, default=None,
                        help="Path to an existing AutoGluon predictor directory.")
    parser.add_argument("--Focal_Loss", action="store_true",
                        help="Enable focal loss to fight imbalance during fine-tuning.")
    parser.add_argument("--focal_loss_gamma", type=float, default=1.0,
                        help="Gamma value for focal loss.")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum epochs for AutoGluon training.")
    parser.add_argument("--time_limit", type=float, default=1.0,
                        help="Training time limit (hours).")
    parser.add_argument("--disable_augment", action="store_true",
                        help="Disable default augmentation list during AutoGluon training.")

    # Embedding extraction
    parser.add_argument("--save_train_embeddings", action="store_true",
                        help="Save training data embeddings to embeddings.train.csv. "
                             "Requires --train_data and --train_images_dir. "
                             "Useful for comparing train/test embedding quality.")
    parser.add_argument("--metrics_sample_size", type=int, default=10000,
                        help="Maximum number of samples used when computing metrics "
                             "(set <=0 to disable subsampling).")

    # AutoGluon evaluation and prediction
    parser.add_argument("--evaluate_classification", action="store_true",
                        help="Evaluate classification performance using AutoGluon's evaluate() method. "
                             "Requires a trained predictor (via --train_data or --load_AutogluonModel). "
                             "Results saved to logs/evaluations.{train|test}.txt")
    parser.add_argument("--predict_classification", action="store_true",
                        help="Predict classifications and probabilities using AutoGluon's predict() and predict_proba(). "
                             "Requires a trained predictor. Results saved to predictions/predictions.{train|test}.csv")

    # Visualization
    parser.add_argument("--visualize_train_data", action="store_true",
                        help="If given, run UMAP on training embeddings (needs --train_data).")
    parser.add_argument("--visualize_test_data", type=str, default=None,
                        help="Optional CSV (image,label) for inference set visualization.")
    parser.add_argument("--visualize_class_number", type=int, default=None,
                        help="Limit number of label categories shown in UMAP plots. "
                             "If unset, show all labels.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--umap_metric", type=str, default="euclidean", help="UMAP metric.")
    parser.add_argument("--umap_seed", type=int, default=42, help="UMAP random seed.")

    return parser.parse_args()


class TeeLogger:
    """Redirect stdout to both console and file (skip progress bars)."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = log_path.open('w', encoding='utf-8')
        self.skip_patterns = [
            '\r',
            '|<E2><96><88>', '|<E2><96><91>',
            '\x1b[',
            '[A',
        ]
        self.last_line_was_progress = False
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        is_progress = any(pattern in message for pattern in self.skip_patterns)
        if not is_progress:
            if self.last_line_was_progress and message.strip():
                self.log.write('\n')
            self.log.write(message)
            self.log.flush()
            self.last_line_was_progress = False
        else:
            self.last_line_was_progress = True

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def ensure_timm_model_exists(model_name: str):
    available = set(timm.list_models(pretrained=True)) | set(timm.list_models(pretrained=False))
    if model_name not in available:
        raise SystemExit(f"[Error] {model_name} is not registered in timm.")


def detect_device(preferred: str = None) -> torch.device:
    if preferred:
        dev = torch.device(preferred)
        if preferred == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available.")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dir(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def collect_images_from_dir(image_dir: str) -> pd.DataFrame:
    image_dir = _resolve_dir(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"{image_dir} does not exist.")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted(p.resolve() for p in image_dir.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"No supported images found under {image_dir}.")
    stems = [f.stem for f in files]
    dup = [name for name, cnt in Counter(stems).items() if cnt > 1]
    if dup:
        raise SystemExit(f"Duplicated filenames (without suffix): {dup[:10]} ... Please rename them.")
    return pd.DataFrame({"image": [str(p) for p in files], "stem": stems, "filename": [p.name for p in files]})


def load_csv_with_dir(csv_path: str, images_dir: str, need_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"image", "label"} if need_label else {"image"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"{csv_path} must contain columns {required_cols}")
    base_dir = _resolve_dir(images_dir)
    full_paths = []
    filenames = []
    for rel in df["image"]:
        rel_path = Path(rel)
        if not rel_path.is_absolute():
            rel_path = base_dir / rel_path
        rel_path = rel_path.expanduser().resolve()
        if not rel_path.exists():
            raise FileNotFoundError(f"Image referenced in {csv_path} not found: {rel_path}")
        full_paths.append(str(rel_path))
        filenames.append(rel_path.name)
    df = df.copy()
    df["image"] = full_paths
    df["stem"] = df["image"].apply(lambda p: Path(p).stem)
    df["filename"] = filenames
    return df


def compute_focal_alpha(train_df: pd.DataFrame):
    class_counts = train_df["label"].value_counts().sort_index()
    weights = 1.0 / (class_counts / class_counts.sum())
    weights = weights / weights.sum()
    return weights.tolist()


def save_processed_train_csv(train_df: pd.DataFrame, out_dir: Path) -> Path:
    copy_df = train_df.copy()
    copy_df["image"] = copy_df["image"].apply(lambda p: str(Path(p).expanduser().resolve()))
    save_path = out_dir / "train.processed.csv"
    copy_df[["image", "label"]].to_csv(save_path, index=False)
    return save_path


def save_processed_test_csv(test_df: pd.DataFrame, out_dir: Path, label_csv: str = None) -> Path:
    """Save processed test/inference CSV with full image paths and optional labels."""
    copy_df = test_df.copy()
    copy_df["image"] = copy_df["image"].apply(lambda p: str(Path(p).expanduser().resolve()))
    
    save_path = out_dir / "test.processed.csv"
    
    if "label" in copy_df.columns:
        copy_df[["image", "label"]].to_csv(save_path, index=False)
    else:
        copy_df[["image"]].to_csv(save_path, index=False)
    
    return save_path


def save_embeddings(image_names, array, out_path: Path):
    cols = [f"E{i:04d}" for i in range(array.shape[1])]
    df = pd.DataFrame(array, columns=cols)
    df.insert(0, "image", image_names)
    df.to_csv(out_path, index=False)
    return df


def run_umap(features, labels, output_path: Path, title: str, args):
    if len(labels) < 2:
        print(f"[Warning] Need >=2 samples to draw UMAP ({output_path}); skipped.")
        return
    feats_norm = Normalizer(norm="l2").fit_transform(features)
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.umap_seed,
        n_components=2,
    )
    coords = reducer.fit_transform(feats_norm)
    df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
    df["label"] = pd.Series(labels).astype(str)
    if args.visualize_class_number is not None and args.visualize_class_number > 0:
        label_counts = df["label"].value_counts()
        keep_labels = label_counts.head(args.visualize_class_number).index.tolist()
        dropped = len(label_counts) - len(keep_labels)
        df = df[df["label"].isin(keep_labels)]
        if df.empty:
            print("[Warning] After applying visualize_class_number filter, no samples remain; skip UMAP.")
            return
        print(f"[Info] Showing top {len(keep_labels)} classes "
              f"out of {len(label_counts)} (dropped {dropped}).")

    unique_labels = df["label"].unique().tolist()
    if len(unique_labels) <= 10:
        palette = sns.color_palette("tab10", len(unique_labels))
    elif len(unique_labels) <= 20:
        palette = sns.color_palette("tab20", len(unique_labels))
    else:
        palette = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    max_labels_per_col = 30
    n_cols_legend = max(1, math.ceil(len(unique_labels) / max_labels_per_col))
    fig_width = 10 + 1.2 * n_cols_legend
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    for lbl in unique_labels:
        sub = df[df["label"] == lbl]
        ax.scatter(sub["UMAP1"], sub["UMAP2"],
                   s=30, alpha=0.85, label=str(lbl),
                   color=color_map[lbl], edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title="Label",
        frameon=True,
        fontsize="small",
        title_fontsize="small",
        ncol=n_cols_legend,
        columnspacing=0.8,
        handlelength=1.0,
        borderaxespad=0.2,
    )
    legend._legend_box.align = "left"
    legend_space = min(0.32, 0.08 * n_cols_legend)
    right_rect = 1 - legend_space
    fig.tight_layout(rect=[0, 0, right_rect, 1])
    fig.savefig(output_path, dpi=300, format="pdf")
    plt.close(fig)

# --------------------------- AutoGluon helpers --------------------------- #
def _build_hyperparameters(train_df, args):
    hyper = {
        "model.timm_image.checkpoint_name": args.base_model,
        "optim.max_epochs": args.max_epochs,
    }
    if not args.disable_augment:
        hyper["model.timm_image.train_transforms"] = [
            "resize_shorter_side",
            "center_crop",
            "random_horizontal_flip",
            "random_vertical_flip",
            "color_jitter",
            "trivial_augment",
        ]
    if args.Focal_Loss:
        weights = compute_focal_alpha(train_df)
        hyper.update({
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": weights,
            "optim.focal_loss.gamma": args.focal_loss_gamma,
            "optim.focal_loss.reduction": "sum",
        })

    accelerator_map = {"cuda": "gpu", "cpu": "cpu", "mps": "auto"}
    if args.device == "cpu":
        hyper["env.accelerator"] = "cpu"
    return hyper


def train_autogluon(train_df: pd.DataFrame, args, out_dir: Path):
    if MultiModalPredictor is None:
        raise SystemExit("AutoGluon is not installed.")
    predictor_dir = out_dir / "AutogluonModels" / args.base_model
    predictor_dir.parent.mkdir(parents=True, exist_ok=True)

    def build_predictor():
        return MultiModalPredictor(label="label", path=str(predictor_dir))

    hyperparameters = _build_hyperparameters(train_df, args)
    time_limit = int(max(args.time_limit, 0.001) * 3600)

    predictor = build_predictor()
    print("[Info] Start AutoGluon training ...")
    try:
        predictor.fit(
            train_data=train_df[["image", "label"]],
            time_limit=time_limit,
            hyperparameters=hyperparameters,
        )
    except Exception as e:
        if "NVMLError_LibraryNotFound" in repr(e):
            print("[Warning] NVML is unavailable; retrying with NVML logging disabled.")
            os.environ["AG_AUTOMM_DISABLE_NVML"] = "1"
            predictor = build_predictor()
            predictor.fit(
                train_data=train_df[["image", "label"]],
                time_limit=time_limit,
                hyperparameters=hyperparameters,
            )
        else:
            raise

    print(f"[Info] AutoGluon model saved under {predictor_dir}")
    return predictor


def load_predictor(path: str):
    if MultiModalPredictor is None:
        raise SystemExit("AutoGluon not installed; cannot load predictor.")
    print(f"[Info] Loading AutoGluon predictor from {path}")
    return MultiModalPredictor.load(path)


def extract_with_predictor(predictor, df: pd.DataFrame):
    cols = ["image"]
    if "label" in df.columns:
        cols.append("label")
    return predictor.extract_embedding(df[cols])


def evaluate_autogluon_classification(predictor, data_df: pd.DataFrame, out_dir: Path, split_name: str):
    """
    Evaluate classification performance using AutoGluon's evaluate() method.
    
    Args:
        predictor: AutoGluon MultiModalPredictor
        data_df: DataFrame with 'image' and 'label' columns
        out_dir: Output directory
        split_name: 'train' or 'test'
    """
    if MultiModalPredictor is None:
        print("[Warning] AutoGluon not installed; skipping evaluation.")
        return
    
    if "label" not in data_df.columns:
        print(f"[Warning] No labels in {split_name} data; skipping evaluation.")
        return
    
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Evaluation] Evaluating {split_name} set classification performance...")
    
    metrics_list = [
        'accuracy', 
        'precision_macro', 
        'precision_micro', 
        'recall_macro', 
        'recall_micro', 
        'f1_macro', 
        'f1_micro', 
        'mcc', 
        'roc_auc_ovo'
    ]
    
    try:
        evaluations = predictor.evaluate(
            data_df[["image", "label"]], 
            metrics=metrics_list
        )
        
        eval_file = logs_dir / f"evaluations.{split_name}.txt"
        with open(eval_file, 'w') as f:
            f.write(f"Classification Evaluation Results ({split_name} set)\n")
            f.write("=" * 60 + "\n\n")
            for key, value in evaluations.items():
                f.write(f"{key}: {value}\n")
        
        print(f"[Info] Evaluation results saved to {eval_file}")
        print(f"\n{split_name.capitalize()} Set Evaluation Metrics:")
        for key, value in evaluations.items():
            print(f"  {key:25s}: {value:.4f}")
    
    except Exception as e:
        print(f"[Warning] Failed to evaluate {split_name} set: {e}")
        import traceback
        traceback.print_exc()


def predict_autogluon_classification(predictor, data_df: pd.DataFrame, out_dir: Path, split_name: str):
    """
    Predict classifications and probabilities using AutoGluon.
    
    Args:
        predictor: AutoGluon MultiModalPredictor
        data_df: DataFrame with 'image' column (and optionally 'label')
        out_dir: Output directory
        split_name: 'train' or 'test'
    """
    if MultiModalPredictor is None:
        print("[Warning] AutoGluon not installed; skipping prediction.")
        return
    
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Prediction] Predicting {split_name} set classifications...")
    
    try:
        input_data = data_df[["image"]].copy()
        predictions = predictor.predict(input_data)
        predictions_proba = predictor.predict_proba(input_data)
        results_dict = {}
        if "filename" in data_df.columns:
            results_dict['image'] = data_df['filename'].values
        else:
            results_dict['image'] = data_df['image'].apply(lambda p: Path(p).name).values
        results_dict['stem'] = data_df['stem'].values
        if 'label' in data_df.columns:
            results_dict['label'] = data_df['label'].values
        results_dict['prediction'] = predictions.values
        for col in predictions_proba.columns:
            results_dict[f'proba_{col}'] = predictions_proba[col].values
        results_df = pd.DataFrame(results_dict)
        results_df.index = results_df['stem']
        results_df.index.name = 'sample'
        
        pred_file = pred_dir / f"predictions.{split_name}.csv"
        results_df.to_csv(pred_file, index=False)
        print(f"[Info] Predictions saved to {pred_file}")
    
    except Exception as e:
        print(f"[Warning] Failed to predict {split_name} set: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------- Metrics functions --------------------------- #

def compute_recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute Recall@K for retrieval tasks.
    
    Args:
        embeddings: [N, D] normalized embeddings
        labels: [N] ground truth labels (can be list or array)
        k_values: list of K values
    
    Returns:
        Dictionary with Recall@K scores
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize
    except ImportError:
        return {f"Recall@{k}": 0.0 for k in k_values}
    
    # Convert labels to numpy array if it's a list
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    if len(embeddings) == 0 or len(labels) == 0:
        return {f"Recall@{k}": 0.0 for k in k_values}
    
    embeddings = normalize(embeddings, norm='l2')
    similarity_matrix = cosine_similarity(embeddings)
    
    # Set diagonal to -inf to exclude self-similarity
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    recalls = {}
    for k in k_values:
        k_actual = min(k, len(labels) - 1)
        if k_actual < 1:
            recalls[f"Recall@{k}"] = 0.0
            continue
        
        correct = 0
        for i in range(len(labels)):
            # Get top-k most similar samples
            top_k_indices = np.argsort(similarity_matrix[i])[-k_actual:]
            top_k_labels = labels[top_k_indices]
            
            # Check if any of top-k has the same label
            if labels[i] in top_k_labels:
                correct += 1
        
        recalls[f"Recall@{k}"] = correct / len(labels)
    
    return recalls


def compute_knn_accuracy(embeddings: np.ndarray, labels: np.ndarray, k_values: List[int] = [1, 5, 20]) -> Dict[str, float]:
    """
    Compute k-NN classification accuracy.
    
    Args:
        embeddings: [N, D] normalized embeddings
        labels: [N] ground truth labels
        k_values: list of K values for k-NN
    
    Returns:
        Dictionary with kNN accuracy for each k
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import normalize
    except ImportError:
        return {f"kNN_Acc_k{k}": 0.0 for k in k_values}
    
    embeddings = normalize(embeddings, norm='l2')
    
    accuracies = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        # Use 5-fold cross-validation
        scores = cross_val_score(knn, embeddings, labels, cv=min(5, len(np.unique(labels))))
        accuracies[f"kNN_Acc_k{k}"] = scores.mean()
    
    return accuracies


def compute_mean_average_precision(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute mean Average Precision (mAP) for retrieval.
    
    Args:
        embeddings: [N, D] normalized embeddings
        labels: [N] ground truth labels (can be list or array)
    
    Returns:
        mAP score
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    # Convert labels to numpy array if it's a list
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    embeddings = normalize(embeddings, norm='l2')
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    aps = []
    for i in range(len(labels)):
        # Get sorted indices by similarity
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        sorted_labels = labels[sorted_indices]
        
        # Compute Average Precision for this query
        relevant = (sorted_labels == labels[i])
        if relevant.sum() == 0:
            continue
        
        precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0

def compute_linear_probing_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Train a linear classifier on frozen embeddings (linear probing).
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] ground truth labels
    
    Returns:
        Linear probing accuracy
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    embeddings = normalize(embeddings, norm='l2')
    
    # Use L2-regularized logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, embeddings, labels, cv=min(5, len(np.unique(labels))))
    
    return scores.mean()


def compute_silhouette_score(embeddings: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
    """
    Compute Silhouette Score (can work without labels using clustering).
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] ground truth labels (optional, will use KMeans if None)
    
    Returns:
        Silhouette score
    """
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    if len(embeddings) < 2:
        return 0.0
    
    embeddings = normalize(embeddings, norm='l2')
    
    # Convert labels to numpy array if it's a list
    if labels is not None and not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # If no labels provided, use KMeans clustering
    if labels is None:
        n_clusters = max(2, int(np.sqrt(len(embeddings) / 2)))
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        except Exception as e:
            print(f"[Warning] KMeans clustering failed: {e}")
            return 0.0
    
    # Check if we have valid labels
    if labels is None or len(labels) == 0:
        return 0.0
    
    # Need at least 2 unique labels for silhouette score
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    try:
        return silhouette_score(embeddings, labels, metric='cosine')
    except Exception as e:
        print(f"[Warning] Silhouette score computation failed: {e}")
        return 0.0


def compute_purity(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute clustering purity using K-Means.
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] ground truth labels (can be list or array)
    
    Returns:
        Purity score
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    # Convert labels to numpy array if it's a list
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    embeddings = normalize(embeddings, norm='l2')
    
    # Get unique labels and create label mapping
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    label_indices = np.array([label_to_idx[label] for label in labels])
    
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute purity
    contingency_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(labels)):
        contingency_matrix[cluster_labels[i], label_indices[i]] += 1
    
    purity = np.sum(np.max(contingency_matrix, axis=1)) / len(labels)
    return purity


def maybe_subsample_for_metrics(embeddings, labels, max_samples, seed=42):
    embeddings = np.asarray(embeddings)
    n = embeddings.shape[0]
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_samples, replace=False))
    subset_embeddings = embeddings[idx]
    subset_labels = None
    if labels is not None:
        labels_array = np.array(labels)
        subset_labels = labels_array[idx]
        if isinstance(labels, list):
            subset_labels = subset_labels.tolist()
    print(f"[Info] Metrics subsample: {len(idx)}/{n} samples")
    return subset_embeddings, subset_labels


def compute_all_metrics(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray],
    mode: str = "pretrain",
    compute_linear_probing: bool = False
) -> Dict[str, float]:
    """
    Compute all metrics at once.
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] ground truth labels (None for unlabeled data, can be list or array)
        mode: "pretrain" or "arcface"
        compute_linear_probing: whether to compute expensive linear probing
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Convert labels to numpy array if it's a list
    if labels is not None and not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    if len(embeddings) == 0:
        print("[Warning] Empty embeddings, skipping metrics computation")
        return {k: "" for k in ["NMI", "ARI", "Recall@1", "Recall@5", "Recall@10",
                                "kNN_Acc_k5", "kNN_Acc_k20", "Linear_Probing_Acc", 
                                "mAP", "Purity", "Silhouette_Score"]}
    
    # Metrics that don't need labels
    try:
        metrics["Silhouette_Score"] = compute_silhouette_score(embeddings, labels)
    except Exception as e:
        print(f"[Warning] Failed to compute Silhouette Score: {e}")
        metrics["Silhouette_Score"] = ""
    
    # Metrics that need labels
    if labels is not None and len(labels) > 0:
        try:
            from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
            
            # NMI and ARI
            embeddings_norm = normalize(embeddings, norm='l2')
            
            # Get unique labels and create label mapping for clustering
            unique_labels = np.unique(labels)
            n_classes = len(unique_labels)
            
            # Create label to index mapping
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            label_indices = np.array([label_to_idx[label] for label in labels])
            
            kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(embeddings_norm)
            
            metrics["NMI"] = normalized_mutual_info_score(label_indices, pred_labels)
            metrics["ARI"] = adjusted_rand_score(label_indices, pred_labels)
            
            # Recall@K
            if len(embeddings) >= 2:
                recall_metrics = compute_recall_at_k(embeddings, labels, k_values=[1, 5, 10])
                metrics.update(recall_metrics)
            else:
                print("[Warning] Not enough samples for Recall@K")
                metrics.update({"Recall@1": "", "Recall@5": "", "Recall@10": ""})
            
            # kNN Accuracy
            if len(embeddings) >= n_classes * 2:
                knn_metrics = compute_knn_accuracy(embeddings, label_indices, k_values=[1, 5, 20])
                metrics.update(knn_metrics)
            else:
                print("[Warning] Not enough samples for kNN")
                metrics.update({"kNN_Acc_k1": "", "kNN_Acc_k5": "", "kNN_Acc_k20": ""})
            
            # Linear Probing (expensive, compute less frequently)
            if compute_linear_probing and len(embeddings) >= n_classes * 2:
                metrics["Linear_Probing_Acc"] = compute_linear_probing_accuracy(embeddings, label_indices)
            else:
                metrics["Linear_Probing_Acc"] = ""
            
            # mAP
            if len(embeddings) >= 2:
                metrics["mAP"] = compute_mean_average_precision(embeddings, labels)
            else:
                metrics["mAP"] = ""
            
            # Purity
            if len(embeddings) >= n_classes:
                metrics["Purity"] = compute_purity(embeddings, labels)
            else:
                metrics["Purity"] = ""
            
            # ArcFace specific metrics
            if mode == "arcface":
                arcface_metrics = compute_arcface_specific_metrics(embeddings, label_indices)
                metrics.update(arcface_metrics)
        
        except Exception as e:
            print(f"[Warning] Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
            # Fill with empty values
            for key in ["NMI", "ARI", "Recall@1", "Recall@5", "Recall@10",
                       "kNN_Acc_k5", "kNN_Acc_k20", "Linear_Probing_Acc", "mAP", "Purity"]:
                if key not in metrics:
                    metrics[key] = ""
    
    return metrics


# --------------------------- Main --------------------------- #
def main():
    args = parse_args()

    ensure_timm_model_exists(args.base_model)
    out_dir = _resolve_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log
    logs_dir = Path(args.out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "log.txt"
    
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    sys.stderr = tee_logger  # Also capture errors
    
    print(f"[Info] Logging to {log_file}")
    print(f"[Info] Command: {' '.join(sys.argv)}")
    #print(f"[Info] Arguments: {vars(args)}")

    if args.visualize_test_data:
        print(f"[Info] Using test CSV: {args.visualize_test_data}")
        inference_df = load_csv_with_dir(
            csv_path=args.visualize_test_data,
            images_dir=args.input_images_dir,
            need_label=False
        )
        print(f"[Info] Processing {len(inference_df)} images from CSV")
    else:
        print(f"[Info] No test CSV provided, processing all images in {args.input_images_dir}")
        inference_df = collect_images_from_dir(args.input_images_dir)
        print(f"[Info] {len(inference_df)} images found in directory to embed.")
    
    inference_filenames = inference_df["filename"].tolist() if "filename" in inference_df.columns else inference_df["image"].apply(lambda p: Path(p).name).tolist()
    inference_stems = inference_df["stem"].tolist()
    test_processed_csv = save_processed_test_csv(inference_df, out_dir)
    print(f"[Info] Processed test CSV saved to {test_processed_csv}")

    predictor = None
    train_df = None
    if args.load_AutogluonModel:
        predictor = load_predictor(args.load_AutogluonModel)
    elif args.train_data:
        if not args.train_images_dir:
            raise SystemExit("--train_images_dir must be provided with --train_data.")
        train_df = load_csv_with_dir(args.train_data, args.train_images_dir, need_label=True)
        processed_csv = save_processed_train_csv(train_df, out_dir)
        print(f"[Info] Processed training CSV saved to {processed_csv}")
        predictor = train_autogluon(train_df, args, out_dir)
    else:
        print("[Info] No training data -> use raw timm backbone.")

    timm_embedder = None
    if predictor is None:
        device = detect_device(args.device)
        print(f"[Info] Using device: {device}")
        timm_embedder = TimmEmbedder(args.base_model, device)

    #Load train_df if needed for evaluation/prediction
    if (args.evaluate_classification or args.predict_classification or args.visualize_train_data or args.save_train_embeddings):
        if train_df is None and args.train_data is not None:
            if args.train_images_dir is None:
                print("[Warning] --train_data provided but --train_images_dir missing. "
                      "Cannot load training data for evaluation/prediction.")
            else:
                print("[Info] Loading training data for evaluation/prediction...")
                train_df = load_csv_with_dir(args.train_data, args.train_images_dir, need_label=True)
                # Save processed CSV if not already saved
                processed_csv = save_processed_train_csv(train_df, out_dir)
                print(f"[Info] Processed training CSV saved to {processed_csv}")

    # AutoGluon Evaluation
    if args.evaluate_classification:
        if predictor is None:
            print("[Warning] --evaluate_classification requires a trained predictor. Skipping evaluation.")
        else:
            # Evaluate on training data
            if train_df is not None:
                evaluate_autogluon_classification(predictor, train_df, out_dir, "train")
            
            # Evaluate on test data (if labels available)
            if "label" in inference_df.columns:
                evaluate_autogluon_classification(predictor, inference_df, out_dir, "test")
            else:
                print("[Info] Test data has no labels; skipping test evaluation.")

    # AutoGluon Prediction
    if args.predict_classification:
        if predictor is None:
            print("[Warning] --predict_classification requires a trained predictor. Skipping prediction.")
        else:
            # Predict on training data
            if train_df is not None:
                predict_autogluon_classification(predictor, train_df, out_dir, "train")
            
            # Predict on test data
            predict_autogluon_classification(predictor, inference_df, out_dir, "test")

    # Extract training embeddings if needed
    train_feats = None
    if args.visualize_train_data or args.save_train_embeddings:
        if train_df is None:
            if args.train_data is not None:
                print("[Warning] Cannot visualize/save train embeddings: train_df not loaded.")
        else:
            print("[Info] Extracting training embeddings...")
            if predictor is not None:
                train_feats = extract_with_predictor(predictor, train_df)
            else:
                train_feats = timm_embedder.extract(train_df, args.batch_size, args.num_workers)
            
            # Save training embeddings if requested
            if args.save_train_embeddings:
                train_embed_path = out_dir / "embeddings.train.csv"
                train_filenames = train_df["filename"].tolist() if "filename" in train_df.columns else train_df["image"].apply(lambda p: Path(p).name).tolist()
                save_embeddings(train_filenames, train_feats, train_embed_path)
                print(f"[Info] Saved training embeddings to {train_embed_path}")

    # Extract test/inference embeddings
    if predictor is not None:
        embeddings_array = extract_with_predictor(predictor, inference_df)
    else:
        embeddings_array = timm_embedder.extract(inference_df, args.batch_size, args.num_workers)
    embeddings_path = out_dir / "embeddings.test.csv"
    save_embeddings(inference_filenames, embeddings_array, embeddings_path)
    print(f"[Info] Saved test embeddings to {embeddings_path}")

    # Visualization for training data
    if args.visualize_train_data:
        if train_feats is None:
            print("[Warning] Train embeddings not extracted; skipping train UMAP.")
        else:
            run_umap(train_feats, train_df["label"].tolist(),
                     out_dir / "umap.train.pdf",
                     f"Training data UMAP ({args.base_model})", args)

    # Visualization for test data
    if args.visualize_test_data:
        label_df = pd.read_csv(args.visualize_test_data)
        if not {"image", "label"}.issubset(label_df.columns):
            raise SystemExit(f"{args.visualize_test_data} must contain image,label.")
        label_df["stem"] = label_df["image"].apply(lambda x: Path(x).stem)
        label_map = dict(zip(label_df["stem"], label_df["label"]))
        matched = [(idx, label_map[stem]) for idx, stem in enumerate(inference_stems) if stem in label_map]
        if not matched:
            print("[Warning] visualize_test_data has no overlap; skipping UMAP.")
        else:
            idxs, labels = zip(*matched)
            run_umap(
                embeddings_array[list(idxs)],
                labels,
                out_dir / "umap.test.pdf",
                f"Test data UMAP ({args.base_model})",
                args,
            )

    # metrics calculation
    if args.visualize_train_data or args.visualize_test_data:
        print("\n" + "="*60)
        print("Computing Comprehensive Metrics...")
        print("="*60)
        metrics_results = {}
        
        # metrics for train data
        if args.visualize_train_data:
            if train_feats is None or train_df is None:
                print("[Warning] Train embeddings or train_df not available; skipping train metrics.")
            else:
                print("\n[Metrics] Computing for training set...")
                train_labels = train_df["label"].tolist() if "label" in train_df.columns else None
                feats_eval, labels_eval = maybe_subsample_for_metrics(
                    train_feats, train_labels, args.metrics_sample_size
                )
                metrics = compute_all_metrics(
                    embeddings=feats_eval,
                    labels=labels_eval,
                    mode="autogluon",
                    compute_linear_probing=True
                )
                metrics_results["train"] = metrics
                
                print("Training Set Metrics:")
                for k, v in metrics.items():
                    if v != "":
                        print(f"  {k:25s}: {v:.4f}")
        
        # metrics for test data
        if args.visualize_test_data:
            print("\n[Metrics] Computing for test/inference set...")
            test_labels = None
            if "label" in inference_df.columns:
                test_labels = inference_df["label"].tolist()
                print(f"[Info] Using {len(test_labels)} labels from test CSV")
            else:
                print("[Info] No labels in test CSV, computing unsupervised metrics only")
            
            feats_eval, labels_eval = maybe_subsample_for_metrics(
                embeddings_array, test_labels, args.metrics_sample_size
            )
            metrics = compute_all_metrics(
                embeddings=feats_eval,
                labels=labels_eval,
                mode="autogluon",
                compute_linear_probing=True
            )
            metrics_results["test"] = metrics
            
            print("Test Set Metrics:")
            for k, v in metrics.items():
                if v != "":
                    print(f"  {k:25s}: {v:.4f}")
        
        # save metrics to CSV
        logs_dir = out_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = logs_dir / "metrics.csv"

        with metrics_csv.open("w", newline="") as f:
            fieldnames = ["split"] + list(next(iter(metrics_results.values())).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for split, metrics in metrics_results.items():
                row = {"split": split}
                row.update(metrics)
                writer.writerow(row)
        
        print(f"\n[Info] Saved comprehensive metrics to {metrics_csv}")

    print("[Info] Done.")


if __name__ == "__main__":
    main()

