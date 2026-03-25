"""entomokit classify embed — extract embeddings and compute quality metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "embed",
        help="Extract embeddings, compute quality metrics, and optionally visualize with UMAP.",
        description=with_examples(
            "Extract embeddings, compute quality metrics, and optionally visualize with UMAP.",
            [
                "entomokit classify embed --images-dir ./images --out-dir ./embed",
                "entomokit classify embed --images-dir ./images --label-csv labels.csv --visualize --out-dir ./embed",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing images to embed.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write embeddings and optional visualizations.",
    )
    p.add_argument(
        "--base-model",
        default="convnextv2_femto",
        help="timm backbone (used if --model-dir not provided).",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="AutoGluon predictor for fine-tuned backbone extraction.",
    )
    p.add_argument(
        "--label-csv",
        default=None,
        help="CSV(image,label) for supervised metrics and UMAP coloring.",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Generate UMAP plot (requires --label-csv).",
    )
    p.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP neighbor count for manifold construction.",
    )
    p.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum distance between embedded points.",
    )
    p.add_argument(
        "--umap-metric",
        default="euclidean",
        help="Distance metric used by UMAP.",
    )
    p.add_argument(
        "--umap-seed",
        type=int,
        default=42,
        help="Random seed for reproducible UMAP layouts.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used during embedding extraction.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes.",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="CPU threads for PyTorch operations (0 = auto).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device for embedding extraction.",
    )
    p.add_argument(
        "--metrics-sample-size",
        type=int,
        default=10000,
        help="Maximum sample size for embedding quality metrics.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import warnings

    import numpy as np
    import pandas as pd
    from pathlib import Path
    from src.classification.utils import select_device, set_num_threads, load_image_csv
    from src.common.cli import save_log

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

    if args.visualize and not args.label_csv:
        print(
            "Error: --visualize requires --label-csv to be provided.", file=sys.stderr
        )
        sys.exit(1)

    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_log(logs_dir, args)

    device = select_device(args.device)
    set_num_threads(args.num_threads)

    # Extract embeddings
    if args.model_dir:
        from src.classification.embedder import extract_embeddings_ag

        embed_df = extract_embeddings_ag(
            images_dir=Path(args.images_dir),
            model_dir=Path(args.model_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
    else:
        from src.classification.embedder import extract_embeddings_timm

        embed_df = extract_embeddings_timm(
            images_dir=Path(args.images_dir),
            base_model=args.base_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

    embed_df.to_csv(out_dir / "embeddings.csv", index=False)
    print(f"Embeddings saved: {len(embed_df)} images, {embed_df.shape[1] - 1} dims")

    # Supervised metrics + UMAP
    if args.label_csv:
        label_df = load_image_csv(Path(args.label_csv), require_label=True)
        merged = embed_df.merge(label_df[["image", "label"]], on="image", how="inner")
        feat_cols = [c for c in merged.columns if c.startswith("feat_")]
        embeddings = merged[feat_cols].values
        labels = merged["label"].values

        from src.classification.embedder import compute_embedding_metrics

        metrics = compute_embedding_metrics(
            embeddings, labels, sample_size=args.metrics_sample_size
        )
        metrics_df = pd.DataFrame([metrics])
        metrics_path = out_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"Metrics saved to: {metrics_path}")

        if args.visualize:
            from src.classification.embedder import visualize_umap

            umap_path = out_dir / "umap.pdf"
            visualize_umap(
                embeddings=embeddings,
                labels=labels,
                out_path=umap_path,
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                seed=args.umap_seed,
            )
            print(f"UMAP saved to: {umap_path}")
