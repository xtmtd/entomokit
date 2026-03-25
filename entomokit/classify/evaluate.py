"""entomokit classify evaluate — evaluate classification performance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "evaluate",
        help="Evaluate classification performance (AutoGluon or ONNX).",
        description=with_examples(
            "Evaluate classification performance (AutoGluon or ONNX).",
            [
                "entomokit classify evaluate --test-csv test.csv --images-dir ./images --model-dir ./model --out-dir ./eval",
                "entomokit classify evaluate --test-csv test.csv --images-dir ./images --onnx-model model.onnx --out-dir ./eval",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument(
        "--test-csv",
        required=True,
        help="CSV with 'image' and 'label' columns for evaluation.",
    )
    p.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing evaluation images.",
    )
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-dir", help="AutoGluon predictor directory for evaluation."
    )
    model_group.add_argument(
        "--onnx-model",
        help="ONNX model file path (requires onnxruntime).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write evaluation logs and metrics.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation inference.",
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
        help="CPU threads for PyTorch/ONNX runtime (0 = auto).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device for evaluation.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.classification.utils import select_device, ag_device_map
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    device = select_device(args.device)

    if args.model_dir:
        from src.classification.evaluator import evaluate

        metrics = evaluate(
            test_csv=Path(args.test_csv),
            images_dir=Path(args.images_dir),
            model_dir=Path(args.model_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_threads=args.num_threads,
            device=ag_device_map(device),
        )
    else:
        from src.classification.evaluator import evaluate_onnx

        metrics = evaluate_onnx(
            test_csv=Path(args.test_csv),
            images_dir=Path(args.images_dir),
            onnx_path=Path(args.onnx_model),
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )

    eval_csv = out_dir / "evaluations.csv"
    metrics_df = pd.DataFrame(
        [
            {"metric": metric_name, "value": metric_value}
            for metric_name, metric_value in metrics.items()
        ]
    )
    metrics_df.to_csv(eval_csv, index=False)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    print(f"\nResults saved to: {eval_csv}")
