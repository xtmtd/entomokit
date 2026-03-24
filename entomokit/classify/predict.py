"""entomokit classify predict — run image classification inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "predict",
        help="Run classification inference (AutoGluon or ONNX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv", help="CSV with 'image' column.")
    input_group.add_argument("--images-dir", help="Directory to scan for images.")
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-dir", help="AutoGluon predictor directory.")
    model_group.add_argument("--onnx-model", help="ONNX model file path.")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write prediction outputs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference.",
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
        help="Compute device for inference.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import pandas as pd
    from pathlib import Path
    from src.classification.utils import select_device, ag_device_map, load_image_csv
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    # Build input DataFrame
    if args.input_csv:
        df = load_image_csv(Path(args.input_csv))
        images_dir = None
    else:
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        imgs = [
            p.name
            for p in Path(args.images_dir).iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ]
        df = pd.DataFrame({"image": sorted(imgs)})
        images_dir = Path(args.images_dir)

    device = select_device(args.device)

    if args.model_dir:
        from src.classification.predictor import predict_ag

        result = predict_ag(
            input_df=df,
            images_dir=images_dir,
            model_dir=Path(args.model_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_threads=args.num_threads,
            device=ag_device_map(device),
        )
    else:
        from src.classification.predictor import predict_onnx

        result = predict_onnx(
            input_df=df,
            images_dir=images_dir,
            onnx_path=Path(args.onnx_model),
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )

    out_csv = pred_dir / "predictions.csv"
    result.to_csv(out_csv, index=False)
    print(f"Predictions saved to: {out_csv}")
