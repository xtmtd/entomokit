"""entomokit classify predict — run image classification inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class PredictInputError(ValueError):
    def __init__(self, message: str, missing_images: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_images = missing_images or []


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "predict",
        help="Run classification inference (AutoGluon or ONNX).",
        description=with_examples(
            "Run classification inference (AutoGluon or ONNX).",
            [
                "entomokit classify predict --images-dir ./images --model-dir ./model --out-dir ./pred",
                "entomokit classify predict --input-csv data.csv --images-dir ./images --onnx-model model.onnx --out-dir ./pred",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument("--input-csv", help="CSV with 'image' column.")
    p.add_argument("--images-dir", help="Directory to scan for images.")
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-dir", help="AutoGluon predictor directory.")
    model_group.add_argument(
        "--onnx-model",
        help="ONNX model file path (requires onnxruntime).",
    )
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


def _has_image_files(images_dir: Path) -> bool:
    return any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in images_dir.iterdir()
    )


def _resolve_predict_inputs(
    input_csv: Path | None,
    images_dir: Path | None,
) -> tuple["pd.DataFrame", Path | None]:
    import pandas as pd
    from src.classification.utils import load_image_csv

    if images_dir is not None and not images_dir.is_dir():
        raise ValueError(f"--images-dir is not a directory: {images_dir}")

    if input_csv is not None:
        df = load_image_csv(input_csv)
        csv_paths = [Path(str(value)) for value in df["image"].tolist()]
        csv_paths_are_readable = all(path.is_file() for path in csv_paths)

        if csv_paths_are_readable:
            if images_dir is not None and _has_image_files(images_dir):
                raise ValueError(
                    "CSV already contains readable image paths; do not pass "
                    "--images-dir at the same time."
                )
            return df, None

        if images_dir is None:
            raise ValueError(
                "CSV image values are not readable paths. Provide --images-dir "
                "to resolve image names."
            )

        missing = [
            str(value)
            for value in df["image"].tolist()
            if not (images_dir / str(value)).is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise PredictInputError(
                f"Some CSV images were not found under --images-dir: {preview}",
                missing_images=missing,
            )
        return df, images_dir

    if images_dir is None:
        raise ValueError("At least one of --input-csv or --images-dir is required.")

    imgs = [
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return pd.DataFrame({"image": sorted(imgs)}), images_dir


def run(args: argparse.Namespace) -> None:
    from src.classification.utils import select_device, ag_device_map
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    input_csv = Path(args.input_csv) if args.input_csv else None
    images_dir = Path(args.images_dir) if args.images_dir else None
    try:
        df, images_dir = _resolve_predict_inputs(
            input_csv=input_csv, images_dir=images_dir
        )
    except PredictInputError as exc:
        logs_dir = out_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        missing_log_path = logs_dir / "missing_images.txt"
        missing_log_path.write_text(
            "\n".join(exc.missing_images) + "\n", encoding="utf-8"
        )
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Missing image list saved to: {missing_log_path}", file=sys.stderr)
        raise SystemExit(2)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)

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
