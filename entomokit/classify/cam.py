"""entomokit classify cam — generate GradCAM heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "cam",
        help="Generate GradCAM heatmaps (PyTorch models only, not ONNX).",
        description=with_examples(
            "Generate GradCAM heatmaps (PyTorch models only, not ONNX).",
            [
                "entomokit classify cam --images-dir ./images --model-dir ./model --out-dir ./cam",
                "entomokit classify cam --images-dir ./images --base-model convnextv2_femto --out-dir ./cam --cam-method gradcampp",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument(
        "--label-csv",
        default=None,
        help="Optional CSV with 'image' and 'label' columns. If omitted, all images in --images-dir are used.",
    )
    p.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing images (and those referenced by --label-csv when provided).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write CAM visualizations and artifacts.",
    )
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-dir", help="AutoGluon predictor directory.")
    model_group.add_argument("--base-model", help="timm backbone name.")
    p.add_argument(
        "--checkpoint-path", default=None, help="Custom .pth weights for timm backbone."
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Class count for custom timm checkpoint loading.",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained timm weights when using --base-model.",
    )
    p.add_argument(
        "--cam-method",
        default="gradcam",
        choices=[
            "gradcam",
            "gradcampp",
            "layercam",
            "scorecam",
            "eigencam",
            "ablationcam",
        ],
        help="CAM algorithm used to generate saliency maps.",
    )
    p.add_argument(
        "--arch",
        default=None,
        choices=["cnn", "vit"],
        help="Force architecture type (auto-detected if not set).",
    )
    p.add_argument(
        "--target-layer-name",
        default=None,
        help="Specific model layer for CAM (auto-selected when omitted).",
    )
    p.add_argument(
        "--image-weight",
        type=float,
        default=0.5,
        help="Blend weight of original image in CAM overlay (0-1).",
    )
    p.add_argument(
        "--fig-format",
        default="png",
        choices=["png", "jpg", "pdf"],
        help="Output format for CAM figures.",
    )
    p.add_argument(
        "--save-npy",
        action="store_true",
        help="Save raw CAM heatmaps as NumPy arrays.",
    )
    p.add_argument(
        "--dump-model-structure",
        action="store_true",
        help="Write model layer names to out-dir/model_layers.txt for --target-layer-name reference.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (None = all).",
    )
    p.add_argument(
        "--cam-batch-size",
        type=int,
        default=32,
        help="Batch size for CAM inference.",
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
        help="Compute device for CAM generation.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.classification.utils import select_device, set_num_threads
    from src.classification.cam import run_cam
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    device = select_device(args.device)
    set_num_threads(args.num_threads)

    run_cam(
        label_csv=Path(args.label_csv) if args.label_csv else None,
        images_dir=Path(args.images_dir),
        out_dir=out_dir,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        base_model=args.base_model,
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        cam_method=args.cam_method,
        arch=args.arch,
        target_layer_name=args.target_layer_name,
        image_weight=args.image_weight,
        fig_format=args.fig_format,
        save_npy=args.save_npy,
        dump_model_structure=args.dump_model_structure,
        max_images=args.max_images,
        cam_batch_size=args.cam_batch_size,
        device=device,
    )
