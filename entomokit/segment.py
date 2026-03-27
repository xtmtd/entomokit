"""entomokit segment — insect image segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "segment",
        help="Segment insects from images using SAM3, Otsu, or GrabCut.",
        description=with_examples(
            "Segment insects from images using SAM3, Otsu, or GrabCut.",
            [
                "entomokit segment --input-dir ./images --out-dir ./segmented",
                "entomokit segment --input-dir ./images --out-dir ./segmented --segmentation-method sam3 --sam3-checkpoint ./sam3.pt",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    # --- I/O ---
    p.add_argument("--input-dir", "-i", required=True, help="Input images directory.")
    p.add_argument("--out-dir", "-o", required=True, help="Output directory.")
    # --- Segmentation method ---
    p.add_argument(
        "--segmentation-method",
        default="sam3",
        choices=["sam3", "sam3-bbox", "otsu", "otsu-bbox", "grabcut", "grabcut-bbox"],
        help="Segmentation method.",
    )
    p.add_argument(
        "--sam3-checkpoint",
        "-c",
        default=None,
        help="Path to SAM3 checkpoint file (required for sam3/sam3-bbox methods).",
    )
    p.add_argument(
        "--hint", "-t", default="insect", help="Text prompt for SAM3 grounding."
    )
    p.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference.",
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence score for masks (0.0 = no filtering).",
    )
    p.add_argument(
        "--padding-ratio",
        type=float,
        default=0.0,
        help="Padding ratio for bounding box (0.0 = no padding).",
    )
    p.add_argument(
        "--repair-strategy",
        "-r",
        default=None,
        choices=["opencv", "sam3-fill", "black-mask", "lama"],
        help="Repair strategy for filling holes.",
    )
    p.add_argument(
        "--lama-model",
        default=None,
        help="Path to LaMa model checkpoint directory.",
    )
    p.add_argument(
        "--lama-mask-dilate",
        type=int,
        default=0,
        help="Number of dilation iterations for LaMa mask.",
    )
    p.add_argument(
        "--out-image-format",
        "-f",
        default="png",
        choices=["png", "jpg"],
        help="Output image format.",
    )
    p.add_argument(
        "--threads",
        "-n",
        type=int,
        default=8,
        help="Number of parallel workers.",
    )
    # --- Annotation output ---
    p.add_argument(
        "--annotation-format",
        default=None,
        choices=["coco", "voc", "yolo"],
        help="Annotation output format. None = no annotations.",
    )
    p.add_argument(
        "--coco-output-mode",
        default="unified",
        choices=["unified", "separate"],
        help="COCO output mode: unified (single annotations.json) or separate (per-image JSON files).",
    )
    p.add_argument(
        "--coco-bbox-format",
        default="xywh",
        choices=["xywh", "xyxy"],
        help="COCO bbox coordinate convention (only used when --annotation-format=coco).",
    )
    # --- Misc ---
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    _ensure_src_on_path()

    from src.common.cli import (
        setup_shutdown_handler,
        get_shutdown_flag,
        setup_logging,
        save_log,
    )
    from src.segmentation.processor import SegmentationProcessor

    setup_shutdown_handler()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: --input-dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: --input-dir is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # setup_logging returns a logger in this project
    logger = setup_logging(out_dir, verbose=args.verbose)
    save_log(out_dir, args)

    # annotation_format: CLI uses None to mean "no annotations"; processor defaults to "coco"
    # Pass the value directly; if None, processor will still run but we avoid enabling annotations.
    annotation_format = args.annotation_format or "coco"

    # Build constructor kwargs; coco_bbox_format is a Phase-2 param not yet in SegmentationProcessor.
    # Pass it only if the constructor accepts it to avoid TypeError.
    import inspect

    processor_sig = inspect.signature(SegmentationProcessor.__init__)
    constructor_kwargs: dict = dict(
        sam3_checkpoint=args.sam3_checkpoint,
        device=args.device,
        hint=args.hint,
        repair_strategy=args.repair_strategy,
        confidence_threshold=args.confidence_threshold,
        padding_ratio=args.padding_ratio,
        segmentation_method=args.segmentation_method,
        lama_model=args.lama_model,
        lama_mask_dilate=max(0, args.lama_mask_dilate),
        annotation_format=annotation_format,
        coco_output_mode=args.coco_output_mode,
    )
    if "coco_bbox_format" in processor_sig.parameters:
        constructor_kwargs["coco_bbox_format"] = args.coco_bbox_format

    logger.info("Starting segmentation process")
    try:
        processor = SegmentationProcessor(**constructor_kwargs)
    except Exception:
        logger.exception("Failed to initialize processor")
        sys.exit(1)

    try:
        results = processor.process_directory(
            input_dir=str(input_dir),
            output_dir=out_dir,
            num_workers=args.threads,
            disable_tqdm=False,
            output_format=args.out_image_format,
            shutdown_flag=get_shutdown_flag(),
        )

        logger.info("Processing complete!")
        logger.info(f"  Successfully processed: {results['processed']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Output files: {len(results['output_files'])}")

    except Exception:
        logger.exception("Processing failed")
        sys.exit(1)


def _ensure_src_on_path() -> None:
    """Add project root to sys.path if needed (for non-editable installs)."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
