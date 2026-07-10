"""entomokit — unified CLI entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def _ensure_project_root_on_path() -> None:
    """Ensure local project root is importable before similarly named packages."""
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
def _build_parser() -> argparse.ArgumentParser:
    description = with_examples(
        "A toolkit for building insect image datasets.",
        [
            "entomokit extract-frames --input-dir ./video.mp4 --out-dir ./frames",
            "entomokit segment --input-dir ./images --out-dir ./out",
            "entomokit classify train --train-csv train.csv --images-dir ./images --out-dir ./model",
        ],
    )
    parser = argparse.ArgumentParser(
        prog="entomokit",
        description=description,
        formatter_class=RichHelpFormatter,
    )
    style_parser(parser)
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
        help="Show entomokit version and exit.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        title="[ Commands ]",
    )

    subparsers.required = False
    # Lazy imports keep startup fast and avoid heavy optional deps at import time
    from entomokit import extract_frames as _extract_frames
    from entomokit import segment as _segment
    from entomokit import synthesize as _synthesize
    from entomokit import clean as _clean
    from entomokit import augment as _augment
    from entomokit import split_csv as _split_csv
    from entomokit import measure as _measure
    from entomokit import doctor as _doctor
    from entomokit import completion as _completion
    from entomokit import update as _update
    from entomokit.classify import register as _register_classify

    _extract_frames.register(subparsers)
    _segment.register(subparsers)
    _measure.register(subparsers)
    _synthesize.register(subparsers)
    _clean.register(subparsers)
    _augment.register(subparsers)
    _split_csv.register(subparsers)
    _register_classify(subparsers)
    _doctor.register(subparsers)
    _update.register(subparsers)
    _completion.register(subparsers)

    return parser


def _get_version() -> str:
    try:
        from entomokit._version import __version__

        return __version__
    except ImportError:
        pass

    try:
        return importlib.metadata.version("entomokit")
    except importlib.metadata.PackageNotFoundError:
        return "0.4.1"


def main(argv: list[str] | None = None) -> None:
    _ensure_project_root_on_path()
    parser = _build_parser()

    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.error("the following arguments are required: <command>")

    args.func(args)


if __name__ == "__main__":
    main()
