"""classify command group — AutoGluon image classification."""

from __future__ import annotations

import argparse

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "classify",
        help="Image classification commands (AutoGluon).",
        description=with_examples(
            "Image classification commands (AutoGluon).",
            [
                "entomokit classify train --train-csv train.csv --images-dir ./images --out-dir ./model",
                "entomokit classify predict --images-dir ./images --model-dir ./model --out-dir ./pred",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    sub = p.add_subparsers(
        dest="subcommand",
        metavar="<subcommand>",
        title="[ Commands ]",
    )
    sub.required = True

    from entomokit.classify import train, predict, evaluate, embed, cam, export_onnx

    train.register(sub)
    predict.register(sub)
    evaluate.register(sub)
    embed.register(sub)
    cam.register(sub)
    export_onnx.register(sub)
