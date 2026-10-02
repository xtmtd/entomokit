"""Shared CLI help style helpers."""

from __future__ import annotations

import argparse

DOCS_BASE_URL = "https://github.com/xtmtd/entomokit/blob/main/"

# Reference for every registered parser, keyed by parser prog. One place owns the
# published URL base and each command's document or README anchor; a new command
# adds its row here, and the documentation tests fail if a pointer is missing.
DOC_LINKS: dict[str, str] = {
    "entomokit": "README.md",
    "entomokit extract-frames": "docs/commands/extract-frames.md",
    "entomokit segment": "docs/commands/segment.md",
    "entomokit measure": "docs/commands/measure.md",
    "entomokit synthesize": "docs/commands/synthesize.md",
    "entomokit clean": "docs/commands/clean.md",
    "entomokit augment": "docs/commands/augment.md",
    "entomokit split-csv": "docs/commands/split-csv.md",
    "entomokit classify": "README.md#classify-commands",
    "entomokit classify train": "docs/commands/classify-train.md",
    "entomokit classify predict": "docs/commands/classify-predict.md",
    "entomokit classify evaluate": "docs/commands/classify-evaluate.md",
    "entomokit classify embed": "docs/commands/classify-embed.md",
    "entomokit classify cam": "docs/commands/classify-cam.md",
    "entomokit classify export-onnx": "docs/commands/classify-export-onnx.md",
    "entomokit doctor": "README.md#doctor-command",
    "entomokit update": "README.md#update-command",
    "entomokit completion": "README.md#completion-command",
    "entomokit completion bash": "README.md#completion-command",
    "entomokit completion zsh": "README.md#completion-command",
    "entomokit completion fish": "README.md#completion-command",
}


class RichHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """Keep defaults while preserving manual newlines in help text."""


def with_examples(summary: str, examples: list[str]) -> str:
    if not examples:
        return summary
    lines = [summary, "", "Quick examples:"]
    lines.extend(f"  {example}" for example in examples)
    return "\n".join(lines)


def with_doc_link(description: str | None, path: str) -> str:
    """Append the published documentation URL to a parser description."""
    url = DOCS_BASE_URL + path
    if description and description.strip():
        return f"{description}\n\n{url}"
    return url


def style_parser(parser: argparse.ArgumentParser) -> None:
    parser._optionals.title = "[ Options ]"
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            action.title = "[ Commands ]"
            break
    path = DOC_LINKS.get(parser.prog)
    if path is not None:
        parser.description = with_doc_link(parser.description, path)
