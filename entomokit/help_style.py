"""Shared CLI help style helpers."""

from __future__ import annotations

import argparse


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


def style_parser(parser: argparse.ArgumentParser) -> None:
    parser._optionals.title = "[ Options ]"
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            action.title = "[ Commands ]"
            break
