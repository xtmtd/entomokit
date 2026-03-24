"""Tests for CLI argument help text coverage."""

from __future__ import annotations

import argparse


def _walk_parsers(parser: argparse.ArgumentParser):
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _walk_parsers(subparser)


def test_all_optional_flags_have_help_text() -> None:
    """Every optional CLI flag should expose a help description."""
    from entomokit.main import _build_parser

    parser = _build_parser()
    missing_help: list[str] = []

    for subparser in _walk_parsers(parser):
        for action in subparser._actions:
            if not action.option_strings:
                continue
            if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
                continue
            if action.help in (None, argparse.SUPPRESS):
                missing_help.extend(action.option_strings)

    assert missing_help == []
