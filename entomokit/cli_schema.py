"""Build machine-readable CLI parameter schemas from argparse."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator


def _subparser_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _leaf_commands(
    parser: argparse.ArgumentParser,
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    sub = _subparser_action(parser)
    if sub is None:
        yield prefix, parser
        return

    for name, child in sub.choices.items():
        yield from _leaf_commands(child, (*prefix, name))


def _stringify_default(value: object) -> object:
    if value is argparse.SUPPRESS:
        return None
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _value_hint(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):
        return "true | false"
    if isinstance(action, argparse._StoreFalseAction):
        return "true | false"
    if action.choices:
        return " | ".join(str(choice) for choice in action.choices)
    if action.type is not None:
        type_name = getattr(action.type, "__name__", str(action.type))
        return f"<{type_name}>"
    return "<value>"


def _action_schema(action: argparse.Action) -> dict[str, object]:
    if action.option_strings:
        name = next(
            (opt for opt in reversed(action.option_strings) if opt.startswith("--")),
            action.option_strings[0],
        )
    else:
        name = action.dest

    return {
        "name": name,
        "dest": action.dest,
        "options": list(action.option_strings),
        "action_kind": _infer_action_kind(action),
        "required": bool(getattr(action, "required", False)),
        "default": _stringify_default(getattr(action, "default", None)),
        "choices": [str(choice) for choice in action.choices]
        if action.choices
        else None,
        "value_type": _infer_value_type(action),
        "value_hint": _value_hint(action),
        "help": None if action.help in (None, argparse.SUPPRESS) else str(action.help),
    }


def _infer_value_type(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"
    if action.type in (int, float, str):
        return action.type.__name__
    return "str"


def _infer_action_kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):
        return "store_true"
    if isinstance(action, argparse._StoreFalseAction):
        return "store_false"
    return "store"


def build_command_schemas(
    parser: argparse.ArgumentParser | None = None,
) -> dict[str, dict[str, object]]:
    """Return schema map for every executable command path."""
    if parser is None:
        from entomokit.main import _build_parser

        parser = _build_parser()

    schemas: dict[str, dict[str, object]] = {}
    for path, leaf_parser in _leaf_commands(parser):
        if not path:
            continue

        params: list[dict[str, object]] = []
        for action in leaf_parser._actions:
            if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
                continue
            params.append(_action_schema(action))

        key = " ".join(path)
        schemas[key] = {"command": key, "parameters": params}

    return schemas


def get_command_schema(command: str) -> dict[str, object] | None:
    """Lookup schema for a command path like 'clean' or 'classify train'."""
    command = " ".join(command.split())
    if not command:
        return None
    return build_command_schemas().get(command)


def dumps_command_schemas(indent: int = 2) -> str:
    """Serialize all command schemas as JSON."""
    return json.dumps(build_command_schemas(), ensure_ascii=True, indent=indent)
