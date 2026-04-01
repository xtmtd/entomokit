#!/usr/bin/env python3
"""Export EntomoKit CLI command schemas as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_project_root_on_path()

from entomokit.cli_schema import build_command_schemas


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export machine-readable entomokit CLI schema"
    )
    p.add_argument(
        "--command",
        default=None,
        help="Optional command path filter, e.g. 'clean' or 'classify train'.",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Output minified JSON instead of pretty JSON.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    schema = build_command_schemas()

    if args.command:
        key = " ".join(args.command.split())
        if key not in schema:
            available = ", ".join(sorted(schema.keys()))
            print(
                f"Unknown command path: {key}. Available commands: {available}",
                file=sys.stderr,
            )
            return 2
        payload: object = schema[key]
    else:
        payload = schema

    if args.compact:
        print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    else:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
