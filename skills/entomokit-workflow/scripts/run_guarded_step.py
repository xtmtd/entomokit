#!/usr/bin/env python3
"""Run one workflow step through policy, schema, and persistence gates."""

from __future__ import annotations

import argparse
import json

from entomokit.workflow_gate import run_guarded_step


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one guarded entomokit workflow step")
    p.add_argument("--step-name", required=True, help="Logical step name, e.g. clean")
    p.add_argument(
        "--command-path",
        required=True,
        help="Schema command path, e.g. clean or classify train",
    )
    p.add_argument("--command", required=True, help="Shell command to execute")
    p.add_argument(
        "--param",
        action="append",
        default=[],
        help="Parameter assignment in key=value format. Repeatable.",
    )
    p.add_argument(
        "--output",
        action="append",
        default=[],
        help="Output artifact path. Repeatable.",
    )
    p.add_argument(
        "--allow-fallback-script",
        action="store_true",
        help="Allow non-entomokit command only with explicit fallback reason.",
    )
    p.add_argument(
        "--fallback-reason",
        default=None,
        help="Required when --allow-fallback-script is set.",
    )
    return p


def _parse_params(raw_items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --param '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --param '{item}'. Key cannot be empty.")
        parsed[key] = value
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        params = _parse_params(args.param)
    except ValueError as exc:
        print(
            json.dumps({"status": "blocked", "errors": [str(exc)]}, ensure_ascii=True)
        )
        return 2

    result = run_guarded_step(
        step_name=args.step_name,
        command_path=args.command_path,
        command=args.command,
        user_inputs=params,
        outputs=list(args.output),
        allow_fallback_script=bool(args.allow_fallback_script),
        fallback_reason=args.fallback_reason,
    )
    print(json.dumps(result, ensure_ascii=True))
    status = result.get("status")
    if status == "success":
        return 0
    if status == "blocked":
        return 2
    if status == "failed":
        code = result.get("return_code")
        return int(code) if isinstance(code, int) and code != 0 else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
