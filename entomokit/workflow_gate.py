"""Guarded workflow execution path for conversational orchestration."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from entomokit.cli_schema import get_command_schema
from entomokit.execution_policy import validate_execution_command
from entomokit.param_guard import render_parameter_card, validate_parameters


Runner = Callable[[list[str]], tuple[int, str, str]]


def _format_command(argv: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _resolve_process_argv(argv: list[str]) -> list[str]:
    if argv and Path(argv[0]).stem.lower() == "entomokit":
        return [sys.executable, "-m", "entomokit.main", *argv[1:]]
    return argv


def _default_runner(argv: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        _resolve_process_argv(argv),
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _extract_entomokit_command_path(argv: list[str]) -> str:
    words: list[str] = []
    for token in argv[1:]:
        if token.startswith("-"):
            break
        words.append(token)
    return " ".join(words)


def _build_entomokit_argv(
    command_path: str,
    approved_params: Mapping[str, object],
) -> list[str]:
    schema = get_command_schema(command_path)
    if schema is None:
        raise ValueError(f"Unknown command schema: {command_path}")

    argv = ["entomokit", *command_path.split()]
    for raw_param in cast(list[object], schema.get("parameters", [])):
        param = cast(dict[str, object], raw_param)
        name = str(param["name"])
        options = [str(opt) for opt in cast(list[object], param.get("options") or [])]
        action_kind = str(param.get("action_kind", "store"))
        value = approved_params.get(name)

        if action_kind == "store_true":
            if bool(value):
                argv.append(name)
            continue

        if action_kind == "store_false":
            if value is False:
                argv.append(name)
            continue

        if value is None:
            continue

        if options:
            argv.extend([name, str(value)])
        else:
            argv.append(str(value))

    return argv


def run_guarded_step(
    *,
    step_name: str,
    command_path: str,
    command: str,
    user_inputs: Mapping[str, object],
    runner: Runner | None = None,
    outputs: list[str] | None = None,
    allow_fallback_script: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Run one workflow step through policy, schema, and validation gates."""

    params = dict(user_inputs)

    policy = validate_execution_command(
        command,
        allow_fallback_script=allow_fallback_script,
        fallback_reason=fallback_reason,
    )
    if not bool(policy["allowed"]):
        message = str(policy["reason"])
        return {
            "status": "blocked",
            "message": message,
            "parameter_card": render_parameter_card(command_path, params),
            "errors": [message],
        }

    validation = validate_parameters(command_path, params)
    card = render_parameter_card(command_path, params)
    validation_errors = cast(list[object], validation.get("errors", []))
    if not bool(validation["passed"]):
        return {
            "status": "blocked",
            "message": "Parameter validation blocked execution.",
            "parameter_card": card,
            "errors": [str(err) for err in validation_errors],
        }

    approved_params = cast(dict[str, Any], validation.get("final_values", {}))
    policy_argv = [str(part) for part in cast(list[object], policy.get("argv", []))]
    command_kind = str(policy.get("command_kind", ""))

    execution_argv = policy_argv
    if command_kind == "entomokit":
        expected_path = " ".join(command_path.split())
        actual_path = _extract_entomokit_command_path(policy_argv)
        if actual_path != expected_path:
            message = (
                "Requested entomokit command path does not match the guarded step. "
                f"Expected {expected_path!r}, got {actual_path or '<root>'!r}."
            )
            return {
                "status": "blocked",
                "message": message,
                "parameter_card": card,
                "errors": [message],
            }

        try:
            execution_argv = _build_entomokit_argv(expected_path, approved_params)
        except ValueError as exc:
            message = str(exc)
            return {
                "status": "blocked",
                "message": message,
                "parameter_card": card,
                "errors": [message],
            }

    active_runner = runner or _default_runner
    code, stdout, stderr = active_runner(execution_argv)

    final_status = "success" if code == 0 else "failed"

    return {
        "status": final_status,
        "step_name": step_name,
        "approved_params": approved_params,
        "executed_command": _format_command(execution_argv),
        "outputs": outputs or [],
        "return_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "parameter_card": card,
        "errors": [] if code == 0 else [stderr.strip() or "command failed"],
    }
