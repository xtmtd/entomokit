"""Guarded workflow execution path for conversational orchestration."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from typing import Any, cast

from entomokit.execution_policy import validate_execution_command
from entomokit.param_guard import render_parameter_card, validate_parameters


Runner = Callable[[str], tuple[int, str, str]]


def _default_runner(command: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


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

    active_runner = runner or _default_runner
    code, stdout, stderr = active_runner(command)

    final_status = "success" if code == 0 else "failed"

    return {
        "status": final_status,
        "step_name": step_name,
        "approved_params": approved_params,
        "outputs": outputs or [],
        "return_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "parameter_card": card,
        "errors": [] if code == 0 else [stderr.strip() or "command failed"],
    }
