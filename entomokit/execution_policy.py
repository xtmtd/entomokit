"""Execution command guardrails for entomokit workflow orchestration."""

from __future__ import annotations

import os
import shlex
from pathlib import Path


ALLOWED_SCHEMA_EXPORT_SCRIPT = "skills/entomokit-workflow/scripts/export_cli_schema.py"
FORBIDDEN_SHELL_TOKENS = ("&&", "||", "|", ";", "&", ">", "<", "`", "\n", "\r")


def _contains_forbidden_shell_syntax(command: str) -> str | None:
    for token in FORBIDDEN_SHELL_TOKENS:
        if token in command:
            return token
    return None


def _strip_wrapping_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def _split_command(command: str) -> list[str]:
    parts = shlex.split(command, posix=os.name != "nt")
    if os.name == "nt":
        return [_strip_wrapping_quotes(part) for part in parts]
    return parts


def _binary_stem(token: str) -> str:
    return Path(token).stem.lower()


def _is_python_binary(token: str) -> bool:
    stem = _binary_stem(token)
    return stem == "python" or stem.startswith("python")


def _normalize_script_path(token: str) -> str:
    normalized = token.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def validate_execution_command(
    command: str,
    *,
    allow_fallback_script: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    trimmed = command.strip()
    if not trimmed:
        return {"allowed": False, "reason": "Command is empty."}

    forbidden = _contains_forbidden_shell_syntax(trimmed)
    if forbidden is not None:
        return {
            "allowed": False,
            "reason": (
                "Shell control syntax is not allowed in guarded commands. "
                f"Found forbidden token: {forbidden!r}"
            ),
        }

    try:
        parts = _split_command(trimmed)
    except ValueError:
        return {"allowed": False, "reason": "Command parsing failed."}

    if not parts:
        return {"allowed": False, "reason": "Command is empty."}

    if _binary_stem(parts[0]) == "entomokit":
        return {
            "allowed": True,
            "reason": "allowed",
            "argv": parts,
            "command_kind": "entomokit",
        }

    if (
        len(parts) >= 2
        and _is_python_binary(parts[0])
        and _normalize_script_path(parts[1]) == ALLOWED_SCHEMA_EXPORT_SCRIPT
    ):
        return {
            "allowed": True,
            "reason": "allowed",
            "argv": parts,
            "command_kind": "schema_export",
        }

    if allow_fallback_script:
        reason = (fallback_reason or "").strip()
        if not reason:
            return {
                "allowed": False,
                "reason": "Fallback script requested but fallback_reason is missing.",
            }
        return {
            "allowed": True,
            "reason": f"fallback-approved: {reason}",
            "argv": parts,
            "command_kind": "fallback",
        }

    return {
        "allowed": False,
        "reason": (
            "Use entomokit CLI for workflow actions. "
            "Custom scripts are blocked unless explicitly approved as fallback."
        ),
    }
