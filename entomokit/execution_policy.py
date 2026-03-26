"""Execution command guardrails for entomokit workflow orchestration."""

from __future__ import annotations

import shlex


ALLOWED_PREFIXES = (
    "entomokit",
    "python skills/entomokit-workflow/scripts/export_cli_schema.py",
)


def validate_execution_command(
    command: str,
    *,
    allow_fallback_script: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    trimmed = command.strip()
    if not trimmed:
        return {"allowed": False, "reason": "Command is empty."}

    if any(trimmed.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        return {"allowed": True, "reason": "allowed"}

    try:
        parts = shlex.split(trimmed)
    except ValueError:
        return {"allowed": False, "reason": "Command parsing failed."}

    binary = parts[0] if parts else ""
    if binary == "entomokit":
        return {"allowed": True, "reason": "allowed"}

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
        }

    return {
        "allowed": False,
        "reason": (
            "Use entomokit CLI for workflow actions. "
            "Custom scripts are blocked unless explicitly approved as fallback."
        ),
    }
