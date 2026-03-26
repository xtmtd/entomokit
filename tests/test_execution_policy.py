"""Tests for entomokit-only execution guardrail."""

from __future__ import annotations

from entomokit.execution_policy import validate_execution_command


def test_validate_execution_command_allows_entomokit() -> None:
    result = validate_execution_command(
        "entomokit clean --input-dir ./raw --out-dir ./out"
    )
    assert result["allowed"] is True


def test_validate_execution_command_blocks_custom_python_script() -> None:
    result = validate_execution_command("python my_custom_clean.py --input ./raw")
    assert result["allowed"] is False
    assert "Use entomokit CLI" in str(result["reason"])


def test_validate_execution_command_allows_schema_export_script() -> None:
    cmd = (
        "python skills/entomokit-workflow/scripts/export_cli_schema.py --command clean"
    )
    result = validate_execution_command(cmd)
    assert result["allowed"] is True


def test_validate_execution_command_allows_fallback_with_reason() -> None:
    result = validate_execution_command(
        "python my_custom_clean.py --input ./raw",
        allow_fallback_script=True,
        fallback_reason="No equivalent entomokit subcommand for this one-off migration.",
    )
    assert result["allowed"] is True
    assert str(result["reason"]).startswith("fallback-approved:")
