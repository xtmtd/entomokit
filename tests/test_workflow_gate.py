"""Tests for guarded workflow step execution."""

from __future__ import annotations

from entomokit.workflow_gate import run_guarded_step


def _ok_runner(command: str) -> tuple[int, str, str]:
    return 0, f"ran: {command}", ""


def test_run_guarded_step_success() -> None:
    result = run_guarded_step(
        step_name="clean",
        command_path="clean",
        command="entomokit clean --input-dir ./raw --out-dir ./out",
        user_inputs={"--input-dir": "./raw", "--out-dir": "./out"},
        runner=_ok_runner,
        outputs=["runs/run001/clean-run001/cleaned_images"],
    )

    assert result["status"] == "success"
    assert result["step_name"] == "clean"
    assert str(result["approved_params"].get("--input-dir")) == "./raw"


def test_run_guarded_step_blocks_non_entomokit_command() -> None:
    result = run_guarded_step(
        step_name="clean",
        command_path="clean",
        command="python my_custom_clean.py --input ./raw",
        user_inputs={"--input-dir": "./raw", "--out-dir": "./out"},
        runner=_ok_runner,
    )

    assert result["status"] == "blocked"
    assert "Use entomokit CLI" in result["message"]


def test_run_guarded_step_allows_explicit_fallback() -> None:
    result = run_guarded_step(
        step_name="fallback-step",
        command_path="doctor",
        command="python -c \"print('ok')\"",
        user_inputs={},
        runner=_ok_runner,
        allow_fallback_script=True,
        fallback_reason="User approved fallback after no matching entomokit subcommand.",
    )

    assert result["status"] == "success"


def test_run_guarded_step_blocks_invalid_params_before_execution() -> None:
    result = run_guarded_step(
        step_name="clean",
        command_path="clean",
        command="entomokit clean --input-dir ./raw --out-dir ./out --out-image-format webp",
        user_inputs={
            "--input-dir": "./raw",
            "--out-dir": "./out",
            "--out-image-format": "webp",
        },
        runner=_ok_runner,
    )

    assert result["status"] == "blocked"
    assert "must be one of" in "\n".join(result["errors"])
