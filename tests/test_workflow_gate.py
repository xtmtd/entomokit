"""Tests for guarded workflow step execution."""

from __future__ import annotations

from entomokit.workflow_gate import run_guarded_step


def _ok_runner(command: list[str]) -> tuple[int, str, str]:
    return 0, f"ran: {' '.join(command)}", ""


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
    assert result["executed_command"].startswith("entomokit clean")


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


def test_run_guarded_step_rebuilds_entomokit_command_from_approved_params() -> None:
    executed: list[list[str]] = []

    def runner(command: list[str]) -> tuple[int, str, str]:
        executed.append(command)
        return 0, "", ""

    result = run_guarded_step(
        step_name="clean",
        command_path="clean",
        command="entomokit clean --input-dir ./raw --out-dir ./MALICIOUS --threads xyz",
        user_inputs={"--input-dir": "./raw", "--out-dir": "./safe"},
        runner=runner,
    )

    assert result["status"] == "success"
    assert executed == [
        [
            "entomokit",
            "clean",
            "--input-dir",
            "./raw",
            "--out-dir",
            "./safe",
            "--out-short-size",
            "512",
            "--out-image-format",
            "jpg",
            "--threads",
            "12",
            "--dedup-mode",
            "md5",
            "--phash-threshold",
            "5",
        ]
    ]


def test_run_guarded_step_blocks_shell_control_syntax() -> None:
    result = run_guarded_step(
        step_name="clean",
        command_path="clean",
        command="entomokit clean --input-dir ./raw --out-dir ./out && echo PWNED",
        user_inputs={"--input-dir": "./raw", "--out-dir": "./out"},
        runner=_ok_runner,
    )

    assert result["status"] == "blocked"
    assert "Shell control syntax is not allowed" in result["message"]


def test_run_guarded_step_blocks_command_path_mismatch() -> None:
    result = run_guarded_step(
        step_name="doctor",
        command_path="doctor",
        command="entomokit --version",
        user_inputs={},
        runner=_ok_runner,
    )

    assert result["status"] == "blocked"
    assert "does not match the guarded step" in result["message"]
