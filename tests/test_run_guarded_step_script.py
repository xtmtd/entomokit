"""Tests for guarded workflow step CLI script."""

from __future__ import annotations

import json
import subprocess
import sys


SCRIPT = "skills/entomokit-workflow/scripts/run_guarded_step.py"


def test_run_guarded_step_script_blocks_custom_script_command() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--step-name",
            "clean",
            "--command-path",
            "clean",
            "--command",
            "python my_custom_clean.py --input ./raw",
            "--param=--input-dir=./raw",
            "--param=--out-dir=./out",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "blocked"
    assert "Use entomokit CLI" in payload["message"]
    assert completed.returncode == 2


def test_run_guarded_step_script_executes_entomokit_command() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--step-name",
            "doctor",
            "--command-path",
            "doctor",
            "--command",
            "entomokit --version",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert payload["return_code"] == 0
    assert completed.returncode == 0


def test_run_guarded_step_script_allows_fallback_when_explicit() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--step-name",
            "custom-fallback",
            "--command-path",
            "doctor",
            "--command",
            f"{sys.executable} -c \"print('ok')\"",
            "--allow-fallback-script",
            "--fallback-reason",
            "User explicitly approved one-off fallback.",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert completed.returncode == 0
