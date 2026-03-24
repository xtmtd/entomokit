from __future__ import annotations

import argparse
import sys


def test_save_log_captures_stdout_stderr_and_filters_progress(tmp_path) -> None:
    from src.common import cli

    log_path = tmp_path / "log.txt"
    args = argparse.Namespace(example=123)

    old_argv = sys.argv[:]
    sys.argv = ["entomokit", "segment", "--input-dir", "in", "--out-dir", "out"]
    try:
        cli.save_log(tmp_path, args)
        print("normal output line")
        sys.stderr.write("error output line\n")
        sys.stdout.write("\rprogress 10%")
        sys.stdout.write("\rprogress 100%\n")
        sys.stdout.flush()
        sys.stderr.flush()
        cli._disable_output_capture()
    finally:
        sys.argv = old_argv
        cli._disable_output_capture()

    content = log_path.read_text(encoding="utf-8")
    assert "Command: entomokit segment --input-dir in --out-dir out" in content
    assert "Arguments:" in content
    assert "example: 123" in content
    assert "normal output line" in content
    assert "error output line" in content
    assert "progress 10%" not in content
    assert "progress 100%" not in content
