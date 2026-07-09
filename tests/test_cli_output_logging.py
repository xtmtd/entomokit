from __future__ import annotations

import argparse
import sys

import pytest


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


def test_save_log_strips_ansi_cursor_sequences(tmp_path) -> None:
    from src.common import cli

    log_path = tmp_path / "log.txt"
    args = argparse.Namespace(example=456)

    old_argv = sys.argv[:]
    sys.argv = ["entomokit", "classify", "train"]
    try:
        cli.save_log(tmp_path, args)
        sys.stdout.write("\x1b[A")
        sys.stdout.write("\x1b[2K")
        sys.stdout.write("clean line\n")
        sys.stdout.flush()
        cli._disable_output_capture()
    finally:
        sys.argv = old_argv
        cli._disable_output_capture()

    content = log_path.read_text(encoding="utf-8")
    assert "clean line" in content
    assert "[A" not in content
    assert "[2K" not in content


def test_guard_exits_on_nonempty_no_flags(tmp_path):
    from src.common.cli import check_output_dir

    p = tmp_path / "out"
    p.mkdir()
    (p / "x.txt").write_text("x")
    with pytest.raises(SystemExit):
        check_output_dir(p, resume=False, overwrite=False)


def test_guard_resume_allows_nonempty(tmp_path):
    from src.common.cli import check_output_dir

    p = tmp_path / "out"
    p.mkdir()
    (p / "x.txt").write_text("x")
    check_output_dir(p, resume=True, overwrite=False)
    assert (p / "x.txt").exists()


def test_guard_overwrite_clears_dir(tmp_path):
    from src.common.cli import check_output_dir

    p = tmp_path / "out"
    p.mkdir()
    (p / "x.txt").write_text("x")
    check_output_dir(p, resume=False, overwrite=True)
    assert not (p / "x.txt").exists()
    assert p.exists()


def test_guard_creates_missing_dir(tmp_path):
    from src.common.cli import check_output_dir

    p = tmp_path / "out" / "nested"
    check_output_dir(p, resume=False, overwrite=False)
    assert p.exists()


def test_guard_empty_dir_no_error(tmp_path):
    from src.common.cli import check_output_dir

    p = tmp_path / "out"
    p.mkdir()
    check_output_dir(p, resume=False, overwrite=False)
    assert p.exists()


def test_signal_handler_first_ctrl_c_sets_flag_then_second_raises():
    from src.common import cli

    cli._shutdown_requested = False
    cli.signal_handler(cli.signal.SIGINT, None)
    assert cli._shutdown_requested is True

    with pytest.raises(KeyboardInterrupt):
        cli.signal_handler(cli.signal.SIGINT, None)


def test_reset_shutdown_flag_clears_state():
    from src.common import cli

    cli._shutdown_requested = True
    cli.reset_shutdown_flag()
    assert cli._shutdown_requested is False
