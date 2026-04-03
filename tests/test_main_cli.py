"""Tests for top-level entomokit CLI behavior."""

from __future__ import annotations

import pytest


def test_help_lists_install_completion(capsys: pytest.CaptureFixture[str]) -> None:
    """Top-level help includes the completion installer flag."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--install-completion" in out


def test_help_includes_quick_examples(capsys: pytest.CaptureFixture[str]) -> None:
    """Top-level help should show quick command examples near the top."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Quick examples:" in out
    assert "entomokit segment --input-dir ./images --out-dir ./out" in out


def test_help_uses_boxed_section_titles(capsys: pytest.CaptureFixture[str]) -> None:
    """Top-level help should render boxed commands/options headings."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "[ Commands ]:" in out
    assert "[ Options ]:" in out


def test_segment_help_has_quick_examples_and_boxed_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Segment help should include quick examples and boxed options section."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["segment", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Quick examples:" in out
    assert "entomokit segment --input-dir ./images --out-dir ./segmented" in out
    assert "[ Options ]:" in out


def test_classify_help_has_quick_examples_and_boxed_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Classify group help should include examples and boxed command list."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["classify", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Quick examples:" in out
    assert (
        "entomokit classify train --train-csv train.csv --images-dir ./images --out-dir ./model"
        in out
    )
    assert "[ Commands ]:" in out
    assert "[ Options ]:" in out


def test_classify_train_help_has_quick_examples_and_boxed_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Classify train help should include examples and boxed options section."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["classify", "train", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Quick examples:" in out
    assert (
        "entomokit classify train --train-csv train.csv --images-dir ./images --out-dir ./model"
        in out
    )
    assert "[ Options ]:" in out


def test_install_completion_works_without_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--install-completion should run as a global option."""
    from entomokit import main as cli_main

    called = {"value": False}

    def _fake_install() -> int:
        called["value"] = True
        return 0

    monkeypatch.setattr(cli_main, "_install_completion", _fake_install)
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["--install-completion"])

    assert exc.value.code == 0
    assert called["value"] is True


def test_main_enables_argcomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() should activate argcomplete hook before parsing."""
    from entomokit import main as cli_main

    called = {"value": False}

    def _fake_activate(parser: object) -> None:
        called["value"] = True

    monkeypatch.setattr(cli_main, "_activate_argcomplete", _fake_activate)
    with pytest.raises(SystemExit):
        cli_main.main(["--help"])

    assert called["value"] is True


def test_top_level_command_order_matches_dataset_workflow() -> None:
    """Top-level commands should follow dataset preparation workflow order."""
    import argparse

    from entomokit.main import _build_parser

    parser = _build_parser()
    commands: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            commands = list(action.choices.keys())
            break

    assert commands[:7] == [
        "extract-frames",
        "segment",
        "synthesize",
        "clean",
        "augment",
        "split-csv",
        "classify",
    ]


def test_version_flag_prints_package_version_long(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--version should print version and exit 0."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip() == "entomokit 0.1.6"


def test_version_flag_prints_package_version_short(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """-v should print version and exit 0 at top-level."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["-v"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip() == "entomokit 0.1.6"
