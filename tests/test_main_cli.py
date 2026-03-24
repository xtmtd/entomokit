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
