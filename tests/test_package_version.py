"""Tests for package version metadata."""

from __future__ import annotations


def test_setup_version_is_0_7_0() -> None:
    """setup.py should publish version 0.7.0."""
    from pathlib import Path

    setup_text = Path("setup.py").read_text(encoding="utf-8")
    assert 'version="0.7.0"' in setup_text


def test_version_txt_is_0_7_0() -> None:
    from pathlib import Path

    assert Path("version.txt").read_text(encoding="utf-8").strip() == "0.7.0"


def test_runtime_version_matches_setup_version() -> None:
    """Runtime __version__ should stay in sync with setup.py."""
    from pathlib import Path

    from entomokit._version import __version__

    assert __version__ == "0.7.0"
    setup_text = Path("setup.py").read_text(encoding="utf-8")
    assert f'version="{__version__}"' in setup_text


def test_main_version_and_fallback_literal() -> None:
    from pathlib import Path

    from entomokit.main import _get_version

    assert _get_version() == "0.7.0"
    main_text = Path("entomokit/main.py").read_text(encoding="utf-8")
    assert 'return "0.7.0"' in main_text


def test_save_log_header_contains_0_7_0(tmp_path) -> None:
    import argparse

    from src.common import cli

    cli.save_log(tmp_path, argparse.Namespace())
    cli._disable_output_capture()
    content = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "EntomoKit version: 0.7.0" in content
