"""Tests for package version metadata."""

from __future__ import annotations

EXPECTED_VERSION = "0.7.1"


def test_setup_version_matches_target() -> None:
    """setup.py should publish the expected version."""
    from pathlib import Path

    setup_text = Path("setup.py").read_text(encoding="utf-8")
    assert f'version="{EXPECTED_VERSION}"' in setup_text


def test_version_txt_matches_target() -> None:
    from pathlib import Path

    assert Path("version.txt").read_text(encoding="utf-8").strip() == EXPECTED_VERSION


def test_runtime_version_matches_setup_version() -> None:
    """Runtime __version__ should stay in sync with setup.py."""
    from pathlib import Path

    from entomokit._version import __version__

    assert __version__ == EXPECTED_VERSION
    setup_text = Path("setup.py").read_text(encoding="utf-8")
    assert f'version="{__version__}"' in setup_text


def test_main_version_and_fallback_literal() -> None:
    from pathlib import Path

    from entomokit.main import _get_version

    assert _get_version() == EXPECTED_VERSION
    main_text = Path("entomokit/main.py").read_text(encoding="utf-8")
    assert f'return "{EXPECTED_VERSION}"' in main_text


def test_save_log_header_contains_version(tmp_path) -> None:
    import argparse

    from src.common import cli

    cli.save_log(tmp_path, argparse.Namespace())
    cli._disable_output_capture()
    content = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert f"EntomoKit version: {EXPECTED_VERSION}" in content
