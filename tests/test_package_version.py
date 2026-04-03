"""Tests for package version metadata."""

from __future__ import annotations


def test_setup_version_is_0_1_6() -> None:
    """setup.py should publish version 0.1.6."""
    from pathlib import Path

    setup_text = Path("setup.py").read_text(encoding="utf-8")
    assert 'version="0.1.6"' in setup_text


def test_runtime_version_matches_setup_version() -> None:
    """Runtime __version__ should stay in sync with setup.py."""
    from pathlib import Path

    from entomokit._version import __version__

    setup_text = Path("setup.py").read_text(encoding="utf-8")
    assert f'version="{__version__}"' in setup_text
