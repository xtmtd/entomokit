"""Tests for package version metadata."""

from __future__ import annotations


def test_setup_version_is_0_1_3() -> None:
    """setup.py should publish version 0.1.3."""
    from pathlib import Path

    setup_text = Path("setup.py").read_text(encoding="utf-8")
    assert 'version="0.1.3"' in setup_text
