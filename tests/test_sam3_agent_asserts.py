"""Regression tests for SAM3 agent guard assertions."""

from __future__ import annotations

from pathlib import Path


def test_agent_core_does_not_use_tuple_assert_pattern() -> None:
    """Tuple-based assert is always truthy and must not appear."""
    source = Path("src/sam3/agent/agent_core.py").read_text(encoding="utf-8")
    assert 'assert (\n            "<tool>" in generated_text,' not in source
