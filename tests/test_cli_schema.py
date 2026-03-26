"""Tests for machine-readable CLI schema export."""

from __future__ import annotations

from entomokit.cli_schema import build_command_schemas, get_command_schema


def test_build_command_schemas_includes_leaf_commands() -> None:
    schemas = build_command_schemas()

    assert "clean" in schemas
    assert "classify train" in schemas
    assert "classify" not in schemas


def test_clean_schema_contains_required_and_enum_metadata() -> None:
    schema = get_command_schema("clean")
    assert schema is not None

    params = {item["name"]: item for item in schema["parameters"]}
    assert params["--input-dir"]["required"] is True
    assert params["--out-dir"]["required"] is True
    assert params["--out-image-format"]["value_hint"] == "jpg | png | tif"


def test_unknown_command_schema_returns_none() -> None:
    assert get_command_schema("not-a-command") is None
