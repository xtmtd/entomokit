"""Tests for machine-readable CLI schema export."""

from __future__ import annotations

from entomokit.cli_schema import build_command_schemas, get_command_schema


def test_build_command_schemas_includes_leaf_commands() -> None:
    schemas = build_command_schemas()

    assert "clean" in schemas
    assert "measure" in schemas
    assert "classify train" in schemas
    assert "classify" not in schemas


def test_clean_schema_contains_required_and_enum_metadata() -> None:
    schema = get_command_schema("clean")
    assert schema is not None

    params = {item["name"]: item for item in schema["parameters"]}
    assert params["--input-dir"]["required"] is True
    assert params["--out-dir"]["required"] is True
    assert params["--out-image-format"]["value_hint"] == "jpg | png | tif"
    assert params["--keep-exif"]["action_kind"] == "store_true"


def test_unknown_command_schema_returns_none() -> None:
    assert get_command_schema("not-a-command") is None


def test_measure_schema_excludes_threads() -> None:
    schema = get_command_schema("measure")
    assert schema is not None
    names = {item["name"] for item in schema["parameters"]}
    assert "--threads" not in names
    assert "-n" not in names


def test_classify_cam_schema_exposes_save_npy_values() -> None:
    from entomokit.cli_schema import get_command_schema

    schema = get_command_schema("classify cam")
    assert schema is not None
    params = {item["name"]: item for item in schema["parameters"]}

    save_npy = params["--save-npy"]
    assert save_npy["choices"] == ["none", "raw", "normalized"]
    assert save_npy["default"] == "none"
    assert save_npy["value_hint"] == "none | raw | normalized"
