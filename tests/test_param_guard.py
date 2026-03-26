"""Tests for pre-run parameter validation and card rendering."""

from __future__ import annotations

from entomokit.param_guard import render_parameter_card, validate_parameters


def test_validate_parameters_blocks_unknown_and_missing_required() -> None:
    result = validate_parameters("clean", {"--input-dir": "./raw", "--bogus": "x"})

    assert result["passed"] is False
    assert any("Unknown parameter: --bogus" in err for err in result["errors"])
    assert any(
        "Missing required parameter: --out-dir" in err for err in result["errors"]
    )


def test_validate_parameters_checks_enum_and_type() -> None:
    result = validate_parameters(
        "clean",
        {
            "--input-dir": "./raw",
            "--out-dir": "./out",
            "--out-image-format": "webp",
            "--threads": "abc",
        },
    )

    assert result["passed"] is False
    assert any("--out-image-format must be one of" in err for err in result["errors"])
    assert any("--threads expects integer" in err for err in result["errors"])


def test_render_parameter_card_shows_validation_status() -> None:
    card = render_parameter_card(
        "clean",
        {
            "--input-dir": "./raw",
            "--out-dir": "./out",
            "--out-image-format": "jpg",
            "--recursive": "true",
        },
    )

    assert "步骤：clean" in card
    assert "参数来源：runtime CLI export" in card
    assert "校验状态：passed" in card
