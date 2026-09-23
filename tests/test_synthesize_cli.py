"""Tests for the synthesize CLI count/seed contract."""

from __future__ import annotations

import pytest


def _parse(*extra: str):
    from entomokit.main import _build_parser

    parser = _build_parser()
    return parser.parse_args(
        ["synthesize", "--target-dir", "t", "--background-dir", "b", "--out-dir", "o", *extra]
    )


def test_num_syntheses_default_is_one():
    assert _parse().num_syntheses == 1


def test_seed_default_is_42():
    assert _parse().seed == 42


@pytest.mark.parametrize("value,expected", [("3", 3), ("0.5", 0.5), ("1.0", 1), ("1", 1)])
def test_num_syntheses_accepts_integer_and_fraction(value, expected):
    assert _parse("--num-syntheses", value).num_syntheses == expected


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "nan", "inf"])
def test_num_syntheses_rejects_invalid(value):
    with pytest.raises(SystemExit):
        _parse("--num-syntheses", value)


def test_run_forwards_count_and_seed(tmp_path, monkeypatch):
    from unittest.mock import patch

    from entomokit import synthesize

    target_dir = tmp_path / "t"
    bg_dir = tmp_path / "b"
    target_dir.mkdir()
    bg_dir.mkdir()

    captured = {}

    def fake_process_directory(self, **kwargs):
        captured.update(kwargs)
        return {"processed": 0, "failed": 0, "output_files": 0, "skipped": 0}

    with patch(
        "src.synthesis.processor.SynthesisProcessor.process_directory",
        fake_process_directory,
    ):
        synthesize.run(
            _parse(
                "--target-dir", str(target_dir),
                "--background-dir", str(bg_dir),
                "--out-dir", str(tmp_path / "o"),
                "--num-syntheses", "0.5",
                "--seed", "9",
            )
        )

    assert captured["num_syntheses"] == 0.5
    assert captured["seed"] == 9
