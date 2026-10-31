"""CLI-level tests for clean --dry-run and the augment manifest rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.common import cli as cli_common


def _write(path: Path, color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def _parse(input_dir: Path, out_dir: Path, *extra: str):
    from entomokit.main import _build_parser

    return _build_parser().parse_args(
        [
            "clean",
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            *extra,
        ]
    )


def test_clean_parser_accepts_dry_run_default_false(tmp_path):
    args = _parse(tmp_path / "in", tmp_path / "out")
    assert args.dry_run is False


def test_clean_dry_run_writes_reports_without_images(tmp_path):
    input_dir = tmp_path / "input"
    _write(input_dir / "a.png")
    out = tmp_path / "out"
    args = _parse(input_dir, out, "--dry-run")

    from entomokit import clean

    try:
        clean.run(args)
    finally:
        cli_common._disable_output_capture()

    for name in (
        "filter_decisions.csv",
        "retained_images.csv",
        "dropped_images.csv",
        "clean_manifest.json",
    ):
        assert (out / name).is_file()
    assert not (out / "cleaned_images").exists()
    manifest = json.loads((out / "clean_manifest.json").read_text())
    assert manifest["dry_run"] is True
    assert manifest["output_files"]["cleaned_images"] is None


def test_clean_dry_run_rejects_overwrite(tmp_path, capsys):
    input_dir = tmp_path / "input"
    _write(input_dir / "a.png")
    args = _parse(input_dir, tmp_path / "out", "--dry-run", "--overwrite")

    from entomokit import clean

    with pytest.raises(SystemExit) as exc:
        clean.run(args)
    assert exc.value.code == 1
    assert "cannot be combined" in capsys.readouterr().err


def test_clean_dry_run_requires_resume_on_nonempty_out_dir(tmp_path):
    input_dir = tmp_path / "input"
    _write(input_dir / "a.png")
    out = tmp_path / "out"
    out.mkdir()
    (out / "log.txt").write_text("previous\n")
    args = _parse(input_dir, out, "--dry-run")

    from entomokit import clean

    with pytest.raises(SystemExit):
        clean.run(args)


def test_clean_real_run_after_dry_run_uses_resume(tmp_path):
    input_dir = tmp_path / "input"
    _write(input_dir / "a.png")
    out = tmp_path / "out"

    from entomokit import clean

    try:
        clean.run(_parse(input_dir, out, "--dry-run"))
        clean.run(_parse(input_dir, out, "--resume"))
    finally:
        cli_common._disable_output_capture()

    assert (out / "cleaned_images" / "a.jpg").is_file()


def test_clean_dry_run_says_would_write(tmp_path, capsys):
    input_dir = tmp_path / "input"
    _write(input_dir / "a.png")
    out = tmp_path / "out"

    from entomokit import clean

    try:
        clean.run(_parse(input_dir, out, "--dry-run"))
    finally:
        cli_common._disable_output_capture()

    printed = capsys.readouterr().out
    assert "would write 1 new image" in printed
    assert "wrote 1 new image" not in printed
