"""Tests for per-file rows in the augment manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


def _write(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def test_augment_manifest_includes_per_file_rows(tmp_path):
    pytest.importorskip("albumentations")
    from src.augment.service import run_augment

    input_dir = tmp_path / "input"
    _write(input_dir / "a.png", (255, 0, 0))
    _write(input_dir / "b.png", (0, 0, 255))

    result = run_augment(
        str(input_dir),
        str(tmp_path / "augmented"),
        preset="light",
        multiply=2,
    )

    assert result.success
    files = result.manifest["files"]
    assert len(files) == 4
    assert all(
        {"original", "augmented", "copy_index"} <= set(item) for item in files
    )
    on_disk = json.loads(
        (tmp_path / "augmented" / "augment_manifest.json").read_text()
    )
    assert on_disk["files"] == files
