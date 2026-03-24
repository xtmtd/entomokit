"""Tests for split-csv val and copy-images features."""

import pytest
import pandas as pd
from pathlib import Path
from src.splitting.splitter import DatasetSplitter


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "image": [f"img_{i:03d}.jpg" for i in range(100)],
            "label": (["cat"] * 50 + ["dog"] * 50),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


def test_val_ratio_creates_val_csv(tmp_path, sample_csv):
    csv_path, _ = sample_csv
    splitter = DatasetSplitter(str(csv_path), str(tmp_path / "out"), seed=42)
    results = splitter.split(mode="ratio", known_test_ratio=0.1, val_ratio=0.1)
    assert (tmp_path / "out" / "val.csv").exists()
    assert results["val"] > 0


def test_no_val_by_default(tmp_path, sample_csv):
    csv_path, _ = sample_csv
    splitter = DatasetSplitter(str(csv_path), str(tmp_path / "out"), seed=42)
    results = splitter.split(mode="ratio", known_test_ratio=0.1)
    assert not (tmp_path / "out" / "val.csv").exists()
    assert results.get("val", 0) == 0


def test_copy_images_creates_subdirs(tmp_path, sample_csv):
    csv_path, df = sample_csv
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in df["image"]:
        (images_dir / name).write_bytes(b"fake")

    splitter = DatasetSplitter(str(csv_path), str(tmp_path / "out"), seed=42)
    splitter.split(
        mode="ratio",
        known_test_ratio=0.1,
        copy_images=True,
        images_dir=images_dir,
    )
    assert (tmp_path / "out" / "images" / "train").is_dir()
    assert (tmp_path / "out" / "images" / "test_known").is_dir()
    assert len(list((tmp_path / "out" / "images" / "train").iterdir())) > 0
