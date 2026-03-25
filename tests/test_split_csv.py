"""Tests for split-csv val and copy-images features."""

import argparse
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


def test_split_csv_run_passes_known_test_count(tmp_path, monkeypatch):
    from entomokit import split_csv as split_csv_cli

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "image": [f"img_{i:03d}.jpg" for i in range(20)],
            "label": (["cat"] * 10 + ["dog"] * 10),
        }
    ).to_csv(csv_path, index=False)

    captured = {}

    class _FakeSplitter:
        def __init__(self, raw_image_csv, out_dir, seed):
            captured["init"] = {
                "raw_image_csv": raw_image_csv,
                "out_dir": out_dir,
                "seed": seed,
            }

        def split(self, **kwargs):
            captured["split_kwargs"] = kwargs
            return {"train": 10, "val": 0, "test_known": 4, "test_unknown": 0}

    monkeypatch.setattr("src.common.cli.setup_shutdown_handler", lambda: None)
    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.splitting.splitter.DatasetSplitter", _FakeSplitter)

    args = argparse.Namespace(
        raw_image_csv=str(csv_path),
        mode="count",
        unknown_test_classes_ratio=0.0,
        known_test_classes_ratio=0.1,
        unknown_test_classes_count=3,
        known_test_classes_count=7,
        val_ratio=0.0,
        val_count=0,
        min_count_per_class=0,
        max_count_per_class=None,
        seed=42,
        out_dir=str(tmp_path / "out"),
        images_dir=None,
        copy_images=False,
        verbose=False,
    )

    split_csv_cli.run(args)

    assert captured["split_kwargs"]["known_test_count"] == 7
    assert captured["split_kwargs"]["unknown_test_count"] == 3
