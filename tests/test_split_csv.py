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
    train_csv = pd.read_csv(tmp_path / "out" / "train.csv")
    train_dir = tmp_path / "out" / "images" / "train"
    assert {path.name for path in train_dir.iterdir()} == {
        Path(image).name for image in train_csv["image"]
    }

def test_copy_images_preserves_nested_same_basename_files(tmp_path):
    csv_path = tmp_path / "data.csv"
    rows = pd.DataFrame(
        {"image": ["a/x.png", "b/x.png", "c/y.png"], "label": ["a", "b", "c"]}
    )
    rows.to_csv(csv_path, index=False)
    images_dir = tmp_path / "source"
    sources = [
        ("a/x.png", b"first"),
        ("b/x.png", b"second"),
        ("c/y.png", b"legacy-flat"),
    ]
    for rel_path, content in sources:
        src = images_dir / rel_path
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(content)

    out_dir = tmp_path / "out"
    splitter = DatasetSplitter(str(csv_path), str(out_dir), seed=42)
    splitter.split(
        mode="ratio", known_test_ratio=0, copy_images=True, images_dir=images_dir
    )

    copied = out_dir / "images" / "train"
    assert (copied / "a" / "x.png").read_bytes() == b"first"
    assert (copied / "b" / "x.png").read_bytes() == b"second"
    assert (copied / "y.png").read_bytes() == b"legacy-flat"
    assert sorted(p.relative_to(copied).as_posix() for p in copied.rglob("*.png")) == [
        "a/x.png",
        "b/x.png",
        "y.png",
    ]
    assert rows.equals(pd.read_csv(out_dir / "train.csv"))


def test_copy_images_allows_same_basename_in_different_splits(tmp_path):
    images_dir = tmp_path / "source"
    out_dir = tmp_path / "out"
    for rel_path, content in [("a/x.png", b"train"), ("b/x.png", b"test")]:
        src = images_dir / rel_path
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(content)
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {"image": ["a/x.png", "b/x.png"], "label": ["a", "b"]}
    ).to_csv(csv_path, index=False)

    splitter = DatasetSplitter(str(csv_path), str(out_dir), seed=42)
    splitter._copy_images(
        images_dir,
        {
            "train": pd.DataFrame({"image": ["a/x.png"]}),
            "test_known": pd.DataFrame({"image": ["b/x.png"]}),
        },
    )

    assert (out_dir / "images" / "train" / "x.png").read_bytes() == b"train"
    assert (out_dir / "images" / "test_known" / "x.png").read_bytes() == b"test"

@pytest.mark.parametrize(
    "images",
    [
        ["a/x.png", "b/x.png", "c/a"],
        ["c/a", "a/x.png", "b/x.png"],
    ],
)
def test_copy_images_rejects_file_directory_destination_conflict(
    tmp_path, images
):
    images_dir = tmp_path / "source"
    for image in images:
        source = images_dir / image
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(image.encode())
    splitter = DatasetSplitter("unused.csv", str(tmp_path / "out"))

    with pytest.raises(ValueError, match="conflicting copied image paths"):
        splitter._copy_images(
            images_dir,
            {"train": pd.DataFrame({"image": images})},
        )

    assert not (tmp_path / "out" / "images").exists()

def test_copy_images_keeps_legacy_parent_symlink_source_behavior(tmp_path):
    images_dir = tmp_path / "source"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"outside source")
    alias = images_dir / "linked"
    images_dir.mkdir()
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    out_dir = tmp_path / "out"
    splitter = DatasetSplitter("unused.csv", str(out_dir))

    splitter._copy_images(
        images_dir,
        {"train": pd.DataFrame({"image": ["linked/secret.png"]})},
    )

    assert (out_dir / "images" / "train" / "secret.png").read_bytes() == b"outside source"


def test_copy_images_keeps_legacy_parent_traversal_source_behavior(tmp_path):
    images_dir = tmp_path / "source"
    images_dir.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside source")
    out_dir = tmp_path / "out"
    splitter = DatasetSplitter("unused.csv", str(out_dir))

    splitter._copy_images(
        images_dir,
        {"train": pd.DataFrame({"image": ["../outside.png"]})},
    )

    assert (out_dir / "images" / "train" / "outside.png").read_bytes() == b"outside source"
@pytest.mark.parametrize("path_kind", ["absolute", "parent"])
def test_collision_group_accepts_absolute_and_parent_source_paths(
    tmp_path, path_kind
):
    images_dir = tmp_path / "source"
    outside = tmp_path / "outside"
    for parent, content in [("a", b"first"), ("b", b"second")]:
        path = outside / parent / "x.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    images_dir.mkdir()
    if path_kind == "parent":
        image_values = ["../outside/a/x.png", "../outside/b/x.png"]
    else:
        image_values = [str(outside / "a" / "x.png"), str(outside / "b" / "x.png")]
    out_dir = tmp_path / "out"
    splitter = DatasetSplitter("unused.csv", str(out_dir))

    splitter._copy_images(
        images_dir,
        {"train": pd.DataFrame({"image": image_values})},
    )

    train_dir = out_dir / "images" / "train"
    copied = list(train_dir.rglob("x.png"))
    assert len(copied) == 2
    assert sorted(path.read_bytes() for path in copied) == [b"first", b"second"]
    assert all(path.is_relative_to(train_dir) for path in copied)


def test_copy_images_allows_repeated_same_source_row(tmp_path):
    images_dir = tmp_path / "source"
    source = images_dir / "a" / "x.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"same source")
    out_dir = tmp_path / "out"
    splitter = DatasetSplitter("unused.csv", str(out_dir))

    splitter._copy_images(
        images_dir,
        {"train": pd.DataFrame({"image": ["a/x.png", "a/x.png"]})},
    )

    assert (out_dir / "images" / "train" / "x.png").read_bytes() == b"same source"



def test_split_csv_copies_nested_same_basename_files(tmp_path, monkeypatch):
    from entomokit.main import main

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {"image": ["a/x.png", "b/x.png"], "label": ["a", "b"]}
    ).to_csv(csv_path, index=False)
    images_dir = tmp_path / "source"
    for rel_path, content in [("a/x.png", b"first"), ("b/x.png", b"second")]:
        src = images_dir / rel_path
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(content)
    monkeypatch.setattr("src.common.cli.setup_shutdown_handler", lambda: None)
    monkeypatch.setattr("src.common.cli.save_log", lambda *_args, **_kwargs: None)

    main(
        [
            "split-csv",
            "--raw-image-csv", str(csv_path),
            "--known-test-sample-ratio", "0",
            "--copy-images",
            "--images-dir", str(images_dir),
            "--out-dir", str(tmp_path / "out"),
        ]
    )

    copied = tmp_path / "out" / "images" / "train"
    assert (copied / "a" / "x.png").read_bytes() == b"first"
    assert (copied / "b" / "x.png").read_bytes() == b"second"

    train_df = pd.read_csv(tmp_path / "out" / "train.csv")
    assert all((copied / image).is_file() for image in train_df["image"])

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
        unknown_test_sample_ratio=0.0,
        known_test_sample_ratio=0.1,
        unknown_test_sample_count=3,
        known_test_sample_count=7,
        val_ratio=0.0,
        val_count=0,
        min_count_per_class=0,
        max_count_per_class=None,
        seed=42,
        out_dir=str(tmp_path / "out"),
        images_dir=None,
        copy_images=False,
        verbose=False,
        overwrite=False,
    )

    split_csv_cli.run(args)

    assert captured["split_kwargs"]["known_test_count"] == 7
    assert captured["split_kwargs"]["unknown_test_count"] == 3


def test_count_mode_val_split_has_no_train_overlap(tmp_path):
    rows = []
    for i in range(10):
        rows.append({"image": f"cat_{i}.jpg", "label": "cat"})
    for i in range(10):
        rows.append({"image": f"dog_{i}.jpg", "label": "dog"})

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    splitter = DatasetSplitter(str(csv_path), str(tmp_path / "out"), seed=0)
    splitter.split(mode="count", known_test_count=4, val_count=4)

    train = pd.read_csv(tmp_path / "out" / "train.csv")
    val = pd.read_csv(tmp_path / "out" / "val.csv")
    overlap = set(train["image"]).intersection(set(val["image"]))
    assert overlap == set()


def test_split_csv_parser_uses_sample_based_argument_names() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "split-csv",
            "--raw-image-csv",
            "data.csv",
            "--unknown-test-sample-ratio",
            "0.2",
            "--known-test-sample-ratio",
            "0.1",
            "--unknown-test-sample-count",
            "5",
            "--known-test-sample-count",
            "10",
        ]
    )

    assert args.unknown_test_sample_ratio == 0.2
    assert args.known_test_sample_ratio == 0.1
    assert args.unknown_test_sample_count == 5
    assert args.known_test_sample_count == 10


def test_class_count_outputs_keep_label_and_count_columns(tmp_path, sample_csv):
    csv_path, _ = sample_csv
    out_dir = tmp_path / "out"
    splitter = DatasetSplitter(str(csv_path), str(out_dir), seed=42)

    splitter.split(mode="ratio", known_test_ratio=0.1, val_ratio=0.1)

    for name in ["class.train.count", "class.test.known.count", "class.val.count"]:
        count_df = pd.read_csv(out_dir / "class_count" / name)
        assert list(count_df.columns) == ["label", "count"]
