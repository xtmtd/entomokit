"""Tests for clean's decision reports and deterministic grouping."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.cleaning.processor import ImageCleaner


def _write(path: Path, color=(255, 0, 0), size=(12, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _clean(input_dir: Path, out_dir: Path, **kwargs) -> dict:
    cleaner = ImageCleaner(
        str(input_dir), str(out_dir / "cleaned_images"), **kwargs
    )
    return cleaner.process_directory(log_path=str(out_dir / "log.txt"))


def _rows(report_dir: Path) -> list[dict]:
    with (report_dir / "filter_decisions.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _manifest(report_dir: Path) -> dict:
    return json.loads(
        (report_dir / "clean_manifest.json").read_text(encoding="utf-8")
    )


def test_filter_decisions_has_one_row_per_input(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    for index in range(3):
        _write(
            input_dir / f"class_{index}" / f"image_{index}.png",
            (index * 60, 10, 10),
        )

    results = _clean(input_dir, out, dedup_mode="none")

    rows = _rows(out)
    assert len(rows) == 3
    assert [row["input_path"] for row in rows] == [
        "class_0/image_0.png",
        "class_1/image_1.png",
        "class_2/image_2.png",
    ]
    assert all(row["status"] == "retained" for row in rows)
    assert results["retained"] == 3
    with (out / "retained_images.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with (out / "dropped_images.csv").open(encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    manifest = _manifest(out)
    assert "index" in manifest
    assert {row["input_path"] for row in manifest["index"]} == {
        row["input_path"] for row in rows
    }


def test_md5_drop_uses_relative_input_and_match_paths(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    _write(input_dir / "train" / "beetles" / "same.png", (200, 10, 10))
    _write(input_dir / "test" / "beetles" / "same.png", (200, 10, 10))

    _clean(input_dir, out, dedup_mode="md5")

    rows = {row["input_path"]: row for row in _rows(out)}
    dropped = rows["train/beetles/same.png"]
    retained = rows["test/beetles/same.png"]
    assert dropped["status"] == "dropped"
    assert dropped["reason"] == "duplicate_md5"
    assert dropped["matched_path"] == "test/beetles/same.png"
    assert dropped["matched_scope"] == "input"
    assert retained["status"] == "retained"
    assert dropped["md5"] == retained["md5"] != ""


def test_phash_group_is_transitive(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    for name, color in (
        ("a.png", (10, 10, 10)),
        ("b.png", (60, 60, 60)),
        ("c.png", (120, 120, 120)),
    ):
        _write(input_dir / name, color)

    values = {"a.png": 0, "b.png": 1, "c.png": 3}

    class FakeHash:
        def __init__(self, value: int):
            self.hash = np.array(
                [int(bit) for bit in f"{value:064b}"], dtype=np.uint8
            )

    def fake_compute_phash(img):
        return FakeHash(values[Path(img.filename).name])

    monkeypatch.setattr(
        "src.cleaning.processor.compute_phash", fake_compute_phash
    )

    _clean(
        input_dir,
        out,
        dedup_mode="phash",
        phash_threshold=1,
    )

    rows = _rows(out)
    groups = {row["group_id"] for row in rows}
    assert len(groups) == 1
    assert all(row["group_size"] == "3" for row in rows)
    assert sorted(row["status"] for row in rows) == [
        "dropped",
        "dropped",
        "retained",
    ]
    assert all(
        row["reason"] == "duplicate_phash"
        for row in rows
        if row["status"] == "dropped"
    )


def test_group_representative_prefers_test_over_val_over_train(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    for split in ("train", "val", "test"):
        _write(input_dir / split / "beetles" / "same.png", (200, 10, 10))

    _clean(input_dir, out, dedup_mode="md5")

    retained = [row for row in _rows(out) if row["status"] == "retained"]
    assert len(retained) == 1
    assert retained[0]["input_path"].startswith("test/")


def test_group_representative_falls_back_to_path_order(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    _write(input_dir / "b" / "same.png", (200, 10, 10))
    _write(input_dir / "a" / "same.png", (200, 10, 10))

    _clean(input_dir, out, dedup_mode="md5")

    retained = [row for row in _rows(out) if row["status"] == "retained"]
    assert len(retained) == 1
    assert retained[0]["input_path"] == "a/same.png"


def test_output_names_are_assigned_in_sorted_path_order(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    _write(input_dir / "a.jpg", (255, 0, 0))
    _write(input_dir / "a.png", (0, 0, 255))

    _clean(input_dir, out, dedup_mode="none")

    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["a.jpg"]["output_path"].endswith("a.jpg")
    assert rows["a.png"]["output_path"].endswith("a_1.jpg")


def test_clean_is_deterministic_across_threads(tmp_path):
    input_dir = tmp_path / "input"
    _write(input_dir / "train" / "beetles" / "same.png", (200, 10, 10))
    _write(input_dir / "test" / "beetles" / "same.png", (200, 10, 10))
    _write(input_dir / "val" / "moths" / "same.png", (10, 200, 10))
    _write(input_dir / "misc" / "other.png", (10, 10, 200))

    outputs = []
    manifests = []
    file_sets = []
    for thread_count in (1, 8, 32):
        out = tmp_path / f"out_{thread_count}"
        _clean(input_dir, out, dedup_mode="md5", threads=thread_count)
        outputs.append((out / "filter_decisions.csv").read_text(encoding="utf-8"))
        manifests.append(_manifest(out))
        file_sets.append(
            sorted(
                path.relative_to(out).as_posix()
                for path in out.rglob("*")
                if path.is_file()
                and path.name
                not in {
                    "log.txt",
                    "filter_decisions.csv",
                    "retained_images.csv",
                    "dropped_images.csv",
                    "clean_manifest.json",
                }
            )
        )

    # A repeated run at one thread count must match itself.
    repeated = tmp_path / "out_repeat"
    _clean(input_dir, repeated, dedup_mode="md5", threads=8)
    outputs.append((repeated / "filter_decisions.csv").read_text(encoding="utf-8"))
    manifests.append(_manifest(repeated))

    assert len(set(outputs)) == 1
    stripped = []
    for manifest in manifests:
        copy = dict(manifest)
        copy.pop("threads", None)
        stripped.append(copy)
    assert all(item == stripped[0] for item in stripped)
    assert all(item == file_sets[0] for item in file_sets)


def test_empty_input_writes_zero_row_reports(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    input_dir.mkdir()

    _clean(input_dir, out, dedup_mode="none")

    for name in (
        "filter_decisions.csv",
        "retained_images.csv",
        "dropped_images.csv",
        "clean_manifest.json",
    ):
        assert (out / name).is_file()
    manifest = _manifest(out)
    assert manifest["n_input"] == 0
    assert manifest["files"] == []
    assert manifest["index"] == []
    assert _rows(out) == []


def test_report_counts_satisfy_the_status_invariant(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    _write(input_dir / "one.png", (200, 10, 10))
    _write(input_dir / "two.png", (200, 10, 10))
    (input_dir / "bad.png").write_bytes(b"not-an-image")

    _clean(input_dir, out, dedup_mode="md5")

    manifest = _manifest(out)
    assert manifest["n_input"] == 3
    assert manifest["n_input"] == (
        manifest["n_retained"]
        + manifest["n_dropped"]
        + manifest["n_existing"]
        + manifest["n_errors"]
    )
    assert manifest["n_errors"] == 1


def test_write_failure_promotes_next_group_member(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    input_dir.mkdir()
    _write(input_dir / "one.png", (255, 0, 0))
    _write(input_dir / "two.png", (255, 0, 0))

    original_save = Image.Image.save
    calls = 0

    def fail_first_save(self, fp, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk full")
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_first_save)
    _clean(input_dir, out, dedup_mode="md5", threads=1)

    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["one.png"]["status"] == "error"
    assert rows["one.png"]["reason"].startswith("write_error:")
    assert rows["two.png"]["status"] == "retained"
    assert (out / "cleaned_images" / "two.jpg").is_file()
    assert not (out / "cleaned_images" / "one.jpg").exists()


def test_md5_failure_does_not_drop_later_duplicate(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    input_dir.mkdir()
    for name in ("one.png", "two.png", "three.png"):
        _write(input_dir / name, (1, 2, 3))

    original_save = Image.Image.save
    calls = 0

    def fail_first_save(self, fp, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk full")
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_first_save)
    results = _clean(input_dir, out, dedup_mode="md5", threads=1)

    assert results["retained"] == 1
    assert results["errors"] == 1
    assert len(list((out / "cleaned_images").glob("*.jpg"))) == 1


def test_group_write_failure_preserves_group_membership(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    input_dir.mkdir()
    for name in ("one.png", "two.png", "three.png"):
        _write(input_dir / name, (1, 2, 3))

    def always_fail(self, fp, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", always_fail)
    _clean(input_dir, out, dedup_mode="md5", threads=1)

    rows = _rows(out)
    assert all(row["status"] == "error" for row in rows)
    reasons = sorted(row["reason"] for row in rows)
    assert reasons.count("group_write_failed") == 2
    assert sum(reason.startswith("write_error:") for reason in reasons) == 1
    assert len({row["group_id"] for row in rows}) == 1
    assert all(row["group_size"] == "3" for row in rows)


def test_promotion_recomputes_reason_and_distance(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    input_dir.mkdir()
    _write(input_dir / "a.png", (10, 10, 10))
    _write(input_dir / "b.png", (60, 60, 60))
    _write(input_dir / "c.png", (10, 10, 10))
    values = {"a.png": 0, "b.png": 1, "c.png": 0}

    class FakeHash:
        def __init__(self, value):
            self.hash = np.array(
                [int(bit) for bit in f"{value:064b}"], dtype=np.uint8
            )

    monkeypatch.setattr(
        "src.cleaning.processor.compute_phash",
        lambda img: FakeHash(values[Path(img.filename).name]),
    )

    original_save = Image.Image.save
    calls = 0

    def fail_first_save(self, fp, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk full")
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_first_save)
    _clean(input_dir, out, dedup_mode="md5+phash", phash_threshold=1, threads=1)

    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["a.png"]["status"] == "error"
    assert rows["b.png"]["status"] == "retained"
    assert rows["c.png"]["status"] == "dropped"
    assert rows["c.png"]["kept_path"] == "b.png"
    assert rows["c.png"]["matched_path"] == "b.png"
    assert rows["c.png"]["reason"] == "duplicate_phash"
    assert rows["c.png"]["distance"] == "1"

    manifest = _manifest(out)
    indexed = {row["input_path"]: row for row in manifest["index"]}
    assert indexed["a.png"]["status"] == "dropped"
    assert indexed["a.png"]["output_path"] == "cleaned_images/b.jpg"
    assert indexed["b.png"]["status"] == "retained"
    assert indexed["c.png"]["status"] == "dropped"
    assert indexed["c.png"]["output_path"] == "cleaned_images/b.jpg"


def test_manifest_contains_cumulative_index_with_dropped_rows(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    for name, color in (
        ("a.png", (10, 10, 10)),
        ("b.png", (60, 60, 60)),
        ("c.png", (120, 120, 120)),
    ):
        _write(input_dir / name, color)
    values = {"a.png": 0, "b.png": 1, "c.png": 2}

    class FakeHash:
        def __init__(self, value):
            self.hash = np.array(
                [int(bit) for bit in f"{value:064b}"], dtype=np.uint8
            )

    monkeypatch.setattr(
        "src.cleaning.processor.compute_phash",
        lambda img: FakeHash(values[Path(img.filename).name]),
    )
    _clean(input_dir, out, dedup_mode="phash", phash_threshold=1, threads=1)

    manifest = _manifest(out)
    assert "index" in manifest
    indexed = {row["input_path"]: row for row in manifest["index"]}
    assert indexed["a.png"]["status"] == "retained"
    assert indexed["b.png"]["status"] == "dropped"
    assert indexed["c.png"]["status"] == "dropped"
