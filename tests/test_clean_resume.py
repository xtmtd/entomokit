"""Tests for clean --resume matching and report semantics."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.cleaning.processor import ImageCleaner


def _write(path: Path, color=(255, 0, 0), size=(200, 140)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _run(input_dir: Path, out: Path, *, resume: bool = False, mode: str = "md5"):
    cleaner = ImageCleaner(
        str(input_dir),
        str(out / "cleaned_images"),
        dedup_mode=mode,
        threads=1,
        resume=resume,
    )
    return cleaner.process_directory(log_path=str(out / "log.txt"))


def _rows(out: Path) -> list[dict]:
    with (out / "filter_decisions.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _manifest(out: Path) -> dict:
    return json.loads((out / "clean_manifest.json").read_text(encoding="utf-8"))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    _write(input_dir / "img0.png", (40, 80, 120))
    _write(input_dir / "img1.png", (200, 80, 20))
    return input_dir, tmp_path / "out"


def test_resume_uses_previous_manifest_source_hash(tmp_path):
    input_dir, out = _fixture(tmp_path)
    first = _run(input_dir, out)
    assert first["retained"] == 2

    second = _run(input_dir, out, resume=True)

    assert second["existing"] == 2
    assert second["retained"] == 0
    assert not list((out / "cleaned_images").glob("*_1.jpg"))
    manifest = _manifest(out)
    assert manifest["n_existing"] == 2


def test_resume_reports_existing_output_relative_paths(tmp_path):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    _run(input_dir, out, resume=True)

    rows = {row["input_path"]: row for row in _rows(out)}
    row = rows["img0.png"]
    assert row["status"] == "existing"
    assert row["reason"] == "duplicate_existing_output"
    assert row["kept_path"] == "cleaned_images/img0.jpg"
    assert row["matched_path"] == "cleaned_images/img0.jpg"
    assert row["kept_scope"] == "existing_output"
    assert row["matched_scope"] == "existing_output"
    assert row["output_path"] == "cleaned_images/img0.jpg"

    with (out / "retained_images.csv").open(encoding="utf-8") as handle:
        retained = list(csv.DictReader(handle))
    assert {item["status"] for item in retained} == {"existing"}


def test_resume_ignores_missing_manifest_output_and_warns(tmp_path, caplog):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    (out / "cleaned_images" / "img0.jpg").unlink()

    with caplog.at_level(logging.WARNING):
        second = _run(input_dir, out, resume=True)

    assert "Ignoring missing output" in caplog.text
    assert second["retained"] == 1
    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["img0.png"]["status"] == "retained"
    assert rows["img1.png"]["status"] == "existing"


def test_resume_none_mode_matches_previous_manifest_without_hashes(tmp_path):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out, mode="none")
    second = _run(input_dir, out, resume=True, mode="none")

    assert second["existing"] == 2
    rows = {row["input_path"]: row for row in _rows(out)}
    assert all(row["status"] == "existing" for row in rows.values())


def test_resume_without_manifest_falls_back_to_output_hashing(tmp_path, caplog):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    (out / "clean_manifest.json").unlink()

    with caplog.at_level(logging.WARNING):
        second = _run(input_dir, out, resume=True)

    assert "no usable clean_manifest.json" in caplog.text
    assert second["existing"] == 0
    assert second["retained"] == 2
    assert list((out / "cleaned_images").glob("*_1.jpg"))


def test_resume_cumulative_index_survives_subset_runs(tmp_path):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)

    backup = tmp_path / "img1.png"
    (input_dir / "img1.png").rename(backup)
    _run(input_dir, out, resume=True)
    backup.rename(input_dir / "img1.png")
    _run(input_dir, out, resume=True)

    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["img1.png"]["status"] == "existing"
    assert not list((out / "cleaned_images").glob("*_1.jpg"))
    indexed = {row["input_path"] for row in _manifest(out)["index"]}
    assert indexed == {"img0.png", "img1.png"}


def test_resume_transitive_group_member_matches_by_own_hash(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    _write(input_dir / "a.png", (10, 10, 10))
    _write(input_dir / "b.png", (60, 60, 60))
    _write(input_dir / "c.png", (120, 120, 120))
    values = {"a.png": 0, "b.png": 1, "c.png": 2}

    class FakeHash:
        def __init__(self, value):
            self.hash = np.array(
                [int(bit) for bit in f"{value:064b}"], dtype=np.uint8
            )

    def fake_compute_phash(img):
        return FakeHash(values[Path(img.filename).name])

    monkeypatch.setattr(
        "src.cleaning.processor.compute_phash", fake_compute_phash
    )
    cleaner = ImageCleaner(
        str(input_dir),
        str(out / "cleaned_images"),
        dedup_mode="phash",
        phash_threshold=1,
        threads=1,
    )
    cleaner.process_directory(log_path=str(out / "log.txt"))

    index = {row["input_path"]: row for row in _manifest(out)["index"]}
    assert index["a.png"]["status"] == "retained"
    assert index["b.png"]["status"] == "dropped"
    assert index["c.png"]["status"] == "dropped"

    input_b = tmp_path / "input_b"
    _write(input_b / "b.png", (60, 60, 60))
    again = ImageCleaner(
        str(input_b),
        str(out / "cleaned_images"),
        dedup_mode="phash",
        phash_threshold=1,
        threads=1,
        resume=True,
    )
    result = again.process_directory(log_path=str(out / "log.txt"))
    assert result["existing"] == 1
    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["b.png"]["status"] == "existing"
    assert not list((out / "cleaned_images").glob("b_*.jpg"))


def test_resume_none_without_manifest_does_not_match_mirrored_path(
    tmp_path, caplog
):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    _write(input_dir / "same.jpg", (10, 20, 30))
    _run(input_dir, out, mode="none")
    (out / "clean_manifest.json").unlink()
    (input_dir / "same.jpg").unlink()
    _write(input_dir / "same.png", (200, 10, 10))

    with caplog.at_level(logging.WARNING):
        result = _run(input_dir, out, resume=True, mode="none")

    assert result["retained"] == 1
    assert result["existing"] == 0
    assert "could not match same.png" in caplog.text
    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["same.png"]["status"] == "retained"


def test_resume_fallback_hashes_output_without_current_mode_hash(
    tmp_path, monkeypatch
):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out, mode="md5")

    class FakeHash:
        def __init__(self, value):
            self.hash = np.array(
                [int(bit) for bit in f"{value:064b}"], dtype=np.uint8
            )

    monkeypatch.setattr(
        "src.cleaning.processor.compute_phash", lambda _img: FakeHash(0)
    )
    second = _run(input_dir, out, resume=True, mode="phash")
    assert second["existing"] == 2


def test_resume_ignores_dry_run_manifest_files(tmp_path):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    manifest = _manifest(out)
    manifest["dry_run"] = True
    manifest["index"] = []
    (out / "clean_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    second = _run(input_dir, out, resume=True)
    assert second["existing"] == 0
    assert second["retained"] == 2


def test_resume_ignores_unsupported_manifest(tmp_path, caplog):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    manifest = _manifest(out)
    manifest["schema_version"] = 999
    (out / "clean_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING):
        _run(input_dir, out, resume=True)
    assert "unsupported schema_version" in caplog.text


def test_resume_rejects_manifest_output_path_outside_out_dir(
    tmp_path, caplog
):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    escape = tmp_path / "escape.jpg"
    Image.new("RGB", (200, 140), (40, 80, 120)).save(escape)
    manifest = _manifest(out)
    manifest["index"] = [
        {
            "input_path": "img0.png",
            "output_path": "../escape.jpg",
            "status": "retained",
            "md5": manifest["index"][0]["md5"],
            "phash": "",
        }
    ]
    (out / "clean_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for path in (out / "cleaned_images").glob("*.jpg"):
        path.unlink()

    with caplog.at_level(logging.WARNING):
        result = _run(input_dir, out, resume=True)
    assert "unsafe path" in caplog.text
    assert result["retained"] == 2


def test_existing_md5_match_has_empty_distance(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    _write(input_dir / "x.png", (50, 60, 70))
    cleaner = ImageCleaner(
        str(input_dir),
        str(out / "cleaned_images"),
        dedup_mode="md5+phash",
        threads=1,
    )
    cleaner.process_directory(log_path=str(out / "log.txt"))
    cleaner = ImageCleaner(
        str(input_dir),
        str(out / "cleaned_images"),
        dedup_mode="md5+phash",
        threads=1,
        resume=True,
    )
    cleaner.process_directory(log_path=str(out / "log.txt"))
    row = _rows(out)[0]
    assert row["status"] == "existing"
    assert row["distance"] == ""


def test_dry_run_preserves_previous_index(tmp_path):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    before = _manifest(out)["index"]

    cleaner = ImageCleaner(
        str(input_dir),
        str(out / "cleaned_images"),
        dedup_mode="md5",
        threads=1,
        dry_run=True,
        resume=True,
    )
    cleaner.process_directory(log_path=str(out / "log.txt"))
    after = _manifest(out)["index"]
    assert after == before

    _run(input_dir, out, resume=True)
    assert not list((out / "cleaned_images").glob("*_1.jpg"))


def test_resume_rejects_manifest_output_path_outside_cleaned_images(
    tmp_path, caplog
):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    other = out / "other.jpg"
    Image.new("RGB", (200, 140), (40, 80, 120)).save(other)
    manifest = _manifest(out)
    manifest["index"] = [
        {
            "input_path": "img0.png",
            "output_path": "other.jpg",
            "status": "retained",
            "md5": manifest["index"][0]["md5"],
            "phash": "",
        }
    ]
    (out / "clean_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for path in (out / "cleaned_images").glob("*.jpg"):
        path.unlink()

    with caplog.at_level(logging.WARNING):
        result = _run(input_dir, out, resume=True)
    assert "unsafe path" in caplog.text
    assert result["retained"] == 2


def test_resume_prefers_index_row_over_scanned_output(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    _write(input_dir / "e.png", (0, 0, 0))
    values = {"e.png": 0, "e.jpg": 0, "other.jpg": 2, "new.png": 2}

    class FakeHash:
        def __init__(self, value):
            self.hash = np.array(
                [int(bit) for bit in f"{value:064b}"], dtype=np.uint8
            )

    monkeypatch.setattr(
        "src.cleaning.processor.compute_phash",
        lambda img: FakeHash(values[Path(img.filename).name]),
    )
    ImageCleaner(
        str(input_dir),
        str(out / "cleaned_images"),
        dedup_mode="phash",
        phash_threshold=3,
        threads=1,
    ).process_directory(log_path=str(out / "log.txt"))

    # A scanned output that is closer to the new input than the index row.
    _write(out / "cleaned_images" / "other.jpg", (20, 20, 20))
    second_input = tmp_path / "input2"
    _write(second_input / "new.png", (10, 10, 10))
    ImageCleaner(
        str(second_input),
        str(out / "cleaned_images"),
        dedup_mode="phash",
        phash_threshold=3,
        threads=1,
        resume=True,
    ).process_directory(log_path=str(out / "log.txt"))

    row = _rows(out)[0]
    assert row["status"] == "existing"
    assert row["output_path"] == "cleaned_images/e.jpg"


def test_resume_falls_back_to_output_hash_for_indexed_output(tmp_path):
    input_dir = tmp_path / "input"
    out = tmp_path / "out"
    input_dir.mkdir()
    Image.effect_noise((160, 120), 40).convert("RGB").save(input_dir / "a.png")
    _run(input_dir, out)

    with Image.open(out / "cleaned_images" / "a.jpg") as img:
        img.load()
        decoded = img.copy()
    second_input = tmp_path / "input2"
    second_input.mkdir()
    decoded.save(second_input / "b.png")

    second = _run(second_input, out, resume=True)
    assert second["existing"] == 1
    assert second["retained"] == 0
    assert not list((out / "cleaned_images").glob("*_1.jpg"))


def test_resume_rejects_truncated_manifest_output(tmp_path, caplog):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    target = out / "cleaned_images" / "img0.jpg"
    raw = target.read_bytes()
    target.write_bytes(raw[: len(raw) // 2])

    with caplog.at_level(logging.WARNING):
        result = _run(input_dir, out, resume=True)

    assert "Ignoring missing output" in caplog.text
    assert result["existing"] == 1
    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["img0.png"]["status"] == "retained"
    assert rows["img1.png"]["status"] == "existing"


def test_resume_rejects_unreadable_manifest_output(tmp_path, caplog):
    input_dir, out = _fixture(tmp_path)
    _run(input_dir, out)
    (out / "cleaned_images" / "img0.jpg").write_bytes(b"not-an-image")

    with caplog.at_level(logging.WARNING):
        result = _run(input_dir, out, resume=True)

    assert "Ignoring missing output" in caplog.text
    assert result["retained"] == 1
    assert result["existing"] == 1
    rows = {row["input_path"]: row for row in _rows(out)}
    assert rows["img0.png"]["status"] == "retained"
    assert rows["img1.png"]["status"] == "existing"
