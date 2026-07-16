"""Tests for clean --recursive flag."""

import pytest
from pathlib import Path
from src.cleaning.processor import ImageCleaner


def test_recursive_finds_images_in_subdirs(tmp_path):
    """Recursive mode collects images from nested directories."""
    sub = tmp_path / "input" / "subdir"
    sub.mkdir(parents=True)
    from PIL import Image

    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(sub / "test.jpg")

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cleaner = ImageCleaner(
        input_dir=str(tmp_path / "input"),
        output_dir=str(out_dir),
        dedup_mode="none",
    )
    results = cleaner.process_directory(
        log_path=str(tmp_path / "log.txt"), recursive=True
    )
    assert results["processed"] == 1


def test_non_recursive_misses_subdir_images(tmp_path):
    """Non-recursive mode should NOT pick up images in subdirs."""
    sub = tmp_path / "input" / "subdir"
    sub.mkdir(parents=True)
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    img.save(sub / "test.jpg")

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    cleaner = ImageCleaner(
        input_dir=str(tmp_path / "input"),
        output_dir=str(out_dir),
        dedup_mode="none",
    )
    results = cleaner.process_directory(
        log_path=str(tmp_path / "log.txt"), recursive=False
    )
    assert results["processed"] == 0


def test_clean_results_count_invalid_images_as_errors(tmp_path):
    from PIL import Image

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)

    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(input_dir / "ok.jpg")
    (input_dir / "bad.jpg").write_bytes(b"not-an-image")

    cleaner = ImageCleaner(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        dedup_mode="none",
        threads=1,
    )
    results = cleaner.process_directory(
        log_path=str(tmp_path / "log.txt"), recursive=False
    )

    assert results["total"] == 2
    assert results["processed"] == 1
    assert results["errors"] == 1


def test_phash_reservation_is_removed_when_saving_fails(tmp_path, monkeypatch):
    from PIL import Image

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(input_dir / "one.png")
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(input_dir / "two.png")

    original_save = Image.Image.save
    calls = 0

    def fail_first_save(self, fp, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk full")
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_first_save)
    cleaner = ImageCleaner(str(input_dir), str(output_dir), dedup_mode="phash", threads=1)
    results = cleaner.process_directory(log_path=str(tmp_path / "log.txt"))

    assert results["errors"] == 1
    assert results["processed"] == 1
