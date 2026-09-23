"""Tests for clean's fixed recursive + mirrored-output contract."""

from pathlib import Path

import pytest
from PIL import Image

from src.cleaning.processor import ImageCleaner


def _write(path: Path, color=(255, 0, 0), size=(10, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def test_clean_always_scans_recursively(tmp_path):
    _write(tmp_path / "input" / "subdir" / "test.jpg")

    cleaner = ImageCleaner(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        dedup_mode="none",
    )
    results = cleaner.process_directory(log_path=str(tmp_path / "log.txt"))
    assert results["processed"] == 1
    assert (tmp_path / "output" / "subdir" / "test.jpg").exists()


def test_same_named_images_in_different_subdirs_do_not_overwrite(tmp_path):
    _write(tmp_path / "input" / "a" / "same.jpg", color=(255, 0, 0))
    _write(tmp_path / "input" / "b" / "same.jpg", color=(0, 0, 255))

    out = tmp_path / "output"
    cleaner = ImageCleaner(
        input_dir=str(tmp_path / "input"), output_dir=str(out), dedup_mode="none"
    )
    results = cleaner.process_directory(log_path=str(tmp_path / "log.txt"))

    assert results["processed"] == 2
    assert (out / "a" / "same.jpg").exists()
    assert (out / "b" / "same.jpg").exists()
    assert (out / "a" / "same.jpg").read_bytes() != (out / "b" / "same.jpg").read_bytes()


def test_clean_parser_has_no_recursive_or_flatten() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["clean", "--input-dir", "in", "--out-dir", "out", "--recursive"])
    with pytest.raises(SystemExit):
        parser.parse_args(["clean", "--input-dir", "in", "--out-dir", "out", "--flatten"])


def test_clean_results_count_invalid_images_as_errors(tmp_path):
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
    results = cleaner.process_directory(log_path=str(tmp_path / "log.txt"))

    assert results["total"] == 2
    assert results["processed"] == 1
    assert results["errors"] == 1


def test_phash_reservation_is_removed_when_saving_fails(tmp_path, monkeypatch):
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
