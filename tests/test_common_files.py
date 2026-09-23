from pathlib import Path

import pytest

from src.common.files import iter_files, relative_output_path


def test_iter_files_is_recursive_sorted_and_case_insensitive(tmp_path: Path):
    root = tmp_path / "input"
    (root / "z").mkdir(parents=True)
    (root / "a").mkdir()
    (root / "z" / "b.JPG").write_bytes(b"")
    (root / "a" / "c.png").write_bytes(b"")
    (root / "ignore.txt").write_bytes(b"")
    assert [p.relative_to(root).as_posix() for p in iter_files(root, {".jpg", ".png"})] == [
        "a/c.png",
        "z/b.JPG",
    ]


def test_relative_output_path_preserves_parent_directories(tmp_path: Path):
    root = tmp_path / "input"
    source = root / "beetles" / "same.jpg"
    assert relative_output_path(source, root, tmp_path / "output") == (
        tmp_path / "output" / "beetles" / "same.jpg"
    )


def test_iter_files_rejects_file_root(tmp_path: Path):
    root_file = tmp_path / "input.txt"
    root_file.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        iter_files(root_file, {".txt"})


def test_iter_files_rejects_missing_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        iter_files(tmp_path / "missing", {".jpg"})


def test_relative_output_path_rejects_outside_source(tmp_path: Path):
    root = tmp_path / "input"
    outside = tmp_path / "elsewhere" / "same.jpg"
    with pytest.raises(ValueError):
        relative_output_path(outside, root, tmp_path / "output")
