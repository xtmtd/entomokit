"""Tests for recursive extract-frames input scanning and output mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.framing.extractor as extractor_module
from src.framing.extractor import VideoFrameExtractor


class _FakeCapture:
    def __init__(self, _path):
        self._frame = 0

    def isOpened(self):
        return True

    def set(self, _prop, _value):
        return True

    def get(self, prop):
        if prop == _FakeCV2.CAP_PROP_FPS:
            return 10.0
        if prop == _FakeCV2.CAP_PROP_FRAME_COUNT:
            return 20
        return 0

    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        return None


class _FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_FRAMES = 1

    VideoCapture = _FakeCapture

    @staticmethod
    def imwrite(path, _frame):
        Path(path).write_bytes(b"frame")
        return True


@pytest.fixture
def fake_cv2(monkeypatch):
    monkeypatch.setattr(extractor_module, "cv2", _FakeCV2)
    return _FakeCV2


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_nested_same_named_videos_get_distinct_frame_trees(tmp_path, fake_cv2):
    input_dir = tmp_path / "videos"
    _touch(input_dir / "nested" / "beetles" / "movie.mp4")
    _touch(input_dir / "nested" / "moths" / "movie.mp4")

    out_dir = tmp_path / "frames"
    extractor = VideoFrameExtractor(str(input_dir), str(out_dir), interval_ms=1000)
    videos = extractor.get_video_files()
    assert [v.relative_to(input_dir).as_posix() for v in videos] == [
        "nested/beetles/movie.mp4",
        "nested/moths/movie.mp4",
    ]

    for video in videos:
        extractor.extract_from_video(video)

    assert (out_dir / "nested" / "beetles" / "movie" / "movie_01.jpg").exists()
    assert (out_dir / "nested" / "moths" / "movie" / "movie_01.jpg").exists()


def test_single_file_input_maps_to_flat_frame_dir(tmp_path, fake_cv2):
    video = tmp_path / "clip.mp4"
    _touch(video)

    out_dir = tmp_path / "frames"
    extractor = VideoFrameExtractor(str(tmp_path), str(out_dir), interval_ms=1000)
    extractor._single_file_filter = video
    assert extractor.get_video_files() == [video]

    extractor.extract_from_video(video)
    assert (out_dir / "clip" / "clip_01.jpg").exists()


def test_single_file_filter_matches_only_the_exact_path(tmp_path, fake_cv2):
    """Same-named videos elsewhere under the input root must not match."""
    root = tmp_path / "videos"
    _touch(root / "a" / "movie.mp4")
    _touch(root / "b" / "movie.mp4")

    extractor = VideoFrameExtractor(str(root), str(tmp_path / "frames"), interval_ms=1000)
    extractor._single_file_filter = root / "a" / "movie.mp4"
    assert extractor.get_video_files() == [root / "a" / "movie.mp4"]
