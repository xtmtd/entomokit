"""Cross-cutting resume parameter guards (measure, extract-frames)."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest


def test_check_resume_params_accepts_same_and_rejects_different(tmp_path):
    from src.common.resume import check_resume_params

    check_resume_params(tmp_path, "measure", {"pixel_size_um": 1.0})
    # Same parameters may resume repeatedly.
    check_resume_params(tmp_path, "measure", {"pixel_size_um": 1.0})
    with pytest.raises(ValueError, match="pixel_size_um"):
        check_resume_params(tmp_path, "measure", {"pixel_size_um": 2.0})


def _measure_args(tmp_path, out_dir, **kw):
    d = dict(
        mask_dir=str(tmp_path / "masks"),
        out_dir=str(out_dir),
        pixel_size_um=1.0,
        verbose=False,
        resume=True,
        overwrite=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_measure_resume_rejects_pixel_size_change(tmp_path):
    from entomokit import measure

    (tmp_path / "masks").mkdir()
    out = tmp_path / "out"
    summary = {"total": 0, "ok": 0, "warn": 0, "fail": 0}
    with patch("src.measurement.service.run_batch", return_value=summary):
        measure.run(_measure_args(tmp_path, out))
        measure.run(_measure_args(tmp_path, out))
        with pytest.raises(ValueError, match="pixel_size_um"):
            measure.run(_measure_args(tmp_path, out, pixel_size_um=2.0))


def _ef_args(tmp_path, out_dir, **kw):
    d = dict(
        input_dir=str(tmp_path / "videos"),
        out_dir=str(out_dir),
        out_image_format="jpg",
        threads=1,
        max_frames=None,
        start_time=0.0,
        end_time=None,
        interval=1000,
        resume=True,
        overwrite=False,
        verbose=False,
        quiet=True,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_extract_frames_resume_rejects_interval_change(tmp_path):
    from entomokit import extract_frames

    (tmp_path / "videos").mkdir()
    out = tmp_path / "out"
    stats = {
        "total_videos": 0,
        "total_frames": 0,
        "errors": 0,
        "processing_time": 0.0,
        "skipped_frames": 0,
    }
    with patch(
        "src.framing.extractor.VideoFrameExtractor.extract_all", return_value=stats
    ):
        extract_frames.run(_ef_args(tmp_path, out))
        extract_frames.run(_ef_args(tmp_path, out))
        with pytest.raises(ValueError, match="interval_ms"):
            extract_frames.run(_ef_args(tmp_path, out, interval=500))


def test_measure_rejects_pixel_size_change_after_normal_run(tmp_path):
    from entomokit import measure

    (tmp_path / "masks").mkdir()
    out = tmp_path / "out"
    summary = {"total": 0, "ok": 0, "warn": 0, "fail": 0}
    with patch("src.measurement.service.run_batch", return_value=summary):
        measure.run(_measure_args(tmp_path, out, resume=False))
        with pytest.raises(ValueError, match="pixel_size_um"):
            measure.run(_measure_args(tmp_path, out, pixel_size_um=2.0))


def test_extract_frames_rejects_interval_change_after_normal_run(tmp_path):
    from entomokit import extract_frames

    (tmp_path / "videos").mkdir()
    out = tmp_path / "out"
    stats = {
        "total_videos": 0,
        "total_frames": 0,
        "errors": 0,
        "processing_time": 0.0,
        "skipped_frames": 0,
    }
    with patch(
        "src.framing.extractor.VideoFrameExtractor.extract_all", return_value=stats
    ):
        extract_frames.run(_ef_args(tmp_path, out, resume=False))
        with pytest.raises(ValueError, match="interval_ms"):
            extract_frames.run(_ef_args(tmp_path, out, interval=500))
