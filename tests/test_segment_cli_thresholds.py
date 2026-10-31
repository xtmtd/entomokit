"""Tests for segment's model threshold and post-filter split."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.segmentation.processor import SegmentationProcessor


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=(100, 120, 140)).save(path)


def _parser():
    from entomokit.main import _build_parser

    return _build_parser()


def test_segment_parser_threshold_defaults_and_ranges():
    parser = _parser()
    args = parser.parse_args(
        ["segment", "--input-dir", "in", "--out-dir", "out"]
    )
    assert args.sam3_confidence_threshold == 0.5
    assert args.post_confidence_threshold == 0.0
    assert args.confidence_threshold == 0.0

    for flag in (
        "--sam3-confidence-threshold",
        "--post-confidence-threshold",
        "--confidence-threshold",
    ):
        for bad in ("-0.01", "1.01"):
            with pytest.raises(SystemExit):
                parser.parse_args(
                    ["segment", "--input-dir", "in", "--out-dir", "out", flag, bad]
                )


def test_processor_passes_sam3_threshold_to_wrapper():
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        SegmentationProcessor(
            "fake.pt",
            device="cpu",
            segmentation_method="sam3",
            sam3_confidence_threshold=0.2,
        )
    mock_sam.assert_called_with(
        "fake.pt", device="cpu", confidence_threshold=0.2
    )


def test_post_filter_uses_post_confidence_threshold():
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        wrapper = MagicMock()
        wrapper.predict_with_scores.return_value = {
            "masks": [np.ones((10, 10), dtype=np.uint8)] * 2,
            "scores": [0.95, 0.45],
        }
        mock_sam.return_value = wrapper
        processor = SegmentationProcessor(
            "fake.pt",
            device="cpu",
            segmentation_method="sam3",
            confidence_threshold=0.7,
        )

    masks, scores = processor._segment_and_filter(
        np.ones((10, 10, 3), dtype=np.uint8)
    )
    assert len(masks) == 1
    assert scores == [0.95]


def test_default_thresholds_do_not_warn(caplog):
    with (
        patch("src.segmentation.processor.SAM3Wrapper"),
        patch("pathlib.Path.exists", return_value=True),
        caplog.at_level(logging.WARNING),
    ):
        SegmentationProcessor(
            "fake.pt", device="cpu", segmentation_method="sam3"
        )
    assert "no additional effect" not in caplog.text


def test_post_threshold_below_sam3_threshold_warns(caplog):
    with (
        patch("src.segmentation.processor.SAM3Wrapper"),
        patch("pathlib.Path.exists", return_value=True),
        caplog.at_level(logging.WARNING),
    ):
        SegmentationProcessor(
            "fake.pt",
            device="cpu",
            segmentation_method="sam3",
            sam3_confidence_threshold=0.5,
            post_confidence_threshold=0.3,
        )
    assert "no additional effect" in caplog.text


def test_no_mask_reason_names_both_thresholds(tmp_path, caplog):
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
        caplog.at_level(logging.WARNING),
    ):
        wrapper = MagicMock()
        wrapper.predict_with_scores.return_value = {"masks": [], "scores": []}
        mock_sam.return_value = wrapper
        processor = SegmentationProcessor(
            "fake.pt",
            device="cpu",
            segmentation_method="sam3",
            sam3_confidence_threshold=0.2,
        )
        processor.process_image(
            image=np.ones((20, 20, 3), dtype=np.uint8),
            output_dir=tmp_path,
            base_name="test",
        )
    assert "sam3=0.2" in caplog.text
    assert "post=0.0" in caplog.text


def test_otsu_ignores_sam3_threshold():
    image = np.full((40, 40, 3), 127, dtype=np.uint8)
    masks_by_threshold = []
    for threshold in (0.5, 0.2):
        processor = SegmentationProcessor(
            "fake.pt",
            device="cpu",
            segmentation_method="otsu",
            sam3_confidence_threshold=threshold,
        )
        masks, _scores = processor._segment_and_filter(image)
        masks_by_threshold.append(
            [mask.astype(np.uint8).tolist() for mask in masks]
        )
    assert masks_by_threshold[0] == masks_by_threshold[1]


def _run_segment_cli(tmp_path, monkeypatch, *extra):
    input_dir = tmp_path / "input"
    _write(input_dir / "a.png")
    out_dir = tmp_path / "out"
    args = _parser().parse_args(
        [
            "segment",
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            "--segmentation-method",
            "otsu",
            *extra,
        ]
    )
    captured = {}

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def process_directory(self, **kwargs):
            return {
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "output_files": [],
            }

    monkeypatch.setattr(
        "src.segmentation.processor.SegmentationProcessor", FakeProcessor
    )
    from src.common import cli as cli_common
    from entomokit import segment

    try:
        segment.run(args)
    finally:
        cli_common._disable_output_capture()
    return captured


def test_legacy_confidence_threshold_is_used_when_post_is_zero(
    tmp_path, monkeypatch, caplog
):
    with caplog.at_level(logging.INFO):
        captured = _run_segment_cli(
            tmp_path, monkeypatch, "--confidence-threshold", "0.7"
        )
    assert captured["post_confidence_threshold"] == 0.7
    assert "0.0 (unset)" in caplog.text


def test_post_wins_when_both_are_nonzero(tmp_path, monkeypatch):
    captured = _run_segment_cli(
        tmp_path,
        monkeypatch,
        "--post-confidence-threshold",
        "0.6",
        "--confidence-threshold",
        "0.7",
    )
    assert captured["post_confidence_threshold"] == 0.6
