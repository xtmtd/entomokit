"""Tests for recursive directory scanning in segment, embed, measure, predict."""

import pytest
from pathlib import Path
from PIL import Image


def _make_nested_images(tmp_path, n=3):
    """Create nested structure: input/{species}/{split}/{img}.jpg"""
    for i in range(n):
        sp_dir = tmp_path / "input" / f"species_{i}" / "train"
        sp_dir.mkdir(parents=True)
        Image.new("RGB", (10, 10), color=(255, 0, 0)).save(sp_dir / f"img_{i}.jpg")
    return tmp_path / "input"


def test_segment_recursive_finds_nested_images(tmp_path):
    """segment --recursive should find images in nested dirs."""
    from src.segmentation.processor import SegmentationProcessor

    input_dir = _make_nested_images(tmp_path)
    out_dir = tmp_path / "output"

    processor = SegmentationProcessor(
        segmentation_method="otsu", sam3_checkpoint="dummy.pt"
    )
    results = processor.process_directory(
        input_dir=str(input_dir),
        output_dir=out_dir,
        recursive=True,
    )
    assert results["processed"] == 3


def test_segment_non_recursive_misses_nested(tmp_path):
    """segment without --recursive should NOT find nested images."""
    from src.segmentation.processor import SegmentationProcessor

    input_dir = _make_nested_images(tmp_path)
    out_dir = tmp_path / "output"

    processor = SegmentationProcessor(
        segmentation_method="otsu", sam3_checkpoint="dummy.pt"
    )
    results = processor.process_directory(
        input_dir=str(input_dir),
        output_dir=out_dir,
        recursive=False,
    )
    assert results["processed"] == 0


def test_embed_timm_recursive_finds_nested(tmp_path):
    """embed timm --recursive should find nested images."""
    from src.classification.embedder import extract_embeddings_timm

    input_dir = _make_nested_images(tmp_path)
    df = extract_embeddings_timm(
        images_dir=input_dir,
        base_model="convnextv2_femto",
        batch_size=2,
        num_workers=0,
        device="cpu",
        recursive=True,
    )
    assert len(df) == 3


def test_measure_recursive_finds_nested(tmp_path):
    """measure --recursive should find nested masks."""
    from src.measurement.io import iter_mask_files

    mask_dir = tmp_path / "masks"
    for i in range(3):
        sub = mask_dir / f"species_{i}" / "train"
        sub.mkdir(parents=True)
        Image.new("L", (10, 10), color=255).save(sub / f"mask_{i}.png")

    files = iter_mask_files(mask_dir, recursive=True)
    assert len(files) == 3
