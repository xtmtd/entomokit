"""Integration tests for --resume / --overwrite across processing commands."""
import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ── segment ──────────────────────────────────────────────────────────────────

def _segment_args(**kw):
    d = dict(
        input_dir="/nonexistent", out_dir="/tmp/seg_out",
        threads=8,
        segmentation_method="otsu", confidence_threshold=0.3,
        min_area_ratio=0.01, max_area_ratio=0.9, sam3_checkpoint=None,
        lama_model=None, coco_output_mode="single", output_format="png",
        num_workers=1, verbose=False, resume=False, overwrite=False,
        annotation_output_format=None, iou_threshold=0.1, hint="insect",
        padding_ratio=0.0, repair_strategy=None, device="auto",
        annotation_format=None, coco_bbox_format="xywh",
        lama_mask_dilate=0, out_image_format="png",
        recursive=False, flatten=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_segment_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "dummy.png").write_bytes(b"")
    from entomokit import segment
    with pytest.raises(SystemExit):
        segment.run(_segment_args(out_dir=str(out), input_dir=str(tmp_path)))


def test_segment_overwrite_clears(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "dummy.png").write_bytes(b"")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    with patch("src.segmentation.processor.SegmentationProcessor") as MockP:
        MockP.return_value.process_directory.return_value = {
            "processed": 0, "failed": 0, "output_files": []
        }
        from entomokit import segment
        segment.run(_segment_args(out_dir=str(out), input_dir=str(in_dir), overwrite=True))
    assert not (out / "dummy.png").exists()


# ── synthesize ───────────────────────────────────────────────────────────────

def _synthesize_args(**kw):
    d = dict(
        target_dir="/nonexistent", background_dir="/nonexistent",
        out_dir="/tmp/syn_out", num_syntheses=10,
        area_ratio_min=0.1, area_ratio_max=0.5,
        color_match_strength=0.3, avoid_black_regions=True,
        rotate=True, out_image_format="png",
        annotation_output_format="none", coco_output_mode="single",
        threads=1, verbose=False, resume=False, overwrite=False,
        coco_bbox_format="xywh", rotate_degrees=0.0,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_synthesize_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.png").write_bytes(b"")
    from entomokit import synthesize
    with pytest.raises(SystemExit):
        synthesize.run(_synthesize_args(out_dir=str(out)))


# ── augment ──────────────────────────────────────────────────────────────────

def _augment_args(**kw):
    d = dict(
        input_dir="/nonexistent", out_dir="/tmp/aug_out",
        preset="light", policy=None, seed=42, multiply=2,
        verbose=False, resume=False, overwrite=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_augment_exits_on_nonempty(tmp_path):
    pytest.importorskip("albumentations")
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.png").write_bytes(b"")
    from entomokit import augment
    with pytest.raises(SystemExit):
        augment.run(_augment_args(out_dir=str(out)))


# ── clean ────────────────────────────────────────────────────────────────────

def _clean_args(**kw):
    d = dict(
        input_dir="/nonexistent", out_dir="/tmp/clean_out",
        out_short_size=None, out_image_format="jpg", threads=4,
        keep_exif=False, dedup_mode="phash", phash_threshold=5,
        recursive=False, verbose=False, resume=False, overwrite=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_clean_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.jpg").write_bytes(b"")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    from entomokit import clean
    with pytest.raises(SystemExit):
        clean.run(_clean_args(out_dir=str(out), input_dir=str(in_dir)))


# ── measure ──────────────────────────────────────────────────────────────────

def _measure_args(**kw):
    d = dict(
        mask_dir="/nonexistent", out_dir="/tmp/meas_out",
        pixel_size_um=None, verbose=False,
        resume=False, overwrite=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_measure_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "metrics.csv").write_text("file_name\na.png\n")
    from entomokit import measure
    with pytest.raises(SystemExit):
        measure.run(_measure_args(out_dir=str(out)))


# ── extract-frames ───────────────────────────────────────────────────────────

def _ef_args(**kw):
    d = dict(
        input_dir="/nonexistent", out_dir="/tmp/ef_out",
        out_image_format="jpg", threads=1, max_frames=None,
        start_time=0.0, end_time=None, interval=1000,
        resume=False, overwrite=False, verbose=False, quiet=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_ef_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "frame_001.jpg").write_bytes(b"")
    from entomokit import extract_frames
    with pytest.raises(SystemExit):
        extract_frames.run(_ef_args(out_dir=str(out)))


def test_ef_no_skip_existing_in_help():
    result = subprocess.run(
        [sys.executable, "-m", "entomokit.main", "extract-frames", "--help"],
        capture_output=True, text=True,
    )
    assert "--skip-existing" not in result.stdout
    assert "--resume" in result.stdout


# ── split-csv ────────────────────────────────────────────────────────────────

def _split_csv_args(**kw):
    d = dict(
        raw_image_csv="/nonexistent",
        out_dir="/tmp/split_out",
        mode="ratio",
        unknown_test_sample_ratio=0.0,
        known_test_sample_ratio=0.1,
        unknown_test_sample_count=0,
        known_test_sample_count=0,
        val_ratio=0.0,
        val_count=0,
        min_count_per_class=0,
        max_count_per_class=None,
        seed=42,
        copy_images=False,
        images_dir=None,
        verbose=False,
        overwrite=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_split_csv_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "train.csv").write_text("image,label\n")
    # Must provide a valid CSV so the guard fires before the missing-file check
    csv = tmp_path / "data.csv"
    csv.write_text("image,label\nimg.png,cat\n")
    from entomokit import split_csv
    with pytest.raises(SystemExit):
        split_csv.run(_split_csv_args(out_dir=str(out), raw_image_csv=str(csv)))


def test_split_csv_overwrite_clears(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "train.csv").write_text("image,label\n")
    csv = tmp_path / "data.csv"
    csv.write_text("image,label\nimg.png,cat\n")
    from entomokit import split_csv
    with patch("src.splitting.splitter.DatasetSplitter") as MockS:
        MockS.return_value.split.return_value = {
            "train": 1, "val": 0, "test_known": 0, "test_unknown": 0,
        }
        split_csv.run(_split_csv_args(
            out_dir=str(out), raw_image_csv=str(csv), overwrite=True,
        ))
    assert not (out / "train.csv").exists()


# ── classify train ───────────────────────────────────────────────────────────

def _train_args(**kw):
    d = dict(
        train_csv="/nonexistent",
        images_dir="/nonexistent",
        base_model="convnextv2_femto",
        out_dir="/tmp/train_out",
        augment="medium",
        max_epochs=50,
        time_limit=1.0,
        resume=False,
        overwrite=False,
        learning_rate=1e-4,
        weight_decay=1e-3,
        warmup_steps=0.1,
        patience=10,
        top_k=3,
        focal_loss=False,
        focal_loss_gamma=1.0,
        device="auto",
        batch_size=32,
        num_workers=4,
        num_threads=0,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_train_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "dummy.txt").write_text("")
    from entomokit.classify import train
    with pytest.raises(SystemExit):
        train.run(_train_args(out_dir=str(out)))


def test_train_resume_allows_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "logs").mkdir()
    from entomokit.classify import train
    with patch("src.classification.trainer.train") as mock_train:
        mock_train.return_value = out
        train.run(_train_args(
            out_dir=str(out), resume=True,
            train_csv=str(tmp_path / "t.csv"),
            images_dir=str(tmp_path),
        ))
    mock_train.assert_called_once()


def test_train_resume_overwrite_mutually_exclusive():
    result = subprocess.run(
        [sys.executable, "-m", "entomokit.main", "classify", "train",
         "--resume", "--overwrite", "--train-csv", "x.csv",
         "--images-dir", "/tmp", "--out-dir", "/tmp/out"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


# ── classify predict ─────────────────────────────────────────────────────────

def _predict_args(**kw):
    d = dict(
        input_csv=None,
        images_dir="/nonexistent",
        model_dir=None,
        onnx_model=None,
        out_dir="/tmp/pred_out",
        batch_size=32,
        num_workers=4,
        num_threads=0,
        overwrite=False,
        device="auto",
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_predict_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "predictions").mkdir()
    from entomokit.classify import predict
    with pytest.raises(SystemExit):
        predict.run(_predict_args(out_dir=str(out)))


# ── classify evaluate ────────────────────────────────────────────────────────

def _evaluate_args(**kw):
    d = dict(
        test_csv="/nonexistent",
        images_dir="/nonexistent",
        model_dir=None,
        onnx_model=None,
        out_dir="/tmp/eval_out",
        batch_size=32,
        num_workers=4,
        num_threads=0,
        overwrite=False,
        device="auto",
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_evaluate_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "evaluations.csv").write_text("")
    from entomokit.classify import evaluate
    with pytest.raises(SystemExit):
        evaluate.run(_evaluate_args(out_dir=str(out)))


# ── classify embed ───────────────────────────────────────────────────────────

def _embed_args(**kw):
    d = dict(
        images_dir="/nonexistent",
        out_dir="/tmp/embed_out",
        base_model="convnextv2_femto",
        model_dir=None,
        label_csv=None,
        visualize=False,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric="euclidean",
        umap_seed=42,
        batch_size=32,
        num_workers=4,
        num_threads=0,
        overwrite=False,
        device="auto",
        metrics_sample_size=10000,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_embed_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "logs").mkdir()
    from entomokit.classify import embed
    with pytest.raises(SystemExit):
        embed.run(_embed_args(out_dir=str(out)))


# ── classify cam ─────────────────────────────────────────────────────────────

def _cam_args(**kw):
    d = dict(
        label_csv=None,
        images_dir="/nonexistent",
        out_dir="/tmp/cam_out",
        model_dir=None,
        base_model="convnextv2_femto",
        checkpoint_path=None,
        num_classes=None,
        no_pretrained=False,
        cam_method="gradcam",
        arch=None,
        target_layer_name=None,
        image_weight=0.5,
        fig_format="png",
        save_npy=False,
        dump_model_structure=False,
        max_images=None,
        cam_batch_size=32,
        num_workers=4,
        num_threads=0,
        overwrite=False,
        device="auto",
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_cam_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "figures").mkdir()
    from entomokit.classify import cam
    with pytest.raises(SystemExit):
        cam.run(_cam_args(out_dir=str(out)))


# ── classify export-onnx ─────────────────────────────────────────────────────

def _export_onnx_args(**kw):
    d = dict(
        model_dir="/nonexistent",
        out_dir="/tmp/onnx_out",
        opset=17,
        overwrite=False,
        sample_image=None,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_export_onnx_exits_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "log.txt").write_text("")
    from entomokit.classify import export_onnx
    with pytest.raises(SystemExit):
        export_onnx.run(_export_onnx_args(out_dir=str(out)))


# ── guard message hint ────────────────────────────────────────────────────────

def test_guard_hint_without_resume(capsys):
    from src.common.cli import check_output_dir
    out = Path("/tmp/_test_guard_hint")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dummy.txt").write_text("")
    try:
        with pytest.raises(SystemExit):
            check_output_dir(out, resume=False, overwrite=False, has_resume=False)
        captured = capsys.readouterr()
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    assert "--resume" not in captured.err
    assert "--overwrite" in captured.err


def test_guard_hint_with_resume(capsys):
    from src.common.cli import check_output_dir
    out = Path("/tmp/_test_guard_hint2")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dummy.txt").write_text("")
    try:
        with pytest.raises(SystemExit):
            check_output_dir(out, resume=False, overwrite=False, has_resume=True)
        captured = capsys.readouterr()
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    assert "--resume" in captured.err
    assert "--overwrite" in captured.err
