# tests/test_segmentation.py
import pytest
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.segmentation.processor import SegmentationProcessor


class _DummyInpainter:
    def __init__(self):
        self.calls = []

    def __call__(self, image, mask):
        self.calls.append((image.copy(), mask.copy()))
        return image


class DummyInpainter:
    def __init__(self):
        self.calls = []

    def __call__(self, image, mask):
        self.calls.append((image, mask))
        return image


def test_segmentation_processor_init():
    """Test processor initialization."""
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            sam3_checkpoint="fake.pt", device="cpu", segmentation_method="sam3"
        )

        assert processor.device == "cpu"
        assert processor.sam_wrapper is not None
        assert processor.hint == "insect"
        assert processor.repair_strategy is None
        assert processor.metadata_manager is not None
        assert processor.insect_category_id > 0


def test_process_single_insect():
    """Test processing single insect image."""
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        # Setup mock
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[30:70, 30:70] = 255
        mock_wrapper.predict_with_scores.return_value = {
            "masks": [mock_mask],
            "scores": [0.95],
        }
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt", device="cpu", segmentation_method="sam3"
        )

        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                image=img, output_dir=tmpdir, base_name="test"
            )

            assert result is not None
            assert "masks" in result
            assert "output_files" in result
            assert len(result["masks"]) == 1
            assert len(result["output_files"]) == 1
            assert result["output_files"][0].endswith(".png")
            assert Path(result["output_files"][0]).parent.name == "images"


def test_confidence_threshold_filtering():
    """Test filtering by confidence score."""
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_mask1 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask1[30:70, 30:70] = 255
        mock_mask2 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask2[50:80, 50:80] = 255
        mock_wrapper.predict_with_scores.return_value = {
            "masks": [mock_mask1, mock_mask2],
            "scores": [0.95, 0.45],
        }
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt",
            device="cpu",
            segmentation_method="sam3",
            confidence_threshold=0.7,
        )

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                image=img, output_dir=tmpdir, base_name="filtered"
            )

            assert result is not None
            assert len(result["masks"]) == 1
            assert len(result["output_files"]) == 1
            assert Path(result["output_files"][0]).parent.name == "images"


def test_process_multiple_masks():
    """Test processing multiple masks."""
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_mask1 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask1[10:30, 10:30] = 255
        mock_mask2 = np.zeros((100, 100), dtype=np.uint8)
        mock_mask2[50:70, 50:70] = 255
        mock_wrapper.predict_with_scores.return_value = {
            "masks": [mock_mask1, mock_mask2],
            "scores": [0.95, 0.85],
        }
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt", device="cpu", segmentation_method="sam3"
        )

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                image=img, output_dir=tmpdir, base_name="multi"
            )

            assert len(result["masks"]) == 2
            assert len(result["output_files"]) == 2
            assert all(Path(f).parent.name == "images" for f in result["output_files"])


def test_process_empty_masks():
    """Test processing when no masks found."""
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_wrapper.predict_with_scores.return_value = {"masks": [], "scores": []}
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt", device="cpu", segmentation_method="sam3"
        )

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            result = processor.process_image(
                image=img, output_dir=tmpdir, base_name="empty"
            )

            assert result is not None
            assert len(result["masks"]) == 0
            assert len(result["output_files"]) == 0


def test_process_image_metadata():
    """Test metadata generation."""
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper = MagicMock()
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[20:50, 30:70] = 255
        mock_wrapper.predict_with_scores.return_value = {
            "masks": [mock_mask],
            "scores": [0.95],
        }
        mock_sam.return_value = mock_wrapper

        processor = SegmentationProcessor(
            "fake.pt", device="cpu", segmentation_method="sam3"
        )

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.process_image(
                image=img,
                output_dir=tmpdir,
                base_name="meta_test",
                original_path="/path/to/original.jpg",
            )

            # Check metadata was added
            assert len(processor.metadata_manager.images) == 1
            assert len(processor.metadata_manager.annotations) == 1

            img_meta = processor.metadata_manager.images[0]
            assert img_meta["file_name"] == "meta_test.png"

            ann_meta = processor.metadata_manager.annotations[0]
            assert ann_meta["category_id"] == processor.insect_category_id
            # bbox is in original image coordinates
            # Object is at [30, 20, 40, 30] in the 100x100 image
            assert ann_meta["bbox"] == [
                30,
                20,
                40,
                30,
            ]  # x, y, w, h (list for JSON serialization)


def test_lama_mask_dilation_applies(monkeypatch):
    dummy_inpainter = _DummyInpainter()
    processor = SegmentationProcessor(
        sam3_checkpoint="fake.pt",
        device="cpu",
        segmentation_method="otsu",
        repair_strategy="lama",
        lama_mask_dilate=1,
    )

    monkeypatch.setattr(
        processor, "_get_lama_inpainter", lambda refine=False: dummy_inpainter
    )

    image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255

    processor._repair_with_lama(image, mask)

    assert dummy_inpainter.calls, "LaMa inpainter should be invoked"
    _, used_mask = dummy_inpainter.calls[-1]
    assert used_mask.sum() > mask.sum(), "Dilated mask should have larger area"


def test_otsu_avoids_full_image_foreground():
    processor = SegmentationProcessor(
        sam3_checkpoint="fake.pt",
        device="cpu",
        segmentation_method="otsu",
    )

    image = np.full((300, 400, 3), 240, dtype=np.uint8)
    image[100:220, 150:260] = 30

    masks = processor._segment_with_otsu(image)
    assert masks, "Otsu should produce at least one contour mask"

    largest = max(masks, key=lambda m: int(np.sum(m)))
    largest_ratio = float(np.sum(largest)) / float(image.shape[0] * image.shape[1])
    assert largest_ratio < 0.5, "Foreground should not collapse to near full image"


def test_grabcut_rescales_masks_for_large_images(monkeypatch):
    processor = SegmentationProcessor(
        sam3_checkpoint="fake.pt",
        device="cpu",
        segmentation_method="grabcut",
    )

    image = np.zeros((2000, 3000, 3), dtype=np.uint8)

    def fake_grabcut(img, mask, rect, bg_model, fg_model, iters, mode):
        mask[:] = cv2.GC_BGD
        mask[350:500, 600:760] = cv2.GC_FGD

    monkeypatch.setattr(cv2, "grabCut", fake_grabcut)

    masks = processor._segment_with_grabcut(image)
    assert masks, "GrabCut path should produce a foreground mask"

    largest = max(masks, key=lambda m: int(np.sum(m)))
    ys, xs = np.where(largest)
    assert len(xs) > 0 and len(ys) > 0

    # With correct scale-back, the object should map near original-image bottom-right.
    assert int(xs.max()) > 2000
    assert int(ys.max()) > 1200


def test_otsu_bbox_outputs_rgb_crop():
    processor = SegmentationProcessor(
        sam3_checkpoint="fake.pt",
        device="cpu",
        segmentation_method="otsu-bbox",
    )

    image = np.full((240, 320, 3), 240, dtype=np.uint8)
    image[80:180, 120:220] = 20

    with tempfile.TemporaryDirectory() as tmpdir:
        result = processor.process_image(image=image, output_dir=tmpdir, base_name="bbox_otsu")
        assert len(result["output_files"]) == 1

        out_path = Path(result["output_files"][0])
        assert out_path.exists()
        assert Image.open(out_path).mode == "RGB"


def test_grabcut_bbox_outputs_rgb_crop(monkeypatch):
    processor = SegmentationProcessor(
        sam3_checkpoint="fake.pt",
        device="cpu",
        segmentation_method="grabcut-bbox",
    )

    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    fake_mask = np.zeros((200, 300), dtype=bool)
    fake_mask[40:160, 80:220] = True
    monkeypatch.setattr(processor, "_segment_with_grabcut", lambda img: [fake_mask])

    with tempfile.TemporaryDirectory() as tmpdir:
        result = processor.process_image(
            image=image,
            output_dir=tmpdir,
            base_name="bbox_grabcut",
        )
        assert len(result["output_files"]) == 1

        out_path = Path(result["output_files"][0])
        assert out_path.exists()
        assert Image.open(out_path).mode == "RGB"


def test_e2e_segment_real_insect_image():
    """End-to-end test: segment real insect image using SAM3 model."""
    import sys

    sys.path.insert(0, "src")
    pytest.importorskip("sam3", reason="SAM3 not available")

    model_path = Path("/Users/zf/data/coding/models/sam3.pt")
    if not model_path.exists():
        pytest.skip(f"SAM3 checkpoint not found: {model_path}")

    test_image_path = Path("data/insects/female_dor_1_Lucanus_brivioi.jpg")
    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        processor = SegmentationProcessor(
            sam3_checkpoint=str(model_path), device="cpu", hint="insect",
            segmentation_method="sam3",
        )

        # Load real image
        from src.utils import load_image

        image = load_image(test_image_path)

        # Process single image
        result = processor.process_image(
            image=image, output_dir=tmpdir, base_name="test_insect"
        )

        # Verify results
        assert result is not None
        assert len(result["masks"]) >= 1  # Should find at least one mask
        assert len(result["output_files"]) >= 1

        # Verify output file exists and is valid
        output_path = Path(result["output_files"][0])
        assert output_path.exists()
        assert output_path.suffix == ".png"

        # Verify image has alpha channel
        from src.utils import load_image as load_rgba

        output_img = load_rgba(output_path)
        assert output_img.shape[2] == 4  # RGBA


def test_e2e_segment_directory_real_images():
    """End-to-end test: process directory of real insect images."""
    import sys

    sys.path.insert(0, "src")
    pytest.importorskip("sam3", reason="SAM3 not available")

    model_path = Path("/Users/zf/data/coding/models/sam3.pt")
    if not model_path.exists():
        pytest.skip(f"SAM3 checkpoint not found: {model_path}")

    test_images = list(Path("data/insects").glob("*.jpg"))
    if len(test_images) < 2:
        pytest.skip(f"Need at least 2 test images, found {len(test_images)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()

        # Copy first 2 images to input dir
        for img_path in test_images[:2]:
            import shutil

            shutil.copy(img_path, input_dir)

        processor = SegmentationProcessor(
            sam3_checkpoint=str(model_path), device="cpu", hint="insect",
            segmentation_method="sam3",
        )

        result = processor.process_directory(input_dir=input_dir, output_dir=output_dir)

        # Verify results
        assert result is not None
        assert result["processed"] == 2
        assert result["failed"] == 0
        assert len(result["output_files"]) >= 2  # At least one per image

        # Verify metadata was saved
        metadata_path = output_dir / "annotations.coco.json"
        assert metadata_path.exists()

        # Verify metadata structure
        import json

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "images" in metadata
        assert "annotations" in metadata
        assert "categories" in metadata
        assert len(metadata["categories"]) > 0
        assert metadata["categories"][0]["name"] == "insect"


# ─── CPU worker dispatch tests ─────────────────────────────────────────


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for stem, color in [("01", 30), ("02", 60), ("03", 100)]:
        img = np.full((240, 320, 3), color, dtype=np.uint8)
        img[80:180, 120:220] = 20
        path = d / f"{stem}.png"
        from PIL import Image as _PIL

        _PIL.fromarray(img).save(path)
    return d


def _make_processor(method="otsu", annotation_format="coco"):
    mock_wrapper = MagicMock()
    with (
        patch("src.segmentation.processor.SAM3Wrapper") as mock_sam,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_wrapper.predict_with_scores.return_value = {"masks": [], "scores": []}
        mock_sam.return_value = mock_wrapper
        return SegmentationProcessor(
            sam3_checkpoint="fake.pt",
            device="cpu",
            segmentation_method=method,
            annotation_format=annotation_format,
        )


@pytest.mark.parametrize("method", ["otsu", "grabcut"])
def test_cpu_methods_use_image_workers(monkeypatch, image_dir, tmp_path, method):
    submitted = []

    class _RecordingExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args):
            submitted.append(args[0])
            return _ImmediateFuture(fn(*args))

    monkeypatch.setattr(
        "src.segmentation.processor.ThreadPoolExecutor", _RecordingExecutor
    )
    processor = _make_processor(method)
    processor.process_directory(image_dir, tmp_path, num_workers=2)

    expected = sorted(image_dir.glob("*.png"))
    assert submitted == expected


def test_sam3_never_constructs_cpu_worker_executor(monkeypatch, image_dir, tmp_path):
    monkeypatch.setattr(
        "src.segmentation.processor.ThreadPoolExecutor",
        lambda **_kwargs: pytest.fail("SAM3 must remain serial"),
    )
    processor = _make_processor("sam3")
    processor.process_directory(image_dir, tmp_path, num_workers=8)


def _tree_bytes(root):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_otsu_outputs_match_with_one_and_two_workers(image_dir, tmp_path):
    out1 = tmp_path / "one"
    out2 = tmp_path / "two"
    p1 = _make_processor("otsu")
    p1.process_directory(image_dir, out1, num_workers=1)
    p2 = _make_processor("otsu")
    p2.process_directory(image_dir, out2, num_workers=2)
    assert _tree_bytes(out1) == _tree_bytes(out2)


def test_cpu_worker_failure_is_counted_and_later_images_write(monkeypatch, image_dir, tmp_path):
    processor = _make_processor("otsu")
    original = processor._compute_image

    def _fail_on_02(path):
        if path.stem == "02":
            raise ValueError("bad image")
        return original(path)

    monkeypatch.setattr(processor, "_compute_image", _fail_on_02)
    result = processor.process_directory(image_dir, tmp_path, num_workers=2)
    assert (result["processed"], result["failed"]) == (2, 1)

    import json

    json.loads((tmp_path / "annotations.coco.json").read_text())


@pytest.mark.parametrize("value", [0, -1])
def test_process_directory_rejects_nonpositive_num_workers(image_dir, tmp_path, value):
    processor = _make_processor("otsu")
    with pytest.raises(ValueError, match="num_workers must be positive"):
        processor.process_directory(image_dir, tmp_path, num_workers=value)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_segment_cli_rejects_nonpositive_threads(value):
    from entomokit.main import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["segment", "--input-dir", "in", "--out-dir", "out", "--threads", value]
        )
