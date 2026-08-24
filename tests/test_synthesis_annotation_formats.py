import numpy as np
import pytest

pytest.importorskip("cv2")

from src.synthesis.processor import SynthesisProcessor


def _build_rgba(height: int = 32, width: int = 32) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, :3] = 255
    image[8:24, 10:22, 3] = 255
    return image


class TestRecursiveScan:
    """Test recursive target scanning for synthesis."""

    def _make_nested_targets(self, tmp_path):
        """Create nested structure: targets/{species}/{target}.png"""
        for i in range(2):
            sp_dir = tmp_path / "targets" / f"species_{i}"
            sp_dir.mkdir(parents=True)
            img = _build_rgba()
            from PIL import Image

            Image.fromarray(img).save(sp_dir / f"target_{i}.png")
        return tmp_path / "targets"

    def test_process_directory_finds_nested_targets_when_recursive(self, tmp_path):
        """recursive=True should find nested targets."""
        target_dir = self._make_nested_targets(tmp_path)
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        from PIL import Image

        Image.new("RGB", (64, 64), color=(0, 128, 0)).save(bg_dir / "bg.png")

        processor = SynthesisProcessor(annotation_format="coco")
        results = processor.process_directory(
            target_dir=target_dir,
            background_dir=bg_dir,
            output_dir=tmp_path / "out",
            num_syntheses=1,
            threads=1,
            recursive=True,
        )
        assert results["processed"] == 2

    def test_process_directory_misses_nested_targets_when_not_recursive(self, tmp_path):
        """recursive=False should NOT find nested targets."""
        target_dir = self._make_nested_targets(tmp_path)
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        from PIL import Image

        Image.new("RGB", (64, 64), color=(0, 128, 0)).save(bg_dir / "bg.png")

        processor = SynthesisProcessor(annotation_format="coco")
        with pytest.raises(ValueError, match="No target images found"):
            processor.process_directory(
                target_dir=target_dir,
                background_dir=bg_dir,
                output_dir=tmp_path / "out",
                num_syntheses=1,
                threads=1,
                recursive=False,
            )

    def test_process_directory_mirrors_target_subdir_in_output(self, tmp_path):
        """recursive=True should mirror target subdirs in image output."""
        target_dir = self._make_nested_targets(tmp_path)
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        from PIL import Image

        Image.new("RGB", (64, 64), color=(0, 128, 0)).save(bg_dir / "bg.png")

        processor = SynthesisProcessor(annotation_format="coco")
        processor.process_directory(
            target_dir=target_dir,
            background_dir=bg_dir,
            output_dir=tmp_path / "out",
            num_syntheses=1,
            threads=1,
            recursive=True,
        )

        # Output images should be under out/images/species_0/ and out/images/species_1/
        assert (tmp_path / "out" / "images" / "species_0").exists()
        assert (tmp_path / "out" / "images" / "species_1").exists()


def test_save_voc_single_does_not_pass_unsupported_mask_area(tmp_path):
    processor = SynthesisProcessor(annotation_format="voc")
    result = _build_rgba()

    processor._save_voc_single(
        output_filename="sample.png",
        result=result,
        scale_ratio=1.0,
        rotation_angle=0.0,
        position_x=0,
        position_y=0,
        output_dir=tmp_path,
        target_rgba=result,
    )

    assert (tmp_path / "Annotations" / "sample.xml").exists()


def test_save_yolo_single_does_not_pass_unsupported_mask_area(tmp_path):
    processor = SynthesisProcessor(annotation_format="yolo")
    result = _build_rgba()

    processor._save_yolo_single(
        output_filename="sample.png",
        result=result,
        scale_ratio=1.0,
        rotation_angle=0.0,
        position_x=0,
        position_y=0,
        output_dir=tmp_path,
        target_rgba=result,
    )

    assert (tmp_path / "labels" / "sample.txt").exists()
    yaml_path = tmp_path / "data.yaml"
    assert yaml_path.exists()
    yaml_content = yaml_path.read_text(encoding="utf-8")
    assert "nc: 1" in yaml_content
    assert 'names: ["insect"]' in yaml_content
    assert "train: images" in yaml_content
