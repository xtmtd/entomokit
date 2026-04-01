import numpy as np
import pytest

pytest.importorskip("cv2")

from src.synthesis.processor import SynthesisProcessor


def _build_rgba(height: int = 32, width: int = 32) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, :3] = 255
    image[8:24, 10:22, 3] = 255
    return image


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
