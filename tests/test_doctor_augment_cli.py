"""Tests for doctor/augment command registration and behavior."""

from __future__ import annotations


def test_augment_parser_supports_input_output_and_multiply() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "augment",
            "--input-dir",
            "images",
            "--out-dir",
            "augmented",
            "--multiply",
            "3",
        ]
    )

    assert args.input_dir == "images"
    assert args.out_dir == "augmented"
    assert args.multiply == 3


def test_augment_parser_rejects_preset_and_policy_together() -> None:
    import pytest

    from entomokit.main import _build_parser

    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "augment",
                "--input-dir",
                "images",
                "--out-dir",
                "augmented",
                "--preset",
                "light",
                "--policy",
                "policy.json",
            ]
        )


def test_augment_shows_progress_for_each_input_image(tmp_path, monkeypatch) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("albumentations")
    from PIL import Image
    from src.augment import service

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    Image.new("RGB", (10, 10)).save(input_dir / "one.png")
    Image.new("RGB", (10, 10)).save(input_dir / "two.png")

    calls = []

    def fake_tqdm(items, **kwargs):
        calls.append((list(items), kwargs))
        return iter(calls[-1][0])

    monkeypatch.setattr(service, "tqdm", fake_tqdm, raising=False)
    service.run_augment(input_dir, tmp_path / "output", preset="light")

    assert calls[0][0] == sorted(input_dir.iterdir())
    assert calls[0][1]["desc"] == "Augmenting"


def test_doctor_command_is_registered() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["doctor"])

    assert args.command == "doctor"
    assert callable(args.func)


def test_doctor_reports_missing_automm_in_recommendations(monkeypatch) -> None:
    from src.doctor import service as doctor_service

    versions = {
        "torch": "2.3.1",
        "opencv-python": "4.11.0",
        "albumentations": "1.4.14",
        "imagehash": "4.3.1",
        "scikit-image": "0.25.2",
        "pandas": "2.2.3",
        "onnxruntime": "1.20.1",
        "autogluon.multimodal": "NOT INSTALLED",
        "autogluon": "NOT INSTALLED",
        "timm": "1.0.19",
    }

    monkeypatch.setattr(
        doctor_service,
        "_check_pkg_version",
        lambda name: versions.get(name, "NOT INSTALLED"),
    )

    report = doctor_service.run_doctor()

    assert any(
        "autogluon.multimodal>=1.4.0" in rec for rec in report["recommendations"]
    )


def test_augment_nested_same_named_images_are_mirrored(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("albumentations")
    from PIL import Image
    from src.augment import service

    input_dir = tmp_path / "input"
    for sub, color in (("a", (255, 0, 0)), ("b", (0, 0, 255))):
        d = input_dir / sub
        d.mkdir(parents=True)
        Image.new("RGB", (10, 10), color=color).save(d / "same.png")

    out_dir = tmp_path / "out"
    service.run_augment(input_dir, out_dir, preset="light", multiply=1)

    assert (out_dir / "images" / "a" / "same_aug1.png").exists()
    assert (out_dir / "images" / "b" / "same_aug1.png").exists()


def test_augment_multiply_one_always_uses_suffix(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("albumentations")
    from PIL import Image
    from src.augment import service

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    Image.new("RGB", (10, 10)).save(input_dir / "source.png")

    out_dir = tmp_path / "out"
    service.run_augment(input_dir, out_dir, preset="light", multiply=1)

    assert (out_dir / "images" / "source_aug1.png").exists()
    assert not (out_dir / "images" / "source.png").exists()


def _noise_image(path, seed=0) -> None:
    import numpy as np
    from PIL import Image

    arr = np.random.default_rng(seed).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_augment_resume_completes_partial_copies(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("albumentations")
    from src.augment import service

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _noise_image(input_dir / "x.png")
    out = tmp_path / "out"

    service.run_augment(input_dir, out, preset="light", multiply=3, seed=7)
    (out / "images" / "x_aug2.png").unlink()
    (out / "images" / "x_aug3.png").unlink()

    result = service.run_augment(
        input_dir, out, preset="light", multiply=3, seed=7, skip_existing=True
    )
    assert result.manifest["images_processed"] == 1
    assert sorted(p.name for p in (out / "images").glob("x_aug*.png")) == [
        "x_aug1.png",
        "x_aug2.png",
        "x_aug3.png",
    ]


def test_augment_resume_regenerates_identical_pixels(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("albumentations")
    from src.augment import service

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _noise_image(input_dir / "x.png")
    out = tmp_path / "out"

    service.run_augment(input_dir, out, preset="light", multiply=2, seed=7)
    first = {
        p.name: p.read_bytes() for p in (out / "images").glob("x_aug*.png")
    }
    (out / "images" / "x_aug2.png").unlink()

    service.run_augment(
        input_dir, out, preset="light", multiply=2, seed=7, skip_existing=True
    )
    second = {
        p.name: p.read_bytes() for p in (out / "images").glob("x_aug*.png")
    }
    assert second == first


def test_augment_resume_multiply_decrease_removes_stale(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("albumentations")
    from src.augment import service

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _noise_image(input_dir / "x.png")
    out = tmp_path / "out"

    service.run_augment(input_dir, out, preset="light", multiply=3, seed=7)
    service.run_augment(
        input_dir, out, preset="light", multiply=2, seed=7, skip_existing=True
    )
    assert sorted(p.name for p in (out / "images").glob("x_aug*.png")) == [
        "x_aug1.png",
        "x_aug2.png",
    ]


def test_augment_resume_does_not_treat_unrelated_prefix_as_complete(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("albumentations")
    from src.augment import service

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _noise_image(input_dir / "x.png")
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    (out / "images" / "x_augment_backup.png").write_bytes(b"")

    result = service.run_augment(
        input_dir, out, preset="light", multiply=1, seed=7, skip_existing=True
    )
    assert result.manifest["images_processed"] == 1
    assert (out / "images" / "x_augment_backup.png").exists()
    assert (out / "images" / "x_aug1.png").exists()
