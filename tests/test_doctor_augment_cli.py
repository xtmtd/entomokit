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
