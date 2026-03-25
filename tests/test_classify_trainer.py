"""Tests for classify training integration with AutoGluon."""

from __future__ import annotations

import csv
import sys
import types
import warnings
from pathlib import Path


def _install_fake_autogluon(monkeypatch):
    class FakePredictor:
        init_calls = []
        load_calls = []
        fit_calls = []
        fail_with_nvml_once = False

        def __init__(self, **kwargs):
            self.__class__.init_calls.append(kwargs)

        @classmethod
        def load(cls, path: str, resume: bool = False):
            cls.load_calls.append({"path": path, "resume": resume})
            return cls.__new__(cls)

        def fit(self, **kwargs):
            if self.__class__.fail_with_nvml_once:
                self.__class__.fail_with_nvml_once = False
                raise RuntimeError(
                    "pynvml.NVMLError_LibraryNotFound: NVML Shared Library Not Found"
                )
            self.__class__.fit_calls.append(kwargs)

    autogluon_module = types.ModuleType("autogluon")
    multimodal_module = types.ModuleType("autogluon.multimodal")
    multimodal_module.MultiModalPredictor = FakePredictor
    autogluon_module.multimodal = multimodal_module

    monkeypatch.setitem(sys.modules, "autogluon", autogluon_module)
    monkeypatch.setitem(sys.modules, "autogluon.multimodal", multimodal_module)

    return FakePredictor


def _write_train_csv(csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerow({"image": "sample_0.jpg", "label": "moth"})
        writer.writerow({"image": "sample_1.jpg", "label": "beetle"})


def _write_train_csv_with_labels(csv_path: Path, labels: list[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label"])
        writer.writeheader()
        for i, label in enumerate(labels):
            writer.writerow({"image": f"sample_{i}.jpg", "label": label})


def test_train_uses_optim_max_epochs_hparam_not_fit_kwarg(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.classification import trainer as trainer_mod

    fake_predictor = _install_fake_autogluon(monkeypatch)

    train_csv = tmp_path / "train.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample_0.jpg").write_bytes(b"fake")
    (images_dir / "sample_1.jpg").write_bytes(b"fake")
    _write_train_csv(train_csv)

    trainer_mod.train(
        train_csv=train_csv,
        images_dir=images_dir,
        base_model="convnextv2_femto",
        out_dir=tmp_path / "out",
        augment_transforms=["center_crop"],
        max_epochs=12,
        time_limit_hours=1.0,
        focal_loss=False,
        focal_loss_gamma=1.0,
        device="cpu",
        batch_size=8,
        num_workers=2,
        num_threads=0,
        resume=False,
        learning_rate=5e-4,
        weight_decay=1e-4,
        warmup_steps=0.2,
        patience=7,
        top_k=2,
    )

    fit_call = fake_predictor.fit_calls[-1]
    assert "max_epochs" not in fit_call
    assert fit_call["hyperparameters"]["optim.max_epochs"] == 12
    assert fit_call["hyperparameters"]["optim.lr"] == 5e-4
    assert fit_call["hyperparameters"]["optim.weight_decay"] == 1e-4
    assert fit_call["hyperparameters"]["optim.warmup_steps"] == 0.2
    assert fit_call["hyperparameters"]["optim.patience"] == 7
    assert fit_call["hyperparameters"]["optim.top_k"] == 2


def test_train_uses_fp32_precision_on_non_cuda_devices(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.classification import trainer as trainer_mod

    fake_predictor = _install_fake_autogluon(monkeypatch)

    train_csv = tmp_path / "train.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample_0.jpg").write_bytes(b"fake")
    (images_dir / "sample_1.jpg").write_bytes(b"fake")
    _write_train_csv(train_csv)

    trainer_mod.train(
        train_csv=train_csv,
        images_dir=images_dir,
        base_model="convnextv2_femto",
        out_dir=tmp_path / "out",
        augment_transforms=["center_crop"],
        max_epochs=3,
        time_limit_hours=1.0,
        focal_loss=False,
        focal_loss_gamma=1.0,
        device="mps",
        batch_size=8,
        num_workers=2,
        num_threads=0,
        resume=False,
        learning_rate=None,
        weight_decay=None,
        warmup_steps=None,
        patience=None,
        top_k=None,
    )

    fit_call = fake_predictor.fit_calls[-1]
    assert fit_call["hyperparameters"]["env.precision"] == 32


def test_train_resume_loads_predictor_with_resume_true(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.classification import trainer as trainer_mod

    fake_predictor = _install_fake_autogluon(monkeypatch)

    train_csv = tmp_path / "train.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample_0.jpg").write_bytes(b"fake")
    (images_dir / "sample_1.jpg").write_bytes(b"fake")
    _write_train_csv(train_csv)

    out_dir = tmp_path / "out"
    model_dir = out_dir / "AutogluonModels" / "convnextv2_femto"
    model_dir.mkdir(parents=True)

    trainer_mod.train(
        train_csv=train_csv,
        images_dir=images_dir,
        base_model="convnextv2_femto",
        out_dir=out_dir,
        augment_transforms=["center_crop"],
        max_epochs=20,
        time_limit_hours=1.0,
        focal_loss=False,
        focal_loss_gamma=1.0,
        device="cpu",
        batch_size=8,
        num_workers=2,
        num_threads=0,
        resume=True,
        learning_rate=None,
        weight_decay=None,
        warmup_steps=None,
        patience=None,
        top_k=None,
    )

    assert fake_predictor.load_calls == [
        {"path": str(model_dir), "resume": True},
    ]


def test_classify_train_parser_accepts_resume_and_common_optim_flags() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "classify",
            "train",
            "--train-csv",
            "train.csv",
            "--images-dir",
            "images",
            "--out-dir",
            "runs/exp1",
            "--resume",
            "--learning-rate",
            "0.0003",
            "--weight-decay",
            "0.0001",
            "--warmup-steps",
            "0.1",
            "--patience",
            "12",
            "--top-k",
            "2",
        ]
    )

    assert args.resume is True
    assert args.learning_rate == 0.0003
    assert args.weight_decay == 0.0001
    assert args.warmup_steps == 0.1
    assert args.patience == 12
    assert args.top_k == 2


def test_classify_train_parser_has_documented_default_optim_values() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "classify",
            "train",
            "--train-csv",
            "train.csv",
            "--images-dir",
            "images",
            "--out-dir",
            "runs/exp1",
        ]
    )

    assert args.learning_rate == 1e-4
    assert args.weight_decay == 1e-3
    assert args.warmup_steps == 0.1
    assert args.patience == 10
    assert args.top_k == 3


def test_train_uses_binary_problem_type_for_two_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.classification import trainer as trainer_mod

    fake_predictor = _install_fake_autogluon(monkeypatch)

    train_csv = tmp_path / "train_binary.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(4):
        (images_dir / f"sample_{i}.jpg").write_bytes(b"fake")
    _write_train_csv_with_labels(train_csv, ["moth", "beetle", "moth", "beetle"])

    trainer_mod.train(
        train_csv=train_csv,
        images_dir=images_dir,
        base_model="convnextv2_femto",
        out_dir=tmp_path / "out",
        augment_transforms=["center_crop"],
        max_epochs=5,
        time_limit_hours=1.0,
        focal_loss=False,
        focal_loss_gamma=1.0,
        device="cpu",
        batch_size=8,
        num_workers=2,
        num_threads=0,
        resume=False,
        learning_rate=None,
        weight_decay=None,
        warmup_steps=None,
        patience=None,
        top_k=None,
    )

    assert fake_predictor.init_calls[-1]["problem_type"] == "binary"


def test_focal_loss_config_does_not_set_invalid_scalar_alpha(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.classification import trainer as trainer_mod

    fake_predictor = _install_fake_autogluon(monkeypatch)

    train_csv = tmp_path / "train.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample_0.jpg").write_bytes(b"fake")
    (images_dir / "sample_1.jpg").write_bytes(b"fake")
    _write_train_csv(train_csv)

    trainer_mod.train(
        train_csv=train_csv,
        images_dir=images_dir,
        base_model="convnextv2_femto",
        out_dir=tmp_path / "out",
        augment_transforms=["center_crop"],
        max_epochs=3,
        time_limit_hours=1.0,
        focal_loss=True,
        focal_loss_gamma=1.0,
        device="cpu",
        batch_size=8,
        num_workers=2,
        num_threads=0,
        resume=False,
        learning_rate=None,
        weight_decay=None,
        warmup_steps=None,
        patience=None,
        top_k=None,
    )

    fit_call = fake_predictor.fit_calls[-1]
    assert fit_call["hyperparameters"]["optim.loss_func"] == "focal_loss"
    assert fit_call["hyperparameters"]["optim.focal_loss.gamma"] == 1.0
    assert "optim.focal_loss.alpha" not in fit_call["hyperparameters"]


def test_train_retries_when_nvml_library_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.classification import trainer as trainer_mod

    fake_predictor = _install_fake_autogluon(monkeypatch)
    fake_predictor.fail_with_nvml_once = True

    train_csv = tmp_path / "train.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample_0.jpg").write_bytes(b"fake")
    (images_dir / "sample_1.jpg").write_bytes(b"fake")
    _write_train_csv(train_csv)
    monkeypatch.delenv("AG_AUTOMM_DISABLE_NVML", raising=False)

    trainer_mod.train(
        train_csv=train_csv,
        images_dir=images_dir,
        base_model="convnextv2_femto",
        out_dir=tmp_path / "out",
        augment_transforms=["center_crop"],
        max_epochs=5,
        time_limit_hours=1.0,
        focal_loss=False,
        focal_loss_gamma=1.0,
        device="mps",
        batch_size=8,
        num_workers=2,
        num_threads=0,
        resume=False,
        learning_rate=None,
        weight_decay=None,
        warmup_steps=None,
        patience=None,
        top_k=None,
    )

    assert len(fake_predictor.fit_calls) == 1
    assert fake_predictor.fail_with_nvml_once is False


def test_install_nvml_stub_when_library_missing(monkeypatch) -> None:
    from src.classification import trainer as trainer_mod

    class _FakeNvmlModule(types.ModuleType):
        def nvmlInit(self):
            raise RuntimeError(
                "pynvml.NVMLError_LibraryNotFound: NVML Shared Library Not Found"
            )

    fake_nvml = _FakeNvmlModule("nvidia_smi")
    monkeypatch.setitem(sys.modules, "nvidia_smi", fake_nvml)

    installed = trainer_mod._install_nvml_stub_if_missing()

    assert installed is True
    stub = sys.modules["nvidia_smi"]
    stub.nvmlInit()
    handle = stub.nvmlDeviceGetHandleByIndex(0)
    info = stub.nvmlDeviceGetMemoryInfo(handle)
    assert info.used == 0
    assert info.total == 0


def test_suppress_non_cuda_amp_warnings_filters_known_messages() -> None:
    from src.classification import trainer as trainer_mod

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        trainer_mod._suppress_non_cuda_amp_warnings(device="mps")
        warnings.warn(
            "User provided device_type of 'cuda', but CUDA is not available. Disabling",
            UserWarning,
        )
        warnings.warn(
            "torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.",
            UserWarning,
        )

    assert records == []


def test_train_uses_multiclass_problem_type_for_three_or_more_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.classification import trainer as trainer_mod

    fake_predictor = _install_fake_autogluon(monkeypatch)

    train_csv = tmp_path / "train_multi.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(6):
        (images_dir / f"sample_{i}.jpg").write_bytes(b"fake")
    _write_train_csv_with_labels(
        train_csv,
        ["moth", "beetle", "fly", "moth", "beetle", "fly"],
    )

    trainer_mod.train(
        train_csv=train_csv,
        images_dir=images_dir,
        base_model="convnextv2_femto",
        out_dir=tmp_path / "out",
        augment_transforms=["center_crop"],
        max_epochs=5,
        time_limit_hours=1.0,
        focal_loss=False,
        focal_loss_gamma=1.0,
        device="cpu",
        batch_size=8,
        num_workers=2,
        num_threads=0,
        resume=False,
        learning_rate=None,
        weight_decay=None,
        warmup_steps=None,
        patience=None,
        top_k=None,
    )

    assert fake_predictor.init_calls[-1]["problem_type"] == "multiclass"
