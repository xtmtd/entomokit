"""AutoGluon MultiModalPredictor training logic."""

from __future__ import annotations

import os
import sys
import types
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _install_nvml_stub_if_missing() -> bool:
    """Install a lightweight nvidia_smi stub when NVML is unavailable.

    AutoGluon may import `nvidia_smi` even on non-NVIDIA hosts (e.g. Apple Silicon).
    If NVML shared library is missing, replace `nvidia_smi` with a no-op stub so
    logging does not crash training.
    """
    try:
        import nvidia_smi  # type: ignore
    except Exception:
        return False

    try:
        nvidia_smi.nvmlInit()
        return False
    except Exception as e:
        if "NVMLError_LibraryNotFound" not in repr(e):
            return False

    stub = types.ModuleType("nvidia_smi")

    class _MemInfo:
        used = 0
        total = 0

    def _nvml_init() -> None:
        return None

    def _device_handle(index: int) -> int:
        return index

    def _memory_info(_handle: int) -> _MemInfo:
        return _MemInfo()

    stub.nvmlInit = _nvml_init  # type: ignore[attr-defined]
    stub.nvmlDeviceGetHandleByIndex = _device_handle  # type: ignore[attr-defined]
    stub.nvmlDeviceGetMemoryInfo = _memory_info  # type: ignore[attr-defined]
    sys.modules["nvidia_smi"] = stub
    return True


def _suppress_non_cuda_amp_warnings(device: str) -> None:
    """Hide known CUDA AMP warnings on non-CUDA devices."""
    if device == "cuda":
        return
    warnings.filterwarnings(
        "ignore",
        message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.",
        category=UserWarning,
    )


def train(
    train_csv: Path,
    images_dir: Path,
    base_model: str,
    out_dir: Path,
    augment_transforms: List[str],
    max_epochs: int,
    time_limit_hours: float,
    resume: bool,
    learning_rate: Optional[float],
    weight_decay: Optional[float],
    warmup_steps: Optional[float],
    patience: Optional[int],
    top_k: Optional[int],
    focal_loss: bool,
    focal_loss_gamma: float,
    device: str,  # 'cpu'/'cuda'/'mps' (already resolved)
    batch_size: int,
    num_workers: int,
    num_threads: int,
) -> Path:
    """Train an AutoGluon MultiModalPredictor for image classification.

    Returns:
        Path to the saved predictor directory.
    """
    from autogluon.multimodal import MultiModalPredictor
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = pd.read_csv(train_csv)
    df["image"] = df["image"].apply(lambda p: str(images_dir / p))
    label_count = int(df["label"].nunique())
    if label_count < 2:
        raise ValueError("Training requires at least 2 unique label values.")
    problem_type = "binary" if label_count == 2 else "multiclass"

    model_dir = out_dir / "AutogluonModels" / base_model
    if not resume:
        model_dir.mkdir(parents=True, exist_ok=True)
    elif not model_dir.exists():
        raise FileNotFoundError(
            f"--resume requested, but model directory does not exist: {model_dir}"
        )

    hyperparameters: dict[str, object] = {
        "model.timm_image.checkpoint_name": base_model,
        "model.timm_image.train_transforms": augment_transforms,
        "optim.max_epochs": max_epochs,
        "env.num_workers": num_workers,
        "env.batch_size": batch_size,
    }
    if learning_rate is not None:
        hyperparameters["optim.lr"] = learning_rate
    if weight_decay is not None:
        hyperparameters["optim.weight_decay"] = weight_decay
    if warmup_steps is not None:
        hyperparameters["optim.warmup_steps"] = warmup_steps
    if patience is not None:
        hyperparameters["optim.patience"] = patience
    if top_k is not None:
        hyperparameters["optim.top_k"] = top_k
    if focal_loss:
        hyperparameters["optim.loss_func"] = "focal_loss"
        hyperparameters["optim.focal_loss.gamma"] = focal_loss_gamma
    if device != "cuda":
        hyperparameters["env.precision"] = 32

    _suppress_non_cuda_amp_warnings(device=device)
    _install_nvml_stub_if_missing()

    if resume:
        predictor = MultiModalPredictor.load(path=str(model_dir), resume=True)
    else:
        predictor = MultiModalPredictor(
            label="label",
            problem_type=problem_type,
            path=str(model_dir),
        )

    if device == "cpu":
        hyperparameters["env.accelerator"] = "cpu"

    fit_kwargs = {
        "train_data": df,
        "hyperparameters": hyperparameters,
        "time_limit": int(time_limit_hours * 3600),
    }
    try:
        predictor.fit(**fit_kwargs)
    except Exception as e:
        if "NVMLError_LibraryNotFound" not in repr(e):
            raise
        os.environ["AG_AUTOMM_DISABLE_NVML"] = "1"
        os.environ["AUTOMM_DISABLE_NVML"] = "1"
        _install_nvml_stub_if_missing()
        if resume:
            predictor = MultiModalPredictor.load(path=str(model_dir), resume=True)
        else:
            predictor = MultiModalPredictor(
                label="label",
                problem_type=problem_type,
                path=str(model_dir),
            )
        predictor.fit(**fit_kwargs)

    # Save processed CSV for traceability
    processed_csv = out_dir / "train.processed.csv"
    df.to_csv(processed_csv, index=False)

    return model_dir
