"""AutoGluon MultiModalPredictor training logic."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
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


def _load_training_frame(
    csv_path: Path,
    images_dir: Path,
    role: str,
    *,
    validate_files: bool,
) -> pd.DataFrame:
    """Load an image,label CSV and resolve image paths.

    Training uses ``validate_files=False``: only column presence and path
    resolution are enforced, preserving the existing lenient training contract.
    Validation uses ``validate_files=True``: the file must be non-empty and every
    image must exist.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"{role} CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = {"image", "label"} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{role} CSV must contain 'image' and 'label' columns; "
            f"missing: {sorted(missing_columns)}"
        )

    if df.empty:
        if validate_files:
            raise ValueError(f"{role} CSV is empty: {csv_path}")
        return df

    if validate_files:
        if df["image"].isna().any() or (
            df["image"].astype(str).str.strip() == ""
        ).any():
            raise ValueError(f"{role} CSV contains blank image values")
        if df["label"].isna().any() or (
            df["label"].astype(str).str.strip() == ""
        ).any():
            raise ValueError(f"{role} CSV contains blank label values")

    df = df.copy()
    df["image"] = df["image"].apply(
        lambda value: str(Path(str(value)))
        if Path(str(value)).is_absolute()
        else str(images_dir / str(value))
    )

    if validate_files:
        resolved = [os.path.normcase(str(Path(value).resolve())) for value in df["image"]]
        duplicates = pd.Series(resolved).duplicated()
        if duplicates.any():
            duplicate_rows = df.loc[duplicates, "image"].tolist()
            raise ValueError(
                f"{role} CSV contains duplicate image rows: {duplicate_rows[:5]}"
            )
        missing_images = [value for value in df["image"] if not Path(value).is_file()]
        if missing_images:
            raise ValueError(
                f"{role} CSV references missing images: {missing_images[:5]}"
            )

    return df


def _resolved_paths(values) -> List[str]:
    return [os.path.normcase(str(Path(value).resolve())) for value in values]


def _append_validation_log(out_dir: Path, lines: List[str]) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "log.txt").open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def _backfill_model_assets(
    model_dir: Path,
    predictor,
    elapsed_seconds: float,
) -> None:
    """Refresh AutoMM's assets.json after fit.

    AutoMM writes assets.json inside top-k averaging before it populates
    ``_best_score`` and ``_total_train_time``, so a reloaded model can show
    null metadata even though training completed.  Update those two standard
    keys in place; keep the file and all other keys untouched.
    """
    assets_path = model_dir / "assets.json"
    if not assets_path.is_file():
        warnings.warn(
            f"AutoMM assets.json not found; skipping metadata backfill: {assets_path}",
            UserWarning,
        )
        return

    try:
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not read AutoMM assets.json: {exc}", UserWarning)
        return
    if not isinstance(assets, dict):
        warnings.warn("AutoMM assets.json is not a JSON object", UserWarning)
        return

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    learner = getattr(predictor, "_learner", None)
    best_score = as_float(getattr(learner, "_best_score", None))
    total_train_time = as_float(getattr(learner, "_total_train_time", None))
    if total_train_time is None:
        total_train_time = float(elapsed_seconds)

    changed = False
    if best_score is not None:
        assets["best_score"] = best_score
        changed = True
    if total_train_time is not None:
        assets["total_train_time"] = total_train_time
        changed = True
    if not changed:
        return

    temp_path = assets_path.with_name(f".{assets_path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(assets, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temp_path, assets_path)
    except OSError as exc:
        warnings.warn(
            f"Could not update AutoMM assets.json: {exc}", UserWarning
        )
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


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
    seed: int = 0,
    val_csv: Optional[Path] = None,
) -> Path:
    """Train an AutoGluon MultiModalPredictor for image classification.

    Returns:
        Path to the saved predictor directory.
    """
    from autogluon.multimodal import MultiModalPredictor
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = _load_training_frame(
        train_csv, images_dir, "training", validate_files=False
    )
    label_count = int(df["label"].nunique())
    if label_count < 2:
        raise ValueError("Training requires at least 2 unique label values.")
    problem_type = "binary" if label_count == 2 else "multiclass"

    val_df: Optional[pd.DataFrame] = None
    validation_lines: List[str] = []
    if val_csv is not None:
        val_df = _load_training_frame(
            val_csv, images_dir, "validation", validate_files=True
        )
        label_lookup = {}
        for label in df["label"]:
            label_lookup.setdefault(str(label), label)

        validation_labels = []
        unknown_labels = []
        for label in val_df["label"]:
            key = str(label)
            if key not in label_lookup:
                unknown_labels.append(str(label))
            else:
                validation_labels.append(label_lookup[key])
        if unknown_labels:
            raise ValueError(
                "Validation CSV contains labels absent from training: "
                f"{sorted(set(unknown_labels))[:5]}"
            )
        val_df = val_df.copy()
        val_df["label"] = validation_labels

        training_paths = set(_resolved_paths(df["image"]))
        validation_paths = set(_resolved_paths(val_df["image"]))
        overlap = sorted(training_paths & validation_paths)
        if overlap:
            raise ValueError(
                "Training and validation images overlap: "
                f"{overlap[:5]}"
            )

        digest = hashlib.sha256(val_csv.read_bytes()).hexdigest()
        validation_lines = [
            f"Validation CSV: {val_csv}",
            f"Validation CSV SHA256: {digest}",
            f"Validation images: {len(val_df)}",
            "Overlap check: passed",
        ]
    else:
        validation_lines = [
            "Validation: AutoMM internal random split (no --val-csv)"
        ]

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
        "seed": seed,
    }
    if val_df is not None:
        fit_kwargs["tuning_data"] = val_df

    _append_validation_log(out_dir, validation_lines)

    training_start = time.monotonic()
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

    elapsed_seconds = time.monotonic() - training_start
    _backfill_model_assets(model_dir, predictor, elapsed_seconds)

    # Save processed CSV for traceability
    processed_csv = out_dir / "train.processed.csv"
    df.to_csv(processed_csv, index=False)

    return model_dir
