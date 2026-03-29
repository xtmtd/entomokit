# entomokit 重构 Phase 3 — classify 命令组实现

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `entomokit classify` 下的全部 6 个子命令（train/predict/evaluate/embed/cam/export-onnx），将 `add_functions/` 中的业务逻辑迁移并整合进统一框架。

**Architecture:** 业务逻辑写在 `src/classification/` 各模块，CLI 参数解析写在 `entomokit/classify/` 各模块，两者通过简洁函数接口解耦。`export-onnx` 直接调用 `MultiModalPredictor.export_onnx()`。

**Tech Stack:** AutoGluon MultiModalPredictor, timm, grad-cam, umap-learn, scikit-learn, onnxruntime, torch

**Spec:** `docs/superpowers/specs/2026-03-24-entomokit-refactor-design.md` — Section 5.2–5.7, Section 7

**前置条件:** Phase 1 已完成（`entomokit classify` stub 可用）

**参考文件（只读）:**
- `add_functions/extract_embeddings20260115.py` — embed/train/evaluate/predict 逻辑来源
- `add_functions/GradCam_heatmap.py` — cam 逻辑来源

---

## 文件结构

### 新建文件

```
src/classification/__init__.py
src/classification/trainer.py       # AutoGluon 训练
src/classification/predictor.py     # 推理（AutoGluon + ONNX）
src/classification/evaluator.py     # 评估（AutoGluon + ONNX）
src/classification/embedder.py      # 嵌入提取 + 质量指标 + UMAP
src/classification/cam.py           # GradCAM 热力图（仅 PyTorch）
src/classification/exporter.py      # ONNX 导出（调用 AutoGluon 原生 API）
src/classification/utils.py         # 共享工具：设备选择、augment 解析、DataLoader 构建

entomokit/classify/__init__.py      # 替换 Phase 1 stub，注册全部 6 子命令
entomokit/classify/train.py
entomokit/classify/predict.py
entomokit/classify/evaluate.py
entomokit/classify/embed.py
entomokit/classify/cam.py
entomokit/classify/export_onnx.py
```

---

## Task 1: `src/classification/utils.py` — 共享工具

**Files:**
- Create: `src/classification/__init__.py`
- Create: `src/classification/utils.py`

- [ ] **Step 1: 创建包**

```bash
mkdir -p src/classification
touch src/classification/__init__.py
```

- [ ] **Step 2: 写 `src/classification/utils.py`**

```python
"""Shared utilities for the classify command group."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Valid augment preset names and their corresponding transform lists
AUGMENT_TRANSFORMS = {
    "none": ["resize_shorter_side", "center_crop"],
    "light": ["resize_shorter_side", "center_crop", "random_horizontal_flip"],
    "medium": [
        "resize_shorter_side", "center_crop",
        "random_horizontal_flip", "color_jitter", "trivial_augment",
    ],
    "heavy": [
        "random_resize_crop", "random_horizontal_flip",
        "random_vertical_flip", "color_jitter", "trivial_augment", "randaug",
    ],
}

# All valid transform string names accepted by AutoGluon
VALID_TRANSFORM_NAMES = {
    "resize_to_square", "resize_shorter_side", "center_crop",
    "random_resize_crop", "random_horizontal_flip", "random_vertical_flip",
    "color_jitter", "affine", "randaug", "trivial_augment",
}


def resolve_augment(augment: str) -> List[str]:
    """Parse --augment value to a list of transform strings.

    Accepts preset names (none/light/medium/heavy) or a JSON array string.
    Raises ValueError on invalid preset or unknown transform names in JSON.
    """
    if augment in AUGMENT_TRANSFORMS:
        return AUGMENT_TRANSFORMS[augment]

    # Try JSON
    try:
        transforms = json.loads(augment)
    except json.JSONDecodeError:
        valid = ", ".join(sorted(AUGMENT_TRANSFORMS))
        raise ValueError(
            f"Invalid --augment value: {augment!r}. "
            f"Use one of [{valid}] or a JSON array of transform names."
        )

    if not isinstance(transforms, list):
        raise ValueError("--augment JSON must be an array, e.g. '[\"random_resize_crop\"]'")

    unknown = [t for t in transforms if t not in VALID_TRANSFORM_NAMES]
    if unknown:
        valid_names = ", ".join(sorted(VALID_TRANSFORM_NAMES))
        raise ValueError(
            f"Unknown transform name(s) in --augment: {unknown}. "
            f"Valid names: [{valid_names}]"
        )

    return transforms


def select_device(device_str: str) -> torch.device:
    """Resolve 'auto'/cuda/mps/cpu to a torch.device, with fallback."""
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    if device_str == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("Warning: MPS not available, falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_str)


def set_num_threads(num_threads: int) -> None:
    """Set PyTorch CPU thread count. 0 = let PyTorch decide."""
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def load_image_csv(csv_path: Path, require_label: bool = False) -> "pd.DataFrame":
    """Load and validate a CSV with at least an 'image' column."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError(f"CSV must have an 'image' column: {csv_path}")
    if require_label and "label" not in df.columns:
        raise ValueError(f"CSV must have a 'label' column: {csv_path}")
    return df


def ag_device_map(device: torch.device) -> str:
    """Convert torch.device to AutoGluon device string."""
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"
```

- [ ] **Step 3: 写单元测试**

在 `tests/test_classify_utils.py`：

```python
"""Tests for src/classification/utils.py"""
import pytest
from src.classification.utils import resolve_augment, AUGMENT_TRANSFORMS


def test_preset_none():
    assert resolve_augment("none") == AUGMENT_TRANSFORMS["none"]


def test_preset_medium():
    assert resolve_augment("medium") == AUGMENT_TRANSFORMS["medium"]


def test_preset_heavy():
    assert resolve_augment("heavy") == AUGMENT_TRANSFORMS["heavy"]


def test_json_custom_valid():
    result = resolve_augment('["random_resize_crop", "color_jitter"]')
    assert result == ["random_resize_crop", "color_jitter"]


def test_invalid_preset_raises():
    with pytest.raises(ValueError, match="Invalid --augment"):
        resolve_augment("ultra")


def test_invalid_json_raises():
    with pytest.raises(ValueError, match="Invalid --augment"):
        resolve_augment("not_json_or_preset")


def test_unknown_transform_in_json_raises():
    with pytest.raises(ValueError, match="Unknown transform name"):
        resolve_augment('["random_resize_crop", "super_augment"]')


def test_json_not_array_raises():
    with pytest.raises(ValueError, match="must be an array"):
        resolve_augment('{"key": "value"}')
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_classify_utils.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/classification/ tests/test_classify_utils.py
git commit -m "feat: add classification utils (device, augment, thread helpers)"
```

---

## Task 2: `src/classification/trainer.py`

**Files:**
- Create: `src/classification/trainer.py`
- Reference: `add_functions/extract_embeddings20260115.py` (训练逻辑，约 L440–L500)

- [ ] **Step 1: 写 `src/classification/trainer.py`**

```python
"""AutoGluon MultiModalPredictor training logic."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def train(
    train_csv: Path,
    images_dir: Path,
    base_model: str,
    out_dir: Path,
    augment_transforms: List[str],
    max_epochs: int,
    time_limit_hours: float,
    focal_loss: bool,
    focal_loss_gamma: float,
    device: str,          # 'cpu'/'cuda'/'mps' (already resolved)
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

    model_dir = out_dir / "AutogluonModels" / base_model
    model_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "model.timm_image.checkpoint_name": base_model,
        "model.timm_image.train_transforms": augment_transforms,
        "env.num_workers": num_workers,
        "env.batch_size": batch_size,
    }
    if focal_loss:
        hyperparameters["optimization.loss_function"] = "focal_loss"
        hyperparameters["optimization.focal_loss.alpha"] = -1
        hyperparameters["optimization.focal_loss.gamma"] = focal_loss_gamma

    predictor = MultiModalPredictor(
        label="label",
        problem_type="multiclass",
        path=str(model_dir),
    )
    predictor.fit(
        train_data=df,
        hyperparameters=hyperparameters,
        max_epochs=max_epochs,
        time_limit=int(time_limit_hours * 3600),
    )

    # Save processed CSV for traceability
    processed_csv = out_dir / "train.processed.csv"
    df.to_csv(processed_csv, index=False)

    return model_dir
```

- [ ] **Step 2: 写 `entomokit/classify/train.py`**

```python
"""entomokit classify train — train an AutoGluon image classifier."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train",
        help="Train an AutoGluon image classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-csv", required=True, help="CSV with 'image' and 'label' columns.")
    p.add_argument("--images-dir", required=True, help="Directory containing training images.")
    p.add_argument("--base-model", default="convnextv2_femto",
                   help="timm backbone name.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--augment", default="medium",
                   help="Augment preset: none/light/medium/heavy or JSON array.")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--time-limit", type=float, default=1.0,
                   help="Training time limit in hours.")
    p.add_argument("--focal-loss", action="store_true")
    p.add_argument("--focal-loss-gamma", type=float, default=1.0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-threads", type=int, default=0,
                   help="CPU threads for PyTorch (0 = auto).")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.classification.utils import resolve_augment, select_device, ag_device_map
    from src.classification.trainer import train
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    save_log(out_dir / "logs", args, log_filename="log.txt")

    augment_transforms = resolve_augment(args.augment)  # raises on invalid
    device = select_device(args.device)

    model_dir = train(
        train_csv=Path(args.train_csv),
        images_dir=Path(args.images_dir),
        base_model=args.base_model,
        out_dir=out_dir,
        augment_transforms=augment_transforms,
        max_epochs=args.max_epochs,
        time_limit_hours=args.time_limit,
        focal_loss=args.focal_loss,
        focal_loss_gamma=args.focal_loss_gamma,
        device=ag_device_map(device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
    )

    print(f"Model saved to: {model_dir}")
```

- [ ] **Step 3: 冒烟测试 (无需真实训练)**

```bash
python -m entomokit.main classify train --help
python -m entomokit.main classify train \
    --train-csv nonexistent.csv \
    --images-dir . \
    --out-dir /tmp/cls_train \
    --augment invalid_preset 2>&1 | grep "Invalid --augment"
```

Expected: help 正确显示；第二条命令输出包含 `Invalid --augment`。

- [ ] **Step 4: Commit**

```bash
git add src/classification/trainer.py entomokit/classify/train.py
git commit -m "feat: classify train command (AutoGluon MultiModalPredictor)"
```

---

## Task 3: `src/classification/predictor.py` + `classify predict`

**Files:**
- Create: `src/classification/predictor.py`
- Create: `entomokit/classify/predict.py`
- Reference: `add_functions/extract_embeddings20260115.py` (predict_classification, ~L530–L600)

- [ ] **Step 1: 写 `src/classification/predictor.py`**

```python
"""Image classification inference — AutoGluon and ONNX."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def predict_ag(
    input_df: pd.DataFrame,
    images_dir: Optional[Path],
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    num_threads: int,
    device: str,
) -> pd.DataFrame:
    """Run inference with AutoGluon predictor.
    Returns DataFrame with columns: image, prediction, proba_*.
    """
    from autogluon.multimodal import MultiModalPredictor
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = input_df.copy()
    if images_dir:
        df["image"] = df["image"].apply(lambda p: str(images_dir / p))

    predictor = MultiModalPredictor.load(str(model_dir))
    proba = predictor.predict_proba(df)
    predictions = predictor.predict(df)

    result = input_df.copy()
    result["prediction"] = predictions.values
    for cls in proba.columns:
        result[f"proba_{cls}"] = proba[cls].values
    return result


def predict_onnx(
    input_df: pd.DataFrame,
    images_dir: Optional[Path],
    onnx_path: Path,
    batch_size: int,
    num_threads: int,
) -> pd.DataFrame:
    """Run inference with ONNX model.
    Returns DataFrame with columns: image, prediction, proba_*.
    """
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import torchvision.transforms as T

    sess_options = ort.SessionOptions()
    if num_threads > 0:
        sess_options.intra_op_num_threads = num_threads
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_name = sess.get_inputs()[0].name
    all_proba = []

    paths = input_df["image"].tolist()
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            full_p = (images_dir / p) if images_dir else Path(p)
            img = Image.open(full_p).convert("RGB")
            tensors.append(transform(img).numpy())
        batch = np.stack(tensors)
        logits = sess.run(None, {input_name: batch})[0]
        # softmax
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
        all_proba.append(proba)

    all_proba = np.vstack(all_proba)
    n_classes = all_proba.shape[1]

    result = input_df.copy()
    result["prediction"] = all_proba.argmax(axis=1)
    for i in range(n_classes):
        result[f"proba_{i}"] = all_proba[:, i]
    return result
```

- [ ] **Step 2: 写 `entomokit/classify/predict.py`**

```python
"""entomokit classify predict — run image classification inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "predict",
        help="Run classification inference (AutoGluon or ONNX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv", help="CSV with 'image' column.")
    input_group.add_argument("--images-dir", help="Directory to scan for images.")
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-dir", help="AutoGluon predictor directory.")
    model_group.add_argument("--onnx-model", help="ONNX model file path.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import pandas as pd
    from pathlib import Path
    from src.classification.utils import select_device, ag_device_map, load_image_csv
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    # Build input DataFrame
    if args.input_csv:
        df = load_image_csv(Path(args.input_csv))
        images_dir = None
    else:
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        imgs = [p.name for p in Path(args.images_dir).iterdir()
                if p.suffix.lower() in IMAGE_EXTS]
        df = pd.DataFrame({"image": sorted(imgs)})
        images_dir = Path(args.images_dir)

    device = select_device(args.device)

    if args.model_dir:
        from src.classification.predictor import predict_ag
        result = predict_ag(
            input_df=df,
            images_dir=images_dir,
            model_dir=Path(args.model_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_threads=args.num_threads,
            device=ag_device_map(device),
        )
    else:
        from src.classification.predictor import predict_onnx
        result = predict_onnx(
            input_df=df,
            images_dir=images_dir,
            onnx_path=Path(args.onnx_model),
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )

    out_csv = pred_dir / "predictions.csv"
    result.to_csv(out_csv, index=False)
    print(f"Predictions saved to: {out_csv}")
```

- [ ] **Step 3: 冒烟测试**

```bash
python -m entomokit.main classify predict --help
```

Expected: help 含 `--input-csv/--images-dir` 互斥组和 `--model-dir/--onnx-model` 互斥组。

- [ ] **Step 4: Commit**

```bash
git add src/classification/predictor.py entomokit/classify/predict.py
git commit -m "feat: classify predict command (AutoGluon + ONNX inference)"
```

---

## Task 4: `src/classification/evaluator.py` + `classify evaluate`

**Files:**
- Create: `src/classification/evaluator.py`
- Create: `entomokit/classify/evaluate.py`
- Reference: `add_functions/extract_embeddings20260115.py` (evaluate_classification, ~L480–L530)

- [ ] **Step 1: 写 `src/classification/evaluator.py`**

```python
"""Classification evaluation — accuracy, F1, MCC, AUC."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np


def evaluate(
    test_csv: Path,
    images_dir: Path,
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    num_threads: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate AutoGluon predictor. Returns dict of metric name → value."""
    from autogluon.multimodal import MultiModalPredictor
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, matthews_corrcoef
    )
    from src.classification.utils import set_num_threads

    set_num_threads(num_threads)

    df = pd.read_csv(test_csv)
    df["image"] = df["image"].apply(lambda p: str(images_dir / p))

    predictor = MultiModalPredictor.load(str(model_dir))
    predictions = predictor.predict(df).values
    labels = df["label"].values

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, predictions, average="micro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "recall_micro": recall_score(labels, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
    }

    # ROC-AUC (OVR, requires probabilities)
    try:
        from sklearn.metrics import roc_auc_score
        proba = predictor.predict_proba(df).values
        metrics["roc_auc_ovo"] = roc_auc_score(
            labels, proba, multi_class="ovo", average="macro"
        )
    except Exception:
        metrics["roc_auc_ovo"] = float("nan")

    return metrics


def evaluate_onnx(
    test_csv: Path,
    images_dir: Path,
    onnx_path: Path,
    batch_size: int,
    num_threads: int,
) -> Dict[str, float]:
    """Evaluate ONNX model using predict_onnx + sklearn metrics."""
    from src.classification.predictor import predict_onnx
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    df = pd.read_csv(test_csv)
    result = predict_onnx(df, images_dir, onnx_path, batch_size, num_threads)

    labels = df["label"].values
    predictions = result["prediction"].values

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
    }
```

- [ ] **Step 2: 写 `entomokit/classify/evaluate.py`**

```python
"""entomokit classify evaluate — evaluate classification performance."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "evaluate",
        help="Evaluate classification performance (AutoGluon or ONNX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test-csv", required=True)
    p.add_argument("--images-dir", required=True)
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-dir")
    model_group.add_argument("--onnx-model")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.classification.utils import select_device, ag_device_map
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_log(logs_dir, args)

    device = select_device(args.device)

    if args.model_dir:
        from src.classification.evaluator import evaluate
        metrics = evaluate(
            test_csv=Path(args.test_csv),
            images_dir=Path(args.images_dir),
            model_dir=Path(args.model_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_threads=args.num_threads,
            device=ag_device_map(device),
        )
    else:
        from src.classification.evaluator import evaluate_onnx
        metrics = evaluate_onnx(
            test_csv=Path(args.test_csv),
            images_dir=Path(args.images_dir),
            onnx_path=Path(args.onnx_model),
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )

    eval_txt = logs_dir / "evaluations.txt"
    lines = [f"{k}: {v:.6f}" for k, v in metrics.items()]
    eval_txt.write_text("\n".join(lines) + "\n")

    for line in lines:
        print(line)
    print(f"\nResults saved to: {eval_txt}")
```

- [ ] **Step 3: 冒烟测试**

```bash
python -m entomokit.main classify evaluate --help
```

- [ ] **Step 4: Commit**

```bash
git add src/classification/evaluator.py entomokit/classify/evaluate.py
git commit -m "feat: classify evaluate command (AutoGluon + ONNX, sklearn metrics)"
```

---

## Task 5: `src/classification/embedder.py` + `classify embed`

**Files:**
- Create: `src/classification/embedder.py`
- Create: `entomokit/classify/embed.py`
- Reference: `add_functions/extract_embeddings20260115.py` (嵌入提取, UMAP, 质量指标, ~L110–L440)

- [ ] **Step 1: 写 `src/classification/embedder.py`**

```python
"""Embedding extraction, quality metrics, and UMAP visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class _ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


def extract_embeddings_timm(
    images_dir: Path,
    base_model: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Extract embeddings using a pretrained timm backbone (no fine-tuning)."""
    import timm
    from timm.data import resolve_model_data_config
    from timm.data.transforms_factory import create_transform

    model = timm.create_model(base_model, pretrained=True, num_classes=0)
    model.eval().to(device)

    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    dataset = _ImageDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    all_embeddings, all_paths = [], []
    with torch.no_grad():
        for batch_tensors, batch_paths in loader:
            feats = model(batch_tensors.to(device)).cpu().numpy()
            all_embeddings.append(feats)
            all_paths.extend(batch_paths)

    embeddings = np.vstack(all_embeddings)
    df = pd.DataFrame(embeddings, columns=[f"feat_{i}" for i in range(embeddings.shape[1])])
    df.insert(0, "image", [Path(p).name for p in all_paths])
    return df


def extract_embeddings_ag(
    images_dir: Path,
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Extract embeddings using a fine-tuned AutoGluon model."""
    from autogluon.multimodal import MultiModalPredictor

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    df_in = pd.DataFrame({"image": [str(p) for p in paths]})

    predictor = MultiModalPredictor.load(str(model_dir))
    embeddings = predictor.extract_embedding(df_in)

    embed_df = pd.DataFrame(embeddings, columns=[f"feat_{i}" for i in range(embeddings.shape[1])])
    embed_df.insert(0, "image", [p.name for p in paths])
    return embed_df


def compute_embedding_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10000,
) -> Dict[str, float]:
    """Compute embedding space quality metrics.

    Returns dict with: NMI, ARI, Recall@1/5/10, kNN_Acc_k1/5/20,
    Linear_Probing_Acc, Purity, Silhouette_Score.
    """
    from sklearn.metrics import (
        normalized_mutual_info_score, adjusted_rand_score, silhouette_score
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Subsample if needed
    n = len(y)
    if 0 < sample_size < n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, sample_size, replace=False)
        X, y = embeddings[idx], y[idx]
    else:
        X = embeddings

    # Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms > 0, norms, 1)

    # Clustering metrics
    from sklearn.cluster import KMeans
    n_clusters = len(np.unique(y))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_norm)

    nmi = normalized_mutual_info_score(y, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)

    # Purity
    from collections import Counter
    purity_sum = sum(
        Counter(y[cluster_labels == c]).most_common(1)[0][1]
        for c in np.unique(cluster_labels)
    )
    purity = purity_sum / len(y)

    # kNN accuracy
    def knn_acc(k):
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_norm, y)
        return knn.score(X_norm, y)

    # Recall@K
    def recall_at_k(k):
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nbrs.fit(X_norm)
        indices = nbrs.kneighbors(X_norm, return_distance=False)[:, 1:]
        hits = sum(any(y[j] == y[i] for j in indices[i]) for i in range(len(y)))
        return hits / len(y)

    # Linear probing
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_norm, y)
    lp_acc = lr.score(X_norm, y)

    # Silhouette (sample 2000 for speed)
    sil_idx = np.random.RandomState(42).choice(len(y), min(2000, len(y)), replace=False)
    sil = silhouette_score(X_norm[sil_idx], y[sil_idx])

    # mAP@R: mean Average Precision at R (R = number of same-class samples)
    def mean_ap_at_r() -> float:
        from sklearn.neighbors import NearestNeighbors
        n = len(y)
        nbrs = NearestNeighbors(n_neighbors=n, metric="euclidean")
        nbrs.fit(X_norm)
        indices = nbrs.kneighbors(X_norm, return_distance=False)[:, 1:]
        aps = []
        for i in range(n):
            r = int((y == y[i]).sum()) - 1  # number of same-class samples excluding self
            if r == 0:
                continue
            retrieved = indices[i, :r]
            hits = (y[retrieved] == y[i])
            precisions = hits.cumsum() / (np.arange(len(hits)) + 1)
            aps.append((precisions * hits).sum() / r)
        return float(np.mean(aps)) if aps else 0.0

    return {
        "NMI": nmi,
        "ARI": ari,
        "Recall@1": recall_at_k(1),
        "Recall@5": recall_at_k(5),
        "Recall@10": recall_at_k(10),
        "kNN_Acc_k1": knn_acc(1),
        "kNN_Acc_k5": knn_acc(5),
        "kNN_Acc_k20": knn_acc(20),
        "Linear_Probing_Acc": lp_acc,
        "mAP@R": mean_ap_at_r(),
        "Purity": purity,
        "Silhouette_Score": sil,
    }


def visualize_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
) -> None:
    """Generate and save a UMAP scatter plot coloured by label."""
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    embedding_2d = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    colors = [color_map[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[color_map[lbl]], label=lbl, s=5, alpha=0.7)
    ax.legend(markerscale=3, bbox_to_anchor=(1, 1), loc="upper left", fontsize=8)
    ax.set_title("UMAP Embedding Visualization")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: 写 `entomokit/classify/embed.py`**

```python
"""entomokit classify embed — extract embeddings and compute quality metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "embed",
        help="Extract embeddings, compute quality metrics, and optionally visualize with UMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-model", default="convnextv2_femto",
                   help="timm backbone (used if --model-dir not provided).")
    p.add_argument("--model-dir", default=None,
                   help="AutoGluon predictor for fine-tuned backbone extraction.")
    p.add_argument("--label-csv", default=None,
                   help="CSV(image,label) for supervised metrics and UMAP coloring.")
    p.add_argument("--visualize", action="store_true",
                   help="Generate UMAP plot (requires --label-csv).")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", default="euclidean")
    p.add_argument("--umap-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--metrics-sample-size", type=int, default=10000)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from src.classification.utils import select_device, set_num_threads, load_image_csv
    from src.common.cli import save_log

    if args.visualize and not args.label_csv:
        print("Error: --visualize requires --label-csv to be provided.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_log(logs_dir, args)

    device = select_device(args.device)
    set_num_threads(args.num_threads)

    # Extract embeddings
    if args.model_dir:
        from src.classification.embedder import extract_embeddings_ag
        embed_df = extract_embeddings_ag(
            images_dir=Path(args.images_dir),
            model_dir=Path(args.model_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
    else:
        from src.classification.embedder import extract_embeddings_timm
        embed_df = extract_embeddings_timm(
            images_dir=Path(args.images_dir),
            base_model=args.base_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

    embed_df.to_csv(out_dir / "embeddings.csv", index=False)
    print(f"Embeddings saved: {len(embed_df)} images, {embed_df.shape[1] - 1} dims")

    # Supervised metrics + UMAP
    if args.label_csv:
        label_df = load_image_csv(Path(args.label_csv), require_label=True)
        merged = embed_df.merge(label_df[["image", "label"]], on="image", how="inner")
        feat_cols = [c for c in merged.columns if c.startswith("feat_")]
        embeddings = merged[feat_cols].values
        labels = merged["label"].values

        from src.classification.embedder import compute_embedding_metrics
        metrics = compute_embedding_metrics(
            embeddings, labels, sample_size=args.metrics_sample_size
        )
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(logs_dir / "metrics.csv", index=False)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if args.visualize:
            from src.classification.embedder import visualize_umap
            umap_path = out_dir / "umap.pdf"
            visualize_umap(
                embeddings=embeddings,
                labels=labels,
                out_path=umap_path,
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                seed=args.umap_seed,
            )
            print(f"UMAP saved to: {umap_path}")
```

- [ ] **Step 3: 冒烟测试**

```bash
python -m entomokit.main classify embed --help
```

- [ ] **Step 4: Commit**

```bash
git add src/classification/embedder.py entomokit/classify/embed.py
git commit -m "feat: classify embed command (timm/AG embeddings + UMAP + quality metrics)"
```

---

## Task 6: `src/classification/cam.py` + `classify cam`

**Files:**
- Create: `src/classification/cam.py`
- Create: `entomokit/classify/cam.py`
- Reference: `add_functions/GradCam_heatmap.py`

- [ ] **Step 1: 写 `src/classification/cam.py`**

从 `add_functions/GradCam_heatmap.py` 提取以下函数，保持原逻辑不变（注意：`select_device` 已在 utils.py，不重复；`timm.data.resolve_model_data_config` 需从 `timm.data.config` 导入，修正原脚本的导入路径错误）：

提取的函数：`load_label_file`, `load_model`, `build_eval_transforms`, `get_module_by_name`, `find_last_conv_module`, `default_vit_target`, `infer_architecture`, `vit_reshape_transform`, `prepare_cam`, `prepare_output_dirs`, `process_image`

新增公开入口函数 `run_cam`，接受显式关键字参数（不使用 argparse.Namespace），供 `entomokit/classify/cam.py` 调用：

```python
def run_cam(
    *,
    label_csv: Path,
    images_dir: Path,
    out_dir: Path,
    model_dir: Optional[Path],
    base_model: Optional[str],
    checkpoint_path: Optional[Path],
    num_classes: Optional[int],
    pretrained: bool,
    cam_method: str,
    arch: Optional[str],
    target_layer_name: Optional[str],
    image_weight: float,
    fig_format: str,
    save_npy: bool,
    max_images: Optional[int],
    cam_batch_size: int,
    device: torch.device,
) -> None:
    """Run CAM heatmap generation for all images in label_csv."""
    import logging
    out_dirs = prepare_output_dirs(out_dir)
    df = load_label_file(label_csv)
    model = load_model_from_args(
        load_ag=str(model_dir) if model_dir else None,
        base_model=base_model,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
    )
    preprocess, display_transform = build_eval_transforms(model)
    inferred_arch = arch or infer_architecture(base_model or "", model)
    cam_extractor, target_layers, reshape_transform = prepare_cam(
        model, inferred_arch, target_layer_name, cam_method, device, cam_batch_size
    )
    records = []
    for idx, row in df.iterrows():
        if max_images and idx >= max_images:
            break
        img_path = (images_dir / row["image"]).resolve()
        if not img_path.exists():
            logging.error("Image not found: %s", img_path)
            continue
        try:
            record = process_image(
                img_path=img_path,
                label=str(row["label"]),
                model=model,
                preprocess=preprocess,
                display_transform=display_transform,
                cam_extractor=cam_extractor,
                device=device,
                fig_dir=out_dirs["fig"],
                array_dir=out_dirs["array"],
                image_weight=image_weight,
                fig_format=fig_format,
                save_npy=save_npy,
            )
            records.append(record)
        except Exception as exc:
            logging.exception("Failed on %s: %s", img_path, exc)
    if records:
        import pandas as pd
        pd.DataFrame(records).to_csv(out_dir / "cam_summary.csv", index=False)
```

- [ ] **Step 2: 写 `entomokit/classify/cam.py`**

```python
"""entomokit classify cam — generate GradCAM heatmaps."""
from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "cam",
        help="Generate GradCAM heatmaps (PyTorch models only, not ONNX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--label-csv", required=True, help="CSV with 'image' and 'label' columns.")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-dir", required=True)
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-dir", help="AutoGluon predictor directory.")
    model_group.add_argument("--base-model", help="timm backbone name.")
    p.add_argument("--checkpoint-path", default=None,
                   help="Custom .pth weights for timm backbone.")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--cam-method", default="gradcam",
                   choices=["gradcam", "gradcampp", "layercam",
                            "scorecam", "eigencam", "ablationcam"])
    p.add_argument("--arch", default=None, choices=["cnn", "vit"],
                   help="Force architecture type (auto-detected if not set).")
    p.add_argument("--target-layer-name", default=None)
    p.add_argument("--image-weight", type=float, default=0.5)
    p.add_argument("--fig-format", default="png",
                   choices=["png", "jpg", "pdf"])
    p.add_argument("--save-npy", action="store_true")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--cam-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.classification.utils import select_device, set_num_threads
    from src.classification.cam import run_cam
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    device = select_device(args.device)
    set_num_threads(args.num_threads)

    run_cam(
        label_csv=Path(args.label_csv),
        images_dir=Path(args.images_dir),
        out_dir=out_dir,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        base_model=args.base_model,
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        cam_method=args.cam_method,
        arch=args.arch,
        target_layer_name=args.target_layer_name,
        image_weight=args.image_weight,
        fig_format=args.fig_format,
        save_npy=args.save_npy,
        max_images=args.max_images,
        cam_batch_size=args.cam_batch_size,
        device=device,
    )
```

- [ ] **Step 3: 冒烟测试**

```bash
python -m entomokit.main classify cam --help
```

Expected: help 含 `--model-dir/--base-model` 互斥组，含 `--save-npy`，含 `--cam-method`。

- [ ] **Step 4: Commit**

```bash
git add src/classification/cam.py entomokit/classify/cam.py
git commit -m "feat: classify cam command (GradCAM heatmaps, PyTorch only)"
```

---

## Task 7: `src/classification/exporter.py` + `classify export-onnx`

**Files:**
- Create: `src/classification/exporter.py`
- Create: `entomokit/classify/export_onnx.py`

- [ ] **Step 1: 写 `src/classification/exporter.py`**

```python
"""ONNX export using AutoGluon's native MultiModalPredictor.export_onnx()."""
from __future__ import annotations

from pathlib import Path


def export_onnx(
    model_dir: Path,
    out_dir: Path,
    opset: int = 17,
    input_size: int = 224,
) -> Path:
    """Export AutoGluon predictor to ONNX format.

    Returns:
        Path to the exported model.onnx file.
    """
    from autogluon.multimodal import MultiModalPredictor

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = MultiModalPredictor.load(str(model_dir))
    predictor.export_onnx(
        save_path=str(out_dir),
        opset=opset,
        input_size=(input_size, input_size),
    )

    onnx_path = out_dir / "model.onnx"
    return onnx_path
```

- [ ] **Step 2: 写 `entomokit/classify/export_onnx.py`**

```python
"""entomokit classify export-onnx — export AutoGluon model to ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "export-onnx",
        help="Export an AutoGluon predictor to ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True,
                   help="AutoGluon predictor directory.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for model.onnx.")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version.")
    p.add_argument("--input-size", type=int, default=224,
                   help="Model input image size (square).")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.classification.exporter import export_onnx
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    save_log(out_dir, args)

    onnx_path = export_onnx(
        model_dir=Path(args.model_dir),
        out_dir=out_dir,
        opset=args.opset,
        input_size=args.input_size,
    )
    print(f"ONNX model saved to: {onnx_path}")
```

- [ ] **Step 3: 冒烟测试**

```bash
python -m entomokit.main classify export-onnx --help
```

- [ ] **Step 4: Commit**

```bash
git add src/classification/exporter.py entomokit/classify/export_onnx.py
git commit -m "feat: classify export-onnx command (AutoGluon native export_onnx)"
```

---

## Task 8: 完整注册 classify 命令组

**Files:**
- Modify: `entomokit/classify/__init__.py` (替换 Phase 1 stub)

- [ ] **Step 1: 替换 stub**

```python
"""classify command group — AutoGluon image classification."""
from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "classify",
        help="Image classification commands (AutoGluon).",
    )
    sub = p.add_subparsers(dest="subcommand", metavar="<subcommand>")
    sub.required = True

    from entomokit.classify import (
        train, predict, evaluate, embed, cam, export_onnx
    )
    train.register(sub)
    predict.register(sub)
    evaluate.register(sub)
    embed.register(sub)
    cam.register(sub)
    export_onnx.register(sub)
```

- [ ] **Step 2: 完整帮助树测试**

```bash
entomokit classify --help
entomokit classify train --help
entomokit classify predict --help
entomokit classify evaluate --help
entomokit classify embed --help
entomokit classify cam --help
entomokit classify export-onnx --help
```

Expected: 全部 6 个子命令有正确帮助输出，无 import error。

- [ ] **Step 3: 运行全套测试**

```bash
pytest tests/ -v --tb=short
```

Expected: 全部 PASS。

- [ ] **Step 4: 最终 commit**

```bash
git add entomokit/classify/__init__.py
git commit -m "feat: complete classify command group registration (train/predict/evaluate/embed/cam/export-onnx)"
```

---

## Phase 3 完成标志

- [ ] `entomokit classify train/predict/evaluate/embed/cam/export-onnx --help` 全部可用
- [ ] `resolve_augment("invalid")` 报错退出
- [ ] `classify embed --visualize` 无 `--label-csv` 时报错退出
- [ ] `classify cam --help` 不含 `--onnx-model` 选项
- [ ] `classify export-onnx` 底层调用 `MultiModalPredictor.export_onnx()`
- [ ] `pytest tests/ -v` 全部通过
