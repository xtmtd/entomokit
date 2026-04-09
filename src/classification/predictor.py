"""Image classification inference — AutoGluon and ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import json

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(iterable, **_kwargs):
        return iterable


def load_onnx_class_labels(onnx_path: Path) -> list[str] | None:
    meta_path = Path(onnx_path).parent / "label_classes.json"
    if not meta_path.is_file():
        return None

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = payload.get("class_labels")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
    except Exception:
        return None

    return None


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
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run inference with ONNX model.
    Returns DataFrame with columns: image, prediction, proba_*.
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ONNX inference requires 'onnxruntime'. Install with "
            "`pip install onnxruntime` or `pip install 'entomokit[classify]'`."
        ) from exc
    from PIL import Image
    import torchvision.transforms as T

    onnx_path = Path(onnx_path)
    if onnx_path.is_dir():
        onnx_path = onnx_path / "model.onnx"
    elif not onnx_path.is_file() and (onnx_path / "model.onnx").is_file():
        onnx_path = onnx_path / "model.onnx"

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    class_labels = load_onnx_class_labels(onnx_path)

    sess_options = ort.SessionOptions()
    if num_threads > 0:
        sess_options.intra_op_num_threads = num_threads
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options)

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_metas = sess.get_inputs()
    all_proba = []

    paths = input_df["image"].tolist()
    batch_starts = range(0, len(paths), batch_size)
    if show_progress:
        batch_starts = tqdm(batch_starts, desc="ONNX infer", unit="batch")

    for i in batch_starts:
        batch_paths = paths[i : i + batch_size]
        tensors = []
        for p in batch_paths:
            full_p = (images_dir / p) if images_dir else Path(p)
            img = Image.open(full_p).convert("RGB")
            tensors.append(transform(img).numpy())
        batch = np.stack(tensors)
        feed = {}
        for meta in input_metas:
            name = meta.name
            if "valid_num" in name.lower():
                feed[name] = np.ones((batch.shape[0],), dtype=np.int64)
                continue

            value = batch
            rank = len(meta.shape) if getattr(meta, "shape", None) is not None else None
            if batch.ndim == 4 and (rank == 5 or name.lower().endswith("_image")):
                value = batch[:, None, ...]
            feed[name] = value

        outputs = sess.run(None, feed)
        logits_candidates = [
            out
            for out in outputs
            if isinstance(out, np.ndarray)
            and out.ndim == 2
            and out.shape[0] == batch.shape[0]
            and out.shape[1] > 1
        ]
        logits = (
            min(logits_candidates, key=lambda arr: arr.shape[1])
            if logits_candidates
            else outputs[0]
        )
        # softmax
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
        all_proba.append(proba)

    all_proba_arr = np.vstack(all_proba)
    n_classes = all_proba_arr.shape[1]

    result = input_df.copy()
    prediction_index = all_proba_arr.argmax(axis=1)
    result["prediction_index"] = prediction_index
    if class_labels and max(prediction_index, default=0) < len(class_labels):
        result["prediction"] = [class_labels[i] for i in prediction_index]
    else:
        result["prediction"] = prediction_index
    for i in range(n_classes):
        result[f"proba_{i}"] = all_proba_arr[:, i]
    return result
