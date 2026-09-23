"""Run augmentation pipelines with deterministic seeding."""

from __future__ import annotations

import random
from typing import Optional

import albumentations as A
import numpy as np


def run_pipeline(
    pipeline: A.Compose,
    image: np.ndarray,
    seed: Optional[int] = None,
) -> dict:
    """Apply an albumentations Compose pipeline to an image."""
    if seed is not None:
        # albumentations >=2 keeps its own RNG on the pipeline, so seeding the
        # global random modules alone is not reproducible.
        set_pipeline_seed = getattr(pipeline, "set_random_seed", None)
        if set_pipeline_seed is not None:
            set_pipeline_seed(seed)
        py_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)

    result = pipeline(image=image)

    if seed is not None:
        random.setstate(py_state)
        np.random.set_state(np_state)

    return result
