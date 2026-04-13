from __future__ import annotations

import cv2
import numpy as np

from src.measurement.core import compute_metrics
from src.measurement.io import keep_largest_component


def _curved_larva_mask(size: int = 160) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.array(
        [
            [20, 120],
            [40, 95],
            [65, 78],
            [95, 70],
            [122, 84],
            [140, 110],
        ],
        dtype=np.int32,
    )
    cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def test_keep_largest_component_discards_small_islands() -> None:
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:20, 5:20] = 1
    mask[0:2, 0:2] = 1
    kept = keep_largest_component(mask)
    assert int(kept.sum()) == 15 * 15


def test_basic_metrics_for_axis_aligned_rectangle() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 1
    m = compute_metrics(mask)
    assert m["quality_flag"] in {"ok", "warn"}
    assert abs(m["area_px"] - (60 * 40)) < 5
    assert m["major_axis_px"] >= m["minor_axis_px"]
    assert 0 < m["circularity"] < 1


def test_curved_body_length_exceeds_major_axis() -> None:
    mask = _curved_larva_mask()
    m = compute_metrics(mask)
    assert m["body_length_px"] >= m["major_axis_px"]
    assert m["curvature_index"] >= 1.0


def test_rotation_preserves_area_and_length_approximately() -> None:
    mask = _curved_larva_mask(180)
    m1 = compute_metrics(mask)

    rot = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    m2 = compute_metrics(rot)

    assert abs(m1["area_px"] - m2["area_px"]) / max(1.0, m1["area_px"]) < 0.05
    assert (
        abs(m1["body_length_px"] - m2["body_length_px"])
        / max(1.0, m1["body_length_px"])
        < 0.15
    )
