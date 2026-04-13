from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.transform import rotate

from src.measurement.skeleton import (
    longest_backbone_path,
    prune_short_branches,
    skeleton_graph,
    skeletonize,
)


def _pca_axes(
    mask: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return 0.0, 0.0, 0.0, 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    l1 = float(max(vals[0], 0.0))
    l2 = float(max(vals[1], 0.0))
    major = 4.0 * math.sqrt(l1) if l1 > 0 else 0.0
    minor = 4.0 * math.sqrt(l2) if l2 > 0 else 0.0
    ecc = math.sqrt(max(0.0, 1.0 - l2 / l1)) if l1 > 0 else 0.0
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    major_vec = vecs[:, 0]
    minor_vec = vecs[:, 1]
    return major, minor, ecc, angle, major_vec, minor_vec


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _estimate_width_series(mask: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    if not path:
        return np.array([], dtype=np.float64)
    dist = distance_transform_edt(mask > 0)
    widths = [2.0 * float(dist[y, x]) for y, x in path if dist[y, x] > 0]
    return np.asarray(widths, dtype=np.float64)


def _symmetry_score(mask: np.ndarray, angle_deg: float) -> float:
    rotated = rotate(
        mask.astype(np.float32),
        angle=-angle_deg,
        order=0,
        preserve_range=True,
    )
    rotated = (rotated > 0.5).astype(np.uint8)
    if not np.any(rotated):
        return 0.0
    cx = int(np.mean(np.where(rotated > 0)[1]))
    left = rotated[:, :cx]
    right = rotated[:, cx:]
    if left.size == 0 or right.size == 0:
        return 0.0
    min_w = min(left.shape[1], right.shape[1])
    left = left[:, left.shape[1] - min_w :]
    right = right[:, :min_w]
    right_flip = np.fliplr(right)
    inter = np.logical_and(left > 0, right_flip > 0).sum()
    union = np.logical_or(left > 0, right_flip > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def _feret_min_from_projection(mask: np.ndarray, minor_vec: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    proj = pts @ minor_vec
    return float(np.max(proj) - np.min(proj)) if proj.size else 0.0


def _largest_region(mask: np.ndarray):
    labeled = measure.label(mask > 0, connectivity=2)
    props = measure.regionprops(labeled)
    if not props:
        return None
    return max(props, key=lambda p: p.area)


def compute_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Compute morphology metrics from a single binary mask (scikit-image definitions)."""
    result: dict[str, Any] = {"quality_flag": "ok", "warn_reason": ""}
    warn: list[str] = []
    bin_mask = (mask > 0).astype(np.uint8)

    rp = _largest_region(bin_mask)
    if rp is None:
        result.update({"quality_flag": "fail", "warn_reason": "no_region"})
        return result

    area = float(rp.area)
    perimeter = float(rp.perimeter)
    minr, minc, maxr, maxc = rp.bbox
    w = int(maxc - minc)
    h = int(maxr - minr)

    major, minor, ecc, angle, _major_vec, minor_vec = _pca_axes(bin_mask)
    major = float(rp.major_axis_length)
    minor = float(rp.minor_axis_length)
    ecc = float(rp.eccentricity)

    max_feret = float(getattr(rp, "feret_diameter_max", major))
    min_feret = _feret_min_from_projection(bin_mask, minor_vec)

    diag = math.hypot(*bin_mask.shape)
    skel = skeletonize(bin_mask)
    graph = skeleton_graph(skel)
    if graph:
        graph = prune_short_branches(graph, min_len=max(3.0, 0.06 * diag))
    path, body_length = longest_backbone_path(graph)

    if not path or body_length <= 0:
        body_length = major
        widths = np.array([minor], dtype=np.float64)
        warn.append("fallback_rect_used")
    else:
        widths = _estimate_width_series(bin_mask, path)

    body_length = max(float(body_length), major)
    body_width = float(np.median(widths)) if widths.size else 0.0
    thickness_cv = (
        float(np.std(widths) / np.mean(widths))
        if widths.size and np.mean(widths) > 0
        else 0.0
    )

    if _touches_border(bin_mask):
        warn.append("touching_border")
    branch_nodes = [n for n in graph if len(graph[n]) >= 3]
    if len(branch_nodes) > 6:
        warn.append("too_many_branches")

    endpoints = [path[0], path[-1]] if len(path) >= 2 else []
    if len(endpoints) == 2:
        straight = float(
            np.hypot(
                endpoints[0][0] - endpoints[1][0], endpoints[0][1] - endpoints[1][1]
            )
        )
    else:
        straight = 0.0
    curvature = float(body_length / straight) if straight > 0 else 1.0

    if warn:
        result["quality_flag"] = "warn"
        result["warn_reason"] = ";".join(sorted(set(warn)))

    convex_area = float(rp.convex_area)
    solidity = float(rp.solidity)
    extent = float(rp.extent)
    equivalent_diameter = float(rp.equivalent_diameter_area)

    result.update(
        {
            "area_px": area,
            "perimeter_px": perimeter,
            "bbox_w_px": float(w),
            "bbox_h_px": float(h),
            "aspect_ratio": float(max(w, h) / max(1.0, min(w, h))),
            "major_axis_px": major,
            "minor_axis_px": minor,
            "eccentricity": ecc,
            "solidity": solidity,
            "extent": extent,
            "circularity": float((4.0 * math.pi * area) / (perimeter * perimeter))
            if perimeter > 0
            else 0.0,
            "convex_area_px": convex_area,
            "equivalent_diameter_px": equivalent_diameter,
            "body_length_px": float(body_length),
            "body_width_px": float(body_width),
            "max_feret_px": float(max_feret),
            "min_feret_px": float(max(0.0, min_feret)),
            "curvature_index": float(max(1.0, curvature)),
            "thickness_cv": float(thickness_cv),
            "symmetry_score": float(_symmetry_score(bin_mask, angle)),
        }
    )
    return result
