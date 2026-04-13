from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.measurement.core import compute_metrics
from src.measurement.io import iter_mask_files, keep_largest_component, load_binary_mask


def _append_um_columns(row: dict[str, Any], pixel_size_um: float) -> None:
    row["body_length_um"] = row.get("body_length_px", 0.0) * pixel_size_um
    row["body_width_um"] = row.get("body_width_px", 0.0) * pixel_size_um
    row["perimeter_um"] = row.get("perimeter_px", 0.0) * pixel_size_um
    row["area_um2"] = row.get("area_px", 0.0) * (pixel_size_um**2)
    row["max_feret_um"] = row.get("max_feret_px", 0.0) * pixel_size_um
    row["min_feret_um"] = row.get("min_feret_px", 0.0) * pixel_size_um


def measure_one_mask(path: Path, pixel_size_um: float | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"file_name": path.name}
    try:
        mask = load_binary_mask(path)
        mask = keep_largest_component(mask)
        metrics = compute_metrics(mask)
        row.update(metrics)
        if pixel_size_um is not None and row.get("quality_flag") != "fail":
            _append_um_columns(row, pixel_size_um)
    except Exception as exc:
        row.update(
            {
                "quality_flag": "fail",
                "warn_reason": "",
                "error_message": str(exc),
            }
        )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    if "file_name" in keys:
        keys.remove("file_name")
        keys = ["file_name"] + keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _metric_definitions_rows() -> list[dict[str, str]]:
    return [
        {
            "metric": "file_name",
            "unit": "-",
            "zh": "掩码文件名",
            "en": "Mask file name",
            "formula_or_note": "input filename",
        },
        {
            "metric": "quality_flag",
            "unit": "-",
            "zh": "质量标记（ok/warn/fail）",
            "en": "Quality flag (ok/warn/fail)",
            "formula_or_note": "derived from warning and failure checks",
        },
        {
            "metric": "warn_reason",
            "unit": "-",
            "zh": "警告原因（分号分隔）",
            "en": "Warning reasons (semicolon-separated)",
            "formula_or_note": "touching_border;too_many_branches;fallback_rect_used",
        },
        {
            "metric": "error_message",
            "unit": "-",
            "zh": "失败原因",
            "en": "Failure reason",
            "formula_or_note": "set only when quality_flag=fail",
        },
        {
            "metric": "area_px",
            "unit": "px^2",
            "zh": "虫体面积（前景像素数）",
            "en": "Body area (foreground pixel count)",
            "formula_or_note": "count_nonzero(mask)",
        },
        {
            "metric": "perimeter_px",
            "unit": "px",
            "zh": "周长（skimage定义）",
            "en": "Perimeter (skimage definition)",
            "formula_or_note": "skimage.measure.regionprops(...).perimeter",
        },
        {
            "metric": "bbox_w_px",
            "unit": "px",
            "zh": "轴对齐外接框宽度",
            "en": "Axis-aligned bounding box width",
            "formula_or_note": "w from cv2.boundingRect",
        },
        {
            "metric": "bbox_h_px",
            "unit": "px",
            "zh": "轴对齐外接框高度",
            "en": "Axis-aligned bounding box height",
            "formula_or_note": "h from cv2.boundingRect",
        },
        {
            "metric": "aspect_ratio",
            "unit": "-",
            "zh": "长宽比（长/短）",
            "en": "Aspect ratio (long/short)",
            "formula_or_note": "max(w,h) / min(w,h)",
        },
        {
            "metric": "major_axis_px",
            "unit": "px",
            "zh": "主轴长度（PCA近似）",
            "en": "Major axis length (PCA approximation)",
            "formula_or_note": "4*sqrt(lambda_max)",
        },
        {
            "metric": "minor_axis_px",
            "unit": "px",
            "zh": "次轴长度（PCA近似）",
            "en": "Minor axis length (PCA approximation)",
            "formula_or_note": "4*sqrt(lambda_min)",
        },
        {
            "metric": "eccentricity",
            "unit": "-",
            "zh": "离心率",
            "en": "Eccentricity",
            "formula_or_note": "sqrt(1 - lambda_min/lambda_max)",
        },
        {
            "metric": "solidity",
            "unit": "-",
            "zh": "实心度（skimage定义）",
            "en": "Solidity (skimage definition)",
            "formula_or_note": "skimage.measure.regionprops(...).solidity",
        },
        {
            "metric": "extent",
            "unit": "-",
            "zh": "填充度",
            "en": "Extent",
            "formula_or_note": "area / (bbox_w*bbox_h)",
        },
        {
            "metric": "circularity",
            "unit": "-",
            "zh": "圆形度",
            "en": "Circularity",
            "formula_or_note": "4*pi*area/perimeter^2",
        },
        {
            "metric": "convex_area_px",
            "unit": "px^2",
            "zh": "凸包面积（skimage定义）",
            "en": "Convex hull area (skimage definition)",
            "formula_or_note": "skimage.measure.regionprops(...).convex_area",
        },
        {
            "metric": "equivalent_diameter_px",
            "unit": "px",
            "zh": "等效圆直径",
            "en": "Equivalent diameter",
            "formula_or_note": "sqrt(4*area/pi)",
        },
        {
            "metric": "body_length_px",
            "unit": "px",
            "zh": "体长（skimage骨架主干测地长度）",
            "en": "Body length (skimage skeleton backbone geodesic length)",
            "formula_or_note": "longest pruned path on skimage.morphology.skeletonize output; fallback to major_axis",
        },
        {
            "metric": "body_width_px",
            "unit": "px",
            "zh": "体宽（中心线宽度中位数）",
            "en": "Body width (median width along backbone)",
            "formula_or_note": "median(2*distance_transform on backbone); fallback rect short side",
        },
        {
            "metric": "max_feret_px",
            "unit": "px",
            "zh": "最大Feret直径（skimage定义）",
            "en": "Maximum Feret diameter (skimage definition)",
            "formula_or_note": "skimage.measure.regionprops(...).feret_diameter_max",
        },
        {
            "metric": "min_feret_px",
            "unit": "px",
            "zh": "最小Feret直径",
            "en": "Minimum Feret diameter",
            "formula_or_note": "short side of minAreaRect",
        },
        {
            "metric": "curvature_index",
            "unit": "-",
            "zh": "弯曲指数",
            "en": "Curvature index",
            "formula_or_note": "body_length / endpoint_straight_distance",
        },
        {
            "metric": "thickness_cv",
            "unit": "-",
            "zh": "厚度变异系数",
            "en": "Thickness coefficient of variation",
            "formula_or_note": "std(width_series) / mean(width_series)",
        },
        {
            "metric": "symmetry_score",
            "unit": "-",
            "zh": "左右对称得分",
            "en": "Left-right symmetry score",
            "formula_or_note": "IoU between mirrored halves after major-axis alignment",
        },
        {
            "metric": "body_length_um",
            "unit": "um",
            "zh": "体长（微米）",
            "en": "Body length in micrometers",
            "formula_or_note": "body_length_px * pixel_size_um",
        },
        {
            "metric": "body_width_um",
            "unit": "um",
            "zh": "体宽（微米）",
            "en": "Body width in micrometers",
            "formula_or_note": "body_width_px * pixel_size_um",
        },
        {
            "metric": "perimeter_um",
            "unit": "um",
            "zh": "周长（微米）",
            "en": "Perimeter in micrometers",
            "formula_or_note": "perimeter_px * pixel_size_um",
        },
        {
            "metric": "area_um2",
            "unit": "um^2",
            "zh": "面积（平方微米）",
            "en": "Area in square micrometers",
            "formula_or_note": "area_px * pixel_size_um^2",
        },
    ]


def _summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ok = sum(1 for r in rows if r.get("quality_flag") == "ok")
    warn = sum(1 for r in rows if r.get("quality_flag") == "warn")
    fail = sum(1 for r in rows if r.get("quality_flag") == "fail")

    summary: dict[str, Any] = {
        "total": total,
        "ok": ok,
        "warn": warn,
        "fail": fail,
        "ok_rate": ok / total if total else 0.0,
        "warn_rate": warn / total if total else 0.0,
        "fail_rate": fail / total if total else 0.0,
    }

    for name in ["area_px", "perimeter_px", "body_length_px", "body_width_px"]:
        vals = np.array(
            [
                float(r[name])
                for r in rows
                if r.get("quality_flag") != "fail" and name in r
            ],
            dtype=np.float64,
        )
        if vals.size:
            summary[f"{name}_p50"] = float(np.quantile(vals, 0.5))
            summary[f"{name}_p90"] = float(np.quantile(vals, 0.9))
            summary[f"{name}_mean"] = float(np.mean(vals))
        else:
            summary[f"{name}_p50"] = 0.0
            summary[f"{name}_p90"] = 0.0
            summary[f"{name}_mean"] = 0.0

    reason_counts: dict[str, int] = {}
    for row in rows:
        reasons = str(row.get("warn_reason", "")).strip()
        if not reasons:
            continue
        for reason in [x.strip() for x in reasons.split(";") if x.strip()]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items()):
        summary[f"warn_count_{reason}"] = count

    return summary


def run_batch(
    mask_dir: Path, out_dir: Path, pixel_size_um: float | None
) -> dict[str, int]:
    files = iter_mask_files(mask_dir)
    rows = [measure_one_mask(path, pixel_size_um=pixel_size_um) for path in files]
    if not rows:
        rows = [
            {
                "file_name": "",
                "quality_flag": "fail",
                "warn_reason": "",
                "error_message": "no mask files found",
            }
        ]

    _write_csv(out_dir / "metrics.csv", rows)
    summary = _summary_row(rows)
    _write_csv(out_dir / "metrics_summary.csv", [summary])
    _write_csv(out_dir / "metric_definitions.csv", _metric_definitions_rows())
    return {
        "total": int(summary["total"]),
        "ok": int(summary["ok"]),
        "warn": int(summary["warn"]),
        "fail": int(summary["fail"]),
    }
