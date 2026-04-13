from __future__ import annotations

import csv

import cv2
import numpy as np


def _write_test_masks(mask_dir) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    m1 = np.zeros((80, 80), dtype=np.uint8)
    m1[20:60, 25:55] = 255
    cv2.imwrite(str(mask_dir / "mask_01.png"), m1)

    m2 = np.zeros((80, 80), dtype=np.uint8)
    cv2.ellipse(m2, (40, 40), (20, 10), 30, 0, 360, 255, -1)
    cv2.imwrite(str(mask_dir / "mask_02.png"), m2)


def test_measure_writes_metrics_and_summary_csv(tmp_path) -> None:
    from entomokit.main import main

    mask_dir = tmp_path / "masks"
    out_dir = tmp_path / "out"
    _write_test_masks(mask_dir)

    main(["measure", "--mask-dir", str(mask_dir), "--out-dir", str(out_dir)])

    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "metrics_summary.csv").exists()
    assert (out_dir / "metric_definitions.csv").exists()

    with (out_dir / "metrics.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "body_length_px" in rows[0]
    assert "quality_flag" in rows[0]

    with (out_dir / "metric_definitions.csv").open("r", encoding="utf-8") as f:
        defs = list(csv.DictReader(f))
    assert defs
    assert {"metric", "unit", "zh", "en", "formula_or_note"}.issubset(defs[0].keys())


def test_measure_adds_um_columns_when_scale_provided(tmp_path) -> None:
    from entomokit.main import main

    mask_dir = tmp_path / "masks"
    out_dir = tmp_path / "out"
    _write_test_masks(mask_dir)

    main(
        [
            "measure",
            "--mask-dir",
            str(mask_dir),
            "--out-dir",
            str(out_dir),
            "--pixel-size-um",
            "2.5",
        ]
    )

    with (out_dir / "metrics.csv").open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert "body_length_um" in row
    assert "body_width_um" in row
    assert "area_um2" in row


def test_measure_threads_default_is_8() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["measure", "--mask-dir", "m", "--out-dir", "o"])
    assert args.threads == 8


def test_measure_prints_length_width_caution(tmp_path, caplog) -> None:
    from entomokit.main import main

    mask_dir = tmp_path / "masks"
    out_dir = tmp_path / "out"
    _write_test_masks(mask_dir)

    main(["measure", "--mask-dir", str(mask_dir), "--out-dir", str(out_dir)])

    assert "body_length/body_width are estimates" in caplog.text
