"""Tests for synthesis count semantics and seeded task planning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.synthesis.processor import (
    SynthesisProcessor,
    derive_seed,
    resolve_num_syntheses,
)


def test_resolve_integer_selects_all_targets_and_count():
    assert resolve_num_syntheses(3, 4) == ([0, 1, 2, 3], 3)


def test_resolve_one_and_one_point_zero_mean_one_each():
    assert resolve_num_syntheses(1, 4) == ([0, 1, 2, 3], 1)
    assert resolve_num_syntheses(1.0, 4) == ([0, 1, 2, 3], 1)


def test_resolve_fraction_is_deterministic_subset():
    indices, per_target = resolve_num_syntheses(0.5, 4, seed=42)
    assert per_target == 1
    assert len(indices) == 2
    assert indices == resolve_num_syntheses(0.5, 4, seed=42)[0]


def test_resolve_small_fraction_selects_at_least_one_target():
    indices, per_target = resolve_num_syntheses(0.01, 3, seed=1)
    assert per_target == 1
    assert len(indices) == 1


@pytest.mark.parametrize("bad", [0, -1, 2.5, float("nan"), float("inf")])
def test_resolve_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        resolve_num_syntheses(bad, 4)


def test_derive_seed_is_stable_and_varies():
    assert derive_seed(42, 0) == derive_seed(42, 0)
    assert derive_seed(42, 0) != derive_seed(42, 1)


def _write_targets(root: Path, names) -> None:
    for name in names:
        d = root / Path(name).parent
        d.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGBA", (20, 20), (200, 30, 30, 0))
        for x in range(5, 15):
            for y in range(5, 15):
                img.putpixel((x, y), (200, 30, 30, 255))
        img.save(root / name)


def _write_backgrounds(root: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        arr = rng.integers(20, 200, size=(60, 60, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / f"bg{i}.png")


def _processor() -> SynthesisProcessor:
    return SynthesisProcessor(
        output_format="png",
        annotation_format="none",
        area_ratio_min=0.05,
        area_ratio_max=0.2,
        rotate_degrees=0.0,
    )


def test_integer_count_has_no_background_cap(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png", "b.png"])
    _write_backgrounds(bg_dir, 1)

    result = _processor().process_directory(
        target_dir, bg_dir, tmp_path / "out", num_syntheses=3, threads=1
    )
    assert result["processed"] == 6


def test_fractional_count_generates_one_per_selected_target(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png", "b.png", "c.png", "d.png"])
    _write_backgrounds(bg_dir, 2)

    result = _processor().process_directory(
        target_dir, bg_dir, tmp_path / "out", num_syntheses=0.5, threads=1, seed=42
    )
    assert result["processed"] == 2


def test_seeded_runs_are_reproducible(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png", "b.png"])
    _write_backgrounds(bg_dir, 2)

    for out_name in ("out1", "out2"):
        _processor().process_directory(
            target_dir, bg_dir, tmp_path / out_name, num_syntheses=2, threads=1, seed=7
        )

    out1 = sorted((tmp_path / "out1" / "images").glob("*.png"))
    out2 = sorted((tmp_path / "out2" / "images").glob("*.png"))
    assert [p.name for p in out1] == [p.name for p in out2]
    assert [p.read_bytes() for p in out1] == [p.read_bytes() for p in out2]


def test_nested_duplicate_target_names_get_distinct_output_dirs(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a/same.png", "b/same.png"])
    _write_backgrounds(bg_dir, 1)

    result = _processor().process_directory(
        target_dir, bg_dir, tmp_path / "out", num_syntheses=1, threads=1
    )
    assert result["processed"] == 2
    assert (tmp_path / "out" / "images" / "a" / "same_01.png").exists()
    assert (tmp_path / "out" / "images" / "b" / "same_01.png").exists()


def test_all_rgb_targets_error_lists_observed_modes(tmp_path):
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    Image.new("RGB", (20, 20), (10, 20, 30)).save(target_dir / "a.jpg")
    Image.new("RGB", (20, 20), (10, 20, 30)).save(target_dir / "b.jpg")
    bg_dir = tmp_path / "bg"
    _write_backgrounds(bg_dir, 1)

    with pytest.raises(ValueError, match=r"2 of 2 selected target image\(s\) failed to load \(observed modes: 2 RGB\)"):
        _processor().process_directory(
            target_dir, bg_dir, tmp_path / "out", num_syntheses=1, threads=1
        )


def test_resume_ignores_unrelated_same_prefix_files(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)

    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    (out / "images" / "a_backup.png").write_bytes(b"")

    result = _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=1, threads=1, skip_existing=True
    )
    assert result["processed"] == 1
    assert result["skipped"] == 0
    assert (out / "images" / "a_01.png").exists()


def test_resume_does_not_skip_partially_completed_target(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)

    out = tmp_path / "out"
    _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=3, threads=1, seed=5
    )
    (out / "images" / "a_02.png").unlink()

    resumed = _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=3, threads=1, seed=5, skip_existing=True
    )
    assert resumed["skipped"] == 0
    assert resumed["processed"] == 3
    assert resumed["failed"] == 0
    assert (out / "images" / "a_02.png").exists()


def test_resume_skips_fully_completed_target(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)

    out = tmp_path / "out"
    first = _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=3, threads=1, seed=5
    )
    assert first["processed"] == 3
    assert first["failed"] == 0

    resumed = _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=3, threads=1, seed=5, skip_existing=True
    )
    assert resumed["skipped"] == 3
    assert resumed["processed"] == 0
    assert resumed["failed"] == 0



def test_same_stem_different_extension_does_not_overwrite(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png", "a.tif"])
    _write_backgrounds(bg_dir, 1)

    out = tmp_path / "out"
    result = _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=1, threads=1, seed=1
    )

    assert result["processed"] == 2
    produced = sorted(p.name for p in (out / "images").glob("*.png"))
    assert len(produced) == 2
    assert produced[0].startswith("a__") and produced[1].startswith("a__")


def test_same_stem_collision_resume_uses_resolved_names(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png", "a.tif"])
    _write_backgrounds(bg_dir, 1)

    out = tmp_path / "out"
    _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=2, threads=1, seed=1
    )
    resumed = _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=2, threads=1, seed=1, skip_existing=True
    )
    assert resumed["skipped"] == 4
    assert resumed["processed"] == 0
    assert resumed["failed"] == 0


def test_unreadable_target_counts_as_uncreated_not_failed(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    Image.new("RGB", (20, 20), (10, 20, 30)).save(target_dir / "b.png")
    _write_backgrounds(bg_dir, 1)

    result = _processor().process_directory(
        target_dir, bg_dir, tmp_path / "out", num_syntheses=1, threads=1
    )
    assert result["processed"] == 1
    assert result["failed"] == 0
    assert result["uncreated"] == 1


def test_failed_background_counts_as_uncreated(tmp_path, monkeypatch):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)

    processor = _processor()
    monkeypatch.setattr(
        processor, "_load_image", lambda _p: (_ for _ in ()).throw(ValueError("bad"))
    )
    result = processor.process_directory(
        target_dir, bg_dir, tmp_path / "out", num_syntheses=2, threads=1
    )
    assert result["processed"] == 0
    assert result["failed"] == 0
    assert result["uncreated"] == 2


def test_resume_preserves_skipped_coco_annotations(tmp_path):
    import json

    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)
    out = tmp_path / "out"

    def _run(skip):
        return SynthesisProcessor(
            output_format="png", annotation_format="coco", rotate_degrees=0.0
        ).process_directory(
            target_dir, bg_dir, out, num_syntheses=1, threads=1, seed=1,
            skip_existing=skip,
        )

    _run(skip=False)
    before = json.loads((out / "annotations.coco.json").read_text(encoding="utf-8"))
    assert len(before["images"]) == 1

    # Pure resume must not drop the existing annotation.
    result = _run(skip=True)
    assert result["processed"] == 0
    after = json.loads((out / "annotations.coco.json").read_text(encoding="utf-8"))
    assert after["images"] == before["images"]
    assert after["annotations"] == before["annotations"]

    # Adding a target must merge and keep the skipped sample's annotation.
    _write_targets(target_dir, ["c.png"])
    result = _run(skip=True)
    merged = json.loads((out / "annotations.coco.json").read_text(encoding="utf-8"))
    assert result["skipped"] == 1
    assert result["processed"] == 1
    assert sorted(i["file_name"] for i in merged["images"]) == ["a_01.png", "c_01.png"]
    assert len(merged["annotations"]) == 2


def test_synthesize_resume_rejects_coco_bbox_format_switch(tmp_path):
    import json

    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)
    out = tmp_path / "out"

    def _run(skip, bbox_format):
        return SynthesisProcessor(
            output_format="png",
            annotation_format="coco",
            rotate_degrees=0.0,
            coco_bbox_format=bbox_format,
        ).process_directory(
            target_dir, bg_dir, out, num_syntheses=1, threads=1, seed=1,
            skip_existing=skip,
        )

    _run(skip=False, bbox_format="xywh")
    before = (out / "annotations.coco.json").read_text(encoding="utf-8")
    assert json.loads(before)["info"]["bbox_format"] == "xywh"

    _write_targets(target_dir, ["c.png"])
    with pytest.raises(ValueError, match="coco-bbox-format"):
        _run(skip=True, bbox_format="xyxy")

    # The switch fails before anything is written; the old JSON is untouched.
    assert (out / "annotations.coco.json").read_text(encoding="utf-8") == before
    assert not (out / "images" / "c_01.png").exists()


def test_resume_reduced_num_syntheses_removes_stale_copy(tmp_path):
    import json

    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)
    out = tmp_path / "out"

    def _run(num, skip):
        return SynthesisProcessor(
            output_format="png", annotation_format="coco", rotate_degrees=0.0
        ).process_directory(
            target_dir, bg_dir, out, num_syntheses=num, threads=1, seed=5,
            skip_existing=skip,
        )

    _run(3, skip=False)
    assert (out / "images" / "a_03.png").exists()

    resumed = _run(2, skip=True)
    assert resumed["skipped"] == 0
    assert resumed["processed"] == 2
    assert sorted(p.name for p in (out / "images").glob("a_*.png")) == [
        "a_01.png",
        "a_02.png",
    ]
    after = json.loads((out / "annotations.coco.json").read_text(encoding="utf-8"))
    assert sorted(i["file_name"] for i in after["images"]) == ["a_01.png", "a_02.png"]


def test_resume_reduced_count_keeps_old_copies_when_target_unreadable(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)
    out = tmp_path / "out"

    def _proc():
        return SynthesisProcessor(
            output_format="png", annotation_format="voc", rotate_degrees=0.0
        )

    _proc().process_directory(
        target_dir, bg_dir, out, num_syntheses=3, threads=1, seed=5
    )
    assert (out / "images" / "a_03.png").exists()
    assert (out / "Annotations" / "a_03.xml").exists()

    # Make the target unreadable, then resume with fewer copies. The failure
    # path must not delete the existing copy set.
    Image.new("RGB", (20, 20), (1, 2, 3)).save(target_dir / "a.png")
    with pytest.raises(ValueError, match="produced no tasks"):
        _proc().process_directory(
            target_dir, bg_dir, out, num_syntheses=2, threads=1, seed=5,
            skip_existing=True,
        )
    assert (out / "images" / "a_03.png").exists()
    assert (out / "Annotations" / "a_03.xml").exists()


def test_resume_ignores_non_canonical_index(tmp_path):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    (out / "images" / "a_1.png").write_bytes(b"")

    result = _processor().process_directory(
        target_dir, bg_dir, out, num_syntheses=1, threads=1, skip_existing=True
    )
    assert result["skipped"] == 0
    assert result["processed"] == 1
    assert (out / "images" / "a_01.png").exists()


def test_resume_reduced_count_keeps_voc_copies_when_background_unreadable(
    tmp_path, monkeypatch
):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)
    out = tmp_path / "out"

    def _proc():
        return SynthesisProcessor(
            output_format="png", annotation_format="voc", rotate_degrees=0.0
        )

    _proc().process_directory(
        target_dir, bg_dir, out, num_syntheses=3, threads=1, seed=5
    )
    assert (out / "images" / "a_03.png").exists()
    assert (out / "Annotations" / "a_03.xml").exists()

    failing = _proc()
    monkeypatch.setattr(
        failing,
        "_load_image",
        lambda _p: (_ for _ in ()).throw(ValueError("bad background")),
    )
    result = failing.process_directory(
        target_dir, bg_dir, out, num_syntheses=2, threads=1, seed=5,
        skip_existing=True,
    )
    assert result["processed"] == 0
    assert (out / "images" / "a_03.png").exists()
    assert (out / "Annotations" / "a_03.xml").exists()


def test_resume_reduced_count_keeps_coco_entries_when_background_unreadable(
    tmp_path, monkeypatch
):
    target_dir = tmp_path / "targets"
    bg_dir = tmp_path / "bg"
    _write_targets(target_dir, ["a.png"])
    _write_backgrounds(bg_dir, 1)
    out = tmp_path / "out"

    def _proc():
        return SynthesisProcessor(
            output_format="png", annotation_format="coco", rotate_degrees=0.0
        )

    _proc().process_directory(
        target_dir, bg_dir, out, num_syntheses=3, threads=1, seed=5
    )
    before = (out / "annotations.coco.json").read_text(encoding="utf-8")
    assert (out / "images" / "a_03.png").exists()

    failing = _proc()
    monkeypatch.setattr(
        failing,
        "_load_image",
        lambda _p: (_ for _ in ()).throw(ValueError("bad background")),
    )
    failing.process_directory(
        target_dir, bg_dir, out, num_syntheses=2, threads=1, seed=5,
        skip_existing=True,
    )
    assert (out / "images" / "a_03.png").exists()
    assert (out / "annotations.coco.json").read_text(encoding="utf-8") == before
