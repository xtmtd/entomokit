"""Tests for clean --pad-color."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.cleaning.processor import ImageCleaner, pad_to_square


def _run_cleaner(tmp_path: Path, img: Image.Image, **kwargs) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    src = input_dir / "sample.png"
    img.save(src)
    out_dir = tmp_path / "output"
    cleaner = ImageCleaner(
        input_dir=str(input_dir),
        output_dir=str(out_dir),
        dedup_mode="none",
        threads=1,
        **kwargs,
    )
    cleaner.process_directory(log_path=str(tmp_path / "log.txt"))
    return next(out_dir.rglob("*"))


def test_pad_none_keeps_resized_dimensions(tmp_path):
    out = _run_cleaner(tmp_path, Image.new("RGB", (4, 2), (10, 20, 30)))
    with Image.open(out) as img:
        assert img.size == (4, 2)


def test_pad_black_and_white_produce_square_with_exact_corners(tmp_path):
    for color, expected in (("black", (0, 0, 0)), ("white", (255, 255, 255))):
        case = tmp_path / color
        out = _run_cleaner(
            case,
            Image.new("RGB", (4, 2), (123, 45, 67)),
            pad_color=color,
            out_image_format="png",
        )
        with Image.open(out).convert("RGB") as img:
            assert img.size == (4, 4)
            assert img.getpixel((0, 0)) == expected
            assert img.getpixel((3, 3)) == expected


def test_pad_median_uses_border_median(tmp_path):
    img = Image.new("RGB", (3, 1))
    for x, value in enumerate([(0, 0, 0), (100, 100, 100), (200, 200, 200)]):
        img.putpixel((x, 0), value)
    out = _run_cleaner(
        tmp_path, img, pad_color="median", out_image_format="png"
    )
    with Image.open(out).convert("RGB") as result:
        assert result.size == (3, 3)
        assert result.getpixel((0, 0)) == (100, 100, 100)


def test_pad_uses_original_long_edge_when_no_resize(tmp_path):
    out = _run_cleaner(
        tmp_path,
        Image.new("RGB", (8, 4), (1, 2, 3)),
        pad_color="black",
        out_short_size=-1,
        out_image_format="png",
    )
    with Image.open(out) as img:
        assert img.size == (8, 8)


def test_transparent_png_is_composited_before_jpeg_save(tmp_path):
    img = Image.new("RGBA", (4, 2), (255, 0, 0, 0))
    out = _run_cleaner(tmp_path, img, pad_color="white", out_image_format="jpg")
    assert out.suffix == ".jpg"
    with Image.open(out) as result:
        assert result.mode == "RGB"
        r, g, b = result.convert("RGB").getpixel((0, 0))
        assert r > 250 and g > 250 and b > 250


def test_pad_to_square_none_is_identity():
    img = Image.new("RGB", (4, 2))
    assert pad_to_square(img, "none") is img


def test_clean_parser_pad_color_default_and_choices() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["clean", "--input-dir", "in", "--out-dir", "out"])
    assert args.pad_color == "none"

    for choice in ("none", "median", "black", "white"):
        parsed = parser.parse_args(
            ["clean", "--input-dir", "in", "--out-dir", "out", "--pad-color", choice]
        )
        assert parsed.pad_color == choice
