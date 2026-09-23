"""Shared recursive file scanning and input-relative output path helpers.

Directory-oriented commands use these helpers so that every command shares one
recursive-scan and mirrored-output policy. CSV-driven commands keep their own
explicit path lists and do not use this module.
"""

from __future__ import annotations

from pathlib import Path

from src.common.validators import IMAGE_EXTENSIONS as _IMAGE_EXTENSIONS
from src.common.validators import VIDEO_EXTENSIONS as _VIDEO_EXTENSIONS

# Re-exported as the single source of truth for accepted extensions.
IMAGE_EXTENSIONS: frozenset = frozenset(_IMAGE_EXTENSIONS)
VIDEO_EXTENSIONS: frozenset = frozenset(_VIDEO_EXTENSIONS)


def iter_files(root: Path, extensions) -> list:
    """Return sorted files below root recursively, filtered case-insensitively.

    Symlinked directories are not followed (pathlib's default for ``rglob``).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    exts = {e.lower() for e in extensions}
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files


def relative_output_path(source: Path, input_root: Path, output_root: Path) -> Path:
    """Map source to output_root/source.relative_to(input_root)."""
    source = Path(source)
    input_root = Path(input_root)
    output_root = Path(output_root)
    try:
        rel = source.relative_to(input_root)
    except ValueError:
        raise ValueError(
            f"Source path is outside the input root: {source} (root: {input_root})"
        ) from None
    return output_root / rel
