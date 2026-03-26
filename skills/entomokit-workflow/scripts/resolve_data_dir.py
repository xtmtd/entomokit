#!/usr/bin/env python3
"""Resolve entomokit example data directory from runtime environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REQUIRED_MARKERS = (
    "video.mp4",
    "Epidorcus/figs.csv",
    "insects",
)


def is_valid_data_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / marker).exists() for marker in REQUIRED_MARKERS)


def candidate_data_dirs_from_path(start: Path) -> list[Path]:
    candidates = []
    for parent in [start] + list(start.parents):
        data_dir = parent / "data"
        candidates.append(data_dir)
    return candidates


def resolve() -> Path | None:
    env_data = os.environ.get("ENTOMOKIT_DATA_DIR")
    if env_data:
        p = Path(env_data).expanduser().resolve()
        if is_valid_data_dir(p):
            return p

    try:
        import inspect
        import entomokit.main as emain

        module_file = Path(inspect.getfile(emain)).resolve()
        for candidate in candidate_data_dirs_from_path(module_file.parent):
            if is_valid_data_dir(candidate):
                return candidate
    except Exception:
        pass

    cwd = Path.cwd().resolve()
    for candidate in candidate_data_dirs_from_path(cwd):
        if is_valid_data_dir(candidate):
            return candidate

    return None


def main() -> int:
    found = resolve()
    if not found:
        print(
            "Could not locate entomokit example data directory. "
            "Set ENTOMOKIT_DATA_DIR to a valid data path.",
            file=sys.stderr,
        )
        return 1

    print(str(found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
