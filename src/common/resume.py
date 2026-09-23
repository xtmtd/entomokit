"""Resume parameter guards.

A ``--resume`` run must not silently mix outputs produced under different
parameters (for example a different measurement scale or frame interval).
``check_resume_params`` records the parameters that affect output and rejects a
resume whose parameters differ from the recorded ones.
"""

from __future__ import annotations

import json
from pathlib import Path

_STATE_DIR = ".entomokit"


def check_resume_params(out_dir, command: str, params: dict) -> None:
    """Record *params* and require a match when a previous run recorded them.

    Raises ``ValueError`` on mismatch so a resume cannot silently produce a
    dataset that mixes outputs from different parameter sets.
    """
    out_dir = Path(out_dir)
    path = out_dir / _STATE_DIR / f"{command}_params.json"
    if path.is_file():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = None
        if isinstance(prior, dict) and prior != params:
            changed = {
                key: [prior.get(key), params.get(key)]
                for key in sorted(set(prior) | set(params))
                if prior.get(key) != params.get(key)
            }
            raise ValueError(
                f"Cannot resume {command}: parameters differ from the previous "
                f"run {changed}. Delete the output directory or keep the "
                "original parameters."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
