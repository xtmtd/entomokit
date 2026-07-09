"""entomokit update — check for updates from GitHub and optionally install."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from typing import Tuple

from entomokit._version import __version__, __commit__, __commit_date__

_REPO = "xtmtd/entomokit"
_API_URL = f"https://api.github.com/repos/{_REPO}/commits/main"
_INSTALL_URL = f"git+https://github.com/{_REPO}.git"


def _local_commit() -> str:
    if __commit__ != "unknown":
        return __commit__
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or "unknown"
    except Exception:
        return "unknown"


def _local_commit_date() -> str:
    if __commit_date__ != "unknown":
        return __commit_date__
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()[:10] or "unknown"
    except Exception:
        return "unknown"


def fetch_latest_commit(timeout: int = 10) -> Tuple[str, str, str]:
    """Return (short_sha, date_str, first_line_of_message)."""
    req = urllib.request.Request(
        _API_URL,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    entry = data
    sha = entry["sha"][:7]
    date = entry["commit"]["author"]["date"][:10]   # YYYY-MM-DD
    message = entry["commit"]["message"].splitlines()[0]
    return sha, date, message


def _status(local_commit: str, local_date: str, remote_sha: str, remote_date: str) -> str:
    """Return same, newer, or unknown."""
    if local_commit == remote_sha:
        return "same"
    if local_commit == "unknown" or local_date == "unknown":
        return "unknown"
    return "newer" if remote_date > local_date else "same"


def register(subparsers: argparse._SubParsersAction) -> None:
    from entomokit.help_style import RichHelpFormatter, style_parser

    p = subparsers.add_parser(
        "update",
        help="Check for updates and optionally install the latest version from GitHub.",
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument(
        "--check",
        action="store_true",
        help="Only show version information; do not install.",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt and install immediately.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    local_ver = __version__
    local_sha = _local_commit()
    local_date = _local_commit_date()
    print(f"Current version : {local_ver} ({local_sha})")
    print("Checking GitHub for updates...")

    try:
        remote_sha, remote_date, remote_msg = fetch_latest_commit()
    except Exception as exc:
        print(f"Error: could not reach GitHub — {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Latest commit   : {remote_sha}  ({remote_date})  {remote_msg}")

    status = _status(local_sha, local_date, remote_sha, remote_date)
    if status == "same":
        print("Already up to date.")
        return

    if status == "unknown" and not args.yes:
        print("Local commit/date is unknown. Re-run with --yes to install anyway.")
        return

    if args.check:
        print("(Run without --check to install the update.)")
        return

    if not args.yes:
        answer = input("Proceed with update? [y/N] ").strip().lower()
        if answer != "y":
            print("Update cancelled.")
            return

    print(f"Installing latest from GitHub ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", _INSTALL_URL],
        check=False,
    )
    if result.returncode == 0:
        print("Update complete. Restart your shell to use the new version.")
    else:
        print("Update failed. Check pip output above.", file=sys.stderr)
        sys.exit(result.returncode)
