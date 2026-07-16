"""entomokit update — check for updates from GitHub and optionally install."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from functools import total_ordering

from entomokit._version import __version__

_REPO = "xtmtd/entomokit"
_VERSION_URL = f"https://raw.githubusercontent.com/{_REPO}/main/version.txt"
_TAGS_API_URL = f"https://api.github.com/repos/{_REPO}/tags"
_INSTALL_URL = f"git+https://github.com/{_REPO}.git"


def fetch_remote_version(timeout: int = 10) -> str:
    """Read main's version file, falling back to the latest release tag."""
    try:
        req = urllib.request.Request(
            _VERSION_URL,
            headers={"Cache-Control": "no-cache", "User-Agent": "entomokit-updater"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            version = resp.read().decode().strip()
        if version:
            return version
    except Exception:
        pass

    req = urllib.request.Request(
        _TAGS_API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "entomokit-updater"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        tags = json.loads(resp.read())
    if not tags:
        raise RuntimeError("No tags found in repository.")
    versions = [tag["name"].lstrip("v") for tag in tags if _SEMVER_RE.match(tag["name"].lstrip("v"))]
    if not versions:
        raise RuntimeError("No Semantic Version tags found in repository.")
    return str(max(versions, key=_parse_version))


@total_ordering
@dataclass(frozen=True)
class _SemVer:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SemVer):
            return NotImplemented
        return (self.major, self.minor, self.patch, self.prerelease) == (
            other.major, other.minor, other.patch, other.prerelease,
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _SemVer):
            return NotImplemented
        core = (self.major, self.minor, self.patch)
        other_core = (other.major, other.minor, other.patch)
        if core != other_core:
            return core < other_core
        if not self.prerelease or not other.prerelease:
            return bool(self.prerelease)
        for left, right in zip(self.prerelease, other.prerelease):
            if left == right:
                continue
            if left.isdigit() and right.isdigit():
                return int(left) < int(right)
            if left.isdigit() != right.isdigit():
                return left.isdigit()
            return left < right
        return len(self.prerelease) < len(other.prerelease)


_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def _parse_version(ver: str) -> _SemVer:
    """Parse a Semantic Version; invalid versions sort below valid releases."""
    match = _SEMVER_RE.match(ver.strip()) if isinstance(ver, str) else None
    if not match:
        return _SemVer(0, 0, 0, ("invalid",))
    prerelease = match.group("prerelease")
    return _SemVer(
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        tuple(prerelease.split(".")) if prerelease else (),
    )


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
    print(f"Current version : {local_ver}")
    print("Checking GitHub for updates...")

    try:
        remote_ver = fetch_remote_version()
    except Exception as exc:
        print(f"Error: could not reach GitHub — {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Latest version  : {remote_ver}")

    local_tuple = _parse_version(local_ver)
    remote_tuple = _parse_version(remote_ver)

    if local_tuple >= remote_tuple:
        print("Already up to date.")
        return

    print(f"Update available: {local_ver} -> {remote_ver}")

    if args.check:
        print("(Run without --check to install the update.)")
        return

    if not args.yes:
        answer = input("Proceed with update? [y/N] ").strip().lower()
        if answer != "y":
            print("Update cancelled.")
            return

    print("Installing latest from GitHub ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", _INSTALL_URL],
        check=False,
    )
    if result.returncode == 0:
        print("Update complete. Restart your shell to use the new version.")
    else:
        print("Update failed. Check pip output above.", file=sys.stderr)
        sys.exit(result.returncode)
