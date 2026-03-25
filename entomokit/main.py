"""entomokit — unified CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def _ensure_project_root_on_path() -> None:
    """Ensure local project root is importable before similarly named packages."""
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _detect_shell() -> str:
    shell = Path(os.environ.get("SHELL", "")).name
    if shell in {"bash", "zsh", "fish"}:
        return shell
    return "bash"


def _rc_path(shell: str) -> Path:
    home = Path.home()
    if shell == "zsh":
        return home / ".zshrc"
    return home / ".bashrc"


def _completion_snippet(shell: str) -> str:
    if shell == "zsh":
        return (
            "autoload -U +X bashcompinit && bashcompinit\n"
            'eval "$(register-python-argcomplete entomokit)"'
        )
    return 'eval "$(register-python-argcomplete entomokit)"'


def _install_completion() -> int:
    try:
        import argcomplete  # noqa: F401
    except ImportError:
        print(
            "Completion requires 'argcomplete'. Install it with: pip install argcomplete",
            file=sys.stderr,
        )
        return 1

    shell = _detect_shell()
    if shell == "fish":
        fish_dir = Path.home() / ".config" / "fish" / "completions"
        fish_dir.mkdir(parents=True, exist_ok=True)
        completion_file = fish_dir / "entomokit.fish"
        completion_file.write_text(
            "register-python-argcomplete --shell fish entomokit | source\n",
            encoding="utf-8",
        )
        print(f"Installed fish completion at: {completion_file}")
        return 0

    rc_path = _rc_path(shell)
    snippet = _completion_snippet(shell)
    block = (
        f"# >>> entomokit completion >>>\n{snippet}\n# <<< entomokit completion <<<\n"
    )

    existing = rc_path.read_text(encoding="utf-8") if rc_path.exists() else ""
    if "# >>> entomokit completion >>>" in existing:
        print(f"Completion already configured in: {rc_path}")
        return 0

    with rc_path.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(block)

    print(f"Installed {shell} completion in: {rc_path}")
    print(f"Run: source {rc_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    description = with_examples(
        "A toolkit for building insect image datasets.",
        [
            "entomokit segment --input-dir ./images --out-dir ./out",
            "entomokit extract-frames --input-dir ./video.mp4 --out-dir ./frames",
            "entomokit classify train --train-csv train.csv --images-dir ./images --out-dir ./model",
        ],
    )
    parser = argparse.ArgumentParser(
        prog="entomokit",
        description=description,
        formatter_class=RichHelpFormatter,
    )
    style_parser(parser)
    parser.add_argument(
        "--install-completion",
        action="store_true",
        help="Install shell completion for entomokit.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        title="[ Commands ]",
    )

    subparsers.required = False
    # Lazy imports keep startup fast and avoid heavy optional deps at import time
    from entomokit import segment as _segment
    from entomokit import extract_frames as _extract_frames
    from entomokit import clean as _clean
    from entomokit import split_csv as _split_csv
    from entomokit import synthesize as _synthesize
    from entomokit.classify import register as _register_classify

    _segment.register(subparsers)
    _extract_frames.register(subparsers)
    _clean.register(subparsers)
    _split_csv.register(subparsers)
    _synthesize.register(subparsers)
    _register_classify(subparsers)

    return parser


def _activate_argcomplete(parser: argparse.ArgumentParser) -> None:
    try:
        import argcomplete
    except ImportError:
        return
    argcomplete.autocomplete(parser)


def main(argv: list[str] | None = None) -> None:
    _ensure_project_root_on_path()
    parser = _build_parser()
    _activate_argcomplete(parser)

    args = parser.parse_args(argv)
    if args.install_completion:
        raise SystemExit(_install_completion())

    if not getattr(args, "command", None):
        parser.error("the following arguments are required: <command>")

    args.func(args)


if __name__ == "__main__":
    main()
