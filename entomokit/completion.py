"""Static shell completion helpers for entomokit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser


def _subparser_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _node_key(path: tuple[str, ...]) -> str:
    return " ".join(path)


def _action_completion_choices(action: argparse.Action) -> list[str]:
    if not action.choices:
        return []
    return [str(choice) for choice in action.choices]


def _parser_completion_tree() -> dict[str, dict[str, object]]:
    from entomokit.main import _build_parser

    parser = _build_parser()
    nodes: dict[str, dict[str, object]] = {}

    def visit(current: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        key = _node_key(path)
        node = nodes.setdefault(
            key,
            {"commands": [], "options": [], "choice_map": {}},
        )
        sub = _subparser_action(current)
        if sub is not None:
            node["commands"] = list(sub.choices.keys())
        for action in current._actions:
            if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
                continue
            if action.option_strings:
                node["options"].extend(action.option_strings)
                choices = _action_completion_choices(action)
                if choices:
                    for option in action.option_strings:
                        node["choice_map"][option] = choices
        if sub is None:
            return
        for name, child in sub.choices.items():
            visit(child, (*path, name))

    visit(parser, ())
    return nodes


def _shell_words(values: list[str]) -> str:
    return " ".join(sorted(dict.fromkeys(values)))


def _case_label(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for char in " ()|*?[]":
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def _render_choice_cases(nodes: dict[str, dict[str, object]], indent: str) -> str:
    lines: list[str] = []
    for path, node in nodes.items():
        for option, choices in node["choice_map"].items():
            choice_key = f"{path if path else '__root__'}:{option}"
            lines.append(
                f"{indent}{_case_label(choice_key)}) _entomokit_reply '{_shell_words(choices)}'; return ;;"
            )
    return "\n".join(lines)


def _render_path_cases(nodes: dict[str, dict[str, object]], indent: str) -> str:
    lines: list[str] = []
    for path, node in nodes.items():
        words = list(node["commands"]) + list(node["options"])
        if not words:
            continue
        case_key = path if path else "__root__"
        lines.append(
            f"{indent}{_case_label(case_key)}) _entomokit_reply '{_shell_words(words)}' ;;"
        )
    return "\n".join(lines)


def _render_path_resolver(nodes: dict[str, dict[str, object]], word_var: str) -> str:
    lines = [
        "  path='__root__'",
        f"  for word in \"${{{word_var}[@]}}\"; do",
        "    [[ \"$word\" == -* ]] && continue",
        "    case \"$path\" in",
    ]
    for path, node in nodes.items():
        if not node["commands"]:
            continue
        case_key = path if path else "__root__"
        path_parts = tuple(path.split()) if path else ()
        lines.append(f"      {_case_label(case_key)})")
        for command in node["commands"]:
            child = _node_key((*path_parts, command))
            child_key = child if child else "__root__"
            lines.append(f"        [[ \"$word\" == \"{command}\" ]] && path='{child_key}'")
        lines.append("        ;;")
    lines.extend(
        [
            "    esac",
            "  done",
        ]
    )
    return "\n".join(lines)


def _render_bash_script(nodes: dict[str, dict[str, object]]) -> str:
    choice_cases = _render_choice_cases(nodes, "    ")
    path_cases = _render_path_cases(nodes, "    ")
    path_resolver = _render_path_resolver(nodes, "words")
    return (
        "_entomokit_completion() {\n"
        "  local cur prev word path choice_key\n"
        "  local -a words\n"
        "  COMPREPLY=()\n"
        "  cur=\"${COMP_WORDS[COMP_CWORD]}\"\n"
        "  prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n"
        "  words=(\"${COMP_WORDS[@]:1:COMP_CWORD-1}\")\n"
        "  _entomokit_reply() { COMPREPLY=( $( compgen -W \"$1\" -- \"$cur\" ) ); }\n"
        f"{path_resolver}\n"
        "  choice_key=\"${path}:$prev\"\n"
        "  case \"$choice_key\" in\n"
        f"{choice_cases}\n"
        "  esac\n"
        "  case \"$path\" in\n"
        f"{path_cases}\n"
        "  esac\n"
        "}\n"
        "complete -o default -F _entomokit_completion entomokit\n"
    )


def _render_zsh_script(nodes: dict[str, dict[str, object]]) -> str:
    choice_cases = _render_choice_cases(nodes, "    ")
    path_cases = _render_path_cases(nodes, "    ")
    path_resolver = _render_path_resolver(nodes, "command_words")
    return (
        "#compdef entomokit\n"
        "_entomokit() {\n"
        "  local cur prev word path choice_key\n"
        "  local -a command_words\n"
        "  cur=$words[CURRENT]\n"
        "  prev=$words[$((CURRENT-1))]\n"
        "  command_words=(\"${words[@]:1:$((CURRENT-2))}\")\n"
        "  _entomokit_reply() { compadd -- ${(z)1}; }\n"
        f"{path_resolver}\n"
        "  choice_key=\"${path}:$prev\"\n"
        "  case \"$choice_key\" in\n"
        f"{choice_cases}\n"
        "  esac\n"
        "  case \"$path\" in\n"
        f"{path_cases}\n"
        "  esac\n"
        "  _files\n"
        "}\n"
        "compdef _entomokit entomokit\n"
    )


def _fish_condition(path: str, child_commands: list[str] | None = None) -> str:
    if not path:
        return "__fish_use_subcommand"
    tokens = path.split()
    if len(tokens) == 1:
        child_words = _shell_words(child_commands or [])
        if child_words:
            return f"__fish_seen_subcommand_from {tokens[0]}; and not __fish_seen_subcommand_from {child_words}"
        return f"__fish_seen_subcommand_from {tokens[0]}"
    return f"__fish_seen_entomokit_command_sequence {' '.join(tokens)}"


def _fish_option_flag(option: str) -> str:
    if option.startswith("--"):
        return f"-l '{option[2:]}'"
    return f"-s '{option[1:]}'"


def _render_fish_script(nodes: dict[str, dict[str, object]]) -> str:
    lines = [
        "complete -c entomokit",
        "function __fish_seen_entomokit_command_sequence",
        "    set -l tokens (commandline -opc)",
        "    set -e tokens[1]",
        "    set -l i 1",
        "    for token in $tokens",
        "        set -l expected $argv[$i]",
        "        if test -n \"$expected\"; and test \"$token\" = \"$expected\"",
        "            set i (math $i + 1)",
        "        end",
        "    end",
        "    test $i -gt (count $argv)",
        "end",
    ]
    for path, node in nodes.items():
        commands = list(node["commands"])
        options = list(node["options"])
        condition = _fish_condition(path, commands)
        if commands:
            lines.append(
                f"complete -c entomokit -n '{condition}' -a '{_shell_words(commands)}'"
            )
        for option in options:
            lines.append(
                f"complete -c entomokit -n '{condition}' {_fish_option_flag(option)}"
            )
        for option, choices in node["choice_map"].items():
            if not option.startswith("--"):
                continue
            choice_condition = f"{condition}; and __fish_seen_argument {option}"
            lines.append(
                f"complete -c entomokit -n '{choice_condition}' -a '{_shell_words(choices)}'"
            )
    return "\n".join(lines) + "\n"


def render_completion_script(shell: str) -> str:
    nodes = _parser_completion_tree()
    if shell == "bash":
        return _render_bash_script(nodes)
    if shell == "zsh":
        return _render_zsh_script(nodes)
    if shell == "fish":
        return _render_fish_script(nodes)
    raise ValueError(f"Unsupported shell: {shell}")


def detect_shell() -> str:
    shell = Path(os.environ.get("SHELL", "")).name
    return shell if shell in {"bash", "zsh", "fish"} else "bash"


def install_for_shell(shell: str | None = None) -> int:
    shell = shell if shell is not None else detect_shell()
    script = render_completion_script(shell)
    if shell == "zsh":
        target = Path.home() / ".zfunc" / "_entomokit"
    elif shell == "fish":
        target = Path.home() / ".config" / "fish" / "completions" / "entomokit.fish"
    else:
        target = Path.home() / ".local" / "share" / "bash-completion" / "completions" / "entomokit"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    print(f"Installed {shell} completion at: {target}")
    if shell == "zsh":
        _install_conda_hook()
    return 0


def _install_conda_hook() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("No active conda environment — completion will work after next 'conda activate'")
        return
    hook_dir = Path(conda_prefix) / "etc" / "conda" / "activate.d"
    hook_dir.mkdir(parents=True, exist_ok=True)
    hook_file = hook_dir / "entomokit_completion.sh"
    hook_content = (
        "if [[ -f ~/.zfunc/_entomokit ]] && ! (( $+functions[_entomokit] )); then\n"
        "  fpath=(~/.zfunc $fpath)\n"
        "  autoload -Uz _entomokit\n"
        "  compdef _entomokit entomokit\n"
        "fi\n"
    )
    hook_file.write_text(hook_content, encoding="utf-8")
    print(f"Installed conda activation hook at: {hook_file}")


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "completion",
        help="Generate static shell completion scripts.",
        formatter_class=RichHelpFormatter,
    )
    style_parser(parser)
    shell_subparsers = parser.add_subparsers(
        dest="shell",
        metavar="<shell>",
        title="[ Commands ]",
    )
    shell_subparsers.required = True
    for shell_name in ("bash", "zsh", "fish"):
        shell_parser = shell_subparsers.add_parser(
            shell_name,
            help=f"Print a {shell_name} completion script.",
            formatter_class=RichHelpFormatter,
        )
        style_parser(shell_parser)
        shell_parser.add_argument(
            "--install",
            action="store_true",
            help="Write the script to the default user location.",
        )
        shell_parser.set_defaults(func=run, completion_shell=shell_name)


def run(args: argparse.Namespace) -> None:
    if args.install:
        raise SystemExit(install_for_shell(args.completion_shell))
    sys.stdout.write(render_completion_script(args.completion_shell))
