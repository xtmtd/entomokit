"""Tests for top-level entomokit CLI behavior."""

from __future__ import annotations

import pytest


def test_help_omits_legacy_install_completion_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Top-level help should only expose the completion command, not a legacy flag."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--install-completion" not in out
    assert "completion" in out


def test_help_includes_quick_examples(capsys: pytest.CaptureFixture[str]) -> None:
    """Top-level help should show quick command examples near the top."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Quick examples:" in out
    assert "entomokit segment --input-dir ./images --out-dir ./out" in out


def test_help_uses_boxed_section_titles(capsys: pytest.CaptureFixture[str]) -> None:
    """Top-level help should render boxed commands/options headings."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "[ Commands ]:" in out
    assert "[ Options ]:" in out


def test_segment_help_has_quick_examples_and_boxed_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Segment help should include quick examples and boxed options section."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["segment", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Quick examples:" in out
    assert "entomokit segment --input-dir ./images --out-dir ./segmented" in out
    assert "[ Options ]:" in out


def test_classify_help_has_quick_examples_and_boxed_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Classify group help should include examples and boxed command list."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["classify", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Quick examples:" in out
    assert (
        "entomokit classify train --train-csv train.csv --images-dir ./images --out-dir ./model"
        in out
    )
    assert "[ Commands ]:" in out
    assert "[ Options ]:" in out


def test_classify_train_help_has_quick_examples_and_boxed_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Classify train help should include examples and boxed options section."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["classify", "train", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Quick examples:" in out
    assert (
        "entomokit classify train --train-csv train.csv --images-dir ./images --out-dir ./model"
        in out
    )
    assert "[ Options ]:" in out


def test_completion_group_help_lists_shells(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["completion", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "bash" in out
    assert "zsh" in out
    assert "fish" in out


def test_completion_zsh_outputs_static_script(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from entomokit.main import main

    main(["completion", "zsh"])
    out = capsys.readouterr().out
    assert "entomokit" in out
    assert "compdef" in out or "complete" in out
    assert "classify" in out
    assert "train" in out
    assert "predict" in out
    assert "evaluate" in out
    assert "--version" in out
    assert "auto" in out
    assert "cuda" in out


def test_completion_fish_supports_nested_classify_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from entomokit.main import main

    main(["completion", "fish"])
    out = capsys.readouterr().out
    assert "complete -c entomokit -n '__fish_seen_subcommand_from classify; and not " in out
    assert "__fish_seen_subcommand_from classify; and not __fish_seen_subcommand_from classify" not in out
    assert " -a 'cam embed evaluate export-onnx predict train'" in out


def test_completion_fish_classify_train_options_require_command_sequence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from entomokit.main import main

    main(["completion", "fish"])
    out = capsys.readouterr().out
    assert "__fish_seen_entomokit_command_sequence classify train" in out
    assert "__fish_seen_subcommand_from classify train" not in out


def test_completion_fish_renders_short_flags_as_short_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from entomokit.main import main

    main(["completion", "fish"])
    out = capsys.readouterr().out
    assert "complete -c entomokit -n '__fish_use_subcommand' -s 'v'" in out
    assert "complete -c entomokit -n '__fish_use_subcommand' -l 'v'" not in out


def test_completion_bash_and_zsh_quote_nested_case_labels() -> None:
    from entomokit.completion import render_completion_script

    for shell in ("bash", "zsh"):
        out = render_completion_script(shell)
        assert "classify\\ train)" in out
        assert "classify train)" not in out


def test_completion_zsh_uses_dollar_CURRENT_in_array_subscripts() -> None:
    from entomokit.completion import render_completion_script

    zsh = render_completion_script("zsh")
    assert "${words[CURRENT]}" not in zsh
    assert "cur=$words[CURRENT]" in zsh
    assert "prev=$words[$((CURRENT-1))]" in zsh


def test_completion_zsh_does_not_shadow_completion_words() -> None:
    from entomokit.completion import render_completion_script

    zsh = render_completion_script("zsh")
    bash = render_completion_script("bash")
    assert "local -a words" not in zsh
    assert "local -a command_words" in zsh
    assert "for word in \"${command_words[@]}\"; do" in zsh
    assert "local -a words" in bash


def test_completion_choice_cases_are_scoped_by_command_path() -> None:
    from entomokit.completion import render_completion_script

    out = render_completion_script("bash")
    assert "extract-frames:--out-image-format) _entomokit_reply 'jpg png tif'; return ;;" in out
    assert "segment:--out-image-format) _entomokit_reply 'jpg png'; return ;;" in out
    assert "\n    --out-image-format) _entomokit_reply" not in out


def test_install_zsh_completion_prints_activation_hint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from entomokit.completion import install_for_shell

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "env"))

    assert install_for_shell("zsh") == 0
    out = capsys.readouterr().out
    assert "Installed zsh completion at:" in out
    assert "Installed conda activation hook at:" in out


def test_top_level_command_order_matches_dataset_workflow() -> None:
    """Top-level commands should follow dataset preparation workflow order."""
    import argparse

    from entomokit.main import _build_parser

    parser = _build_parser()
    commands: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            commands = list(action.choices.keys())
            break

    assert commands[:8] == [
        "extract-frames",
        "segment",
        "measure",
        "synthesize",
        "clean",
        "augment",
        "split-csv",
        "classify",
    ]


def test_version_flag_prints_package_version_long(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--version should print version and exit 0."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip() == "entomokit 0.4.1"


def test_version_flag_prints_package_version_short(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """-v should print version and exit 0 at top-level."""
    from entomokit.main import main

    with pytest.raises(SystemExit) as exc:
        main(["-v"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip() == "entomokit 0.4.1"


def test_get_version_returns_current():
    from entomokit.main import _get_version
    assert _get_version() == "0.4.1"


def test_bash_falls_back_to_files():
    from entomokit.completion import render_completion_script

    script = render_completion_script("bash")
    assert "complete -o default -F _entomokit_completion entomokit" in script


def test_zsh_calls_files_fallback():
    from entomokit.completion import render_completion_script

    script = render_completion_script("zsh")
    assert "_files" in script


def test_fish_no_global_no_files_flag():
    from entomokit.completion import render_completion_script

    script = render_completion_script("fish")
    first_line = script.splitlines()[0]
    assert "-f" not in first_line
