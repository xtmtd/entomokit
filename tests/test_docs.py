"""Documentation checks for the README and docs/commands.

The contract is
docs/superpowers/specs/2026-10-01-entomokit-documentation-conventions-design.md
(Section 7). Checks consume the existing runtime CLI schema and argparse tree
only; they never execute command callbacks or import optional analysis backends.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Spec constants
# --------------------------------------------------------------------------- #

SPEC = (
    "docs/superpowers/specs/"
    "2026-10-01-entomokit-documentation-conventions-design.md"
)
COMMANDS_DIR = Path("docs/commands")
README_EN = Path("README.md")
README_CN = Path("README.cn.md")
PLAYBOOK = Path("skills/entomokit-workflow/references/teaching-playbook.md")

# Spec Section 4 skeleton: (English, Chinese, required) in fixed order.
SECTION_ORDER = (
    ("Purpose", "目的", True),
    ("Usage", "用法", True),
    ("Parameters", "参数", True),
    ("Inputs", "输入", False),
    ("Outputs", "输出", False),
    ("Examples", "示例", False),
    ("Notes", "说明", False),
    ("Version Notes", "版本注记", False),
)
REQUIRED_SECTIONS = tuple(
    english for english, _chinese, required in SECTION_ORDER if required
)
_SECTION_INDEX = {
    title: index
    for index, pair in enumerate(SECTION_ORDER)
    for title in pair[:2]
}

# Spec Section 3 fixes these Commands H2 titles and the operational anchors.
README_COMMANDS_HEADINGS = ("Commands", "命令")
OPERATIONAL_ANCHORS = {
    "doctor": "doctor-command",
    "update": "update-command",
    "completion": "completion-command",
}
README_ONLY = ("doctor", "update", "completion")
DOCS_BASE_URL = "https://github.com/xtmtd/entomokit/blob/main/"

OPTION_RE = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9-]*$")
HELP_OPTIONS = {"-h", "--help"}
RELEASE_RE = re.compile(r"\b0\.\d+\.\d+\b")


# --------------------------------------------------------------------------- #
# Extraction helpers (stdlib only; no Markdown parser or framework)
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^```[^\n]*\n.*?^```[^\n]*$", re.M | re.S)
_H2_RE = re.compile(r"^## (?!#)(.*)$", re.M)
_H3_RE = re.compile(r"^### (.+)$", re.M)
_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)")
_ANCHOR_RE = re.compile(r'<a\s+(?:id|name)="([^"]+)"\s*>', re.I)


def _strip_fenced_blocks(text: str) -> str:
    """Drop fenced code blocks: example Markdown inside them is not live."""
    return _FENCE_RE.sub("", text)


def option_tokens(text: str) -> set[str]:
    """Option names appearing as code spans, excluding the built-in help."""
    tokens: set[str] = set()
    for span in _CODE_SPAN_RE.findall(_strip_fenced_blocks(text)):
        for token in re.split(r"[,\s|]+", span.strip()):
            if OPTION_RE.match(token) and token not in HELP_OPTIONS:
                tokens.add(token)
    return tokens


def h2_titles(text: str) -> list[str]:
    return [match.group(1).strip() for match in _H2_RE.finditer(_strip_fenced_blocks(text))]


def h2_body(text: str, titles: tuple[str, ...]) -> str | None:
    """Body of the first H2 whose title is in ``titles``, bound by the next H2."""
    clean = _strip_fenced_blocks(text)
    matches = list(_H2_RE.finditer(clean))
    for index, match in enumerate(matches):
        if match.group(1).strip() in titles:
            end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
            return clean[match.end():end]
    return None


def anchored_body(text: str, anchor_id: str) -> str | None:
    """Section introduced by an explicit anchor, bound by the following H2.

    The spec places each anchor immediately before its own heading, so the
    section starts at the first H2 after the anchor.
    """
    clean = _strip_fenced_blocks(text)
    pattern = r'<a\s+(?:id|name)="' + re.escape(anchor_id) + r'"\s*>'
    match = re.search(pattern, clean, re.I)
    if match is None:
        return None
    rest = clean[match.end():]
    headings = list(_H2_RE.finditer(rest))
    if not headings:
        return rest
    start = headings[0].start()
    end = headings[1].start() if len(headings) > 1 else len(rest)
    return rest[start:end]


def h3_titles(body: str) -> list[str]:
    return [match.group(1).strip() for match in _H3_RE.finditer(_strip_fenced_blocks(body))]


def h3_option_tokens(body: str) -> set[str]:
    tokens: set[str] = set()
    for title in h3_titles(body):
        tokens |= option_tokens(title)
    return tokens


def tables(text: str) -> list[list[list[str]]]:
    """Markdown tables in the text, grouped, with separator rows removed."""
    tables_found: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in _strip_fenced_blocks(text).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if current:
                tables_found.append(current)
                current = []
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        current.append(cells)
    if current:
        tables_found.append(current)
    return tables_found


def table_rows(text: str) -> list[list[str]]:
    return [row for table in tables(text) for row in table]


def table_first_column_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for row in table_rows(text):
        if row:
            tokens |= option_tokens(row[0])
    return tokens


def normalize_label(cell: str) -> str:
    return " ".join(cell.replace("`", "").split())


def invocation_stem(cell: str) -> str:
    """Displayed invocation label (``classify train``) as the document stem."""
    return normalize_label(cell).replace(" ", "-")


def row_label_problems(labels: list[str], expected: set[str]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            problems.append(f"duplicate row label: {label}")
        seen.add(label)
    for label in sorted(set(labels) - expected):
        problems.append(f"unexpected row label: {label}")
    for label in sorted(expected - set(labels)):
        problems.append(f"missing row label: {label}")
    return problems


def explicit_anchors(text: str) -> set[str]:
    return set(_ANCHOR_RE.findall(_strip_fenced_blocks(text)))


def markdown_links(text: str) -> list[str]:
    return _LINK_RE.findall(_strip_fenced_blocks(text))


# --------------------------------------------------------------------------- #
# Extraction helper tests (pass/fail independently of the migration)
# --------------------------------------------------------------------------- #


def test_extraction_option_tokens_accept_aliases_and_separators() -> None:
    assert option_tokens("| `--out-dir`, `-o` | Output |") == {"--out-dir", "-o"}
    assert option_tokens("### `--yes`, `-y`") == {"--yes", "-y"}


def test_extraction_option_tokens_exclude_help_globally() -> None:
    assert option_tokens("`-h`, `--help`, `--install`") == {"--install"}


def test_extraction_option_tokens_ignore_prose_and_fenced_examples() -> None:
    text = (
        "Use --out-dir in prose.\n\n"
        "```bash\n"
        "entomokit clean --input-dir in --out-dir out\n"
        "```\n"
    )
    assert option_tokens(text) == set()


def test_extraction_parameters_h3_scope_is_bounded_by_the_h2() -> None:
    document = (
        "# entomokit demo\n\n"
        "## Usage\n\n`--usage-flag`\n\n"
        "## Parameters\n\n"
        "### `--a`, `-a`\n\nRequired.\n\n"
        "### `--b`\n\nNo default.\n\n"
        "## Notes\n\n### `--leaked`\n\nText.\n"
    )
    body = h2_body(document, ("Parameters", "参数"))
    assert body is not None
    assert h3_option_tokens(body) == {"--a", "-a", "--b"}


def test_extraction_operational_section_is_bounded_by_the_next_h2() -> None:
    document = (
        '<a id="update-command"></a>\n'
        "## Update Command\n\n"
        "| Parameter | Description |\n|---|---|\n"
        "| `--check` | only check |\n"
        "| `--yes`, `-y` | skip prompt |\n\n"
        "## Common Behaviours\n\n"
        "| `--unrelated` | noise |\n"
    )
    body = anchored_body(document, "update-command")
    assert body is not None
    assert table_first_column_tokens(body) == {"--check", "--yes", "-y"}


def test_extraction_table_option_column_uses_code_spans() -> None:
    table = (
        "| Parameter | Description |\n|---|---|\n"
        "| `--check` | mentions --verbose in prose |\n"
    )
    assert table_first_column_tokens(table) == {"--check"}


def test_extraction_table_rows_ignore_separator_rows() -> None:
    table = "intro\n\n| A | B |\n|---|---|\n| x | y |\n\nprose\n"
    assert table_rows(table) == [["A", "B"], ["x", "y"]]


def test_extraction_invocation_labels_normalize_to_stems() -> None:
    assert normalize_label("`classify train`") == "classify train"
    assert normalize_label("extract-frames") == "extract-frames"


def test_extraction_row_label_problems_report_duplicates_and_extras() -> None:
    problems = row_label_problems(["clean", "clean", "ghost"], {"clean", "measure"})
    assert any("duplicate" in problem for problem in problems)
    assert any("ghost" in problem for problem in problems)
    assert any("measure" in problem for problem in problems)


def test_extraction_fragment_anchors_accept_id_and_name() -> None:
    text = '<a id="directory-policy"></a>\n\n## X\n\n<a name="other"></a>\n'
    assert explicit_anchors(text) == {"directory-policy", "other"}


def test_extraction_missing_fragment_is_not_reported_as_present() -> None:
    assert "absent" not in explicit_anchors('<a id="present"></a>\n')


def test_extraction_ignores_fenced_links() -> None:
    text = "```markdown\n[nope](missing.md)\n```\n\n[ok](README.md)\n"
    assert markdown_links(text) == ["README.md"]


# --------------------------------------------------------------------------- #
# Runtime schema and argparse helpers
# --------------------------------------------------------------------------- #


def _schema() -> dict[str, dict]:
    from entomokit.cli_schema import build_command_schemas

    return build_command_schemas()


def _stem(command_key: str) -> str:
    return command_key.replace(" ", "-")


def _functional_keys() -> list[str]:
    return sorted(
        key
        for key in _schema()
        if key not in README_ONLY and key.split(" ")[0] not in README_ONLY
    )


def _functional_stems() -> set[str]:
    return {_stem(key) for key in _functional_keys()}


def _options_of(command_key: str) -> set[str]:
    return {
        option
        for parameter in _schema()[command_key]["parameters"]
        for option in parameter["options"]
    } - HELP_OPTIONS


def _subparser(parser, path: tuple[str, ...]):
    import argparse

    current = parser
    for name in path:
        action = next(
            (
                candidate
                for candidate in current._actions
                if isinstance(candidate, argparse._SubParsersAction)
            ),
            None,
        )
        if action is None or name not in action.choices:
            return None
        current = action.choices[name]
    return current


def _registered_parser_paths(parser, prefix: tuple[str, ...] = ()):
    """Every registered parser path in the tree, parent parsers included."""
    import argparse

    paths = [prefix]
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                paths.extend(_registered_parser_paths(child, (*prefix, name)))
    return paths


def _heading_marks(text: str) -> list[tuple[int, int, str]]:
    """H2/H3 heading positions in the raw text, ignoring fenced code blocks."""
    marks: list[tuple[int, int, str]] = []
    fenced = False
    for index, line in enumerate(text.splitlines()):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = re.match(r"^(#{2,3}) (.+)$", line)
        if match:
            marks.append((index, len(match.group(1)), match.group(2).strip()))
    return marks


def _heading_bodies(text: str) -> list[tuple[str, str]]:
    """Every H2/H3 heading with its raw body, bounded by the next heading of the
    same or higher level, so a section made only of subsections or of a fenced
    example is not "empty"."""
    lines = text.splitlines()
    marks = _heading_marks(text)
    bodies: list[tuple[str, str]] = []
    for position, (line_number, level, title) in enumerate(marks):
        end = len(lines)
        for next_line_number, next_level, _next_title in marks[position + 1:]:
            if next_level <= level:
                end = next_line_number
                break
        bodies.append((f"{'#' * level} {title}", "\n".join(lines[line_number + 1:end])))
    return bodies


def _normalized_section_order(titles: list[str]) -> list[int]:
    return [_SECTION_INDEX[title] for title in titles if title in _SECTION_INDEX]


# --------------------------------------------------------------------------- #
# Repository checks
# --------------------------------------------------------------------------- #


def test_functional_document_closure() -> None:
    assert COMMANDS_DIR.is_dir(), f"{COMMANDS_DIR} is missing"
    stems = _functional_stems()
    assert stems, "runtime schema exposes no functional commands"
    english = {
        path.stem for path in COMMANDS_DIR.glob("*.md") if not path.name.endswith(".cn.md")
    }
    chinese = {
        path.name[: -len(".cn.md")] for path in COMMANDS_DIR.glob("*.cn.md")
    }
    assert not (
        (stems - english)
        or (english - stems)
        or (stems - chinese)
        or (chinese - stems)
    ), (
        f"missing English documents: {sorted(stems - english)}; "
        f"stale English documents: {sorted(english - stems)}; "
        f"missing Chinese documents: {sorted(stems - chinese)}; "
        f"stale Chinese documents: {sorted(chinese - stems)}"
    )


@pytest.mark.parametrize("command_key", _functional_keys(), ids=_stem)
def test_command_document_structure(command_key: str) -> None:
    stem = _stem(command_key)
    titles_by_language: dict[str, list[str]] = {}
    orders_by_language: dict[str, list[int]] = {}
    for language, suffix in (("English", ".md"), ("Chinese", ".cn.md")):
        path = COMMANDS_DIR / f"{stem}{suffix}"
        assert path.exists(), f"{language} document missing: {path}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines and lines[0].strip() == f"# entomokit {command_key}", (
            f"{path}: H1 must be the literal invocation '# entomokit {command_key}'"
        )
        expected_switch = f"[English]({stem}.md) | [中文]({stem}.cn.md)"
        assert len(lines) > 1 and lines[1].strip() == expected_switch, (
            f"{path}: language switch must sit immediately below the H1"
        )
        titles = h2_titles(text)
        unknown = [title for title in titles if title not in _SECTION_INDEX]
        assert not unknown, f"{path}: unknown H2 sections {unknown}"
        required = {
            english if language == "English" else chinese
            for english, chinese, needed in SECTION_ORDER
            if needed
        }
        missing = sorted(required - set(titles))
        assert not missing, f"{path}: missing required sections {missing}"
        orders_by_language[language] = _normalized_section_order(titles)
        titles_by_language[language] = titles
        empty = [
            heading
            for heading, body in _heading_bodies(text)
            if not body.strip()
        ]
        assert not empty, f"{path}: empty headings {empty}"
        numbered = [
            heading
            for heading, _body in _heading_bodies(text)
            if RELEASE_RE.search(heading)
        ]
        assert not numbered, f"{path}: headings carry release numbers {numbered}"
    english_order = orders_by_language["English"]
    assert english_order == sorted(english_order), (
        "English sections are not in spec order: "
        f"{titles_by_language['English']}"
    )
    assert orders_by_language["English"] == orders_by_language["Chinese"], (
        f"section sequences differ: {titles_by_language['English']} vs "
        f"{titles_by_language['Chinese']}"
    )


@pytest.mark.parametrize("command_key", _functional_keys(), ids=_stem)
def test_command_option_coverage(command_key: str) -> None:
    stem = _stem(command_key)
    expected = _options_of(command_key)
    for language, suffix in (("English", ".md"), ("Chinese", ".cn.md")):
        path = COMMANDS_DIR / f"{stem}{suffix}"
        assert path.exists(), f"{language} document missing: {path}"
        text = path.read_text(encoding="utf-8")
        body = h2_body(text, ("Parameters", "参数"))
        assert body is not None, f"{path}: no Parameters section"
        documented = h3_option_tokens(body)
        missing = sorted(expected - documented)
        extra = sorted(documented - expected)
        assert not missing and not extra, (
            f"{path}: undocumented options {missing}; unexpected options {extra}"
        )


def test_readme_command_index() -> None:
    expected_labels = {_stem(key) for key in _functional_keys()} | set(README_ONLY)
    for readme, suffix in ((README_EN, ".md"), (README_CN, ".cn.md")):
        path = Path(readme)
        assert path.exists(), f"missing README: {path}"
        text = path.read_text(encoding="utf-8")
        section = h2_body(text, README_COMMANDS_HEADINGS)
        assert section is not None, f"{path}: no Commands/命令 section"
        rows = [row for table in tables(section) for row in table[1:]]
        assert rows, f"{path}: Commands section has no data rows"
        labels = [invocation_stem(row[0]) for row in rows]
        problems = row_label_problems(labels, expected_labels)
        assert not problems, f"{path}: {problems}"
        for row in rows:
            label = invocation_stem(row[0])
            links = markdown_links(" | ".join(row[1:]))
            assert links, f"{path}: row {label!r} has no link"
            target = links[0]
            if label in README_ONLY:
                fragment = f"#{OPERATIONAL_ANCHORS[label]}"
                assert target in (fragment, f"{path.name}{fragment}"), (
                    f"{path}: row {label!r} points to {target!r}, expected {fragment!r}"
                )
            else:
                expected_target = f"docs/commands/{label}{suffix}"
                assert target == expected_target, (
                    f"{path}: row {label!r} points to {target!r}, expected {expected_target!r}"
                )


def test_readme_operational_options() -> None:
    shells = sorted(
        key.split(" ")[1] for key in _schema() if key.startswith("completion ")
    )
    for readme in (README_EN, README_CN):
        path = Path(readme)
        text = path.read_text(encoding="utf-8")
        for command, anchor in OPERATIONAL_ANCHORS.items():
            body = anchored_body(text, anchor)
            assert body is not None, f"{path}: anchor {anchor!r} is missing"
            rows = tables(body)
            data_rows = rows[0][1:] if rows else []
            if command == "completion":
                labels = sorted(normalize_label(row[0]) for row in data_rows)
                assert labels == shells, (
                    f"{path}: completion shells {labels}, expected {shells}"
                )
                for row in data_rows:
                    shell = normalize_label(row[0])
                    expected = _options_of(f"completion {shell}")
                    documented = option_tokens(row[1]) if len(row) > 1 else set()
                    assert documented == expected, (
                        f"{path}: completion {shell} documents {sorted(documented)}, "
                        f"expected {sorted(expected)}"
                    )
            else:
                expected = _options_of(command)
                documented = (
                    set().union(*(option_tokens(row[0]) for row in data_rows))
                    if data_rows
                    else set()
                )
                if expected:
                    assert documented == expected, (
                        f"{path}: {command} documents {sorted(documented)}, "
                        f"expected {sorted(expected)}"
                    )
                else:
                    assert not rows, (
                        f"{path}: {command} has no user-settable options; "
                        "omit its option table"
                    )


def _is_readme(target: Path) -> bool:
    return target.name in {README_EN.name, README_CN.name}


def _in_commands_dir(target: Path) -> bool:
    return target.parent.name == COMMANDS_DIR.name and "docs" in target.parts


def test_document_local_links() -> None:
    sources = [README_EN, README_CN] + sorted(COMMANDS_DIR.glob("*.md"))
    problems: list[str] = []
    for source in sources:
        assert source.exists(), f"missing document: {source}"
        text = source.read_text(encoding="utf-8")
        lines = text.splitlines()
        switch_line = lines[1].strip() if len(lines) > 1 else ""
        for link in markdown_links(text):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", link):
                continue
            path_part, _, fragment = link.partition("#")
            target = source
            if path_part:
                if link in switch_line:
                    continue
                target = (source.parent / path_part).resolve()
                if not target.exists():
                    problems.append(f"{source}: broken link {link}")
                    continue
                if target.is_dir():
                    continue
                if _in_commands_dir(target) or _is_readme(target):
                    source_is_chinese = source.name.endswith(".cn.md")
                    target_is_chinese = target.name.endswith(".cn.md")
                    if source_is_chinese != target_is_chinese:
                        problems.append(
                            f"{source}: cross-language link {link} is not a language switch"
                        )
                        continue
            if fragment:
                target_text = (
                    target.read_text(encoding="utf-8")
                    if path_part and target.is_file()
                    else text
                )
                if fragment not in explicit_anchors(target_text):
                    problems.append(f"{source}: missing fragment {link}")
    assert not problems, "\n".join(problems)


def test_with_doc_link_appends_description() -> None:
    from entomokit.help_style import DOCS_BASE_URL, with_doc_link

    text = with_doc_link(
        "Summary.\n\nQuick examples:\n  entomokit segment",
        "docs/commands/segment.md",
    )
    assert text.startswith("Summary.")
    assert "entomokit segment" in text
    assert text.endswith(DOCS_BASE_URL + "docs/commands/segment.md")
    assert "\n\n" in text


def test_with_doc_link_handles_empty_description() -> None:
    from entomokit.help_style import DOCS_BASE_URL, with_doc_link

    url = DOCS_BASE_URL + "README.md#doctor-command"
    assert with_doc_link(None, "README.md#doctor-command") == url
    assert with_doc_link("", "README.md#doctor-command") == url
    assert with_doc_link("   ", "README.md#doctor-command") == url


def test_parser_help_links() -> None:
    from entomokit.main import _build_parser

    parser = _build_parser()
    expected: dict[tuple[str, ...], str] = {(): DOCS_BASE_URL + "README.md"}
    for command_key in _functional_keys():
        expected[tuple(command_key.split())] = (
            DOCS_BASE_URL + f"docs/commands/{_stem(command_key)}.md"
        )
    expected[("classify",)] = DOCS_BASE_URL + "README.md#classify-commands"
    expected[("doctor",)] = DOCS_BASE_URL + "README.md#doctor-command"
    expected[("update",)] = DOCS_BASE_URL + "README.md#update-command"
    expected[("completion",)] = DOCS_BASE_URL + "README.md#completion-command"
    for command_key in _schema():
        if command_key.startswith("completion "):
            expected[tuple(command_key.split())] = (
                DOCS_BASE_URL + "README.md#completion-command"
            )
    registered = {tuple(path) for path in _registered_parser_paths(parser)}
    missing_from_expectations = sorted(
        " ".join(path) or "root" for path in registered - set(expected)
    )
    assert not missing_from_expectations, (
        "registered parsers without an expected help URL: "
        f"{missing_from_expectations}"
    )
    problems: list[str] = []
    for path, url in expected.items():
        subparser = _subparser(parser, path)
        name = " ".join(path) or "root"
        if subparser is None:
            problems.append(f"{name}: parser path not found")
            continue
        if url not in subparser.format_help():
            problems.append(f"{name}: help omits {url}")
    assert not problems, "\n".join(problems)
