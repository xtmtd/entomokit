import json
from unittest.mock import MagicMock, patch

from entomokit.update import fetch_remote_version, _parse_version


def _make_tags_response(tags: list) -> MagicMock:
    mock = MagicMock()
    mock.read.return_value = json.dumps(tags).encode()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _make_raw_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.read.return_value = content.encode()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_fetch_remote_version_from_version_txt():
    """Reads the main-branch version file before consulting release tags."""
    with patch("urllib.request.urlopen", return_value=_make_raw_response("0.6.0\n")):
        assert fetch_remote_version() == "0.6.0"


def test_fetch_remote_version_falls_back_to_tags():
    """Uses tags only when the version file cannot be fetched."""
    tags_payload = [{"name": "v0.5.0"}, {"name": "v0.4.1"}]
    with patch(
        "urllib.request.urlopen",
        side_effect=[OSError("404"), _make_tags_response(tags_payload)],
    ):
        assert fetch_remote_version() == "0.5.0"


def test_fetch_remote_version_strips_v_prefix():
    """Tag names with 'v' prefix are stripped."""
    tags_payload = [{"name": "v1.2.3"}]
    with patch(
        "urllib.request.urlopen",
        side_effect=[OSError("404"), _make_tags_response(tags_payload)],
    ):
        ver = fetch_remote_version()
    assert ver == "1.2.3"


def test_fetch_remote_version_uses_highest_semver_tag():
    tags_payload = [{"name": "nightly"}, {"name": "v0.4.1"}, {"name": "v0.5.0"}]
    with patch(
        "urllib.request.urlopen",
        side_effect=[OSError("404"), _make_tags_response(tags_payload)],
    ):
        assert fetch_remote_version() == "0.5.0"


def test_parse_version_normal():
    assert _parse_version("0.5.0") < _parse_version("1.2.3")


def test_parse_version_invalid_returns_zero():
    assert _parse_version("unknown") < _parse_version("0.0.0")
    assert _parse_version("") < _parse_version("0.0.0")


def test_parse_version_ignores_prerelease_suffix():
    assert _parse_version("0.5.0-rc.1") < _parse_version("0.5.0")
    assert _parse_version("0.5.0+build.1") == _parse_version("0.5.0")


def test_version_comparison_newer():
    assert _parse_version("0.5.0") > _parse_version("0.4.1")


def test_version_comparison_same():
    assert _parse_version("0.5.0") == _parse_version("0.5.0")
