import json
from unittest.mock import MagicMock, patch

from entomokit.update import _status, _local_commit, _local_commit_date, fetch_latest_commit


def _make_response(sha: str, date: str, message: str) -> MagicMock:
    payload = json.dumps({
        "sha": sha,
        "commit": {
            "author": {"date": date},
            "message": message,
        }
    }).encode()
    mock = MagicMock()
    mock.read.return_value = payload
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_fetch_latest_commit_parses():
    with patch("urllib.request.urlopen", return_value=_make_response("abc1234def", "2026-07-10T00:00:00Z", "feat: X")):
        sha, date, msg = fetch_latest_commit()
    assert sha == "abc1234"  # first 7 chars
    assert "2026-07-10" in date
    assert "feat: X" in msg


def test_status_same_sha():
    assert _status("abc1234", "2026-07-10", "abc1234", "2026-07-11") == "same"


def test_status_remote_newer():
    assert _status("abc1234", "2026-07-09", "def5678", "2026-07-10") == "newer"


def test_status_unknown_local():
    assert _status("unknown", "unknown", "abc1234", "2026-07-10") == "unknown"


def test_local_commit_resolves_from_git_when_unknown():
    """When __commit__ is 'unknown', try git rev-parse at runtime."""
    with patch("entomokit.update.__commit__", "unknown"), \
         patch("subprocess.check_output", return_value="abcd123\n") as mock_check:
        result = _local_commit()
    assert result == "abcd123"
    mock_check.assert_called_once()


def test_local_commit_returns_unknown_when_git_fails():
    with patch("entomokit.update.__commit__", "unknown"), \
         patch("subprocess.check_output", side_effect=OSError):
        result = _local_commit()
    assert result == "unknown"


def test_local_commit_returns_static_value_when_known():
    with patch("entomokit.update.__commit__", "abc1234"):
        result = _local_commit()
    assert result == "abc1234"


def test_local_commit_date_resolves_from_git_when_unknown():
    with patch("entomokit.update.__commit_date__", "unknown"), \
         patch("subprocess.check_output", return_value="2026-07-10 abc\n") as mock_check:
        result = _local_commit_date()
    assert result == "2026-07-10"
    mock_check.assert_called_once()


def test_local_commit_date_returns_unknown_when_git_fails():
    with patch("entomokit.update.__commit_date__", "unknown"), \
         patch("subprocess.check_output", side_effect=OSError):
        result = _local_commit_date()
    assert result == "unknown"


def test_local_commit_date_returns_static_value_when_known():
    with patch("entomokit.update.__commit_date__", "2026-07-10"):
        result = _local_commit_date()
    assert result == "2026-07-10"
