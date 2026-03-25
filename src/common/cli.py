"""Common CLI utilities for all scripts."""

import argparse
import atexit
import logging
import os
import re
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


_shutdown_requested = False
_capture_file_handle = None
_capture_stdout = None
_capture_stderr = None
_capture_log_path: Optional[Path] = None
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _TeeStream:
    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        written = self._stream.write(text)
        if text and "\r" not in text:
            clean = _ANSI_CSI_RE.sub("", text)
            if clean:
                self._log_file.write(clean)
            self._log_file.flush()
        return written

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):
        return getattr(self._stream, "errors", None)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _disable_output_capture() -> None:
    global _capture_file_handle, _capture_stdout, _capture_stderr, _capture_log_path
    if _capture_stdout is not None:
        sys.stdout = _capture_stdout
        _capture_stdout = None
    if _capture_stderr is not None:
        sys.stderr = _capture_stderr
        _capture_stderr = None
    if _capture_file_handle is not None:
        _capture_file_handle.flush()
        _capture_file_handle.close()
        _capture_file_handle = None
    _capture_log_path = None


def _enable_output_capture(log_path: Path) -> None:
    global _capture_file_handle, _capture_stdout, _capture_stderr, _capture_log_path
    if _capture_log_path == log_path and _capture_file_handle is not None:
        return

    _disable_output_capture()
    _capture_log_path = log_path
    _capture_file_handle = open(log_path, "a", encoding="utf-8")
    _capture_stdout = sys.stdout
    _capture_stderr = sys.stderr
    sys.stdout = _TeeStream(_capture_stdout, _capture_file_handle)
    sys.stderr = _TeeStream(_capture_stderr, _capture_file_handle)


atexit.register(_disable_output_capture)


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) by setting shutdown flag."""
    global _shutdown_requested
    _shutdown_requested = True
    print(
        "\nShutdown requested. Finishing current task and exiting gracefully...",
        file=sys.stderr,
    )


def setup_shutdown_handler():
    """Register signal handler for graceful shutdown on Ctrl+C."""
    signal.signal(signal.SIGINT, signal_handler)


def get_shutdown_flag() -> Callable[[], bool]:
    """Return a callable that returns True when shutdown is requested."""
    return lambda: _shutdown_requested


def setup_logging(
    output_dir: Path, verbose: bool = False, log_filename: str = "log.txt"
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        output_dir: Output directory for log file
        verbose: Enable DEBUG level logging
        log_filename: Name of the log file

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    log_path = output_dir / log_filename

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )

    return logging.getLogger(__name__)


def save_log(output_dir: Path, args, log_filename: str = "log.txt") -> None:
    """Save command log to file.

    Args:
        output_dir: Output directory
        args: Parsed arguments (from argparse)
        log_filename: Name of the log file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / log_filename
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

    _enable_output_capture(log_path)


def parse_args(
    parser: argparse.ArgumentParser, args: Optional[list] = None
) -> argparse.Namespace:
    """Parse arguments with graceful shutdown support.

    Args:
        parser: Pre-configured ArgumentParser instance
        args: List of arguments. If None, uses sys.argv

    Returns:
        Parsed arguments
    """
    return parser.parse_args(args)


def validate_directory(
    path: Path, must_exist: bool = True, must_be_dir: bool = True
) -> Path:
    """Validate directory path."""
    from src.common.validators import validate_directory as _validate_directory

    return _validate_directory(path, must_exist=must_exist, must_be_dir=must_be_dir)


def validate_file(path: Path, must_exist: bool = True) -> Path:
    """Validate file path."""
    from src.common.validators import validate_file as _validate_file

    return _validate_file(path, must_exist=must_exist)


def validate_image_extensions(files: list, extensions: Optional[set] = None) -> list:
    """Filter files by valid image extensions.

    Args:
        files: List of file paths
        extensions: Set of valid extensions. Defaults to common image formats

    Returns:
        Filtered list of image files
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    return [f for f in files if Path(f).suffix.lower() in extensions]
