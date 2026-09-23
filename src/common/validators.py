"""Common validation utilities for all scripts."""

import os
from pathlib import Path
from typing import List, Optional, Set


IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_EXTENSIONS: Set[str] = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v', '.mpeg', '.mpg', '.wmv', '.3gp', '.ts'}


def validate_directory(path: Path, must_exist: bool = True, must_be_dir: bool = True) -> Path:
    """Validate directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether directory must exist
        must_be_dir: Whether path must be a directory
    
    Returns:
        Validated Path object
    
    Raises:
        FileNotFoundError: If directory doesn't exist and must_exist is True
        NotADirectoryError: If path is not a directory and must_be_dir is True
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    
    if must_be_dir and not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    return path


def validate_file(path: Path, must_exist: bool = True) -> Path:
    """Validate file path.
    
    Args:
        path: Path to validate
        must_exist: Whether file must exist
    
    Returns:
        Validated Path object
    
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist is True
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")
    
    return path


def find_images(directory: Path, extensions: Optional[Set[str]] = None) -> List[Path]:
    """Find all image files in directory.
    
    Args:
        directory: Directory to search
        extensions: Set of extensions to include. Defaults to IMAGE_EXTENSIONS
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    return [f for f in directory.iterdir() 
            if f.is_file() and f.suffix.lower() in extensions]


def find_videos(directory: Path, extensions: Optional[Set[str]] = None) -> List[Path]:
    """Find all video files in directory.
    
    Args:
        directory: Directory to search
        extensions: Set of extensions to include. Defaults to VIDEO_EXTENSIONS
    
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = VIDEO_EXTENSIONS
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    return [f for f in directory.iterdir() 
            if f.is_file() and f.suffix.lower() in extensions]


def count_files(directory: Path, extensions: Optional[Set[str]] = None) -> int:
    """Count files with given extensions in directory.
    
    Args:
        directory: Directory to count files in
        extensions: Set of extensions to count. Defaults to all
    
    Returns:
        Number of files matching extensions
    """
    files = find_images(directory, extensions) if extensions is None or extensions <= IMAGE_EXTENSIONS else find_videos(directory, extensions)
    return len(files)


def validate_thread_count(threads: int, max_multiplier: int = 2) -> int:
    """Validate and cap thread count.
    
    Args:
        threads: Requested thread count
        max_multiplier: Maximum multiplier of CPU count
    
    Returns:
        Validated thread count
    """
    max_threads = os.cpu_count() or 8
    max_allowed = max_threads * max_multiplier
    
    if threads > max_allowed:
        print(f"Warning: Thread count {threads} exceeds recommended maximum {max_allowed}", 
              file=__import__('sys').stderr)
        return max_allowed
    
    return threads


def validate_range(value: float, min_val: float, max_val: float, name: str) -> float:
    """Validate value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error message
    
    Returns:
        Validated value
    
    Raises:
        ValueError: If value is out of range
    """
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value
