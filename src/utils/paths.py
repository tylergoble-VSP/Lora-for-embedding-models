"""
Used in: ALL scripts writing files to disk.
Purpose:
    Provide a single, consistent mechanism for generating timestamped paths.
"""

from pathlib import Path  # Handles directories and file paths
from datetime import datetime  # Used for generating timestamps
from typing import Optional  # Type hints for optional return values
import glob  # For finding files matching patterns


def timestamp() -> str:
    """
    Create a timestamp string in the format YYYYMMDD_HHMMSS.

    Returns:
        A formatted timestamp string.
    """

    # datetime.now() gets the current system time
    # strftime() formats it into a filename-safe string
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamped_path(base_folder: str, base_name: str, ext: str) -> Path:
    """
    Generate a timestamped file path inside base_folder.

    Args:
        base_folder: Directory where file will be stored.
        base_name: Descriptive filename prefix.
        ext: File extension without dot (e.g., 'csv').

    Returns:
        A Path object for the timestamped file.
    """

    folder = Path(base_folder)

    # Ensure the directory exists before writing
    folder.mkdir(parents=True, exist_ok=True)

    # Construct name: baseName_timestamp.extension
    ts = timestamp()
    filename = f"{base_name}_{ts}.{ext}"

    # Combine folder path and filename
    return folder / filename


def find_latest_timestamped_file(
    base_folder: str,
    base_name: str,
    ext: str
) -> Optional[Path]:
    """
    Find the most recently created timestamped file matching the pattern.

    This is useful for loading the latest generated file without manually
    specifying the timestamp. Files are matched by pattern:
    {base_name}_YYYYMMDD_HHMMSS.{ext}

    Args:
        base_folder: Directory to search in.
        base_name: Filename prefix to match.
        ext: File extension without dot (e.g., 'csv').

    Returns:
        Path to the most recent matching file, or None if no files found.
    """
    folder = Path(base_folder)

    # Construct glob pattern to match timestamped files
    # Pattern: baseName_YYYYMMDD_HHMMSS.ext
    pattern = folder / f"{base_name}_*.{ext}"

    # Find all matching files
    matching_files = glob.glob(str(pattern))

    if not matching_files:
        return None

    # Return the most recently modified file
    # This works because timestamped files are created in chronological order
    latest_file = max(matching_files, key=lambda p: Path(p).stat().st_mtime)

    return Path(latest_file)

