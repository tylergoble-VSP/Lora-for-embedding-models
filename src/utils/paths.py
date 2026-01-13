"""
Used in: ALL scripts writing files to disk.
Purpose:
    Provide a single, consistent mechanism for generating timestamped paths.
"""

from pathlib import Path  # Handles directories and file paths
from datetime import datetime  # Used for generating timestamps


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

