"""
Used in: Analytics notebooks and any performance-critical workflows.
Purpose:
    Provide decorators and context managers for recording execution time.
    Automatically logs results into timestamped CSV files.
"""

import time  # Used for capturing timestamps before/after execution
import csv  # For writing time logs as CSV
from pathlib import Path  # Robust path handling for cross-platform support
from src.utils.paths import timestamped_path  # Centralized timestamped output paths

# Generate a timestamped log file name for each run.
# This ensures logs never overwrite each other.
LOG_PATH = timestamped_path("outputs/logs", "timing", "csv")


def timeit(fn):
    """
    Decorator that measures how long a function takes to execute.

    Args:
        fn: The function being decorated.

    Returns:
        A wrapped version of fn that logs its execution time.
    """

    def wrapper(*args, **kwargs):
        start = time.time()  # Start the timer

        result = fn(*args, **kwargs)  # Execute the wrapped function

        duration = time.time() - start  # Compute elapsed time

        # Append a row containing the function name, duration, and timestamp.
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([fn.__name__, duration, time.time()])

        return result  # Return original function output

    return wrapper  # Return the wrapper to be used as decorator


class TimeBlock:
    """
    Context manager for timing code blocks.

    Usage:
        with TimeBlock("my_operation"):
            # code to time
            pass
    """

    def __init__(self, operation_name: str):
        """
        Initialize the time block context manager.

        Args:
            operation_name: Name of the operation being timed.
        """
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log the duration when exiting the context."""
        duration = time.time() - self.start_time

        # Append a row containing the operation name, duration, and timestamp.
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.operation_name, duration, time.time()])

        return False  # Don't suppress exceptions

