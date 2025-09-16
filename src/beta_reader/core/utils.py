"""Common utilities for beta-reader application."""

import time
from pathlib import Path

from ..llm.exceptions import FileProcessingError


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string (e.g., "2m 34s", "45.2s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"


def load_system_prompt(config) -> str:
    """Load the system prompt for beta reading.

    Args:
        config: Application configuration instance.

    Returns:
        System prompt content.

    Raises:
        FileProcessingError: If system prompt cannot be loaded.
    """
    # Check config path first, then fall back to config method
    config_prompt_path = Path("config") / "system_prompt.txt"
    if config_prompt_path.exists():
        try:
            return read_file_content(config_prompt_path)
        except Exception as e:
            raise FileProcessingError(f"Failed to load system prompt: {e}") from e

    # Fall back to config method
    try:
        system_prompt_path = config.get_system_prompt_path()
        return read_file_content(system_prompt_path)
    except Exception as e:
        raise FileProcessingError(f"Failed to load system prompt: {e}") from e


def read_file_content(file_path: Path) -> str:
    """Read content from a file.

    Args:
        file_path: Path to the file to read.

    Returns:
        File content as string.

    Raises:
        FileProcessingError: If file cannot be read.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise FileProcessingError(f"Failed to read file {file_path}: {e}") from e


def write_file_content(file_path: Path, content: str) -> None:
    """Write content to a file.

    Args:
        file_path: Path to the file to write.
        content: Content to write.

    Raises:
        FileProcessingError: If file cannot be written.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise FileProcessingError(f"Failed to write file {file_path}: {e}") from e


def get_safe_filename(name: str) -> str:
    """Convert a string to a safe filename by replacing problematic characters.

    Args:
        name: String to convert to safe filename.

    Returns:
        Safe filename string.
    """
    return name.replace(":", "_").replace("/", "_").replace(" ", "_").replace("-", "_")