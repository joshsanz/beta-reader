"""Base processor interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from ..core.config import Config
from ..core.utils import format_duration, load_system_prompt, read_file_content, write_file_content
from ..llm.client import OllamaClient


class BaseProcessor(ABC):
    """Abstract base class for file processors."""

    def __init__(self, client: OllamaClient, config: Config) -> None:
        """Initialize processor.

        Args:
            client: Ollama client for LLM communication.
            config: Application configuration.
        """
        self.client = client
        self.config = config

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the given file.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if this processor can handle the file, False otherwise.
        """
        pass

    @abstractmethod
    def process_file(
        self,
        file_path: Path,
        output_path: Path | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> str:
        """Process a file with the LLM.

        Args:
            file_path: Path to the input file.
            output_path: Optional path to save the output. If None, returns the result.
            stream: Whether to stream output to terminal.
            model: Optional model override.

        Returns:
            Processed content as string.

        Raises:
            FileProcessingError: If processing fails.
        """
        pass

    @abstractmethod
    def process_stream(
        self,
        file_path: Path,
        model: str | None = None,
    ) -> Iterator[str]:
        """Process a file with streaming output.

        Args:
            file_path: Path to the input file.
            model: Optional model override.

        Yields:
            Chunks of processed text.

        Raises:
            FileProcessingError: If processing fails.
        """
        pass

    def _get_model(self, model_override: str | None = None) -> str:
        """Get the model to use for processing.

        Args:
            model_override: Optional model override.

        Returns:
            Model name to use.
        """
        return model_override or self.config.ollama.default_model

    def _read_file_content(self, file_path: Path) -> str:
        """Read content from a file.

        Args:
            file_path: Path to the file to read.

        Returns:
            File content as string.

        Raises:
            FileProcessingError: If file cannot be read.
        """
        return read_file_content(file_path)

    def _write_file_content(self, file_path: Path, content: str) -> None:
        """Write content to a file.

        Args:
            file_path: Path to the file to write.
            content: Content to write.

        Raises:
            FileProcessingError: If file cannot be written.
        """
        return write_file_content(file_path, content)

    def _load_system_prompt(self) -> str:
        """Load the system prompt for beta reading.

        Returns:
            System prompt content.

        Raises:
            FileProcessingError: If system prompt cannot be loaded.
        """
        return load_system_prompt(self.config)

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable format.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string (e.g., "2m 34s", "45.2s").
        """
        return format_duration(seconds)
