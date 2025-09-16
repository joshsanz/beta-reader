"""Text file processor.

This module provides processing capabilities for plain text files (.txt),
including chunking for large files and both streaming and non-streaming modes.
"""

import time
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console

from ..core.chunk_processor import ChunkProcessor
from ..core.text_chunker import TextChunker
from ..llm.exceptions import FileProcessingError
from .base import BaseProcessor


class TextProcessor(BaseProcessor):
    """Processor for plain text files.

    Handles processing of .txt files with automatic chunking for large content,
    progress reporting, and flexible output options.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize text processor.

        Sets up console output, loads system prompt, configures text chunker
        with application settings, and initializes the chunk processor.

        Args:
            *args: Positional arguments passed to BaseProcessor.
            **kwargs: Keyword arguments passed to BaseProcessor.
        """
        super().__init__(*args, **kwargs)
        self.console = Console()
        self.system_prompt = self._load_system_prompt()
        # Initialize chunker with config values
        self.chunker = TextChunker(
            target_word_count=self.config.chunking.target_word_count,
            max_word_count=self.config.chunking.max_word_count
        )
        self.chunk_processor = ChunkProcessor(self.client, self.chunker, self.console)

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle text files.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the file has a .txt extension, False otherwise.
        """
        return file_path.suffix.lower() == ".txt"

    # ========================================================================
    # Public Processing Methods
    # ========================================================================

    def process_file(
        self,
        file_path: Path,
        output_path: Path | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> str:
        """Process a text file with the LLM.

        Args:
            file_path: Path to the input text file.
            output_path: Optional path to save the output. If None, returns the result.
            stream: Whether to stream output to terminal.
            model: Optional model override.

        Returns:
            Processed content as string.

        Raises:
            FileProcessingError: If processing fails.
        """
        if not file_path.exists():
            raise FileProcessingError(f"File not found: {file_path}")

        if not self.can_process(file_path):
            raise FileProcessingError(f"Cannot process file type: {file_path.suffix}")

        start_time = time.time()

        try:
            content = self._read_file_content(file_path)
            model_name = self._get_model(model)

            if stream:
                result = self._process_with_streaming(content, model_name, output_path)
            else:
                result = self._process_without_streaming(content, model_name, output_path)

            # Report total processing time
            end_time = time.time()
            duration = end_time - start_time
            word_count = len(content.split())

            self.console.print(f"\n📊 [bold green]Processing completed in {self._format_duration(duration)}[/bold green]")
            self.console.print(f"📈 Processed {word_count:,} words")

            return result

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Processing failed for {file_path}: {e}") from e

    def process_stream(
        self,
        file_path: Path,
        model: str | None = None,
    ) -> Iterator[str]:
        """Process a text file with streaming output.

        Args:
            file_path: Path to the input text file.
            model: Optional model override.

        Yields:
            Chunks of processed text.

        Raises:
            FileProcessingError: If processing fails.
        """
        if not file_path.exists():
            raise FileProcessingError(f"File not found: {file_path}")

        if not self.can_process(file_path):
            raise FileProcessingError(f"Cannot process file type: {file_path.suffix}")

        try:
            content = self._read_file_content(file_path)
            model_name = self._get_model(model)

            yield from self.client.generate_stream(
                model=model_name,
                prompt=f"[BEGINNING OF CONTENT]\n{content}\n[END OF CONTENT]",
                system_prompt=self.system_prompt,
            )

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Streaming failed for {file_path}: {e}") from e

    # ========================================================================
    # Private Processing Methods
    # ========================================================================

    def _process_with_streaming(
        self,
        content: str,
        model: str,
        output_path: Path | None = None
    ) -> str:
        """Process content with streaming output to terminal.

        Args:
            content: Text content to process.
            model: Model name to use.
            output_path: Optional path to save output.

        Returns:
            Complete processed text.
        """
        self.console.print(f"\n[bold blue]Processing with model:[/bold blue] {model}")

        debug_chunking = getattr(self, '_debug_chunking', False)
        return self.chunk_processor.process_content_streaming(
            content, model, self.system_prompt, output_path, "text", debug_chunking
        )


    def _process_without_streaming(
        self,
        content: str,
        model: str,
        output_path: Path | None = None
    ) -> str:
        """Process content without streaming.

        Args:
            content: Text content to process.
            model: Model name to use.
            output_path: Optional path to save output.

        Returns:
            Complete processed text.
        """
        self.console.print(f"[bold blue]Processing with model:[/bold blue] {model}")

        debug_chunking = getattr(self, '_debug_chunking', False)
        return self.chunk_processor.process_content_non_streaming(
            content, model, self.system_prompt, output_path, "text", debug_chunking
        )


