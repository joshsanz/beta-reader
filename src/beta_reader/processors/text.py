"""Text file processor."""

import sys
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console
from rich.text import Text

from ..llm.exceptions import FileProcessingError
from .base import BaseProcessor


class TextProcessor(BaseProcessor):
    """Processor for plain text files."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize text processor."""
        super().__init__(*args, **kwargs)
        self.console = Console()
        self._system_prompt = self._load_system_prompt()

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle text files.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if the file has a .txt extension, False otherwise.
        """
        return file_path.suffix.lower() == ".txt"

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

        try:
            content = self._read_file_content(file_path)
            model_name = self._get_model(model)

            if stream:
                return self._process_with_streaming(content, model_name, output_path)
            else:
                return self._process_without_streaming(content, model_name, output_path)

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

            for chunk in self.client.generate_stream(
                model=model_name,
                prompt=content,
                system_prompt=self._system_prompt,
            ):
                yield chunk

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Streaming failed for {file_path}: {e}") from e

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
        self.console.print("[dim]Streaming output...[/dim]\n")

        result_chunks = []

        try:
            for chunk in self.client.generate_stream(
                model=model,
                prompt=content,
                system_prompt=self._system_prompt,
            ):
                result_chunks.append(chunk)
                # Stream to console with color
                text = Text(chunk, style="green")
                self.console.print(text, end="")
                sys.stdout.flush()

            self.console.print("\n")  # Add final newline

            result = "".join(result_chunks)

            if output_path:
                self._write_file_content(output_path, result)
                self.console.print(f"\n[bold green]Output saved to:[/bold green] {output_path}")

            return result

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Processing interrupted by user[/yellow]")
            raise FileProcessingError("Processing interrupted by user")

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

        with self.console.status("[dim]Processing...[/dim]"):
            result = self.client.generate(
                model=model,
                prompt=content,
                system_prompt=self._system_prompt,
            )

        if output_path:
            self._write_file_content(output_path, result)
            self.console.print(f"[bold green]Output saved to:[/bold green] {output_path}")

        return result

    def _load_system_prompt(self) -> str:
        """Load the system prompt for beta reading.
        
        Returns:
            System prompt content.
            
        Raises:
            FileProcessingError: If system prompt cannot be loaded.
        """
        # Check config path first, then fall back to config method
        config_prompt_path = Path("config") / "system_prompt.txt"
        if config_prompt_path.exists():
            try:
                return self._read_file_content(config_prompt_path)
            except Exception as e:
                raise FileProcessingError(f"Failed to load system prompt: {e}") from e

        # Fall back to config method
        try:
            system_prompt_path = self.config.get_system_prompt_path()
            return self._read_file_content(system_prompt_path)
        except Exception as e:
            raise FileProcessingError(f"Failed to load system prompt: {e}") from e
