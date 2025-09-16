"""Shared chunk processing utilities for processors.

This module provides a unified interface for processing content in chunks,
supporting both streaming and non-streaming modes. It abstracts the common
patterns used across text and EPUB processors.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from ..llm.client import OllamaClient
from ..llm.exceptions import FileProcessingError
from .text_chunker import TextChunk, TextChunker
from .utils import write_file_content


class ChunkProcessor:
    """Handles common chunk processing operations for streaming and non-streaming modes.

    This class provides a unified interface for processing content that may need
    to be chunked for large texts, supporting both streaming output to console
    and direct processing modes.
    """

    def __init__(self, client: OllamaClient, chunker: TextChunker, console: Console) -> None:
        """Initialize chunk processor.

        Args:
            client: Ollama client for LLM communication.
            chunker: Text chunker instance configured with target/max word counts.
            console: Rich console instance for formatted output.
        """
        self.client = client
        self.chunker = chunker
        self.console = console

    def process_content_streaming(
        self,
        content: str,
        model: str,
        system_prompt: str,
        output_path: Path | None = None,
        title: str = "Content",
        debug_chunking: bool = False,
    ) -> str:
        """Process content with streaming output.

        Args:
            content: Content to process.
            model: Model name to use.
            system_prompt: System prompt for processing.
            output_path: Optional output file path.
            title: Title for progress display.
            debug_chunking: Whether to show chunk debugging info.

        Returns:
            Processed content.

        Raises:
            FileProcessingError: If processing fails.
        """
        word_count = len(content.split())

        if word_count <= self.chunker.target_word_count:
            return self._process_single_chunk_streaming(
                content, model, system_prompt, output_path, title
            )
        else:
            return self._process_chunked_content_streaming(
                content, model, system_prompt, output_path, title, debug_chunking
            )

    def process_content_non_streaming(
        self,
        content: str,
        model: str,
        system_prompt: str,
        output_path: Path | None = None,
        title: str = "Content",
        debug_chunking: bool = False,
    ) -> str:
        """Process content without streaming.

        Args:
            content: Content to process.
            model: Model name to use.
            system_prompt: System prompt for processing.
            output_path: Optional output file path.
            title: Title for progress display.
            debug_chunking: Whether to show chunk debugging info.

        Returns:
            Processed content.

        Raises:
            FileProcessingError: If processing fails.
        """
        word_count = len(content.split())

        if word_count <= self.chunker.target_word_count:
            return self._process_single_chunk_non_streaming(
                content, model, system_prompt, output_path, title
            )
        else:
            return self._process_chunked_content_non_streaming(
                content, model, system_prompt, output_path, title, debug_chunking
            )

    def _process_single_chunk_streaming(
        self,
        content: str,
        model: str,
        system_prompt: str,
        output_path: Path | None = None,
        title: str = "Content",
    ) -> str:
        """Process content as a single chunk with streaming."""
        if output_path:
            self.console.print(f"[dim]Streaming output for {title}...[/dim]")
        else:
            self.console.print(f"[dim]Streaming output for {title}...[/dim]\n")

        result_chunks = []

        try:
            for chunk in self.client.generate_stream(
                model=model,
                prompt=f"[BEGINNING OF CONTENT]\n{content}\n[END OF CONTENT]",
                system_prompt=system_prompt,
            ):
                result_chunks.append(chunk)
                if not output_path:  # Only stream to console if not saving to file
                    text = Text(chunk, style="green")
                    self.console.print(text, end="")
                    sys.stdout.flush()

            if not output_path:
                self.console.print("\n")

            result = "".join(result_chunks)

            if output_path:
                write_file_content(output_path, result)
                self.console.print(f"[bold green]Output saved to:[/bold green] {output_path}")

            return result

        except KeyboardInterrupt:
            self.console.print(f"\n[yellow]Processing interrupted for {title}[/yellow]")
            raise FileProcessingError(f"Processing interrupted for {title}")

    def _process_chunked_content_streaming(
        self,
        content: str,
        model: str,
        system_prompt: str,
        output_path: Path | None = None,
        title: str = "Content",
        debug_chunking: bool = False,
    ) -> str:
        """Process content in chunks with streaming."""
        chunks = self.chunker.chunk_text(content)

        if not chunks:
            return content

        self.console.print(f"[blue]Chunking {title}: {len(chunks)} chunks ({len(content.split())} words)[/blue]")

        # Show debug information about chunk boundaries
        if debug_chunking:
            self.console.print(f"\n[yellow]Chunk Boundaries Debug Information for {title}:[/yellow]")
            boundaries = self.chunker.get_chunk_boundaries_debug(content)
            for boundary in boundaries:
                self.console.print(f"[dim]{boundary}[/dim]")
            self.console.print()

        processed_chunks = []

        try:
            for i, chunk in enumerate(chunks, 1):
                if output_path:
                    self.console.print(f"[dim]Processing chunk {i}/{len(chunks)}...[/dim]")
                else:
                    self.console.print(f"[dim]Processing chunk {i}/{len(chunks)}...[/dim]\n")

                result_parts = []
                for stream_chunk in self.client.generate_stream(
                    model=model,
                    prompt=f"[BEGINNING OF CONTENT]\n{chunk.content}\n[END OF CONTENT]",
                    system_prompt=system_prompt,
                ):
                    result_parts.append(stream_chunk)
                    if not output_path:  # Only stream to console if not saving to file
                        text = Text(stream_chunk, style="green")
                        self.console.print(text, end="")
                        sys.stdout.flush()

                if not output_path:
                    self.console.print("\n")

                chunk_result = "".join(result_parts)
                processed_chunks.append(TextChunk(
                    content=chunk_result,
                    chunk_number=chunk.chunk_number,
                    total_chunks=chunk.total_chunks,
                    word_count=len(chunk_result.split()),
                    start_position=chunk.start_position,
                    end_position=chunk.end_position
                ))

            # Reassemble the processed chunks
            result = self.chunker.reassemble_chunks(processed_chunks)

            if output_path:
                write_file_content(output_path, result)
                self.console.print(f"[bold green]Output saved to:[/bold green] {output_path}")

            return result

        except KeyboardInterrupt:
            self.console.print(f"\n[yellow]Processing interrupted for {title}[/yellow]")
            raise FileProcessingError(f"Processing interrupted for {title}")

    def _process_single_chunk_non_streaming(
        self,
        content: str,
        model: str,
        system_prompt: str,
        output_path: Path | None = None,
        title: str = "Content",
    ) -> str:
        """Process content as a single chunk without streaming."""
        with self.console.status(f"[dim]Processing {title}...[/dim]"):
            result = self.client.generate(
                model=model,
                prompt=f"[BEGINNING OF CONTENT]\n{content}\n[END OF CONTENT]",
                system_prompt=system_prompt,
            )

        if output_path:
            write_file_content(output_path, result)
            self.console.print(f"[bold green]Output saved to:[/bold green] {output_path}")

        return result

    def _process_chunked_content_non_streaming(
        self,
        content: str,
        model: str,
        system_prompt: str,
        output_path: Path | None = None,
        title: str = "Content",
        debug_chunking: bool = False,
    ) -> str:
        """Process content in chunks without streaming."""
        chunks = self.chunker.chunk_text(content)

        if not chunks:
            return content

        self.console.print(f"[blue]Chunking {title}: {len(chunks)} chunks ({len(content.split())} words)[/blue]")

        # Show debug information about chunk boundaries
        if debug_chunking:
            self.console.print(f"\n[yellow]Chunk Boundaries Debug Information for {title}:[/yellow]")
            boundaries = self.chunker.get_chunk_boundaries_debug(content)
            for boundary in boundaries:
                self.console.print(f"[dim]{boundary}[/dim]")
            self.console.print()

        processed_chunks = []

        try:
            with Progress() as progress:
                task = progress.add_task(f"Processing {title}", total=len(chunks))

                for chunk in chunks:
                    progress.update(task, description=f"Processing chunk {chunk.chunk_number}/{chunk.total_chunks}")

                    chunk_result = self.client.generate(
                        model=model,
                        prompt=f"[BEGINNING OF CONTENT]\n{chunk.content}\n[END OF CONTENT]",
                        system_prompt=system_prompt,
                    )

                    processed_chunks.append(TextChunk(
                        content=chunk_result,
                        chunk_number=chunk.chunk_number,
                        total_chunks=chunk.total_chunks,
                        word_count=len(chunk_result.split()),
                        start_position=chunk.start_position,
                        end_position=chunk.end_position
                    ))

                    progress.advance(task)

            # Reassemble the processed chunks
            result = self.chunker.reassemble_chunks(processed_chunks)

            if output_path:
                write_file_content(output_path, result)
                self.console.print(f"[bold green]Output saved to:[/bold green] {output_path}")

            return result

        except KeyboardInterrupt:
            self.console.print(f"\n[yellow]Processing interrupted for {title}[/yellow]")
            raise FileProcessingError(f"Processing interrupted for {title}")