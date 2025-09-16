"""Epub file processor."""

import re
import time
from collections.abc import Iterator
from pathlib import Path

import ebooklib
from ebooklib import epub
from rich.console import Console
from rich.progress import Progress

from ..core.batch_state import BatchStateManager
from ..core.format_converter import ensure_markdown_format, ensure_html_format
from ..core.text_chunker import TextChunk, TextChunker
from ..llm.exceptions import FileProcessingError
from .base import BaseProcessor


class EpubProcessor(BaseProcessor):
    """Processor for EPUB files."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize epub processor."""
        super().__init__(*args, **kwargs)
        self.console = Console()
        self.system_prompt = self._load_system_prompt()
        self.batch_manager = BatchStateManager()
        # Initialize chunker with config values
        self.chunker = TextChunker(
            target_word_count=self.config.chunking.target_word_count,
            max_word_count=self.config.chunking.max_word_count
        )

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle epub files.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the file has an .epub extension, False otherwise.
        """
        return file_path.suffix.lower() == ".epub"

    # ========================================================================
    # Public Processing Methods
    # ========================================================================

    def process_file(
        self,
        file_path: Path,
        output_path: Path | None = None,
        stream: bool = False,
        model: str | None = None,
        chapter: int | None = None,
        batch: bool = False,
        resume_batch_id: str | None = None,
    ) -> str:
        """Process an epub file with the LLM.

        Args:
            file_path: Path to the input epub file.
            output_path: Optional path to save the output epub. If None, returns text.
            stream: Whether to stream output to terminal.
            model: Optional model override.
            chapter: Optional specific chapter number to process (1-indexed).
            batch: Whether to process all chapters in batch mode.
            resume_batch_id: Optional batch ID to resume from.

        Returns:
            Processed content as string (or path to output epub if output_path specified).

        Raises:
            FileProcessingError: If processing fails.
        """
        if not file_path.exists():
            raise FileProcessingError(f"File not found: {file_path}")

        if not self.can_process(file_path):
            raise FileProcessingError(f"Cannot process file type: {file_path.suffix}")

        try:
            model_name = self._get_model(model)

            # Load the epub
            book = epub.read_epub(str(file_path))

            if chapter is not None:
                # Process single chapter
                return self._process_single_chapter(book, chapter, model_name, stream, output_path)
            elif batch or output_path or resume_batch_id:
                # Process all chapters and create new epub
                return self._process_batch(book, file_path, model_name, stream, output_path, resume_batch_id)
            else:
                # Just list chapters for user to choose
                self._list_chapters(book)
                return "Use --chapter N to process a specific chapter, or --batch to process all chapters."

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Processing failed for {file_path}: {e}") from e

    def process_stream(
        self,
        file_path: Path,
        model: str | None = None,
        chapter: int | None = None,
    ) -> Iterator[str]:
        """Process an epub file chapter with streaming output.

        Args:
            file_path: Path to the input epub file.
            model: Optional model override.
            chapter: Chapter number to process (1-indexed). Required for streaming.

        Yields:
            Chunks of processed text.

        Raises:
            FileProcessingError: If processing fails or chapter not specified.
        """
        if chapter is None:
            raise FileProcessingError("Chapter number required for streaming epub processing")

        if not file_path.exists():
            raise FileProcessingError(f"File not found: {file_path}")

        if not self.can_process(file_path):
            raise FileProcessingError(f"Cannot process file type: {file_path.suffix}")

        try:
            model_name = self._get_model(model)
            book = epub.read_epub(str(file_path))
            chapters = self._extract_chapters(book)

            if chapter < 1 or chapter > len(chapters):
                raise FileProcessingError(f"Chapter {chapter} not found. Available chapters: 1-{len(chapters)}")

            chapter_title, chapter_content = chapters[chapter - 1]

            yield from self.client.generate_stream(
                model=model_name,
                prompt=f"[BEGINNING OF CONTENT]\n{chapter_content}\n[END OF CONTENT]",
                system_prompt=self.system_prompt,
            )

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Streaming failed for {file_path}: {e}") from e

    # ========================================================================
    # Chapter Extraction and Content Processing
    # ========================================================================

    def _extract_chapters(self, book: epub.EpubBook) -> list[tuple[str, str]]:
        """Extract chapters from epub book.

        Args:
            book: The epub book object.

        Returns:
            List of (title, content) tuples for each chapter.
        """
        chapters = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Get the content as text
                content = item.get_content().decode('utf-8')

                # Extract title from content or use filename
                title = self._extract_title(content) or item.get_name()

                # Clean HTML and get text content
                text_content = self._html_to_text(content)

                # Skip very short content (likely not actual chapters)
                if len(text_content.strip()) > 100:
                    chapters.append((title, text_content))

        return chapters

    def _extract_title(self, html_content: str) -> str | None:
        """Extract title from HTML content.

        Args:
            html_content: HTML content to extract title from.

        Returns:
            Extracted title or None if not found.
        """
        # Try to find title in h1, h2, or title tags
        title_patterns = [
            r'<h1[^>]*>(.*?)</h1>',
            r'<h2[^>]*>(.*?)</h2>',
            r'<title[^>]*>(.*?)</title>',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
                if title:
                    return title

        return None

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML content to plain text while preserving basic formatting.

        Args:
            html_content: HTML content to convert.

        Returns:
            Plain text with basic formatting preserved.
        """
        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert common HTML elements to text equivalents
        html_content = re.sub(r'<br[^>]*>', '\n', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<p[^>]*>', '\n\n', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'</p>', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<h[1-6][^>]*>', '\n\n', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'</h[1-6]>', '\n\n', html_content, flags=re.IGNORECASE)

        # Remove all other HTML tags
        html_content = re.sub(r'<[^>]+>', '', html_content)

        # Clean up whitespace
        html_content = re.sub(r'\n\s*\n', '\n\n', html_content)
        html_content = re.sub(r'^\s+', '', html_content, flags=re.MULTILINE)

        return html_content.strip()

    def _list_chapters(self, book: epub.EpubBook) -> None:
        """List chapters in the epub book.

        Args:
            book: The epub book object.
        """
        chapters = self._extract_chapters(book)

        self.console.print(f"\n[bold blue]Found {len(chapters)} chapters in epub:[/bold blue]\n")

        for i, (title, content) in enumerate(chapters, 1):
            word_count = len(content.split())
            title_display = title[:50] + "..." if len(title) > 50 else title
            self.console.print(f"  {i:2d}. [cyan]{title_display}[/cyan] ({word_count:,} words)")

    # ========================================================================
    # Single Chapter Processing
    # ========================================================================

    def _process_single_chapter(
        self,
        book: epub.EpubBook,
        chapter_num: int,
        model: str,
        stream: bool,
        output_path: Path | None = None,
    ) -> str:
        """Process a single chapter from the epub.

        Args:
            book: The epub book object.
            chapter_num: Chapter number (1-indexed).
            model: Model name to use.
            stream: Whether to stream output.
            output_path: Optional output file path.

        Returns:
            Processed chapter content.
        """
        chapters = self._extract_chapters(book)

        if chapter_num < 1 or chapter_num > len(chapters):
            raise FileProcessingError(f"Chapter {chapter_num} not found. Available chapters: 1-{len(chapters)}")

        chapter_title, chapter_content = chapters[chapter_num - 1]

        self.console.print(f"\n[bold blue]Processing chapter {chapter_num}:[/bold blue] {chapter_title}")

        if stream:
            return self._process_with_streaming(chapter_content, model, output_path, chapter_title)
        else:
            return self._process_without_streaming(chapter_content, model, output_path, chapter_title)

    # ========================================================================
    # Batch Processing
    # ========================================================================

    def _process_batch(
        self,
        book: epub.EpubBook,
        original_path: Path,
        model: str,
        stream: bool,
        output_path: Path | None = None,
        resume_batch_id: str | None = None,
    ) -> str:
        """Process all chapters in batch mode with resume support.

        Args:
            book: The epub book object.
            original_path: Path to original epub file.
            model: Model name to use.
            stream: Whether to stream output.
            output_path: Output epub path.
            resume_batch_id: Optional batch ID to resume from.

        Returns:
            Path to output epub or summary text.
        """
        start_time = time.time()

        chapters = self._extract_chapters(book)

        if not chapters:
            raise FileProcessingError("No chapters found in epub file")

        # Load or create batch state
        if resume_batch_id:
            try:
                # Resolve short hash to full batch ID if needed
                full_batch_id = self.batch_manager.resolve_short_hash(resume_batch_id)
                batch_state = self.batch_manager.load_batch_state(full_batch_id)
                short_hash = self.batch_manager.get_short_hash(full_batch_id)
                self.console.print(f"\n[yellow]Resuming batch processing:[/yellow] {short_hash}")
                self.console.print(f"[dim]Full ID: {full_batch_id}[/dim]")
                self.console.print(f"[dim]Progress: {batch_state.completed_chapters}/{batch_state.total_chapters} chapters completed[/dim]")

                # Check if batch is already completed
                if batch_state.status == 'completed' and batch_state.completed_chapters == batch_state.total_chapters:
                    self.console.print(f"\n[green]Batch {short_hash} is already completed![/green]")

                    if output_path:
                        # Check if the requested output file already exists
                        if output_path.exists():
                            self.console.print(f"[green]Requested output file already exists:[/green] {output_path}")
                            return str(output_path)

                        # Check if original output file exists and user wants same path
                        original_output = Path(batch_state.output_directory) if batch_state.output_directory else None
                        if original_output and original_output == output_path and original_output.exists():
                            self.console.print(f"[green]Output file already exists:[/green] {original_output}")
                            return str(original_output)
                        else:
                            # Regenerate the output EPUB from existing processed chapters
                            self.console.print(f"[blue]Regenerating output EPUB from completed batch to:[/blue] {output_path}")
                            # Continue with normal processing to regenerate the EPUB
                    else:
                        # Just return the summary text
                        self.console.print("[blue]Generating summary from completed batch...[/blue]")
                        # Continue with normal processing to generate summary

            except Exception as e:
                self.console.print(f"[red]Could not resume batch {resume_batch_id}: {e}[/red]")
                self.console.print("[yellow]Starting new batch instead...[/yellow]")
                resume_batch_id = None

        if not resume_batch_id:
            # Create new batch state
            batch_id = self.batch_manager.generate_batch_id(original_path, model)
            batch_state = self.batch_manager.create_batch_state(
                batch_id, original_path, chapters, model, output_path
            )
            short_hash = self.batch_manager.get_short_hash(batch_id)
            self.console.print(f"\n[bold blue]Starting batch processing:[/bold blue] {short_hash}")
            self.console.print(f"[dim]Full ID: {batch_id}[/dim]")

        self.console.print(f"\n[bold blue]Processing {len(chapters)} chapters with model:[/bold blue] {model}")

        # Collect completed chapters if resuming
        processed_chapters = []
        if resume_batch_id:
            # Load previously completed chapters
            for i, chapter_state in enumerate(batch_state.chapters):
                if chapter_state.status == 'completed' and chapter_state.output_file:
                    try:
                        with open(chapter_state.output_file, encoding='utf-8') as f:
                            content = f.read()
                        # Note: Previously processed content might be in HTML format
                        # The format converter will handle this appropriately when creating EPUB
                        processed_chapters.append((chapter_state.chapter_title, content))
                    except Exception:
                        # If we can't load previous output, reprocess this chapter
                        chapter_state.status = 'pending'

        start_chapter = len(processed_chapters)

        with Progress() as progress:
            task = progress.add_task("[green]Processing chapters...", total=len(chapters))
            progress.update(task, completed=start_chapter)

            try:
                for i, (title, content) in enumerate(chapters[start_chapter:], start_chapter + 1):
                    chapter_index = i - 1
                    progress.update(task, description=f"[green]Chapter {i}: {title[:20]}...")

                    # Update chapter status to processing
                    self.batch_manager.update_chapter_status(batch_state, chapter_index, 'processing')

                    # Start timing for this chapter
                    chapter_start_time = time.time()

                    try:
                        # Convert HTML to Markdown for cleaner model input
                        markdown_content = ensure_markdown_format(content)

                        if stream:
                            self.console.print(f"\n[bold]Chapter {i}: {title}[/bold]")
                            processed_content = self._process_with_streaming(markdown_content, model, None, title)
                        else:
                            processed_content = self._process_without_streaming(markdown_content, model, None, title)

                        # Save chapter output for resume capability
                        if output_path:
                            output_dir = output_path.parent / f".{output_path.stem}_chapters"
                            output_dir.mkdir(exist_ok=True)
                            chapter_file = output_dir / f"chapter_{i:03d}_{title[:50].replace('/', '_')}.txt"
                            with open(chapter_file, 'w', encoding='utf-8') as f:
                                f.write(processed_content)

                            chapter_output_file = str(chapter_file)
                        else:
                            chapter_output_file = None

                        processed_chapters.append((title, processed_content))

                        # Calculate chapter processing time
                        chapter_end_time = time.time()
                        chapter_duration = chapter_end_time - chapter_start_time
                        chapter_word_count = len(processed_content.split())

                        # Update chapter status to completed
                        self.batch_manager.update_chapter_status(
                            batch_state,
                            chapter_index,
                            'completed',
                            output_file=chapter_output_file,
                            word_count=chapter_word_count
                        )

                        # Report individual chapter completion with timing
                        completed_count = len(processed_chapters)
                        total_count = len(chapters)
                        self.console.print(f"\n✓ [green]Chapter {i}: {title[:30]}{'...' if len(title) > 30 else ''}[/green] ([bold]{self._format_duration(chapter_duration)}[/bold]) - {completed_count}/{total_count} completed")

                        progress.update(task, advance=1)

                    except Exception as e:
                        # Check if this was an interruption - if so, re-raise as KeyboardInterrupt
                        if isinstance(e, FileProcessingError) and "interrupted" in str(e):
                            # Mark chapter as paused and re-raise for batch handling
                            self.batch_manager.update_chapter_status(
                                batch_state,
                                chapter_index,
                                'paused',
                                error_message=str(e)
                            )
                            raise KeyboardInterrupt()

                        # Mark chapter as failed but continue with others
                        self.batch_manager.update_chapter_status(
                            batch_state,
                            chapter_index,
                            'failed',
                            error_message=str(e)
                        )
                        self.console.print(f"[red]Failed to process chapter {i}: {e}[/red]")
                        # Add empty content to maintain chapter order
                        processed_chapters.append((title, f"[ERROR: Failed to process - {e}]"))
                        progress.update(task, advance=1)

            except KeyboardInterrupt:
                # Mark batch as paused for resume
                batch_state.status = 'paused'
                self.batch_manager.save_batch_state(batch_state)
                self.console.print(f"\n[yellow]Batch processing paused. Resume with batch ID:[/yellow] {batch_state.batch_id}")
                raise FileProcessingError("Batch processing interrupted by user")

        if output_path:
            # Validate processed_chapters before creating EPUB
            if not processed_chapters:
                raise FileProcessingError("No processed chapters available to create EPUB. This might indicate a problem with loading completed chapters from previous batch run.")


            # Create new epub with processed content
            output_epub_path = self._create_processed_epub(book, processed_chapters, original_path, output_path)
            self.console.print(f"\n[bold green]Processed epub saved to:[/bold green] {output_epub_path}")

            # Report total processing time and batch statistics
            end_time = time.time()
            duration = end_time - start_time
            total_words = sum(len(content.split()) for _, content in processed_chapters)
            avg_time_per_chapter = duration / len(processed_chapters) if processed_chapters else 0

            self.console.print(f"\n📊 [bold green]Processing completed in {self._format_duration(duration)}[/bold green]")
            self.console.print(f"📈 Processed {len(processed_chapters)} chapters, {total_words:,} words")
            self.console.print(f"⚡ Average: {self._format_duration(avg_time_per_chapter)} per chapter")

            return str(output_epub_path)
        else:
            # Return combined text
            result = "\n\n".join([f"# {title}\n\n{content}" for title, content in processed_chapters])

            # Report total processing time for text output
            end_time = time.time()
            duration = end_time - start_time
            total_words = sum(len(content.split()) for _, content in processed_chapters)

            self.console.print(f"\n📊 [bold green]Processing completed in {self._format_duration(duration)}[/bold green]")
            self.console.print(f"📈 Processed {len(processed_chapters)} chapters, {total_words:,} words")

            return result

    # ========================================================================
    # EPUB Creation and File Utilities
    # ========================================================================

    def _create_processed_epub(
        self,
        original_book: epub.EpubBook,
        processed_chapters: list[tuple[str, str]],
        original_path: Path,
        output_path: Path,
    ) -> Path:
        """Create a new epub with processed content.

        Args:
            original_book: Original epub book object.
            processed_chapters: List of (title, processed_content) tuples.
            original_path: Path to original epub.
            output_path: Path for output epub.

        Returns:
            Path to created epub file.
        """
        # Create new epub
        new_book = epub.EpubBook()

        # Copy metadata from original
        try:
            identifier_metadata = original_book.get_metadata('DC', 'identifier')
            identifier = identifier_metadata[0][0] if identifier_metadata else 'processed-book'
            new_book.set_identifier(identifier)
        except Exception:
            new_book.set_identifier('processed-book')

        try:
            title_metadata = original_book.get_metadata('DC', 'title')
            title = f"{title_metadata[0][0]} (Beta Read)" if title_metadata else 'Processed Book'
            new_book.set_title(title)
        except Exception:
            new_book.set_title('Processed Book')

        new_book.set_language('en')

        # Copy authors
        authors = original_book.get_metadata('DC', 'creator')
        if authors:
            for author in authors:
                new_book.add_author(author[0])
        else:
            new_book.add_author('Unknown Author')

        # Add processed chapters
        spine = ['nav']
        toc = []

        for i, (title, content) in enumerate(processed_chapters, 1):
            # Create chapter
            chapter = epub.EpubHtml(title=title, file_name=f'chapter_{i}.xhtml', lang='en')

            # Convert processed Markdown content back to HTML for EPUB
            html_content = ensure_html_format(content)

            # Add chapter title if not already present
            if not html_content.startswith('<h1'):
                html_content = f'<h1>{title}</h1>\n{html_content}'

            # Skip empty content
            if not html_content or not html_content.strip():
                self.console.print(f"[yellow]Warning: Empty content for chapter {i}: {title}[/yellow]")
                continue

            # Set content directly as per ebooklib example
            chapter.content = html_content

            new_book.add_item(chapter)
            spine.append(chapter)
            toc.append(chapter)

        # Add navigation - use simple TOC structure for now
        new_book.toc = toc

        # Add navigation files
        new_book.add_item(epub.EpubNcx())
        new_book.add_item(epub.EpubNav())

        # Define spine - make sure it's not empty
        if len(spine) <= 1:  # Only 'nav' in spine
            raise FileProcessingError(f"No chapters were successfully added to EPUB spine. Expected {len(processed_chapters)} chapters.")

        new_book.spine = spine

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write epub
        try:
            epub.write_epub(str(output_path), new_book)
        except Exception as e:
            self.console.print(f"[red]Error writing EPUB: {e}[/red]")
            raise

        return output_path

    def _wrap_html_content(self, html_content: str, title: str) -> str:
        """Wrap existing HTML content in proper XHTML structure.

        Args:
            html_content: HTML content that's already formatted.
            title: Chapter title.

        Returns:
            Complete XHTML document.
        """
        import html
        return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>{html.escape(title)}</title>
</head>
<body>
<h1>{html.escape(title)}</h1>
{html_content}
</body>
</html>'''

    def _text_to_simple_html(self, text_content: str, title: str) -> str:
        """Convert plain text to simple HTML format for EPUB chapters.

        Args:
            text_content: Plain text content.
            title: Chapter title.

        Returns:
            Simple HTML content (not a complete document).
        """
        import html
        escaped_text = html.escape(text_content)

        # Convert line breaks to paragraphs
        paragraphs = escaped_text.split('\n\n')
        html_paragraphs = [f'<p>{para.replace(chr(10), "<br/>")}</p>' for para in paragraphs if para.strip()]

        return f'<h1>{html.escape(title)}</h1>\n' + '\n'.join(html_paragraphs)

    def _text_to_html(self, text_content: str, title: str) -> str:
        """Convert plain text back to HTML format.

        Args:
            text_content: Plain text content.
            title: Chapter title.

        Returns:
            HTML formatted content.
        """
        # Escape HTML characters
        import html
        escaped_text = html.escape(text_content)

        # Convert line breaks to paragraphs
        paragraphs = escaped_text.split('\n\n')
        html_paragraphs = [f'<p>{para.replace(chr(10), "<br/>")}</p>' for para in paragraphs if para.strip()]

        return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>{html.escape(title)}</title>
</head>
<body>
<h1>{html.escape(title)}</h1>
{chr(10).join(html_paragraphs)}
</body>
</html>'''

    def _process_with_streaming(
        self,
        content: str,
        model: str,
        output_path: Path | None = None,
        chapter_title: str = "Chapter"
    ) -> str:
        """Process content with streaming output to terminal."""
        # Check if content needs chunking
        word_count = len(content.split())
        debug_chunking = getattr(self, '_debug_chunking', False)
        if word_count <= self.chunker.target_word_count:
            return self._process_single_chunk_streaming(content, model, output_path, chapter_title)
        else:
            return self._process_chunked_content_streaming(content, model, output_path, chapter_title, debug_chunking)

    def _process_single_chunk_streaming(
        self,
        content: str,
        model: str,
        output_path: Path | None = None,
        chapter_title: str = "Chapter"
    ) -> str:
        """Process content as a single chunk with streaming."""
        if output_path:
            self.console.print(f"[dim]Streaming output for {chapter_title}...[/dim]")
        else:
            self.console.print(f"[dim]Streaming output for {chapter_title}...[/dim]\n")

        result_chunks = []

        try:
            for chunk in self.client.generate_stream(
                model=model,
                prompt=f"[BEGINNING OF CONTENT]\n{content}\n[END OF CONTENT]",
                system_prompt=self.system_prompt,
            ):
                result_chunks.append(chunk)
                if not output_path:  # Only stream to console if not saving to file
                    from rich.text import Text
                    text = Text(chunk, style="green")
                    self.console.print(text, end="")
                    import sys
                    sys.stdout.flush()

            if not output_path:
                self.console.print("\n")

            result = "".join(result_chunks)

            if output_path:
                self._write_file_content(output_path, result)
                self.console.print(f"[bold green]Chapter saved to:[/bold green] {output_path}")

            return result

        except KeyboardInterrupt:
            self.console.print(f"\n[yellow]Processing interrupted for {chapter_title}[/yellow]")
            raise FileProcessingError(f"Processing interrupted for {chapter_title}")

    def _process_chunked_content_streaming(
        self,
        content: str,
        model: str,
        output_path: Path | None = None,
        chapter_title: str = "Chapter",
        debug_chunking: bool = False
    ) -> str:
        """Process content in chunks with streaming."""
        chunks = self.chunker.chunk_text(content)

        if not chunks:
            return content

        self.console.print(f"[blue]Chunking {chapter_title}: {len(chunks)} chunks ({len(content.split())} words)[/blue]")

        # Show debug information about chunk boundaries
        if debug_chunking:
            self.console.print(f"\n[yellow]Chunk Boundaries Debug Information for {chapter_title}:[/yellow]")
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
                    system_prompt=self.system_prompt,
                ):
                    result_parts.append(stream_chunk)
                    if not output_path:  # Only stream to console if not saving to file
                        from rich.text import Text
                        text = Text(stream_chunk, style="green")
                        self.console.print(text, end="")
                        import sys
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
                self._write_file_content(output_path, result)
                self.console.print(f"[bold green]Chapter saved to:[/bold green] {output_path}")

            return result

        except KeyboardInterrupt:
            self.console.print(f"\n[yellow]Processing interrupted for {chapter_title}[/yellow]")
            raise FileProcessingError(f"Processing interrupted for {chapter_title}")

    def _process_without_streaming(
        self,
        content: str,
        model: str,
        output_path: Path | None = None,
        chapter_title: str = "Chapter"
    ) -> str:
        """Process content without streaming."""
        # Check if content needs chunking
        word_count = len(content.split())
        debug_chunking = getattr(self, '_debug_chunking', False)
        if word_count <= self.chunker.target_word_count:
            return self._process_single_chunk_no_streaming(content, model, output_path, chapter_title)
        else:
            return self._process_chunked_content_no_streaming(content, model, output_path, chapter_title, debug_chunking)

    def _process_single_chunk_no_streaming(
        self,
        content: str,
        model: str,
        output_path: Path | None = None,
        chapter_title: str = "Chapter"
    ) -> str:
        """Process content as a single chunk without streaming."""
        with self.console.status(f"[dim]Processing {chapter_title}...[/dim]"):
            result = self.client.generate(
                model=model,
                prompt=f"[BEGINNING OF CONTENT]\n{content}\n[END OF CONTENT]",
                system_prompt=self.system_prompt,
            )

        if output_path:
            self._write_file_content(output_path, result)
            self.console.print(f"[bold green]Chapter saved to:[/bold green] {output_path}")

        return result

    def _process_chunked_content_no_streaming(
        self,
        content: str,
        model: str,
        output_path: Path | None = None,
        chapter_title: str = "Chapter",
        debug_chunking: bool = False
    ) -> str:
        """Process content in chunks without streaming."""
        chunks = self.chunker.chunk_text(content)

        if not chunks:
            return content

        self.console.print(f"[blue]Chunking {chapter_title}: {len(chunks)} chunks ({len(content.split())} words)[/blue]")

        # Show debug information about chunk boundaries
        if debug_chunking:
            self.console.print(f"\n[yellow]Chunk Boundaries Debug Information for {chapter_title}:[/yellow]")
            boundaries = self.chunker.get_chunk_boundaries_debug(content)
            for boundary in boundaries:
                self.console.print(f"[dim]{boundary}[/dim]")
            self.console.print()

        processed_chunks = []

        try:
            with Progress() as progress:
                task = progress.add_task(f"Processing {chapter_title}", total=len(chunks))

                for chunk in chunks:
                    progress.update(task, description=f"Processing chunk {chunk.chunk_number}/{chunk.total_chunks}")

                    chunk_result = self.client.generate(
                        model=model,
                        prompt=f"[BEGINNING OF CONTENT]\n{chunk.content}\n[END OF CONTENT]",
                        system_prompt=self.system_prompt,
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
                self._write_file_content(output_path, result)
                self.console.print(f"[bold green]Chapter saved to:[/bold green] {output_path}")

            return result

        except KeyboardInterrupt:
            self.console.print(f"\n[yellow]Processing interrupted for {chapter_title}[/yellow]")
            raise FileProcessingError(f"Processing interrupted for {chapter_title}")

