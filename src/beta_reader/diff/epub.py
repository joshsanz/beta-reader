"""Epub diff utilities."""

from pathlib import Path

import ebooklib
from ebooklib import epub
from rich.console import Console
from rich.table import Table

from ..llm.exceptions import FileProcessingError
from .text import TextDiffer


class EpubDiffer:
    """Generate and display diffs between epub files."""

    def __init__(self) -> None:
        """Initialize epub differ."""
        self.console = Console()
        self.text_differ = TextDiffer()

    def diff_files(
        self,
        original_path: Path,
        edited_path: Path,
        format: str = "unified"
    ) -> str:
        """Generate diff between two epub files.

        Args:
            original_path: Path to the original epub file.
            edited_path: Path to the edited epub file.
            format: Diff format - "unified" or "split".

        Returns:
            Combined diff output as string.

        Raises:
            FileProcessingError: If files cannot be read or diff format is invalid.
        """
        if format not in ("unified", "split"):
            raise FileProcessingError(f"Invalid diff format: {format}. Must be 'unified' or 'split'.")

        try:
            # Load both epub files
            original_book = epub.read_epub(str(original_path))
            edited_book = epub.read_epub(str(edited_path))

            # Extract chapters from both books
            original_chapters = self._extract_chapters(original_book)
            edited_chapters = self._extract_chapters(edited_book)

            # Generate diff for each chapter
            diff_sections = []

            # Add header
            diff_sections.append(f"EPUB DIFF: {original_path} → {edited_path}")
            diff_sections.append("=" * 80)

            max_chapters = max(len(original_chapters), len(edited_chapters))

            for i in range(max_chapters):
                # Get chapter content or empty if chapter doesn't exist
                orig_title, orig_content = original_chapters[i] if i < len(original_chapters) else ("", "")
                edit_title, edit_content = edited_chapters[i] if i < len(edited_chapters) else ("", "")

                # Skip if both chapters are empty
                if not orig_content and not edit_content:
                    continue

                # Create chapter diff
                chapter_header = f"\nCHAPTER {i+1}: {orig_title or edit_title}"
                diff_sections.append(chapter_header)
                diff_sections.append("-" * len(chapter_header))

                # Create temporary files for text differ
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as orig_temp, \
                     tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as edit_temp:

                    orig_temp.write(orig_content)
                    edit_temp.write(edit_content)
                    orig_temp_path = Path(orig_temp.name)
                    edit_temp_path = Path(edit_temp.name)

                try:
                    chapter_diff = self.text_differ.diff_files(orig_temp_path, edit_temp_path, format)
                    if chapter_diff.strip():  # Only add if there are actual differences
                        diff_sections.append(chapter_diff)
                    else:
                        diff_sections.append("No differences found in this chapter.")
                finally:
                    # Clean up temp files
                    orig_temp_path.unlink(missing_ok=True)
                    edit_temp_path.unlink(missing_ok=True)

            return "\n".join(diff_sections)

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Failed to generate epub diff: {e}") from e

    def display_diff(
        self,
        original_path: Path,
        edited_path: Path,
        format: str = "unified"
    ) -> None:
        """Display diff between epub files with rich formatting.

        Args:
            original_path: Path to the original epub file.
            edited_path: Path to the edited epub file.
            format: Diff format - "unified" or "split".

        Raises:
            FileProcessingError: If files cannot be read or diff format is invalid.
        """
        if format not in ("unified", "split"):
            raise FileProcessingError(f"Invalid diff format: {format}. Must be 'unified' or 'split'.")

        try:
            # Load both epub files
            original_book = epub.read_epub(str(original_path))
            edited_book = epub.read_epub(str(edited_path))

            # Extract chapters from both books
            original_chapters = self._extract_chapters(original_book)
            edited_chapters = self._extract_chapters(edited_book)

            # Check if books are identical
            if self._books_identical(original_chapters, edited_chapters):
                self.console.print("[green]✓ Epub files are identical - no differences found.[/green]")
                return

            # Display header
            self.console.print(f"\n[bold blue]Epub diff between files[/bold blue] ({format} format)")
            self.console.print(f"[dim]Original:[/dim] {original_path}")
            self.console.print(f"[dim]Edited:[/dim] {edited_path}")

            # Display summary table
            self._display_chapter_summary(original_chapters, edited_chapters)

            # Display chapter-by-chapter diffs
            max_chapters = max(len(original_chapters), len(edited_chapters))

            for i in range(max_chapters):
                # Get chapter content or empty if chapter doesn't exist
                orig_title, orig_content = original_chapters[i] if i < len(original_chapters) else ("", "")
                edit_title, edit_content = edited_chapters[i] if i < len(edited_chapters) else ("", "")

                # Skip if both chapters are empty
                if not orig_content and not edit_content:
                    continue

                # Check if this chapter has differences
                if orig_content == edit_content:
                    continue  # Skip identical chapters

                # Display chapter header
                chapter_title = orig_title or edit_title
                self.console.print(f"\n[bold yellow]Chapter {i+1}: {chapter_title}[/bold yellow]")

                # Create temporary files for text differ
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as orig_temp, \
                     tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as edit_temp:

                    orig_temp.write(orig_content)
                    edit_temp.write(edit_content)
                    orig_temp_path = Path(orig_temp.name)
                    edit_temp_path = Path(edit_temp.name)

                try:
                    # Display diff for this chapter
                    self.text_differ.display_diff(orig_temp_path, edit_temp_path, format)
                finally:
                    # Clean up temp files
                    orig_temp_path.unlink(missing_ok=True)
                    edit_temp_path.unlink(missing_ok=True)

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Failed to display epub diff: {e}") from e

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
        """Extract title from HTML content."""
        import re

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
        """Convert HTML content to plain text while preserving basic formatting."""
        import re

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

    def _books_identical(self, original_chapters: list[tuple[str, str]], edited_chapters: list[tuple[str, str]]) -> bool:
        """Check if two books are identical."""
        if len(original_chapters) != len(edited_chapters):
            return False

        for (_orig_title, orig_content), (_edit_title, edit_content) in zip(original_chapters, edited_chapters, strict=False):
            if orig_content != edit_content:
                return False

        return True

    def _display_chapter_summary(self, original_chapters: list[tuple[str, str]], edited_chapters: list[tuple[str, str]]) -> None:
        """Display a summary table of chapter differences."""
        table = Table(title="Chapter Summary")
        table.add_column("Chapter", style="cyan")
        table.add_column("Original", style="dim")
        table.add_column("Edited", style="dim")
        table.add_column("Status", justify="center")

        max_chapters = max(len(original_chapters), len(edited_chapters))

        for i in range(max_chapters):
            orig_title, orig_content = original_chapters[i] if i < len(original_chapters) else ("", "")
            edit_title, edit_content = edited_chapters[i] if i < len(edited_chapters) else ("", "")

            # Skip if both chapters are empty
            if not orig_content and not edit_content:
                continue

            chapter_num = str(i + 1)
            orig_display = f"{orig_title[:30]}..." if len(orig_title) > 30 else orig_title
            edit_display = f"{edit_title[:30]}..." if len(edit_title) > 30 else edit_title

            if not orig_content:
                status = "[green]+ Added[/green]"
                orig_display = "[dim]—[/dim]"
            elif not edit_content:
                status = "[red]- Removed[/red]"
                edit_display = "[dim]—[/dim]"
            elif orig_content == edit_content:
                status = "[dim]= Same[/dim]"
            else:
                status = "[yellow]~ Changed[/yellow]"

            table.add_row(chapter_num, orig_display, edit_display, status)

        self.console.print("\n")
        self.console.print(table)
        self.console.print()
