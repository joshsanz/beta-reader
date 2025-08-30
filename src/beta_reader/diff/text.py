"""Text diff utilities."""

import difflib
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..llm.exceptions import FileProcessingError


class TextDiffer:
    """Generate and display diffs between text files."""

    def __init__(self) -> None:
        """Initialize text differ."""
        self.console = Console()

    def diff_files(
        self,
        original_path: Path,
        edited_path: Path,
        format: str = "unified"
    ) -> str:
        """Generate diff between two files.

        Args:
            original_path: Path to the original file.
            edited_path: Path to the edited file.
            format: Diff format - "unified" or "split".

        Returns:
            Diff output as string.

        Raises:
            FileProcessingError: If files cannot be read or diff format is invalid.
        """
        if format not in ("unified", "split"):
            raise FileProcessingError(f"Invalid diff format: {format}. Must be 'unified' or 'split'.")

        try:
            original_content = self._read_file_lines(original_path)
            edited_content = self._read_file_lines(edited_path)

            if format == "unified":
                return self._generate_unified_diff(
                    original_content,
                    edited_content,
                    str(original_path),
                    str(edited_path)
                )
            else:  # split format
                return self._generate_split_diff(
                    original_content,
                    edited_content,
                    str(original_path),
                    str(edited_path)
                )

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Failed to generate diff: {e}") from e

    def display_diff(
        self,
        original_path: Path,
        edited_path: Path,
        format: str = "unified"
    ) -> None:
        """Display diff with rich formatting.

        Args:
            original_path: Path to the original file.
            edited_path: Path to the edited file.
            format: Diff format - "unified" or "split".

        Raises:
            FileProcessingError: If files cannot be read or diff format is invalid.
        """
        if format not in ("unified", "split"):
            raise FileProcessingError(f"Invalid diff format: {format}. Must be 'unified' or 'split'.")

        try:
            original_content = self._read_file_lines(original_path)
            edited_content = self._read_file_lines(edited_path)

            # Check if files are identical
            if original_content == edited_content:
                self.console.print("[green]✓ Files are identical - no differences found.[/green]")
                return

            # Display header
            self.console.print(f"\n[bold blue]Diff between files[/bold blue] ({format} format)")
            self.console.print(f"[dim]Original:[/dim] {original_path}")
            self.console.print(f"[dim]Edited:[/dim] {edited_path}")
            self.console.print()

            if format == "unified":
                self._display_unified_diff(
                    original_content,
                    edited_content,
                    str(original_path),
                    str(edited_path)
                )
            else:  # split format
                self._display_split_diff(
                    original_content,
                    edited_content,
                    str(original_path),
                    str(edited_path)
                )

        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Failed to display diff: {e}") from e

    def _read_file_lines(self, file_path: Path) -> list[str]:
        """Read file content as list of lines.

        Args:
            file_path: Path to the file to read.

        Returns:
            List of lines from the file.

        Raises:
            FileProcessingError: If file cannot be read.
        """
        if not file_path.exists():
            raise FileProcessingError(f"File not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                return f.readlines()
        except Exception as e:
            raise FileProcessingError(f"Failed to read file {file_path}: {e}") from e

    def _generate_unified_diff(
        self,
        original_lines: list[str],
        edited_lines: list[str],
        original_name: str,
        edited_name: str
    ) -> str:
        """Generate unified diff format.

        Args:
            original_lines: Lines from original file.
            edited_lines: Lines from edited file.
            original_name: Name of original file.
            edited_name: Name of edited file.

        Returns:
            Unified diff as string.
        """
        diff_lines = list(difflib.unified_diff(
            original_lines,
            edited_lines,
            fromfile=original_name,
            tofile=edited_name,
            lineterm=""
        ))
        return "\n".join(diff_lines)

    def _generate_split_diff(
        self,
        original_lines: list[str],
        edited_lines: list[str],
        original_name: str,
        edited_name: str
    ) -> str:
        """Generate split diff format.

        Args:
            original_lines: Lines from original file.
            edited_lines: Lines from edited file.
            original_name: Name of original file.
            edited_name: Name of edited file.

        Returns:
            Split diff as string.
        """
        # Use context diff for split format
        diff_lines = list(difflib.context_diff(
            original_lines,
            edited_lines,
            fromfile=original_name,
            tofile=edited_name,
            lineterm=""
        ))
        return "\n".join(diff_lines)

    def _display_unified_diff(
        self,
        original_lines: list[str],
        edited_lines: list[str],
        original_name: str,
        edited_name: str
    ) -> None:
        """Display unified diff with rich formatting.

        Args:
            original_lines: Lines from original file.
            edited_lines: Lines from edited file.
            original_name: Name of original file.
            edited_name: Name of edited file.
        """
        diff_lines = list(difflib.unified_diff(
            original_lines,
            edited_lines,
            fromfile=original_name,
            tofile=edited_name,
            lineterm=""
        ))

        for line in diff_lines:
            if line.startswith("@@"):
                self.console.print(Text(line, style="cyan bold"))
            elif line.startswith("+++") or line.startswith("---"):
                self.console.print(Text(line, style="white bold"))
            elif line.startswith("+"):
                self.console.print(Text(line, style="green"))
            elif line.startswith("-"):
                self.console.print(Text(line, style="red"))
            else:
                self.console.print(line)

    def _display_split_diff(
        self,
        original_lines: list[str],
        edited_lines: list[str],
        original_name: str,
        edited_name: str
    ) -> None:
        """Display split diff with rich formatting using side-by-side view.

        Args:
            original_lines: Lines from original file.
            edited_lines: Lines from edited file.
            original_name: Name of original file.
            edited_name: Name of edited file.
        """
        # Create a table for side-by-side comparison
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Original", style="dim", width=50)
        table.add_column("Edited", style="dim", width=50)

        # Get diff ops using SequenceMatcher
        matcher = difflib.SequenceMatcher(None, original_lines, edited_lines)

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                # Show equal lines in dim style
                for i in range(i1, i2):
                    orig_line = original_lines[i].rstrip('\n')
                    edit_line = edited_lines[j1 + (i - i1)].rstrip('\n')
                    table.add_row(
                        Text(orig_line, style="dim"),
                        Text(edit_line, style="dim")
                    )
            elif op == "delete":
                # Show deleted lines in red on left, empty on right
                for i in range(i1, i2):
                    orig_line = original_lines[i].rstrip('\n')
                    table.add_row(
                        Text(f"- {orig_line}", style="red"),
                        Text("", style="dim")
                    )
            elif op == "insert":
                # Show inserted lines in green on right, empty on left
                for j in range(j1, j2):
                    edit_line = edited_lines[j].rstrip('\n')
                    table.add_row(
                        Text("", style="dim"),
                        Text(f"+ {edit_line}", style="green")
                    )
            elif op == "replace":
                # Show replaced lines - old on left (red), new on right (green)
                max_lines = max(i2 - i1, j2 - j1)
                for k in range(max_lines):
                    orig_text = ""
                    edit_text = ""

                    if k < (i2 - i1):
                        orig_line = original_lines[i1 + k].rstrip('\n')
                        orig_text = f"- {orig_line}"

                    if k < (j2 - j1):
                        edit_line = edited_lines[j1 + k].rstrip('\n')
                        edit_text = f"+ {edit_line}"

                    table.add_row(
                        Text(orig_text, style="red" if orig_text else "dim"),
                        Text(edit_text, style="green" if edit_text else "dim")
                    )

        self.console.print(table)
