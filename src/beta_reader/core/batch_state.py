"""Batch processing state management for resume functionality."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..llm.exceptions import FileProcessingError


@dataclass
class ChapterState:
    """State of a single chapter in batch processing."""

    chapter_index: int
    chapter_title: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    start_time: float | None = None
    end_time: float | None = None
    output_file: str | None = None
    error_message: str | None = None
    word_count: int | None = None
    processing_time: float | None = None


@dataclass
class BatchState:
    """State of batch processing operation."""

    batch_id: str
    input_file: str
    output_directory: str | None
    model: str
    start_time: float
    last_update: float
    status: str  # 'running', 'paused', 'completed', 'failed'
    current_chapter: int
    total_chapters: int
    chapters: list[ChapterState]
    completed_chapters: int = 0
    failed_chapters: int = 0

    def update_progress(self) -> None:
        """Update progress counters."""
        self.completed_chapters = sum(1 for c in self.chapters if c.status == 'completed')
        self.failed_chapters = sum(1 for c in self.chapters if c.status == 'failed')
        self.last_update = time.time()


class BatchStateManager:
    """Manages batch processing state for resume functionality."""

    def __init__(self, state_dir: Path | None = None) -> None:
        """Initialize batch state manager.

        Args:
            state_dir: Directory to store state files. Defaults to .batch_states/
        """
        if state_dir is None:
            state_dir = Path.cwd() / ".batch_states"

        self.state_dir = state_dir
        self.state_dir.mkdir(exist_ok=True)

    def generate_batch_id(self, input_file: Path, model: str) -> str:
        """Generate a unique batch ID.

        Args:
            input_file: Input file being processed.
            model: Model being used.

        Returns:
            Unique batch ID.
        """
        timestamp = int(time.time())
        safe_filename = input_file.stem.replace(" ", "_").replace("-", "_")
        safe_model = model.replace(":", "_").replace("/", "_")
        return f"{safe_filename}_{safe_model}_{timestamp}"

    def get_short_hash(self, batch_id: str, length: int = 8) -> str:
        """Generate a short hash for a batch ID.

        Args:
            batch_id: Full batch ID.
            length: Length of short hash.

        Returns:
            Short hash string.
        """
        return hashlib.sha256(batch_id.encode()).hexdigest()[:length]

    def resolve_short_hash(self, short_hash: str) -> str:
        """Resolve a short hash to a full batch ID.

        Args:
            short_hash: Short hash to resolve.

        Returns:
            Full batch ID.

        Raises:
            FileProcessingError: If hash cannot be resolved or is ambiguous.
        """
        # First check if it's already a full batch ID
        if self._is_full_batch_id(short_hash):
            return short_hash

        # Find all matching batch IDs
        matches = []
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, encoding='utf-8') as f:
                    data = json.load(f)
                batch_id = data['batch_id']
                if self.get_short_hash(batch_id).startswith(short_hash.lower()):
                    matches.append(batch_id)
            except Exception:
                # Skip corrupted state files
                continue

        if not matches:
            raise FileProcessingError(f"No batch found with hash '{short_hash}'")
        elif len(matches) > 1:
            raise FileProcessingError(f"Hash '{short_hash}' is ambiguous. Matches: {matches}")

        return matches[0]

    def _is_full_batch_id(self, batch_id: str) -> bool:
        """Check if string looks like a full batch ID.

        Args:
            batch_id: String to check.

        Returns:
            True if it looks like a full batch ID.
        """
        # Full batch IDs are long and contain underscores and timestamp
        return len(batch_id) > 20 and "_" in batch_id and batch_id.split("_")[-1].isdigit()

    def create_batch_state(
        self,
        batch_id: str,
        input_file: Path,
        chapters: list[tuple[str, str]],
        model: str,
        output_directory: Path | None = None,
    ) -> BatchState:
        """Create a new batch state.

        Args:
            batch_id: Unique batch identifier.
            input_file: Input file being processed.
            chapters: List of (title, content) tuples.
            model: Model being used.
            output_directory: Optional output directory.

        Returns:
            New batch state object.
        """
        chapter_states = [
            ChapterState(
                chapter_index=i,
                chapter_title=title,
                status='pending'
            )
            for i, (title, _) in enumerate(chapters)
        ]

        state = BatchState(
            batch_id=batch_id,
            input_file=str(input_file),
            output_directory=str(output_directory) if output_directory else None,
            model=model,
            start_time=time.time(),
            last_update=time.time(),
            status='running',
            current_chapter=0,
            total_chapters=len(chapters),
            chapters=chapter_states,
        )

        self.save_batch_state(state)
        return state

    def save_batch_state(self, state: BatchState) -> None:
        """Save batch state to disk.

        Args:
            state: Batch state to save.

        Raises:
            FileProcessingError: If state cannot be saved.
        """
        try:
            state_file = self.state_dir / f"{state.batch_id}.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(state), f, indent=2)
        except Exception as e:
            raise FileProcessingError(f"Failed to save batch state: {e}") from e

    def load_batch_state(self, batch_id: str) -> BatchState:
        """Load batch state from disk.

        Args:
            batch_id: Batch identifier.

        Returns:
            Loaded batch state.

        Raises:
            FileProcessingError: If state cannot be loaded.
        """
        try:
            state_file = self.state_dir / f"{batch_id}.json"
            if not state_file.exists():
                raise FileProcessingError(f"Batch state not found: {batch_id}")

            with open(state_file, encoding='utf-8') as f:
                data = json.load(f)

            # Convert chapter dictionaries back to ChapterState objects
            chapters = [ChapterState(**chapter_data) for chapter_data in data['chapters']]
            data['chapters'] = chapters

            return BatchState(**data)
        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise
            raise FileProcessingError(f"Failed to load batch state: {e}") from e

    def list_batch_states(self) -> list[dict[str, Any]]:
        """List all available batch states.

        Returns:
            List of batch state summaries.
        """
        states = []

        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, encoding='utf-8') as f:
                    data = json.load(f)

                states.append({
                    'batch_id': data['batch_id'],
                    'short_hash': self.get_short_hash(data['batch_id']),
                    'input_file': data['input_file'],
                    'model': data['model'],
                    'status': data['status'],
                    'progress': f"{data.get('completed_chapters', 0)}/{data['total_chapters']}",
                    'start_time': data['start_time'],
                    'last_update': data['last_update'],
                })
            except Exception:
                # Skip corrupted state files
                continue

        # Sort by last update time (most recent first)
        return sorted(states, key=lambda x: x['last_update'], reverse=True)

    def update_chapter_status(
        self,
        state: BatchState,
        chapter_index: int,
        status: str,
        output_file: str | None = None,
        error_message: str | None = None,
        word_count: int | None = None,
        processing_time: float | None = None,
    ) -> None:
        """Update status of a specific chapter.

        Args:
            state: Batch state to update.
            chapter_index: Index of chapter to update.
            status: New status for the chapter.
            output_file: Optional path to output file.
            error_message: Optional error message.
            word_count: Optional word count of processed text.
            processing_time: Optional processing time in seconds.
        """
        if chapter_index >= len(state.chapters):
            raise FileProcessingError(f"Invalid chapter index: {chapter_index}")

        chapter = state.chapters[chapter_index]
        chapter.status = status

        if status == 'processing':
            chapter.start_time = time.time()
        elif status in ('completed', 'failed'):
            chapter.end_time = time.time()
            if chapter.start_time:
                chapter.processing_time = chapter.end_time - chapter.start_time

        if output_file:
            chapter.output_file = output_file
        if error_message:
            chapter.error_message = error_message
        if word_count is not None:
            chapter.word_count = word_count
        if processing_time is not None:
            chapter.processing_time = processing_time

        # Update overall state
        state.update_progress()

        # Update current chapter to next pending chapter
        if status in ('completed', 'failed'):
            next_pending = next(
                (i for i, c in enumerate(state.chapters) if c.status == 'pending'),
                None
            )
            if next_pending is not None:
                state.current_chapter = next_pending
            else:
                # All chapters processed
                if state.failed_chapters == 0:
                    state.status = 'completed'
                else:
                    state.status = 'failed'

        self.save_batch_state(state)

    def mark_batch_failed(self, state: BatchState, error_message: str) -> None:
        """Mark entire batch as failed.

        Args:
            state: Batch state to update.
            error_message: Error message describing the failure.
        """
        state.status = 'failed'

        # Mark currently processing chapter as failed if any
        current_chapter = state.chapters[state.current_chapter]
        if current_chapter.status == 'processing':
            current_chapter.status = 'failed'
            current_chapter.error_message = error_message
            current_chapter.end_time = time.time()

        state.update_progress()
        self.save_batch_state(state)

    def cleanup_old_states(self, max_age_days: int = 7) -> int:
        """Clean up old batch state files.

        Args:
            max_age_days: Maximum age in days for state files.

        Returns:
            Number of files cleaned up.
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_up = 0

        for state_file in self.state_dir.glob("*.json"):
            try:
                if state_file.stat().st_mtime < cutoff_time:
                    state_file.unlink()
                    cleaned_up += 1
            except Exception:
                # Skip files we can't process
                continue

        return cleaned_up

    def delete_batch_state(self, batch_id: str) -> bool:
        """Delete a batch state file.

        Args:
            batch_id: Batch identifier.

        Returns:
            True if deleted successfully, False if not found.
        """
        try:
            state_file = self.state_dir / f"{batch_id}.json"
            if state_file.exists():
                state_file.unlink()
                return True
            return False
        except Exception:
            return False

    def get_resumable_batches(self) -> list[dict[str, Any]]:
        """Get list of batches that can be resumed.

        Returns:
            List of resumable batch summaries.
        """
        states = self.list_batch_states()
        return [
            state for state in states
            if state['status'] in ('running', 'paused') and
            state['progress'] != f"{state.get('completed_chapters', 0)}/0"
        ]
