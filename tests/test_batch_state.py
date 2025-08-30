"""Tests for batch processing state management."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from beta_reader.core.batch_state import (
    BatchState,
    BatchStateManager,
    ChapterState,
)
from beta_reader.llm.exceptions import FileProcessingError


class TestChapterState:
    """Test ChapterState dataclass."""

    def test_chapter_state_creation(self):
        """Test ChapterState creation with defaults."""
        state = ChapterState(
            chapter_index=0,
            chapter_title="Chapter 1",
            status="pending"
        )
        
        assert state.chapter_index == 0
        assert state.chapter_title == "Chapter 1"
        assert state.status == "pending"
        assert state.start_time is None
        assert state.end_time is None
        assert state.output_file is None
        assert state.error_message is None
        assert state.word_count is None
        assert state.processing_time is None

    def test_chapter_state_with_all_fields(self):
        """Test ChapterState creation with all fields."""
        state = ChapterState(
            chapter_index=1,
            chapter_title="Chapter 2", 
            status="completed",
            start_time=1000.0,
            end_time=1100.0,
            output_file="chapter2.txt",
            error_message=None,
            word_count=500,
            processing_time=100.0
        )
        
        assert state.chapter_index == 1
        assert state.status == "completed"
        assert state.processing_time == 100.0
        assert state.word_count == 500


class TestBatchState:
    """Test BatchState dataclass."""

    def test_batch_state_creation(self):
        """Test BatchState creation."""
        chapters = [
            ChapterState(0, "Ch 1", "pending"),
            ChapterState(1, "Ch 2", "pending")
        ]
        
        state = BatchState(
            batch_id="test_batch",
            input_file="test.txt",
            output_directory="/tmp/output",
            model="llama3.1:8b", 
            start_time=time.time(),
            last_update=time.time(),
            status="running",
            current_chapter=0,
            total_chapters=2,
            chapters=chapters
        )
        
        assert state.batch_id == "test_batch"
        assert state.total_chapters == 2
        assert len(state.chapters) == 2
        assert state.completed_chapters == 0
        assert state.failed_chapters == 0

    def test_update_progress(self):
        """Test progress update calculation."""
        chapters = [
            ChapterState(0, "Ch 1", "completed"),
            ChapterState(1, "Ch 2", "failed"),
            ChapterState(2, "Ch 3", "pending")
        ]
        
        state = BatchState(
            batch_id="test",
            input_file="test.txt",
            output_directory=None,
            model="test_model",
            start_time=1000.0,
            last_update=1000.0,
            status="running",
            current_chapter=2,
            total_chapters=3,
            chapters=chapters
        )
        
        with patch('time.time', return_value=2000.0):
            state.update_progress()
        
        assert state.completed_chapters == 1
        assert state.failed_chapters == 1
        assert state.last_update == 2000.0


class TestBatchStateManager:
    """Test BatchStateManager class."""

    def test_init_default_directory(self):
        """Test initialization with default directory."""
        manager = BatchStateManager()
        expected_path = Path.cwd() / ".batch_states"
        assert manager.state_dir == expected_path
        assert manager.state_dir.exists()

    def test_init_custom_directory(self):
        """Test initialization with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "custom_states"
            manager = BatchStateManager(custom_dir)
            assert manager.state_dir == custom_dir
            assert manager.state_dir.exists()

    def test_generate_batch_id(self):
        """Test batch ID generation."""
        manager = BatchStateManager()
        input_file = Path("test file-name.txt")
        model = "llama3.1:8b"
        
        with patch('time.time', return_value=1234567890):
            batch_id = manager.generate_batch_id(input_file, model)
        
        assert "test_file_name" in batch_id
        assert "llama3.1_8b" in batch_id
        assert "1234567890" in batch_id

    def test_create_batch_state(self):
        """Test batch state creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            chapters = [("Chapter 1", "content1"), ("Chapter 2", "content2")]
            input_file = Path("test.txt")
            output_dir = Path("/tmp/output")
            
            with patch('time.time', return_value=1000.0):
                state = manager.create_batch_state(
                    "test_batch",
                    input_file,
                    chapters,
                    "test_model",
                    output_dir
                )
            
            assert state.batch_id == "test_batch"
            assert state.input_file == str(input_file)
            assert state.output_directory == str(output_dir)
            assert state.total_chapters == 2
            assert len(state.chapters) == 2
            assert state.chapters[0].chapter_title == "Chapter 1"
            assert state.chapters[1].chapter_title == "Chapter 2"
            
            # Verify file was saved
            state_file = manager.state_dir / "test_batch.json"
            assert state_file.exists()

    def test_save_and_load_batch_state(self):
        """Test saving and loading batch state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            chapters = [ChapterState(0, "Ch 1", "pending")]
            original_state = BatchState(
                batch_id="test_save_load",
                input_file="test.txt",
                output_directory=None,
                model="test_model",
                start_time=1000.0,
                last_update=1100.0, 
                status="running",
                current_chapter=0,
                total_chapters=1,
                chapters=chapters
            )
            
            # Save state
            manager.save_batch_state(original_state)
            
            # Load state
            loaded_state = manager.load_batch_state("test_save_load")
            
            assert loaded_state.batch_id == original_state.batch_id
            assert loaded_state.input_file == original_state.input_file
            assert loaded_state.model == original_state.model
            assert loaded_state.total_chapters == original_state.total_chapters
            assert len(loaded_state.chapters) == len(original_state.chapters)
            assert loaded_state.chapters[0].chapter_title == original_state.chapters[0].chapter_title

    def test_load_nonexistent_batch_state(self):
        """Test loading non-existent batch state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            with pytest.raises(FileProcessingError, match="Batch state not found"):
                manager.load_batch_state("nonexistent")

    def test_save_batch_state_error(self):
        """Test save batch state with file system error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            # Create a state that will cause JSON serialization issues
            state = BatchState(
                batch_id="test",
                input_file="test.txt",
                output_directory=None,
                model="test_model",
                start_time=1000.0,
                last_update=1000.0,
                status="running", 
                current_chapter=0,
                total_chapters=1,
                chapters=[]
            )
            
            # Make directory read-only to cause save error
            temp_path = Path(temp_dir)
            temp_path.chmod(0o444)  # Read-only
            
            try:
                with pytest.raises(FileProcessingError, match="Failed to save batch state"):
                    manager.save_batch_state(state)
            finally:
                # Restore permissions for cleanup
                temp_path.chmod(0o755)

    def test_list_batch_states(self):
        """Test listing batch states."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            # Create test states
            for i in range(3):
                state = BatchState(
                    batch_id=f"batch_{i}",
                    input_file=f"test_{i}.txt",
                    output_directory=None,
                    model="test_model",
                    start_time=1000.0 + i,
                    last_update=1100.0 + i,
                    status="running",
                    current_chapter=0,
                    total_chapters=2,
                    chapters=[],
                    completed_chapters=i,
                    failed_chapters=0
                )
                manager.save_batch_state(state)
            
            states = manager.list_batch_states()
            
            assert len(states) == 3
            # Should be sorted by last_update (most recent first)
            assert states[0]['batch_id'] == "batch_2"
            assert states[1]['batch_id'] == "batch_1" 
            assert states[2]['batch_id'] == "batch_0"
            
            # Check state info
            assert states[0]['progress'] == "2/2"
            assert states[0]['model'] == "test_model"

    def test_list_batch_states_with_corrupted_file(self):
        """Test listing batch states with corrupted file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            # Create valid state
            valid_state = BatchState(
                batch_id="valid",
                input_file="test.txt",
                output_directory=None,
                model="test_model",
                start_time=1000.0,
                last_update=1000.0,
                status="running",
                current_chapter=0,
                total_chapters=1,
                chapters=[]
            )
            manager.save_batch_state(valid_state)
            
            # Create corrupted state file
            corrupted_file = manager.state_dir / "corrupted.json"
            with open(corrupted_file, 'w') as f:
                f.write("invalid json content {")
            
            states = manager.list_batch_states()
            
            # Should only return valid state, skip corrupted
            assert len(states) == 1
            assert states[0]['batch_id'] == "valid"

    def test_update_chapter_status(self):
        """Test updating chapter status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            chapters = [
                ChapterState(0, "Ch 1", "pending"),
                ChapterState(1, "Ch 2", "pending")
            ]
            
            state = BatchState(
                batch_id="test_update",
                input_file="test.txt",
                output_directory=None,
                model="test_model",
                start_time=1000.0,
                last_update=1000.0,
                status="running",
                current_chapter=0,
                total_chapters=2,
                chapters=chapters
            )
            
            # Update chapter to processing
            with patch('time.time', return_value=2000.0):
                manager.update_chapter_status(
                    state, 0, "processing",
                    output_file="ch1_output.txt",
                    word_count=500
                )
            
            assert state.chapters[0].status == "processing"
            assert state.chapters[0].start_time == 2000.0
            assert state.chapters[0].output_file == "ch1_output.txt"
            assert state.chapters[0].word_count == 500
            
            # Update chapter to completed
            with patch('time.time', return_value=2100.0):
                manager.update_chapter_status(
                    state, 0, "completed",
                    processing_time=100.0
                )
            
            assert state.chapters[0].status == "completed"
            assert state.chapters[0].end_time == 2100.0
            assert state.chapters[0].processing_time == 100.0
            assert state.current_chapter == 1  # Should advance to next pending
            assert state.completed_chapters == 1

    def test_update_chapter_status_invalid_index(self):
        """Test updating chapter status with invalid index."""
        manager = BatchStateManager()
        
        state = BatchState(
            batch_id="test",
            input_file="test.txt", 
            output_directory=None,
            model="test_model",
            start_time=1000.0,
            last_update=1000.0,
            status="running",
            current_chapter=0,
            total_chapters=1,
            chapters=[ChapterState(0, "Ch 1", "pending")]
        )
        
        with pytest.raises(FileProcessingError, match="Invalid chapter index"):
            manager.update_chapter_status(state, 5, "completed")

    def test_update_chapter_status_completes_batch(self):
        """Test that updating last chapter completes batch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            state = BatchState(
                batch_id="test_complete",
                input_file="test.txt",
                output_directory=None,
                model="test_model", 
                start_time=1000.0,
                last_update=1000.0,
                status="running",
                current_chapter=0,
                total_chapters=1,
                chapters=[ChapterState(0, "Ch 1", "pending")]
            )
            
            manager.update_chapter_status(state, 0, "completed")
            
            assert state.status == "completed"
            assert state.completed_chapters == 1
            assert state.failed_chapters == 0

    def test_update_chapter_status_with_failures(self):
        """Test batch status when there are failed chapters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            chapters = [
                ChapterState(0, "Ch 1", "completed"),
                ChapterState(1, "Ch 2", "pending")
            ]
            
            state = BatchState(
                batch_id="test_failure",
                input_file="test.txt",
                output_directory=None,
                model="test_model",
                start_time=1000.0,
                last_update=1000.0,
                status="running",
                current_chapter=1,
                total_chapters=2,
                chapters=chapters,
                completed_chapters=1
            )
            
            manager.update_chapter_status(
                state, 1, "failed",
                error_message="Processing failed"
            )
            
            assert state.status == "failed"  # Batch failed due to failed chapter
            assert state.completed_chapters == 1
            assert state.failed_chapters == 1

    def test_mark_batch_failed(self):
        """Test marking entire batch as failed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            state = BatchState(
                batch_id="test_fail",
                input_file="test.txt",
                output_directory=None,
                model="test_model",
                start_time=1000.0,
                last_update=1000.0,
                status="running",
                current_chapter=0,
                total_chapters=2,
                chapters=[
                    ChapterState(0, "Ch 1", "processing"),
                    ChapterState(1, "Ch 2", "pending")
                ]
            )
            
            with patch('time.time', return_value=2000.0):
                manager.mark_batch_failed(state, "System error occurred")
            
            assert state.status == "failed"
            assert state.chapters[0].status == "failed"
            assert state.chapters[0].error_message == "System error occurred"
            assert state.chapters[0].end_time == 2000.0
            assert state.chapters[1].status == "pending"  # Unchanged

    def test_cleanup_old_states(self):
        """Test cleanup of old state files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            # Create old and new state files
            old_file = manager.state_dir / "old_batch.json"
            new_file = manager.state_dir / "new_batch.json"
            
            old_file.write_text("{}")
            new_file.write_text("{}")
            
            # Set old file to be old (8 days ago)
            import os
            old_time = time.time() - (8 * 24 * 60 * 60)
            os.utime(old_file, (old_time, old_time))
            
            cleaned_up = manager.cleanup_old_states(max_age_days=7)
            
            assert cleaned_up == 1
            assert not old_file.exists()
            assert new_file.exists()

    def test_delete_batch_state(self):
        """Test deleting batch state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            # Create state
            state_file = manager.state_dir / "to_delete.json"
            state_file.write_text("{}")
            
            # Delete existing state
            result = manager.delete_batch_state("to_delete")
            assert result is True
            assert not state_file.exists()
            
            # Try to delete non-existent state
            result = manager.delete_batch_state("nonexistent")
            assert result is False

    def test_get_resumable_batches(self):
        """Test getting resumable batches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            # Create various states
            states_data = [
                {"batch_id": "running", "status": "running", "completed_chapters": 1, "total_chapters": 3},
                {"batch_id": "paused", "status": "paused", "completed_chapters": 2, "total_chapters": 5},
                {"batch_id": "completed", "status": "completed", "completed_chapters": 3, "total_chapters": 3},
                {"batch_id": "failed", "status": "failed", "completed_chapters": 1, "total_chapters": 2},
            ]
            
            for data in states_data:
                state_file = manager.state_dir / f"{data['batch_id']}.json"
                state_data = {
                    "batch_id": data["batch_id"],
                    "input_file": "test.txt",
                    "model": "test_model",
                    "start_time": 1000.0,
                    "last_update": 1000.0,
                    "status": data["status"],
                    "current_chapter": 0,
                    "total_chapters": data["total_chapters"],
                    "chapters": [],
                    "completed_chapters": data["completed_chapters"],
                    "failed_chapters": 0
                }
                state_file.write_text(json.dumps(state_data))
            
            resumable = manager.get_resumable_batches()
            
            # Should only include running and paused batches
            assert len(resumable) == 2
            batch_ids = [batch['batch_id'] for batch in resumable]
            assert "running" in batch_ids
            assert "paused" in batch_ids
            assert "completed" not in batch_ids
            assert "failed" not in batch_ids

    def test_load_corrupted_state_file(self):
        """Test loading corrupted state file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BatchStateManager(Path(temp_dir))
            
            # Create corrupted file
            corrupted_file = manager.state_dir / "corrupted.json"
            with open(corrupted_file, 'w') as f:
                f.write("invalid json {")
            
            with pytest.raises(FileProcessingError, match="Failed to load batch state"):
                manager.load_batch_state("corrupted")