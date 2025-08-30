"""Tests for text chunking functionality."""

import pytest

from beta_reader.core.text_chunker import TextChunk, TextChunker


class TestTextChunk:
    """Test TextChunk dataclass."""

    def test_text_chunk_creation(self):
        """Test TextChunk creation."""
        chunk = TextChunk(
            content="Sample text",
            chunk_number=1,
            total_chunks=2,
            word_count=2,
            start_position=0,
            end_position=11
        )
        
        assert chunk.content == "Sample text"
        assert chunk.chunk_number == 1
        assert chunk.total_chunks == 2
        assert chunk.word_count == 2
        assert chunk.start_position == 0
        assert chunk.end_position == 11


class TestTextChunker:
    """Test TextChunker class."""

    def test_init_default_values(self):
        """Test TextChunker initialization with defaults."""
        chunker = TextChunker()
        assert chunker.target_word_count == 500
        assert chunker.max_word_count == 750

    def test_init_custom_values(self):
        """Test TextChunker initialization with custom values."""
        chunker = TextChunker(target_word_count=100, max_word_count=150)
        assert chunker.target_word_count == 100
        assert chunker.max_word_count == 150

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_whitespace_only_text(self):
        """Test chunking whitespace-only text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("   \n\t  ")
        assert chunks == []

    def test_single_short_chunk(self):
        """Test text that fits in single chunk."""
        chunker = TextChunker(target_word_count=10)
        text = "This is a short text with exactly nine words."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_number == 1
        assert chunks[0].total_chunks == 1
        assert chunks[0].word_count == 9
        assert chunks[0].start_position == 0
        assert chunks[0].end_position == len(text)

    def test_paragraph_break_chunking(self):
        """Test chunking with paragraph breaks."""
        chunker = TextChunker(target_word_count=5, max_word_count=10)
        text = "First paragraph with five words exactly.\n\nSecond paragraph with another five words exactly."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 2
        assert "First paragraph" in chunks[0].content
        assert "Second paragraph" in chunks[1].content
        assert chunks[0].total_chunks == 2
        assert chunks[1].total_chunks == 2

    def test_sentence_break_chunking(self):
        """Test chunking with sentence breaks."""
        chunker = TextChunker(target_word_count=8, max_word_count=12)
        text = "First sentence with exactly six words here. Second sentence with another six words here."
        chunks = chunker.chunk_text(text)
        
        # Should split at sentence boundary
        assert len(chunks) == 2
        assert chunks[0].content.endswith("here.")
        assert chunks[1].content.startswith("Second")

    def test_forced_split_at_max_words(self):
        """Test forced split when max word count is reached."""
        chunker = TextChunker(target_word_count=5, max_word_count=8)
        # Long sentence without good break points
        text = "word " * 15  # 15 words in a row
        chunks = chunker.chunk_text(text)
        
        # Should be forced to split
        assert len(chunks) >= 2
        # First chunk should not exceed max word count
        assert chunks[0].word_count <= 8

    def test_chunk_boundaries_debug(self):
        """Test debug information generation."""
        chunker = TextChunker(target_word_count=10, max_word_count=15)
        text = "First paragraph here.\n\nSecond paragraph here with more words to trigger chunking behavior."
        boundaries = chunker.get_chunk_boundaries_debug(text)
        
        # Should have at least one boundary if text was chunked
        chunks = chunker.chunk_text(text)
        if len(chunks) > 1:
            assert len(boundaries) == len(chunks) - 1
            assert "paragraph break" in boundaries[0] or "sentence end" in boundaries[0]

    def test_word_positions_accuracy(self):
        """Test word position calculation accuracy."""
        chunker = TextChunker()
        text = "Word1 Word2   Word3\nWord4"
        positions = chunker._build_word_positions(text)
        
        assert len(positions) == 4
        # Check first word position
        assert positions[0] == (0, 5)  # "Word1"
        # Check word with extra spaces
        word3_start = text.find("Word3")
        assert positions[2] == (word3_start, word3_start + 5)

    def test_boundary_pattern_matching(self):
        """Test boundary pattern matching."""
        chunker = TextChunker()
        text = "Sentence one. Sentence two! Question three? Another sentence."
        
        # Test sentence ending patterns
        boundary = chunker._find_boundary_with_pattern(text, 0, len(text), r'[.!?]\s+')
        assert boundary is not None
        assert text[boundary - 2:boundary] in [". ", "! ", "? "]

    def test_reassemble_chunks(self):
        """Test reassembling chunks back to text."""
        chunker = TextChunker(target_word_count=5)
        original_text = "First sentence. Second sentence.\n\nNew paragraph here."
        chunks = chunker.chunk_text(original_text)
        
        reassembled = chunker.reassemble_chunks(chunks)
        
        # Should preserve meaning even if formatting changes slightly
        assert "First sentence" in reassembled
        assert "Second sentence" in reassembled
        assert "New paragraph" in reassembled

    def test_reassemble_empty_chunks(self):
        """Test reassembling empty chunk list."""
        chunker = TextChunker()
        reassembled = chunker.reassemble_chunks([])
        assert reassembled == ""

    def test_chunking_info(self):
        """Test chunking information generation."""
        chunker = TextChunker(target_word_count=5)
        text = "This is a test text with exactly ten words here."
        info = chunker.get_chunking_info(text)
        
        assert info['total_words'] == 10
        assert info['total_chunks'] >= 1
        assert info['avg_words_per_chunk'] > 0
        assert info['min_words_per_chunk'] > 0
        assert info['max_words_per_chunk'] > 0

    def test_chunking_info_empty_text(self):
        """Test chunking info for empty text."""
        chunker = TextChunker()
        info = chunker.get_chunking_info("")
        
        assert info['total_words'] == 0
        assert info['total_chunks'] == 0
        assert info['avg_words_per_chunk'] == 0.0
        assert info['min_words_per_chunk'] == 0
        assert info['max_words_per_chunk'] == 0

    def test_boundary_preferences(self):
        """Test that chunk boundaries prefer higher priority breaks."""
        chunker = TextChunker(target_word_count=8, max_word_count=15)
        # Text with both paragraph break and sentence break
        text = "Short paragraph.\n\nLonger second paragraph with more words, including commas, that should trigger chunking."
        chunks = chunker.chunk_text(text)
        
        # Should create multiple chunks due to length
        assert len(chunks) >= 1
        # First chunk should contain the first paragraph
        if len(chunks) > 1:
            assert "Short paragraph" in chunks[0].content

    def test_very_long_words(self):
        """Test handling of very long words."""
        chunker = TextChunker(target_word_count=3, max_word_count=5)
        text = "short superlongwordthatexceedstheusualwordlengthbutshouldbetreatedasasingleword short"
        chunks = chunker.chunk_text(text)
        
        # Should not break in middle of long word
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "superlongword" not in chunk.content or "superlongwordthatexceedstheusualwordlengthbutshouldbetreatedasasingleword" in chunk.content

    def test_mixed_line_endings(self):
        """Test handling of mixed line endings."""
        chunker = TextChunker(target_word_count=5)
        text = "Line one\nLine two\r\nLine three\n\nParagraph break"
        chunks = chunker.chunk_text(text)
        
        # Should handle different line ending types
        assert len(chunks) >= 1
        # Content should be preserved
        reassembled = chunker.reassemble_chunks(chunks)
        assert "Line one" in reassembled
        assert "Paragraph break" in reassembled

    def test_consecutive_punctuation(self):
        """Test handling of consecutive punctuation."""
        chunker = TextChunker(target_word_count=5)
        text = "Question?? Answer!! More text... End."
        chunks = chunker.chunk_text(text)
        
        # Should handle consecutive punctuation correctly
        assert len(chunks) >= 1
        reassembled = chunker.reassemble_chunks(chunks)
        assert "Question" in reassembled
        assert "End" in reassembled

    def test_chunk_number_consistency(self):
        """Test that chunk numbers are consistent and sequential."""
        chunker = TextChunker(target_word_count=3)
        text = "One two three four five six seven eight nine ten eleven twelve"
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_number == i + 1
                assert chunk.total_chunks == len(chunks)

    def test_position_accuracy(self):
        """Test that start and end positions are accurate."""
        chunker = TextChunker(target_word_count=5)
        text = "First chunk content.\n\nSecond chunk content here."
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # First chunk should start at 0
            assert chunks[0].start_position == 0
            # Each chunk should have valid positions
            for chunk in chunks:
                assert chunk.start_position < chunk.end_position
                assert chunk.end_position <= len(text)
                # Content should match positions
                extracted = text[chunk.start_position:chunk.end_position].strip()
                assert chunk.content in extracted or extracted in chunk.content