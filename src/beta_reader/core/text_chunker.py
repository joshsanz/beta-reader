"""Text chunking utilities for splitting long content into manageable pieces."""

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    content: str
    chunk_number: int
    total_chunks: int
    word_count: int
    start_position: int
    end_position: int


class TextChunker:
    """Handles splitting text into chunks while preserving logical boundaries."""

    def __init__(self, target_word_count: int = 500, max_word_count: int = 750):
        """Initialize text chunker.

        Args:
            target_word_count: Target number of words per chunk.
            max_word_count: Maximum number of words per chunk before forced split.
        """
        self.target_word_count = target_word_count
        self.max_word_count = max_word_count

    def chunk_text(self, text: str) -> list[TextChunk]:
        """Split text into chunks, preserving logical boundaries.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        if not text.strip():
            return []

        # Count total words in the text
        total_words = len(text.split())

        # If text is short enough, return as single chunk
        if total_words <= self.target_word_count:
            return [TextChunk(
                content=text,
                chunk_number=1,
                total_chunks=1,
                word_count=total_words,
                start_position=0,
                end_position=len(text)
            )]

        chunks = []
        current_position = 0
        chunk_number = 1

        while current_position < len(text):
            # Find the end of the current chunk
            chunk_end = self._find_chunk_boundary(text, current_position)

            # Extract chunk content
            chunk_content = text[current_position:chunk_end].strip()

            if chunk_content:  # Only add non-empty chunks
                chunk_word_count = len(chunk_content.split())
                chunks.append(TextChunk(
                    content=chunk_content,
                    chunk_number=chunk_number,
                    total_chunks=0,  # Will be updated after all chunks are created
                    word_count=chunk_word_count,
                    start_position=current_position,
                    end_position=chunk_end
                ))
                chunk_number += 1

            current_position = chunk_end

        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks

        return chunks

    def _find_chunk_boundary(self, text: str, start_pos: int) -> int:
        """Find the optimal boundary for ending a chunk.

        Args:
            text: The full text.
            start_pos: Starting position for this chunk.

        Returns:
            Position where the chunk should end.
        """
        if start_pos >= len(text):
            return len(text)

        # Build word positions mapping for accurate position calculation
        remaining_text = text[start_pos:]
        word_positions = self._build_word_positions(remaining_text)

        if len(word_positions) <= self.target_word_count:
            # Remaining text fits in one chunk
            return len(text)

        # Find the approximate target position after target_word_count words
        if self.target_word_count <= len(word_positions):
            # Position after the target_word_count-th word
            target_word_end = word_positions[self.target_word_count - 1][1]
            approx_target_pos = start_pos + target_word_end
        else:
            approx_target_pos = start_pos + len(remaining_text) // 2

        # Find the search end position (after max_word_count words)
        if self.max_word_count <= len(word_positions):
            max_word_end = word_positions[self.max_word_count - 1][1]
            search_end = start_pos + max_word_end
        else:
            search_end = len(text)

        # Try to find a good breaking point in order of preference
        # First, look for boundaries after the target position
        breaking_points = [
            # Double newline (paragraph break) - highest priority
            r'\n\s*\n',
            # Single newline followed by whitespace (likely paragraph)
            r'\n\s+',
            # Single newline
            r'\n',
            # Sentence ending followed by space or newline
            r'[.!?]\s+',
            r'[.!?]\n',
            # Clause ending (semicolon, colon) followed by space
            r'[;:]\s+',
            # Comma followed by space (last resort)
            r',\s+',
        ]

        # Look forward from target position
        for pattern in breaking_points:
            boundary = self._find_boundary_with_pattern(
                text, approx_target_pos, search_end, pattern
            )
            if boundary is not None:
                return boundary

        # If no good boundary found after target, look backwards for high-priority boundaries
        # But don't go too far back - stay within reasonable range
        min_search_pos = max(start_pos, approx_target_pos - (approx_target_pos - start_pos) // 3)

        high_priority_patterns = [
            r'\n\s*\n',  # Paragraph breaks
            r'[.!?]\s+', # Sentence endings
            r'[.!?]\n',  # Sentence endings with newline
        ]

        for pattern in high_priority_patterns:
            boundary = self._find_boundary_with_pattern_backwards(
                text, min_search_pos, approx_target_pos, pattern
            )
            if boundary is not None:
                return boundary

        # If no good boundary found, split at max_word_count
        if len(word_positions) > self.max_word_count:
            max_word_end = word_positions[self.max_word_count - 1][1]
            return start_pos + max_word_end

        # Fallback: return end of text
        return len(text)

    def _build_word_positions(self, text: str) -> list[tuple[int, int]]:
        """Build a list of (start, end) positions for each word in text.

        Args:
            text: Text to analyze.

        Returns:
            List of (start_pos, end_pos) tuples for each word.
        """
        word_positions = []
        words = text.split()
        current_pos = 0

        for word in words:
            # Find the word in the remaining text
            word_start = text.find(word, current_pos)
            if word_start == -1:
                # Fallback: assume words are separated by single spaces
                word_start = current_pos
            word_end = word_start + len(word)
            word_positions.append((word_start, word_end))
            current_pos = word_end

        return word_positions

    def _find_boundary_with_pattern(
        self, text: str, start_search: int, end_search: int, pattern: str
    ) -> int | None:
        """Find a boundary using a regex pattern.

        Args:
            text: The text to search.
            start_search: Start searching from this position.
            end_search: End search at this position.
            pattern: Regex pattern to find.

        Returns:
            Position after the pattern match, or None if not found.
        """
        search_text = text[start_search:end_search]
        matches = list(re.finditer(pattern, search_text))

        if matches:
            # Take the first match (closest to target)
            match = matches[0]
            return start_search + match.end()

        return None

    def _find_boundary_with_pattern_backwards(
        self, text: str, start_search: int, end_search: int, pattern: str
    ) -> int | None:
        """Find a boundary using a regex pattern, searching backwards.

        Args:
            text: The text to search.
            start_search: Start searching from this position.
            end_search: End search at this position (exclusive).
            pattern: Regex pattern to find.

        Returns:
            Position after the pattern match, or None if not found.
        """
        search_text = text[start_search:end_search]
        matches = list(re.finditer(pattern, search_text))

        if matches:
            # Take the last match (closest to target from behind)
            match = matches[-1]
            return start_search + match.end()

        return None

    def reassemble_chunks(self, chunks: list[TextChunk]) -> str:
        """Reassemble processed chunks back into a single text.

        Args:
            chunks: List of processed chunks.

        Returns:
            Reassembled text.
        """
        if not chunks:
            return ""

        # Sort chunks by chunk_number to ensure correct order
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_number)

        # Join chunks with appropriate spacing
        result_parts = []

        for i, chunk in enumerate(sorted_chunks):
            content = chunk.content.strip()
            if content:
                result_parts.append(content)

                # Add spacing between chunks if needed
                if i < len(sorted_chunks) - 1:
                    # Check if content ends with sentence punctuation
                    if content[-1] in '.!?':
                        result_parts.append('\n\n')  # Paragraph break
                    elif content[-1] in ',;:':
                        result_parts.append(' ')     # Continue sentence
                    else:
                        result_parts.append('\n')    # Line break

        return ''.join(result_parts)

    def get_chunking_info(self, text: str) -> dict[str, int | float]:
        """Get information about how text would be chunked.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary with chunking statistics.
        """
        chunks = self.chunk_text(text)

        if not chunks:
            return {
                'total_words': 0,
                'total_chunks': 0,
                'avg_words_per_chunk': 0.0,
                'min_words_per_chunk': 0,
                'max_words_per_chunk': 0,
            }

        word_counts = [chunk.word_count for chunk in chunks]

        return {
            'total_words': len(text.split()),
            'total_chunks': len(chunks),
            'avg_words_per_chunk': sum(word_counts) / len(word_counts),
            'min_words_per_chunk': min(word_counts),
            'max_words_per_chunk': max(word_counts),
        }

    def get_chunk_boundaries_debug(self, text: str, context_words: int = 5) -> list[str]:
        """Get debug information about chunk boundaries.

        Args:
            text: The text to analyze.
            context_words: Number of words to show on each side of boundary.

        Returns:
            List of strings showing chunk boundaries with context.
        """
        chunks = self.chunk_text(text)

        if len(chunks) <= 1:
            return []

        boundaries = []
        text_words = text.split()

        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            chunks[i + 1]

            # Find the boundary position in the original text
            boundary_pos = current_chunk.end_position

            # Get words around the boundary
            # Find word positions to extract context
            char_pos = 0
            word_positions = []

            for word_idx, word in enumerate(text_words):
                word_positions.append((char_pos, char_pos + len(word), word_idx))
                char_pos += len(word) + 1  # +1 for space

            # Find the word index closest to the boundary
            boundary_word_idx = 0
            for pos_start, pos_end, word_idx in word_positions:
                if pos_start <= boundary_pos <= pos_end:
                    boundary_word_idx = word_idx
                    break
                elif pos_start > boundary_pos:
                    boundary_word_idx = word_idx - 1 if word_idx > 0 else 0
                    break
                boundary_word_idx = word_idx

            # Get context words
            max(0, boundary_word_idx - context_words + 1)
            min(len(text_words), boundary_word_idx + context_words + 1)

            # Get actual text context around the boundary position for accurate display
            context_chars = 40
            context_start = max(0, boundary_pos - context_chars)
            context_end = min(len(text), boundary_pos + context_chars)

            before_text = text[context_start:boundary_pos]
            after_text = text[boundary_pos:context_end]

            # Clean up the display text (replace newlines with \\n for visibility)
            before_display = before_text.replace('\n', '\\n').replace('\t', '\\t')
            after_display = after_text.replace('\n', '\\n').replace('\t', '\\t')

            # Determine the type of boundary
            boundary_type = "unknown"
            if boundary_pos >= 2 and text[boundary_pos-2:boundary_pos] == '\n\n':
                boundary_type = "paragraph break"
            elif boundary_pos >= 1 and text[boundary_pos-1:boundary_pos] == '\n':
                boundary_type = "line break"
            elif boundary_pos > 0 and text[boundary_pos-1] in '.!?':
                boundary_type = "sentence end"
            elif boundary_pos > 0 and text[boundary_pos-1] in ',;:':
                boundary_type = "punctuation break"

            boundary_info = (
                f"Chunk {i+1}/{len(chunks)} → Chunk {i+2}/{len(chunks)} (pos {boundary_pos}, {boundary_type})\n"
                f"  Before: ...{before_display[-30:]}\n"
                f"  After:  {after_display[:30]}..."
            )
            boundaries.append(boundary_info)

        return boundaries
