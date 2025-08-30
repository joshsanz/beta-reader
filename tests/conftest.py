"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_text():
    """Sample text for testing text chunking."""
    return """First paragraph with some content here. This is the first paragraph of our test text.

Second paragraph with different content. This paragraph should be separated from the first one by a paragraph break.

Third paragraph with even more content. This final paragraph completes our sample text for testing purposes."""


@pytest.fixture
def long_sample_text():
    """Longer sample text that will definitely trigger chunking."""
    paragraphs = []
    for i in range(10):
        paragraphs.append(
            f"This is paragraph number {i+1} with sufficient content to ensure that "
            f"we have enough text to trigger the chunking algorithm. Each paragraph "
            f"contains multiple sentences to make it more realistic. The content "
            f"varies slightly between paragraphs to make testing more comprehensive."
        )
    return "\n\n".join(paragraphs)