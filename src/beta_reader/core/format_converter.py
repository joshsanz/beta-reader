"""Format conversion utilities for beta-reader processing.

This module provides conversion between HTML and Markdown formats to ensure
consistent input/output handling for language models and EPUB processing.
"""

import re
from html_to_markdown import convert_to_markdown
import markdown


class FormatConverter:
    """Handles conversion between HTML and Markdown formats."""

    def __init__(self):
        """Initialize the format converter with markdown extensions."""
        # Configure markdown processor with commonly needed extensions
        self._md_extensions = [
            'extra',  # Includes tables, fenced code blocks, etc.
            'nl2br',  # Convert newlines to <br> tags
        ]

        # Initialize markdown processor
        self._md_processor = markdown.Markdown(extensions=self._md_extensions)

    def html_to_markdown(self, html_content: str) -> str:
        """Convert HTML content to Markdown format.

        Args:
            html_content: HTML content string

        Returns:
            Markdown formatted string
        """
        if not html_content.strip():
            return html_content

        # Convert HTML to markdown
        markdown_content = convert_to_markdown(html_content)

        # Clean up common conversion artifacts
        markdown_content = self._clean_markdown_output(markdown_content)

        return markdown_content

    def markdown_to_html(self, markdown_content: str) -> str:
        """Convert Markdown content to HTML format.

        Args:
            markdown_content: Markdown content string

        Returns:
            HTML formatted string
        """
        if not markdown_content.strip():
            return markdown_content

        # Convert markdown to HTML
        html_content = self._md_processor.convert(markdown_content)

        # Reset the markdown processor state for next use
        self._md_processor.reset()

        # Clean up and format the HTML output
        html_content = self._clean_html_output(html_content)

        return html_content

    def _clean_markdown_output(self, content: str) -> str:
        """Clean up markdown conversion artifacts."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Trim leading/trailing whitespace but preserve paragraph structure
        content = content.strip()

        return content

    def _clean_html_output(self, content: str) -> str:
        """Clean up HTML conversion artifacts."""
        # Ensure proper paragraph wrapping
        if content and not content.startswith('<'):
            # Wrap bare text in paragraphs
            paragraphs = content.split('\n\n')
            wrapped_paragraphs = []
            for para in paragraphs:
                para = para.strip()
                if para:
                    if not para.startswith('<'):
                        para = f'<p>{para}</p>'
                    wrapped_paragraphs.append(para)
            content = '\n'.join(wrapped_paragraphs)

        # Clean up excessive whitespace in HTML
        content = re.sub(r'>\s+<', '><', content)
        content = re.sub(r'\n\s*\n', '\n', content)

        return content.strip()

    def detect_format(self, content: str) -> str:
        """Detect if content is HTML or plain text/markdown.

        Args:
            content: Content to analyze

        Returns:
            'html' if HTML tags detected, 'markdown' otherwise
        """
        # Look for HTML tags
        html_tag_pattern = r'<[^>]+>'
        if re.search(html_tag_pattern, content):
            return 'html'
        return 'markdown'

    def ensure_markdown_format(self, content: str) -> str:
        """Ensure content is in markdown format.

        Converts HTML to markdown if needed, otherwise returns as-is.

        Args:
            content: Content to process

        Returns:
            Content in markdown format
        """
        if self.detect_format(content) == 'html':
            return self.html_to_markdown(content)
        return content

    def ensure_html_format(self, content: str) -> str:
        """Ensure content is in HTML format.

        Converts markdown to HTML if needed, otherwise returns as-is.

        Args:
            content: Content to process

        Returns:
            Content in HTML format
        """
        if self.detect_format(content) == 'markdown':
            return self.markdown_to_html(content)
        return content


# Global converter instance
_converter = None


def get_converter() -> FormatConverter:
    """Get the global format converter instance."""
    global _converter
    if _converter is None:
        _converter = FormatConverter()
    return _converter


def html_to_markdown(content: str) -> str:
    """Convert HTML to markdown using global converter."""
    return get_converter().html_to_markdown(content)


def markdown_to_html(content: str) -> str:
    """Convert markdown to HTML using global converter."""
    return get_converter().markdown_to_html(content)


def ensure_markdown_format(content: str) -> str:
    """Ensure content is in markdown format using global converter."""
    return get_converter().ensure_markdown_format(content)


def ensure_html_format(content: str) -> str:
    """Ensure content is in HTML format using global converter."""
    return get_converter().ensure_html_format(content)