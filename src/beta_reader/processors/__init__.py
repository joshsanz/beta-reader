"""File processors module."""

from .base import BaseProcessor
from .epub import EpubProcessor
from .text import TextProcessor

__all__ = [
    "BaseProcessor",
    "EpubProcessor",
    "TextProcessor",
]
