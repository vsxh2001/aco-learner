"""Abstract base for all document-to-Markdown converters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseConverter(ABC):
    """Convert a document (as raw bytes) into Markdown text."""

    @abstractmethod
    def convert(self, data: bytes, filename: str = "") -> str:
        """
        Convert raw document bytes to a Markdown string.

        Args:
            data: Raw bytes of the input document.
            filename: Original filename (used for logging / metadata).

        Returns:
            A Markdown-formatted string.
        """
        ...
