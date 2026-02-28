"""Abstract base for TTS synthesizers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable


class BaseSynthesizer(ABC):
    """Convert text to audio bytes, with per-chunk progress reporting."""

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> bytes:
        """
        Synthesize speech from text, processing chunk-by-chunk with progress.

        Args:
            text: The full text to speak.
            voice: Voice identifier (provider-specific).
            on_chunk_done: Callback(completed_chunks, total_chunks) after each chunk.

        Returns:
            Raw audio bytes (MP3 or WAV depending on provider).
        """
        chunks = self._split_into_chunks(text)
        total = len(chunks)
        audio_parts: list[bytes] = []

        for i, chunk in enumerate(chunks):
            part = await self._synthesize_chunk(chunk, voice)
            audio_parts.append(part)
            if on_chunk_done:
                on_chunk_done(i + 1, total)

        return self._concatenate_audio(audio_parts)

    @abstractmethod
    async def _synthesize_chunk(self, text: str, voice: str | None = None) -> bytes:
        """
        Synthesize a single chunk of text. Subclasses must implement this.

        Args:
            text: A short text chunk to synthesize.
            voice: Voice identifier.

        Returns:
            Raw audio bytes for this chunk.
        """
        ...

    @abstractmethod
    def list_voices(self) -> list[str]:
        """Return a list of available voice identifiers."""
        ...

    def _split_into_chunks(self, text: str, max_chars: int = 2000) -> list[str]:
        """
        Split text into chunks suitable for TTS processing.
        Splits on paragraph boundaries, merges small paragraphs,
        and splits oversized ones on sentence boundaries.
        """
        paragraphs = re.split(r"\n{2,}", text.strip())
        chunks: list[str] = []
        buffer = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds the limit, flush buffer first
            if buffer and len(buffer) + len(para) + 2 > max_chars:
                chunks.append(buffer)
                buffer = ""

            # If a single paragraph is oversized, split on sentences
            if len(para) > max_chars:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sent_buf = ""
                for sent in sentences:
                    if sent_buf and len(sent_buf) + len(sent) + 1 > max_chars:
                        chunks.append(sent_buf)
                        sent_buf = ""
                    sent_buf = f"{sent_buf} {sent}" if sent_buf else sent
                if sent_buf:
                    chunks.append(sent_buf)
            else:
                buffer = f"{buffer}\n\n{para}" if buffer else para

        if buffer:
            chunks.append(buffer)

        return chunks if chunks else [text]

    def _concatenate_audio(self, parts: list[bytes]) -> bytes:
        """Concatenate audio byte segments. Works for raw byte concatenation (MP3)."""
        return b"".join(parts)
