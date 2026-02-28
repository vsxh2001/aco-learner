"""Table of Contents extraction and timestamp estimation from Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TOCEntry:
    """A single entry in the Table of Contents."""

    title: str
    level: int  # 1 = h1, 2 = h2, … 6 = h6
    timestamp_seconds: float = 0.0


def extract_toc(markdown: str) -> list[TOCEntry]:
    """Extract headings from Markdown text as TOC entries (no timestamps).

    Args:
        markdown: Raw Markdown text.

    Returns:
        Ordered list of TOCEntry objects with ``timestamp_seconds = 0``.
    """
    entries: list[TOCEntry] = []
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        entries.append(TOCEntry(title=title, level=level))
    return entries


def estimate_timestamps(
    markdown: str,
    audio_duration_seconds: float,
) -> list[TOCEntry]:
    """Extract TOC from Markdown and attach estimated audio timestamps.

    Timestamps are estimated proportionally: a heading found at character
    position *p* in a document of total length *n* is assigned a timestamp
    of ``(p / n) * audio_duration_seconds``.

    Args:
        markdown: Raw Markdown text containing ATX headings (``# …``).
        audio_duration_seconds: Total audio duration in seconds.

    Returns:
        Ordered list of TOCEntry objects with estimated timestamps.
    """
    # Guard against empty markdown: division would be zero/one anyway
    total_len = max(len(markdown), 1)
    entries: list[TOCEntry] = []
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        fraction = match.start() / total_len
        timestamp = fraction * audio_duration_seconds
        entries.append(TOCEntry(title=title, level=level, timestamp_seconds=timestamp))
    return entries


def estimate_audio_duration(voice_script: str, wpm: int = 150) -> float:
    """Estimate audio duration from word count and speaking rate.

    Args:
        voice_script: The voice-ready script text.
        wpm: Average words per minute (default 150).

    Returns:
        Estimated duration in seconds.
    """
    words = len(voice_script.split())
    return (words / max(wpm, 1)) * 60.0


def get_audio_duration(audio_bytes: bytes, audio_format: str) -> float:
    """Get actual audio duration in seconds using pydub.

    Falls back to 0.0 if pydub or the underlying codec is unavailable.

    Args:
        audio_bytes: Raw audio data.
        audio_format: Format string, e.g. ``"mp3"`` or ``"wav"``.

    Returns:
        Duration in seconds, or 0.0 on failure.
    """
    try:
        import io

        from pydub import AudioSegment  # type: ignore

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
        return len(audio) / 1000.0
    except Exception:
        return 0.0
