"""Tests for the Table of Contents (TOC) extraction and timestamp estimation."""

from __future__ import annotations

import pytest

from app.transformers.toc import (
    TOCEntry,
    estimate_audio_duration,
    estimate_timestamps,
    extract_toc,
)


class TestExtractToc:
    def test_returns_empty_for_no_headings(self):
        assert extract_toc("Just some plain text.") == []

    def test_single_h1(self):
        toc = extract_toc("# Hello World")
        assert len(toc) == 1
        assert toc[0].title == "Hello World"
        assert toc[0].level == 1

    def test_multiple_levels(self):
        md = "# Chapter 1\n\n## Section 1.1\n\n### Sub-section 1.1.1\n\n## Section 1.2"
        toc = extract_toc(md)
        assert len(toc) == 4
        assert toc[0].level == 1
        assert toc[1].level == 2
        assert toc[2].level == 3
        assert toc[3].level == 2

    def test_all_heading_levels(self):
        md = "\n".join(f"{'#' * i} Level {i}" for i in range(1, 7))
        toc = extract_toc(md)
        assert len(toc) == 6
        for i, entry in enumerate(toc, start=1):
            assert entry.level == i

    def test_heading_text_stripped(self):
        toc = extract_toc("##   Spaced Title  ")
        assert toc[0].title == "Spaced Title"

    def test_default_timestamps_zero(self):
        toc = extract_toc("# Section A\n\n## Sub-section")
        for entry in toc:
            assert entry.timestamp_seconds == 0.0

    def test_inline_hash_not_treated_as_heading(self):
        """A # not at the start of a line should not be a heading."""
        toc = extract_toc("Some text with # inline hash")
        assert len(toc) == 0


class TestEstimateTimestamps:
    def test_empty_markdown_returns_empty(self):
        assert estimate_timestamps("", 60.0) == []

    def test_no_headings_returns_empty(self):
        assert estimate_timestamps("Just plain text.", 60.0) == []

    def test_first_heading_near_zero(self):
        md = "# Introduction\n\nSome text here."
        toc = estimate_timestamps(md, 100.0)
        assert len(toc) == 1
        # The first heading is at the very start, so timestamp should be ~0
        assert toc[0].timestamp_seconds < 5.0

    def test_timestamps_increase_with_position(self):
        md = "# First\n\n" + "word " * 200 + "\n\n# Second\n\n" + "word " * 200 + "\n\n# Third"
        toc = estimate_timestamps(md, 120.0)
        assert len(toc) == 3
        assert toc[0].timestamp_seconds < toc[1].timestamp_seconds
        assert toc[1].timestamp_seconds < toc[2].timestamp_seconds

    def test_timestamps_bounded_by_duration(self):
        md = "# Section A\n\n" + "x " * 100 + "\n\n# Section B"
        duration = 90.0
        toc = estimate_timestamps(md, duration)
        for entry in toc:
            assert 0.0 <= entry.timestamp_seconds <= duration

    def test_levels_preserved(self):
        md = "# H1\n\n## H2\n\n### H3"
        toc = estimate_timestamps(md, 60.0)
        assert [e.level for e in toc] == [1, 2, 3]

    def test_zero_duration(self):
        md = "# Section\n\nText."
        toc = estimate_timestamps(md, 0.0)
        assert all(e.timestamp_seconds == 0.0 for e in toc)


class TestEstimateAudioDuration:
    def test_empty_text_returns_zero(self):
        assert estimate_audio_duration("") == 0.0

    def test_known_word_count(self):
        # 150 words at 150 wpm → 60 seconds
        text = " ".join(["word"] * 150)
        duration = estimate_audio_duration(text, wpm=150)
        assert abs(duration - 60.0) < 0.5

    def test_scales_with_word_count(self):
        short = estimate_audio_duration("word " * 50)
        long = estimate_audio_duration("word " * 200)
        assert long > short

    def test_custom_wpm(self):
        text = " ".join(["word"] * 300)
        slow = estimate_audio_duration(text, wpm=100)
        fast = estimate_audio_duration(text, wpm=200)
        assert slow > fast
