"""Tests for voice script transformer — preprocessing and preamble stripping."""

from __future__ import annotations

import pytest

from app.transformers.voice_script import (
    _strip_llm_preamble,
    _strip_markdown,
    _expand_symbols,
    _split_into_sections,
    preprocess,
)


class TestStripMarkdown:
    def test_headings_removed(self):
        assert _strip_markdown("# Hello") == "Hello"
        assert _strip_markdown("### Sub heading") == "Sub heading"

    def test_bold_italic_removed(self):
        assert "bold" in _strip_markdown("**bold** text")
        assert "italic" in _strip_markdown("*italic* text")

    def test_links_keep_text(self):
        result = _strip_markdown("[click here](http://example.com)")
        assert "click here" in result
        assert "http" not in result

    def test_images_keep_alt(self):
        result = _strip_markdown("![a diagram](img.png)")
        assert "a diagram" in result
        assert "img.png" not in result

    def test_code_blocks_removed(self):
        result = _strip_markdown("before\n```python\ncode\n```\nafter")
        assert "code" not in result
        assert "before" in result


class TestExpandSymbols:
    def test_percent(self):
        assert "percent" in _expand_symbols("50%")

    def test_ampersand(self):
        assert "and" in _expand_symbols("A & B")


class TestSplitSections:
    def test_merges_small_sections(self):
        text = "Hi.\n\nBye."
        sections = _split_into_sections(text)
        # Both are tiny, should be merged into one
        assert len(sections) == 1

    def test_splits_large_sections(self):
        big = ("Word " * 100 + "\n\n") * 5
        sections = _split_into_sections(big)
        assert len(sections) > 1


class TestStripPreamble:
    @pytest.mark.parametrize("preamble", [
        "Here is the script:\n",
        "Here's the rewritten voice-ready script:\n",
        "Sure! Here is your converted text:\n",
        "Certainly!\n",
        "Of course.\n",
        "Here is the voice script:\n",
        "I've rewritten the text:\n",
        "Below is the narration:\n",
        "Okay, here is the script.\n",
    ])
    def test_common_preambles_stripped(self, preamble):
        body = "The quick brown fox jumps over the lazy dog."
        result = _strip_llm_preamble(preamble + body)
        assert result.startswith("The quick"), f"Preamble not stripped: {result[:50]}"

    def test_clean_text_unchanged(self):
        text = "Exercise has many benefits. First, it improves your health."
        assert _strip_llm_preamble(text) == text

    def test_does_not_strip_real_content(self):
        text = "Here we discuss the main findings of the study."
        result = _strip_llm_preamble(text)
        assert "findings" in result
