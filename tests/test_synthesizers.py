"""Tests for TTS synthesizers — base class, edge-tts, and kokoro."""

from __future__ import annotations

import asyncio
import logging
import pytest

from app.synthesizers.base import BaseSynthesizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _has_network() -> bool:
    """Quick check for network connectivity."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


# ── Base class tests ───────────────────────────────────────────────────────

class TestBaseSynthesizer:
    """Test the chunking and progress logic enforced by the base class."""

    def test_split_short_text_single_chunk(self):
        """Short text should produce exactly one chunk."""
        synth = _DummySynthesizer()
        chunks = synth._split_into_chunks("Hello world. This is a test.")
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_split_long_text_multiple_chunks(self):
        """Text exceeding max_chars should be split into multiple chunks."""
        synth = _DummySynthesizer()
        # Build text with clear paragraph boundaries
        paragraphs = [f"Paragraph {i}. " + "x " * 100 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = synth._split_into_chunks(text, max_chars=500)
        assert len(chunks) > 1
        # All original text should be present
        joined = " ".join(chunks)
        for i in range(10):
            assert f"Paragraph {i}" in joined

    def test_split_respects_sentence_boundaries(self):
        """An oversized paragraph should split on sentence boundaries."""
        synth = _DummySynthesizer()
        sentences = [f"Sentence number {i} is here." for i in range(50)]
        single_para = " ".join(sentences)
        chunks = synth._split_into_chunks(single_para, max_chars=200)
        assert len(chunks) > 1
        # No chunk should be much larger than max_chars (allow some slack)
        for chunk in chunks:
            assert len(chunk) < 400, f"Chunk too large: {len(chunk)} chars"

    def test_progress_callback_fires(self):
        """on_chunk_done should fire once per chunk with correct counts."""
        synth = _DummySynthesizer()
        paragraphs = [f"Paragraph {i}. " + "word " * 80 for i in range(5)]
        text = "\n\n".join(paragraphs)

        progress_log: list[tuple[int, int]] = []

        def on_done(completed: int, total: int):
            progress_log.append((completed, total))

        audio = asyncio.run(synth.synthesize(text, on_chunk_done=on_done))

        assert len(audio) > 0
        assert len(progress_log) > 0
        # Last callback should be (total, total)
        last_completed, last_total = progress_log[-1]
        assert last_completed == last_total
        # Progress should be monotonically increasing
        for i in range(1, len(progress_log)):
            assert progress_log[i][0] == progress_log[i - 1][0] + 1

    def test_concatenate_audio(self):
        """Default byte concatenation should combine parts."""
        synth = _DummySynthesizer()
        result = synth._concatenate_audio([b"aaa", b"bbb", b"ccc"])
        assert result == b"aaabbbccc"

    def test_empty_text(self):
        """Empty text should still return something (single chunk)."""
        synth = _DummySynthesizer()
        audio = asyncio.run(synth.synthesize(""))
        assert isinstance(audio, bytes)


class _DummySynthesizer(BaseSynthesizer):
    """Minimal synthesizer for testing base class logic."""

    async def _synthesize_chunk(self, text: str, voice: str | None = None) -> bytes:
        return f"[audio:{len(text)}]".encode()

    def list_voices(self) -> list[str]:
        return ["dummy-voice"]


# ── Edge-TTS tests ─────────────────────────────────────────────────────────

class TestEdgeTTS:
    """Test edge-tts synthesizer (requires network access)."""

    def test_import(self):
        """edge-tts module should be importable."""
        from app.synthesizers.edge import EdgeTTSSynthesizer
        synth = EdgeTTSSynthesizer()
        assert "en-US-AriaNeural" in synth.list_voices()

    @pytest.mark.skipif(
        not _has_network(), reason="No network access for edge-tts"
    )
    def test_synthesize_short_text(self):
        """Synthesize a short sentence and verify we get audio bytes."""
        from app.synthesizers.edge import EdgeTTSSynthesizer
        synth = EdgeTTSSynthesizer()

        progress_log = []
        audio = asyncio.run(
            synth.synthesize(
                "Hello, this is a test.",
                voice="en-US-AriaNeural",
                on_chunk_done=lambda c, t: progress_log.append((c, t)),
            )
        )

        assert isinstance(audio, bytes)
        assert len(audio) > 100, f"Audio too small: {len(audio)} bytes"
        logger.info(f"Edge-TTS produced {len(audio)} bytes, {len(progress_log)} progress callbacks")


# ── Kokoro tests ───────────────────────────────────────────────────────────

class TestKokoro:
    """Test Kokoro TTS synthesizer (requires kokoro package)."""

    def test_import(self):
        """Check if kokoro is importable — skip tests if not."""
        try:
            from app.synthesizers.kokoro_synth import KokoroSynthesizer
            synth = KokoroSynthesizer()
            assert len(synth.list_voices()) > 0
            logger.info("Kokoro imported successfully")
        except ImportError:
            pytest.skip("Kokoro not installed")
        except Exception as e:
            logger.error(f"Kokoro import failed with: {type(e).__name__}: {e}")
            raise

    def test_synthesize_short_text(self):
        """Synthesize a short sentence with Kokoro."""
        try:
            from app.synthesizers.kokoro_synth import KokoroSynthesizer
        except ImportError:
            pytest.skip("Kokoro not installed")

        synth = KokoroSynthesizer()
        progress_log = []

        try:
            audio = asyncio.run(
                synth.synthesize(
                    "Hello, this is a test of the Kokoro synthesizer.",
                    voice="af_heart",
                    on_chunk_done=lambda c, t: progress_log.append((c, t)),
                )
            )
            assert isinstance(audio, bytes)
            assert len(audio) > 100, f"Audio too small: {len(audio)} bytes"
            logger.info(f"Kokoro produced {len(audio)} bytes, {len(progress_log)} progress callbacks")
        except Exception as e:
            logger.error(f"Kokoro synthesize failed: {type(e).__name__}: {e}", exc_info=True)
            raise

    def test_pipeline_init(self):
        """Test that the Kokoro pipeline initializes correctly."""
        try:
            from app.synthesizers.kokoro_synth import KokoroSynthesizer
        except ImportError:
            pytest.skip("Kokoro not installed")

        synth = KokoroSynthesizer()
        try:
            pipeline = synth._get_pipeline()
            assert pipeline is not None
            logger.info(f"Kokoro pipeline initialized: {type(pipeline).__name__}")
        except Exception as e:
            logger.error(f"Kokoro pipeline init failed: {type(e).__name__}: {e}", exc_info=True)
            raise

    def test_blocking_synth_directly(self):
        """Call _synthesize_blocking directly to isolate async issues."""
        try:
            from app.synthesizers.kokoro_synth import KokoroSynthesizer
        except ImportError:
            pytest.skip("Kokoro not installed")

        synth = KokoroSynthesizer()
        try:
            audio = synth._synthesize_blocking("Hello world.", "af_heart")
            assert isinstance(audio, bytes)
            assert len(audio) > 0
            logger.info(f"Kokoro blocking synth produced {len(audio)} bytes")
        except Exception as e:
            logger.error(f"Kokoro blocking synth failed: {type(e).__name__}: {e}", exc_info=True)
            raise

    def test_concatenate_wav(self):
        """Test WAV concatenation works correctly."""
        try:
            from app.synthesizers.kokoro_synth import KokoroSynthesizer
        except ImportError:
            pytest.skip("Kokoro not installed")

        synth = KokoroSynthesizer()
        try:
            part1 = synth._synthesize_blocking("First part.", "af_heart")
            part2 = synth._synthesize_blocking("Second part.", "af_heart")
            combined = synth._concatenate_audio([part1, part2])
            assert len(combined) > len(part1)
            logger.info(f"WAV concat: {len(part1)} + {len(part2)} = {len(combined)} bytes")
        except Exception as e:
            logger.error(f"Kokoro WAV concat failed: {type(e).__name__}: {e}", exc_info=True)
            raise


# ── Helpers ────────────────────────────────────────────────────────────────
