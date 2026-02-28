"""Kokoro TTS synthesizer — high-quality local TTS (optional dependency)."""

from __future__ import annotations

import asyncio

from app.synthesizers.base import BaseSynthesizer

DEFAULT_VOICES = [
    "af_heart",
    "af_bella",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bm_george",
]


class KokoroSynthesizer(BaseSynthesizer):
    """Local TTS using Kokoro (small, fast, high quality)."""

    def __init__(self):
        # Import here so we fail fast if kokoro isn't installed
        import kokoro  # noqa: F401
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")  # "a" = American English
        return self._pipeline

    async def _synthesize_chunk(self, text: str, voice: str | None = None) -> bytes:
        voice = voice or "af_heart"
        # Kokoro is synchronous, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_blocking, text, voice)

    def _synthesize_blocking(self, text: str, voice: str) -> bytes:
        import io
        import soundfile as sf

        pipeline = self._get_pipeline()
        audio_segments = []
        for _, _, audio in pipeline(text, voice=voice):
            audio_segments.append(audio)

        if not audio_segments:
            raise ValueError("Kokoro produced no audio output.")

        # Concatenate all segments
        import numpy as np
        full_audio = np.concatenate(audio_segments)

        # Write to WAV bytes (Streamlit can play WAV)
        buf = io.BytesIO()
        sf.write(buf, full_audio, 24000, format="WAV")
        return buf.getvalue()

    def _concatenate_audio(self, parts: list[bytes]) -> bytes:
        """Concatenate WAV files properly using numpy + soundfile."""
        import io
        import numpy as np
        import soundfile as sf

        arrays = []
        for part in parts:
            data, _sr = sf.read(io.BytesIO(part))
            arrays.append(data)

        full = np.concatenate(arrays)
        buf = io.BytesIO()
        sf.write(buf, full, 24000, format="WAV")
        return buf.getvalue()

    def list_voices(self) -> list[str]:
        return DEFAULT_VOICES
