"""Edge-TTS synthesizer — free, no API key, 400+ neural voices."""

from __future__ import annotations

import asyncio
import io

import edge_tts

from app.synthesizers.base import BaseSynthesizer

# Curated list of high-quality English voices
DEFAULT_VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-DavisNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
]


class EdgeTTSSynthesizer(BaseSynthesizer):
    """Uses Microsoft Edge's free online TTS endpoint."""

    async def _synthesize_chunk(self, text: str, voice: str | None = None) -> bytes:
        voice = voice or "en-US-AriaNeural"
        communicate = edge_tts.Communicate(text, voice)

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        return audio_buffer.getvalue()

    def list_voices(self) -> list[str]:
        return DEFAULT_VOICES


def synthesize_sync(text: str, voice: str | None = None) -> bytes:
    """Synchronous wrapper for use in Streamlit / non-async contexts."""
    synth = EdgeTTSSynthesizer()
    return asyncio.run(synth.synthesize(text, voice))
