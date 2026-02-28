"""OpenAI TTS synthesizer — high quality cloud voices (requires API key)."""

from __future__ import annotations

import asyncio

from app.synthesizers.base import BaseSynthesizer

DEFAULT_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class OpenAITTSSynthesizer(BaseSynthesizer):
    """Uses OpenAI's TTS API (tts-1 or tts-1-hd)."""

    def __init__(self, model: str = "tts-1"):
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model

    async def _synthesize_chunk(self, text: str, voice: str | None = None) -> bytes:
        voice = voice or "nova"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._synthesize_blocking, text, voice
        )

    def _synthesize_blocking(self, text: str, voice: str) -> bytes:
        response = self._client.audio.speech.create(
            model=self._model,
            voice=voice,
            input=text,
            response_format="mp3",
        )
        return response.content

    def list_voices(self) -> list[str]:
        return DEFAULT_VOICES
