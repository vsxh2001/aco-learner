"""Centralized configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class LLMConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "ollama/llama3.2"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3")))
    api_base: str | None = field(default_factory=lambda: os.getenv("OLLAMA_API_BASE"))
    max_concurrent: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_CONCURRENT", "4")))


@dataclass
class TTSConfig:
    provider: str = field(default_factory=lambda: os.getenv("TTS_PROVIDER", "edge-tts"))
    voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "en-US-AriaNeural"))
    openai_voice: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_VOICE", "nova"))


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


def get_config() -> AppConfig:
    """Return a fresh config instance (re-reads env vars)."""
    return AppConfig()
