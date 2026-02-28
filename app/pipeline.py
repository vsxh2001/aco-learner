"""Pipeline orchestrator — chains PDF → Markdown → Voice Script → Audio."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.config import AppConfig, get_config
from app.converters import get_converter
from app.synthesizers import get_synthesizer
from app.transformers.voice_script import TransformResult, transform


@dataclass
class PipelineResult:
    """Full output of the conversion pipeline."""
    filename: str
    raw_markdown: str
    cleaned_markdown: str
    voice_script: str
    audio_bytes: bytes | None = None
    audio_format: str = "mp3"
    sections_processed: int = 0
    metadata: dict = field(default_factory=dict)


async def _run_pipeline_async(
    file_bytes: bytes,
    filename: str,
    config: AppConfig,
    skip_llm: bool,
    skip_tts: bool,
    progress_callback: Callable[[str, float], None] | None,
) -> PipelineResult:
    """Async inner implementation of the pipeline."""

    def _progress(stage: str, pct: float):
        if progress_callback:
            progress_callback(stage, pct)

    # ── Step 1: Document → Markdown ────────────────────────────────────
    _progress("Converting PDF to Markdown…", 0.1)
    ext = Path(filename).suffix
    converter = get_converter(ext)
    raw_markdown = converter.convert(file_bytes, filename)
    _progress("Markdown extracted", 0.3)

    # ── Step 2: Markdown → Voice Script ────────────────────────────────
    _progress("Transforming to voice script…", 0.4)

    def _on_section_done(completed: int, total: int):
        pct = 0.4 + 0.3 * (completed / max(total, 1))
        _progress(f"Rewriting section {completed}/{total}…", pct)

    result: TransformResult = await transform(
        markdown=raw_markdown,
        model=config.llm.model,
        temperature=config.llm.temperature,
        api_base=config.llm.api_base,
        skip_llm=skip_llm,
        on_section_done=_on_section_done,
        max_concurrent=config.llm.max_concurrent,
    )
    _progress("Voice script ready", 0.7)

    # ── Step 3: Voice Script → Audio ───────────────────────────────────
    audio_bytes = None
    audio_format = "mp3"
    if not skip_tts:
        _progress("Synthesizing audio…", 0.75)
        try:
            synth = get_synthesizer(config.tts.provider)

            def _on_tts_chunk(completed: int, total: int):
                pct = 0.75 + 0.2 * (completed / max(total, 1))
                _progress(f"Synthesizing audio chunk {completed}/{total}…", pct)

            audio_bytes = await synth.synthesize(
                result.voice_script,
                config.tts.voice,
                on_chunk_done=_on_tts_chunk,
            )
            if config.tts.provider == "kokoro":
                audio_format = "wav"
            _progress("Audio ready", 1.0)
        except Exception as e:
            _progress(f"TTS failed: {e}", 0.8)
    else:
        _progress("Done (audio skipped)", 1.0)

    return PipelineResult(
        filename=filename,
        raw_markdown=raw_markdown,
        cleaned_markdown=result.cleaned_markdown,
        voice_script=result.voice_script,
        audio_bytes=audio_bytes,
        audio_format=audio_format,
        sections_processed=result.sections_processed,
        metadata={
            "model": config.llm.model,
            "tts_provider": config.tts.provider,
            "tts_voice": config.tts.voice,
            "llm_skipped": skip_llm,
            "tts_skipped": skip_tts,
            "llm_concurrency": config.llm.max_concurrent,
        },
    )


def run_pipeline(
    file_bytes: bytes,
    filename: str,
    config: AppConfig | None = None,
    skip_llm: bool = False,
    skip_tts: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PipelineResult:
    """
    Execute the full conversion pipeline (sync wrapper).

    Creates a new event loop if needed, or uses the running one.
    Streamlit runs its own loop, so we handle both cases.

    Args:
        file_bytes: Raw uploaded file content.
        filename: Original filename (used to pick the right converter).
        config: App configuration. Uses defaults if None.
        skip_llm: Skip the LLM rewriting step (rule-based only).
        skip_tts: Skip audio synthesis.
        progress_callback: Optional fn(stage_name, progress_0_to_1) for UI updates.

    Returns:
        PipelineResult with all intermediate and final outputs.
    """
    config = config or get_config()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = _run_pipeline_async(
        file_bytes, filename, config, skip_llm, skip_tts, progress_callback
    )

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g. Streamlit).
        # Use nest_asyncio to allow nested event loop usage,
        # or fall back to a thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
