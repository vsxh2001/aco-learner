"""Transform Markdown into a voice-agent-ready spoken script.

Two-stage process:
  1. Rule-based pre-processing — strip Markdown syntax, expand symbols.
  2. LLM rewriting — turn cleaned text into natural spoken prose.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable

import litellm

# ── Symbol / abbreviation expansion table ──────────────────────────────────

SYMBOL_MAP: dict[str, str] = {
    "%": " percent",
    "&": " and ",
    "+": " plus ",
    "=": " equals ",
    "@": " at ",
    "#": " number ",
    "~": " approximately ",
    "→": " leads to ",
    "←": " comes from ",
    "≥": " greater than or equal to ",
    "≤": " less than or equal to ",
    "≠": " is not equal to ",
}

# ── System prompt for the LLM rewriter ─────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional script writer who converts written content into \
natural spoken-word scripts suitable for text-to-speech narration.

CRITICAL: Output ONLY the spoken script itself. Do NOT include any \
preamble, introduction, or meta-commentary like "Here is the script", \
"Sure!", "Below is the converted text", etc. Start directly with the \
narrated content.

Rules:
- Convert bullet points and numbered lists into flowing prose \
  ("There are three key points. First, … Second, … Finally, …").
- Narrate tables as natural sentences describing the data.
- Spell out abbreviations on first use: "PDF" → "P-D-F", "API" → "A-P-I".
- Convert symbols: "$10" → "ten dollars", "50%" → "fifty percent".
- Write numbers contextually: small numbers as words, years as digits.
- Use short, simple sentences. Avoid nested clauses.
- Add natural transitions between sections: "Now, moving on to…", \
  "Next, let's look at…"
- Remove any leftover Markdown formatting, URLs, or reference links.
- Preserve the full meaning and all information from the original text.

Example input:
  ## Benefits of Exercise
  - Improves cardiovascular health
  - Boosts mood and mental clarity
  - Aids in weight management
  Regular exercise is recommended by the WHO for adults aged 18–64.

Example output:
  Let's talk about the benefits of exercise. There are three main benefits \
to highlight. First, it improves cardiovascular health. Second, it boosts \
your mood and mental clarity. And third, it aids in weight management. \
The World Health Organization recommends regular exercise for adults \
between the ages of eighteen and sixty-four.
"""


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class TransformResult:
    cleaned_markdown: str
    voice_script: str
    sections_processed: int


# ── Rule-based pre-processing ──────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove Markdown syntax while preserving readable text."""
    # Remove images: ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove links but keep text: [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove reference-style links: [text][ref]
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
    # Remove code blocks (must happen before inline backtick removal)
    text = re.sub(r"```[^\n]*\n[\s\S]*?```", "", text)
    text = re.sub(r"~~~[^\n]*\n[\s\S]*?~~~", "", text)
    # Remove headings markers (keep text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove strikethrough
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove blockquote markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Clean up bullet markers (keep text)
    text = re.sub(r"^[\s]*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _expand_symbols(text: str) -> str:
    """Replace common symbols with their spoken equivalents."""
    for symbol, replacement in SYMBOL_MAP.items():
        text = text.replace(symbol, replacement)
    return text


def _split_into_sections(text: str) -> list[str]:
    """Split text on double newlines into manageable sections."""
    sections = re.split(r"\n{2,}", text)
    # Merge tiny sections with the next one
    merged: list[str] = []
    buffer = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        buffer = f"{buffer}\n\n{section}" if buffer else section
        if len(buffer) > 200:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def preprocess(markdown: str) -> str:
    """Run rule-based cleaning on Markdown text."""
    text = _strip_markdown(markdown)
    text = _expand_symbols(text)
    return text


# ── LLM-powered rewriting ─────────────────────────────────────────────────

# Patterns that LLMs commonly prepend despite being told not to
_PREAMBLE_PATTERNS = re.compile(
    r"^\s*(?:"
    r"here(?:'s| is) (?:the |your |a )?(?:re)?(?:written |converted |transformed )?"
    r"(?:voice[- ]?(?:ready )?)?(?:script|version|text|narration|output|result)?[^.\n]*"
    r"|sure[!,.\s]*"
    r"|okay[!,.\s]*"
    r"|certainly[!,.\s]*"
    r"|of course[!,.\s]*"
    r"|absolutely[!,.\s]*"
    r"|i'?(?:ve| have) (?:re)?(?:written|converted|transformed)[^.\n]*"
    r"|below is[^.\n]*"
    r"|the following is[^.\n]*"
    r")[\s:.\-!]*\n*",
    re.IGNORECASE,
)


def _strip_llm_preamble(text: str) -> str:
    """Remove common LLM preamble/meta-commentary from the output."""
    # Apply repeatedly to handle compound preambles like "Sure! Here is the script:\n"
    cleaned = text
    for _ in range(3):
        new = _PREAMBLE_PATTERNS.sub("", cleaned, count=1).strip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned if cleaned else text


async def rewrite_with_llm(
    text: str,
    model: str = "ollama/llama3.2",
    temperature: float = 0.3,
    api_base: str | None = None,
    on_section_done: Callable[[int, int], None] | None = None,
    max_concurrent: int = 4,
) -> str:
    """
    Rewrite cleaned text into a voice-ready script using an LLM.

    Sends up to *max_concurrent* section requests in parallel so Ollama
    (or any LLM backend) can batch GPU work.  Section order is preserved.
    Progress is reported as each section completes.

    Args:
        text: Pre-processed text.
        model: LiteLLM model string.
        temperature: Sampling temperature.
        api_base: Optional API base URL.
        on_section_done: Callback(completed, total) after each section finishes.
        max_concurrent: Maximum number of LLM requests in flight at once.
    """
    sections = _split_into_sections(text)

    if not sections:
        return text

    total = len(sections)
    results: list[str | None] = [None] * total
    completed_count = 0
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_section(index: int, section: str) -> None:
        nonlocal completed_count
        user_msg = section
        if index > 0:
            user_msg = (
                f"(Continuing from the previous section. "
                f"Maintain the same tone and style.)\n\n{section}"
            )

        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
        }
        if api_base:
            kwargs["api_base"] = api_base

        async with semaphore:
            response = await litellm.acompletion(**kwargs)

        result_text = _strip_llm_preamble(
            response.choices[0].message.content.strip()
        )
        results[index] = result_text

        async with lock:
            completed_count += 1
            if on_section_done:
                on_section_done(completed_count, total)

    tasks = [
        asyncio.create_task(_process_section(i, sec))
        for i, sec in enumerate(sections)
    ]
    await asyncio.gather(*tasks)

    return "\n\n".join(r for r in results if r is not None)


# ── Public API ─────────────────────────────────────────────────────────────

async def transform(
    markdown: str,
    model: str = "ollama/llama3.2",
    temperature: float = 0.3,
    api_base: str | None = None,
    skip_llm: bool = False,
    on_section_done: Callable[[int, int], None] | None = None,
    max_concurrent: int = 4,
) -> TransformResult:
    """
    Full transformation pipeline: preprocess + LLM rewrite.

    Args:
        markdown: Raw Markdown text.
        model: LiteLLM model string (e.g., "ollama/llama3.2", "gpt-4o").
        temperature: LLM sampling temperature.
        api_base: Optional API base URL (for Ollama, etc.).
        skip_llm: If True, return only the rule-based cleaned version.
        on_section_done: Callback(completed, total) for progress tracking.
        max_concurrent: Maximum number of concurrent LLM requests (Ollama default: 4).

    Returns:
        TransformResult with cleaned markdown and final voice script.
    """
    cleaned = preprocess(markdown)
    sections = _split_into_sections(cleaned)

    if skip_llm:
        return TransformResult(
            cleaned_markdown=cleaned,
            voice_script=cleaned,
            sections_processed=len(sections),
        )

    voice_script = await rewrite_with_llm(
        cleaned, model, temperature, api_base, on_section_done, max_concurrent
    )

    return TransformResult(
        cleaned_markdown=cleaned,
        voice_script=voice_script,
        sections_processed=len(sections),
    )
