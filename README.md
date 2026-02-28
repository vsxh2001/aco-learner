# ACO Learner

Convert documents (PDF, etc.) into voice-ready spoken scripts and audio. Upload a PDF, get a natural narration script rewritten by an LLM, and synthesize it to audio — all from a simple web UI.

## Pipeline

```
PDF  ──►  Markdown  ──►  Voice Script  ──►  Audio
     pymupdf4llm     rule-based cleanup    edge-tts / Kokoro
                      + LLM rewriting      / OpenAI TTS
```

1. **PDF → Markdown** — Extracts structured Markdown via `pymupdf4llm`.
2. **Markdown → Voice Script** — Two-stage transform:
   - *Rule-based*: strips Markdown syntax, expands symbols (`%` → "percent", `&` → "and", etc.).
   - *LLM rewriting*: converts bullet lists to flowing prose, narrates tables, spells out abbreviations, adds spoken transitions. Sends sections concurrently for better GPU utilization.
3. **Voice Script → Audio** — Text-to-speech synthesis with multiple provider options.

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) (for local LLM inference)

### Install

```bash
git clone <repo-url> && cd aco-learner
uv sync                       # core deps
uv sync --extra dev           # + test deps
uv sync --extra all           # + Kokoro + OpenAI TTS
```

### Set up Ollama

```bash
ollama pull llama3.2          # or any model you prefer
```

### Configure

```bash
cp .env.example .env
# Edit .env to taste — defaults work for local Ollama + edge-tts
```

### Run

```bash
uv run streamlit run app/ui/main.py
```

Open http://localhost:8501, upload a PDF, and hit **Convert**.

## Configuration

All settings can be configured via environment variables (`.env`) or the web UI sidebar.

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `ollama/llama3.2` | LiteLLM model string |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_MAX_CONCURRENT` | `4` | Parallel LLM requests (match `OLLAMA_NUM_PARALLEL`) |
| `OLLAMA_API_BASE` | *(auto)* | Ollama server URL |
| `TTS_PROVIDER` | `edge-tts` | TTS backend: `edge-tts`, `kokoro`, `openai-tts` |
| `TTS_VOICE` | `en-US-AriaNeural` | Voice name for the selected provider |

## LLM Providers

The UI provides a provider dropdown with auto-detected models:

| Provider | Setup | Notes |
|---|---|---|
| **Ollama (local)** | `ollama pull <model>` | Models auto-detected from local server |
| **OpenAI** | Set `OPENAI_API_KEY` in `.env` | gpt-4o, gpt-4o-mini, etc. |
| **Anthropic** | Set `ANTHROPIC_API_KEY` in `.env` | Claude Sonnet, Haiku, etc. |
| **Other** | Free-text model string | Any [LiteLLM](https://docs.litellm.ai/docs/providers)-supported provider |

## TTS Providers

| Provider | Install | Cost | Quality |
|---|---|---|---|
| **edge-tts** | Included | Free | Good (Microsoft Edge neural voices) |
| **Kokoro** | `uv sync --extra kokoro` | Free / local | High (~82M params, needs GPU) |
| **OpenAI TTS** | `uv sync --extra openai-tts` | Paid API | Excellent |

## Project Structure

```
app/
├── config.py                  # Centralized config from .env
├── pipeline.py                # Orchestrator: PDF → MD → Script → Audio
├── converters/
│   ├── base.py                # Abstract converter interface
│   └── pdf_to_md.py           # PDF extraction via pymupdf4llm
├── transformers/
│   └── voice_script.py        # MD cleanup + async LLM rewriting
├── synthesizers/
│   ├── base.py                # Abstract TTS with chunk-based progress
│   ├── edge.py                # edge-tts (free, default)
│   ├── kokoro_synth.py        # Kokoro local TTS
│   └── openai_tts.py          # OpenAI TTS API
└── ui/
    └── main.py                # Streamlit web interface
tests/
├── test_voice_script.py       # Preprocessing + preamble stripping tests
└── test_synthesizers.py       # TTS chunking, progress, provider tests
```

## Development

```bash
uv sync --extra dev
uv run pytest -v
```

## GPU Utilization (Ollama)

LLM rewriting sends up to `LLM_MAX_CONCURRENT` (default 4) section requests in parallel, matching Ollama's `OLLAMA_NUM_PARALLEL` setting. This keeps the GPU busy processing multiple sequences simultaneously instead of idling between sequential requests.

To increase parallelism on the Ollama side:

```bash
# /etc/systemd/system/ollama.service → [Service] section:
Environment="OLLAMA_NUM_PARALLEL=8"
# Then:
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Then set **Concurrent requests** in the UI sidebar (or `LLM_MAX_CONCURRENT` in `.env`) to match.

## License

MIT
