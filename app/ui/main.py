"""
ACO Learner — Convert documents to voice-agent-ready scripts.

Streamlit web UI: upload a PDF, get a voice script + audio.
"""

from __future__ import annotations

import requests
import streamlit as st

from app.config import AppConfig, LLMConfig, TTSConfig
from app.pipeline import run_pipeline
from app.synthesizers import SYNTHESIZERS
from app.synthesizers.edge import EdgeTTSSynthesizer


# ── Helpers ────────────────────────────────────────────────────────────────

LLM_PROVIDERS = ["Ollama (local)", "OpenAI", "Anthropic", "Other"]


@st.cache_data(ttl=30)
def _fetch_ollama_models(api_base: str) -> list[str]:
    """Fetch locally available Ollama models via the /api/tags endpoint."""
    try:
        resp = requests.get(f"{api_base}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return sorted(m["name"] for m in models) or ["llama3.2:latest"]
    except Exception:
        return ["llama3.2:latest"]

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ACO Learner",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ ACO Learner")
st.caption("Upload a PDF → get a voice-ready script and audio.")

# ── Sidebar: settings ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("LLM")
    llm_provider = st.selectbox("Provider", LLM_PROVIDERS, index=0)

    if llm_provider == "Ollama (local)":
        api_base = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="Ollama server address",
        )
        ollama_models = _fetch_ollama_models(api_base)
        selected_model = st.selectbox(
            "Model",
            ollama_models,
            help="Locally available Ollama models",
        )
        llm_model = f"ollama/{selected_model}"
    elif llm_provider == "OpenAI":
        api_base = ""
        llm_model = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            help="Requires OPENAI_API_KEY in .env",
        )
    elif llm_provider == "Anthropic":
        api_base = ""
        llm_model = st.selectbox(
            "Model",
            ["anthropic/claude-sonnet-4-20250514", "anthropic/claude-haiku-4-20250414", "anthropic/claude-3-haiku-20240307"],
            help="Requires ANTHROPIC_API_KEY in .env",
        )
    else:  # Other
        api_base = st.text_input("API Base URL (optional)", value="")
        llm_model = st.text_input(
            "Model",
            value="ollama/llama3.2",
            help="Any LiteLLM model string",
        )

    llm_temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    llm_concurrency = st.slider(
        "Concurrent requests",
        1, 8, 4,
        help="Number of parallel LLM requests. Match to Ollama's OLLAMA_NUM_PARALLEL (default 4).",
    )
    skip_llm = st.checkbox("Skip LLM (rule-based only)", value=False)

    st.divider()
    st.subheader("TTS")
    tts_providers = list(SYNTHESIZERS.keys())
    tts_provider = st.selectbox("Provider", tts_providers, index=0)

    # Get voice list for selected provider
    try:
        synth_instance = SYNTHESIZERS[tts_provider]()
        voices = synth_instance.list_voices()
    except Exception:
        voices = ["en-US-AriaNeural"]
    tts_voice = st.selectbox("Voice", voices, index=0)

    skip_tts = st.checkbox("Skip audio synthesis", value=False)

# ── Build config from sidebar values ──────────────────────────────────────


def build_config() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(
            model=llm_model,
            temperature=llm_temp,
            api_base=api_base or None,
            max_concurrent=llm_concurrency,
        ),
        tts=TTSConfig(
            provider=tts_provider,
            voice=tts_voice,
        ),
    )


# ── Main area: upload + convert ───────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf"],
    help="Supported formats: PDF (more coming soon)",
)

if uploaded_file is not None:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info(f"📄 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")
    with col_btn:
        convert_btn = st.button("🚀 Convert", type="primary", use_container_width=True)

    if convert_btn:
        config = build_config()
        progress_bar = st.progress(0.0, text="Starting…")
        status_text = st.empty()

        def on_progress(stage: str, pct: float):
            progress_bar.progress(min(pct, 1.0), text=stage)

        try:
            result = run_pipeline(
                file_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name,
                config=config,
                skip_llm=skip_llm,
                skip_tts=skip_tts,
                progress_callback=on_progress,
            )

            progress_bar.progress(1.0, text="✅ Done!")

            # ── Results in tabs ────────────────────────────────────────
            tab_toc, tab_script, tab_md, tab_audio, tab_meta = st.tabs(
                ["📋 Table of Contents", "📝 Voice Script", "📄 Raw Markdown", "🔊 Audio", "ℹ️ Metadata"]
            )

            with tab_toc:
                toc = result.table_of_contents
                if toc:
                    lines: list[str] = []
                    for entry in toc:
                        indent = "&nbsp;" * 4 * (entry.level - 1)
                        mins = int(entry.timestamp_seconds) // 60
                        secs = int(entry.timestamp_seconds) % 60
                        timestamp_str = f"{mins:02d}:{secs:02d}"
                        lines.append(
                            f"{indent}{'◦' if entry.level > 1 else '•'} "
                            f"**{entry.title}** — `{timestamp_str}`"
                        )
                    st.markdown("\n\n".join(lines), unsafe_allow_html=True)
                else:
                    st.info("No headings found in the document.")

            with tab_script:
                st.text_area(
                    "Voice-ready script",
                    value=result.voice_script,
                    height=400,
                    label_visibility="collapsed",
                )
                st.download_button(
                    "⬇️ Download Script (.md)",
                    data=result.voice_script,
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_voice_script.md",
                    mime="text/markdown",
                )

            with tab_md:
                st.text_area(
                    "Extracted Markdown",
                    value=result.raw_markdown,
                    height=400,
                    label_visibility="collapsed",
                )
                st.download_button(
                    "⬇️ Download Markdown (.md)",
                    data=result.raw_markdown,
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.md",
                    mime="text/markdown",
                )

            with tab_audio:
                if result.audio_bytes:
                    fmt = result.audio_format
                    mime = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
                    st.audio(result.audio_bytes, format=mime)
                    st.download_button(
                        f"⬇️ Download Audio (.{fmt})",
                        data=result.audio_bytes,
                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.{fmt}",
                        mime=mime,
                    )
                else:
                    st.warning("Audio synthesis was skipped or failed.")

            with tab_meta:
                st.json(result.metadata)
                st.metric("Sections processed", result.sections_processed)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Pipeline failed: {e}")
            st.exception(e)

else:
    st.markdown(
        """
        ### How it works

        1. **Upload** a PDF document
        2. **Extract** structured Markdown from it
        3. **Transform** the Markdown into a natural spoken-word script
           - Rule-based cleanup (strip formatting, expand symbols)
           - LLM rewriting (bullet lists → prose, tables → narration, etc.)
        4. **Synthesize** audio from the voice script
        5. **Download** the script and/or audio

        ---

        Configure the **LLM model** and **TTS provider** in the sidebar.
        For local-only usage, set the model to `ollama/llama3` and TTS to `edge-tts`.
        """
    )
