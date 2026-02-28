from app.synthesizers.edge import EdgeTTSSynthesizer

SYNTHESIZERS: dict[str, type] = {
    "edge-tts": EdgeTTSSynthesizer,
}

# Optional providers — register only if their deps are installed
try:
    from app.synthesizers.kokoro_synth import KokoroSynthesizer
    SYNTHESIZERS["kokoro"] = KokoroSynthesizer
except ImportError:
    pass

try:
    from app.synthesizers.openai_tts import OpenAITTSSynthesizer
    SYNTHESIZERS["openai-tts"] = OpenAITTSSynthesizer
except ImportError:
    pass


def get_synthesizer(provider: str):
    """Return the synthesizer instance for a given provider name."""
    synth_cls = SYNTHESIZERS.get(provider)
    if synth_cls is None:
        raise ValueError(f"Unknown TTS provider '{provider}'. Available: {list(SYNTHESIZERS.keys())}")
    return synth_cls()
