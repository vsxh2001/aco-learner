from app.converters.pdf_to_md import PdfToMarkdown

CONVERTERS: dict[str, type] = {
    ".pdf": PdfToMarkdown,
}


def get_converter(extension: str):
    """Return the converter class for a given file extension."""
    ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    converter_cls = CONVERTERS.get(ext)
    if converter_cls is None:
        raise ValueError(f"No converter registered for '{ext}'. Supported: {list(CONVERTERS.keys())}")
    return converter_cls()
