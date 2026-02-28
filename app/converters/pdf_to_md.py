"""Convert PDF documents to Markdown using pymupdf4llm."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pymupdf4llm

from app.converters.base import BaseConverter


class PdfToMarkdown(BaseConverter):
    """Extract structured Markdown from a PDF file."""

    def convert(self, data: bytes, filename: str = "") -> str:
        """
        Write bytes to a temp file, then use pymupdf4llm to extract Markdown.
        """
        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            md_text: str = pymupdf4llm.to_markdown(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not md_text or not md_text.strip():
            raise ValueError(f"No text could be extracted from '{filename or 'the PDF'}'. "
                             "It may be image-only or encrypted.")

        return md_text
