from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def _extract(reader: PdfReader, name: str) -> str:
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise ValueError(f"No text found in {name} (scanned PDF?)")
    return text


def read_pdf_text(path: str | Path) -> str:
    """Reads the text of a PDF file on disk."""
    return _extract(PdfReader(str(path)), str(path))


def read_pdf_bytes(data: bytes) -> str:
    """Reads the text of a PDF held in memory (e.g. an upload)."""
    return _extract(PdfReader(BytesIO(data)), "the uploaded PDF")
