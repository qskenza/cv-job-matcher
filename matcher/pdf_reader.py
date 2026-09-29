from pathlib import Path
from pypdf import PdfReader


def read_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise ValueError(f"No text found in {path} (scanned PDF?)")
    return text
