"""Usage: python -m scripts.parse_cv data/cvs/my_cv.pdf"""
import sys
from pathlib import Path

from matcher import tracing
from matcher.extractor import Extractor
from matcher.pdf_reader import read_pdf_text

if __name__ == "__main__":
    pdf = Path(sys.argv[1])
    try:
        with tracing.trace_step("parse-cv", input={"file": pdf.name}):
            profile = Extractor().extract_cv(read_pdf_text(pdf))
    finally:
        tracing.flush()

    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pdf.stem}.json"
    out.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    print(profile.model_dump_json(indent=2))
    print(f"\nSaved to {out}")
