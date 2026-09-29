"""Extracts every data/jobs/*.txt offer into data/output/jobs/*.json.

Already-parsed offers are skipped, so you only pay for new ones.
Usage: python -m scripts.parse_jobs  (add --force to re-parse everything)
"""
import sys
from pathlib import Path

from matcher import tracing
from matcher.extractor import Extractor

JOBS_DIR = Path("data/jobs")
OUT_DIR = Path("data/output/jobs")

if __name__ == "__main__":
    force = "--force" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(JOBS_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {JOBS_DIR}")

    extractor = Extractor()
    try:
        with tracing.trace_step("parse-jobs", input={"files": [f.name for f in files]}) as run:
            parsed = 0
            for txt in files:
                out = OUT_DIR / f"{txt.stem}.json"
                if out.exists() and not force:
                    print(f"skip   {txt.name} (already parsed)")
                    continue
                job = extractor.extract_job(txt.read_text(encoding="utf-8"))
                out.write_text(job.model_dump_json(indent=2), encoding="utf-8")
                print(f"parsed {txt.name} -> {job.title} @ {job.company}")
                parsed += 1
            run.update(output={"parsed": parsed, "skipped": len(files) - parsed})
    finally:
        tracing.flush()
