"""Ranks all parsed job offers for one parsed CV.

Usage: python -m scripts.rank_jobs data/output/cv_eng.json [--threshold 0.9] [--debug]
  --threshold  similarity needed for two skills to count as the same
  --debug      also shows the closest CV skill for every missing skill
"""
import argparse
from pathlib import Path

from matcher import tracing
from matcher.embeddings import Embedder
from matcher.ranking import SKILL_MATCH_THRESHOLD, rank_jobs
from matcher.schemas import CVProfile, JobOffer
from matcher.vector_index import faiss

JOBS_DIR = Path("data/output/jobs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cv_json")
    parser.add_argument("--threshold", type=float, default=SKILL_MATCH_THRESHOLD)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cv = CVProfile.model_validate_json(Path(args.cv_json).read_text(encoding="utf-8"))
    jobs = {
        f.stem: JobOffer.model_validate_json(f.read_text(encoding="utf-8"))
        for f in sorted(JOBS_DIR.glob("*.json"))
    }
    if not jobs:
        raise SystemExit("No parsed jobs found. Run: python -m scripts.parse_jobs")

    embedder = Embedder()
    try:
        with tracing.trace_step(
            "rank-jobs",
            input={"cv": cv.full_name, "jobs": list(jobs), "threshold": args.threshold},
        ) as run:
            results = rank_jobs(cv, jobs, embedder.embed, args.threshold)
            run.update(output=[
                {"job": r.job_id, "score": round(r.score, 3), "semantic": round(r.semantic, 3),
                 "coverage": round(r.coverage, 3), "missing": r.missing}
                for r in results
            ])
    finally:
        tracing.flush()

    print(f"Ranking {len(jobs)} offers for {cv.full_name} "
          f"(backend: {'faiss' if faiss else 'numpy'}, threshold: {args.threshold})\n")
    for rank, r in enumerate(results, 1):
        print(f"{rank}. {r.job.title} @ {r.job.company or '?'}  [{r.job_id}]")
        print(f"   score {r.score:.2f} | semantic {r.semantic:.2f} | "
              f"skills {len(r.matched)}/{len(r.job.required_skills)}")
        for m in r.matched:
            label = m.job_skill if m.job_skill == m.cv_skill else f"{m.job_skill} ~ {m.cv_skill}"
            print(f"   + {label} ({m.similarity:.2f})")
        if r.missing and args.debug:
            import numpy as np
            cv_skills = cv.all_skills()
            cv_vecs = embedder.embed(cv_skills)
            miss_vecs = embedder.embed(r.missing)
            for skill, vec in zip(r.missing, miss_vecs):
                sims = cv_vecs @ vec
                best = int(np.argmax(sims))
                print(f"   - {skill} (closest: {cv_skills[best]}, {sims[best]:.2f})")
        elif r.missing:
            print("   - missing: " + ", ".join(r.missing))
        print()
