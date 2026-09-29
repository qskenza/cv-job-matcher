"""Ranks job offers for a CV with an explainable score.

score = SEMANTIC_WEIGHT * semantic + (1 - SEMANTIC_WEIGHT) * coverage
- semantic: cosine similarity between the whole CV and the whole offer (FAISS search)
- coverage: share of the offer's required skills that match a CV skill, where two
  skills match if their embeddings are close ("JWT" ~ "JWT Authentication")
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from matcher.schemas import CVProfile, JobOffer
from matcher.vector_index import VectorIndex

EmbedFn = Callable[[list[str]], np.ndarray]

SKILL_MATCH_THRESHOLD = 0.90  # calibrated on real results: see README Findings
SEMANTIC_WEIGHT = 0.4


@dataclass
class SkillMatch:
    job_skill: str
    cv_skill: str
    similarity: float


@dataclass
class MatchResult:
    job_id: str
    job: JobOffer
    semantic: float
    coverage: float
    score: float
    matched: list[SkillMatch] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def cv_to_text(cv: CVProfile) -> str:
    parts = [cv.summary or ""]
    parts += [f"{e.title}: {e.description or ''}" for e in cv.experiences]
    parts += [f"{p.name}: {p.description or ''}" for p in cv.projects]
    parts.append("Skills: " + ", ".join(cv.all_skills()))
    return "\n".join(parts)


def job_to_text(job: JobOffer) -> str:
    parts = [job.title, job.summary or ""]
    parts.append("Required skills: " + ", ".join(job.required_skills))
    if job.nice_to_have_skills:
        parts.append("Nice to have: " + ", ".join(job.nice_to_have_skills))
    return "\n".join(parts)


def skill_coverage(
    job_skills: list[str],
    cv_skills: list[str],
    vectors: dict[str, np.ndarray],
    threshold: float = SKILL_MATCH_THRESHOLD,
) -> tuple[list[SkillMatch], list[str]]:
    """For each job skill, finds the closest CV skill and keeps it if above threshold."""
    if not job_skills:
        return [], []
    if not cv_skills:
        return [], list(job_skills)

    cv_matrix = np.stack([vectors[s] for s in cv_skills])
    matched, missing = [], []
    for skill in job_skills:
        sims = cv_matrix @ vectors[skill]
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            matched.append(SkillMatch(skill, cv_skills[best], float(sims[best])))
        else:
            missing.append(skill)
    return matched, missing


def rank_jobs(
    cv: CVProfile,
    jobs: dict[str, JobOffer],
    embed: EmbedFn,
    threshold: float = SKILL_MATCH_THRESHOLD,
) -> list[MatchResult]:
    job_ids = list(jobs)
    cv_skills = cv.all_skills()

    # 1. Embed every unique skill once, in a single batch
    all_skills = list(dict.fromkeys(cv_skills + [s for j in jobs.values() for s in j.required_skills]))
    skill_vectors = dict(zip(all_skills, embed(all_skills))) if all_skills else {}

    # 2. Semantic search: whole CV against every whole offer
    doc_vectors = embed([cv_to_text(cv)] + [job_to_text(jobs[i]) for i in job_ids])
    index = VectorIndex(doc_vectors[1:])
    semantic = {job_ids[i]: sim for i, sim in index.search(doc_vectors[0], k=len(job_ids))}

    # 3. Combine with skill coverage
    results = []
    for job_id in job_ids:
        job = jobs[job_id]
        matched, missing = skill_coverage(job.required_skills, cv_skills, skill_vectors, threshold)
        coverage = len(matched) / len(job.required_skills) if job.required_skills else 0.0
        score = SEMANTIC_WEIGHT * semantic[job_id] + (1 - SEMANTIC_WEIGHT) * coverage
        results.append(MatchResult(job_id, job, semantic[job_id], coverage, score, matched, missing))

    return sorted(results, key=lambda r: r.score, reverse=True)
