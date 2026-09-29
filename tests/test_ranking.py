"""Ranking tests with a fake embedder, so they run without any API call."""
import numpy as np

from matcher.embeddings import normalize
from matcher.ranking import rank_jobs, skill_coverage
from matcher.schemas import CVProfile, JobOffer
from matcher.vector_index import VectorIndex

# Fake embedder: texts sharing the same first word get the same vector,
# so "JWT" and "JWT Authentication" count as the same skill.
VOCAB = ["python", "jwt", "sql", "react", "kubernetes", "fastapi"]


def fake_embed(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for i, text in enumerate(texts):
        words = text.lower().replace(",", " ").replace(":", " ").split()
        for w in words:
            if w in VOCAB:
                vectors[i, VOCAB.index(w)] += 1
        if not vectors[i].any():
            vectors[i, 0] = 0.01
    return normalize(vectors)


CV = CVProfile(full_name="Test", cv_language="en", skills=["Python", "JWT Authentication", "SQL"])


def job(title: str, skills: list[str]) -> JobOffer:
    return JobOffer(title=title, offer_language="en", required_skills=skills)


def test_similar_skill_names_match():
    skills = ["JWT", "Kubernetes", "JWT Authentication", "Python", "SQL"]
    vectors = dict(zip(skills, fake_embed(skills)))
    matched, missing = skill_coverage(["JWT", "Kubernetes"], ["JWT Authentication", "Python", "SQL"], vectors)
    assert [(m.job_skill, m.cv_skill) for m in matched] == [("JWT", "JWT Authentication")]
    assert missing == ["Kubernetes"]


def test_best_fitting_job_ranks_first():
    jobs = {
        "frontend": job("Frontend dev", ["React", "Kubernetes"]),
        "backend": job("Backend dev", ["Python", "SQL", "JWT"]),
    }
    results = rank_jobs(CV, jobs, fake_embed)
    assert results[0].job_id == "backend"
    assert results[0].coverage == 1.0
    assert results[1].missing == ["React", "Kubernetes"]


def test_vector_index_returns_best_first():
    vectors = normalize(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    hits = VectorIndex(vectors).search(np.array([1, 0], dtype=np.float32), k=3)
    assert [i for i, _ in hits] == [0, 2, 1]
