"""Streamlit demo: upload a CV, see job offers ranked with matched and missing skills.

Run: streamlit run app.py
"""
import re
from pathlib import Path

import streamlit as st

from matcher import tracing
from matcher.embeddings import Embedder
from matcher.extractor import Extractor
from matcher.pdf_reader import read_pdf_bytes
from matcher.ranking import SEMANTIC_WEIGHT, SKILL_MATCH_THRESHOLD, MatchResult, rank_jobs
from matcher.schemas import CVProfile, JobOffer

OUTPUT_DIR = Path("data/output")
JOBS_DIR = OUTPUT_DIR / "jobs"

st.set_page_config(page_title="CV–Job Matcher", page_icon="🎯", layout="wide")


# ---------- Cached resources: created once, reused across reruns ----------

@st.cache_resource
def get_extractor() -> Extractor:
    return Extractor()


@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_data(show_spinner=False)
def parse_cv(pdf_bytes: bytes) -> dict:
    """Parses an uploaded CV once; the same file is never sent to the LLM twice."""
    try:
        with tracing.trace_step("app-parse-cv"):
            profile = get_extractor().extract_cv(read_pdf_bytes(pdf_bytes))
    finally:
        tracing.flush()
    return profile.model_dump()


# ---------- Data helpers ----------

def load_jobs() -> dict[str, JobOffer]:
    return {
        f.stem: JobOffer.model_validate_json(f.read_text(encoding="utf-8"))
        for f in sorted(JOBS_DIR.glob("*.json"))
    }


def parsed_cvs() -> list[Path]:
    return sorted(p for p in OUTPUT_DIR.glob("*.json") if p.name != "embedding_cache.json")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "offer"


def add_job(text: str) -> JobOffer:
    try:
        with tracing.trace_step("app-add-job"):
            job = get_extractor().extract_job(text)
    finally:
        tracing.flush()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = JOBS_DIR / f"{slugify(f'{job.company or ''} {job.title}')}.json"
    path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    return job


def badges(labels: list[str], color: str) -> str:
    return " ".join(f":{color}-background[{label}]" for label in labels)


def show_result(rank: int, r: MatchResult) -> None:
    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.subheader(f"{rank}. {r.job.title}")
        details = [r.job.company, r.job.location, r.job.seniority]
        left.caption(" · ".join(d for d in details if d))
        right.metric("Score", f"{r.score:.0%}")
        st.progress(min(max(r.score, 0.0), 1.0))

        required = len(r.job.required_skills)
        st.caption(f"Semantic fit {r.semantic:.2f} · Required skills matched {len(r.matched)}/{required}")

        if r.matched:
            labels = [m.job_skill if m.job_skill == m.cv_skill else f"{m.job_skill} ≈ {m.cv_skill}"
                      for m in r.matched]
            st.markdown("**Matched:** " + badges(labels, "green"))
        if r.missing:
            st.markdown("**Missing:** " + badges(r.missing, "red"))
        if not required:
            st.info("This offer lists no technical skills, so only the overall fit counts.")


# ---------- Page ----------

st.title("🎯 CV–Job Matcher")
st.caption("French or English CVs and job offers, parsed with Gemini, validated with Pydantic v2, "
           "ranked with embeddings and FAISS.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Skill match threshold", 0.80, 0.95, SKILL_MATCH_THRESHOLD, 0.01,
        help="How similar two skill names must be to count as the same skill. "
             "0.90 was calibrated on real data.",
    )
    st.markdown(f"**Score** = {SEMANTIC_WEIGHT:.0%} semantic fit + {1 - SEMANTIC_WEIGHT:.0%} "
                "required-skill coverage")
    st.caption("Tracing: " + ("on (Langfuse)" if tracing.enabled() else "off"))

cv_col, jobs_col = st.columns(2)

with cv_col:
    st.header("1. Your CV")
    upload = st.file_uploader("Upload a CV (PDF, French or English)", type="pdf")
    existing = parsed_cvs()
    choice = None
    if not upload and existing:
        choice = st.selectbox("…or use a CV you already parsed", existing, format_func=lambda p: p.name)

    cv = None
    try:
        if upload:
            with st.spinner("Reading your CV with Gemini…"):
                cv = CVProfile.model_validate(parse_cv(upload.getvalue()))
        elif choice:
            cv = CVProfile.model_validate_json(choice.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Could not read this CV: {e}")

    if cv:
        st.success(f"{cv.full_name} · CV in {cv.cv_language.upper()} · {len(cv.all_skills())} skills found")
        with st.expander("See the extracted profile"):
            st.markdown(badges(cv.all_skills(), "blue"))
            st.json(cv.model_dump(), expanded=False)

with jobs_col:
    st.header("2. Job offers")
    jobs = load_jobs()
    st.write(f"{len(jobs)} offers loaded")
    for job in jobs.values():
        st.markdown(f"- {job.title} · {job.company or '?'}")

    with st.expander("Add a job offer"):
        text = st.text_area("Paste the full offer (French or English)", height=200)
        if st.button("Parse and add", disabled=not text.strip()):
            try:
                with st.spinner("Extracting the offer with Gemini…"):
                    job = add_job(text)
                st.success(f"Added: {job.title} · {job.company or '?'}")
                st.rerun()
            except Exception as e:
                st.error(f"Could not parse this offer: {e}")

st.divider()
st.header("3. Ranking")

if not cv:
    st.info("Upload or pick a CV to see the ranking.")
elif not jobs:
    st.info("Add at least one job offer to see the ranking.")
else:
    try:
        with st.spinner("Comparing your CV with each offer…"):
            try:
                with tracing.trace_step("app-rank-jobs", input={"cv": cv.full_name, "jobs": list(jobs)}):
                    results = rank_jobs(cv, jobs, get_embedder().embed, threshold)
            finally:
                tracing.flush()
        for rank, r in enumerate(results, 1):
            show_result(rank, r)
    except Exception as e:
        st.error(f"Ranking failed: {e}")
