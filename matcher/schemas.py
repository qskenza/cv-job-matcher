"""Pydantic v2 models: the contract every LLM output must satisfy."""
from pydantic import BaseModel, Field, field_validator

LANGUAGE_LEVELS = {"native", "fluent", "intermediate", "basic"}


def _clean_list(items: list[str]) -> list[str]:
    """Strip, drop empties, dedupe case-insensitively, keep order."""
    seen, out = set(), []
    for item in (x.strip() for x in items):
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def _check_doc_language(v: str) -> str:
    v = v.strip().lower()[:2]
    if v not in {"fr", "en"}:
        raise ValueError("language must be 'fr' or 'en'")
    return v


# ---------- CV ----------

class Experience(BaseModel):
    title: str = Field(description="Job title, e.g. 'Data Analyst Intern'")
    company: str
    start_date: str | None = Field(default=None, description="Format YYYY-MM if known")
    end_date: str | None = Field(default=None, description="Format YYYY-MM, or 'present'")
    description: str | None = Field(default=None, description="One-sentence summary")


class Project(BaseModel):
    name: str
    description: str | None = Field(default=None, description="One-sentence summary, in English")
    technologies: list[str] = []


class Education(BaseModel):
    degree: str
    institution: str
    graduation_year: int | None = None


class Language(BaseModel):
    name: str = Field(description="Language name in English, e.g. 'French'")
    level: str = Field(description="One of: native, fluent, intermediate, basic")

    @field_validator("level")
    @classmethod
    def normalize_level(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in LANGUAGE_LEVELS:
            raise ValueError(f"level must be one of {sorted(LANGUAGE_LEVELS)}, got '{v}'")
        return v


class CVProfile(BaseModel):
    full_name: str
    email: str | None = None
    cv_language: str = Field(description="Language the CV is written in: 'fr' or 'en'")
    summary: str | None = Field(default=None, description="2-sentence summary, in English")
    skills: list[str] = Field(description="Technical skills, e.g. 'Python', 'SQL'")
    experiences: list[Experience] = []
    projects: list[Project] = []
    education: list[Education] = []
    languages: list[Language] = []

    @field_validator("skills")
    @classmethod
    def clean_skills(cls, v: list[str]) -> list[str]:
        return _clean_list(v)

    @field_validator("cv_language")
    @classmethod
    def check_lang(cls, v: str) -> str:
        return _check_doc_language(v)

    def all_skills(self) -> list[str]:
        """Skills section plus every technology used in projects."""
        techs = [t for p in self.projects for t in p.technologies]
        return _clean_list(self.skills + techs)


# ---------- Job offer ----------

class JobOffer(BaseModel):
    title: str
    company: str | None = None
    location: str | None = None
    offer_language: str = Field(description="Language the offer is written in: 'fr' or 'en'")
    seniority: str | None = Field(default=None, description="One of: intern, junior, mid, senior")
    required_skills: list[str] = Field(
        description="Technical skills, tools and technologies explicitly required. "
        "No soft skills, no spoken languages."
    )
    nice_to_have_skills: list[str] = Field(
        default=[], description="Technical skills listed as a plus or an interest"
    )
    languages: list[str] = Field(default=[], description="Spoken languages required, in English")
    summary: str | None = Field(default=None, description="2-sentence summary of the role, in English")

    @field_validator("required_skills", "nice_to_have_skills")
    @classmethod
    def clean_skills(cls, v: list[str]) -> list[str]:
        return _clean_list(v)

    @field_validator("offer_language")
    @classmethod
    def check_lang(cls, v: str) -> str:
        return _check_doc_language(v)
