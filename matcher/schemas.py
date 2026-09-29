"""Pydantic v2 models: the contract every LLM output must satisfy."""
from pydantic import BaseModel, Field, field_validator


class Experience(BaseModel):
    title: str = Field(description="Job title, e.g. 'Data Analyst Intern'")
    company: str
    start_date: str | None = Field(default=None, description="Format YYYY-MM if known")
    end_date: str | None = Field(default=None, description="Format YYYY-MM, or 'present'")
    description: str | None = Field(default=None, description="One-sentence summary")


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
        allowed = {"native", "fluent", "intermediate", "basic"}
        if v not in allowed:
            raise ValueError(f"level must be one of {sorted(allowed)}, got '{v}'")
        return v

class Project(BaseModel):
    name: str
    description: str | None = Field(default=None, description="One-sentence summary, in English")
    technologies: list[str] = []

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
        # Strip, drop empties, dedupe case-insensitively, keep order
        seen, out = set(), []
        for s in (x.strip() for x in v):
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    @field_validator("cv_language")
    @classmethod
    def check_lang(cls, v: str) -> str:
        v = v.strip().lower()[:2]
        if v not in {"fr", "en"}:
            raise ValueError("cv_language must be 'fr' or 'en'")
        return v
