import pytest
from pydantic import ValidationError

from matcher.schemas import CVProfile

VALID = {
    "full_name": "Test User",
    "cv_language": "FR",
    "skills": [" Python", "python", "SQL", ""],
    "languages": [{"name": "French", "level": "Fluent"}],
}


def test_skills_are_cleaned_and_deduped():
    cv = CVProfile.model_validate(VALID)
    assert cv.skills == ["Python", "SQL"]
    assert cv.cv_language == "fr"
    assert cv.languages[0].level == "fluent"


def test_invalid_language_level_rejected():
    bad = {**VALID, "languages": [{"name": "French", "level": "C1-ish"}]}
    with pytest.raises(ValidationError):
        CVProfile.model_validate(bad)


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        CVProfile.model_validate({"cv_language": "en", "skills": []})
