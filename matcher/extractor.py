"""LLM extraction: CV text -> validated CVProfile, with a self-correcting retry."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from matcher.schemas import CVProfile

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash-lite"

PROMPT = """You extract structured data from CVs written in French or English.

Rules:
- Only use information present in the CV. Never invent anything.
- Write ALL output in English, whatever the CV language: summary, descriptions,
  job titles, degree names, institution names and project names.
- Use standard English names for skills and technologies
  (e.g. 'REST APIs' not 'API REST', 'Gemini API' not 'API Gemini').
- Language levels must be one of: native, fluent, intermediate, basic.
- Dates as YYYY-MM when possible, or 'present' for ongoing roles.

CV:
"""

RETRY_PROMPT = """

Your previous answer failed validation with these errors:
{errors}

Return corrected JSON that fixes these errors."""


class CVExtractor:
    """Extracts a validated CVProfile from raw CV text using Gemini."""

    def __init__(self, model: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    def _call(self, contents: str) -> str:
        """Sends one request and returns the raw JSON text."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CVProfile,
                temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        if not response.text:
            raise RuntimeError("The model returned an empty response.")
        return response.text

    def extract(self, cv_text: str, max_retries: int = 1) -> CVProfile:
        """Extracts and validates a CV.

        If validation fails, the errors are sent back to the model so it can
        correct its output, up to `max_retries` times.
        """
        base_prompt = PROMPT + cv_text
        contents = base_prompt

        for attempt in range(max_retries + 1):
            raw = self._call(contents)
            try:
                return CVProfile.model_validate_json(raw)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                contents = base_prompt + RETRY_PROMPT.format(errors=e)