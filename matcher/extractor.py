"""LLM extraction: raw text -> validated Pydantic model, with a self-correcting retry."""
import os
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from matcher.schemas import CVProfile, JobOffer
from matcher.tracing import trace_step

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash-lite"

T = TypeVar("T", bound=BaseModel)

COMMON_RULES = """Rules:
- Only use information present in the document. Never invent anything.
- Write ALL output in English, whatever the document language.
- Use standard English names for skills and technologies
  (e.g. 'REST APIs' not 'API REST', 'Gemini API' not 'API Gemini').
"""

CV_PROMPT = f"""You extract structured data from CVs written in French or English.

{COMMON_RULES}- Language levels must be one of: native, fluent, intermediate, basic.
- Dates as YYYY-MM when possible, or 'present' for ongoing roles.

CV:
"""

JOB_PROMPT = f"""You extract structured data from job offers written in French or English.

{COMMON_RULES}- required_skills and nice_to_have_skills contain ONLY technical skills, tools and
  technologies (e.g. 'Python', 'SQL', 'Pydantic v2'). Never put soft skills
  (leadership, teamwork, problem-solving) or spoken languages in them.
- Spoken languages go in 'languages'.
- A skill goes in required_skills only if the offer requires it; skills described
  as a plus, a bonus, an interest or "apprécié" go in nice_to_have_skills.
- Seniority must be one of: intern, junior, mid, senior (or null if unclear).

Job offer:
"""

RETRY_PROMPT = """

Your previous answer failed validation with these errors:
{errors}

Return corrected JSON that fixes these errors."""


def make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    return genai.Client(api_key=api_key)


def _usage(response) -> dict[str, int]:
    """Token counts in Langfuse's format (each token counted in exactly one bucket)."""
    meta = response.usage_metadata
    if meta is None:
        return {}
    usage = {
        "input": meta.prompt_token_count,
        "output": meta.candidates_token_count,
        "output_reasoning": getattr(meta, "thoughts_token_count", None),
    }
    return {k: v for k, v in usage.items() if v}


class Extractor:
    """Extracts validated Pydantic models from raw text using Gemini."""

    def __init__(self, client: genai.Client | None = None, model: str | None = None):
        self.client = client or make_client()
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    def _call(self, contents: str, schema: type[BaseModel], attempt: int) -> str:
        """Sends one request, traced as an LLM generation (input, output, tokens, time)."""
        with trace_step(
            f"gemini-{schema.__name__}",
            as_type="generation",
            model=self.model,
            input=contents,
            model_parameters={"temperature": 0},
            metadata={"attempt": attempt + 1},
        ) as generation:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            if not response.text:
                raise RuntimeError("The model returned an empty response.")
            generation.update(output=response.text, usage_details=_usage(response))
            return response.text

    def extract(self, prompt: str, text: str, schema: type[T], max_retries: int = 1) -> T:
        """Extracts and validates text into `schema`.

        If validation fails, the errors are sent back to the model so it can
        correct its output, up to `max_retries` times. Failed attempts are
        flagged as warnings in the trace.
        """
        base_prompt = prompt + text
        contents = base_prompt

        with trace_step(f"extract-{schema.__name__}", input={"characters": len(text)}) as span:
            for attempt in range(max_retries + 1):
                raw = self._call(contents, schema, attempt)
                try:
                    result = schema.model_validate_json(raw)
                    span.update(output=result.model_dump(), metadata={"attempts": attempt + 1})
                    return result
                except ValidationError as e:
                    span.update(
                        level="WARNING",
                        status_message=f"Attempt {attempt + 1} failed validation "
                        f"({e.error_count()} errors)",
                    )
                    if attempt == max_retries:
                        raise
                    contents = base_prompt + RETRY_PROMPT.format(errors=e)

    def extract_cv(self, cv_text: str) -> CVProfile:
        return self.extract(CV_PROMPT, cv_text, CVProfile)

    def extract_job(self, job_text: str) -> JobOffer:
        return self.extract(JOB_PROMPT, job_text, JobOffer)
