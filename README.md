# Multilingual CV–Job Matcher

Matches CVs (French/English) to job offers using LLM extraction, Pydantic v2 validation, embeddings and Langfuse tracing.

## Status
- [x] Step 1: CV parsing → validated Pydantic v2 models (with self-correcting retry)
- [ ] Step 2: Embeddings + ranking (FAISS)
- [ ] Step 3: Langfuse tracing
- [ ] Step 4: Streamlit demo

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then add your Gemini key
```

## Usage
```bash
python -m scripts.parse_cv data/cvs/your_cv.pdf
pytest
```
## Findings
- Parsing the same CV in French and English initially gave different skill names
    ("API REST" vs "REST APIs") and French job titles. Instructing the model to
    normalize all output to English fixed most of it.
- Small variations remain ("JWT" vs "JWT Authentication"), which is why ranking
    uses embeddings rather than exact keyword matching.