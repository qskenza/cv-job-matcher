"""Text embeddings with Gemini, L2-normalized so dot product = cosine similarity.

Every embedding is cached on disk, so each text is only sent to the API once.
This keeps reruns free and fast, and avoids hitting the free-tier rate limit.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types

from matcher.extractor import make_client

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_CACHE = Path("data/output/embedding_cache.json")
BATCH_SIZE = 50
MAX_RETRIES = 3


class Embedder:
    def __init__(
        self,
        client: genai.Client | None = None,
        model: str | None = None,
        dim: int = 768,
        cache_path: Path | None = DEFAULT_CACHE,
    ):
        self.client = client or make_client()
        self.model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.dim = dim
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _key(self, text: str) -> str:
        # Vectors from different models or sizes are not comparable
        return f"{self.model}|{self.dim}|{text}"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns a (len(texts), dim) float32 array of unit vectors."""
        new = list(dict.fromkeys(t for t in texts if self._key(t) not in self.cache))
        for i in range(0, len(new), BATCH_SIZE):
            batch = new[i : i + BATCH_SIZE]
            for text, vector in zip(batch, self._embed_batch(batch)):
                self.cache[self._key(text)] = vector
        if new:
            self._save()
        vectors = np.array([self.cache[self._key(t)] for t in texts], dtype=np.float32)
        return normalize(vectors)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Calls the API, waiting and retrying if the rate limit is hit."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=self.dim,
                    ),
                )
                return [e.values for e in result.embeddings]
            except errors.ClientError as e:
                if e.code != 429 or attempt == MAX_RETRIES:
                    raise
                print(f"Rate limit hit, waiting 60s (retry {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(60)

    def _save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)