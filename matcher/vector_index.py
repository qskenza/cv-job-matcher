"""Cosine-similarity search with FAISS, falling back to NumPy if FAISS isn't installed."""
import numpy as np

try:
    import faiss
except ImportError:  # e.g. no wheel for your Python version
    faiss = None


class VectorIndex:
    """Exact inner-product index. Vectors must be L2-normalized."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.backend = "faiss" if faiss else "numpy"
        if faiss:
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Returns the top-k (row index, cosine similarity) pairs, best first."""
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        k = min(k, len(self.vectors))
        if faiss:
            scores, ids = self.index.search(query, k)
            return list(zip(ids[0].tolist(), scores[0].tolist()))
        scores = (self.vectors @ query.T).ravel()
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]
