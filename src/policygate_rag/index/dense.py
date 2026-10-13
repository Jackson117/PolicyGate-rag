"""Dense retrieval: cosine similarity over precomputed document embeddings.

The index holds vectors; it never calls an embedding API (hard constraint 7).
Vectors arrive from the embedder in ``retrieval/``, are cached on disk with
the model name and the corpus hash, and a cache whose model or corpus does not
match is refused rather than silently reused.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class DenseIndex:
    def __init__(self, doc_ids: Sequence[str], vectors: np.ndarray, *, model: str) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(doc_ids):
            raise ValueError(f"{len(doc_ids)} ids but vectors of shape {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("a document vector is all zeros")
        self.doc_ids = tuple(doc_ids)
        self.model = model
        self._unit = vectors / norms

    @property
    def dimensions(self) -> int:
        return self._unit.shape[1]

    def search(self, query_vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        """Top ``k`` documents by cosine similarity, best first; ties by corpus order."""
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape != (self.dimensions,):
            raise ValueError(f"query has shape {q.shape}, index has {self.dimensions} dims")
        norm = float(np.linalg.norm(q))
        if norm == 0:
            raise ValueError("query vector is all zeros")
        sims = self._unit @ (q / norm)
        order = np.lexsort((np.arange(len(sims)), -sims))[:k]
        return [(self.doc_ids[i], float(sims[i])) for i in order]

    # --- the on-disk cache ---------------------------------------------------

    def save(self, path: Path, *, corpus_hash: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self._unit)
        path.with_suffix(".json").write_text(json.dumps(
            {"model": self.model, "corpus_hash": corpus_hash, "doc_ids": list(self.doc_ids)},
            indent=1,
        ))

    @classmethod
    def load(cls, path: Path, *, model: str, corpus_hash: str) -> DenseIndex:
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["model"] != model:
            raise ValueError(f"cache built with {meta['model']}, wanted {model}")
        if meta["corpus_hash"] != corpus_hash:
            raise ValueError("cache was built from a different corpus")
        return cls(meta["doc_ids"], np.load(path.with_suffix(".npy")), model=model)
