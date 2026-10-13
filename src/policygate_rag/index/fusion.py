"""Reciprocal rank fusion, and the hybrid search built on it.

RRF scores a document by ``sum over lists of 1 / (k + rank)``, rank starting
at 1. It uses ranks only, so BM25 scores and cosine similarities, which live on
unrelated scales, never have to be calibrated against each other. Ties are
broken by best single rank, then by document id, so the fused order is a
function of the input rankings alone.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from policygate_rag.index.bm25 import BM25Index
from policygate_rag.index.dense import DenseIndex


class Fused(BaseModel):
    """One fused result, with where it came from (for the retrieval trace)."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    score: float
    ranks: dict[str, int]
    """Rank in each source list that contained it, e.g. ``{"bm25": 3, "dense": 1}``."""


def rrf(rankings: dict[str, Sequence[str]], *, k: int = 60) -> list[Fused]:
    if k < 1:
        raise ValueError("the RRF constant must be at least 1")
    ranks: dict[str, dict[str, int]] = {}
    for source, ids in rankings.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"{source} ranks a document twice")
        for rank, doc_id in enumerate(ids, start=1):
            ranks.setdefault(doc_id, {})[source] = rank
    fused = [
        Fused(doc_id=d, score=sum(1 / (k + r) for r in rs.values()), ranks=rs)
        for d, rs in ranks.items()
    ]
    fused.sort(key=lambda f: (-f.score, min(f.ranks.values()), f.doc_id))
    return fused


class HybridIndex:
    """BM25 and dense search over the same corpus, fused with RRF."""

    def __init__(self, bm25: BM25Index, dense: DenseIndex) -> None:
        if bm25.doc_ids != dense.doc_ids:
            raise ValueError("BM25 and dense indexes cover different documents")
        self.bm25, self.dense = bm25, dense

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        k_bm25: int,
        k_dense: int,
        rrf_k: int,
    ) -> list[Fused]:
        return rrf(
            {
                "bm25": [d for d, _ in self.bm25.search(query, k_bm25)],
                "dense": [d for d, _ in self.dense.search(query_vector, k_dense)],
            },
            k=rrf_k,
        )
