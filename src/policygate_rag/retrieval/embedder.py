"""Embedding calls, with their cost. The one place this project calls an embedder.

``index/`` holds vectors and never calls an API (hard constraint 7); this module
produces the vectors. Every call reports what it cost, and an unknown cost is
an error rather than a zero: the ledger books what is spent, and spend it
cannot see would make the budget guard optimistic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class Embedding(BaseModel):
    model_config = ConfigDict(frozen=True)

    vectors: list[list[float]]
    tokens: int
    usd: float


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> Embedding: ...


class UnknownEmbeddingCost(Exception):
    pass


class LiteLLMEmbedder:
    """``litellm.embedding`` in batches, keeping input order and summing cost."""

    def __init__(self, model: str, *, batch_size: int = 64) -> None:
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> Embedding:
        import litellm

        vectors: list[list[float]] = []
        tokens, usd = 0, 0.0
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            response = litellm.embedding(model=self.model, input=batch)
            rows = sorted(response.data, key=lambda r: r["index"])
            if len(rows) != len(batch):
                raise ValueError(f"asked for {len(batch)} embeddings, got {len(rows)}")
            vectors.extend(r["embedding"] for r in rows)
            tokens += response.usage.prompt_tokens
            cost = (response._hidden_params or {}).get("response_cost")
            if cost is None:
                raise UnknownEmbeddingCost(f"no cost reported for {self.model}")
            usd += cost
        return Embedding(vectors=vectors, tokens=tokens, usd=usd)
