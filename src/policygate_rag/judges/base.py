"""The Judge interface: a relevance score per retrieved candidate.

Pure Python. API-backed judges keep their network calls in a ``*_client.py``
module (hard constraint 7); the judge itself only turns a query and candidates
into scores, so the offline benchmark in stage 1 can run every judge on the
same inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from policygate_rag.index.corpus import Document


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc: Document
    fused_score: float
    """The candidate's RRF score from the round that retrieved it."""


class Judgement(BaseModel):
    model_config = ConfigDict(frozen=True)

    scores: tuple[float, ...]
    """One per candidate, in order. Probabilities where the judge produces them."""
    usd: float = 0.0
    is_probability: bool = False
    """Whether ``scores`` are relevance probabilities, so a threshold applies."""


class Judge(Protocol):
    name: str

    def judge(self, query: str, candidates: Sequence[Candidate]) -> Judgement: ...
