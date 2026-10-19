"""JudgeKind.JEV: one Noul question per query-candidate pair.

Following TypeSafe's own re-ranking recipe: each request carries one pair, so no
candidate is a distractor for another (Jev 1.13's documented weakness is
accuracy loss from unrelated content in the state). The Noul is P(yes) that the
document holds information the request needs, so ``judge_threshold`` applies
to it directly.

Pure Python: the network is in ``jev_client.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from policygate_rag.judges.base import Candidate, Judgement

INSTRUCTIONS = (
    "The request comes from a customer of a bank. Does the document contain information "
    "that a customer service agent needs to handle this request, such as the policy, "
    "procedure, product terms or eligibility rules it involves?"
)
CRITERIA = {
    "true": "The document states rules, terms, steps or facts the agent must apply to this "
            "specific request.",
    "false": "The document is about a related topic or product but gives the agent nothing "
             "it needs for this specific request.",
}

Nouls = Callable[[Any, dict[str, dict]], tuple[dict[str, float], int, float]]


class JevJudge:
    name = "jev"

    def __init__(self, nouls: Nouls, *, workers: int = 8) -> None:
        self._nouls = nouls
        self._workers = workers

    def _one(self, query: str, candidate: Candidate) -> tuple[float, float]:
        state = {
            "request": query,
            "document": {"title": candidate.doc.title, "content": candidate.doc.content},
        }
        probabilities, _, usd = self._nouls(
            state, {"relevant": {"instructions": INSTRUCTIONS, "criteria": CRITERIA}}
        )
        return probabilities["relevant"], usd

    def judge(self, query: str, candidates: Sequence[Candidate]) -> Judgement:
        if not candidates:
            return Judgement(scores=(), is_probability=True)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            results = list(pool.map(lambda c: self._one(query, c), candidates))
        return Judgement(
            scores=tuple(p for p, _ in results),
            external_usd=sum(u for _, u in results),  # Jev bills its own account
            is_probability=True,
        )
