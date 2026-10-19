"""JudgeKind.LLM: an LLM grades a batch of candidates per call.

Each call carries the request and up to ``judge_batch_size`` numbered documents
and must return a relevance probability for every number, in JSON. A missing or
malformed probability is recorded as 0.0 and counted, not silently dropped:
the benchmark reports how often the grader failed to answer.

Pure Python: the model call is injected (``llm_client.py`` in runs, a stub in
tests), and its cost goes to the study's ledger like any other model call.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from policygate_rag.judges.base import Candidate, Judgement

SYSTEM = """You grade documents from a bank's internal knowledge base for relevance to a \
customer's request. A document is relevant if a customer service agent needs information in \
it (a policy, procedure, product term or eligibility rule) to handle this specific request; \
being about a related product or topic is not enough.

For every document number, give the probability (0 to 1) that it is relevant. Reply in JSON: \
{"probabilities": {"<number>": <probability>, ...}}"""

MAX_DOC_CHARS = 4000

Ask = Callable[[str, str], tuple[str, float]]
"""(system, user) -> (reply text, USD)."""


class LLMJudge:
    name = "llm"

    def __init__(self, ask: Ask, *, batch_size: int = 10) -> None:
        self._ask = ask
        self.batch_size = batch_size
        self.failures = 0
        """Probabilities the grader did not return, across all calls."""

    def _batch(self, query: str, batch: Sequence[Candidate]) -> tuple[list[float], float]:
        docs = "\n\n".join(
            f"<document number=\"{i}\" title=\"{c.doc.title}\">\n"
            f"{c.doc.content[:MAX_DOC_CHARS]}\n</document>"
            for i, c in enumerate(batch, start=1)
        )
        reply, usd = self._ask(SYSTEM, f"<request>{query}</request>\n\n{docs}")
        try:
            given = json.loads(reply).get("probabilities") or {}
        except (json.JSONDecodeError, AttributeError):
            given = {}
        scores = []
        for i in range(1, len(batch) + 1):
            try:
                scores.append(min(1.0, max(0.0, float(given[str(i)]))))
            except (KeyError, TypeError, ValueError):
                scores.append(0.0)
                self.failures += 1
        return scores, usd

    def judge(self, query: str, candidates: Sequence[Candidate]) -> Judgement:
        scores: list[float] = []
        usd = 0.0
        for start in range(0, len(candidates), self.batch_size):
            s, u = self._batch(query, candidates[start:start + self.batch_size])
            scores += s
            usd += u
        return Judgement(scores=tuple(scores), usd=usd, is_probability=True)
