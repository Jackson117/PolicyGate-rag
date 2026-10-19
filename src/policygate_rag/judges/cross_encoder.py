"""JudgeKind.CROSS_ENCODER: a small local reranker, pinned to a revision.

``cross-encoder/ms-marco-MiniLM-L6-v2`` (Apache-2.0), revision pinned below: free
to run, deterministic, and trained on web search (MS MARCO) rather than banking,
which is part of what stage 1 measures. Its output is passed through a sigmoid;
that makes it a number in 0..1, not a calibrated probability, and stage 1's
reliability diagram is what says how far it is from one.

sentence-transformers is imported only when the judge is built, so the rest of
the package does not need it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from policygate_rag.judges.base import Candidate, Judgement

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
"""The Hugging Face commit, looked up 2026-10-19. The model was renamed from
``ms-marco-MiniLM-L-6-v2``; the old name redirects."""

MAX_DOC_CHARS = 4000


class CrossEncoderJudge:
    name = "cross_encoder"

    def __init__(self, model: str = MODEL, revision: str = REVISION, *, scorer=None) -> None:
        if scorer is None:
            from sentence_transformers import CrossEncoder

            encoder = CrossEncoder(model, revision=revision)
            scorer = lambda pairs: encoder.predict(pairs, activation_fn=None).tolist()  # noqa: E731
        self._score = scorer
        self.model, self.revision = model, revision

    def judge(self, query: str, candidates: Sequence[Candidate]) -> Judgement:
        if not candidates:
            return Judgement(scores=(), is_probability=True)
        pairs = [(query, f"{c.doc.title}\n{c.doc.content[:MAX_DOC_CHARS]}") for c in candidates]
        logits = self._score(pairs)
        return Judgement(
            scores=tuple(1 / (1 + math.exp(-x)) for x in logits),
            is_probability=True,
        )
