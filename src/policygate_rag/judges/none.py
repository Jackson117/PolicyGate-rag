"""JudgeKind.NONE: no judgement. Candidates keep their fused rank.

The baseline for whether a judge is needed at all: filtering then keeps the
top k_final by RRF score, with no threshold, because RRF scores are not
probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence

from policygate_rag.judges.base import Candidate, Judgement


class NoJudge:
    name = "none"

    def judge(self, query: str, candidates: Sequence[Candidate]) -> Judgement:
        return Judgement(scores=tuple(c.fused_score for c in candidates), is_probability=False)
