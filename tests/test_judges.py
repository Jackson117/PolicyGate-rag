"""LLM and cross-encoder judges with stubbed model calls. No network, no download."""

import json

import pytest

from policygate_rag.index.corpus import Document
from policygate_rag.judges.base import Candidate
from policygate_rag.judges.cross_encoder import CrossEncoderJudge
from policygate_rag.judges.llm import LLMJudge
from policygate_rag.judges.none import NoJudge


def cands(n):
    return [Candidate(doc=Document(doc_id=f"d{i}", title=f"T{i}", content="C"),
                      fused_score=1 / (60 + i)) for i in range(1, n + 1)]


def test_no_judge_keeps_fused_scores_and_is_not_a_probability():
    v = NoJudge().judge("q", cands(3))
    assert v.scores == tuple(1 / (60 + i) for i in (1, 2, 3)) and not v.is_probability


class TestLLMJudge:
    def test_batches_and_maps_numbers_back_to_candidates(self):
        calls = []

        def ask(system, user):
            calls.append(user)
            n = user.count("<document number=")
            return json.dumps({"probabilities": {str(i): i / 10 for i in range(1, n + 1)}}), 0.003

        v = LLMJudge(ask, batch_size=4).judge("dispute", cands(10))
        assert len(calls) == 3  # 4 + 4 + 2
        assert v.scores == (0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2)
        assert v.usd == pytest.approx(0.009) and v.is_probability and v.external_usd == 0

    def test_missing_or_malformed_probabilities_are_zero_and_counted(self):
        judge = LLMJudge(lambda s, u: (json.dumps({"probabilities": {"1": 0.9, "2": "x",
                                                                     "3": 7}}), 0.0))
        v = judge.judge("q", cands(4))
        assert v.scores == (0.9, 0.0, 1.0, 0.0)  # "x" and #4 missing; 7 clamped
        assert judge.failures == 2
        assert LLMJudge(lambda s, u: ("not json", 0.0)).judge("q", cands(2)).scores == (0, 0)


def test_cross_encoder_applies_a_sigmoid_to_logits():
    seen = []

    def scorer(pairs):
        seen.extend(pairs)
        return [0.0, 10.0, -10.0]

    v = CrossEncoderJudge(scorer=scorer).judge("dispute", cands(3))
    assert v.scores[0] == pytest.approx(0.5)
    assert v.scores[1] > 0.99 and v.scores[2] < 0.01
    assert seen[0] == ("dispute", "T1\nC")
