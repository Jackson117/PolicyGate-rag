"""Agentic RAG (A3) through the real agent and graph, with scripted model calls.

The scripted model answers by call type: the agent's turn, the sufficiency
check, the query rewrite. A one-candidate-per-round setup (k_wide=1) makes
each round's new document predictable, so every stop reason can be forced.
No network.
"""

import json
from datetime import datetime

import numpy as np
import pytest
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.domains.banking_knowledge.environment import get_environment

from policygate_rag.agent import RagAgent
from policygate_rag.config import RagConfig, RetrievalMode
from policygate_rag.index.bm25 import BM25Index
from policygate_rag.index.dense import DenseIndex
from policygate_rag.index.fusion import HybridIndex
from policygate_rag.judges.base import Judgement
from policygate_rag.retrieval.naive import RetrievalResources
from tests.test_agent import HARNESS
from tests.test_naive import DOCS, FakeEmbedder


class Model:
    """Scripted by call_name; records every call it receives."""

    def __init__(self, sufficiency=(), rewrites=(), agent="ok"):
        self.sufficiency, self.rewrites, self.agent = list(sufficiency), list(rewrites), agent
        self.calls = []

    def __call__(self, model, messages, tools=None, call_name=None, **kwargs):
        self.calls.append({"name": call_name, "messages": messages, "kwargs": kwargs})
        if call_name == "sufficiency_check":
            content, cost = self.sufficiency.pop(0), 0.002
        elif call_name == "query_rewrite":
            content, cost = self.rewrites.pop(0), 0.001
        elif isinstance(self.agent, AssistantMessage):
            return self.agent
        else:
            content, cost = self.agent, 0.01
        return AssistantMessage(role="assistant", content=content, cost=cost)

    def names(self):
        return [c["name"] for c in self.calls]


def suff(p, missing=""):
    return json.dumps({"probability": p, "missing": missing})


def rw(query):
    return json.dumps({"query": query})


def setup(model, *, judge=None, **overrides):
    embedder = FakeEmbedder()
    resources = RetrievalResources(
        index=HybridIndex(BM25Index(DOCS),
                          DenseIndex([d.doc_id for d in DOCS], np.eye(3), model="fake")),
        docs={d.doc_id: d for d in DOCS}, embedder=embedder,
        **({"judge": judge} if judge else {}),
    )
    params = dict(name="A3", harness=HARNESS, retrieval=RetrievalMode.AGENTIC,
                  k_final=2, k_wide=1, k_bm25=3, k_dense=3) | overrides
    env = get_environment(retrieval_variant="no_knowledge")
    agent = RagAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(), config=RagConfig(**params),
        toolkit=env.tools, task_id="t", generate=model, retrieval=resources,
        clock=lambda: datetime(2026, 1, 1), session_id="s",
    )
    agent.set_seed(1)
    return agent, agent.get_init_state(), embedder


def turn(agent, state, text):
    return agent.generate_next_message(UserMessage(role="user", content=text), state)


class TestStopReasons:
    def test_sufficient_on_the_first_round(self):
        model = Model(sufficiency=[suff(0.9)])
        agent, state, embedder = setup(model)
        out, state = turn(agent, state, "I want to dispute a charge")
        assert model.names() == ["sufficiency_check", "agent_response"]
        assert embedder.queries == ["I want to dispute a charge"]
        assert [d["doc_id"] for d in state.context] == ["doc_dispute"]
        assert out.cost == pytest.approx(0.0001 + 0.002 + 0.01)

    def test_a_rewrite_finds_more_and_both_rounds_feed_the_context(self):
        model = Model(sufficiency=[suff(0.2, "credit rules"), suff(0.8)], rewrites=[rw("credit")])
        agent, state, embedder = setup(model)
        out, state = turn(agent, state, "I want to dispute a charge")
        assert model.names() == ["sufficiency_check", "query_rewrite", "sufficiency_check",
                                 "agent_response"]
        assert embedder.queries == ["I want to dispute a charge", "credit"]
        assert {d["doc_id"] for d in state.context} == {"doc_dispute", "doc_credit"}
        assert out.cost == pytest.approx(2 * 0.0001 + 2 * 0.002 + 0.001 + 0.01)

    def test_the_round_budget_ends_the_loop(self):
        model = Model(sufficiency=[suff(0.1)] * 3, rewrites=[rw("credit"), rw("savings")])
        agent, state, _ = setup(model, agentic_max_rounds=3, k_final=3)
        _, state = turn(agent, state, "dispute")
        assert model.names().count("sufficiency_check") == 3
        assert model.names().count("query_rewrite") == 2
        assert len(state.context) == 3

    def test_a_rewrite_that_finds_nothing_new_stops_the_loop(self):
        model = Model(sufficiency=[suff(0.1)], rewrites=[rw("dispute")])
        agent, state, embedder = setup(model)
        _, state = turn(agent, state, "dispute")
        assert embedder.queries == ["dispute", "dispute"]
        assert model.names() == ["sufficiency_check", "query_rewrite", "agent_response"]
        assert [d["doc_id"] for d in state.context] == ["doc_dispute"]

    @pytest.mark.parametrize("expected,model", [
        ("sufficient", Model(sufficiency=[suff(0.9)])),
        ("no_new_docs", Model(sufficiency=[suff(0.1)], rewrites=[rw("dispute")])),
        ("budget_exhausted", Model(sufficiency=[suff(0.1)] * 2, rewrites=[rw("credit")])),
    ])
    def test_the_stop_reason_is_recorded(self, tmp_path, expected, model):
        from policygate_rag.tracing.recorder import TraceRecorder

        agent, state, _ = setup(model, agentic_max_rounds=2)
        agent.recorder = TraceRecorder(tmp_path / "t.jsonl")
        turn(agent, state, "dispute")
        rows = [json.loads(r) for r in (tmp_path / "t.jsonl").read_text().splitlines()]
        r = next(x["data"]["retrieve"]["retrieval"] for x in rows
                 if x["type"] == "updates" and not x["ns"] and "retrieve" in x["data"])
        assert r["stop_reason"] == expected
        json.dumps(r)  # JSON-native throughout


class TestWhatTheModelCallsSee:
    def test_the_check_sees_the_earlier_request_on_an_identity_turn(self):
        model = Model(sufficiency=[suff(0.9), suff(0.9)])
        agent, state, _ = setup(model)
        _, state = turn(agent, state, "I want to dispute a charge")
        _, state = turn(agent, state, "My name is Kenji Tanaka")
        check = [c for c in model.calls if c["name"] == "sufficiency_check"][-1]
        prompt = check["messages"][1].content
        assert "dispute a charge" in prompt and "Kenji Tanaka" in prompt
        assert check["kwargs"]["response_format"] == {"type": "json_object"}

    def test_a_tool_result_turn_reuses_the_documents(self):
        from tau2.data_model.message import ToolCall

        model = Model(sufficiency=[suff(0.9)], agent=AssistantMessage(
            role="assistant", content=None, cost=0.01,
            tool_calls=[ToolCall(id="t1", name="get_current_time", arguments={})]))
        agent, state, embedder = setup(model)
        out, state = turn(agent, state, "dispute")
        env = get_environment(retrieval_variant="no_knowledge")
        _, state = agent.generate_next_message(env.get_response(out.tool_calls[0]), state)
        assert len(embedder.queries) == 1
        assert model.names().count("sufficiency_check") == 1
        assert [d["doc_id"] for d in state.context] == ["doc_dispute"]

class TestRobustness:
    def test_unparseable_model_output_counts_as_insufficient_and_falls_back(self):
        model = Model(sufficiency=["not json", suff(0.9)], rewrites=["also not json"])
        agent, state, embedder = setup(model)
        _, state = turn(agent, state, "dispute")
        # No 'missing' text to search for, so the rewrite falls back to the request.
        assert embedder.queries == ["dispute", "dispute"]

    def test_a_probability_judge_filters_below_the_threshold(self):
        class Strict:
            name = "strict"

            def judge(self, query, candidates):
                return Judgement(scores=tuple(0.1 for _ in candidates), is_probability=True)

        model = Model(sufficiency=[suff(0.9)])
        agent, state, _ = setup(model, judge=Strict(), judge_threshold=0.5)
        _, state = turn(agent, state, "dispute")
        assert state.context == []
