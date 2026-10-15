"""Naive RAG (A2) through the real agent and graph, with a fake embedder.

The banking toolkit is real; the model, the embedder and the corpus are small
stand-ins, so these tests assert on what retrieval puts in front of the model
and when, not on retrieval quality. No network.
"""

import json
from datetime import datetime

import numpy as np
import pytest
from tau2.data_model.message import ToolMessage, UserMessage
from tau2.domains.banking_knowledge.environment import get_environment

from policygate_rag.agent import RagAgent
from policygate_rag.config import RagConfig, RetrievalMode
from policygate_rag.index.bm25 import BM25Index
from policygate_rag.index.corpus import Document
from policygate_rag.index.dense import DenseIndex
from policygate_rag.index.fusion import HybridIndex
from policygate_rag.retrieval.embedder import Embedding
from policygate_rag.retrieval.naive import RetrievalResources
from tests.test_agent import HARNESS, Scripted, say, use

DOCS = [
    Document(doc_id="doc_dispute", title="Debit card disputes", content="File within 60 days."),
    Document(doc_id="doc_credit", title="Statement credits", content="Apply after review."),
    Document(doc_id="doc_savings", title="Savings rates", content="Rates change monthly."),
]
TOPICS = {"dispute": 0, "credit": 1, "savings": 2}


class FakeEmbedder:
    """One-hot on the first topic word found; costs 0.0001 per call."""

    model = "fake"

    def __init__(self):
        self.queries = []

    def embed(self, texts):
        self.queries.extend(texts)
        vectors = []
        for t in texts:
            v = np.full(3, 0.01)
            for word, i in TOPICS.items():
                if word in t.lower():
                    v[i] = 1.0
                    break
            vectors.append(v.tolist())
        return Embedding(vectors=vectors, tokens=5 * len(texts), usd=0.0001)


@pytest.fixture
def setup():
    embedder = FakeEmbedder()
    resources = RetrievalResources(
        index=HybridIndex(BM25Index(DOCS), DenseIndex([d.doc_id for d in DOCS], np.eye(3),
                                                       model="fake")),
        docs={d.doc_id: d for d in DOCS},
        embedder=embedder,
    )
    config = RagConfig(name="A2", harness=HARNESS, retrieval=RetrievalMode.NAIVE, k_final=1,
                       k_bm25=3, k_dense=3)
    return config, resources, embedder


def make(script, config, resources):
    env = get_environment(retrieval_variant="no_knowledge")
    agent = RagAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(), config=config,
        toolkit=env.tools, task_id="t", generate=script, retrieval=resources,
        clock=lambda: datetime(2026, 1, 1), session_id="s",
    )
    agent.set_seed(1)
    return env, agent, agent.get_init_state()


def system_prompt(call):
    return call["messages"][0].content


def test_a_user_turn_retrieves_and_the_documents_reach_the_prompt(setup):
    config, resources, embedder = setup
    script = Scripted(say("Here is how disputes work.", cost=0.01))
    _, agent, state = make(script, config, resources)
    out, state = agent.generate_next_message(
        UserMessage(role="user", content="I want to dispute a charge"), state)
    prompt = system_prompt(script.calls[0])
    assert "<retrieved_knowledge>" in prompt and "Debit card disputes" in prompt
    assert "Savings rates" not in prompt  # k_final=1
    assert embedder.queries == ["I want to dispute a charge"]  # the user's words, verbatim
    assert out.cost == pytest.approx(0.0101)  # the embedding is paid for in this turn
    assert [d["doc_id"] for d in state.context] == ["doc_dispute"]


def test_a_tool_result_turn_keeps_the_documents_without_retrieving(setup):
    config, resources, embedder = setup
    script = Scripted(use("get_current_time", "t1"), say("done"))
    env, agent, state = make(script, config, resources)
    out, state = agent.generate_next_message(
        UserMessage(role="user", content="dispute please"), state)
    result = env.get_response(out.tool_calls[0])
    assert isinstance(result, ToolMessage)
    out, state = agent.generate_next_message(result, state)
    assert len(embedder.queries) == 1  # no second retrieval
    assert "Debit card disputes" in system_prompt(script.calls[1])
    assert out.cost == pytest.approx(0.01)  # no embedding this turn


def test_the_next_user_turn_replaces_the_documents(setup):
    config, resources, embedder = setup
    script = Scripted(say("ok"), say("ok"))
    _, agent, state = make(script, config, resources)
    _, state = agent.generate_next_message(UserMessage(role="user", content="dispute"), state)
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="what about savings"), state)
    prompt = system_prompt(script.calls[1])
    assert "Savings rates" in prompt and "Debit card disputes" not in prompt


def test_the_trace_records_query_candidates_and_kept_ids(setup, tmp_path):
    from policygate_rag.tracing.recorder import TraceRecorder

    config, resources, _ = setup
    env = get_environment(retrieval_variant="no_knowledge")
    agent = RagAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(), config=config,
        toolkit=env.tools, task_id="t", generate=Scripted(say("ok")), retrieval=resources,
        recorder=TraceRecorder(tmp_path / "trace.jsonl"),
    )
    agent.set_seed(1)
    agent.generate_next_message(UserMessage(role="user", content="credit"),
                                agent.get_init_state())
    rows = [json.loads(r) for r in (tmp_path / "trace.jsonl").read_text().splitlines()]
    retrieval = next(r["data"]["retrieve"]["retrieval"] for r in rows
                     if r["type"] == "updates" and not r["ns"] and "retrieve" in r["data"])
    assert retrieval["query"] == "credit"
    assert retrieval["kept"] == ["doc_credit"]
    assert {c["doc_id"] for c in retrieval["candidates"]} >= {"doc_credit"}
    assert "content" not in json.dumps(retrieval["candidates"])  # ids and ranks, not text
