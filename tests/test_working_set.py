"""The working set and product lookup (arm A5) through the real agent and graph. No network."""

import json
from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError
from tau2.data_model.message import UserMessage
from tau2.domains.banking_knowledge.environment import get_environment

from policygate_rag.agent import RagAgent
from policygate_rag.config import RagConfig, RetrievalMode
from policygate_rag.index.bm25 import BM25Index
from policygate_rag.index.corpus import Document
from policygate_rag.index.dense import DenseIndex
from policygate_rag.index.fusion import HybridIndex
from policygate_rag.retrieval.naive import RetrievalResources
from policygate_rag.retrieval.products import ProductLookup, mentioned_products, product_names
from policygate_rag.retrieval.working_set import named_now
from policygate_rag.tracing.read import read_trace
from policygate_rag.tracing.recorder import TraceRecorder
from tests.test_agent import HARNESS
from tests.test_agentic import Model, suff
from tests.test_naive import DOCS, FakeEmbedder

PRODUCT_DOCS = [
    Document(doc_id="doc_checking_accounts_light_blue_account_001",
             title="Light Blue Account at a glance", content="A starter account."),
    Document(doc_id="doc_checking_accounts_light_blue_account_002",
             title="Light Blue Account referral bonus", content="Referral bonus of 50."),
    Document(doc_id="doc_checking_accounts_blue_account_001",
             title="Blue Account at a glance", content="A standard account."),
]
ALL = DOCS + PRODUCT_DOCS


def setup(model, **overrides):
    vectors = np.vstack([np.eye(3), np.full((len(PRODUCT_DOCS), 3), 0.001)])
    resources = RetrievalResources(
        index=HybridIndex(BM25Index(ALL), DenseIndex([d.doc_id for d in ALL], vectors,
                                                     model="fake")),
        docs={d.doc_id: d for d in ALL}, embedder=FakeEmbedder(),
    )
    params = dict(name="A5", harness=HARNESS, retrieval=RetrievalMode.AGENTIC, k_final=1,
                  k_wide=1, k_bm25=1, k_dense=1, context_capacity=2) | overrides
    env = get_environment(retrieval_variant="no_knowledge")
    agent = RagAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(), config=RagConfig(**params),
        toolkit=env.tools, task_id="t", generate=model, retrieval=resources,
        clock=lambda: datetime(2026, 1, 1), session_id="s",
    )
    agent.set_seed(1)
    return agent, agent.get_init_state()


def turn(agent, state, text):
    return agent.generate_next_message(UserMessage(role="user", content=text), state)


def ids(state):
    return [d["doc_id"] for d in state.context]


def test_documents_are_kept_across_turns_and_the_oldest_leave_first():
    agent, state = setup(Model(sufficiency=[suff(0.9)] * 4))
    _, state = turn(agent, state, "dispute")
    assert ids(state) == ["doc_dispute"]
    _, state = turn(agent, state, "credit")
    assert ids(state) == ["doc_dispute", "doc_credit"]
    _, state = turn(agent, state, "savings")
    assert ids(state) == ["doc_credit", "doc_savings"]
    _, state = turn(agent, state, "dispute again")
    assert ids(state) == ["doc_savings", "doc_dispute"]


def test_the_prompt_shows_the_whole_set_newest_last():
    model = Model(sufficiency=[suff(0.9)] * 2)
    agent, state = setup(model)
    turn(agent, state, "dispute")
    _, state = turn(agent, state, "credit")
    system = model.calls[-1]["messages"][0].content
    assert "retrieved during this conversation, most recent last" in system
    assert system.index("doc_dispute") < system.index("doc_credit")


def test_a_named_product_brings_its_summary_and_best_match(tmp_path):
    recorder = TraceRecorder(tmp_path / "trace.jsonl")
    agent, state = setup(Model(sufficiency=[suff(0.9)]), context_capacity=5,
                         product_lookup=True)
    agent.recorder = recorder
    _, state = turn(agent, state, "what referral bonus does the Light Blue Account pay?")
    assert set(ids(state)) >= {"doc_checking_accounts_light_blue_account_001",
                               "doc_checking_accounts_light_blue_account_002"}
    assert "doc_checking_accounts_blue_account_001" not in ids(state)  # masked by "light blue"
    [episode] = read_trace(tmp_path / "trace.jsonl")
    assert episode.shown == set(ids(state))


def test_a_product_counts_once_unless_the_customer_names_it_again():
    look = ProductLookup([d.doc_id for d in ALL], BM25Index(ALL))
    named = [{"role": "user", "content": "my Blue Account"},
             {"role": "assistant", "content": "Looking at your Blue Account."}]
    assert named_now(named, look) == set()
    tool_turn = named + [{"role": "tool", "content": "accounts: Light Blue Account"}]
    assert named_now(tool_turn, look) == {"checking_accounts_light_blue_account"}
    user_turn = named + [{"role": "user", "content": "the Blue Account, yes"}]
    assert named_now(user_turn, look) == {"checking_accounts_blue_account"}


def test_names_come_from_ids_and_the_longest_wins():
    general = "doc_bank_accounts_bank_accounts_(general)_009"
    names = product_names([d.doc_id for d in ALL] + [general])
    assert set(names) == {"light blue account", "blue account"}
    assert mentioned_products("Light-Blue account", names) == {
        "checking_accounts_light_blue_account"}


@pytest.mark.parametrize("kw", [
    dict(retrieval=RetrievalMode.NONE, context_capacity=10),
    dict(retrieval=RetrievalMode.AGENTIC, context_capacity=2, k_final=5),
    dict(retrieval=RetrievalMode.AGENTIC, product_lookup=True),
])
def test_inconsistent_settings_are_refused(kw):
    with pytest.raises(ValidationError):
        RagConfig(name="x", harness=HARNESS, **kw)


def test_the_a5_arm_differs_from_a3_only_in_the_working_set():
    from experiments.arms import arms

    a = arms()
    a3, a5 = a["A3"].config.model_dump(), a["A5"].config.model_dump()
    diff = {k for k in a3 if a3[k] != a5[k]}
    assert diff == {"name", "context_capacity", "product_lookup"}
    assert a["A5"].env_variant == a["A3"].env_variant
    assert json.loads(json.dumps(a5))["context_capacity"] == 20
