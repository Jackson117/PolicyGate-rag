"""RagAgent and the per-turn graph, driven by a scripted model.

The tau2 side is real -- the banking toolkit, its tool objects, tool execution --
and the model is a stub returning a fixed sequence of proposals, so these assert
on what the graph and harness do, not on what a model would. No network.
"""

import json
import sqlite3
from datetime import datetime

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from policygate.harness.config import HarnessConfig
from policygate.harness.state import SessionState
from policygate.harness.types import AuditEvent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.banking_knowledge.environment import get_db, get_environment

from policygate_rag.agent import RagAgent
from policygate_rag.config import RagConfig, RetrievalMode
from policygate_rag.tracing.recorder import TraceRecorder

HARNESS = HarnessConfig(name="test", gating_enabled=True, audit_enabled=True)


class Scripted:
    """Returns the given proposals in order, recording what each call was offered."""

    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.calls = []

    def __call__(self, model, messages, tools=None, call_name=None, **kwargs):
        self.calls.append({"tools": [t.name for t in tools or []], "messages": messages,
                           "kwargs": kwargs})
        return self.proposals.pop(0)


def say(text, cost=0.01):
    return AssistantMessage(role="assistant", content=text, cost=cost)


def use(tool, id, cost=0.01, **args):
    return AssistantMessage(role="assistant", content=None, cost=cost,
                            tool_calls=[ToolCall(id=id, name=tool, arguments=args)])


@pytest.fixture(scope="module")
def user():
    users = get_db().model_dump()["users"]["data"]
    return sorted(users.items())[0]


def make(script, *, config=None, variant="no_knowledge", **kwargs):
    env = get_environment(retrieval_variant=variant)
    config = config or RagConfig(name="A0", harness=HARNESS)
    agent = RagAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(), config=config,
        toolkit=env.tools, task_id="task_001", generate=script,
        clock=lambda: datetime(2026, 1, 1), session_id="s", **kwargs,
    )
    agent.set_seed(7)
    return env, agent, agent.get_init_state()


def hello():
    return UserMessage(role="user", content="Hi, I need help with my card.")


class TestOneTurn:
    def test_a_text_reply_passes_through_and_carries_its_cost(self):
        script = Scripted(say("Happy to help.", cost=0.02))
        _, agent, state = make(script)
        out, state = agent.generate_next_message(hello(), state)
        assert out.content == "Happy to help."
        assert out.cost == pytest.approx(0.02)
        assert script.calls[0]["kwargs"] == {"temperature": 0.0, "seed": 7}

    def test_a_refused_proposal_is_retried_inside_the_turn_and_its_cost_kept(self):
        script = Scripted(
            use("get_credit_card_accounts_by_user", "c1", cost=0.01, user_id="123"),
            use("get_user_information_by_id", "c2", cost=0.03, user_id="123"),
        )
        _, agent, state = make(script)
        out, state = agent.generate_next_message(hello(), state)
        assert out.tool_calls[0].name == "get_user_information_by_id"
        assert out.cost == pytest.approx(0.04)  # the refused proposal is paid for too
        refusal = next(m for m in state.messages if isinstance(m, ToolMessage))
        assert refusal.error and "gating.identity_first" in refusal.content
        # The model saw the refusal before its second proposal.
        assert isinstance(script.calls[1]["messages"][-1], ToolMessage)

    def test_the_budget_runs_out_into_an_escalation(self):
        refused = [use("get_credit_card_accounts_by_user", f"c{i}", user_id="123")
                   for i in range(3)]
        _, agent, state = make(Scripted(*refused))
        out, state = agent.generate_next_message(hello(), state)
        assert out.tool_calls[0].name == "transfer_to_human_agents"
        assert out.cost == pytest.approx(0.03)
        rules = [r.decision.rule for r in state.session.chain.records if r.decision]
        assert rules[-1] == "gating.budget_exhausted"

    def test_customer_data_is_not_offered_before_verification_when_pruning(self):
        config = RagConfig(name="A0", harness=HARNESS.model_copy(
            update={"context_pruning_enabled": True}))
        script = Scripted(say("ok"))
        _, agent, state = make(script, config=config)
        agent.generate_next_message(hello(), state)
        offered = set(script.calls[0]["tools"])
        assert "get_user_information_by_id" in offered
        assert "get_credit_card_accounts_by_user" not in offered


class TestAcrossTurns:
    def test_verification_then_a_discoverable_write(self, user):
        uid, rec = user
        script = Scripted(
            use("get_user_information_by_id", "l1", user_id=uid),
            use("log_verification", "v1", name=rec["name"], user_id=uid,
                address=rec["address"], email=rec["email"], phone_number=rec["phone_number"],
                date_of_birth=rec["date_of_birth"], time_verified="2025-11-14 03:40:00 EST"),
            use("call_discoverable_agent_tool", "w1",
                agent_tool_name="apply_statement_credit_8472", arguments="{}"),
        )
        env, agent, state = make(script)

        out, state = agent.generate_next_message(hello(), state)
        out, state = agent.generate_next_message(env.get_response(out.tool_calls[0]), state)
        assert state.session.state is SessionState.UNVERIFIED  # a lookup is not verification
        out, state = agent.generate_next_message(env.get_response(out.tool_calls[0]), state)
        assert state.session.state is SessionState.SCOPED
        assert out.tool_calls[0].name == "call_discoverable_agent_tool"  # tau2 gets the original
        events = [r.event for r in state.session.chain.records]
        assert AuditEvent.TOOL_EXECUTED in events  # the verification log was recorded
        proposed = [r.tool_call.name for r in state.session.chain.records
                    if r.event is AuditEvent.TOOL_PROPOSED]
        assert proposed[-1] == "apply_statement_credit_8472"  # the harness saw the real op


class TestPersistence:
    def test_checkpoints_are_json_native_and_keyed_by_episode(self, tmp_path):
        saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
        recorder = TraceRecorder(tmp_path / "trace.jsonl")
        script = Scripted(say("one"), say("two"))
        _, agent, state = make(script, checkpointer=saver, recorder=recorder)
        _, state = agent.generate_next_message(hello(), state)
        _, state = agent.generate_next_message(
            UserMessage(role="user", content="thanks"), state)

        assert agent.thread_id == "A0-task_001-seed7"
        snapshot = agent.graph.get_state({"configurable": {"thread_id": agent.thread_id}})
        json.dumps(snapshot.values)  # raises if anything non-JSON got in
        assert snapshot.values["emitted"]["content"] == "two"

        rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        assert {r["turn"] for r in rows} == {0, 1}
        assert {r["thread_id"] for r in rows} == {"A0-task_001-seed7"}
        assert {"retrieve", "reason", "gate"} <= {k for r in rows for k in r["data"]}


def test_a_retrieval_arm_without_resources_is_refused():
    config = RagConfig(name="A2", harness=HARNESS, retrieval=RetrievalMode.NAIVE)
    with pytest.raises(ValueError, match="needs RetrievalResources"):
        make(Scripted(say("x")), config=config)


def test_agentic_retrieval_is_not_built_yet():
    from policygate_rag.graph.build import build_graph

    with pytest.raises(NotImplementedError):
        build_graph(RagConfig(name="A3", harness=HARNESS, retrieval=RetrievalMode.AGENTIC))


class TestLoopGuard:
    def test_a_repeated_read_is_refused_and_persistence_escalates(self):
        config = RagConfig(name="A0", harness=HARNESS.model_copy(
            update={"loop_guards_enabled": True}))
        same = [use("get_current_time", f"t{i}") for i in range(7)]
        env, agent, state = make(Scripted(*same), config=config)
        out, state = agent.generate_next_message(hello(), state)
        for _ in range(2):  # the second and third identical calls go through
            out, state = agent.generate_next_message(env.get_response(out.tool_calls[0]), state)
            assert out.tool_calls[0].name == "get_current_time"
        # The fourth is refused; the model repeats it until the attempts run out.
        out, state = agent.generate_next_message(env.get_response(out.tool_calls[0]), state)
        assert out.tool_calls[0].name == "transfer_to_human_agents"
        rules = [r.decision.rule for r in state.session.chain.records if r.decision]
        assert rules.count("loop_guard.repeated_action") == 3
