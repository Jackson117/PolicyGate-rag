"""The whole pipeline offline: tau2's run_domain, the real banking environment
and evaluator, RagAgent, the guard and the ledger -- with the agent and the user
simulator scripted, so nothing is spent. Proves the wiring, not any number."""

import json

import pytest
import tau2.user.user_simulator as user_simulator
from tau2.data_model.message import AssistantMessage

from policygate_rag import runner
from policygate_rag.cost.ledger import BudgetLine, Ledger

AGENT_USD, USER_USD = 0.002, 0.001


def scripted_agent(model, messages, tools=None, call_name=None, **kwargs):
    return AssistantMessage(role="assistant", content="Could you tell me more?", cost=AGENT_USD)


@pytest.fixture
def scripted_user(monkeypatch):
    turns = {}

    def generate(model, messages, tools=None, call_name=None, **kwargs):
        key = id(messages[0])
        turns[key] = turns.get(key, 0) + 1
        text = "I have a question about my card." if turns[key] == 1 else "###STOP###"
        return AssistantMessage(role="assistant", content=text, cost=USER_USD)

    monkeypatch.setattr(user_simulator, "generate", generate)


def test_a_scripted_batch_runs_end_to_end_and_is_booked(tmp_path, scripted_user):
    ledger_path = tmp_path / "ledger.json"
    args = runner.argparse.Namespace(
        out=str(tmp_path / "run"), line="development", arms=["A0", "A4"],
        tasks="calibration", trials=1, seed=300, model="openai/gpt-4.1-mini",
        prior=0.01, chunk_size=3, max_concurrency=1, ledger=ledger_path, dry_run=False,
    )
    assert runner.run(args, generate=scripted_agent) == 0

    ledger = Ledger.load(ledger_path)
    episodes = [e for e in ledger.entries if e.kind.value == "episode"]
    assert len(episodes) == 6  # 2 arms x 3 calibration tasks x 1 trial
    assert all(e.line is BudgetLine.DEVELOPMENT for e in episodes)
    assert all(e.usd > 0 for e in episodes)
    # No discarded attempts here, so the trace and tau2's accounting agree.
    assert not [e for e in ledger.entries if e.kind.value == "call"]

    run_dir = tmp_path / "run"
    plan = json.loads((run_dir / "plan.json").read_text())
    assert plan["arms"]["A4"]["env_variant"] == "golden_retrieval"
    for arm in ("A0", "A4"):
        assert (run_dir / arm / "results_chunk_000.json").exists()
        assert (run_dir / arm / "checkpoints.sqlite").exists()
        rows = (run_dir / arm / "trace.jsonl").read_text().splitlines()
        assert any(json.loads(r)["type"] == "turn_cost" for r in rows)


def test_an_existing_output_directory_is_refused(tmp_path):
    (tmp_path / "run").mkdir()
    args = runner.argparse.Namespace(
        out=str(tmp_path / "run"), line="development", arms=["A0"], tasks="calibration",
        trials=1, seed=300, model="openai/gpt-4.1-mini", prior=0.01, chunk_size=3,
        max_concurrency=1, ledger=tmp_path / "ledger.json", dry_run=False,
    )
    assert runner.run(args, generate=scripted_agent) == 2
