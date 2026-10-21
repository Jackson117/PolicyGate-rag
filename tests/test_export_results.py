"""export_results: arms without complete runs never carry numbers. No runs needed."""

from types import SimpleNamespace

from policygate_rag.eval import export_results as ex


def sim(task, ok=True):
    return SimpleNamespace(
        task_id=task, agent_cost=0.01, user_cost=0.002, seed=1,
        reward_info=SimpleNamespace(reward=1.0 if ok else 0.0),
        termination_reason=SimpleNamespace(value="user_stop"))


def test_partial_and_missing_arms_carry_status_not_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    (tmp_path / "done").mkdir()
    (tmp_path / "half").mkdir()
    fake = {"done": [sim(f"t{i}", i % 2 == 0) for i in range(4) for _ in range(2)],
            "half": [sim("t0")]}
    monkeypatch.setattr(ex, "_sims", lambda dirs: [s for d in dirs for s in fake[d]])
    monkeypatch.setattr(ex, "_arm_config", lambda dirs, arm: {})
    out = ex.export_arms({
        "A0": {"label": "floor", "runs": ["done"]},
        "A3": {"label": "agentic", "runs": ["half"]},
        "A1": {"label": "bm25", "runs": []},
    }, expected=8)
    assert out["arms"]["A0"]["status"] == "complete"
    assert out["arms"]["A0"]["pass_k"]["pass^1"]["estimate"] == 0.5
    assert out["arms"]["A3"]["status"] == "partial" and "pass_k" not in out["arms"]["A3"]
    assert out["arms"]["A1"] == {"label": "bm25", "runs": [], "episodes": 0, "status": "not run"}
    assert out["paired"] == {}  # only one complete arm


def test_a_partial_arm_with_every_task_reports_pass1_only(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    for d in ("done", "most"):
        (tmp_path / d).mkdir()
    fake = {"done": [sim(f"t{i}", i == 0) for i in range(2) for _ in range(3)],
            # t0 has three trials, t1 one: every task, not every trial.
            "most": [sim("t0", True), sim("t0", False), sim("t0", True), sim("t1", False)]}
    monkeypatch.setattr(ex, "_sims", lambda dirs: [s for d in dirs for s in fake[d]])
    monkeypatch.setattr(ex, "_arm_config", lambda dirs, arm: {})
    out = ex.export_arms({"A0": {"label": "floor", "runs": ["done"]},
                          "A5": {"label": "ws", "runs": ["most"]}},
                         expected=6, task_ids={"t0", "t1"})
    a5 = out["arms"]["A5"]
    assert a5["status"] == "partial" and a5["covers_all_tasks"]
    assert list(a5["pass_k"]) == ["pass^1"]
    assert a5["pass_k"]["pass^1"]["estimate"] == (2 / 3 + 0) / 2
    assert a5["per_task_trials"] == {"t0": 3, "t1": 1}
    assert list(out["paired"]["A5 - A0"]) == ["pass^1"]
