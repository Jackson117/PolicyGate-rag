"""eval/retrieval_metrics.py: formulas by hand, and reading a real naive trace shape."""

import json

import pytest

from policygate_rag.eval.retrieval_metrics import (
    coverage,
    episode_scores,
    parse_thread,
    precision,
    read_trace,
    reciprocal_rank,
)


def test_coverage_precision_and_rr_by_hand():
    assert coverage(["a", "b", "x"], ["a", "c"]) == 0.5
    assert precision(["a", "b", "x"], ["a", "c"]) == pytest.approx(1 / 3)
    assert precision([], ["a"]) is None
    assert reciprocal_rank(["x", "y", "a"], ["a", "b"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], ["a"]) == 0.0
    with pytest.raises(ValueError):
        coverage(["a"], [])


def test_thread_ids_parse_and_bad_ones_are_refused():
    assert parse_thread("A2-task_017-seed626729") == ("A2", "task_017", "626729")
    with pytest.raises(ValueError):
        parse_thread("nonsense")


def _row(thread, turn, retrieval):
    return json.dumps({"thread_id": thread, "turn": turn, "type": "updates", "ns": [],
                       "data": {"retrieve": {"retrieval": retrieval}}})


def test_episodes_union_what_was_shown_and_list_every_query(tmp_path):
    cand = lambda *ids: [{"doc_id": i, "score": 0.1, "ranks": {}} for i in ids]  # noqa: E731
    lines = [
        _row("A2-t1-seed1", 0, {"mode": "naive", "reused": False, "query": "q1",
                                "candidates": cand("x", "a", "b"), "kept": ["x", "a"]}),
        _row("A2-t1-seed1", 1, {"mode": "naive", "reused": True, "kept": ["x", "a"]}),
        _row("A2-t1-seed1", 2, {"mode": "naive", "reused": False, "query": "q2",
                                "candidates": cand("c"), "kept": ["c"]}),
        _row("A3dev-t1-seed1", 0, {"mode": "agentic", "reused": False, "query": "q",
                                   "rounds": [{"query": "q", "candidates": cand("x")},
                                              {"query": "q'", "candidates": cand("a")}],
                                   "kept": ["a"], "stop_reason": "sufficient"}),
        json.dumps({"thread_id": "A2-t1-seed1", "turn": 0, "type": "turn_cost", "ns": [],
                    "data": {"usd": 0.01}}),
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(lines) + "\n")
    episodes = {e.arm: e for e in read_trace(path)}
    naive = episodes["A2"]
    assert naive.shown == {"x", "a", "c"}
    assert [q.query for q in naive.queries] == ["q1", "q2"]
    assert [q.round for q in episodes["A3dev"].queries] == [0, 1]

    [score] = episode_scores([naive], {"t1": ["a", "z"]})
    assert score["coverage"] == 0.5
    assert score["precision"] == pytest.approx(1 / 3)
    assert score["mrr"] == pytest.approx((1 / 2 + 0) / 2)
