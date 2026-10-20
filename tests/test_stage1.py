"""Stage 1 collect and bench, end to end on synthetic queries. No network."""

import json

import pytest

from policygate_rag.config import RagConfig
from policygate_rag.eval.judge_bench import bench, calibration, query_metrics, rerank
from policygate_rag.index.corpus import Document
from policygate_rag.judges.base import Judgement
from policygate_rag.judges.none import NoJudge
from policygate_rag.stage1.collect import collect, fused_scores, unique_pairs
from policygate_rag.tracing.read import Query


def q(task, text, cands, seed="1"):
    return Query(arm="A2", task_id=task, seed=seed, turn=0, round=0, query=text,
                 candidates=tuple(cands), shown=())


def test_rerank_breaks_ties_by_fused_order():
    assert rerank(["a", "b", "c"], {"a": 0.1, "b": 0.9, "c": 0.1}) == ["b", "a", "c"]


def test_query_metrics_by_hand():
    m = query_metrics(["x", "a", "y", "b"], {"x": .9, "a": .8, "y": .3, "b": .2}, {"a", "b"},
                      k=2, threshold=0.5)
    assert m["mrr"] == 0.5 and m["coverage_at_k"] == 0.5 and m["precision_at_k"] == 0.5
    assert m["coverage_at_threshold"] == 0.5 and m["kept_at_threshold"] == 2


def test_a_perfectly_calibrated_judge_has_zero_ece():
    pairs = [(0.25, i < 1) for i in range(4)] + [(0.75, i < 3) for i in range(4)]
    c = calibration(pairs)
    assert c["ece"] == pytest.approx(0.0)
    assert c["brier"] == pytest.approx(0.1875)


def test_unique_pairs_score_each_query_document_once():
    pairs = unique_pairs([q("t1", "same", ["a", "b"]), q("t1", "same", ["b", "c"], seed="2")])
    assert pairs == {"same": ["a", "b", "c"]}


def test_collect_resumes_and_bench_prefers_the_better_judge(tmp_path):
    docs = {d: Document(doc_id=d, title=d, content="") for d in "abcd"}
    queries = [q("t1", "q1", ["a", "b", "c", "d"]),
               q("t2", "q2", ["c", "d", "a", "b"])]
    fused = fused_scores(queries)
    pairs = unique_pairs(queries)

    class Oracle:
        name = "oracle"

        def judge(self, query, candidates):
            good = {"q1": {"d"}, "q2": {"b"}}[query]
            return Judgement(scores=tuple(0.9 if c.doc.doc_id in good else 0.1
                                          for c in candidates), is_probability=True)

    out_none, out_or = tmp_path / "scores_none.jsonl", tmp_path / "scores_oracle.jsonl"
    assert collect("none", NoJudge(), pairs, docs, fused, out_none)["pairs"] == 8
    assert collect("none", NoJudge(), pairs, docs, fused, out_none)["pairs"] == 0  # resumed
    collect("oracle", Oracle(), pairs, docs, fused, out_or)

    from policygate_rag.eval.judge_bench import load_scores
    judges = {"none": load_scores(out_none), "oracle": load_scores(out_or)}
    report = bench(queries, judges, {"t1": {"d"}, "t2": {"b"}}, RagConfig(k_final=1))
    assert report["judges"]["oracle"]["mrr"].startswith("1.000")
    assert report["judges"]["none"]["mrr"].startswith("0.250")
    assert report["paired_vs_none"]["oracle"]["mrr"].startswith("0.750")
    assert report["judges"]["oracle"]["calibration"]["pairs"] == 8
    json.dumps(report)


def test_the_llm_sample_keeps_every_task_and_is_seeded():
    from policygate_rag.stage1.collect import sample_per_task

    queries = [q(f"t{t}", f"t{t}-q{i}", ["a"]) for t in range(5) for i in range(4)]
    a = sample_per_task(queries, 2, seed=7)
    assert len(a) == 10 and {x.split("-")[0] for x in a} == {f"t{t}" for t in range(5)}
    assert a == sample_per_task(queries, 2, seed=7)
    assert len(sample_per_task(queries, 9, seed=7)) == 20  # capped by what a task has


def test_bench_restricts_the_primary_comparison_to_queries_every_judge_scored(tmp_path):
    from policygate_rag.eval.judge_bench import scored_everywhere

    qa, qb = q("t1", "qa", ["a", "b"]), q("t2", "qb", ["a", "b"])
    partial = {("qa", "a"): 0.5, ("qa", "b"): 0.1}
    assert scored_everywhere(qa, partial) and not scored_everywhere(qb, partial)
