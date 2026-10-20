"""Stage 1, step 2: score the judges against required_documents.

    python -m policygate_rag.eval.judge_bench runs/stage1

For every harvested query, each judge re-ranks the same top-k_wide candidates
(ties broken by fused rank). Per query:

* MRR of the re-ranked list (any required document counts as relevant);
* coverage@k and precision@k of the top k_final -- what would reach the model;
* for judges that output probabilities, coverage and precision of the set at or
  above ``judge_threshold``, and calibration: Brier score, expected calibration
  error and a reliability table over all scored pairs.

Queries are averaged within a task, then tasks are the unit: intervals are
bootstrap over tasks, and judge-vs-judge differences are paired by task.
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from policygate_rag.config import RagConfig
from policygate_rag.eval.passk import bootstrap_mean
from policygate_rag.eval.retrieval_metrics import required_documents
from policygate_rag.tracing.read import Query

BINS = 10


def rerank(candidates: Sequence[str], scores: Mapping[str, float]) -> list[str]:
    order = {d: i for i, d in enumerate(candidates)}
    return sorted(candidates, key=lambda d: (-scores[d], order[d]))


def query_metrics(ranked: Sequence[str], scores: Mapping[str, float], required: set[str], *,
                  k: int, threshold: float | None) -> dict[str, float | None]:
    top = ranked[:k]
    hit_rank = next((i for i, d in enumerate(ranked, start=1) if d in required), None)
    m: dict[str, float | None] = {
        "mrr": 0.0 if hit_rank is None else 1.0 / hit_rank,
        "coverage_at_k": len(set(top) & required) / len(required),
        "precision_at_k": len(set(top) & required) / len(top) if top else None,
    }
    if threshold is not None:
        kept = {d for d in ranked if scores[d] >= threshold}
        m["coverage_at_threshold"] = len(kept & required) / len(required)
        m["precision_at_threshold"] = len(kept & required) / len(kept) if kept else None
        m["kept_at_threshold"] = float(len(kept))
    return m


def calibration(pairs: Sequence[tuple[float, bool]], bins: int = BINS) -> dict:
    """Brier score, expected calibration error, and the reliability table."""
    if not pairs:
        return {}
    brier = statistics.fmean((p - y) ** 2 for p, y in pairs)
    table, ece = [], 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not inside:
            continue
        mean_p = statistics.fmean(p for p, _ in inside)
        rate = statistics.fmean(1.0 if y else 0.0 for _, y in inside)
        ece += len(inside) / len(pairs) * abs(mean_p - rate)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(inside),
                      "mean_predicted": mean_p, "observed_rate": rate})
    return {"brier": brier, "ece": ece, "reliability": table, "pairs": len(pairs)}


def load_scores(path: Path) -> tuple[dict[tuple[str, str], float], bool]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {(r["query"], r["doc_id"]): r["score"] for r in rows}, bool(
        rows and rows[0]["is_probability"])


def bench(queries: Sequence[Query], judges: Mapping[str, tuple[dict, bool]],
          required: Mapping[str, set[str]], config: RagConfig) -> dict:
    per_task: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    calib: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for q in queries:
        req = set(required[q.task_id])
        for name, (scores, is_prob) in judges.items():
            s = {d: scores[(q.query, d)] for d in q.candidates}
            ranked = rerank(q.candidates, s)
            m = query_metrics(ranked, s, req, k=config.k_final,
                              threshold=config.judge_threshold if is_prob else None)
            for key, value in m.items():
                if value is not None:
                    per_task[name][q.task_id][key].append(value)
            if is_prob:
                calib[name].extend((s[d], d in req) for d in q.candidates)

    report: dict = {"queries": len(queries), "tasks": len({q.task_id for q in queries}),
                    "k": config.k_final, "threshold": config.judge_threshold, "judges": {},
                    "per_task": {}, "paired": {}}
    task_means: dict[str, dict[str, dict[str, float]]] = {}
    for name, tasks in per_task.items():
        means = {t: {k: statistics.fmean(v) for k, v in ms.items()} for t, ms in tasks.items()}
        task_means[name] = means
        report["per_task"][name] = means
        metrics = sorted({k for m in means.values() for k in m})
        report["judges"][name] = {
            k: bootstrap_mean([m[k] for m in means.values() if k in m]).model_dump()
            for k in metrics
        }
        if calib.get(name):
            report["judges"][name]["calibration"] = calibration(calib[name])
    # Every pair of judges once, differences paired by task: each judge minus the
    # no-judge baseline, and each other pair in alphabetical order ("a - b").
    others = sorted(n for n in task_means if n != "none")
    pairs = [(n, "none") for n in others if "none" in task_means]
    pairs += list(itertools.combinations(others, 2))
    for a, b in pairs:
        shared = sorted(set(task_means[a]) & set(task_means[b]))
        report["paired"][f"{a} - {b}"] = {
            k: bootstrap_mean([task_means[a][t][k] - task_means[b][t][k] for t in shared]
                              ).model_dump()
            for k in ("mrr", "coverage_at_k", "precision_at_k")
        }
    return report


def scored_everywhere(q: Query, scores: Mapping[tuple[str, str], float]) -> bool:
    return all((q.query, d) in scores for d in q.candidates)


def main() -> None:
    out = Path(sys.argv[1])
    queries = [Query.model_validate_json(line)
               for line in (out / "queries.jsonl").read_text().splitlines()]
    judges = {p.stem.removeprefix("scores_"): load_scores(p)
              for p in sorted(out.glob("scores_*.jsonl"))}
    required = {k: set(v) for k, v in
                required_documents({q.task_id for q in queries}).items()}
    config = RagConfig()
    # Primary: every judge on the same queries (the LLM judge scored a per-task
    # sample), so differences are paired on identical inputs.
    common = [q for q in queries if all(scored_everywhere(q, s) for s, _ in judges.values())]
    # Secondary: the judges that scored every query, on all of them.
    full_judges = {n: j for n, j in judges.items()
                   if all(scored_everywhere(q, j[0]) for q in queries)}
    report = {
        "common": bench(common, judges, required, config),
        "full": bench(queries, full_judges, required, config),
    }
    (out / "bench.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
