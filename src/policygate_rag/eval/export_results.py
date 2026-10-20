"""Export every measured result to versioned JSON under ``results/``, for the dashboard.

    python -m policygate_rag.eval.export_results

Reads the run directories named in ``experiments/results_manifest.json``, the
ledger and the labels, and writes:

* ``stage1_judges.json`` -- every judge tested in stage 1, on the queries all judges
  scored and on the full set: ranking metrics, per-task values, every pairwise
  difference, calibration tables, costs, the LLM judge's sample;
* ``stage2_arms.json``   -- per arm: pass^1..3 with intervals, per-task outcomes, cost,
  terminations, loop-guard refusals, retrieval metrics; every pair of arms paired by task;
* ``w1.json``            -- W1 calibration, and the floor/ceiling runs from before the
  loop guard, marked superseded;
* ``spend.json``         -- the ledger by line, and spend billed to other accounts;
* ``index.json``         -- schema version, when and from which commit, what is complete.

Every interval is ``{"estimate", "low", "high", "n_tasks"}``: 95% bootstrap over tasks.
Arms with no or partial runs are exported with their status, never with numbers that
look final. Regenerate after every batch; the files are committed, runs/ is not.
"""

from __future__ import annotations

import itertools
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from policygate_rag.config import RagConfig
from policygate_rag.eval.passk import arm_pass_k, paired_difference
from policygate_rag.eval.retrieval_metrics import episode_scores, read_trace, required_documents

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results"
ARM_ORDER = ("A4", "A3", "A2", "A1", "A0")
"""Pairs are reported as earlier-minus-later in this order: ceiling first, floor last."""


def _sims(dirs: list[str]):
    from tau2.data_model.simulation import Results

    return [s for d in dirs for p in sorted((ROOT / d).glob("results_chunk_*.json"))
            for s in Results.load(p).simulations]


def _outcomes(sims) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = defaultdict(list)
    for s in sims:
        out[s.task_id].append(bool(s.reward_info and s.reward_info.reward >= 1.0))
    return dict(out)


def _guard_refusals(dirs: list[str]) -> int:
    n = 0
    for d in dirs:
        trace = ROOT / d / "trace.jsonl"
        if not trace.exists():
            continue
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            if row["type"] == "updates" and not row["ns"] and "gate" in row["data"]:
                n += sum(v["decision"]["rule"] == "loop_guard.repeated_action"
                         for v in (row["data"]["gate"] or {}).get("verdicts", []))
    return n


def _arm_config(dirs: list[str], arm: str) -> dict | None:
    for d in dirs:
        plan = ROOT / Path(d).parent / "plan.json"
        if plan.exists():
            entry = json.loads(plan.read_text())["arms"].get(arm)
            if entry:
                c = entry["config"]
                return {"env_variant": entry["env_variant"], "retrieval": c["retrieval"],
                        "judge": c["judge"], "agent_llm": c["agent_llm"],
                        "harness": {"name": c["harness"]["name"]} | {
                            k: v for k, v in c["harness"].items()
                            if k.endswith("_enabled") and v},
                        "max_steps": c.get("max_steps"), "k_final": c["k_final"]}
    return None


def _external(dirs: list[str]) -> float:
    return sum(json.loads((ROOT / d / "external_spend.json").read_text())["usd"]
               for d in dirs if (ROOT / d / "external_spend.json").exists())


def export_arms(spec: dict[str, Any], expected: int) -> dict:
    arms, outcomes = {}, {}
    for arm, meta in spec.items():
        if arm.startswith("_") or not isinstance(meta, dict) or "runs" not in meta:
            continue
        dirs = [d for d in meta["runs"] if (ROOT / d).exists()]
        sims = _sims(dirs) if dirs else []
        entry: dict[str, Any] = {"label": meta["label"], "runs": dirs, "episodes": len(sims)}
        if not sims:
            entry["status"] = "not run"
            arms[arm] = entry
            continue
        entry["status"] = "complete" if len(sims) >= expected else "partial"
        entry["config"] = _arm_config(dirs, arm)
        if entry["status"] == "partial":
            # Progress only: a pass rate over a handful of episodes would read as a result.
            entry["tasks_started"] = len({s.task_id for s in sims})
            arms[arm] = entry
            continue
        o = _outcomes(sims)
        trials = min(map(len, o.values()))
        cost = [(s.agent_cost or 0.0) + (s.user_cost or 0.0) for s in sims]
        entry |= {
            "tasks": len(o),
            "trials_per_task": trials,
            "pass_k": {f"pass^{k}": arm_pass_k(o, k).model_dump() for k in range(1, trials + 1)},
            "per_task_successes": {t: sum(v) for t, v in sorted(o.items())},
            "cost": {"usd_total": sum(cost), "usd_mean": statistics.fmean(cost),
                     "usd_max": max(cost), "external_usd": _external(dirs)},
            "terminations": dict(Counter(s.termination_reason.value for s in sims)),
            "infrastructure_errors": sum(s.reward_info is None for s in sims),
            "loop_guard_refusals": _guard_refusals(dirs),
        }
        retrieval_rows = []
        for d in dirs:
            trace = ROOT / d / "trace.jsonl"
            episodes = read_trace(trace) if trace.exists() else []
            if episodes:
                labels = required_documents({e.task_id for e in episodes})
                retrieval_rows += episode_scores(episodes, {k: set(v) for k, v in labels.items()})
        if not retrieval_rows and (entry["config"] or {}).get("env_variant") == "bm25":
            retrieval_rows = _tool_retrieval_rows(sims)
            entry["retrieval_source"] = "tau2 KB_search tool results"
        elif retrieval_rows:
            entry["retrieval_source"] = "retrieval trace"
        if retrieval_rows:
            entry["retrieval"] = _retrieval_summary(retrieval_rows)
        if entry["status"] == "complete":
            outcomes[arm] = o
        arms[arm] = entry
    from policygate_rag.eval.passk import bootstrap_mean

    retrieval_paired = {}
    with_retrieval = [a for a in ARM_ORDER if a in outcomes and "retrieval" in arms[a]]
    for a, b in itertools.combinations(with_retrieval, 2):
        ta, tb = arms[a]["retrieval"]["per_task"], arms[b]["retrieval"]["per_task"]
        shared = sorted(set(ta) & set(tb))
        retrieval_paired[f"{a} - {b}"] = {
            m: bootstrap_mean([ta[t][m] - tb[t][m] for t in shared
                               if m in ta[t] and m in tb[t]]).model_dump()
            for m in RETRIEVAL_METRICS
        }
    paired = {}
    present = [a for a in ARM_ORDER if a in outcomes]
    for a, b in itertools.combinations(present, 2):
        trials = min(min(map(len, outcomes[a].values())), min(map(len, outcomes[b].values())))
        paired[f"{a} - {b}"] = {f"pass^{k}": paired_difference(outcomes[a], outcomes[b], k
                                                               ).model_dump()
                                for k in range(1, trials + 1)}
    return {"arms": arms, "paired": paired, "retrieval_paired": retrieval_paired}


RETRIEVAL_METRICS = ("coverage", "precision", "mrr")
TOOL_RESULT_ID = re.compile(r"^\s*ID: (\S+)", re.M)


def _tool_retrieval_rows(sims, tool: str = "KB_search") -> list[dict]:
    """Retrieval through tau2's own search tool (arm A1), scored like the trace-based arms.

    Each call is a query, its result lists document ids in rank order; what an
    episode was shown is the union of every result.
    """
    from policygate_rag.eval.retrieval_metrics import coverage, precision, reciprocal_rank

    labels = required_documents({s.task_id for s in sims})
    rows = []
    for sim in sims:
        call_ids = {c.id for m in sim.messages if m.role == "assistant"
                    for c in (m.tool_calls or []) if c.name == tool}
        ranked = [TOOL_RESULT_ID.findall(m.content or "") for m in sim.messages
                  if m.role == "tool" and m.id in call_ids]
        shown = {d for r in ranked for d in r}
        req = set(labels[sim.task_id])
        rrs = [reciprocal_rank(r, req) for r in ranked]
        rows.append({"task_id": sim.task_id, "coverage": coverage(shown, req),
                     "precision": precision(shown, req),
                     "mrr": sum(rrs) / len(rrs) if rrs else None,
                     "shown": len(shown), "queries": len(rrs)})
    return rows


def _retrieval_per_task(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    out: dict[str, dict[str, float]] = {}
    for task, rs in sorted(by_task.items()):
        out[task] = {m: statistics.fmean(x[m] for x in rs if x[m] is not None)
                     for m in RETRIEVAL_METRICS if any(x[m] is not None for x in rs)}
    return out


def _retrieval_summary(rows: list[dict]) -> dict:
    from policygate_rag.eval.passk import bootstrap_mean

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    out: dict[str, Any] = {"per_task": _retrieval_per_task(rows)}
    for m in ("coverage", "precision", "mrr"):
        vals = [statistics.fmean(x[m] for x in rs if x[m] is not None)
                for rs in by_task.values() if any(x[m] is not None for x in rs)]
        if vals:
            out[m] = bootstrap_mean(vals).model_dump()
    out["docs_shown_median"] = statistics.median(r["shown"] for r in rows)
    out["queries_median"] = statistics.median(r["queries"] for r in rows)
    return out


def export_stage1(spec: dict) -> dict:
    from policygate_rag.eval.judge_bench import bench, load_scores, scored_everywhere
    from policygate_rag.tracing.read import Query

    d = ROOT / spec["dir"]
    if not (d / "queries.jsonl").exists():
        return {"status": "not run"}
    queries = [Query.model_validate_json(x)
               for x in (d / "queries.jsonl").read_text().splitlines()]
    judges = {p.stem.removeprefix("scores_"): load_scores(p)
              for p in sorted(d.glob("scores_*.jsonl"))}
    required = {k: set(v) for k, v in required_documents({q.task_id for q in queries}).items()}
    config = RagConfig()
    common = [q for q in queries if all(scored_everywhere(q, s) for s, _ in judges.values())]
    full = {n: j for n, j in judges.items() if all(scored_everywhere(q, j[0]) for q in queries)}
    costs_file, sample_file = d / "costs.json", d / "llm_sample.json"
    costs = json.loads(costs_file.read_text()) if costs_file.exists() else {}
    sample = json.loads(sample_file.read_text()) if sample_file.exists() else None
    return {
        "status": "complete",
        "harvested_from": "A2 stage-2 trace",
        "queries": len(queries),
        "unique_queries": len({q.query for q in queries}),
        "judges": {n: spec["judges"].get(n, {}) | {"cost": costs.get(n)} for n in judges},
        "llm_sample": None if sample is None else (
            {k: v for k, v in sample.items() if k != "queries"}
            | {"unique_queries": len(sample["queries"])}),
        "common": bench(common, judges, required, config),
        "full": bench(queries, full, required, config),
    }


def export_w1(manifest: dict) -> dict:
    from policygate_rag.eval.calibration import summarise

    out: dict[str, Any] = {}
    cal = ROOT / manifest["w1_calibration"]["dir"]
    out["calibration"] = summarise(cal) if (cal / "plan.json").exists() else {"status": "not run"}
    unguarded = manifest["w1_unguarded"]
    out["unguarded"] = {"superseded_by": unguarded["superseded_by"], "reason": unguarded["reason"]}
    out["unguarded"] |= export_arms({k: v for k, v in unguarded.items()
                                     if isinstance(v, dict) and "runs" in v},
                                    manifest["expected_episodes_per_arm"])
    return out


def export_spend() -> dict:
    from policygate_rag.cost.ledger import ALLOCATIONS, TOTAL_USD, Ledger

    ledger = Ledger.load(ROOT / "runs" / "ledger.json")
    external = {}
    stage1 = ROOT / "runs" / "stage1" / "costs.json"
    if stage1.exists():
        external["stage1_judges"] = {k: v["external_usd"]
                                     for k, v in json.loads(stage1.read_text()).items()
                                     if v["external_usd"]}
    for p in sorted((ROOT / "runs").glob("*/*/external_spend.json")):
        e = json.loads(p.read_text())
        external[f"{p.parent.parent.name}/{e['arm']}"] = e["usd"]
    return {
        "ceiling_usd": TOTAL_USD,
        "lines": {k.value: {"base_allocation": v, "allocated": ledger.allocation(k),
                            "spent": ledger.spent(k)}
                  for k, v in ALLOCATIONS.items()},
        "transfers": [t.model_dump(mode="json") for t in ledger.transfers],
        "total_spent": ledger.total(),
        "reconciliations": [e.model_dump(mode="json") for e in ledger.entries
                            if e.kind.value == "reconciliation"],
        "billed_elsewhere_usd": external,
    }


def main() -> None:
    manifest = json.loads((ROOT / "experiments" / "results_manifest.json").read_text())
    OUT.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    stage2 = export_arms(manifest["stage2"], manifest["expected_episodes_per_arm"])
    files = {
        "stage1_judges.json": export_stage1(manifest["stage1"]),
        "stage2_arms.json": stage2,
        "w1.json": export_w1(manifest),
        "spend.json": export_spend(),
    }
    for name, payload in files.items():
        (OUT / name).write_text(json.dumps({"schema_version": SCHEMA_VERSION} | payload,
                                           indent=1, default=str) + "\n")
    (OUT / "index.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "files": sorted(files),
        "arms": {a: e["status"] for a, e in stage2["arms"].items()},
        "intervals": ("95% percentile bootstrap over tasks, 10,000 resamples; "
                      "differences paired by task"),
    }, indent=1) + "\n")
    print(json.dumps({a: (e["status"], e["episodes"]) for a, e in stage2["arms"].items()}))


if __name__ == "__main__":
    main()
