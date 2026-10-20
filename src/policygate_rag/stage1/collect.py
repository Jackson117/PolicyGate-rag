"""Stage 1, step 1: collect every judge's scores on the queries naive RAG issued.

    python -m policygate_rag.stage1.collect --trace runs/stage2_guarded/A2/trace.jsonl \\
        --judges none cross_encoder llm jev --dry-run

Queries and their fused candidates come from the A2 trace, so every judge scores
exactly what naive retrieval returned. The top ``k_wide`` candidates of each
query are scored -- the candidate set the agentic arm would judge. Each unique
(query text, document) pair is scored once per judge: trials of a task often
open with the same message, and paying twice for one judgement adds nothing.

Spend: the LLM judge is booked to the stage-1 line and checked against it
before any call; Jev bills its own account and is tallied separately. Scores
are appended to ``scores_<judge>.jsonl`` as they come, and a re-run skips what
is already there, so an interrupted collection resumes rather than repays.

No labels are read here. Scoring against the labels is
``eval/judge_bench.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from policygate_rag.config import RagConfig
from policygate_rag.index.corpus import Document, load_corpus
from policygate_rag.judges.base import Candidate
from policygate_rag.tracing.read import Query, read_trace

ROOT = Path(__file__).resolve().parents[3]
JUDGES = ("none", "cross_encoder", "llm", "jev")


def harvest(traces: Iterable[Path], k_wide: int) -> list[Query]:
    """Every fresh query in the given traces, candidates cut to ``k_wide``."""
    out = []
    for trace in traces:
        for episode in read_trace(trace):
            for q in episode.queries:
                out.append(q.model_copy(update={"candidates": q.candidates[:k_wide]}))
    return out


def unique_pairs(queries: Iterable[Query]) -> dict[str, list[str]]:
    """Query text -> its candidate ids, first-seen order, deduplicated."""
    pairs: dict[str, list[str]] = {}
    for q in queries:
        seen = pairs.setdefault(q.query, [])
        seen.extend(d for d in q.candidates if d not in seen)
    return pairs


def already_scored(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(r["query"], r["doc_id"]) for r in map(json.loads, path.read_text().splitlines())}


def estimate(pairs: dict[str, list[str]], docs: dict[str, Document]) -> dict[str, float]:
    """Rough input tokens (~4 chars/token) the paid judges would read."""
    chars = sum(len(q) + len(docs[d].title) + len(docs[d].content)
                for q, ids in pairs.items() for d in ids)
    tokens = chars / 4
    return {
        "pairs": sum(len(v) for v in pairs.values()),
        "tokens": tokens,
        "llm_usd": tokens * 0.40 / 1e6 * 1.15,  # gpt-4.1-mini input, +15% for prompts and output
        "jev_usd": (tokens + 150 * sum(len(v) for v in pairs.values())) * 0.042 / 1e6,
    }


def build_judge(name: str, config: RagConfig):
    if name == "none":
        from policygate_rag.judges.none import NoJudge
        return NoJudge()
    if name == "cross_encoder":
        raise ValueError("score the cross-encoder with scripts/cross_encoder_score.py in its "
                         "own environment (--export-pairs first); see that script")
    if name == "llm":
        from policygate_rag.judges.llm import LLMJudge
        from policygate_rag.judges.llm_client import make_ask
        return LLMJudge(make_ask(config.effective_aux_llm, temperature=0.0),
                        batch_size=config.judge_batch_size)
    if name == "jev":
        from policygate_rag.judges.jev import JevJudge
        from policygate_rag.judges.jev_client import JevClient
        return JevJudge(JevClient().nouls)
    raise ValueError(f"unknown judge {name!r}")


def collect(name: str, judge, pairs: dict[str, list[str]], docs: dict[str, Document],
            fused: dict[tuple[str, str], float], out: Path, *, on_spend=None) -> dict:
    """Score every pair not already in ``out``; return totals for this call."""
    done = already_scored(out)
    totals = {"pairs": 0, "usd": 0.0, "external_usd": 0.0}
    out.parent.mkdir(parents=True, exist_ok=True)
    for query, ids in pairs.items():
        todo = [d for d in ids if (query, d) not in done]
        if not todo:
            continue
        verdict = judge.judge(query, [Candidate(doc=docs[d], fused_score=fused[(query, d)])
                                      for d in todo])
        with out.open("a") as f:
            for d, s in zip(todo, verdict.scores, strict=True):
                f.write(json.dumps({"judge": name, "query": query, "doc_id": d, "score": s,
                                    "is_probability": verdict.is_probability}) + "\n")
        totals["pairs"] += len(todo)
        totals["usd"] += verdict.usd
        totals["external_usd"] += verdict.external_usd
        if on_spend is not None and verdict.usd:
            on_spend(verdict.usd)
    return totals


def export_pairs(pairs: dict[str, list[str]], docs: dict[str, Document], path: Path) -> None:
    """Pairs with the text the cross-encoder reads, as judges/cross_encoder.py builds it."""
    from policygate_rag.judges.cross_encoder import MAX_DOC_CHARS

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for query, ids in pairs.items():
            for d in ids:
                text = f"{docs[d].title}\n{docs[d].content[:MAX_DOC_CHARS]}"
                f.write(json.dumps({"query": query, "doc_id": d, "text": text}) + "\n")


def sample_per_task(queries: Iterable[Query], per_task: int, seed: int) -> set[str]:
    """Up to ``per_task`` unique query texts from each task, drawn with a fixed seed.

    Every task stays in the sample, so the judge comparison keeps its unit of
    analysis (tasks) while paying for fewer queries within each.
    """
    import random

    by_task: dict[str, list[str]] = {}
    for q in queries:
        texts = by_task.setdefault(q.task_id, [])
        if q.query not in texts:
            texts.append(q.query)
    rng = random.Random(seed)
    chosen: set[str] = set()
    for task in sorted(by_task):
        texts = sorted(by_task[task])
        chosen.update(rng.sample(texts, min(per_task, len(texts))))
    return chosen


def largest_sample_within(queries: list[Query], pairs: dict[str, list[str]],
                          docs: dict[str, Document], budget: float, seed: int) -> int:
    """The largest per-task count whose estimated LLM-judge cost fits ``budget``."""
    best = 0
    for n in range(1, 200):
        chosen = sample_per_task(queries, n, seed)
        if estimate({q: pairs[q] for q in chosen}, docs)["llm_usd"] > budget:
            break
        best = n
        if len(chosen) == len(pairs):
            break
    return best


def fused_scores(queries: Iterable[Query]) -> dict[tuple[str, str], float]:
    """A stand-in RRF score per (query, doc) from rank, for the NONE judge's order."""
    out = {}
    for q in queries:
        for rank, d in enumerate(q.candidates, start=1):
            out.setdefault((q.query, d), 1 / (60 + rank))
    return out


def record_costs(path: Path, judge: str, totals: dict) -> None:
    """Accumulate each judge's pairs and spend across collection runs."""
    costs = json.loads(path.read_text()) if path.exists() else {}
    c = costs.setdefault(judge, {"pairs": 0, "usd": 0.0, "external_usd": 0.0})
    for k in c:
        c[k] += totals[k]
    path.write_text(json.dumps(costs, indent=1))


def main(argv=None) -> int:
    from policygate_rag.cost.ledger import BudgetLine, Entry, EntryKind, Ledger
    from policygate_rag.retrieval.build_index import DOCS_DIR, LEDGER

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--trace", type=Path, nargs="+", required=True)
    p.add_argument("--judges", nargs="+", choices=JUDGES, default=list(JUDGES))
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "stage1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export-pairs", type=Path,
                   help="write the unique pairs for scripts/cross_encoder_score.py and stop")
    p.add_argument("--llm-budget", type=float, default=None,
                   help="score only a per-task sample of queries with the LLM judge, as "
                        "large as fits this many USD (estimated)")
    p.add_argument("--seed", type=int, default=20261020)
    args = p.parse_args(argv)

    config = RagConfig()
    docs = {d.doc_id: d for d in load_corpus(DOCS_DIR)}
    queries = harvest(args.trace, config.k_wide)
    pairs = unique_pairs(queries)
    est = estimate(pairs, docs)
    print(f"{len(queries)} queries ({len(pairs)} unique), {est['pairs']} unique pairs, "
          f"~{est['tokens'] / 1e6:.2f}M tokens: llm ~{est['llm_usd']:.2f} USD (stage 1), "
          f"jev ~{est['jev_usd']:.2f} USD (Jev account)")
    if args.dry_run:
        return 0
    if args.export_pairs:
        export_pairs(pairs, docs, args.export_pairs)
        (args.out).mkdir(parents=True, exist_ok=True)
        (args.out / "queries.jsonl").write_text(
            "".join(q.model_dump_json() + "\n" for q in queries))
        print(f"wrote {est['pairs']} pairs to {args.export_pairs}")
        return 0

    ledger = Ledger.load(LEDGER)
    line = BudgetLine.STAGE1
    llm_pairs = pairs
    if "llm" in args.judges and args.llm_budget is not None:
        per_task = largest_sample_within(queries, pairs, docs, args.llm_budget, args.seed)
        chosen = sample_per_task(queries, per_task, args.seed)
        llm_pairs = {q: ids for q, ids in pairs.items() if q in chosen}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "llm_sample.json").write_text(json.dumps(
            {"per_task": per_task, "seed": args.seed, "budget_usd": args.llm_budget,
             "queries": sorted(chosen)}, indent=1))
        sub = estimate(llm_pairs, docs)
        print(f"LLM judge sample: {per_task} per task, {len(chosen)} unique queries, "
              f"{sub['pairs']} pairs, ~{sub['llm_usd']:.2f} USD")
    if "llm" in args.judges:
        remaining = ledger.allocation(line) - ledger.spent(line)
        if estimate(llm_pairs, docs)["llm_usd"] > remaining:
            print(f"LLM judge ~{est['llm_usd']:.2f} USD exceeds the {remaining:.2f} left on "
                  "the stage-1 line; not starting", file=sys.stderr)
            return 2

    def book(usd: float) -> None:
        ledger.record(Entry(line=line, kind=EntryKind.CALL, usd=usd,
                            model=config.effective_aux_llm, label="stage 1: LLM judge"))
        if ledger.spent(line) > ledger.allocation(line):
            raise RuntimeError("stage-1 line exhausted mid-collection")

    fused = fused_scores(queries)
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "queries.jsonl").write_text(
        "".join(q.model_dump_json() + "\n" for q in queries))
    for name in args.judges:
        totals = collect(name, build_judge(name, config),
                         llm_pairs if name == "llm" else pairs, docs, fused,
                         args.out / f"scores_{name}.jsonl", on_spend=book)
        print(f"{name}: {totals['pairs']} new pairs, {totals['usd']:.4f} USD (stage 1), "
              f"{totals['external_usd']:.4f} USD (external)")
        record_costs(args.out / "costs.json", name, totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
