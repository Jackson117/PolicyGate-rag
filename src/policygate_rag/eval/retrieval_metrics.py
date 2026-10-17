"""Retrieval metrics against required_documents. Labels are read here and only here.

Labels are per task, not per query, so every metric counts any of the task's
required documents as relevant:

* episode coverage  = |shown ∩ required| / |required|
* episode precision = |shown ∩ required| / |shown|     (undefined when nothing was shown)
* query MRR         = 1 / rank of the first required document, 0 if none is ranked

"Shown" is the union of the documents placed in the model's context across an
episode's turns, read from the retrieval trace. Queries and their ranked
candidates come from the same trace, so the offline judge benchmark scores
exactly the queries the arms issued.

Pure Python: no LangGraph, no network.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

THREAD = re.compile(r"^(?P<arm>[^-]+)-(?P<task>.+)-seed(?P<seed>-?\d+|None)$")


def coverage(shown: Iterable[str], required: Iterable[str]) -> float:
    required = set(required)
    if not required:
        raise ValueError("a task with no required documents has no coverage")
    return len(set(shown) & required) / len(required)


def precision(shown: Iterable[str], required: Iterable[str]) -> float | None:
    shown = set(shown)
    return None if not shown else len(shown & set(required)) / len(shown)


def reciprocal_rank(ranked: Sequence[str], required: Iterable[str]) -> float:
    required = set(required)
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in required:
            return 1.0 / rank
    return 0.0


class Query(BaseModel):
    """One retrieval query as issued, with the ranked candidates it got."""

    model_config = ConfigDict(frozen=True)

    arm: str
    task_id: str
    seed: str
    turn: int
    round: int
    query: str
    candidates: tuple[str, ...]
    shown: tuple[str, ...]


class EpisodeRetrieval(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    task_id: str
    seed: str
    shown: frozenset[str]
    queries: tuple[Query, ...]


def parse_thread(thread_id: str) -> tuple[str, str, str]:
    m = THREAD.match(thread_id)
    if m is None:
        raise ValueError(f"not an episode thread id: {thread_id!r}")
    return m["arm"], m["task"], m["seed"]


def read_trace(path: Path) -> list[EpisodeRetrieval]:
    """Every episode's retrieval from one arm's trace.jsonl."""
    shown: dict[str, set[str]] = defaultdict(set)
    queries: dict[str, list[Query]] = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["type"] != "updates" or row["ns"] or "retrieve" not in row["data"]:
            continue
        r = (row["data"]["retrieve"] or {}).get("retrieval") or {}
        if r.get("mode") in (None, "none"):
            continue
        thread = row["thread_id"]
        arm, task, seed = parse_thread(thread)
        shown[thread].update(r.get("kept") or [])
        if r.get("reused"):
            continue
        rounds = r.get("rounds") or [{"query": r["query"], "candidates": r["candidates"]}]
        for i, rnd in enumerate(rounds):
            queries[thread].append(Query(
                arm=arm, task_id=task, seed=seed, turn=row["turn"], round=i,
                query=rnd["query"],
                candidates=tuple(c["doc_id"] for c in rnd["candidates"]),
                shown=tuple(r.get("kept") or []),
            ))
    out = []
    for thread in sorted(shown.keys() | queries.keys()):
        arm, task, seed = parse_thread(thread)
        out.append(EpisodeRetrieval(arm=arm, task_id=task, seed=seed,
                                    shown=frozenset(shown[thread]),
                                    queries=tuple(queries[thread])))
    return out


def episode_scores(
    episodes: Sequence[EpisodeRetrieval], required: Mapping[str, Iterable[str]]
) -> list[dict]:
    """Coverage, precision and mean query MRR per episode."""
    rows = []
    for e in episodes:
        req = set(required[e.task_id])
        rrs = [reciprocal_rank(q.candidates, req) for q in e.queries]
        rows.append({
            "arm": e.arm, "task_id": e.task_id, "seed": e.seed,
            "coverage": coverage(e.shown, req),
            "precision": precision(e.shown, req),
            "mrr": sum(rrs) / len(rrs) if rrs else None,
            "shown": len(e.shown), "required": len(req), "queries": len(rrs),
        })
    return rows


def required_documents(task_ids: Iterable[str] | None = None) -> dict[str, frozenset[str]]:
    """The labels, from tau2's loader (never the stale aggregate tasks.json)."""
    from tau2.domains.banking_knowledge.environment import get_tasks

    wanted = None if task_ids is None else set(task_ids)
    return {
        t.id: frozenset(t.required_documents or ())
        for t in get_tasks() if wanted is None or t.id in wanted
    }
