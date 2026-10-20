"""Reading retrieval traces back: which queries were issued, what they returned,
what reached the model. Label-free, so any package may use it; scoring against
labels happens in ``eval/``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

THREAD = re.compile(r"^(?P<arm>[^-]+)-(?P<task>.+)-seed(?P<seed>-?\d+|None)$")


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


def last_attempt_rows(path: Path) -> list[dict]:
    """The trace's rows, keeping each thread's last attempt only.

    An episode stopped by a stall and re-run with ``--resume`` appears twice under
    the same thread id. The re-run's turn numbers restart at 0, and only it is the
    episode tau2 scored, so an earlier attempt's rows are dropped. Attempts are
    told apart on graph updates only; other row types carry no ordering promise.
    """
    rows: dict[str, list[dict]] = defaultdict(list)
    last_turn: dict[str, int] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        thread, turn = row["thread_id"], row["turn"]
        if row["type"] == "updates":
            if turn < last_turn.get(thread, -1):
                rows[thread] = []
            last_turn[thread] = turn
        rows[thread].append(row)
    return [row for thread_rows in rows.values() for row in thread_rows]


def read_trace(path: Path) -> list[EpisodeRetrieval]:
    """Every episode's retrieval from one arm's trace.jsonl (last attempts only)."""
    shown: dict[str, set[str]] = defaultdict(set)
    queries: dict[str, list[Query]] = defaultdict(list)
    for row in last_attempt_rows(path):
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
