"""Offline replay: what would the model have had in context, under other retrieval policies?

    python -m policygate_rag.eval.replay runs/stage2_a3/A3

Replays A3's recorded episodes (conversation from tau2's results, retrieval from
the trace) and measures **coverage at action time**: at every agent step that
proposes a tool call, the share of the task's required documents in the
model's context at that step. A3's reported coverage is the union over the
episode, which overstates what the model ever has at once.

Context policies, all fed only with what was available at that step:

* ``replace``      -- what A3 did: the top 5 of the latest user turn, replaced every user turn.
* ``ws{N}``        -- a working set of N documents kept across turns, oldest evicted first.
* source ``kept``  -- A3's top 5 per user turn enter the set; ``pool``: every candidate
                      the judge admitted that turn.
* ``+ent``         -- products named anywhere in the conversation so far (user, agent,
                      or tool output) add that product's summary (``_001``) and its best
                      BM25 match for the latest user message.
* ``+act``         -- a proposed tool call adds the documents that name the tool (for
                      discoverable tools, the tool named in the arguments), as a harness
                      that intercepts the call and retrieves before letting it through would.

Counterfactual on fixed conversations: a different context would have changed
the conversation, so these numbers bound what the policies can deliver, they do
not predict pass^1. No model calls and no network: BM25 and string matching only.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from policygate_rag.eval.passk import bootstrap_mean
from policygate_rag.eval.retrieval_metrics import required_documents
from policygate_rag.index.bm25 import BM25Index
from policygate_rag.index.corpus import Document, load_corpus
from policygate_rag.retrieval.products import DOC_ID, mentioned_products, product_names
from policygate_rag.tracing.read import last_attempt_rows, parse_thread

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "third_party/tau2-bench/data/tau2/domains/banking_knowledge/documents"
ACT_DOCS_PER_TOOL = 3
WRAPPER_TOOLS = frozenset({
    "unlock_discoverable_agent_tool", "call_discoverable_agent_tool", "give_discoverable_user_tool",
})
"""Generic wrappers: the tool they name is in their arguments; their own name is noise."""


# --- one episode as a sequence of agent steps -------------------------------

@dataclass
class Step:
    turn: int
    seen_text: str
    """Every message before this agent step, concatenated (what a lookup could read)."""
    last_user: str
    tool_calls: list[tuple[str, dict]]


@dataclass
class Episode:
    thread: str
    task_id: str
    steps: list[Step]
    retrieval: dict[int, dict] = field(default_factory=dict)
    """Turn -> A3's retrieval state for user turns."""


def read_retrievals(trace: Path) -> dict[str, dict[int, dict]]:
    """Per thread, turn -> A3's retrieval on user turns (last attempt only)."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in last_attempt_rows(trace):
        if row["type"] != "updates" or row["ns"] or "retrieve" not in row["data"]:
            continue
        r = (row["data"]["retrieve"] or {}).get("retrieval") or {}
        if r.get("mode") == "agentic" and not r.get("reused"):
            out[row["thread_id"]][row["turn"]] = r
    return dict(out)


def episode_steps(messages: Sequence) -> list[Step]:
    """Agent steps in order. The first assistant message is tau2's fixed greeting,
    not an agent turn, so agent turn k is assistant message k + 1."""
    steps, seen, last_user, k = [], [], "", -1
    for m in messages:
        if m.role == "assistant":
            if k >= 0:
                calls = [(c.name, c.arguments or {}) for c in (m.tool_calls or [])]
                steps.append(Step(turn=k, seen_text="\n".join(seen), last_user=last_user,
                                  tool_calls=calls))
            k += 1
        elif m.role == "user":
            last_user = m.content or ""
        if m.content:
            seen.append(m.content)
    return steps


def live_contexts(trace: Path) -> dict[tuple[str, str], dict[int, set[str]]]:
    """What was actually in front of the model, per (task, seed) and turn, from a
    trace: the working set where there is one, else the turn's retrieved documents
    (which a tool-result turn reuses)."""
    out: dict[tuple[str, str], dict[int, set[str]]] = defaultdict(dict)
    for row in last_attempt_rows(trace):
        if row["type"] != "updates" or row["ns"]:
            continue
        _, task, seed = parse_thread(row["thread_id"])
        data = row["data"]
        if "remember" in data:
            ctx = {d["doc_id"] for d in (data["remember"] or {}).get("context") or []}
        elif "retrieve" in data:
            r = (data["retrieve"] or {}).get("retrieval") or {}
            if r.get("mode") in (None, "none"):
                continue
            ctx = set(r.get("kept") or [])
        else:
            continue
        out[(task, seed)][row["turn"]] = ctx
    return dict(out)


def action_coverage(sims, trace: Path) -> dict[tuple[str, str], float]:
    """Per episode: mean share of required documents in context at the steps that
    propose a tool call. Episodes with no such step are left out."""
    contexts = live_contexts(trace)
    required = {k: set(v) for k, v in required_documents({s.task_id for s in sims}).items()}
    out = {}
    for s in sims:
        key = (s.task_id, str(s.seed))
        turns = contexts.get(key, {})
        req = required[s.task_id]
        covs = [len(turns.get(step.turn, set()) & req) / len(req)
                for step in episode_steps(s.messages) if step.tool_calls]
        if covs:
            out[key] = statistics.fmean(covs)
    return out


# --- context policies ------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    name: str
    capacity: int | None
    """``None``: replace on every user turn (A3). ``0``: unbounded working set."""
    source: str = "kept"
    entities: bool = False
    actions: bool = False


class Lookups:
    def __init__(self, docs: Sequence[Document]) -> None:
        self.docs = {d.doc_id: d for d in docs}
        self.bm25 = BM25Index(docs)
        self.names = product_names(self.docs)
        self.by_product: dict[str, list[str]] = defaultdict(list)
        for d in self.docs:
            m = DOC_ID.match(d)
            if m:
                self.by_product[m["product"]].append(d)

    def entity_docs(self, products: Iterable[str], step: Step) -> list[str]:
        out = []
        scores = dict(self.bm25.search(step.last_user, len(self.docs)))
        for product in sorted(products):
            members = sorted(self.by_product[product])
            out.append(members[0])
            rest = [d for d in members[1:] if scores.get(d, 0) > 0]
            if rest:
                out.append(max(rest, key=lambda d: (scores[d], d)))
        return out

    def action_docs(self, step: Step) -> list[str]:
        out = []
        for name, args in step.tool_calls:
            targets = ([] if name in WRAPPER_TOOLS else [name]) + [v for v in args.values()
                                if isinstance(v, str) and re.fullmatch(r"[a-z0-9_]+", v)
                                and "_" in v]
            for t in targets:
                hits = [(d.content.count(t), d.doc_id) for d in self.docs.values()
                        if t in d.content]
                hits.sort(key=lambda h: (-h[0], h[1]))
                out += [d for _, d in hits[:ACT_DOCS_PER_TOOL]]
        return out


def admitted(r: dict) -> list[str]:
    docs = []
    for rnd in r["rounds"]:
        prob = rnd.get("judge_is_probability", True)
        docs += [j["doc_id"] for j in rnd.get("judged", []) if not prob or j["score"] >= 0.5]
    return docs


def replay(ep: Episode, policy: Policy, look: Lookups) -> list[set[str]]:
    """The context at each step that proposes a tool call."""
    ws: OrderedDict[str, None] = OrderedDict()
    replaced: list[str] = []
    named: set[str] = set()
    contexts = []

    def add(ids: Iterable[str]) -> None:
        for d in ids:
            ws.pop(d, None)
            ws[d] = None
        while policy.capacity and len(ws) > policy.capacity:
            ws.popitem(last=False)

    for step in ep.steps:
        r = ep.retrieval.get(step.turn)
        if r is not None:
            new = (r.get("kept") or []) if policy.source == "kept" else admitted(r)
            replaced = list(r.get("kept") or [])
            # Pool order: lowest priority first, so the top-ranked survive eviction.
            add(reversed(new) if policy.source == "kept" else new)
        if policy.entities:
            # A product enters when first named, and again when the customer names it
            # in a new message; naming it once does not pin it in the set forever.
            now = mentioned_products(step.seen_text, look.names)
            fresh = now - named
            if r is not None:
                fresh |= mentioned_products(step.last_user, look.names)
            named |= now
            add(look.entity_docs(fresh, step))
        if policy.actions and step.tool_calls:
            add(look.action_docs(step))
        if step.tool_calls:
            contexts.append(set(replaced) if policy.capacity is None else set(ws))
    return contexts


POLICIES = (
    Policy("replace (A3)", None),
    Policy("ws10 kept", 10),
    Policy("ws20 kept", 20),
    Policy("ws20 pool", 20, "pool"),
    Policy("ws20 kept +ent", 20, entities=True),
    Policy("ws20 kept +act", 20, actions=True),
    Policy("ws20 kept +ent +act", 20, entities=True, actions=True),
    Policy("ws30 kept +ent +act", 30, entities=True, actions=True),
    Policy("ws30 pool +ent +act", 30, "pool", entities=True, actions=True),
    Policy("unbounded pool +ent +act", 0, "pool", entities=True, actions=True),
)


PAIRS = (
    ("ws10 kept", "replace (A3)"),
    ("ws20 kept", "ws10 kept"),
    ("ws20 pool", "ws20 kept"),
    ("ws20 kept +ent", "ws20 kept"),
    ("ws20 kept +act", "ws20 kept"),
    ("ws30 kept +ent +act", "ws20 kept +ent +act"),
)
"""Paired contrasts that isolate one change each: (policy, the policy it changes)."""


def load_episodes(run: Path) -> list[Episode]:
    from tau2.data_model.simulation import Results

    retrievals = read_retrievals(run / "trace.jsonl")
    episodes = []
    for p in sorted(run.glob("results_chunk_*.json")):
        for s in Results.load(p).simulations:
            thread = f"A3-{s.task_id}-seed{s.seed}"
            _, task, _ = parse_thread(thread)
            episodes.append(Episode(thread=thread, task_id=task,
                                    steps=episode_steps(s.messages),
                                    retrieval=retrievals.get(thread, {})))
    return episodes


def evaluate(episodes: Sequence[Episode], policies: Sequence[Policy], look: Lookups) -> dict:
    required = {k: set(v) for k, v in required_documents({e.task_id for e in episodes}).items()}
    per_task: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for ep in episodes:
        req = required[ep.task_id]
        for pol in policies:
            ctxs = replay(ep, pol, look)
            if not ctxs:
                continue
            covs = [len(c & req) / len(req) for c in ctxs]
            m = per_task[pol.name][ep.task_id]
            m["mean_action_coverage"].append(statistics.fmean(covs))
            m["max_action_coverage"].append(max(covs))
            m["docs_in_context"].append(statistics.fmean(len(c) for c in ctxs))
            m["tokens_in_context"].append(statistics.fmean(
                sum(len(look.docs[d].text) for d in c) / 4 for c in ctxs))
    means = {p: {t: {k: statistics.fmean(v) for k, v in ms.items()} for t, ms in tasks.items()}
             for p, tasks in per_task.items()}
    report: dict = {"episodes": len(episodes), "policies": {}, "paired": {}}
    for pol in policies:
        tm = means[pol.name]
        report["policies"][pol.name] = {
            k: bootstrap_mean([m[k] for m in tm.values()]).model_dump()
            for k in ("mean_action_coverage", "max_action_coverage",
                      "docs_in_context", "tokens_in_context")}
    names = [p.name for p in policies]
    for a, b in [(n, names[0]) for n in names[1:]] + [p for p in PAIRS if set(p) <= set(names)]:
        shared = sorted(set(means[a]) & set(means[b]))
        report["paired"][f"{a} - {b}"] = bootstrap_mean(
            [means[a][t]["mean_action_coverage"] - means[b][t]["mean_action_coverage"]
             for t in shared]).model_dump()
    return report


def main() -> None:
    run = Path(sys.argv[1])
    look = Lookups(load_corpus(DOCS))
    report = evaluate(load_episodes(run), POLICIES, look)
    (run.parent / "replay.json").write_text(json.dumps(report, indent=2))
    fmt = lambda i: f"{i['estimate']:.3f} [{i['low']:.3f}, {i['high']:.3f}]"  # noqa: E731
    print(f"{'policy':28s} {'mean cov @action':24s} {'max cov @action':24s} docs  tokens")
    for name, r in report["policies"].items():
        print(f"{name:28s} {fmt(r['mean_action_coverage']):24s} "
              f"{fmt(r['max_action_coverage']):24s} {r['docs_in_context']['estimate']:5.1f} "
              f"{r['tokens_in_context']['estimate']:6.0f}")
    print("\npaired, mean coverage at action time:")
    for name, i in report["paired"].items():
        print(f"  {name:48s} {i['estimate']:+.3f} [{i['low']:+.3f}, {i['high']:+.3f}]")


if __name__ == "__main__":
    main()
