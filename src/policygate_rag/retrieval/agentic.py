"""Agentic RAG (arm A3): retrieve, judge, check sufficiency, rewrite, repeat.

    START -> plan --(tool-result turn)--------------------------------> END  (documents kept)
               \\--(user turn)--> retrieve_wide --(nothing new)--> finish -> END
                                     ^             \\--> judge -> filter -> sufficiency
                                     |                                       |
                                     +----------- rewrite <--(not enough, rounds left)
                                                  sufficiency --(enough, or budget spent)--> finish

Like the naive arm it runs on user turns only and its first query is the user's
message verbatim, so the two arms differ only in what happens next. The
sufficiency check and the rewrite see the recent conversation, not just the
latest message: on a turn like "My name is Kenji Tanaka" the verbatim query
retrieves nothing useful, and it is the conversation that tells the check what
the customer actually needs and the rewrite what to search for.

Every loop ends with a recorded :class:`~policygate_rag.config.StopReason`.
Model calls here (sufficiency, rewrite) and query embeddings are added to the
turn's cost, which tau2 sees on the emitted message.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from tau2.data_model.message import SystemMessage, UserMessage

from policygate_rag.config import RagConfig, RetrievalMode, StopReason
from policygate_rag.graph.state import TurnContext, TurnState
from policygate_rag.judges.base import Candidate
from policygate_rag.retrieval.naive import RetrievalResources, user_query

SUFFICIENCY_PROMPT = """You check whether retrieved knowledge base documents are enough for a \
bank customer service agent to handle the customer's current need.

Read the conversation to work out what the customer currently needs. If the latest message \
only supplies details (a name, a date of birth) for a request made earlier, judge against that \
earlier request. If the customer's need requires no bank policy or product knowledge at all, \
the documents are sufficient.

Reply in JSON: {"probability": <0..1, how likely the documents contain what the agent needs>, \
"missing": "<what information is still missing, or empty>"}"""

REWRITE_PROMPT = """You write search queries for a bank's internal knowledge base of policy and \
product documents. Given the conversation, the queries already tried and what is still \
missing, write one new query that is likely to find the missing information. Use the bank's \
terms (product names, procedures), not the customer's personal details.

Reply in JSON: {"query": "<the new query>"}"""

MAX_MESSAGE_CHARS = 600


def recent_conversation(state: TurnState, window: int) -> str:
    """The last ``window`` user and assistant text messages, oldest first."""
    lines = []
    for m in state.get("messages") or []:
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip():
            lines.append(f"{m['role']}: {m['content'].strip()[:MAX_MESSAGE_CHARS]}")
    return "\n".join(lines[-window:])


def _ask_json(ctx: TurnContext, model: str, system: str, user: str, call_name: str):
    """One JSON-mode model call. Returns (parsed or None, usd, raw text)."""
    response = ctx.generate(
        model=model,
        messages=[SystemMessage(role="system", content=system),
                  UserMessage(role="user", content=user)],
        tools=None,
        call_name=call_name,
        response_format={"type": "json_object"},
        **ctx.llm_args,
    )
    raw = response.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    return (parsed if isinstance(parsed, dict) else None), (response.cost or 0.0), raw


def _top(pool: list[dict], k: int) -> list[dict]:
    return sorted(pool, key=lambda p: (-p["score"], p["doc_id"]))[:k]


def build_agentic(config: RagConfig):
    if config.retrieval is not RetrievalMode.AGENTIC:
        raise ValueError(f"agentic subgraph built for retrieval={config.retrieval.value}")
    aux_llm = config.effective_aux_llm

    def plan(state: TurnState) -> TurnState:
        query = user_query(state)
        if query is None:
            kept = [d["doc_id"] for d in state.get("context") or []]
            return {"retrieval": {"mode": "agentic", "reused": True, "kept": kept}}
        return {"retrieval": {
            "mode": "agentic", "reused": False, "query": query,
            "next_query": query, "rounds": [], "pool": [], "seen": [], "usd": 0.0,
        }}

    def after_plan(state: TurnState) -> Literal["retrieve_wide", "__end__"]:
        return END if state["retrieval"]["reused"] else "retrieve_wide"

    def retrieve_wide(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        r = dict(state["retrieval"])
        res: RetrievalResources = runtime.context.retrieval
        query = r["next_query"]
        embedded = res.embedder.embed([query])
        fused = res.index.search(query, embedded.vectors[0], k_bm25=config.k_bm25,
                                 k_dense=config.k_dense, rrf_k=config.rrf_k)[: config.k_wide]
        new = [f for f in fused if f.doc_id not in set(r["seen"])]
        r["rounds"] = r["rounds"] + [{
            "query": query,
            "candidates": [f.model_dump(mode="json") for f in fused],
            "new": [f.doc_id for f in new],
            "usd": embedded.usd,
        }]
        r["seen"] = r["seen"] + [f.doc_id for f in new]
        r["usd"] += embedded.usd
        return {"retrieval": r, "cost_usd": state.get("cost_usd", 0.0) + embedded.usd}

    def after_retrieve(state: TurnState) -> Literal["judge", "finish"]:
        return "judge" if state["retrieval"]["rounds"][-1]["new"] else "finish"

    def judge(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        r = dict(state["retrieval"])
        res: RetrievalResources = runtime.context.retrieval
        last = dict(r["rounds"][-1])
        by_id = {c["doc_id"]: c for c in last["candidates"]}
        candidates = [Candidate(doc=res.docs[d], fused_score=by_id[d]["score"])
                      for d in last["new"]]
        verdict = res.judge.judge(last["query"], candidates)
        last["judged"] = [{"doc_id": c.doc.doc_id, "score": s}
                          for c, s in zip(candidates, verdict.scores, strict=True)]
        last["judge_is_probability"] = verdict.is_probability
        last["judge_external_usd"] = verdict.external_usd
        r["rounds"] = r["rounds"][:-1] + [last]
        r["usd"] += verdict.usd
        return {"retrieval": r, "cost_usd": state.get("cost_usd", 0.0) + verdict.usd}

    def filter_(state: TurnState) -> TurnState:
        r = dict(state["retrieval"])
        last = r["rounds"][-1]
        admitted = [
            j for j in last["judged"]
            if not last["judge_is_probability"] or j["score"] >= config.judge_threshold
        ]
        r["pool"] = r["pool"] + admitted
        r["kept"] = [p["doc_id"] for p in _top(r["pool"], config.k_final)]
        return {"retrieval": r}

    def sufficiency(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        r = dict(state["retrieval"])
        res: RetrievalResources = runtime.context.retrieval
        docs = "\n\n".join(
            f"<document title=\"{res.docs[d].title}\">\n{res.docs[d].content}\n</document>"
            for d in r["kept"]
        ) or "(no documents)"
        parsed, usd, raw = _ask_json(
            runtime.context, aux_llm, SUFFICIENCY_PROMPT,
            f"<conversation>\n{recent_conversation(state, config.conversation_window)}\n"
            f"</conversation>\n\n<documents>\n{docs}\n</documents>",
            "sufficiency_check",
        )
        last = dict(r["rounds"][-1])
        try:
            probability = min(1.0, max(0.0, float(parsed["probability"])))
            last["missing"] = str(parsed.get("missing") or "")
        except (TypeError, KeyError, ValueError):
            probability, last["missing"], last["sufficiency_error"] = 0.0, "", raw[:200]
        last["sufficiency"] = probability
        last["usd"] += usd
        r["rounds"] = r["rounds"][:-1] + [last]
        r["usd"] += usd
        return {"retrieval": r, "cost_usd": state.get("cost_usd", 0.0) + usd}

    def after_sufficiency(state: TurnState) -> Literal["rewrite", "finish"]:
        r = state["retrieval"]
        if r["rounds"][-1]["sufficiency"] >= config.sufficiency_threshold:
            return "finish"
        if len(r["rounds"]) >= config.agentic_max_rounds:
            return "finish"
        return "rewrite"

    def rewrite(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        r = dict(state["retrieval"])
        tried = "\n".join(f"- {x['query']}" for x in r["rounds"])
        parsed, usd, raw = _ask_json(
            runtime.context, aux_llm, REWRITE_PROMPT,
            f"<conversation>\n{recent_conversation(state, config.conversation_window)}\n"
            f"</conversation>\n\n<tried>\n{tried}\n</tried>\n\n"
            f"<missing>{r['rounds'][-1].get('missing', '')}</missing>",
            "query_rewrite",
        )
        query = (parsed or {}).get("query")
        if not isinstance(query, str) or not query.strip():
            # Fall back to searching for what was reported missing, then to the
            # original request; never invent a query.
            query = r["rounds"][-1].get("missing") or r["query"]
            r["rewrite_error"] = raw[:200]
        r["next_query"] = query.strip()
        r["usd"] += usd
        return {"retrieval": r, "cost_usd": state.get("cost_usd", 0.0) + usd}

    def finish(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        r = dict(state["retrieval"])
        res: RetrievalResources = runtime.context.retrieval
        last = r["rounds"][-1]
        if not last["new"]:
            reason = StopReason.NO_NEW_DOCS
        elif last.get("sufficiency", 0.0) >= config.sufficiency_threshold:
            reason = StopReason.SUFFICIENT
        else:
            reason = StopReason.BUDGET_EXHAUSTED
        kept = _top(r["pool"], config.k_final)
        r["kept"] = [p["doc_id"] for p in kept]
        r["stop_reason"] = reason.value
        context: list[dict[str, Any]] = [
            {"doc_id": p["doc_id"], "title": res.docs[p["doc_id"]].title,
             "content": res.docs[p["doc_id"]].content, "score": p["score"]}
            for p in kept
        ]
        return {"retrieval": r, "context": context}

    graph = StateGraph(TurnState, context_schema=TurnContext)
    graph.add_node("plan", plan)
    graph.add_node("retrieve_wide", retrieve_wide)
    graph.add_node("judge", judge)
    graph.add_node("filter", filter_)
    graph.add_node("sufficiency", sufficiency)
    graph.add_node("rewrite", rewrite)
    graph.add_node("finish", finish)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan)
    graph.add_conditional_edges("retrieve_wide", after_retrieve)
    graph.add_edge("judge", "filter")
    graph.add_edge("filter", "sufficiency")
    graph.add_conditional_edges("sufficiency", after_sufficiency)
    graph.add_edge("rewrite", "retrieve_wide")
    graph.add_edge("finish", END)
    return graph.compile()
