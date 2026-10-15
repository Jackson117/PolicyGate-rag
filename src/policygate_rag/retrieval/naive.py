"""Naive RAG (arm A2): one hybrid retrieval per user turn, top-k into the prompt.

    START -> plan --(turn starts with a user message)--> search -> END
                  \\--(it starts with a tool result)------------> END   (documents kept)

The query is the user's latest message, verbatim. Rewriting it is what makes
the agentic arm agentic, so the naive arm does not. On turns that begin with a
tool result there is no new request to retrieve for; the documents retrieved
for the last user message stay in the prompt until the next one replaces them,
so context never accumulates across turns.

Every turn records what retrieval did -- the query, every fused candidate with
its rank in each source, the documents kept, the embedding cost -- in the
``retrieval`` field, which the trace keeps. Only ids leave this module; whether
they were the right ones is decided in ``eval/`` against the labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from policygate_rag.config import RagConfig, RetrievalMode
from policygate_rag.graph.state import TurnContext, TurnState
from policygate_rag.index.corpus import Document
from policygate_rag.index.fusion import HybridIndex
from policygate_rag.retrieval.embedder import Embedder


@dataclass
class RetrievalResources:
    """What retrieval needs at run time. Built once per arm, shared by its episodes."""

    index: HybridIndex
    docs: Mapping[str, Document]
    embedder: Embedder


def user_query(state: TurnState) -> str | None:
    """The latest message if it is a user message with text; otherwise None."""
    messages = state.get("messages") or []
    if not messages:
        return None
    last = messages[-1]
    if last.get("role") != "user":
        return None
    text = (last.get("content") or "").strip()
    return text or None


def build_naive(config: RagConfig):
    if config.retrieval is not RetrievalMode.NAIVE:
        raise ValueError(f"naive subgraph built for retrieval={config.retrieval.value}")

    def plan(state: TurnState) -> TurnState:
        query = user_query(state)
        if query is None:
            kept = [d["doc_id"] for d in state.get("context") or []]
            return {"retrieval": {"mode": "naive", "reused": True, "kept": kept}}
        return {"retrieval": {"mode": "naive", "reused": False, "query": query}}

    def route(state: TurnState) -> Literal["search", "__end__"]:
        return END if state["retrieval"]["reused"] else "search"

    def search(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        resources: RetrievalResources = runtime.context.retrieval
        query = state["retrieval"]["query"]
        embedded = resources.embedder.embed([query])
        fused = resources.index.search(
            query, embedded.vectors[0],
            k_bm25=config.k_bm25, k_dense=config.k_dense, rrf_k=config.rrf_k,
        )
        kept = fused[: config.k_final]
        context = [
            {
                "doc_id": f.doc_id,
                "title": resources.docs[f.doc_id].title,
                "content": resources.docs[f.doc_id].content,
                "score": f.score,
            }
            for f in kept
        ]
        return {
            "context": context,
            "retrieval": state["retrieval"] | {
                "candidates": [f.model_dump(mode="json") for f in fused],
                "kept": [f.doc_id for f in kept],
                "embedding_usd": embedded.usd,
                "embedding_tokens": embedded.tokens,
            },
            "cost_usd": state.get("cost_usd", 0.0) + embedded.usd,
        }

    graph = StateGraph(TurnState, context_schema=TurnContext)
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route)
    graph.add_edge("search", END)
    return graph.compile()
