"""The working set: documents kept in front of the model across turns (arm A5).

    START -> retrieve -> remember -> reason -> ...

A2 and A3 replace the documents on every user turn, so the model never has more
than k_final of them at once, against a median of 10 required per task. Here
each turn's retrieved documents join the previous turn's set instead, and the
oldest leave once it holds ``context_capacity``. With ``product_lookup``, a
product named for the first time -- or named again by the customer -- adds its
documents too, on any turn, including those that begin with a tool result.

Order is oldest first, so the newest documents sit last in the prompt; a
document that comes back is moved to the end rather than duplicated. Every
turn records what was added and evicted in ``working_set``.
"""

from __future__ import annotations

from collections import OrderedDict

from langgraph.runtime import Runtime

from policygate_rag.config import RagConfig
from policygate_rag.graph.state import TurnContext, TurnState
from policygate_rag.retrieval.naive import RetrievalResources, user_query


def _text(messages: list[dict]) -> str:
    return "\n".join(m.get("content") or "" for m in messages if isinstance(m.get("content"), str))


def named_now(messages: list[dict], lookup) -> set[str]:
    """Products named in the messages since the agent last spoke, but never before;
    plus those the customer names in a new message, which brings them back."""
    last_spoke = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"),
                     default=-1)
    before = lookup.mentioned(_text(messages[: last_spoke + 1]))
    fresh = lookup.mentioned(_text(messages)) - before
    if messages and messages[-1].get("role") == "user":
        fresh |= lookup.mentioned(messages[-1].get("content") or "")
    return fresh


def build_remember(config: RagConfig):
    capacity = config.context_capacity
    if capacity is None:
        raise ValueError("the working set needs context_capacity")

    def remember(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        res: RetrievalResources = runtime.context.retrieval
        ws: OrderedDict[str, dict] = OrderedDict(
            (d["doc_id"], d) for d in state.get("memory") or [])

        def add(docs: list[dict]) -> list[str]:
            for d in docs:
                ws.pop(d["doc_id"], None)
                ws[d["doc_id"]] = d
            return [d["doc_id"] for d in docs]

        retrieval = state.get("retrieval") or {}
        fresh_turn = user_query(state) is not None and not retrieval.get("reused")
        # The retrieval's own ranking puts its best first; add it last so it is
        # the newest and the last to be evicted.
        retrieved = add(list(reversed(state.get("context") or []))) if fresh_turn else []
        products, from_products = [], []
        if config.product_lookup:
            messages = state.get("messages") or []
            products = sorted(named_now(messages, res.products))
            ids = res.products.documents(products, user_query(state) or "")
            from_products = add([{"doc_id": d, "title": res.docs[d].title,
                                  "content": res.docs[d].content, "score": 0.0} for d in ids])
        evicted = []
        while len(ws) > capacity:
            evicted.append(ws.popitem(last=False)[0])
        return {
            "context": list(ws.values()),
            "working_set": {"retrieved": retrieved, "products": products,
                            "from_products": from_products, "evicted": evicted,
                            "size": len(ws)},
        }

    return remember
