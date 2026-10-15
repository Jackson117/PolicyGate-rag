"""The per-turn decision graph.

tau2 owns the outer loop and executes every tool, so this graph runs once per
``generate_next_message`` call and returns exactly one AssistantMessage. It
never executes a tool and never loops across turns::

    START -> retrieve -> reason -> gate --(allowed or text)--> END
                           ^        |
                           +--------+ refused, attempts left
                                    |
                                    +--(budget spent)--> escalate -> END

The reason -> gate loop is the harness's internal retry: a refused proposal
never reaches tau2; the refusal goes back to the model as an errored tool
result and it proposes again, within ``max_internal_attempts``. This is the
same protocol policygate's own tau2 agent runs -- both call
:class:`~policygate.harness.session.HarnessSession` -- so "harness on" means the
same thing here as in policygate's results.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from policygate.harness.session import HarnessSession
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage

from policygate_rag.config import RagConfig, RetrievalMode
from policygate_rag.graph.state import MESSAGES, TurnContext, TurnState
from policygate_rag.retrieval.naive import build_naive


def _dump(message) -> dict:
    return message.model_dump(mode="json")


def build_graph(config: RagConfig, *, checkpointer=None):
    """Compile the graph for one arm. ``checkpointer`` is a LangGraph saver or None."""
    if config.max_history_tokens is not None:
        raise NotImplementedError("history trimming is a W2 item; max_history_tokens must be None")

    def no_retrieval(state: TurnState) -> TurnState:
        return {"context": [], "retrieval": {"mode": RetrievalMode.NONE.value}}

    if config.retrieval is RetrievalMode.NONE:
        retrieve = no_retrieval
    elif config.retrieval is RetrievalMode.NAIVE:
        retrieve = build_naive(config)
    else:
        raise NotImplementedError(f"retrieval={config.retrieval.value} is built later in W2")

    def reason(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        ctx = runtime.context
        history = MESSAGES.validate_python(state["messages"] + state.get("scratch", []))
        offered = set(ctx.session.offered(t.name for t in ctx.tools))
        response = ctx.generate(
            model=ctx.llm,
            tools=[t for t in ctx.tools if t.name in offered],
            messages=[SystemMessage(role="system", content=_system(ctx, state))] + history,
            call_name="agent_response",
            **ctx.llm_args,
        )
        return {
            "scratch": state.get("scratch", []) + [_dump(response)],
            "attempts": state.get("attempts", 0) + 1,
            "cost_usd": state.get("cost_usd", 0.0) + (response.cost or 0.0),
        }

    def gate(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        ctx = runtime.context
        proposal = AssistantMessage.model_validate(state["scratch"][-1])
        if not proposal.tool_calls:
            return {
                "outcome": "message",
                "verdicts": [],
                "emitted": _emit(proposal, state),
                "harness_state": ctx.session.state.value,
            }
        calls = [ctx.to_harness_call(c) for c in proposal.tool_calls]
        verdicts = ctx.session.review(calls)
        dumped = [v.model_dump(mode="json") for v in verdicts]
        if all(v.allowed for v in verdicts):
            ctx.session.note_handoff(calls)
            return {
                "outcome": "tool_calls",
                "verdicts": dumped,
                "emitted": _emit(proposal, state),
                "harness_state": ctx.session.state.value,
            }
        refusals = [
            _dump(ToolMessage(id=call_id, role="tool", content=text,
                              requestor="assistant", error=True))
            for call_id, text in HarnessSession.refusals(verdicts)
        ]
        return {
            "outcome": "refused",
            "verdicts": dumped,
            "scratch": state["scratch"] + refusals,
            "harness_state": ctx.session.state.value,
        }

    def after_gate(
        state: TurnState, runtime: Runtime[TurnContext]
    ) -> Literal["reason", "escalate", "__end__"]:
        if state["outcome"] != "refused":
            return END
        if state["attempts"] < runtime.context.max_attempts:
            return "reason"
        return "escalate"

    def escalate(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        ctx = runtime.context
        tool, decision = ctx.session.escalate(state["attempts"])
        if tool is None:
            message = AssistantMessage(
                role="assistant",
                content="I am unable to complete this request. Please hold for a human agent.",
            )
        else:
            # Deterministic id: the turn's attempts make it unique within a thread.
            call = ToolCall(id=f"escalation-{len(state['messages'])}", name=tool,
                            arguments={"summary": decision.reason})
            message = AssistantMessage(role="assistant", content=None, tool_calls=[call])
        return {
            "outcome": "escalated",
            "emitted": _emit(message, state),
            "harness_state": ctx.session.state.value,
        }

    graph = StateGraph(TurnState, context_schema=TurnContext)
    graph.add_node("retrieve", retrieve)
    graph.add_node("reason", reason)
    graph.add_node("gate", gate)
    graph.add_node("escalate", escalate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "gate")
    graph.add_conditional_edges("gate", after_gate)
    graph.add_edge("escalate", END)
    return graph.compile(checkpointer=checkpointer)


def _system(ctx: TurnContext, state: TurnState) -> str:
    context = state.get("context") or []
    if not context:
        return ctx.system_prompt
    docs = "\n\n".join(
        f"<document id=\"{d['doc_id']}\" title=\"{d['title']}\">\n{d['content']}\n</document>"
        for d in context
    )
    return (
        f"{ctx.system_prompt}\n<retrieved_knowledge>\n"
        "Knowledge base documents retrieved for the customer's latest message:\n\n"
        f"{docs}\n</retrieved_knowledge>"
    )


def _emit(message: AssistantMessage, state: TurnState) -> dict:
    """The message tau2 receives, carrying the cost of the whole turn.

    tau2 sums the cost of emitted messages only, so the refused proposals and
    any retrieval calls of a turn would vanish from its accounting. Putting the
    turn's total on the one message it sees keeps tau2's ``agent_cost`` -- and
    so the ledger -- complete.
    """
    return _dump(message.model_copy(update={"cost": state.get("cost_usd", 0.0)}))
