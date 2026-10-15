"""Per-turn graph state and runtime context.

State is JSON-native -- strings, numbers, lists, dicts -- so a checkpoint can be
read by any tool, not only by this code. tau2 messages cross the boundary as
``model_dump(mode="json")`` and are rebuilt with :data:`MESSAGES`.

What cannot be serialised lives in :class:`TurnContext`, passed as LangGraph's
runtime context and never checkpointed: the harness session (stateful: tracker
and audit chain), the tau2 tool objects, and the model call.

No field uses a reducer. Each turn's input sets every per-turn field, so a
thread's checkpoint history is a sequence of complete turns and nothing
accumulates across turns by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from policygate.harness.session import ExtractedFacts, HarnessSession
from policygate.harness.types import ToolCall
from pydantic import TypeAdapter
from tau2.data_model.message import AssistantMessage, Message
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.environment.tool import Tool

if TYPE_CHECKING:
    from policygate_rag.retrieval.naive import RetrievalResources

MESSAGES: TypeAdapter = TypeAdapter(list[Message])


class TurnState(TypedDict, total=False):
    messages: list[dict]
    """The agent's history before this turn, as tau2 message dumps. Input only."""

    context: list[dict]
    """Documents retrieval put in front of the model this turn:
    ``{"doc_id", "title", "content", "score"}``. Empty for retrieval NONE."""

    retrieval: dict
    """What retrieval did this turn, for the trace: queries, candidates, stop reason."""

    scratch: list[dict]
    """This turn's internal messages: each proposal, and the refusals fed back."""

    attempts: int
    """Proposals made this turn."""

    verdicts: list[dict]
    """The harness's verdicts on the latest proposal."""

    outcome: str
    """``message`` | ``tool_calls`` | ``escalated``."""

    emitted: dict
    """The AssistantMessage handed to tau2, its cost being the whole turn's."""

    cost_usd: float
    """Every model call made this turn: proposals, and later retrieval and judging."""

    harness_state: str
    """Session state after the gate, for the trace."""


@dataclass
class TurnContext:
    session: HarnessSession
    tools: Sequence[Tool]
    system_prompt: str
    llm: str
    llm_args: dict[str, Any]
    max_attempts: int
    to_harness_call: Callable[[Tau2ToolCall], ToolCall]
    generate: Callable[..., AssistantMessage]
    """tau2's ``generate`` in a run; a scripted stub in tests."""

    retrieval: RetrievalResources | None = None
    """Index, documents and embedder. Required unless retrieval is NONE."""

    extract: Callable[..., ExtractedFacts] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
