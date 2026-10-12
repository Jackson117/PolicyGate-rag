"""RagAgent: tau2's HalfDuplexAgent contract, with the decision graph behind it.

Every arm uses this class. Arms differ only in their ``RagConfig`` and in the
retrieval variant the environment runs (floor, ceiling and the bm25 baseline
get their knowledge from the environment and run retrieval NONE here).

Label isolation (hard constraint 3): the agent is never handed a tau2 ``Task``.
The factory in the runner keeps only the task id, which names the checkpoint
thread and is not a label.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from policygate.adapters.tau2_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from policygate.harness.approval import auto_approve
from policygate.harness.gating import ToolGate
from policygate.harness.session import HarnessSession
from policygate.harness.validation import DomainConstraints, Validator
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate as tau2_generate

from policygate_rag import banking
from policygate_rag.config import RagConfig
from policygate_rag.graph.build import build_graph
from policygate_rag.graph.state import MESSAGES, TurnContext
from policygate_rag.tracing.recorder import TraceRecorder


class RagAgentState:
    """The agent's own history plus the harness session. Not serialised.

    ``messages`` includes what tau2 never sees: refused proposals and the
    refusals fed back, so the model keeps its own record of what was refused.
    """

    def __init__(self, messages: list, session: HarnessSession) -> None:
        self.messages = messages
        self.session = session
        self.turn = 0


class RagAgent(HalfDuplexAgent[RagAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        config: RagConfig,
        *,
        toolkit: Any,
        task_id: str = "unknown",
        checkpointer: Any = None,
        recorder: TraceRecorder | None = None,
        generate: Callable[..., AssistantMessage] = tau2_generate,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        session_id: str | None = None,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.config = config
        self.task_id = task_id
        self.recorder = recorder
        self.generate = generate
        self.clock = clock
        self.session_id = session_id or str(uuid.uuid4())
        self.seed: int | None = None
        self.llm_args: dict[str, Any] = {"temperature": config.agent_temperature}
        self.gate = ToolGate(banking.catalogue(toolkit), config.harness)
        self.validator = Validator(DomainConstraints(), config.harness)
        self.graph = build_graph(config, checkpointer=checkpointer)
        self._checkpointed = checkpointer is not None

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    @property
    def thread_id(self) -> str:
        return f"{self.config.name}-{self.task_id}-seed{self.seed}"

    def set_seed(self, seed: int) -> None:
        # As tau2's own LLM agent does: the seed reaches the model call.
        self.seed = seed
        self.llm_args["seed"] = seed

    def get_init_state(self, message_history: list[Message] | None = None) -> RagAgentState:
        return RagAgentState(
            messages=list(message_history or []),
            session=HarnessSession(
                gate=self.gate,
                validator=self.validator,
                config=self.config.harness,
                approve=auto_approve,
                session_id=self.session_id,
                clock=self.clock,
                profile=banking.BANKING_PROFILE,
            ),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: RagAgentState
    ) -> tuple[AssistantMessage, RagAgentState]:
        self._absorb(message, state)
        context = TurnContext(
            session=state.session,
            tools=self.tools,
            system_prompt=self.system_prompt,
            llm=self.config.agent_llm,
            llm_args=dict(self.llm_args),
            max_attempts=self.config.max_internal_attempts,
            to_harness_call=banking.unwrap,
            generate=self.generate,
        )
        turn_input = {
            "messages": [m.model_dump(mode="json") for m in state.messages],
            "context": [], "retrieval": {}, "scratch": [], "attempts": 0,
            "verdicts": [], "outcome": "", "emitted": {}, "cost_usd": 0.0,
        }
        config = {"configurable": {"thread_id": self.thread_id}}
        final: dict[str, Any] = dict(turn_input)
        for chunk in self.graph.stream(
            turn_input, config, context=context,
            stream_mode=["updates", "custom"], subgraphs=True, version="v2",
        ):
            if self.recorder is not None:
                self.recorder.write(self.thread_id, state.turn, chunk)
            if chunk["type"] == "updates" and not chunk["ns"]:
                for update in chunk["data"].values():
                    if update:
                        final.update(update)

        emitted = AssistantMessage.model_validate(final["emitted"])
        if self.recorder is not None:
            # The runner sums these to catch spend on attempts tau2 discards.
            self.recorder.write(self.thread_id, state.turn, {
                "type": "turn_cost", "ns": (), "data": {"usd": emitted.cost or 0.0},
            })
        state.messages.extend(MESSAGES.validate_python(final["scratch"]))
        if final["outcome"] == "escalated":
            state.messages.append(emitted)
        state.turn += 1
        return emitted, state

    def _absorb(self, message: ValidAgentInputMessage, state: RagAgentState) -> None:
        """Record what the world reported, then let the harness learn from it."""
        if isinstance(message, MultiToolMessage):
            results = list(message.tool_messages)
            state.messages.extend(results)
        else:
            state.messages.append(message)
            results = [message] if isinstance(message, ToolMessage) else []
        for result in results:
            name = self._tool_name_for(result.id, state)
            facts = (
                None if name is None
                else banking.facts(name, result.content, error=result.error)
            )
            state.session.absorb_result(result.id, error=result.error, facts=facts)

    @staticmethod
    def _tool_name_for(call_id: str, state: RagAgentState) -> str | None:
        for message in reversed(state.messages):
            for call in getattr(message, "tool_calls", None) or []:
                if call.id == call_id:
                    return call.name
        return None
