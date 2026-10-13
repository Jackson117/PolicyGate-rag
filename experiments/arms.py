"""Arm definitions: a RagConfig plus the tau2 retrieval variant the environment runs.

Nothing else differs between arms. Every arm in a comparison shares the model
(hard constraint 6) and the harness configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from policygate.harness.config import HarnessConfig

from policygate_rag.config import RagConfig, RetrievalMode

HARNESS = HarnessConfig(
    name="banking_grounded",
    gating_enabled=True,
    audit_enabled=True,
    loop_guards_enabled=True,
)
"""Decided 2026-10-12, loop guard added 2026-10-13.

Gating enforces "verify before accessing customer information" (through
banking.py's roles and facts); audit observes without changing behaviour. The
loop guard (policygate 0.3.0, max_repeated_actions=3) refuses a fourth
consecutive identical call: W1 found 34 of 90 golden-retrieval episodes stuck
repeating one read call until the step cap, none successful, 60% of the arm's
cost. Off, because banking states no such rule or the harness has nothing
banking-specific to check: parallel-call blocking (a retail rule), validation
(no banking constraints), context pruning, and the approval gate. This differs
from policygate's ``HarnessConfig.full()``; the README must say so.
"""


@dataclass(frozen=True)
class Arm:
    id: str
    env_variant: str
    """tau2 ``retrieval_config``: what the environment gives the agent."""
    config: RagConfig


def arms(model: str = "openai/gpt-4.1-mini") -> dict[str, Arm]:
    def cfg(name: str, **kw) -> RagConfig:
        return RagConfig(name=name, agent_llm=model, user_llm=model, harness=HARNESS, **kw)

    return {
        "A0": Arm("A0", "no_knowledge", cfg("A0")),
        "A4": Arm("A4", "golden_retrieval", cfg("A4")),
        "A1": Arm("A1", "bm25", cfg("A1")),
        "A2": Arm("A2", "no_knowledge", cfg("A2", retrieval=RetrievalMode.NAIVE)),
        # A3 (agentic, best judge from stage 1) is defined after W3 picks the judge.
    }
