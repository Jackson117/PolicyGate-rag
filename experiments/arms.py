"""Arm definitions: a RagConfig plus the tau2 retrieval variant the environment runs.

Nothing else differs between arms. Every arm in a comparison shares the model
(hard constraint 6) and the harness configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from policygate.harness.config import HarnessConfig

from policygate_rag.config import RagConfig, RetrievalMode

HARNESS = HarnessConfig(name="banking_grounded", gating_enabled=True, audit_enabled=True)
"""PENDING DECISION -- see the open question in CLAUDE.md, "Harness configuration".

Proposed: only the mechanisms banking's policy text grounds. Gating enforces
"verify before accessing customer information" (via banking.py's roles and
facts); audit observes without changing behaviour. Left off, because banking
states no such rule or the harness has nothing banking-specific to check:
parallel-call blocking (a retail rule), validation (no banking constraints are
written), context pruning, and the approval gate.
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
