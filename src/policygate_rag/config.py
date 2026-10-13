"""RagConfig: every switch the study varies or holds fixed, defined before any module.

As with policygate's ``HarnessConfig``, a setting that is not a field here is a
setting nobody can report, so every number that changes behaviour lives here
rather than as a constant in code. Arms are built in ``experiments/arms.py``
from this class plus the tau2 retrieval variant the environment runs; nothing
else differs between arms.

Frozen, and JSON-native once dumped (``model_dump(mode="json")``), so a config
can be hashed into a run's identity and read back by any tool.
"""

from __future__ import annotations

from enum import StrEnum

from policygate.harness.config import HarnessConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalMode(StrEnum):
    """Which retrieval subgraph runs inside the per-turn graph."""

    NONE = "none"
    """No retrieval in the graph. Floor, ceiling and the tau2 bm25 baseline: their
    knowledge, if any, comes from the environment (a tool) or the prompt."""

    NAIVE = "naive"
    """One hybrid retrieval per user turn, top-k injected as context."""

    AGENTIC = "agentic"
    """retrieve_wide -> judge -> filter -> sufficiency -> (reason | rewrite -> retrieve_wide)."""


class JudgeKind(StrEnum):
    """Which relevance judge scores retrieved candidates. One interface, four kinds."""

    NONE = "none"
    """Raw RRF order, fixed top-k: the baseline for whether a judge is needed at all."""

    CROSS_ENCODER = "cross_encoder"
    """A small open-source reranker run locally. Free, reproducible."""

    LLM = "llm"
    """An LLM grader, candidates batched per call. Counts against the budget."""

    JEV = "jev"
    """TypeSafe Jev, pinned version. Separate account and billing."""


class StopReason(StrEnum):
    """Why an agentic retrieval loop ended. Stored as ``.value`` in graph state."""

    SUFFICIENT = "sufficient"
    NO_NEW_DOCS = "no_new_docs"
    BUDGET_EXHAUSTED = "budget_exhausted"


JEV_PINNED = "jev-1.13.0"
"""Hard constraint 8: never ``jev-latest``."""


class RagConfig(BaseModel):
    """One arm's agent-side configuration."""

    model_config = ConfigDict(frozen=True)

    name: str = "unnamed"

    # --- models: one configuration per comparison (hard constraint 6) --------
    agent_llm: str = "openai/gpt-4.1-mini"
    """The model that reasons and proposes tool calls. Decided at the end of W1
    calibration by the rule in CLAUDE.md; mini is the starting point."""

    user_llm: str = "openai/gpt-4.1-mini"
    """tau2's user simulator. Always the same model as ``agent_llm`` in this study."""

    aux_llm: str | None = None
    """Model for query rewriting, sufficiency checks and the LLM judge. ``None``
    means ``agent_llm``, so a comparison never mixes models by accident."""

    agent_temperature: float = Field(default=0.0, ge=0.0)
    user_temperature: float = Field(default=0.0, ge=0.0)

    # --- the harness (policygate), on in every arm ----------------------------
    harness: HarnessConfig = HarnessConfig(name="unset")
    """Which policygate mechanisms are active. Identical across the arms of a
    comparison; ``experiments/arms.py`` holds the one this study uses."""

    max_steps: int = Field(default=100, ge=1)
    """tau2's step cap per episode. 100, decided 2026-10-12 after calibration: a
    looping episode's cost grows roughly quadratically with its length, and 100 is
    policygate's setting. tau2's own default is 200."""

    max_internal_attempts: int = Field(default=3, ge=1)
    """Proposals the harness may refuse within one turn before escalating."""

    # --- retrieval -----------------------------------------------------------
    retrieval: RetrievalMode = RetrievalMode.NONE

    embedding_model: str = "text-embedding-3-large"
    """Dense embedder. tau2's default OpenAI embedder, so a difference between our
    pipeline and tau2's dense variant is not an embedder difference."""

    bm25_k1: float = Field(default=1.5, gt=0)
    """BM25 term-frequency saturation. 1.5 as in rank_bm25, which tau2's bm25 uses."""

    bm25_b: float = Field(default=0.75, ge=0, le=1)
    """BM25 length normalisation. 0.75 as in rank_bm25."""

    k_bm25: int = Field(default=20, ge=1)
    """Candidates taken from BM25 before fusion."""

    k_dense: int = Field(default=20, ge=1)
    """Candidates taken from the dense index before fusion."""

    rrf_k: int = Field(default=60, ge=1)
    """Reciprocal rank fusion constant: score = sum 1 / (rrf_k + rank)."""

    k_final: int = Field(default=5, ge=1)
    """Documents injected into the reasoning context."""

    max_history_tokens: int | None = None
    """Trim message history to this many tokens before reasoning, starting on a
    user message so tool calls and their results are never split. ``None``
    keeps the whole history."""

    # --- judge (agentic retrieval only) --------------------------------------
    judge: JudgeKind = JudgeKind.NONE

    judge_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Relevance probability at or above which a candidate is kept."""

    judge_batch_size: int = Field(default=10, ge=1)
    """Candidates per LLM-judge call."""

    cross_encoder_model: str | None = None
    """Hugging Face id, pinned with a revision in W3. Required for CROSS_ENCODER."""

    jev_model: str = JEV_PINNED

    # --- agentic loop --------------------------------------------------------
    agentic_max_rounds: int = Field(default=3, ge=1)
    """retrieve_wide calls per turn before stopping with BUDGET_EXHAUSTED."""

    sufficiency_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Sufficiency probability at or above which the loop stops with SUFFICIENT."""

    @model_validator(mode="after")
    def _consistent(self) -> RagConfig:
        if self.judge is not JudgeKind.NONE and self.retrieval is not RetrievalMode.AGENTIC:
            raise ValueError(
                f"judge={self.judge.value} only applies to agentic retrieval, "
                f"got retrieval={self.retrieval.value}"
            )
        if self.judge is JudgeKind.CROSS_ENCODER and not self.cross_encoder_model:
            raise ValueError("judge=cross_encoder needs cross_encoder_model")
        if self.jev_model != JEV_PINNED:
            raise ValueError(f"Jev must be pinned to {JEV_PINNED}, got {self.jev_model!r}")
        if self.agent_llm != self.user_llm:
            raise ValueError(
                "agent and user simulator use the same model in this study "
                f"(got agent={self.agent_llm!r}, user={self.user_llm!r})"
            )
        if self.k_final > self.k_bm25 + self.k_dense:
            raise ValueError("k_final cannot exceed the candidates fusion receives")
        return self

    @property
    def effective_aux_llm(self) -> str:
        return self.aux_llm or self.agent_llm
