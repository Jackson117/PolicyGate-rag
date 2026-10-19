"""The LLM judge's model call: tau2's ``generate`` (litellm) in JSON mode.

The one judges/ module allowed to reach a model API (hard constraint 7). Same
client, provider and model family as the agent, so the LLM judge costs land in
the same ledger and reconcile against the same dashboard.
"""

from __future__ import annotations

from typing import Any


def make_ask(model: str, **llm_args: Any):
    from tau2.data_model.message import SystemMessage, UserMessage
    from tau2.utils.llm_utils import generate

    def ask(system: str, user: str) -> tuple[str, float]:
        response = generate(
            model=model,
            messages=[SystemMessage(role="system", content=system),
                      UserMessage(role="user", content=user)],
            call_name="llm_judge",
            response_format={"type": "json_object"},
            **llm_args,
        )
        if response.cost is None:
            raise ValueError(f"no cost reported for {model}")
        return response.content or "", response.cost

    return ask
