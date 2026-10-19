"""Thin HTTP client for TypeSafe Jev. The only module that talks to Jev.

API (docs.typesafe.ai, read 2026-10-19): ``POST https://api.typesafe.ai/v1/systemone``
with ``Authorization: Bearer <key>``; body ``{"model", "state", "questions"}``;
a Noul answer is ``{"type": "noul", "noul": <0..1, probability of yes>}``;
``usage`` reports ``input_tokens`` and ``output_tokens``.

Pricing (docs.typesafe.ai/models, read 2026-10-19): input tokens only, USD 0.042 per
million; output tokens are free. Responses carry no cost field, so cost is
computed here from the reported input tokens. An unreported token count is an
error, not a zero.

The model is pinned (hard constraint 8): ``jev-latest`` moves when a release
ships, and a judge whose threshold was set on one version is not the same
judge on the next.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from policygate_rag.config import JEV_PINNED

URL = "https://api.typesafe.ai/v1/systemone"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
KEY_ENV = "JEV_API_KEY"


class JevError(Exception):
    pass


class JevClient:
    def __init__(self, *, model: str = JEV_PINNED, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        if model != JEV_PINNED:
            raise ValueError(f"Jev must be pinned to {JEV_PINNED}, got {model!r}")
        key = os.environ.get(KEY_ENV)
        if not key:
            raise JevError(f"{KEY_ENV} is not set")
        self.model = model
        self._http = httpx.Client(
            timeout=timeout, transport=transport,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    def nouls(self, state: Any, questions: dict[str, dict]) -> tuple[dict[str, float], int, float]:
        """Ask Noul questions about one state. Returns (probabilities, input tokens, USD)."""
        response = self._http.post(URL, json={
            "model": self.model, "state": state,
            "questions": {k: {"type": "noul", **q} for k, q in questions.items()},
        })
        if response.status_code != 200:
            raise JevError(f"HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        if body.get("model") != self.model:
            raise JevError(f"asked for {self.model}, answered by {body.get('model')!r}")
        tokens = (body.get("usage") or {}).get("input_tokens")
        if tokens is None:
            raise JevError("no input_tokens reported; cost unknown")
        answers = body.get("answers") or {}
        try:
            probabilities = {k: float(answers[k]["noul"]) for k in questions}
        except (KeyError, TypeError, ValueError) as e:
            raise JevError(f"malformed answers: {str(answers)[:300]}") from e
        return probabilities, tokens, tokens * USD_PER_INPUT_TOKEN

    def close(self) -> None:
        self._http.close()
