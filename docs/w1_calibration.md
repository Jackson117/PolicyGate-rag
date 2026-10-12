# W1 cost calibration (2026-10-12)

Run: `runs/calibration_w1` (not committed; ledger in `runs/ledger.json`).
gpt-4.1-mini for agent and user simulator, temperature 0, harness = gating + audit,
tau2 `max_steps` 200 (default). Tasks: the first task of each subset stratum
(`task_017`, `task_025`, `task_022`), 1 trial, seed 300. 9 episodes.

Costs are litellm's price table applied to reported tokens; **not yet reconciled
against the provider dashboard**.

| Arm | Env variant | USD/episode (mean of 3) | Max | Agent / user | Messages/episode | Terminations |
|---|---|---|---|---|---|---|
| A0 floor | no_knowledge | 0.0077 | 0.0115 | 0.0059 / 0.0018 | 18.7 | 3 user_stop |
| A1 bm25 | bm25 | 0.0981 | 0.2686 | 0.0795 / 0.0186 | 84.3 | 2 user_stop, 1 max_steps |
| A4 ceiling | golden_retrieval | 0.0126 | 0.0211 | 0.0101 / 0.0025 | 21.0 | 3 user_stop |

Mean over arms: **0.0395 USD/episode** against the stage-2 allowance of 0.0556.
Decision rule: **mini fits; mini is kept.** Total calibration spend 0.3552 USD of 1.00.

Agent cost from our trace equals tau2's `agent_cost` in every arm: no retried
attempts were discarded.

## What the mean hides

The A1 mean is one episode. `task_022` under bm25 ran into the 200-step cap: agent
and simulated user looped ("I will now proceed with the detailed review" / "please
continue"), 101 agent turns, ~2.0M prompt tokens, 0.2686 USD. The other two A1
episodes cost ~0.013. Because context grows each turn, a runaway episode costs
roughly quadratically in its length, so stage-2 spend is governed by the tail and
by `max_steps`, not by the typical episode.

A2 and A3 are not measured: both add retrieved context to every call, and A3 adds
judge and rewrite calls.

Rewards are recorded as counts only (A0 0/3, A1 0/3, A4 2/3). Three tasks per arm
support no claim about pass rates.
