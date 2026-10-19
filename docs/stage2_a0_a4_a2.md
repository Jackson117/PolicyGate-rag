# Stage 2, first three arms: floor, ceiling, naive RAG (2026-10-19)

All three arms under the same harness (gating + audit + loop guard, policygate 0.3.0),
gpt-4.1-mini for agent and user simulator, temperature 0, `max_steps` 100, seed 300, the
30-task subset x 3 trials. 270 episodes, no infrastructure errors.

Runs (not committed; ledger committed): `runs/stage2_guarded` (A0; A4 tasks 1-8),
`runs/stage2_guarded_a4b` (A4 tasks 9-30), `runs/stage2_guarded_a2` (A2). The first batch
deadlocked on A4's ninth task (`task_051`): every thread parked on a Python lock, none on
the network, for 35+ minutes. It was stopped and A4 resumed from that task with the same
seed; the hang did not recur and its cause is unknown. The runner now dumps all thread
stacks on SIGUSR1, and the watcher sends it after 10 minutes without trace activity.

Intervals: 95% percentile bootstrap over tasks (10,000 resamples); differences paired by task.

## Task success

| | A0 floor (`no_knowledge`) | A2 naive RAG | A4 ceiling (`golden_retrieval`) |
|---|---|---|---|
| pass^1 | 0.022 [0.000, 0.067] | 0.044 [0.000, 0.100] | 0.222 [0.089, 0.378] |
| pass^2 | 0.011 [0.000, 0.033] | 0.011 [0.000, 0.033] | 0.211 [0.078, 0.367] |
| pass^3 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.200 [0.067, 0.367] |

| Paired difference | pass^1 | pass^3 |
|---|---|---|
| A4 - A0 (what perfect documents are worth) | 0.200 [0.056, 0.367] | 0.200 [0.067, 0.367] |
| A2 - A0 (what naive RAG recovers) | 0.022 [-0.022, 0.078] | 0.000 [0.000, 0.000] |
| A4 - A2 (failure attributable to naive retrieval) | 0.178 [0.056, 0.322] | 0.200 [0.067, 0.367] |

**W1 stop checks, re-run under the guard:** retrieval effect present (A4 - A0 excludes
zero); no floor effect (A4 pass^1 0.222, above ~0.15). The study continues.

What the numbers say, at n = 30 tasks:

- **Perfect documents are worth about 0.20 pass^1.** With them, 0.78 of attempts still fail
  (1 - golden): most failure in this domain is not retrieval.
- **Naive RAG recovers almost none of that 0.20.** A2 is indistinguishable from the floor
  (0.022 [-0.022, 0.078]) and clearly below the ceiling.
- **The ceiling is consistent:** A4's pass^3 (0.200) is close to its pass^1 (0.222). Tasks
  it solves, it mostly solves on every trial.

## Why naive RAG falls short: retrieval quality (A2)

| Episode metric (mean over tasks) | Value |
|---|---|
| Coverage of required documents | 0.318 [0.262, 0.375] |
| Precision of documents shown | 0.122 [0.098, 0.146] |
| Query MRR | 0.361 [0.299, 0.421] |

A median A2 episode issues 7 queries and shows the model 24 distinct documents; about a
third of the required documents are ever among them, and seven in eight documents shown
are not required.

## Cost and behaviour

| | A0 | A2 | A4 |
|---|---|---|---|
| USD/episode (max) | 0.0139 (0.0566) | 0.0333 (0.1559) | 0.0380 (0.2374) |
| USD total | 1.25 | 3.00 | 3.42 |
| Hit the 100-step cap | 1 | 2 | 4 |

The loop guard cut A4's capped episodes from 34/90 (W1, unguarded) to 4/90 and its cost per
episode from 0.053 to 0.038; it refused 100 calls across A4.

Stage-2 line: 13.60 of 25.00 USD spent, including the superseded unguarded W1 floor/ceiling
runs (5.93). Not yet reconciled against the provider dashboard; the three hung `task_051`
episodes may have spent before stalling.
