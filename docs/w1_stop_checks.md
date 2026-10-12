# W1 stop checks: floor and ceiling (2026-10-12)

Run: `runs/w1_floor_ceiling` (not committed; ledger committed). A0 (`no_knowledge`) and
A4 (`golden_retrieval`), the 30-task subset x 3 trials, gpt-4.1-mini for agent and user
simulator, temperature 0, harness = gating + audit, `max_steps` 100, seed 300.
180 episodes, no infrastructure errors. Intervals: 95% percentile bootstrap over tasks
(10,000 resamples); A4 - A0 is paired by task.

| | A0 floor | A4 ceiling | A4 - A0 (paired) |
|---|---|---|---|
| pass^1 | 0.011 [0.000, 0.033] | 0.178 [0.089, 0.278] | 0.167 [0.078, 0.278] |
| pass^2 | 0.000 [0.000, 0.000] | 0.078 [0.011, 0.156] | 0.078 [0.011, 0.167] |
| pass^3 | 0.000 [0.000, 0.000] | 0.033 [0.000, 0.100] | 0.033 [0.000, 0.100] |

## The three stop conditions

1. **Budget: not triggered.** mini kept. Stage 2 has spent 5.93 of 25.00 USD
   (6.28 in total across lines), not yet reconciled against the dashboard.
2. **No retrieval effect: not triggered.** The paired pass^1 interval of A4 - A0,
   [0.078, 0.278], excludes zero. Perfect documents help.
3. **Floor effect: not triggered, narrowly.** A4 pass^1 is 0.178, above the ~0.15
   threshold, but its interval reaches down to 0.089.

## What this means for the rest of the study

- **Headroom is small.** Every retrieval arm should land between 0.011 and 0.178 pass^1.
  With 30 tasks, a paired interval on pass^1 spans roughly +-0.1 at these rates, so
  naive vs agentic will only separate if the difference is large.
- **1 - golden = 0.82.** With the right documents in the prompt, most failures remain:
  reasoning, policy and simulator behaviour, not retrieval. That is already a finding
  for the decomposition.

## Cost, and why

| | A0 | A4 |
|---|---|---|
| USD/episode | 0.0129 | 0.0530 (few 0.031, middle 0.058, many 0.070) |
| Episodes hitting the 100-step cap | 2 / 90 | 34 / 90 |
| Successes among capped episodes | 0 | 0 |
| Capped episodes' share of arm cost | 11% | 60% |

The capped episodes are tool-call loops: the model repeats the same read call
(e.g. `get_user_information_by_id` with identical arguments) until the cap. **The harness
does not cause them**: capped episodes contain zero harness refusals (completed episodes
average 0.17-0.27). Lowering the cap further is not clean: one successful A4 episode used
89 messages.

Calibration's A4 mean (0.0126, n=3) underestimated stage 2 by 4x: its three tasks
happened not to loop. The guard uses measured means, so it adapted as the run went.
