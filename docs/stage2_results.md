# Stage 2 results: all five arms (2026-10-20)

One model (gpt-4.1-mini, agent and user simulator, temperature 0), one harness (gating +
audit + loop guard, policygate 0.3.0), `max_steps` 100, seed 300, the committed 30-task
subset x 3 trials: 450 episodes, no infrastructure errors. Arms differ only in retrieval and
the tau2 retrieval variant. Machine-readable results: `results/stage2_arms.json`
(regenerate with `python -m policygate_rag.eval.export_results`). Per-arm details:
`docs/stage2_a0_a4_a2.md`, `docs/stage2_a3.md`; judges: `docs/stage1_judges.md`.

Intervals: 95% percentile bootstrap over tasks (10,000 resamples); differences paired by task.

## Task success

| Arm | Retrieval | pass^1 | pass^3 | USD/episode |
|---|---|---|---|---|
| A0 floor | none (`no_knowledge`) | 0.022 [0.000, 0.067] | 0.000 | 0.014 |
| A1 external baseline | tau2's BM25 tool (`bm25`) | 0.022 [0.000, 0.056] | 0.000 | 0.055 |
| A2 naive RAG (ours) | hybrid BM25 + dense, top 5 per user turn | 0.044 [0.000, 0.100] | 0.000 | 0.033 |
| A3 agentic RAG (ours) | loop with Jev judge, sufficiency, rewrite | 0.022 [0.000, 0.056] | 0.000 | 0.045 (+ Jev 0.008) |
| A4 ceiling | required documents in the prompt (`golden_retrieval`) | 0.222 [0.089, 0.378] | 0.200 | 0.038 |

| Paired difference (pass^1) | Estimate |
|---|---|
| A4 - A0: what perfect documents are worth | 0.200 [0.056, 0.367] |
| A1 - A0, A2 - A0, A3 - A0: what each retrieval arm recovers | 0.000, 0.022, 0.000 (all intervals include 0) |
| A4 - A2 (best retrieval arm by point estimate) | 0.178 [0.056, 0.322] |
| A4 - A3 | 0.200 [0.078, 0.344] |
| A4 - A1 | 0.200 [0.067, 0.356] |

## Retrieval quality

| | A1 tau2 BM25 tool | A2 naive | A3 agentic + Jev |
|---|---|---|---|
| Coverage of required documents | 0.329 [0.262, 0.398] | 0.318 [0.262, 0.375] | 0.427 [0.361, 0.495] |
| Precision of documents shown | 0.181 [0.142, 0.227] | 0.122 [0.098, 0.146] | 0.207 [0.168, 0.247] |
| Query MRR | 0.582 [0.475, 0.688] | 0.361 [0.299, 0.421] | 0.442 [0.389, 0.497] |
| Queries per episode (median) | 2 | 7 | 11 |

Paired: A3 - A2 coverage +0.109 [+0.057, +0.169]; A3 - A1 coverage +0.098 [+0.025, +0.174].
A1's documents come from the `KB_search` results in tau2's messages (ids in rank order), the
others' from the retrieval trace. MRR is not comparable across arms: A1's few queries are
written by the model, A2's are the user's words, A3's include rewrites.

## The decomposition

```
no_knowledge   tau2 bm25   naive RAG   agentic RAG   golden
   0.022    ~    0.022   ~   0.044   ~    0.022    <   0.222     (pass^1)
```

- **Retrieval is a real bottleneck**: perfect documents add 0.200 [0.056, 0.367].
- **None of the retrieval arms recovers a measurable part of it.** Agentic RAG covers
  about a third more of the required documents than naive RAG or tau2's BM25 tool, at the
  best precision, and is no more successful.
- **Failure attributable to retrieval** (golden - best retrieval arm): 0.178 [0.056, 0.322].
- **Failure that remains with perfect documents** (1 - golden): 0.778 of attempts.
  Retrieval explains less of this domain's failure than everything else does.
- **Coverage looks like a threshold.** 0.32-0.43 coverage buys nothing measurable; the
  complete set buys about a fifth of the tasks. Whether intermediate coverage (say 0.7)
  helps is not answered by these arms.

## A5: the working set (added 2026-10-21, partial)

A diagnosis of A3 (`docs/replay_a3.md`) found that the model rarely had the required
documents when it acted: 0.144 of them at each proposed tool call, against 0.427 over the
episode, because 5 documents were replaced every user turn. A5 = A3 + a 20-document
working set kept across turns + lookup of products named in the conversation.

| | A3 | A5 | A5 − A3 (paired) |
|---|---|---|---|
| pass^1 | 0.022 [0.000, 0.056] | 0.056 [0.000, 0.133] | +0.033 [−0.011, +0.100] |
| Coverage at each proposed tool call | 0.144 [0.105, 0.183] | 0.395 [0.333, 0.458] | +0.260 [+0.210, +0.309] |
| Coverage over the episode | 0.427 [0.361, 0.495] | 0.535 [0.456, 0.614] | +0.108 [+0.042, +0.177] |
| USD / episode | 0.045 | 0.068 | |

The budget guard stopped A5 at 72 of 90 episodes (stage 2: projected 29.08 > 29.02):
every task has trial 1, 21 have all three, and the run was left partial by decision. pass^1
is the per-task mean of the trials run (unbiased); pass^3 is not reported. Trials 2–3 ran
as a second batch, seed 301, so their seeds differ from the other arms'.

The working set does what the replay predicted (0.393 predicted, 0.395 measured), and
task success does not detectably follow. With A4 at 0.222, the retrieval fix has at most
about 0.17 left to recover, below what 30 tasks can resolve.

## Limits

- n = 30 tasks; differences between the retrieval arms are within about +-0.05 and cannot
  be resolved at this size. The full 87 tasks were not run (budget).
- One model (gpt-4.1-mini); the floor and the retrieval arms are near zero, so a stronger
  model might separate them.
- Prompt templates differ by arm (tau2's per-variant templates for A0, A1, A4; A2 and A3 use
  `no_knowledge.md` plus a retrieved-documents block).
- The harness is gating + audit + loop guard, not policygate's `full()`.
- The harness cannot verify the 2-of-4 identity check; it only knows a verification was logged.
- Three batches stalled on leaked HTTP connections (tau2's 10-connection pool); stalled
  episodes were re-run from the start with the same seed, and their partial spend is booked.

## Spend

OpenAI (via AIHubMix), with A5: USD 28.88 of 33.80; stage 2 27.89 of 29.02 after
transfers from unused lines (recorded in the ledger). Jev (own account): USD 1.74.
Reserve untouched. Not yet reconciled against the provider dashboard since 6.29.
