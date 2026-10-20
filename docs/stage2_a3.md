# Stage 2: agentic RAG with the Jev judge (A3) (2026-10-20)

A3: agentic retrieval (retrieve wide -> Jev judge -> filter -> sufficiency -> rewrite, at
most 3 rounds per user turn), judge chosen in stage 1 (`docs/stage1_judges.md`). Same
harness, model, step cap, seed and 30 tasks x 3 trials as A0, A2, A4. Run:
`runs/stage2_a3` (not committed).

The run stalled twice with every model call queued for an HTTP connection (see
"Infrastructure"). The first stall was stopped by hand; after that the runner gained
`--resume` and a supervisor, which caught the second stall at chunk 24, dumped stacks and
resumed. Turns paid for inside stalled episodes (USD 0.31) are booked; the episodes
themselves were re-run from the start with the same seed.

Intervals: 95% bootstrap over tasks; differences paired by task.

## Task success

| | A0 floor | A2 naive | **A3 agentic + Jev** | A4 golden |
|---|---|---|---|---|
| pass^1 | 0.022 [0.000, 0.067] | 0.044 [0.000, 0.100] | **0.022 [0.000, 0.056]** | 0.222 [0.089, 0.378] |
| pass^3 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | **0.000 [0.000, 0.000]** | 0.200 [0.067, 0.367] |

| Paired difference | pass^1 |
|---|---|
| A3 - A2 (what agentic adds over naive) | -0.022 [-0.056, 0.000] |
| A3 - A0 (what agentic RAG recovers) | 0.000 [-0.056, 0.044] |
| A4 - A3 (failure attributable to retrieval, after agentic RAG) | 0.200 [0.078, 0.344] |

## Retrieval

| | A2 naive | A3 agentic + Jev | A3 - A2 (paired) |
|---|---|---|---|
| Coverage of required documents | 0.318 [0.262, 0.375] | 0.432 [0.366, 0.499] | **+0.114 [+0.061, +0.175]** |
| Precision of documents shown | 0.122 [0.098, 0.146] | 0.206 [0.167, 0.246] | **+0.083 [+0.057, +0.111]** |
| Query MRR | 0.361 [0.299, 0.421] | 0.438 [0.385, 0.492] | **+0.077 [+0.037, +0.115]** |

The agentic loop stopped as `sufficient` in 740 of 816 fresh retrievals (91%), 608 of them
after a single round; `budget_exhausted` 74, `no_new_docs` 2.

## What it means

1. **Agentic retrieval works as retrieval and does nothing for the task.** It covers about
   a third more of the required documents, at nearly twice the precision, and task success
   is unchanged: at the floor, like naive RAG.
2. **Partial coverage buys nothing here; complete coverage buys 0.20.** Golden retrieval
   (every required document, every call) succeeds on about a fifth of tasks; 43% coverage
   succeeds on about none. On this domain success looks like a threshold on having the
   whole document set, not a slope.
3. **The sufficiency check is the weak link.** It declares the documents sufficient after
   one round 91% of the time while coverage is 0.43: the model judging its own evidence
   stops early. Coverage is per task, so a single turn may legitimately need fewer
   documents, but the gap is large.
4. **The decomposition, so far:** of the 0.20 that perfect documents are worth, neither
   naive nor agentic RAG recovers a measurable part; 0.78 of attempts fail even with
   perfect documents.

## Cost

| | A2 | A3 |
|---|---|---|
| USD/episode (OpenAI) | 0.0333 | 0.0452 |
| Jev (own account), total | - | 0.748 |
| Loop-guard refusals | 11 | 9 |
| Hit the 100-step cap | 2 | 2 |

## Infrastructure

Both stalls (and A4's earlier one) had the same signature: model calls queued in httpcore's
`wait_for_connection` while no thread did network I/O, i.e. connections leaked from litellm's
pool of 100. Root cause not yet found. Runs now recover (resume + supervisor), and the
runner logs the pool every minute (`http_pool.jsonl`) to identify the leak next time.
