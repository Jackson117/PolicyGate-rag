# W2: naive RAG (A2) smoke test (2026-10-15)

Run: `runs/smoke_a2` (not committed). A2, the three calibration tasks (`task_017`,
`task_025`, `task_022`) x 1 trial, gpt-4.1-mini, harness = gating + audit + loop guard,
`max_steps` 100. Development line. **Three episodes: a check that A2 works with a real
model, not a measurement of anything.**

| Task | USD | Messages | Ended | Reward | Retrievals (fresh / reused) | Harness refusals |
|---|---|---|---|---|---|---|
| task_017 | 0.0223 | 37 | user_stop | 0 | 6 / 9 | 0 |
| task_025 | 0.0105 | 12 | user_stop | 1 | 4 / 0 | 0 |
| task_022 | 0.0365 | 25 | user_stop | 0 | 7 / 4 | 0 |

Mean 0.0231 USD/episode. The agent's trace cost equals tau2's `agent_cost`. No episode hit
the step cap and the loop guard never fired.

## The query policy, checked against the labels

Naive retrieval uses the latest user message verbatim and replaces the documents on each
user turn. Identity answers ("My full name is Kenji Tanaka.") therefore become queries
and replace the previous documents. To see whether that throws away good documents, each
user turn of the three episodes was re-retrieved offline under two policies and scored
against `required_documents` (36 query embeddings, 0.00046 USD):

| Task | Required docs | Latest message: required docs in top 5, per user turn | All user messages so far |
|---|---|---|---|
| task_017 | 4 | 0 0 0 1 0 1 | 0 0 0 0 0 1 |
| task_025 | 10 | 1 3 2 2 2 | 1 1 2 2 2 |
| task_022 | 13 | 0 0 0 1 1 0 0 | 0 0 0 0 1 0 0 |

Concatenating the conversation does not help, and the replaced documents were mostly not
relevant to begin with: the first-turn retrieval found 0 of 4, 1 of 10 and 0 of 13
required documents. **The latest-message policy is kept.** What these three tasks suggest,
and stage 1 will measure properly, is that one literal query rarely reaches the required
documents on this corpus. That gap is what the agentic arm is meant to close.
