# Why A3 covers 0.43 of the required documents, and an offline replay of fixes (2026-10-20)

A3 (agentic RAG with Jev) shows the model 0.427 [0.361, 0.495] of each task's required
documents over an episode, and passes 0.022. This note finds where the rest are lost, then
replays A3's 90 recorded episodes under other context policies to see which fixes are worth
a new arm. No model calls were made; replay is BM25 and string matching
(`python -m policygate_rag.eval.replay runs/stage2_a3/A3`).

Each episode's scored attempt only: 16 A3 episodes also carry an earlier attempt, stopped by
a stall, in the trace. The shared trace reader now drops those (`last_attempt_rows`), which
moved A3's reported coverage from 0.432 to 0.427; no other arm was affected.

## Where required documents are lost

1,014 (episode, required document) slots:

| Stage | Share |
|---|---|
| Never in the top 20 of any round | 0.349 |
| In the top 20, rejected by Jev (score < 0.5) | 0.184 |
| Admitted by Jev, cut by the top-5 limit | 0.095 |
| Shown to the model | 0.372 |

By document type (assigned by title and content heuristics, so approximate):

| Type | Share of slots | Never in top 20 | Shown |
|---|---|---|---|
| Product-specific ("Light Blue Account at a glance") | 0.52 | 0.49 | 0.22 |
| Internal procedure (how to use a discoverable tool) | 0.42 | 0.21 | 0.51 |
| General policy | 0.05 | 0.09 | 0.74 |

Four causes, from the traces:

1. **The model never has most of the documents at once.** At most 5 documents are in the
   prompt, replaced on every user turn, against a median of 10 required. Episode-union
   coverage hides this; see the replay's `replace` row below.
2. **The sufficiency check rarely asks for more.** It stopped as `sufficient` on 673 of 738
   retrievals (91%), 557 after one round. It cannot miss a document it has not seen.
3. **Queries are the customer's words.** Product documents are needed per product (a
   referral comparing ten checking accounts, a stolen wallet holding three account types);
   a top-k search returns the closest few, not the set. 55% of the never-retrieved product
   documents belong to a product named in the conversation. Procedure documents are needed
   for the agent's next action, which is never used as a query.
4. **Jev scores against the query, not the need.** On "My name is Kenji Tanaka" a document
   the task needs is irrelevant to the sentence.

Ranking within the top 20 is not the bottleneck: retrieved required documents sit at rank
1 more often than at any other rank.

## Replay

Metric: **coverage at action time**, the share of required documents in the model's
context at each step that proposes a tool call; mean over those steps, then over episodes
within a task, then over tasks. Intervals are bootstrap over 30 tasks; differences paired.

Policies (details in `eval/replay.py`): `replace` is what A3 did. `wsN` is a working set of
N documents kept across turns, oldest evicted first, fed by A3's top 5 per turn (`kept`) or
everything Jev admitted (`pool`). `+ent` adds a product's summary document and its best
BM25 match for the latest message when the product is first named (by anyone, tool output
included) or named again by the customer. `+act` adds up to 3 documents naming a tool the
agent proposes to call, as a harness that intercepts the call would.

| Policy | Mean coverage at action | Max coverage at action | Docs | Tokens |
|---|---|---|---|---|
| replace (A3) | 0.144 [0.105, 0.184] | 0.235 [0.177, 0.297] | 4.4 | 2,114 |
| ws10 kept | 0.266 [0.208, 0.326] | 0.349 [0.280, 0.422] | 9.7 | 4,656 |
| ws20 kept | 0.333 [0.274, 0.396] | 0.406 [0.337, 0.476] | 14.1 | 6,468 |
| ws20 pool | 0.339 [0.283, 0.398] | 0.425 [0.361, 0.491] | 18.9 | 8,851 |
| ws20 kept +ent | 0.393 [0.332, 0.455] | 0.502 [0.427, 0.579] | 16.1 | 7,006 |
| ws20 kept +act | 0.345 [0.284, 0.408] | 0.425 [0.354, 0.496] | 15.1 | 7,239 |
| ws20 kept +ent +act | 0.394 [0.333, 0.456] | 0.512 [0.437, 0.587] | 16.7 | 7,578 |
| ws30 kept +ent +act | 0.431 [0.373, 0.490] | 0.552 [0.482, 0.621] | 20.2 | 8,909 |
| ws30 pool +ent +act | 0.442 [0.385, 0.502] | 0.556 [0.489, 0.625] | 26.4 | 11,864 |
| unbounded pool +ent +act | 0.509 [0.457, 0.562] | 0.629 [0.571, 0.688] | 36.7 | 15,696 |

Tokens are characters / 4. For scale, A4 inlines its required documents, up to ~13k tokens.

One change at a time (paired, mean coverage at action):

| Change | Difference |
|---|---|
| Keep 10 across turns instead of replacing 5 | +0.122 [+0.093, +0.153] |
| 20 instead of 10 | +0.068 [+0.049, +0.087] |
| Feed the set with everything admitted, not the top 5 | +0.006 [-0.020, +0.037] |
| Product lookup (`+ent`) | +0.060 [+0.026, +0.099] |
| Action trigger (`+act`) | +0.012 [+0.004, +0.020] |
| 30 instead of 20, with `+ent +act` | +0.037 [+0.026, +0.049] |

## Reading

- **Context management is the largest fix and needs no new retrieval.** Keeping 20
  documents across turns more than doubles what the model has when it acts (0.144 to
  0.333) for about 4k more tokens.
- **Product lookup adds a further 0.06.** Its documents are noisy (0.25 of those added are
  required), which the working set's capacity absorbs.
- **The action trigger barely helps, and the replay says why.** 32% of required documents
  name a tool in the task's reference actions; only 13% name a tool the agent actually
  called. The agent calls tools it already read about, so documents fetched on the call are
  mostly ones it had. A trigger can only help once something else gets the agent to the
  right tool.
- **Feeding the set with everything Jev admitted adds nothing over the top 5.**
- **The ceiling of these policies is about half.** Even unbounded, with every admitted
  document, product lookup and the action trigger, coverage at action time is 0.509: about
  a third of required documents are never retrieved by anything here. Reaching them needs
  better queries (a list of needs per turn, one query each, and enumerating a product
  family for "which account is best" questions), which needs model calls.

## Limits

- Counterfactual on fixed conversations. A different context would have changed what the
  agent did and so what was retrieved next; the replay bounds what these policies supply,
  it does not predict pass^1.
- Coverage of the whole required set at every step is a proxy: not every document is needed
  at every step, and task_024 passed at 0.33 while task_038 and task_085 failed at 1.0.
- Even complete documents (A4) pass 0.222, so the most any of this can recover is about 0.2.
- Product names are matched from document ids, longest first; ambiguous names ("green
  account" is a checking and a savings account) add both products.

## The next arm (run 2026-10-21)

A5 = A3 + a 20-document working set + product lookup (`retrieval/working_set.py`,
`retrieval/products.py`). Live, its coverage at action time was 0.395 [0.333, 0.458],
against the replay's 0.393 for the same policy; pass^1 0.056 [0.000, 0.133], A5 − A3
+0.033 [−0.011, +0.100]. The budget guard stopped it at 72 of 90 episodes; see
`docs/stage2_results.md`.
