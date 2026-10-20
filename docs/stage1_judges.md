# Stage 1: offline judge benchmark (2026-10-20)

Queries: every fresh query A2 (naive RAG) issued in stage 2, 773 in all (569 unique), with
the top `k_wide` = 20 fused candidates each: 11,381 unique (query, document) pairs.
Each judge re-ranks the same candidates; ties fall back to fused order. Labels:
`required_documents`, per task (any required document counts as relevant). Metrics at
`k_final` = 5; threshold 0.5 (`judge_threshold`, set before any data). Intervals: 95%
bootstrap over the 30 tasks; differences paired by task.

| Judge | What it is | Pairs scored | Cost |
|---|---|---|---|
| none | fused RRF order | 11,381 | 0 |
| cross_encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d2`, local | 11,381 | 0 |
| jev | TypeSafe `jev-1.13.0`, one Noul per pair | 11,381 | USD 0.43 (Jev account) |
| llm | gpt-4.1-mini, 10 candidates per JSON call | 2,980 (sample) | USD 0.54 (stage 1) |

The LLM judge on all queries was estimated at ~USD 2.98 against 0.98 left on the stage-1
line, so it scored a seeded sample of 5 unique queries per task (149 unique queries; every
task kept). **The comparison below is on the 203 queries all four judges scored**; the
full-set results for the other three judges agree with it.

## Ranking (shared queries, 30 tasks)

| Judge | MRR | Coverage@5 | Precision@5 |
|---|---|---|---|
| none | 0.366 [0.292, 0.444] | 0.089 [0.066, 0.114] | 0.164 [0.125, 0.205] |
| cross_encoder | 0.321 [0.255, 0.387] | 0.089 [0.067, 0.112] | 0.165 [0.124, 0.208] |
| llm | 0.483 [0.406, 0.562] | 0.133 [0.105, 0.165] | 0.242 [0.191, 0.301] |
| **jev** | **0.551 [0.479, 0.627]** | **0.141 [0.113, 0.170]** | **0.256 [0.206, 0.310]** |

| Paired difference | MRR | Coverage@5 |
|---|---|---|
| jev - none | +0.185 [0.125, 0.250] | +0.052 [0.035, 0.071] |
| llm - none | +0.117 [0.067, 0.169] | +0.044 [0.031, 0.058] |
| cross_encoder - none | -0.045 [-0.107, 0.011] | -0.000 [-0.014, 0.013] |
| **jev - llm** | **+0.068 [0.022, 0.115]** | +0.007 [-0.004, 0.019] |

## Calibration (all scored pairs on the shared queries)

| Judge | Brier | ECE | Kept at 0.5 (of 20) | Precision at 0.5 |
|---|---|---|---|---|
| cross_encoder | 0.102 | 0.097 | 0.96 | 0.281 |
| llm | 0.145 | 0.157 | 4.6 | 0.288 |
| jev | 0.197 | 0.269 | 6.8 | 0.287 |

## Findings

1. **Jev ranks best.** It beats the LLM judge on MRR by 0.068 [0.022, 0.115]; on
   coverage and precision of the top 5 the two are not distinguishable at n = 30.
2. **A judge helps at all:** both Jev and the LLM judge improve on fused order by every
   ranking metric.
3. **The web-search cross-encoder does not help on banking policy text.** It is no better
   than fused order on the shared queries and slightly worse on the full set (MRR -0.052
   [-0.091, -0.016]). Its low ECE is not a virtue: it rates nearly everything irrelevant
   and keeps about one document in twenty.
4. **Jev is the worst calibrated.** It is overconfident: 6.8 of 20 candidates clear 0.5,
   at 0.29 precision. In A3 the cap of k_final = 5 limits the damage; its probabilities
   should not be read as probabilities.
5. **Even the best judge leaves most required documents out:** coverage@5 of 0.14 per
   query. Labels are per task, and one query rarely needs all of a task's documents, so
   this understates per-query relevance, but the ceiling on what judging alone can do is
   visible.

Per pair, Jev cost about USD 0.000038 and the LLM judge about USD 0.00018, and Jev bills a
separate account, so choosing Jev also keeps A3's judging out of the OpenAI budget.
The 0.5 threshold was not tuned on this data and is not changed after seeing it.
