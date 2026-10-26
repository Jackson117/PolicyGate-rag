# Hard constraints

Nine rules fixed before the first episode ran. Source comments cite them by
number — "hard constraint 3" means the third one here. Three of them are
enforced by tests that must stay green in CI.

### 1. Python `>=3.12,<3.14`

The same range as policygate.

### 2. The same τ²-bench pin as policygate

Commit `a2c0247`, vendored under `third_party/`. The retrieval variants this
study depends on — `no_knowledge`, `bm25`, `golden_retrieval` — were confirmed
to exist at that pin and to build an environment offline before any arm was
designed. Bumping the pin would affect both projects, so it is not done here
alone.

### 3. Label isolation

`required_documents`, and any other evaluation field in the task files, may be
read **only** by code under `src/policygate_rag/eval/`. Agent, graph, retrieval
and judge code never import from `eval/` and never read task files.

τ²'s `Task` object carries `required_documents` as a plain attribute, so
`RagAgent` is never handed a `Task` at all: the agent factory drops the `task`
argument τ² passes and keeps only the id. `tests/test_label_isolation.py`
enforces this by static scan.

This is the constraint the study's credibility rests on. A retrieval pipeline
that can see which documents are required is not measuring retrieval.

### 4. policygate is a dependency, not a workspace

Installed as a git dependency pinned to tag `v0.3.0` — not a path dependency,
which would track whatever is in its working tree. Its code is never edited
from here. A change it needs is made there, tested against its own retail
results, tagged, and re-pinned here.

### 5. USD 33.80 for all model usage, development included

A hard ceiling, not a target. The cost guard existed before the first
multi-episode run and refuses to start, or to continue, a batch whose projected
spend crosses its budget line.

### 6. One model configuration per comparison

Every arm in a comparison uses the same agent model and the same user-simulator
model. Models are never mixed across arms — that would confound the thing being
measured with the thing held fixed.

### 7. Evaluation code is independent of the system under test

`eval/`, `index/` and `judges/` are pure Python: no LangGraph imports and no
network calls in unit tests. The exception is the thin client wrappers for the
API-backed judges, which is why they are their own modules — `index/` holds
vectors and never calls an embedding API. `tests/test_import_boundaries.py`
enforces it.

### 8. Every external model version is pinned

Jev is called as `jev-1.13.0`, never `jev-latest`: a floating tag moves under a
running experiment. Arms that do not use Jev run without a Jev key at all.

### 9. Never write to τ²'s world-state database

Inherited from policygate. Scoring is a hash comparison of world state, so one
extra byte fails every task. Whatever audit storage policygate uses stays where
policygate puts it.
