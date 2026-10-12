"""The committed task subset: 30 DB-reward tasks, stratified by required-document count.

Lives in ``eval/`` because stratifying reads ``required_documents``, a label.
What leaves this module is task ids and stratum names; nothing outside
``eval/`` needs more than the ids.

Stratification: the tasks whose reward is the DB hash alone (87 at the pin),
ordered by number of required documents (task id breaking ties), are cut into
three rank terciles of near equal size, and ``per_stratum`` tasks are drawn
from each with a fixed seed. Rank terciles rather than value cut points,
because counts tie heavily and a value cut point would put most tasks in one
stratum.

Excluded and listed separately: ACTION-reward tasks (trajectory matching,
reported on their own) and any task whose reward also needs an LLM judge
(``NL_ASSERTION``: task_102 at the pin), which would bring a judge model and its
cost into the comparison.

Tasks are read through tau2's loader, which reads ``tasks/task_*.json``. The
aggregate ``tasks.json`` beside it is stale at the pin (13 of 97 tasks differ,
7 of them in their labels) and is never used.

Usage (writes ``experiments/subset.json``):

    python -m policygate_rag.eval.subset
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

SEED = 20261011
"""Fixed when the subset was first drawn. Changing it is changing the experiment."""

PER_STRATUM = 10
STRATA = ("few", "middle", "many")


class TaskInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    reward_basis: tuple[str, ...]
    n_required: int


class Stratum(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    n_required_min: int
    n_required_max: int
    population: int
    task_ids: tuple[str, ...]


class Subset(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    per_stratum: int
    tau2_commit: str
    population: int
    """DB-reward tasks the subset was drawn from."""
    strata: tuple[Stratum, ...]
    action_task_ids: tuple[str, ...]
    """ACTION-reward tasks, excluded from the subset and reported separately."""
    judged_task_ids: tuple[str, ...]
    """Tasks whose reward also needs an LLM judge (NL_ASSERTION). Excluded."""

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(t for s in self.strata for t in s.task_ids)


def task_infos(tasks: Sequence) -> list[TaskInfo]:
    """Reduce tau2 tasks to what stratification needs."""
    infos = []
    for task in tasks:
        criteria = task.evaluation_criteria
        basis = tuple(sorted(b.value if hasattr(b, "value") else str(b)
                             for b in (criteria.reward_basis or ()))) if criteria else ()
        infos.append(TaskInfo(
            task_id=task.id, reward_basis=basis, n_required=len(task.required_documents or ()),
        ))
    return infos


def draw(
    infos: Sequence[TaskInfo], *, seed: int, per_stratum: int, tau2_commit: str
) -> Subset:
    db = sorted(
        (i for i in infos if i.reward_basis == ("DB",)), key=lambda i: (i.n_required, i.task_id)
    )
    action = sorted(i.task_id for i in infos if "ACTION" in i.reward_basis)
    judged = sorted(
        i.task_id for i in infos
        if "NL_ASSERTION" in i.reward_basis and "ACTION" not in i.reward_basis
    )
    other = [
        i.task_id for i in infos
        if i.reward_basis != ("DB",) and i.task_id not in action and i.task_id not in judged
    ]
    if other:
        raise ValueError(f"tasks with an unexpected reward basis: {other}")

    rng = random.Random(seed)
    n = len(db)
    cuts = [0, round(n / 3), round(2 * n / 3), n]
    strata = []
    for name, lo, hi in zip(STRATA, cuts, cuts[1:], strict=False):
        members = db[lo:hi]
        if len(members) < per_stratum:
            raise ValueError(f"stratum {name} has {len(members)} tasks, fewer than {per_stratum}")
        chosen = sorted(rng.sample([m.task_id for m in members], per_stratum))
        strata.append(Stratum(
            name=name,
            n_required_min=members[0].n_required,
            n_required_max=members[-1].n_required,
            population=len(members),
            task_ids=tuple(chosen),
        ))
    return Subset(
        seed=seed, per_stratum=per_stratum, tau2_commit=tau2_commit, population=n,
        strata=tuple(strata), action_task_ids=tuple(action), judged_task_ids=tuple(judged),
    )


def summary(subset: Subset, infos: Sequence[TaskInfo]) -> str:
    by_id = {i.task_id: i for i in infos}
    lines = [f"{subset.population} DB tasks -> {len(subset.task_ids)} drawn (seed {subset.seed})"]
    for s in subset.strata:
        counts = [by_id[t].n_required for t in s.task_ids]
        lines.append(
            f"  {s.name:6s} required docs {s.n_required_min}-{s.n_required_max} "
            f"(pop {s.population}): drawn median {statistics.median(counts)}"
        )
    lines.append(f"  ACTION tasks reported separately: {len(subset.action_task_ids)}")
    lines.append(f"  LLM-judged tasks excluded: {list(subset.judged_task_ids)}")
    return "\n".join(lines)


def main() -> None:
    import subprocess

    from tau2.domains.banking_knowledge.environment import get_tasks

    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "-C", str(root / "third_party" / "tau2-bench"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    infos = task_infos(get_tasks())
    subset = draw(infos, seed=SEED, per_stratum=PER_STRATUM, tau2_commit=commit)
    out = root / "experiments" / "subset.json"
    out.parent.mkdir(exist_ok=True)
    payload = subset.model_dump(mode="json") | {"task_ids": list(subset.task_ids)}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(summary(subset, infos))
    print(f"wrote {out}")


def load(path: Path) -> Subset:
    raw = json.loads(path.read_text())
    raw.pop("task_ids", None)
    return Subset.model_validate(raw)


if __name__ == "__main__":
    main()
