"""pass^k with bootstrap intervals over tasks, and paired arm differences.

pass^k for one task with n trials and c successes is C(c, k) / C(n, k): the
probability that k trials drawn without replacement all succeed (tau-bench's
estimator). An arm's pass^k is the mean over tasks.

Intervals resample *tasks*, the unit the claim generalises over; trials within
a task are not independent draws of the population. Arms share tasks, so a
difference between arms is computed per task first and the per-task
differences are resampled together (paired bootstrap).

    python -m policygate_rag.eval.passk runs/w1_floor_ceiling A4 A0
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import comb
from pathlib import Path

from pydantic import BaseModel, ConfigDict

SUCCESS = 1.0
"""A DB-scored episode passes only at full reward."""

BOOTSTRAP = 10_000
SEED = 20261012


class Interval(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimate: float
    low: float
    high: float
    n_tasks: int

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n_tasks} tasks)"


def pass_hat_k(successes: int, trials: int, k: int) -> float:
    if k > trials:
        raise ValueError(f"pass^{k} needs at least {k} trials, got {trials}")
    return comb(successes, k) / comb(trials, k)


def per_task(outcomes: Mapping[str, Sequence[bool]], k: int) -> dict[str, float]:
    return {t: pass_hat_k(sum(o), len(o), k) for t, o in outcomes.items()}


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    idx = q * (len(values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


def bootstrap_mean(values: Sequence[float], *, alpha: float = 0.05,
                   n: int = BOOTSTRAP, seed: int = SEED) -> Interval:
    rng = random.Random(seed)
    values = list(values)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n)]
    return Interval(
        estimate=statistics.fmean(values),
        low=_percentile(means, alpha / 2),
        high=_percentile(means, 1 - alpha / 2),
        n_tasks=len(values),
    )


def arm_pass_k(outcomes: Mapping[str, Sequence[bool]], k: int, **kw) -> Interval:
    return bootstrap_mean(list(per_task(outcomes, k).values()), **kw)


def paired_difference(a: Mapping[str, Sequence[bool]], b: Mapping[str, Sequence[bool]],
                      k: int, **kw) -> Interval:
    """pass^k of ``a`` minus ``b`` over the tasks both ran, paired by task."""
    shared = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError(f"arms ran different tasks: {sorted(set(a) ^ set(b))}")
    pa, pb = per_task(a, k), per_task(b, k)
    return bootstrap_mean([pa[t] - pb[t] for t in shared], **kw)


def load_outcomes(run_dir: Path, arm: str) -> dict[str, list[bool]]:
    from tau2.data_model.simulation import Results

    outcomes: dict[str, list[bool]] = defaultdict(list)
    for path in sorted((run_dir / arm).glob("results_chunk_*.json")):
        for sim in Results.load(path).simulations:
            reward = None if sim.reward_info is None else sim.reward_info.reward
            outcomes[sim.task_id].append(reward is not None and reward >= SUCCESS)
    return dict(outcomes)


def main() -> None:
    run_dir, arm_a, arm_b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    a, b = load_outcomes(run_dir, arm_a), load_outcomes(run_dir, arm_b)
    trials = min(len(o) for o in [*a.values(), *b.values()])
    report = {}
    for k in range(1, trials + 1):
        report[f"pass^{k}"] = {
            arm_a: str(arm_pass_k(a, k)),
            arm_b: str(arm_pass_k(b, k)),
            f"{arm_a} - {arm_b} (paired)": str(paired_difference(a, b, k)),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
