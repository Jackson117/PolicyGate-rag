"""W1 cost calibration: measured cost per episode, and the model decision rule.

    python -m policygate_rag.eval.calibration runs/calibration_w1

The rule (CLAUDE.md, "Model decision rule"): stage 2 allows at most USD 0.055
per episode, averaged over arms (25 / 450). If the measured mini cost fits, mini
is used for agent and simulator; otherwise nano for both; if nano does not fit
either, the subset shrinks to 20 tasks, never below 3 trials.

Rewards are printed as counts only. Three tasks per arm say nothing about
pass rates and no interval is computed from them; calibration measures cost.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

STAGE2_PER_EPISODE = 25.0 / 450


def summarise(run_dir: Path) -> dict:
    from tau2.data_model.simulation import Results

    plan = json.loads((run_dir / "plan.json").read_text())
    arms = {}
    for arm in plan["arms"]:
        sims = []
        for path in sorted((run_dir / arm).glob("results_chunk_*.json")):
            sims.extend(Results.load(path).simulations)
        if not sims:
            continue
        agent = [s.agent_cost or 0.0 for s in sims]
        user = [s.user_cost or 0.0 for s in sims]
        arms[arm] = {
            "episodes": len(sims),
            "usd_per_episode": statistics.fmean(a + u for a, u in zip(agent, user, strict=True)),
            "agent_usd": statistics.fmean(agent),
            "user_usd": statistics.fmean(user),
            "max_usd": max(a + u for a, u in zip(agent, user, strict=True)),
            "minutes_per_episode": statistics.fmean((s.duration or 0.0) / 60 for s in sims),
            "messages_per_episode": statistics.fmean(len(s.messages or []) for s in sims),
            "terminations": dict(Counter(
                str(getattr(s.termination_reason, "value", s.termination_reason)) for s in sims
            )),
            "rewards": [
                None if s.reward_info is None else s.reward_info.reward for s in sims
            ],
        }
    mean_over_arms = statistics.fmean(a["usd_per_episode"] for a in arms.values())
    return {
        "model": plan["model"],
        "arms": arms,
        "mean_usd_per_episode_over_arms": mean_over_arms,
        "stage2_allowance": STAGE2_PER_EPISODE,
        "fits": mean_over_arms <= STAGE2_PER_EPISODE,
    }


def main() -> None:
    summary = summarise(Path(sys.argv[1]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
