"""Run arms through tau2 behind the budget guard, booking every episode.

    python -m policygate_rag.runner --out runs/calibration --line calibration \\
        --arms A0 A1 A4 --tasks calibration --trials 1 --prior 0.10 [--dry-run]

Per arm, the episodes are split into chunks; the guard re-projects the budget
line before each chunk using what earlier chunks actually cost, and a chunk
that would overrun the allocation never starts. Each episode is booked from
tau2's results (agent plus user cost). The agent's own per-turn cost rows in
the trace are summed as a cross-check: tau2 keeps only the last attempt of an
episode it retried, so agent spend on discarded attempts shows up as a gap,
which is booked too. The user simulator's spend on discarded attempts is only
visible on the provider dashboard; reconcile after each batch.

Each invocation gets a new run id and a tau2 save path of its own: tau2 resumes
any checkpoint under the same save path, which once let one PolicyGate run load
another's simulations without a model call. An existing output directory is
refused rather than resumed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from policygate_rag.cost.ledger import BudgetLine, Entry, EntryKind, Ledger, guarded_batches

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "runs" / "ledger.json"


def load_arms(model: str):
    spec = importlib.util.spec_from_file_location("arms", ROOT / "experiments" / "arms.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["arms"] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module.arms(model)


def select_tasks(which: str) -> list[str]:
    """``subset`` (all 30), ``calibration`` (first task of each stratum), or ids."""
    # Ids only, read from the committed file: importing eval/ from here would
    # break label isolation (hard constraint 3), and ids are all a run needs.
    subset = json.loads((ROOT / "experiments" / "subset.json").read_text())
    if which == "subset":
        return list(subset["task_ids"])
    if which == "calibration":
        return [stratum["task_ids"][0] for stratum in subset["strata"]]
    return which.split(",")


def register(arm, out_dir: Path, run_id: str, generate=None):
    """Register a tau2 agent factory for ``arm`` and return its name."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from tau2.domains.banking_knowledge.environment import get_environment
    from tau2.registry import registry

    from policygate_rag.agent import RagAgent
    from policygate_rag.tracing.recorder import TraceRecorder

    name = f"policygate_rag_{run_id}_{arm.id}"
    toolkit = get_environment(retrieval_variant=arm.env_variant).tools
    (out_dir / arm.id).mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(
        sqlite3.connect(out_dir / arm.id / "checkpoints.sqlite", check_same_thread=False)
    )
    recorder = TraceRecorder(out_dir / arm.id / "trace.jsonl")
    extra = {} if generate is None else {"generate": generate}

    def factory(tools, domain_policy, llm=None, llm_args=None, task=None, **_ignored):
        # Only the id survives: the Task object carries labels (hard constraint 3).
        return RagAgent(
            tools=tools, domain_policy=domain_policy, config=arm.config, toolkit=toolkit,
            task_id=getattr(task, "id", "unknown"), checkpointer=saver, recorder=recorder,
            **extra,
        )

    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(factory, name)
    return name


def run_config(arm, agent_name: str, task_ids, *, trials: int, seed: int, save_to: str,
               max_concurrency: int):
    from tau2.data_model.simulation import TextRunConfig

    c = arm.config
    return TextRunConfig(
        domain="banking_knowledge",
        task_ids=list(task_ids),
        agent=agent_name,
        llm_agent=c.agent_llm,
        llm_args_agent={"temperature": c.agent_temperature},
        user="user_simulator",
        llm_user=c.user_llm,
        llm_args_user={"temperature": c.user_temperature},
        num_trials=trials,
        seed=seed,
        max_concurrency=max_concurrency,
        retrieval_config=arm.env_variant,
        save_to=save_to,
    )


def traced_agent_cost(trace: Path, thread_prefix: str) -> float:
    if not trace.exists():
        return 0.0
    total = 0.0
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        if row["type"] == "turn_cost" and row["thread_id"].startswith(thread_prefix):
            total += row["data"]["usd"]
    return total


def run(args, *, generate=None) -> int:
    from tau2.runner import run_domain

    out_dir = (ROOT / args.out).resolve()
    if out_dir.exists() and not args.dry_run:
        print(f"{out_dir} exists; use a new directory (results are never mixed)", file=sys.stderr)
        return 2
    line = BudgetLine(args.line)
    all_arms = load_arms(args.model)
    arms = [all_arms[a] for a in args.arms]
    tasks = select_tasks(args.tasks)
    ledger = Ledger.load(args.ledger)
    run_id = uuid.uuid4().hex[:10]

    plan = {
        "run_id": run_id, "started_at": datetime.now(UTC).isoformat(), "line": line.value,
        "model": args.model, "arms": {a.id: {"env_variant": a.env_variant,
                                             "config": a.config.model_dump(mode="json")}
                                      for a in arms},
        "tasks": tasks, "trials": args.trials, "seed": args.seed, "chunk_size": args.chunk_size,
        "prior_per_episode": args.prior,
    }
    for arm in arms:
        p = ledger.project(line, episodes=len(tasks) * args.trials, model=args.model,
                           arm=arm.id, prior_per_episode=args.prior)
        print(f"{arm.id}: {len(tasks)} tasks x {args.trials} trials, "
              f"{p.per_episode:.4f}/episode ({p.basis}); line {line.value} "
              f"spent {p.spent:.4f} of {p.allocation:.2f}")
    if args.dry_run:
        return 0
    if generate is None and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    for arm in arms:
        agent_name = register(arm, out_dir, run_id, generate)
        chunks = [tasks[i:i + args.chunk_size] for i in range(0, len(tasks), args.chunk_size)]
        booked_agent = 0.0

        def run_chunk(i, _size, arm=arm, agent_name=agent_name, chunks=chunks):
            nonlocal booked_agent
            config = run_config(
                arm, agent_name, chunks[i], trials=args.trials, seed=args.seed,
                # Absolute, so tau2 writes inside this run's directory, not its own data/.
                save_to=str(out_dir / arm.id / "tau2" / f"chunk{i:03d}"),
                max_concurrency=args.max_concurrency,
            )
            results = run_domain(config)
            results.save(out_dir / arm.id / f"results_chunk_{i:03d}.json")
            for sim in results.simulations:
                ledger.record_episode(
                    line, arm=arm.id, task_id=sim.task_id, seed=sim.seed, model=args.model,
                    agent_usd=sim.agent_cost, user_usd=sim.user_cost, batch=run_id,
                )
                booked_agent += sim.agent_cost

        guarded_batches(
            ledger, line, chunks=[len(c) * args.trials for c in chunks], model=args.model,
            arm=arm.id, prior_per_episode=args.prior, run_chunk=run_chunk,
        )
        traced = traced_agent_cost(out_dir / arm.id / "trace.jsonl", f"{arm.id}-")
        gap = round(traced - booked_agent, 6)
        if gap > 0:
            ledger.record(Entry(
                line=line, kind=EntryKind.CALL, usd=gap, model=args.model, arm=arm.id,
                batch=run_id, label=f"{arm.id}: agent spend on attempts tau2 discarded",
            ))
        print(f"{arm.id}: booked; agent trace {traced:.4f} vs results {booked_agent:.4f}")
    print(f"line {line.value}: spent {ledger.spent(line):.4f}; total {ledger.total():.4f}. "
          "Reconcile against the provider dashboard now.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="output directory, relative to the repo root")
    p.add_argument("--line", required=True, choices=[b.value for b in BudgetLine
                                                       if b is not BudgetLine.RESERVE])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--tasks", default="calibration", help="subset | calibration | id,id,...")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    p.add_argument("--prior", type=float, required=True,
                   help="assumed USD per episode until 3 of the arm are measured")
    p.add_argument("--chunk-size", type=int, default=3, help="tasks per guarded chunk")
    p.add_argument("--max-concurrency", type=int, default=3)
    p.add_argument("--ledger", type=Path, default=LEDGER)
    p.add_argument("--dry-run", action="store_true")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
