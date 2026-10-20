"""Run the runner, and restart it with --resume when it stalls.

    python -m policygate_rag.supervise --log runs/x.log -- --out runs/x --line stage2 \\
        --arms A3 --tasks subset --trials 3 --prior 0.07 --chunk-size 1 --max-concurrency 3

Two batches have stalled with every model call queued for an HTTP connection
that never came: litellm's client pool (100 connections) held connections no
thread was using, and the cause is not yet found. Rather than depend on a fix,
runs recover: if nothing in the run directory is written for ``--stall-minutes``,
the supervisor asks the runner for a stack dump (SIGUSR1; faulthandler writes
every thread to the log), stops it, and restarts it with ``--resume``, which
skips finished chunks. It gives up after ``--max-restarts``.

Prints one line per event (start, stall, restart, done, failed) for whoever is
watching; the runner's own output goes to ``--log``.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def newest_write(run_dir: Path) -> float:
    paths = list(run_dir.glob("*/trace.jsonl")) + list(run_dir.glob("*/results_chunk_*.json"))
    return max((p.stat().st_mtime for p in paths), default=0.0)


def stop(proc: subprocess.Popen, grace: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def supervise(runner_args: list[str], log: Path, *, stall_s: float, max_restarts: int,
              poll_s: float = 30.0, python: str = sys.executable) -> int:
    out = runner_args[runner_args.index("--out") + 1]
    run_dir = ROOT / out
    restarts = 0
    while True:
        args = list(runner_args)
        if run_dir.joinpath("plan.json").exists() and "--resume" not in args:
            args.append("--resume")
        with log.open("a") as f:
            proc = subprocess.Popen([python, "-m", "policygate_rag.runner", *args],
                                    cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"start pid={proc.pid} resume={'--resume' in args}", flush=True)
        started = time.time()
        while proc.poll() is None:
            time.sleep(poll_s)
            idle = time.time() - max(started, newest_write(run_dir))
            if idle > stall_s and proc.poll() is None:
                print(f"STALL: nothing written for {idle:.0f}s; dumping stacks, stopping "
                      f"pid={proc.pid}", flush=True)
                proc.send_signal(signal.SIGUSR1)
                time.sleep(5)
                stop(proc)
                break
        code = proc.returncode
        if code == 0:
            print("done", flush=True)
            return 0
        if restarts >= max_restarts:
            print(f"failed: exit {code} after {restarts} restarts", flush=True)
            return code or 1
        restarts += 1
        print(f"restart {restarts}/{max_restarts} (exit {code})", flush=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: supervise [options] -- <runner arguments>", file=sys.stderr)
        return 2
    split = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--stall-minutes", type=float, default=15.0)
    p.add_argument("--max-restarts", type=int, default=4)
    opts = p.parse_args(argv[:split])
    return supervise(argv[split + 1:], opts.log, stall_s=opts.stall_minutes * 60,
                     max_restarts=opts.max_restarts)


if __name__ == "__main__":
    sys.exit(main())
