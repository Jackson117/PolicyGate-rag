"""An HTTP client the study owns for litellm, and a monitor of its connection pool.

Batches have stalled with every model call queued for a pool connection while no
thread used any connection in the pool. The pool is tau2's: importing
``tau2.utils.llm_utils`` sets ``litellm.client_session`` to an ``httpx.Client`` with
10 connections, 5 kept alive. To find out what kind of leak fills it, the runner
replaces that client -- after tau2's import, or tau2 overwrites it -- with this one,
on the same limits so every model call behaves exactly as in earlier runs, and
logs the pool every minute: how many connections exist, how many are idle or
active, how many requests wait.
Connections that stay "active" with nobody using them are leaked responses;
a pool that fills with closed connections is another story.

The pool is read through httpcore internals (httpcore 1.0.9); any failure to read
them loses the diagnostic, never the run.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)
"""tau2's own (tau2/utils/llm_utils.py at the pin): the transport every run has used."""


def make_client(timeout_s: float) -> httpx.Client:
    return httpx.Client(limits=LIMITS, timeout=timeout_s, follow_redirects=True)


def pool_state(client: httpx.Client) -> dict:
    try:
        pool = client._transport._pool
        conns = list(pool._connections)
        return {
            "connections": len(conns),
            "idle": sum(1 for c in conns if c.is_idle()),
            "available": sum(1 for c in conns if c.is_available()),
            "closed": sum(1 for c in conns if c.is_closed()),
            "waiting_requests": sum(1 for r in list(pool._requests) if r.connection is None),
            "assigned_requests": sum(1 for r in list(pool._requests) if r.connection is not None),
        }
    except Exception as e:  # internals moved: lose the diagnostic, not the run
        return {"error": f"{type(e).__name__}: {e}"}


def start_monitor(client: httpx.Client, path: Path, every_s: float = 60.0) -> threading.Thread:
    def loop() -> None:
        while True:
            row = {"at": datetime.now(UTC).isoformat()} | pool_state(client)
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            time.sleep(every_s)

    thread = threading.Thread(target=loop, name="http-pool-monitor", daemon=True)
    thread.start()
    return thread
