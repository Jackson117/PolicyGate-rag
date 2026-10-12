"""JSONL trace rows from LangGraph's v2 stream.

One row per stream chunk (``stream_mode=["updates", "custom"]``,
``subgraphs=True``, ``version="v2"``), keyed by the episode's thread id and the
turn number, so a row can be joined to its checkpoint and to tau2's results.
Thread-safe: tau2 runs episodes concurrently in one process.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class TraceRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, thread_id: str, turn: int, chunk: dict[str, Any]) -> None:
        row = {
            "thread_id": thread_id,
            "turn": turn,
            "type": chunk["type"],
            "ns": list(chunk.get("ns", ())),
            "data": chunk["data"],
        }
        line = json.dumps(row, default=str, ensure_ascii=False)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")
