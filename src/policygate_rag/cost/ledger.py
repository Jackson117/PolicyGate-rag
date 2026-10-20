"""The budget ledger and the guard that stops a batch before it overspends.

USD 33.80 is a hard ceiling for all model usage, development included. The
ledger is the project's own record of spend (``runs/ledger.json``, committed);
it is reconciled against the provider's usage dashboard after each batch, and
a reconciliation that disagrees is itself an entry, so the record never
silently drifts from the bill.

The guard is deliberately simple and conservative: before a batch, projected
spend on its budget line is what the line has spent plus the remaining
episodes times the mean cost of comparable episodes. Comparable means same
line, model and arm once there are enough of them, because arms differ in cost
(golden inlines up to ~13k tokens of documents); until then it is a prior the
caller must state. If the projection exceeds the line's allocation, or the
ceiling minus the reserve, the batch does not start.

Pure Python: no model, no network.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class BudgetLine(StrEnum):
    RESERVE = "reserve"
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    STAGE1 = "stage1"
    STAGE2 = "stage2"


TOTAL_USD = 33.80

ALLOCATIONS: dict[BudgetLine, float] = {
    BudgetLine.RESERVE: 3.80,
    BudgetLine.DEVELOPMENT: 3.00,
    BudgetLine.CALIBRATION: 1.00,
    BudgetLine.STAGE1: 1.00,
    BudgetLine.STAGE2: 25.00,
}
assert abs(sum(ALLOCATIONS.values()) - TOTAL_USD) < 1e-9, "allocations must sum to the total"

MIN_COMPARABLE = 3
"""Episodes of the same line, model and arm before their mean replaces the prior."""


class EntryKind(StrEnum):
    EPISODE = "episode"
    """One tau2 simulation: agent plus user simulator."""

    CALL = "call"
    """Spend outside an episode: embedding the KB, offline judge calls."""

    RECONCILIATION = "reconciliation"
    """The difference between the provider dashboard and the ledger, booked as spend."""


class Entry(BaseModel):
    model_config = ConfigDict(frozen=True)

    line: BudgetLine
    kind: EntryKind
    usd: float
    label: str
    """What was paid for: ``{arm}/{task_id}/seed{seed}`` for an episode, free text otherwise."""
    model: str | None = None
    arm: str | None = None
    batch: str | None = None
    at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Transfer(BaseModel):
    """Money moved between budget lines, by decision, with the reason recorded."""

    model_config = ConfigDict(frozen=True)

    source: BudgetLine
    target: BudgetLine
    usd: float = Field(gt=0)
    reason: str
    at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class BudgetExceeded(Exception):
    """A batch would exceed its allocation. Raised before anything is spent."""


class UnknownCost(Exception):
    """An episode came back without a cost. Unaccountable spend stops the run."""


class Projection(BaseModel):
    model_config = ConfigDict(frozen=True)

    line: BudgetLine
    spent: float
    remaining_episodes: int
    per_episode: float
    basis: str
    """Where ``per_episode`` came from: ``"measured (n=...)"`` or ``"prior"``."""
    projected: float
    allocation: float

    @property
    def headroom(self) -> float:
        return self.allocation - self.projected


class Ledger:
    """Append-only spend record, persisted as JSON after every write."""

    def __init__(self, path: Path, entries: Iterable[Entry] = (),
                 transfers: Iterable[Transfer] = ()) -> None:
        self.path = path
        self._entries: list[Entry] = list(entries)
        self._transfers: list[Transfer] = list(transfers)

    @classmethod
    def load(cls, path: Path) -> Ledger:
        if not path.exists():
            return cls(path)
        raw = json.loads(path.read_text())
        if raw.get("total_usd") != TOTAL_USD:
            raise ValueError(f"{path} was written for a different budget: {raw.get('total_usd')}")
        return cls(path, (Entry.model_validate(e) for e in raw["entries"]),
                   (Transfer.model_validate(t) for t in raw.get("transfers", [])))

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    # --- writing -------------------------------------------------------------

    def record(self, entry: Entry) -> None:
        if entry.line is BudgetLine.RESERVE:
            raise BudgetExceeded("the reserve is untouchable; book spend to another line")
        if entry.usd < 0 and entry.kind is not EntryKind.RECONCILIATION:
            raise ValueError(f"negative spend outside a reconciliation: {entry}")
        self._entries.append(entry)
        self._save()

    def record_episode(
        self,
        line: BudgetLine,
        *,
        arm: str,
        task_id: str,
        seed: int,
        model: str,
        agent_usd: float | None,
        user_usd: float | None,
        batch: str | None = None,
    ) -> Entry:
        if agent_usd is None or user_usd is None:
            # tau2 reports None when any message on that side lacks a cost.
            raise UnknownCost(
                f"{arm}/{task_id}/seed{seed}: agent={agent_usd} user={user_usd}; "
                "stopping rather than booking unknown spend as zero"
            )
        entry = Entry(
            line=line, kind=EntryKind.EPISODE, usd=agent_usd + user_usd,
            label=f"{arm}/{task_id}/seed{seed}", model=model, arm=arm, batch=batch,
        )
        self.record(entry)
        return entry

    def reconcile(self, line: BudgetLine, dashboard_usd: float, note: str) -> Entry | None:
        """Book the gap between the provider's figure and the ledger's total.

        ``dashboard_usd`` is cumulative spend on the account for this project.
        A positive gap is booked to ``line``; a negative one is booked too,
        because a ledger that overstates spend is also wrong.
        """
        gap = round(dashboard_usd - self.total(), 6)
        if gap == 0:
            return None
        entry = Entry(
            line=line, kind=EntryKind.RECONCILIATION, usd=gap,
            label=f"dashboard {dashboard_usd:.4f} vs ledger {self.total():.4f}: {note}",
        )
        self.record(entry)
        return entry

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_usd": TOTAL_USD,
            "allocations": {k.value: v for k, v in ALLOCATIONS.items()},
            "effective_allocations": {k.value: round(self.allocation(k), 6) for k in BudgetLine},
            "transfers": [t.model_dump(mode="json") for t in self._transfers],
            "spent": {k.value: round(self.spent(k), 6) for k in BudgetLine},
            "entries": [e.model_dump(mode="json") for e in self._entries],
        }
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, self.path)

    # --- reading -------------------------------------------------------------

    @property
    def transfers(self) -> tuple[Transfer, ...]:
        return tuple(self._transfers)

    def allocation(self, line: BudgetLine) -> float:
        """The line's allocation after every recorded transfer."""
        moved_in = sum(t.usd for t in self._transfers if t.target is line)
        moved_out = sum(t.usd for t in self._transfers if t.source is line)
        return ALLOCATIONS[line] + moved_in - moved_out

    def transfer(self, source: BudgetLine, target: BudgetLine, usd: float, reason: str) -> Transfer:
        """Move unspent money from one line to another. The reserve is untouchable, and a
        line can only give what it has not spent; the total never changes."""
        if BudgetLine.RESERVE in (source, target):
            raise BudgetExceeded("the reserve is untouchable")
        if source is target:
            raise ValueError("a transfer needs two different lines")
        unspent = self.allocation(source) - self.spent(source)
        if usd > unspent + 1e-9:
            raise BudgetExceeded(f"{source.value} has {unspent:.4f} unspent, not {usd:.4f}")
        t = Transfer(source=source, target=target, usd=usd, reason=reason)
        self._transfers.append(t)
        self._save()
        return t

    def spent(self, line: BudgetLine | None = None) -> float:
        return sum(e.usd for e in self._entries if line is None or e.line is line)

    def total(self) -> float:
        return self.spent()

    def episode_costs(
        self, *, line: BudgetLine | None = None, model: str | None = None, arm: str | None = None
    ) -> list[float]:
        return [
            e.usd for e in self._entries
            if e.kind is EntryKind.EPISODE
            and (line is None or e.line is line)
            and (model is None or e.model == model)
            and (arm is None or e.arm == arm)
        ]

    # --- the guard -------------------------------------------------------------

    def project(
        self,
        line: BudgetLine,
        *,
        episodes: int,
        model: str,
        arm: str,
        prior_per_episode: float,
    ) -> Projection:
        """Projected spend on ``line`` if ``episodes`` more of this arm run.

        Uses the mean of this model and arm's measured episodes once there are
        ``MIN_COMPARABLE`` of them, from any line (calibration episodes inform
        stage 2); otherwise the prior, which the caller has to state.
        """
        measured = self.episode_costs(model=model, arm=arm)
        if len(measured) >= MIN_COMPARABLE:
            per_episode, basis = statistics.fmean(measured), f"measured (n={len(measured)})"
        else:
            per_episode, basis = prior_per_episode, "prior"
        spent = self.spent(line)
        return Projection(
            line=line, spent=spent, remaining_episodes=episodes, per_episode=per_episode,
            basis=basis, projected=spent + episodes * per_episode,
            allocation=self.allocation(line),
        )

    def guard(
        self,
        line: BudgetLine,
        *,
        episodes: int,
        model: str,
        arm: str,
        prior_per_episode: float,
    ) -> Projection:
        """Raise :class:`BudgetExceeded` unless the batch fits; return the projection."""
        projection = self.project(
            line, episodes=episodes, model=model, arm=arm, prior_per_episode=prior_per_episode
        )
        if projection.projected > projection.allocation:
            raise BudgetExceeded(
                f"{line.value}: projected {projection.projected:.4f} > allocation "
                f"{projection.allocation:.2f} (spent {projection.spent:.4f} + {episodes} x "
                f"{projection.per_episode:.4f}, {projection.basis})"
            )
        usable = TOTAL_USD - ALLOCATIONS[BudgetLine.RESERVE]
        overall = self.total() + episodes * projection.per_episode
        if overall > usable:
            raise BudgetExceeded(
                f"overall: projected {overall:.4f} > {usable:.2f} usable (reserve excluded)"
            )
        return projection


def guarded_batches(
    ledger: Ledger,
    line: BudgetLine,
    *,
    chunks: Iterable[int],
    model: str,
    arm: str,
    prior_per_episode: float,
    run_chunk: Callable[[int, int], None],
) -> list[Projection]:
    """Run chunks of episodes, re-checking the projection before each.

    ``run_chunk(index, size)`` runs one chunk and books its episodes in the
    ledger; the next projection then includes what the chunk actually cost.
    Remaining episodes always means all chunks still to run, so a chunk that
    turns out expensive stops the batch while there is still budget left.
    """
    sizes = list(chunks)
    projections = []
    for i, size in enumerate(sizes):
        projections.append(
            ledger.guard(
                line, episodes=sum(sizes[i:]), model=model, arm=arm,
                prior_per_episode=prior_per_episode,
            )
        )
        run_chunk(i, size)
    return projections
