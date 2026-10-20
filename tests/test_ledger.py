"""Tests for cost/ledger.py. No model, no network: spend is simulated."""

import json

import pytest

from policygate_rag.cost.ledger import (
    ALLOCATIONS,
    TOTAL_USD,
    BudgetExceeded,
    BudgetLine,
    Entry,
    EntryKind,
    Ledger,
    UnknownCost,
    guarded_batches,
)

MINI = "openai/gpt-4.1-mini"


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.json")


def episode(ledger, usd, *, arm="A0", line=BudgetLine.CALIBRATION, seed=0):
    return ledger.record_episode(
        line, arm=arm, task_id="task_001", seed=seed, model=MINI,
        agent_usd=usd * 0.7, user_usd=usd * 0.3,
    )


class TestRecording:
    def test_allocations_sum_to_the_ceiling(self):
        assert sum(ALLOCATIONS.values()) == pytest.approx(TOTAL_USD)

    def test_an_episode_books_agent_plus_user(self, ledger):
        entry = episode(ledger, 0.02)
        assert entry.usd == pytest.approx(0.02)
        assert entry.label == "A0/task_001/seed0"
        assert ledger.spent(BudgetLine.CALIBRATION) == pytest.approx(0.02)

    @pytest.mark.parametrize("agent,user", [(None, 0.01), (0.01, None)])
    def test_unknown_cost_stops_rather_than_counting_as_zero(self, ledger, agent, user):
        with pytest.raises(UnknownCost):
            ledger.record_episode(
                BudgetLine.CALIBRATION, arm="A0", task_id="t", seed=0, model=MINI,
                agent_usd=agent, user_usd=user,
            )
        assert ledger.entries == ()

    def test_the_reserve_cannot_be_charged(self, ledger):
        with pytest.raises(BudgetExceeded):
            ledger.record(Entry(line=BudgetLine.RESERVE, kind=EntryKind.CALL, usd=0.01, label="x"))

    def test_persisted_after_every_write_and_reloadable(self, ledger):
        episode(ledger, 0.02)
        episode(ledger, 0.03, seed=1)
        reloaded = Ledger.load(ledger.path)
        assert reloaded.entries == ledger.entries
        on_disk = json.loads(ledger.path.read_text())
        assert on_disk["spent"]["calibration"] == pytest.approx(0.05)

    def test_a_ledger_for_another_budget_is_refused(self, ledger):
        episode(ledger, 0.02)
        raw = json.loads(ledger.path.read_text())
        raw["total_usd"] = 50
        ledger.path.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            Ledger.load(ledger.path)


class TestReconciliation:
    def test_a_gap_is_booked_in_either_direction(self, ledger):
        episode(ledger, 0.10)
        up = ledger.reconcile(BudgetLine.CALIBRATION, 0.12, "dashboard after batch 1")
        assert up.usd == pytest.approx(0.02)
        down = ledger.reconcile(BudgetLine.CALIBRATION, 0.11, "refund")
        assert down.usd == pytest.approx(-0.01)
        assert ledger.total() == pytest.approx(0.11)

    def test_no_gap_books_nothing(self, ledger):
        episode(ledger, 0.10)
        assert ledger.reconcile(BudgetLine.CALIBRATION, 0.10, "ok") is None


class TestGuard:
    def test_the_prior_is_used_until_enough_comparable_episodes(self, ledger):
        episode(ledger, 0.5)
        p = ledger.project(BudgetLine.CALIBRATION, episodes=4, model=MINI, arm="A0",
                           prior_per_episode=0.05)
        assert p.basis == "prior"
        assert p.projected == pytest.approx(0.5 + 4 * 0.05)

    def test_measured_cost_replaces_the_prior(self, ledger):
        for seed, usd in enumerate([0.02, 0.04, 0.06]):
            episode(ledger, usd, seed=seed)
        p = ledger.project(BudgetLine.STAGE2, episodes=10, model=MINI, arm="A0",
                           prior_per_episode=1.0)
        assert p.basis == "measured (n=3)"
        assert p.per_episode == pytest.approx(0.04)

    def test_other_arms_do_not_count_as_comparable(self, ledger):
        for seed in range(3):
            episode(ledger, 0.02, arm="A4", seed=seed)
        p = ledger.project(BudgetLine.STAGE2, episodes=1, model=MINI, arm="A0",
                           prior_per_episode=0.09)
        assert p.basis == "prior"

    def test_a_batch_over_its_line_is_refused_before_it_starts(self, ledger):
        with pytest.raises(BudgetExceeded, match="calibration"):
            ledger.guard(BudgetLine.CALIBRATION, episodes=21, model=MINI, arm="A0",
                         prior_per_episode=0.05)

    def test_a_batch_that_fits_returns_its_projection(self, ledger):
        p = ledger.guard(BudgetLine.CALIBRATION, episodes=9, model=MINI, arm="A0",
                         prior_per_episode=0.10)
        assert p.headroom == pytest.approx(0.10)

    def test_chunks_stop_when_actual_cost_outruns_the_prior(self, ledger):
        ran = []

        def run_chunk(i, size):
            ran.append(i)
            for k in range(size):  # each episode costs 0.15, far above the 0.05 prior
                episode(ledger, 0.15, seed=100 * i + k)

        with pytest.raises(BudgetExceeded):
            guarded_batches(
                ledger, BudgetLine.CALIBRATION, chunks=[3, 3, 3], model=MINI, arm="A0",
                prior_per_episode=0.05, run_chunk=run_chunk,
            )
        assert ran == [0]  # after chunk 0 the measured mean (0.15) projects 0.45 + 6 x 0.15
        assert ledger.spent(BudgetLine.CALIBRATION) < ALLOCATIONS[BudgetLine.CALIBRATION]


class TestTransfers:
    def test_unspent_money_moves_and_the_guard_follows(self, ledger):
        episode(ledger, 0.40)  # calibration has 0.60 left
        ledger.transfer(BudgetLine.CALIBRATION, BudgetLine.STAGE2, 0.60, "fund A1")
        assert ledger.allocation(BudgetLine.STAGE2) == pytest.approx(25.60)
        assert ledger.allocation(BudgetLine.CALIBRATION) == pytest.approx(0.40)
        assert sum(ledger.allocation(k) for k in BudgetLine) == pytest.approx(TOTAL_USD)
        p = ledger.project(BudgetLine.STAGE2, episodes=1, model=MINI, arm="A1",
                           prior_per_episode=0.1)
        assert p.allocation == pytest.approx(25.60)

    def test_only_unspent_money_moves_and_never_the_reserve(self, ledger):
        episode(ledger, 0.40)
        with pytest.raises(BudgetExceeded, match="unspent"):
            ledger.transfer(BudgetLine.CALIBRATION, BudgetLine.STAGE2, 0.61, "too much")
        with pytest.raises(BudgetExceeded, match="reserve"):
            ledger.transfer(BudgetLine.RESERVE, BudgetLine.STAGE2, 1.0, "no")
        with pytest.raises(BudgetExceeded, match="reserve"):
            ledger.transfer(BudgetLine.STAGE2, BudgetLine.RESERVE, 1.0, "no")

    def test_transfers_persist_with_their_reason(self, ledger):
        ledger.transfer(BudgetLine.DEVELOPMENT, BudgetLine.STAGE2, 1.5, "fund A1")
        back = Ledger.load(ledger.path)
        assert back.transfers[0].reason == "fund A1"
        assert back.allocation(BudgetLine.DEVELOPMENT) == pytest.approx(1.5)
