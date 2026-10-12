"""Tests for eval/passk.py. Pure arithmetic: no tau2 objects needed."""

import pytest

from policygate_rag.eval.passk import (
    arm_pass_k,
    bootstrap_mean,
    paired_difference,
    pass_hat_k,
)


def test_pass_hat_k_matches_the_combinatorial_definition():
    assert pass_hat_k(3, 3, 3) == 1.0
    assert pass_hat_k(2, 3, 1) == pytest.approx(2 / 3)
    assert pass_hat_k(2, 3, 2) == pytest.approx(1 / 3)
    assert pass_hat_k(2, 3, 3) == 0.0
    with pytest.raises(ValueError):
        pass_hat_k(1, 2, 3)


def test_pass_k_never_increases_with_k():
    outcomes = {"a": [True, True, False], "b": [True, False, False], "c": [True] * 3}
    est = [arm_pass_k(outcomes, k).estimate for k in (1, 2, 3)]
    assert est == sorted(est, reverse=True)


def test_constant_values_give_a_degenerate_interval():
    i = bootstrap_mean([0.5] * 10)
    assert (i.low, i.estimate, i.high) == (0.5, 0.5, 0.5)


def test_the_interval_contains_the_estimate_and_is_reproducible():
    vals = [0.0, 1.0] * 15
    a, b = bootstrap_mean(vals), bootstrap_mean(vals)
    assert a == b
    assert a.low < a.estimate < a.high


def test_pairing_removes_shared_task_difficulty():
    # Task difficulty varies wildly, but arm A beats B by the same margin on
    # every task: the paired interval is tight, an unpaired view would not be.
    a = {f"t{i}": [True] * 3 if i % 2 else [True, False, False] for i in range(20)}
    b = {f"t{i}": [True, True, False] if i % 2 else [False] * 3 for i in range(20)}
    d = paired_difference(a, b, 1)
    assert d.estimate == pytest.approx(1 / 3)
    assert d.high - d.low == pytest.approx(0.0)


def test_arms_on_different_tasks_are_refused():
    with pytest.raises(ValueError, match="different tasks"):
        paired_difference({"a": [True]}, {"b": [True]}, 1)
