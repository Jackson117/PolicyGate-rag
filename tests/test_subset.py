"""Tests for eval/subset.py on synthetic task summaries. No tau2 needed."""

import pytest

from policygate_rag.eval.subset import TaskInfo, draw


def infos(n_db=87):
    out = [TaskInfo(task_id=f"t{i:03d}", reward_basis=("DB",), n_required=1 + i % 30)
           for i in range(n_db)]
    out.append(TaskInfo(task_id="act", reward_basis=("ACTION",), n_required=4))
    out.append(TaskInfo(task_id="nl", reward_basis=("DB", "NL_ASSERTION"), n_required=4))
    return out


def test_three_strata_of_ten_ordered_by_required_documents():
    s = draw(infos(), seed=1, per_stratum=10, tau2_commit="x")
    assert [len(st.task_ids) for st in s.strata] == [10, 10, 10]
    assert s.strata[0].n_required_max <= s.strata[1].n_required_min
    assert s.strata[1].n_required_max <= s.strata[2].n_required_min
    assert sum(st.population for st in s.strata) == s.population == 87


def test_the_draw_is_fixed_by_the_seed():
    a = draw(infos(), seed=7, per_stratum=10, tau2_commit="x")
    assert a == draw(infos(), seed=7, per_stratum=10, tau2_commit="x")
    assert a.task_ids != draw(infos(), seed=8, per_stratum=10, tau2_commit="x").task_ids


def test_action_and_judged_tasks_are_excluded_and_listed():
    s = draw(infos(), seed=1, per_stratum=10, tau2_commit="x")
    assert s.action_task_ids == ("act",) and s.judged_task_ids == ("nl",)
    assert not {"act", "nl"} & set(s.task_ids)


def test_an_unknown_reward_basis_stops_the_draw():
    bad = infos() + [TaskInfo(task_id="c", reward_basis=("COMMUNICATE",), n_required=1)]
    with pytest.raises(ValueError, match="unexpected reward basis"):
        draw(bad, seed=1, per_stratum=10, tau2_commit="x")
