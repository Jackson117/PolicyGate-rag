"""eval/replay.py: the product dictionary, step alignment, and the context policies."""

from types import SimpleNamespace as NS

from policygate_rag.eval.replay import (
    Episode,
    Lookups,
    Policy,
    episode_steps,
    mentioned_products,
    product_names,
    replay,
)
from policygate_rag.index.corpus import Document

IDS = [
    "doc_checking_accounts_green_account_(checking)_001",
    "doc_savings_accounts_green_account_(savings)_001",
    "doc_checking_accounts_dark_green_account_001",
    "doc_checking_accounts_dark_green_account_002",
    "doc_bank_accounts_bank_accounts_(general)_009",
    "doc_credit_cards_ecocard_001",
]


def test_names_come_from_ids_and_general_sections_are_not_products():
    names = product_names(IDS)
    assert names["green account"] == {"checking_accounts_green_account_(checking)",
                                      "savings_accounts_green_account_(savings)"}
    assert "dark green account" in names and "ecocard" in names
    assert not any("general" in n for n in names)


def test_the_longest_name_wins_and_is_masked():
    names = product_names(IDS)
    assert mentioned_products("I have a Dark-Green Account.", names) == {
        "checking_accounts_dark_green_account"}
    both = mentioned_products("dark green account and a green account", names)
    assert "checking_accounts_green_account_(checking)" in both
    assert mentioned_products("greenaccount ecocards", names) == set()


def _msg(role, content="", calls=()):
    return NS(role=role, content=content,
              tool_calls=[NS(name=n, arguments=a) for n, a in calls] or None)


def test_the_greeting_is_not_an_agent_turn():
    steps = episode_steps([
        _msg("assistant", "Hi! How can I help you today?"),
        _msg("user", "lost my EcoCard"),
        _msg("assistant", calls=[("get_user", {})]),
        _msg("tool", "found"),
        _msg("assistant", "done"),
    ])
    assert [s.turn for s in steps] == [0, 1]
    assert steps[0].tool_calls == [("get_user", {})] and steps[1].tool_calls == []
    assert "found" in steps[1].seen_text and "found" not in steps[0].seen_text


def test_replace_forgets_and_a_working_set_remembers_up_to_its_capacity():
    docs = [Document(doc_id=d, title=d, content="text") for d in IDS]
    look = Lookups(docs)
    msgs = [_msg("assistant", "hi")]
    for k in range(3):
        msgs += [_msg("user", f"turn {k}"), _msg("assistant", calls=[("t", {})])]
    retrieval = {k: {"rounds": [], "kept": [f"d{k}"]} for k in range(3)}
    ep = Episode(thread="A3-t-seed1", task_id="t", steps=episode_steps(msgs), retrieval=retrieval)
    assert replay(ep, Policy("replace", None), look) == [{"d0"}, {"d1"}, {"d2"}]
    assert replay(ep, Policy("ws2", 2), look) == [{"d0"}, {"d0", "d1"}, {"d1", "d2"}]
    assert replay(ep, Policy("all", 0), look)[-1] == {"d0", "d1", "d2"}
