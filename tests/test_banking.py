"""Tests for banking.py against the real banking_knowledge toolkit at the pin.

The environment executes real tool calls here, so the fact extractor is checked
against the text tau2 actually returns, not against a string written to match it.
No LLM, no network.
"""

import json
from datetime import datetime

import pytest
from policygate.harness.approval import auto_reject
from policygate.harness.config import HarnessConfig
from policygate.harness.gating import GateRule, ToolGate
from policygate.harness.session import HarnessSession, all_allowed
from policygate.harness.state import SessionState
from policygate.harness.types import ToolKind, ToolRole
from policygate.harness.validation import DomainConstraints, Validator
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.domains.banking_knowledge.environment import get_db, get_environment

from policygate_rag import banking


@pytest.fixture(scope="module")
def user():
    users = get_db().model_dump()["users"]["data"]
    user_id, record = sorted(users.items())[0]
    return user_id, record


def env(variant="no_knowledge"):
    return get_environment(retrieval_variant=variant)


def tcall(tool, id="c1", **args):
    return Tau2ToolCall(id=id, name=tool, arguments=args)


class TestCatalogue:
    def test_roles_follow_the_policy(self):
        roles = {s.name: s.role for s in banking.catalogue(env().tools)}
        assert roles["get_user_information_by_email"] is ToolRole.IDENTITY_LOOKUP
        assert roles["log_verification"] is ToolRole.IDENTITY_COMMIT
        assert roles["transfer_to_human_agents"] is ToolRole.ESCALATION
        assert roles["get_current_time"] is ToolRole.SUBJECT_INDEPENDENT
        assert roles["get_credit_card_accounts_by_user"] is ToolRole.ORDINARY

    def test_the_bm25_arm_has_kb_search_and_it_needs_no_verification(self):
        roles = {s.name: s.role for s in banking.catalogue(env("bm25").tools)}
        assert roles["KB_search"] is ToolRole.SUBJECT_INDEPENDENT

    def test_discoverable_tools_are_in_the_catalogue_with_tau2_kinds(self):
        toolkit = env().tools
        specs = {s.name: s for s in banking.catalogue(toolkit)}
        discoverable = toolkit.get_discoverable_tools()
        assert discoverable, "no discoverable tools found: the toolkit API moved"
        for name in discoverable:
            assert specs[name].kind.value == toolkit.tool_type(name).value
            assert specs[name].role is ToolRole.ORDINARY
        assert any(specs[n].kind is ToolKind.WRITE for n in discoverable)


class TestUnwrap:
    def test_a_dispatcher_call_becomes_the_operation(self):
        call = tcall(
            "call_discoverable_agent_tool", id="x9",
            agent_tool_name="apply_statement_credit_8472",
            arguments=json.dumps({"user_id": "u1", "amount": 10}),
        )
        unwrapped = banking.unwrap(call)
        assert unwrapped.id == "x9"
        assert unwrapped.name == "apply_statement_credit_8472"
        assert unwrapped.arguments == {"user_id": "u1", "amount": 10}

    @pytest.mark.parametrize("arguments", ["not json", "[1, 2]"])
    def test_unparseable_arguments_leave_the_dispatcher_in_place(self, arguments):
        call = tcall("call_discoverable_agent_tool", agent_tool_name="t", arguments=arguments)
        assert banking.unwrap(call).name == "call_discoverable_agent_tool"

    def test_other_calls_pass_through(self):
        assert banking.unwrap(tcall("get_current_time")).name == "get_current_time"


class TestFacts:
    def _log(self, e, user):
        user_id, rec = user
        result = e.get_response(tcall(
            "log_verification", name=rec["name"], user_id=user_id, address=str(rec["address"]),
            email=rec["email"], phone_number=rec["phone_number"],
            date_of_birth=rec["date_of_birth"], time_verified="2025-11-14 03:40:00 EST",
        ))
        return result

    def test_a_real_verification_log_settles_identity_and_scope(self, user):
        result = self._log(env(), user)
        f = banking.facts("log_verification", result.content, error=result.error)
        assert f.user_id == user[0]
        assert f.scope_targets == (user[0],)

    def test_a_repeated_log_establishes_nothing(self, user):
        e = env()
        self._log(e, user)
        again = self._log(e, user)
        assert "Failed" in again.content
        assert banking.facts("log_verification", again.content).user_id is None

    def test_a_lookup_does_not_settle_identity(self, user):
        result = env().get_response(tcall("get_user_information_by_id", user_id=user[0]))
        assert not result.error
        assert banking.facts("get_user_information_by_id", result.content) == banking.facts(
            "get_current_time", None
        )


class TestUnderTheHarness:
    def _session(self, toolkit, **overrides):
        config = HarnessConfig(name="t", gating_enabled=True, audit_enabled=True, **overrides)
        return HarnessSession(
            gate=ToolGate(banking.catalogue(toolkit), config),
            validator=Validator(DomainConstraints(), config),
            config=config,
            approve=auto_reject,
            session_id="s",
            clock=lambda: datetime(2026, 1, 1),
            profile=banking.BANKING_PROFILE,
        )

    def test_the_verification_flow_reaches_writes_and_only_then(self, user):
        e = env()
        s = self._session(e.tools)
        dispute = tcall(
            "call_discoverable_agent_tool", id="w",
            agent_tool_name="apply_statement_credit_8472", arguments="{}",
        )
        [refused] = s.review([banking.unwrap(dispute)])
        assert refused.decision.rule == GateRule.IDENTITY_FIRST
        assert "log_verification" in refused.decision.remedy

        # Lookup, then commit, both executed by tau2.
        lookup = tcall("get_user_information_by_id", id="l", user_id=user[0])
        assert all_allowed(s.review([banking.unwrap(lookup)]))
        r = e.get_response(lookup)
        s.absorb_result("l", error=r.error, facts=banking.facts(lookup.name, r.content))
        assert s.state is SessionState.UNVERIFIED

        uid, rec = user
        commit = tcall(
            "log_verification", id="v", name=rec["name"], user_id=uid,
            address=str(rec["address"]), email=rec["email"], phone_number=rec["phone_number"],
            date_of_birth=rec["date_of_birth"], time_verified="2025-11-14 03:40:00 EST",
        )
        assert all_allowed(s.review([banking.unwrap(commit)]))
        s.note_handoff([banking.unwrap(commit)])
        r = e.get_response(commit)
        s.absorb_result("v", error=r.error, facts=banking.facts(commit.name, r.content))
        assert s.state is SessionState.SCOPED

        assert all_allowed(s.review([banking.unwrap(dispute)]))
        [again] = s.review([banking.unwrap(commit)])
        assert again.decision.rule == GateRule.IDENTITY_ALREADY_ESTABLISHED

    def test_knowledge_and_the_clock_need_no_verification(self):
        s = self._session(env("bm25").tools)
        assert all_allowed(s.review([banking.unwrap(tcall("KB_search", query="dispute"))]))
        assert all_allowed(s.review([banking.unwrap(tcall("get_current_time", id="t"))]))
