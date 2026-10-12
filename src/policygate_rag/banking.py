"""banking_knowledge, as the policygate harness needs to see it.

policygate ships the retail binding; this is the banking one, written here
because the domain knowledge belongs to this study. Every rule below is grounded
in the domain's policy text at the pinned tau2 commit
(``prompts/components/policy_header.md`` and ``additional_instructions.md``),
quoted where it decides something.

Three jobs:

* the tool catalogue, with the roles the gate needs;
* unwrapping ``call_discoverable_agent_tool`` so the harness judges the real
  operation rather than the dispatcher;
* reading tool results into facts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from policygate.adapters.tool_mapping import to_harness_call, tool_specs
from policygate.harness.session import ExtractedFacts
from policygate.harness.state import DomainStateProfile
from policygate.harness.types import ToolCall, ToolKind, ToolSpec
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.environment.toolkit import ToolKitBase

BANKING_PROFILE = DomainStateProfile(
    name="banking_knowledge",
    # "for any scenario involving accessing customer information in internal
    # databases, you must first verify their identify before proceeding."
    requires_identity_lookup=True,
    # The policy states no one-customer-per-conversation rule, so none is imposed.
    single_subject_per_session=False,
)

IDENTITY_LOOKUPS = (
    "get_user_information_by_email",
    "get_user_information_by_id",
    "get_user_information_by_name",
)
"""Find a candidate. "call the appropriate read tools, and ensure that they are able
to give correctly any 2 out of [DOB, email, phone, address]" -- the lookup is
how the agent checks, so it must be callable before verification. It does not
settle identity: "Knowing full name or userID is not enough to verify"."""

IDENTITY_COMMITS = ("log_verification",)
"""Settles identity. "After verification, you must call the verification logging
tool". Typed WRITE by tau2 because it writes the verification record."""

ESCALATION = ("transfer_to_human_agents",)

SUBJECT_INDEPENDENT = (
    "get_current_time",
    "KB_search",
    "unlock_discoverable_agent_tool",
    "give_discoverable_user_tool",
    "list_discoverable_agent_tools",
)
"""Touch no customer's data, so they are legal before verification ("You should
ONLY verify a user's identity if you need to access or modify their customer
information"). KB_search exists only in the bm25 arm; the discovery tools act on
the knowledge base's tool registry, not on a customer. Names absent from a
variant's toolkit are skipped."""

DISPATCHER = "call_discoverable_agent_tool"

_VERIFIED_ID = re.compile(r"Verification logged successfully\.\s*\n\s*- User: .*\(ID: ([^)]+)\)")


def catalogue(toolkit: ToolKitBase) -> tuple[ToolSpec, ...]:
    """The gate's catalogue: the visible toolkit plus every agent discoverable tool.

    tau2's ``get_tool_types`` lists only the non-discoverable tools, but once a
    dispatcher call is unwrapped the harness sees discoverable names, and an
    unknown name would be refused as a hallucination. Kinds come from tau2's own
    decorator; discoverable tools have no role, so writes among them need a
    verified customer like any other write.
    """
    present = set(toolkit.get_tools())
    base = tool_specs(
        toolkit,
        identity_lookups=[n for n in IDENTITY_LOOKUPS if n in present],
        escalation=[n for n in ESCALATION if n in present],
        identity_commits=[n for n in IDENTITY_COMMITS if n in present],
        subject_independent=[n for n in SUBJECT_INDEPENDENT if n in present],
    )
    known = {s.name for s in base}
    discoverable = tuple(
        ToolSpec(name=name, kind=ToolKind(toolkit.tool_type(name).value))
        for name in sorted(toolkit.get_discoverable_tools())
        if name not in known
    )
    return base + discoverable


def unwrap(call: Tau2ToolCall) -> ToolCall:
    """The operation a proposed call performs, in the harness's type.

    ``call_discoverable_agent_tool(agent_tool_name, arguments)`` becomes a call
    to ``agent_tool_name`` with the parsed arguments, keeping the call id so the
    result still finds it. If the arguments are not a JSON object the call is
    left as the dispatcher: the environment will reject it, and the harness
    should not invent an interpretation.
    """
    harness_call = to_harness_call(call)
    if harness_call.name != DISPATCHER:
        return harness_call
    name = harness_call.arguments.get("agent_tool_name")
    raw: Any = harness_call.arguments.get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return harness_call
    if not isinstance(name, str) or not name or not isinstance(parsed, dict):
        return harness_call
    return ToolCall(id=harness_call.id, name=name, arguments=parsed)


def facts(tool_name: str, content: str | None, *, error: bool = False) -> ExtractedFacts:
    """Read a banking tool result into facts.

    Only a successful verification log establishes identity; lookups never do.
    Scope: banking's policy has no per-object rule like retail's single order in
    scope, so the verified customer is the write target. With gating on, that
    reduces the write rule to "writes only after verification", which is what
    the policy states, and adds nothing it does not.
    """
    if error or not content or tool_name not in IDENTITY_COMMITS:
        return ExtractedFacts()
    match = _VERIFIED_ID.search(content)
    if match is None:
        return ExtractedFacts()  # e.g. "Failed to log verification: Record may already exist."
    user_id = match.group(1).strip()
    return ExtractedFacts(user_id=user_id, scope_targets=(user_id,))
