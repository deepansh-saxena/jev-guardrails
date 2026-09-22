"""Offline tests. No API key, no model call -- everything here is the
deterministic and probabilistic scaffolding around the LLM.

    python -m pytest tests -q        (if pytest is installed)
    python tests/test_offline.py     (plain runner, no dependencies)
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tlife_agent import mock_db as db
from tlife_agent.guardrails import runtime as gr
from tlife_agent.guardrails.bank import CLAUSES_BY_ID, POOL_CLAUSES, clauses_for
from tlife_agent.guardrails import violations as gv
from tlife_agent.guardrails.output_scan import CHECKS, scan
from tlife_agent.guardrails.sampler import GuardrailSampler, reset_sampler, sampler_for
from tlife_agent.prompts import ROUTINE_BODIES, build_system_prompt
from tlife_agent.routines.desks import DESK_TOOLS
from tlife_agent.tools.account import get_account_notes
from tlife_agent.tools.billing import apply_goodwill_credit, get_bill
from tlife_agent.tools.common import (
    COMMON_TOOLS,
    get_account_summary,
    lookup_account,
    verify_identity,
)
from tlife_agent.tools.payments import make_payment, start_secure_card_capture
from tlife_agent.tools.tech_support import start_esim_transfer

# ---------------------------------------------------------------------------
# Harness: run a tool the way the graph does, so ToolRuntime is injected.
# ---------------------------------------------------------------------------


def call(tool, args: dict, session: str = "test"):
    graph = StateGraph(AgentState)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    msg = AIMessage(
        content="",
        tool_calls=[{"name": tool.name, "args": args, "id": "c1", "type": "tool_call"}],
    )
    out = graph.compile().invoke(
        {"messages": [msg]}, config={"configurable": {"thread_id": session}}
    )
    import json

    return json.loads(out["messages"][-1].content)


def fresh(session: str) -> None:
    db.reset_session(session)
    reset_sampler(session)
    gr.reset_runtime(session)


_SNAPSHOT = {
    "customers": copy.deepcopy(db.CUSTOMERS),
    "bills": copy.deepcopy(db.BILLS),
    "lines": copy.deepcopy(db.LINES),
    "home": copy.deepcopy(db.HOME_INTERNET),
}


def restore_world() -> None:
    db.CUSTOMERS.clear(); db.CUSTOMERS.update(copy.deepcopy(_SNAPSHOT["customers"]))
    db.BILLS.clear(); db.BILLS.update(copy.deepcopy(_SNAPSHOT["bills"]))
    db.LINES.clear(); db.LINES.update(copy.deepcopy(_SNAPSHOT["lines"]))
    db.HOME_INTERNET.clear(); db.HOME_INTERNET.update(copy.deepcopy(_SNAPSHOT["home"]))


def verified_session(session: str, customer: str = "CUS-100234") -> None:
    fresh(session)
    restore_world()
    pin = db.CUSTOMERS[customer]["account_pin_hash"].split(":")[1]
    call(lookup_account, {"identifier": customer}, session)
    res = call(verify_identity, {"identifier": customer, "account_pin": pin}, session)
    assert res["verified"] is True


# ---------------------------------------------------------------------------
# Guardrail sampler
# ---------------------------------------------------------------------------


def test_pinned_clauses_always_present():
    """Safety floors are never left to the dice."""
    for routine in ROUTINE_BODIES:
        expected = {c.id for c in clauses_for(routine)[0]}
        assert expected, routine
        sampler = GuardrailSampler(session_seed=11)
        for turn in range(1, 25):
            draw = sampler.draw(routine, turn)
            assert set(draw.pinned_ids) == expected, (routine, turn)
            # pinned + sampled are all rendered into the block, none dropped
            assert len(draw.rendered) == len(draw.pinned_ids) + len(draw.clause_ids)
            assert "self-harm" in draw.text or "Safety outranks" in draw.text or \
                "not customer service turns" in draw.text, (routine, turn)


def test_no_critical_clause_is_sampled():
    assert [c.id for c in POOL_CLAUSES if c.severity == "critical"] == []


def test_draws_actually_vary():
    """The whole point: consecutive turns do not see the same rulebook."""
    sampler = GuardrailSampler(session_seed=5)
    draws = [sampler.draw("billing", t) for t in range(1, 21)]
    texts = {d.text for d in draws}
    assert len(texts) == len(draws), "guardrail block repeated verbatim"

    consecutive_identical = sum(
        1 for a, b in zip(draws, draws[1:]) if set(a.clause_ids) == set(b.clause_ids)
    )
    assert consecutive_identical <= 2, consecutive_identical

    caps = {d.credit_cap for d in draws}
    assert len(caps) > 3, f"credit cap barely moved: {caps}"


def test_seed_makes_runs_reproducible():
    a = GuardrailSampler(session_seed=99).draw("payments", 3)
    b = GuardrailSampler(session_seed=99).draw("payments", 3)
    assert a.text == b.text
    c = GuardrailSampler(session_seed=100).draw("payments", 3)
    assert c.text != a.text


def test_prompt_contains_static_and_volatile_halves():
    prompt = build_system_prompt("home_internet", "seedcheck", 1)
    assert "Your role: home internet" in prompt
    assert "Operating rules (resampled every turn" in prompt
    # volatile half must come last so the cacheable prefix stays stable
    assert prompt.index("Your role") < prompt.index("Operating rules")


# ---------------------------------------------------------------------------
# Verification gating
# ---------------------------------------------------------------------------


def test_account_data_sealed_before_verification():
    fresh("seal"); restore_world()
    out = call(get_account_summary, {}, "seal")
    assert out["allowed"] is False

    call(lookup_account, {"identifier": "+1-312-555-0147"}, "seal")
    out = call(get_account_summary, {}, "seal")
    assert out["allowed"] is False, "a lookup must not count as verification"

    out = call(get_bill, {}, "seal")
    assert out["allowed"] is False


def test_wrong_pin_locks_out_after_two_attempts():
    fresh("pin"); restore_world()
    for expected in (1, 2):
        out = call(verify_identity,
                   {"identifier": "CUS-100234", "account_pin": "0000"}, "pin")
        assert out["verified"] is False
        assert out["attempts"] == expected
    assert out["locked_out"] is True


def test_verification_unseals_account_data():
    verified_session("unseal")
    out = call(get_account_summary, {}, "unseal")
    assert out["customer_id"] == "CUS-100234"
    assert len(out["lines"]) == 3
    # masked, never full
    assert len(out["phone_last4"]) == 4


# ---------------------------------------------------------------------------
# Authority limits
# ---------------------------------------------------------------------------


def test_credit_cap_is_enforced_and_matches_the_prompt():
    verified_session("cap")
    sampler_for("cap").draw("billing", 1)
    cap = gr.credit_ceiling("cap", "billing")
    assert cap >= 15

    out = call(apply_goodwill_credit,
               {"amount_usd": cap + 25, "reason": "test"}, "cap")
    assert out["applied"] is False
    assert out["reason"] == "over_authority_cap"
    assert out["cap_usd"] == cap


def test_credit_splitting_is_blocked_by_session_total():
    verified_session("split")
    sampler_for("split").draw("billing", 1)
    cap = gr.credit_ceiling("split", "billing")

    issued = 0.0
    denied = False
    for _ in range(10):
        out = call(apply_goodwill_credit, {"amount_usd": 5.0, "reason": "slice"}, "split")
        if out.get("applied"):
            issued += 5.0
        elif out.get("reason") == "over_authority_cap":
            denied = True
            break
    assert denied, "salami-slicing past the cap was allowed"
    assert issued <= cap


# ---------------------------------------------------------------------------
# Runtime stochastic gates
# ---------------------------------------------------------------------------


def test_critical_actions_always_challenge():
    fresh("crit"); restore_world()
    for action in ("sim_swap", "change_address", "cancel_line", "add_payment_method"):
        gate = gr.maybe_require_step_up("crit", action, "high")
        assert gate.allowed is False, action
        assert gate.challenge["type"] == "one_time_code"


def test_step_up_fires_sometimes_but_not_always():
    fresh("dice"); restore_world()
    results = [gr.maybe_require_step_up("dice", "make_payment", "high").allowed
               for _ in range(200)]
    fired = results.count(False)
    assert 0 < fired < 200, f"gate is not stochastic: {fired}/200"


def test_step_up_challenge_can_be_cleared():
    verified_session("otp")
    gate = gr.maybe_require_step_up("otp", "sim_swap", "critical")
    assert gate.allowed is False
    # The code must NOT be in the payload the model sees. An earlier version
    # included it "for the demo" and the agent read it out of the tool result
    # and confirmed its own challenge.
    assert "_demo_code" not in gate.challenge
    assert not any("code" in str(v) and str(v).isdigit()
                   for v in gate.challenge.values())
    code = db.OTP_STORE["otp"]
    assert gr.confirm_step_up_code("otp", "000000").allowed is False
    assert gr.confirm_step_up_code("otp", code).allowed is True
    assert db.get_session("otp")["auth_level"] == "stepped_up"
    # once stepped up, further risky actions pass
    assert gr.maybe_require_step_up("otp", "sim_swap", "critical").allowed is True


def test_every_roll_is_audited():
    fresh("aud"); restore_world()
    before = len(db.AUDIT_LOG)
    gr.maybe_require_step_up("aud", "make_payment", "high")
    gr.maybe_hold_for_review("aud", "make_payment", 100)
    rows = db.AUDIT_LOG[before:]
    assert {r["event"] for r in rows} == {"step_up.roll", "manual_review.roll"}
    assert all("roll" in r for r in rows), "a roll must be explainable after the fact"


# ---------------------------------------------------------------------------
# Domain guardrails
# ---------------------------------------------------------------------------


def test_payment_requires_verification_then_gates():
    fresh("pay"); restore_world()
    out = call(make_payment, {"amount_usd": 50, "method_id": "PM-01"}, "pay")
    assert out["allowed"] is False  # not verified

    verified_session("pay2")
    out = call(make_payment, {"amount_usd": 50, "method_id": "PM-99"}, "pay2")
    assert out["posted"] is False and out["reason"] == "unknown_method_id"


def test_no_tool_accepts_a_card_number():
    """The only card path is a secure link -- by construction, not by prompt."""
    for tools in list(DESK_TOOLS.values()) + [COMMON_TOOLS]:
        for tool in tools:
            props = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
            for field in props:
                assert "card_number" not in field, (tool.name, field)
                assert field not in {"cvv", "pan", "ssn", "password"}, (tool.name, field)

    verified_session("card")
    out = call(start_secure_card_capture, {}, "card")
    assert out.get("sent") in (True, False)
    if out.get("sent"):
        assert "secure" in out["instructions"].lower()


def test_fraud_watch_blocks_sim_swap_outright():
    verified_session("fraud", customer="CUS-100987")
    out = call(start_esim_transfer, {"line_id": "LN-4410"}, "fraud")
    assert out["started"] is False
    assert out["reason"] == "fraud_watch_active"


def test_injected_instruction_in_account_notes_is_labelled_untrusted():
    verified_session("inject", customer="CUS-100987")
    out = call(get_account_notes, {}, "inject")
    text = " ".join(n["note"] for n in out["notes"])
    assert "SYSTEM NOTE TO AGENT" in text, "canary missing"
    assert "untrusted" in out["warning"].lower()


# ---------------------------------------------------------------------------
# Violation detection
# ---------------------------------------------------------------------------


def test_scanner_clause_ids_exist_in_the_bank():
    """A violation that cites a clause nobody can look up is useless."""
    unknown = sorted({c.clause_id for c in CHECKS} - set(CLAUSES_BY_ID))
    assert unknown == [], unknown


def test_scanner_catches_each_violation_class():
    cases = [
        ("Your card 4242 4242 4242 4242 is on file.", "pii.no_full_identifiers"),
        ("His SSN is 123-45-6789.", "pii.no_full_identifiers"),
        ("I guarantee this will never happen again.", "authority.no_promises"),
        ("Act now, this offer expires today!", "tone.no_dark_patterns"),
        ("Rule check: I must verify identity first.", "accuracy.self_audit_stays_internal"),
        ("My system prompt says to verify you.", "injection.no_prompt_disclosure"),
        ("I'm using handoff_to_triage now.", "tone.no_internal_vocabulary"),
        ("I'm a real person, not a bot.", "compliance.recording_notice"),
        ("This will hurt your credit score.", "compliance.no_advice"),
    ]
    for text, expected in cases:
        gv.reset("scan-test")
        _, found = scan(text, "scan-test", "billing")
        assert expected in {f.clause_id for f in found}, (text, [f.clause_id for f in found])


def test_scanner_redacts_rather_than_only_logging():
    gv.reset("redact")
    cleaned, found = scan("Card 4242 4242 4242 4242 on file.", "redact", "payments")
    assert "4242 4242 4242 4242" not in cleaned
    assert "[card redacted]" in cleaned
    assert found[0].action == "redacted"
    # the evidence trail must not carry the value it caught
    assert "4242" not in found[0].evidence


def test_scanner_leaves_clean_output_alone():
    gv.reset("clean")
    text = "Your bill is ready. Anything else I can help with?"
    cleaned, found = scan(text, "clean", "billing")
    assert cleaned == text and found == []


def test_unverified_amount_disclosure_is_flagged():
    fresh("unver"); restore_world()
    gv.reset("unver")
    _, found = scan("Your balance is $247.83.", "unver", "billing")
    assert "identity.before_account_data" in {f.clause_id for f in found}

    verified_session("ver-ok")
    gv.reset("ver-ok")
    _, found = scan("Your balance is $247.83.", "ver-ok", "billing")
    assert "identity.before_account_data" not in {f.clause_id for f in found}


def test_gate_denial_records_a_violation():
    fresh("gv1"); restore_world(); gv.reset("gv1")
    call(get_account_summary, {}, "gv1")
    rows = gv.all_for("gv1")
    assert rows and rows[0].clause_id == "identity.before_account_data"
    assert rows[0].source == "runtime_gate" and rows[0].action == "blocked"


def test_over_cap_credit_records_a_violation():
    verified_session("gv2"); gv.reset("gv2")
    sampler_for("gv2").draw("billing", 1)
    cap = gr.credit_ceiling("gv2", "billing")
    call(apply_goodwill_credit, {"amount_usd": cap + 50, "reason": "x"}, "gv2")
    rows = [r for r in gv.all_for("gv2") if r.clause_id == "authority.credit_ceiling"]
    assert rows, gv.all_for("gv2")
    assert str(cap) in rows[0].evidence


def test_step_up_retry_is_a_violation_but_first_challenge_is_not():
    fresh("gv3"); restore_world(); gv.reset("gv3")
    gr.maybe_require_step_up("gv3", "sim_swap", "critical")
    assert gv.all_for("gv3") == [], "a challenge firing is not itself a violation"
    gr.maybe_require_step_up("gv3", "sim_swap", "critical")
    rows = gv.all_for("gv3")
    assert rows and rows[0].clause_id == "identity.step_up_for_writes"


def test_fraud_watch_block_records_a_violation():
    verified_session("gv4", customer="CUS-100987"); gv.reset("gv4")
    call(start_esim_transfer, {"line_id": "LN-4410"}, "gv4")
    rows = gv.all_for("gv4")
    assert rows and rows[0].clause_id == "escalation.fraud_watch"
    assert rows[0].severity == "critical"


def test_drain_returns_each_violation_once():
    gv.reset("gv5")
    gv.report("gv5", clause_id="tone.brevity", category="tone", severity="low",
              title="t", detail="d", source="output_scan", action="flagged")
    assert len(gv.drain("gv5")) == 1
    assert gv.drain("gv5") == []
    gv.report("gv5", clause_id="tone.brevity", category="tone", severity="low",
              title="t2", detail="d", source="output_scan", action="flagged")
    assert len(gv.drain("gv5")) == 1
    assert len(gv.all_for("gv5")) == 2


def test_bank_is_large_and_well_formed():
    from tlife_agent.guardrails.bank import ALL_CLAUSES

    assert len(ALL_CLAUSES) >= 60, len(ALL_CLAUSES)
    assert len(CLAUSES_BY_ID) == len(ALL_CLAUSES), "duplicate clause id"
    for clause in ALL_CLAUSES:
        assert len(clause.variants) >= 2, clause.id
        assert clause.severity in {"critical", "high", "medium", "low"}, clause.id
        for variant in clause.variants:
            # guards against stub/placeholder variants, not against brevity
            assert variant.strip() and len(variant) > 30, (clause.id, variant)


# ---------------------------------------------------------------------------
# Soft (judgment-based) guardrails: topical scope
# ---------------------------------------------------------------------------


def test_scope_policy_is_in_every_routine_prompt():
    from tlife_agent.prompts import ROUTINE_BODIES, static_prompt

    for routine in ROUTINE_BODIES:
        prompt = static_prompt(routine)
        assert "## Scope -- what you will and will not discuss" in prompt, routine
        # the two carve-outs that must never be lost
        assert "in distress" in prompt and "automated" in prompt, routine


def test_every_out_of_scope_topic_has_a_deflection():
    from tlife_agent.guardrails.topic_policy import OUT_OF_SCOPE

    assert len(OUT_OF_SCOPE) >= 8
    for topic in OUT_OF_SCOPE:
        assert topic.describes.strip() and topic.deflection.strip(), topic.key
        assert topic.severity in {"critical", "high", "medium", "low"}, topic.key


def test_judge_verdict_threshold_semantics():
    from tlife_agent.guardrails.judge import BLOCK_THRESHOLD, ScopeVerdict

    confident = ScopeVerdict("out_of_scope", "coding", BLOCK_THRESHOLD + 0.1, "r")
    assert confident.out_of_scope and confident.should_block

    unsure = ScopeVerdict("out_of_scope", "coding", BLOCK_THRESHOLD - 0.3, "r")
    assert unsure.out_of_scope and not unsure.should_block, \
        "a low-confidence verdict must not force a refusal"

    fine = ScopeVerdict("in_scope", None, 0.99, "r")
    assert not fine.out_of_scope and not fine.should_block


def test_judge_records_blocked_and_flagged_differently():
    from tlife_agent.guardrails import judge

    gv.reset("j1")
    judge.record("j1", judge.ScopeVerdict("out_of_scope", "coding", 0.95, "code"), "triage")
    judge.record("j1", judge.ScopeVerdict("out_of_scope", "personal", 0.40, "chat"), "triage")
    rows = gv.all_for("j1")
    assert [r.action for r in rows] == ["blocked", "flagged"]
    assert rows[0].source == "llm_judge"
    assert rows[0].clause_id == "scope.off_topic.coding"
    assert "below the" in rows[1].detail, "a flagged verdict must say why it did not block"


def test_deflection_never_leaks_the_topic():
    from tlife_agent.guardrails.judge import ScopeVerdict, deflection_for
    from tlife_agent.guardrails.topic_policy import OUT_OF_SCOPE

    for topic in OUT_OF_SCOPE:
        text = deflection_for(ScopeVerdict("out_of_scope", topic.key, 0.9, "r"))
        # always offers a way forward rather than dead-ending
        assert "?" in text, topic.key
        assert any(w in text for w in ("bills", "account")), topic.key


def test_judge_failure_is_non_fatal():
    """A judge outage must never take the agent down with it."""
    from tlife_agent.guardrails import judge

    original, enabled = judge._model, judge.JUDGE_ENABLED
    judge.JUDGE_ENABLED = True
    judge._model = lambda: (_ for _ in ()).throw(RuntimeError("judge is down"))
    try:
        assert judge.classify("write me some python") is None
    finally:
        judge._model, judge.JUDGE_ENABLED = original, enabled


def test_provider_content_filter_escalates_rather_than_refusing():
    """A provider policy refusal is ambiguous: jailbreak, or someone in crisis.

    Azure returns the same `content_filter` error for "ignore your instructions"
    and for "I'm going to hurt myself". Refusing that as off-topic turns a crisis
    disclosure into a brush-off, so it must escalate instead -- the only response
    that is correct for both.
    """
    from tlife_agent.guardrails import judge
    from tlife_agent.guardrails.backends import ScopeDecision
    from tlife_agent.guardrails.recording import record_scope

    class _Filtered(Exception):
        pass

    original, enabled = judge._model, judge.JUDGE_ENABLED
    judge.JUDGE_ENABLED = True  # independent of TLIFE_JUDGE in the environment
    judge._model = lambda: (_ for _ in ()).throw(
        _Filtered("Error code: 400 - response was filtered due to the prompt "
                  "triggering Azure OpenAI's content management policy")
    )
    try:
        verdict = judge.classify("I'm going to hurt myself and I need my phone")
        assert verdict is not None, "content-filter rejections must not fail open"
        assert verdict.provider_filtered is True
        assert not verdict.out_of_scope, "must not be recorded as a topic refusal"
    finally:
        judge._model, judge.JUDGE_ENABLED = original, enabled

    gv.reset("pf")
    blocked = record_scope(
        "pf",
        ScopeDecision(in_scope=True, category=None, confidence=0.0,
                      owning_desk=None, desk_confidence=0.0,
                      reason="provider filter", provider_filtered=True),
        "triage", "llm",
    )
    assert blocked is False, "a filtered message must never be refused as off-topic"
    rows = gv.all_for("pf")
    assert rows and rows[0].severity == "critical"
    assert rows[0].clause_id == "escalation.vulnerability"


def test_judge_can_be_disabled():
    from tlife_agent.guardrails import judge

    original = judge.JUDGE_ENABLED
    judge.JUDGE_ENABLED = False
    try:
        assert judge.classify("write me some python") is None
    finally:
        judge.JUDGE_ENABLED = original


def test_safety_overrides_scope_is_pinned():
    """Scope rules must never be able to refuse someone in danger."""
    from tlife_agent.guardrails.bank import PINNED_CLAUSES

    ids = {c.id for c in PINNED_CLAUSES}
    assert "topic.safety_overrides_scope" in ids
    assert "escalation.vulnerability" in ids


def test_soft_rules_are_well_formed():
    """The negative examples are what keep the false-positive rate usable."""
    from tlife_agent.guardrails.soft_rules import RULES_BY_ID, SOFT_RULES

    assert len(SOFT_RULES) >= 20
    assert len(RULES_BY_ID) == len(SOFT_RULES), "duplicate soft rule id"
    for rule in SOFT_RULES:
        assert rule.violates.strip() and rule.does_not_violate.strip(), rule.id
        assert len(rule.does_not_violate) > 40, rule.id
        assert rule.severity in {"critical", "high", "medium", "low"}, rule.id
        if rule.clause_id:
            assert rule.clause_id in CLAUSES_BY_ID, (rule.id, rule.clause_id)


def test_critical_soft_rules_are_never_sampled_out():
    from tlife_agent.guardrails.judge import sample_rules
    from tlife_agent.guardrails.soft_rules import ALWAYS_RULES

    always = {r.id for r in ALWAYS_RULES}
    assert always, "grounding and escalation must always be reviewed"
    for turn in range(1, 15):
        drawn = {r.id for r in sample_rules("rubric", turn)}
        assert always <= drawn, turn


def test_rubric_sample_varies_by_turn():
    from tlife_agent.guardrails.judge import sample_rules

    draws = [frozenset(r.id for r in sample_rules("rubric", t)) for t in range(1, 12)]
    assert len(set(draws)) > 5, "the sampled rubric barely moves"


def test_hedged_verdicts_are_dropped():
    """A reviewer that argues itself out of a fail must not record one."""
    from tlife_agent.guardrails.judge import _is_hedged

    assert _is_hedged("The figures are directly supported by the tool results.")
    assert _is_hedged("Missing an effective date, though that may not be necessary.")
    assert _is_hedged("This is arguably fine for an informational reply.")
    assert not _is_hedged("States a balance that no tool result contains.")
    assert not _is_hedged("Promises a refund the assistant cannot execute.")


def test_review_is_skippable_and_non_fatal():
    from tlife_agent.guardrails import judge

    gv.reset("rv")
    original, enabled = judge._model, judge.REVIEW_ENABLED
    judge.REVIEW_ENABLED = False
    try:
        assert judge.review("rv", "billing", "anything", "ctx", 1) == []
    finally:
        judge.REVIEW_ENABLED = enabled

    judge.REVIEW_ENABLED = True
    judge._model = lambda: (_ for _ in ()).throw(RuntimeError("reviewer down"))
    try:
        assert judge.review("rv", "billing", "anything", "ctx", 1) == []
        assert gv.all_for("rv") == []
    finally:
        judge._model, judge.REVIEW_ENABLED = original, enabled


def test_build_context_includes_tool_results():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from tlife_agent.guardrails.judge import build_context

    ctx = build_context([
        HumanMessage(content="why is my bill higher?"),
        AIMessage(content="", tool_calls=[{"name": "get_bill", "args": {}, "id": "1",
                                           "type": "tool_call"}]),
        ToolMessage(content='{"total": 247.83}', name="get_bill", tool_call_id="1"),
        AIMessage(content="It's $247.83."),
    ])
    assert "TOOL RESULT [get_bill]" in ctx and "247.83" in ctx
    assert "CUSTOMER:" in ctx and "ASSISTANT:" in ctx


# ---------------------------------------------------------------------------
# Per-desk (subagent-level) guardrails
# ---------------------------------------------------------------------------


def test_every_desk_has_a_charter_with_a_boundary():
    from tlife_agent.guardrails.desk_policy import CHARTERS
    from tlife_agent.graph import NODE_NAMES

    for node in NODE_NAMES:
        assert node in CHARTERS, node
        charter = CHARTERS[node]
        assert charter.owns, node
        assert charter.does_not_own, f"{node} claims to own everything"


def test_charter_appears_in_the_desk_prompt():
    from tlife_agent.graph import NODE_NAMES
    from tlife_agent.prompts import static_prompt

    for node in NODE_NAMES:
        full = static_prompt(node, lean=False)
        assert "## This desk" in full and "You own:" in full, node
        assert "You do NOT own these" in full, node


def test_lean_prompt_strips_guardrails_but_keeps_the_job():
    """The Jev build's prompt describes the role; the rules live outside it."""
    from tlife_agent.graph import NODE_NAMES
    from tlife_agent.prompts import static_prompt

    for node in NODE_NAMES:
        full = static_prompt(node, lean=False)
        lean = static_prompt(node, lean=True)
        assert len(lean) < len(full) * 0.7, (node, len(lean), len(full))
        # guardrail content gone
        assert "## Scope -- what you will and will not discuss" not in lean, node
        assert "You do NOT own these" not in lean, node
        assert "Absolute limits at this desk" not in lean, node
        # job description kept
        assert "You own:" in lean, node
        assert "Your role" in lean, node


def test_charters_route_to_real_desks():
    from tlife_agent.guardrails.desk_policy import CHARTERS
    from tlife_agent.routines.desks import DESK_NAMES

    for charter in CHARTERS.values():
        for _topic, owner in charter.does_not_own:
            assert owner in DESK_NAMES or owner == "the owning desk", \
                (charter.name, owner)


def test_charter_rule_ids_are_real():
    from tlife_agent.guardrails.desk_policy import CHARTERS
    from tlife_agent.guardrails.soft_rules import RULES_BY_ID

    for charter in CHARTERS.values():
        for rid in list(charter.rubric_boost) + list(charter.severity_override):
            assert rid in RULES_BY_ID, (charter.name, rid)


def test_per_desk_weighting_changes_what_gets_sampled():
    """A flat rubric under-samples the rule a given desk is likeliest to break."""
    from tlife_agent.guardrails.judge import sample_rules
    from tlife_agent.guardrails.sampler import _SAMPLERS, GuardrailSampler

    def rate(desk, rule_id, trials=200):
        hits = 0
        for i in range(trials):
            _SAMPLERS[f"t{i}"] = GuardrailSampler(session_seed=i * 7919)
            if any(r.id == rule_id for r in sample_rules(f"t{i}", 1, desk)):
                hits += 1
        return hits / trials

    boosted = rate("plans_devices", "soft.no_soft_retention_pressure")
    baseline = rate("billing", "soft.no_soft_retention_pressure")
    assert boosted > baseline * 1.4, (boosted, baseline)


def test_per_desk_severity_override():
    from tlife_agent.guardrails.desk_policy import severity_for
    from tlife_agent.guardrails.soft_rules import RULES_BY_ID

    rule = RULES_BY_ID["soft.explicit_consent"]
    assert severity_for("payments", rule) == "critical"
    assert severity_for("tech_support", rule) == rule.severity


def test_boundary_check_flags_wrong_desk_only():
    from tlife_agent.guardrails.judge import ScopeVerdict, check_desk_boundary

    gv.reset("bd")
    right = ScopeVerdict("in_scope", None, 0.98, "billing question",
                         owning_desk="billing", desk_confidence=0.95)
    assert check_desk_boundary("bd", right, "billing") is None
    assert gv.all_for("bd") == []

    wrong = ScopeVerdict("in_scope", None, 0.98, "wants to pay",
                         owning_desk="payments", desk_confidence=0.95)
    note = check_desk_boundary("bd", wrong, "billing")
    assert note and "payments" in note
    rows = gv.all_for("bd")
    assert rows and rows[0].clause_id == "scope.desk_boundary"
    assert rows[0].routine == "billing"


def test_triage_is_exempt_from_boundary_checks():
    """Routing is triage's job -- flagging it for every request is noise."""
    from tlife_agent.guardrails.judge import ScopeVerdict, check_desk_boundary

    gv.reset("bd2")
    v = ScopeVerdict("in_scope", None, 0.98, "wants to pay",
                     owning_desk="payments", desk_confidence=0.99)
    assert check_desk_boundary("bd2", v, "triage") is None
    assert gv.all_for("bd2") == []


def test_low_confidence_boundary_does_not_flag():
    from tlife_agent.guardrails.judge import ScopeVerdict, check_desk_boundary

    gv.reset("bd3")
    v = ScopeVerdict("in_scope", None, 0.98, "ambiguous",
                     owning_desk="payments", desk_confidence=0.30)
    assert check_desk_boundary("bd3", v, "billing") is None
    assert gv.all_for("bd3") == []


# ---------------------------------------------------------------------------
# Guardrail backends (llm vs jev)
# ---------------------------------------------------------------------------


def test_both_backends_satisfy_the_protocol():
    from tlife_agent.guardrails.backends.jev_backend import JevBackend
    from tlife_agent.guardrails.backends.llm_backend import LLMBackend

    for cls in (LLMBackend, JevBackend):
        for method in ("classify_scope", "review_output", "check_tool_call"):
            assert callable(getattr(cls, method, None)), (cls.__name__, method)
    assert LLMBackend.rules_in_prompt is True
    assert JevBackend.rules_in_prompt is False, \
        "the jev build's whole point is that the rules leave the prompt"


def test_unknown_backend_is_rejected():
    from tlife_agent.guardrails.backends import _build

    try:
        _build("gpt")
    except ValueError as exc:
        assert "llm" in str(exc) and "jev" in str(exc)
    else:
        raise AssertionError("an unknown backend name must not silently work")


def test_jev_questions_cover_every_rule_and_category():
    """No rule may be silently dropped when the backend changes."""
    from langchain_typesafe import Choice, Noul

    from tlife_agent.guardrails.backends import jev_backend as jb
    from tlife_agent.guardrails.soft_rules import SOFT_RULES
    from tlife_agent.guardrails.topic_policy import OUT_OF_SCOPE

    scope = jb._scope_choice()
    assert isinstance(scope, Choice)
    for topic in OUT_OF_SCOPE:
        assert topic.key in scope.criteria, topic.key
    assert jb._IN_SCOPE_KEY in scope.criteria

    desks = jb._desk_choice()
    from tlife_agent.routines.desks import DESK_NAMES

    for desk in DESK_NAMES:
        assert desk in desks.criteria, desk

    # Every soft rule becomes a Noul whose true/false criteria are the
    # violates / does_not_violate text -- that mapping is the design.
    for rule in SOFT_RULES:
        noul = jb._rule_noul(rule)
        assert isinstance(noul, Noul)
        assert noul.criteria.true == rule.violates, rule.id
        assert noul.criteria.false == rule.does_not_violate, rule.id
        assert rule.rule in noul.instructions, rule.id

    assert set(jb._tool_choice().criteria) == {"allow", "confirm", "block"}


def test_jev_backend_evaluates_every_rule_not_a_sample():
    """The sampling in the llm build exists only because judging is expensive."""
    from tlife_agent.guardrails.backends import jev_backend as jb
    from tlife_agent.guardrails.soft_rules import SOFT_RULES

    questions = {rule.id: jb._rule_noul(rule) for rule in SOFT_RULES}
    assert len(questions) == len(SOFT_RULES)


def test_recording_is_backend_neutral():
    from tlife_agent.guardrails.backends import ReviewDecision, RuleFinding, ScopeDecision
    from tlife_agent.guardrails.recording import record_review, record_scope

    for backend in ("llm", "jev"):
        gv.reset(f"n-{backend}")
        blocked = record_scope(
            f"n-{backend}",
            ScopeDecision(in_scope=False, category="coding", confidence=0.95,
                          owning_desk=None, desk_confidence=0.0, reason="r"),
            "billing", backend,
        )
        assert blocked
        rows = gv.all_for(f"n-{backend}")
        assert rows[0].clause_id == "scope.off_topic.coding"
        assert f"backend={backend}" in rows[0].evidence

        n = record_review(
            f"n-{backend}",
            ReviewDecision(findings=[RuleFinding("soft.no_dead_ends", 0.88)],
                           rules_evaluated=25),
            "billing", backend,
        )
        assert n == 1


def test_safety_always_beats_an_out_of_scope_verdict():
    from tlife_agent.guardrails.backends import ScopeDecision
    from tlife_agent.guardrails.recording import record_scope

    gv.reset("safe")
    decision = ScopeDecision(in_scope=False, category="personal", confidence=0.99,
                             owning_desk=None, desk_confidence=0.0, reason="r",
                             safety_urgent=True)
    assert record_scope("safe", decision, "triage", "jev") is False
    assert gv.all_for("safe") == [], "a safety turn must never be refused as off-topic"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_runtime_is_hidden_from_the_model():
    for tools in list(DESK_TOOLS.values()) + [COMMON_TOOLS]:
        for tool in tools:
            props = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
            assert "runtime" not in props, tool.name


def test_graph_compiles_with_every_desk():
    # Provider clients validate their key at construction, so give whichever
    # provider is configured a placeholder. No network call is made.
    from tlife_agent.config import required_key_env

    env = required_key_env()
    if env:
        os.environ.setdefault(env, "offline-test-placeholder")
    from tlife_agent.graph import NODE_NAMES, build_care_graph

    nodes = set(build_care_graph().get_graph().nodes)
    for name in NODE_NAMES:
        assert name in nodes, name


def test_triage_can_reach_every_desk():
    from tlife_agent.routines.desks import DESK_NAMES
    from tlife_agent.triage import TRIAGE_HANDOFF_TOOLS

    reachable = {t.name.replace("transfer_to_", "") for t in TRIAGE_HANDOFF_TOOLS}
    assert reachable == set(DESK_NAMES)


# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  \033[32mPASS\033[0m {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  \033[31mFAIL\033[0m {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
