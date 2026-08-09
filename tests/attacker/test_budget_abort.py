"""05-04-PLAN.md Task 1: ledger accounting, the between-round abort edge, and
the within-role step cap -- both independent halves of D-73 mitigation 1.

Everything here runs fully offline: the between-round test builds a small
standalone `StateGraph` directly reusing `budget_check_node`/`over_budget_edge`
(reproducing 05-RESEARCH.md's own executed Wave 0 spike numbers -- cap 0.75,
per-round cost 0.40, round cap 10, terminates after exactly 2 rounds at
spent 0.80); the within-role test drives a REAL `create_deep_agent()` graph
with `StepCapMiddleware` attached against a scripted model that would
otherwise call its bound tool forever.
"""

from __future__ import annotations

import pytest
from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from llmsec.attacker.budget import (
    ATTACKER_MODEL_PRICES_PER_MILLION_TOKENS,
    DEFAULT_MAX_MODEL_CALLS_PER_ROLE,
    StepCapMiddleware,
    attacker_call_cost,
    budget_check_node,
    build_step_cap_middleware,
    over_budget_edge,
    record_agent_spend,
    record_target_spend,
    remaining_calls,
    remaining_usd,
    role_over_share,
    target_call_cost,
)
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.state import BudgetLedger, CampaignState, RoleSpend, new_campaign_state

from .conftest import ScriptedToolCallChatModel


def _fresh_ledger(
    *, cap_usd: float = 10.0, warn_usd: float = 9.0, agent_call_ceiling: int = 1000
) -> BudgetLedger:
    """A bare, hand-constructed `BudgetLedger` -- bypasses `AttackerConfig`'s
    own validators so tests can freely exercise edge values (e.g. a
    `warn_usd` deliberately set above any spend this test will ever reach)."""
    return BudgetLedger(
        cap_usd=cap_usd,
        warn_usd=warn_usd,
        spent_usd=0.0,
        attacker_spent_usd=0.0,
        target_spent_usd=0.0,
        agent_calls=0,
        agent_call_ceiling=agent_call_ceiling,
        per_role={},
        truncated=False,
        overshoot_rounds=0,
        warn_approved=False,
        unpriced_calls=0,
    )


# --- Ledger operations -----------------------------------------------------


def test_record_agent_spend_increases_attacker_and_total_never_agent_calls_of_target():
    ledger = _fresh_ledger()
    record_agent_spend(ledger, role="mutator", usd=0.4, calls=1)
    assert ledger["attacker_spent_usd"] == pytest.approx(0.4)
    assert ledger["target_spent_usd"] == 0.0
    assert ledger["spent_usd"] == pytest.approx(0.4)
    assert ledger["agent_calls"] == 1
    assert ledger["per_role"]["mutator"]["calls"] == 1
    assert ledger["per_role"]["mutator"]["usd"] == pytest.approx(0.4)


def test_record_agent_spend_spent_usd_always_equals_both_sides_summed():
    ledger = _fresh_ledger()
    record_agent_spend(ledger, role="strategist", usd=0.1)
    record_target_spend(ledger, usd=0.05)
    record_agent_spend(ledger, role="mutator", usd=0.2)
    assert ledger["spent_usd"] == pytest.approx(ledger["attacker_spent_usd"] + ledger["target_spent_usd"])
    assert ledger["spent_usd"] == pytest.approx(0.35)


def test_record_target_spend_increases_total_and_target_only_never_agent_calls():
    ledger = _fresh_ledger()
    record_target_spend(ledger, usd=0.12)
    assert ledger["target_spent_usd"] == pytest.approx(0.12)
    assert ledger["attacker_spent_usd"] == 0.0
    assert ledger["spent_usd"] == pytest.approx(0.12)
    assert ledger["agent_calls"] == 0


def test_role_over_share_true_once_role_spend_exceeds_its_ceiling():
    ledger = _fresh_ledger()
    ledger["per_role"]["strategist"] = RoleSpend(calls=1, usd=0.6, share_ceiling_usd=0.5)
    assert role_over_share(ledger, "strategist") is True


def test_role_over_share_false_when_role_has_no_configured_share():
    ledger = _fresh_ledger()
    ledger["per_role"]["mutator"] = RoleSpend(calls=1, usd=100.0, share_ceiling_usd=None)
    assert role_over_share(ledger, "mutator") is False


def test_role_over_share_false_when_role_absent_from_ledger():
    ledger = _fresh_ledger()
    assert role_over_share(ledger, "recon") is False


def test_remaining_usd_and_remaining_calls_floor_at_zero():
    ledger = _fresh_ledger(cap_usd=1.0, agent_call_ceiling=5)
    record_agent_spend(ledger, role="mutator", usd=5.0, calls=10)
    assert remaining_usd(ledger) == 0.0
    assert remaining_calls(ledger) == 0


def test_new_campaign_state_threads_role_shares_into_per_role_ceilings():
    """D-81: `AttackerConfig.roles[*].budget_share` (05-02) reaches the
    ledger's `per_role[role]["share_ceiling_usd"]` via
    `new_campaign_state(role_shares=...)` -- `role_over_share()` can only
    ever fire once a real ceiling is populated here."""
    cfg = AttackerConfig(enabled=True, profile="light", budget_usd=1.0)
    settings = resolve_settings(cfg)
    state = new_campaign_state(
        "scan-1", settings, ["stub_module"], [], role_shares={"strategist": 0.25}
    )
    ledger = state["budget_ledger"]
    assert ledger["per_role"]["strategist"]["share_ceiling_usd"] == pytest.approx(0.25)
    assert ledger["per_role"]["mutator"]["share_ceiling_usd"] is None


def test_remaining_usd_and_remaining_calls_positive_before_exhaustion():
    ledger = _fresh_ledger(cap_usd=1.0, agent_call_ceiling=5)
    record_agent_spend(ledger, role="mutator", usd=0.25, calls=1)
    assert remaining_usd(ledger) == pytest.approx(0.75)
    assert remaining_calls(ledger) == 4


# --- Pricing -----------------------------------------------------------


def test_attacker_call_cost_positive_for_a_priced_model():
    assert "openai:gpt-4o-mini" in ATTACKER_MODEL_PRICES_PER_MILLION_TOKENS
    cost = attacker_call_cost(
        {"input_tokens": 1000, "output_tokens": 500}, "openai:gpt-4o-mini"
    )
    assert cost > 0.0


def test_attacker_call_cost_zero_and_ledger_unpriced_calls_incremented_for_unknown_model():
    ledger = _fresh_ledger()
    cost = attacker_call_cost(
        {"input_tokens": 1000, "output_tokens": 500}, "not-a-real-model", ledger=ledger
    )
    assert cost == 0.0
    assert ledger["unpriced_calls"] == 1


def test_attacker_call_cost_zero_never_guessed_for_missing_usage_metadata():
    assert attacker_call_cost(None, "openai:gpt-4o-mini") == 0.0
    assert attacker_call_cost({}, "openai:gpt-4o-mini") == 0.0


def test_target_call_cost_zero_and_ledger_unpriced_calls_incremented_for_unpriced_model():
    ledger = _fresh_ledger()
    cost = target_call_cost("not-a-real-model", 500, ledger=ledger)
    assert cost == 0.0
    assert ledger["unpriced_calls"] == 1


def test_target_call_cost_zero_for_missing_model_or_tokens():
    assert target_call_cost(None, 500) == 0.0
    assert target_call_cost("openai/gpt-4o-mini", None) == 0.0
    assert target_call_cost("openai/gpt-4o-mini", 0) == 0.0


def test_target_call_cost_positive_for_a_priced_target_model():
    cost = target_call_cost("openai/gpt-4o-mini", 1000)
    assert cost > 0.0


# --- budget_check_node / over_budget_edge -----------------------------


def test_budget_check_node_over_budget_true_once_spent_reaches_cap():
    ledger = _fresh_ledger(cap_usd=0.75)
    record_agent_spend(ledger, role="mutator", usd=0.75)
    state = CampaignState(budget_ledger=ledger)
    result = budget_check_node(state)
    assert result["over_budget"] is True


def test_budget_check_node_over_budget_true_on_call_ceiling_alone_with_zero_spend():
    """The unpriced-model case (D-80): the dollar side reads 0.0, but the
    independent call ceiling alone still trips `over_budget`."""
    ledger = _fresh_ledger(cap_usd=100.0, agent_call_ceiling=3)
    ledger["agent_calls"] = 3
    state = CampaignState(budget_ledger=ledger)
    result = budget_check_node(state)
    assert result["over_budget"] is True
    assert ledger["spent_usd"] == 0.0


def test_budget_check_node_performs_no_mutation():
    ledger = _fresh_ledger(cap_usd=0.75)
    state = CampaignState(budget_ledger=ledger)
    before = dict(ledger)
    budget_check_node(state)
    assert ledger == before


def test_budget_check_node_warn_pending_true_once_warn_crossed_and_not_over_budget():
    ledger = _fresh_ledger(cap_usd=10.0, warn_usd=0.5)
    record_agent_spend(ledger, role="mutator", usd=0.6)
    state = CampaignState(budget_ledger=ledger)
    result = budget_check_node(state)
    assert result["over_budget"] is False
    assert result["warn_pending"] is True


def test_budget_check_node_warn_pending_false_once_already_approved():
    ledger = _fresh_ledger(cap_usd=10.0, warn_usd=0.5)
    record_agent_spend(ledger, role="mutator", usd=0.6)
    ledger["warn_approved"] = True
    state = CampaignState(budget_ledger=ledger)
    result = budget_check_node(state)
    assert result["warn_pending"] is False


def test_over_budget_edge_three_outcomes():
    assert over_budget_edge(CampaignState(over_budget=True, warn_pending=False)) == "finalize"
    assert (
        over_budget_edge(CampaignState(over_budget=False, warn_pending=True)) == "budget_approval"
    )
    assert over_budget_edge(CampaignState(over_budget=False, warn_pending=False)) == "continue"


# --- Between-round abort: reproduces the Wave 0 spike's own numbers -----


async def test_between_round_conditional_edge_terminates_after_two_rounds_at_cap():
    """05-RESEARCH `## Wave 0 Spike Results` Mitigation 1 (between-round
    half): cap 0.75, per-round cost 0.40, round cap 10 (which would spend
    4.00 uncapped) -- terminates after exactly 2 rounds at spent 0.80."""
    ledger = _fresh_ledger(cap_usd=0.75, warn_usd=999.0)

    async def round_node(state: CampaignState) -> dict:
        new_ledger = dict(state["budget_ledger"])
        record_agent_spend(new_ledger, role="mutator", usd=0.40, calls=1)
        return {"budget_ledger": new_ledger, "round": state.get("round", 0) + 1}

    def post_budget_edge(state: CampaignState) -> str:
        outcome = over_budget_edge(state)
        if outcome == "continue":
            if state.get("round", 0) >= state.get("max_rounds", 1):
                return "finalize"
            return "round"
        return outcome

    async def finalize_node(state: CampaignState) -> dict:
        if state.get("over_budget"):
            return {"termination_reason": "BUDGET_CAP_EXCEEDED"}
        return {"termination_reason": "ROUND_CAP_REACHED"}

    builder: StateGraph = StateGraph(CampaignState)
    builder.add_node("round", round_node)
    builder.add_node("budget_check", budget_check_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "round")
    builder.add_edge("round", "budget_check")
    builder.add_conditional_edges(
        "budget_check",
        post_budget_edge,
        # "budget_approval" is unreachable here (warn_usd is set above any
        # spend this test reaches) but every declared outcome still needs
        # a mapped target for `add_conditional_edges()` to compile.
        {"round": "round", "finalize": "finalize", "budget_approval": "finalize"},
    )
    builder.add_edge("finalize", END)
    compiled = builder.compile()

    initial_state: CampaignState = CampaignState(round=0, max_rounds=10, budget_ledger=ledger)
    final_state = await compiled.ainvoke(initial_state)

    assert final_state["round"] == 2
    assert final_state["budget_ledger"]["spent_usd"] == pytest.approx(0.80)
    assert final_state["termination_reason"] == "BUDGET_CAP_EXCEEDED"


# --- Within-role step cap: StepCapMiddleware ----------------------------


def test_build_step_cap_middleware_uses_default_cap():
    middleware = build_step_cap_middleware("mutator")
    assert len(middleware) == 1
    assert isinstance(middleware[0], StepCapMiddleware)
    assert middleware[0].role == "mutator"
    assert middleware[0].max_model_calls == DEFAULT_MAX_MODEL_CALLS_PER_ROLE


async def test_step_cap_middleware_bounds_real_handler_invocations_against_infinite_loop():
    """05-RESEARCH `mitigation1b_withinrole_abort.py`: a deliberately
    infinite-tool-calling model would otherwise never stop -- the
    middleware intercepts BEFORE the handler once the cap is reached, so
    the number of REAL model invocations equals exactly `max_model_calls`."""

    class _NoopOutput(BaseModel):
        value: str

    @tool
    def noop_tool(note: str) -> str:
        """A harmless no-op tool -- never satisfies the structured-output
        requirement, so a model scripted to keep calling it would loop
        forever without the step cap."""
        return "ok"

    script = [{"name": "noop_tool", "args": {"note": f"n{i}"}} for i in range(10)]
    model = ScriptedToolCallChatModel(script=script)

    middleware = StepCapMiddleware(role="mutator", max_model_calls=3)
    agent = create_deep_agent(
        model=model,
        tools=[noop_tool],
        middleware=[middleware],
        response_format=ToolStrategy(_NoopOutput, handle_errors=False),
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert result.get("structured_response") is None
    assert len(model.received_message_batches) == 3
    assert middleware._calls_made == 3
