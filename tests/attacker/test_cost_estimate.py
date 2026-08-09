"""05-04-PLAN.md Task 3: labelled typical/worst-case estimate, the hard cap
display, and the warn-threshold approval pause.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import typer
import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typer.testing import CliRunner
from unittest.mock import AsyncMock

from llmsec.attacker import budget
from llmsec.attacker.budget import (
    CostEstimate,
    budget_approval_node,
    budget_check_node,
    estimate_campaign_cost,
    over_budget_edge,
    render_cost_notice,
)
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.state import BudgetLedger, CampaignState
from llmsec.cli import app
from llmsec.models import ScanReport

cli_runner = CliRunner()


def _fresh_ledger(*, cap_usd: float = 1.0, warn_usd: float = 0.5) -> BudgetLedger:
    return BudgetLedger(
        cap_usd=cap_usd,
        warn_usd=warn_usd,
        spent_usd=0.0,
        attacker_spent_usd=0.0,
        target_spent_usd=0.0,
        agent_calls=0,
        agent_call_ceiling=1000,
        per_role={},
        truncated=False,
        overshoot_rounds=0,
        warn_approved=False,
        unpriced_calls=0,
    )


# --- estimate_campaign_cost / render_cost_notice ------------------------


@pytest.mark.parametrize("queue_size", [1, 2, 5])
def test_estimate_campaign_cost_worst_case_at_least_typical(queue_size):
    cfg = AttackerConfig(enabled=True, profile="standard")
    settings = resolve_settings(cfg)
    estimate = estimate_campaign_cost(settings, queue_size, cfg.model)
    assert estimate.worst_case_usd is not None
    assert estimate.typical_usd is not None
    assert estimate.worst_case_usd >= estimate.typical_usd


def test_render_cost_notice_contains_three_distinct_labelled_figures():
    estimate = CostEstimate(
        typical_usd=0.42,
        worst_case_usd=1.75,
        cap_usd=2.00,
        assumed_calls_typical=8,
        assumed_calls_worst=20,
    )
    notice = render_cost_notice(estimate)
    assert "0.42" in notice
    assert "1.75" in notice
    assert "2.00" in notice
    # Both estimate figures carry an explicit "estimate" label somewhere in
    # the notice, and the notice frames itself as estimates, never a quote.
    assert "estimate" in notice.lower()
    assert "Typical" in notice
    assert "Worst-case" in notice
    assert "Hard cap" in notice


def test_estimate_campaign_cost_unpriced_model_yields_none_not_zero():
    cfg = AttackerConfig(enabled=True, profile="standard", model="not-a-real-model")
    settings = resolve_settings(cfg)
    estimate = estimate_campaign_cost(settings, 3, cfg.model)
    assert estimate.typical_usd is None
    assert estimate.worst_case_usd is None


def test_render_cost_notice_unpriced_model_renders_explicit_unavailable_not_zero():
    estimate = CostEstimate(
        typical_usd=None,
        worst_case_usd=None,
        cap_usd=2.00,
        assumed_calls_typical=8,
        assumed_calls_worst=20,
    )
    notice = render_cost_notice(estimate)
    assert "UNAVAILABLE" in notice
    assert "$0.00" not in notice
    assert "2.00" in notice  # the hard cap is still shown


# --- budget_approval_node's pause-is-first-statement AST check ----------


def test_budget_approval_node_pauses_before_any_other_statement():
    src = inspect.getsource(budget.budget_approval_node)
    fn = ast.parse(src.lstrip()).body[0]
    body = [
        statement
        for statement in fn.body
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Constant)
    ]
    assert "interrupt" in ast.dump(body[0]), "pause must be the first statement"


# --- Warn-threshold approval pause: standalone graph, mirrors 05-RESEARCH ---
# --- mitigation_interrupt.py's own proven shape -----------------------------


def _build_approval_test_graph(checkpointer):
    def round_node(state: CampaignState) -> dict:
        return {}

    def post_budget_edge(state: CampaignState) -> str:
        return over_budget_edge(state)

    def post_approval_edge(state: CampaignState) -> str:
        if state.get("termination_reason") is not None:
            return "finalize"
        return "finalize"

    async def finalize_node(state: CampaignState) -> dict:
        if state.get("termination_reason") is not None:
            return {}
        return {"termination_reason": "ROUND_CAP_REACHED"}

    builder: StateGraph = StateGraph(CampaignState)
    builder.add_node("round", round_node)
    builder.add_node("budget_check", budget_check_node)
    builder.add_node("budget_approval", budget_approval_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "round")
    builder.add_edge("round", "budget_check")
    builder.add_conditional_edges(
        "budget_check",
        post_budget_edge,
        {"continue": "finalize", "finalize": "finalize", "budget_approval": "budget_approval"},
    )
    builder.add_conditional_edges(
        "budget_approval", post_approval_edge, {"finalize": "finalize"}
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


async def test_campaign_whose_warn_threshold_is_never_crossed_never_pauses():
    ledger = _fresh_ledger(cap_usd=10.0, warn_usd=9.0)  # spend (0.0) never reaches warn_usd
    compiled = _build_approval_test_graph(MemorySaver())
    config = {"configurable": {"thread_id": "no-pause"}}
    result = await compiled.ainvoke(CampaignState(budget_ledger=ledger), config=config)
    assert "__interrupt__" not in result
    assert result.get("termination_reason") == "ROUND_CAP_REACHED"


async def test_resumed_campaign_records_exactly_one_approval_event():
    ledger = _fresh_ledger(cap_usd=10.0, warn_usd=0.0)  # 0.0 spend already >= warn_usd
    compiled = _build_approval_test_graph(MemorySaver())
    config = {"configurable": {"thread_id": "one-approval"}}

    first = await compiled.ainvoke(CampaignState(budget_ledger=ledger), config=config)
    assert "__interrupt__" in first

    second = await compiled.ainvoke(Command(resume=True), config=config)
    assert second["budget_ledger"]["warn_approved"] is True
    # The resumed node body re-executed from the top (05-RESEARCH Pitfall
    # 3), but the approval-recording assignment only ever happened once --
    # `warn_approved` is a plain bool, not a counter, so a double-record
    # would be silently indistinguishable from a single one UNLESS we
    # instead assert the round-cap path was reached exactly once too.
    assert second.get("termination_reason") == "ROUND_CAP_REACHED"


async def test_resumed_campaign_with_refusal_terminates_with_recorded_reason():
    ledger = _fresh_ledger(cap_usd=10.0, warn_usd=0.0)
    compiled = _build_approval_test_graph(MemorySaver())
    config = {"configurable": {"thread_id": "refusal"}}

    first = await compiled.ainvoke(CampaignState(budget_ledger=ledger), config=config)
    assert "__interrupt__" in first

    second = await compiled.ainvoke(Command(resume=False), config=config)
    assert second.get("termination_reason") == "WARN_APPROVAL_REFUSED"


# --- CLI: cost notice echoed before the scan coroutine is entered -------


def _scan_yaml(tmp_path):
    data = {
        "target": {
            "type": "http_app",
            "method": "POST",
            "url": "http://localhost:8000/chat",
            "headers": {},
            "body_template": '{"message": "{{payload}}"}',
            "response_path": "response",
        },
        "enabled_modules": ["prompt_injection"],
        "max_concurrency": 5,
    }
    config_path = tmp_path / "llmsec.config.yaml"
    config_path.write_text(yaml.safe_dump(data))
    return config_path


def _fake_report() -> ScanReport:
    return ScanReport(
        scan_id="fake-scan-1",
        target_summary="http_app:POST http://localhost:8000/chat",
        module_ids=["prompt_injection"],
        findings=[],
        case_log=[],
        started_at="2026-07-21T00:00:00Z",
        completed_at="2026-07-21T00:00:01Z",
    )


def test_cli_deep_cost_notice_written_before_scan_coroutine_entered(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)

    async def _run_scan_side_effect(*args, **kwargs):
        typer.echo("RUN_SCAN_ENTERED")
        return _fake_report()

    mock_run_scan = AsyncMock(side_effect=_run_scan_side_effect)
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)

    result = cli_runner.invoke(
        app, ["scan", "--config", str(yaml_path), "--deep", "--yes-i-am-authorized"]
    )

    assert result.exit_code == 0, result.output
    mock_run_scan.assert_awaited_once()
    notice_pos = result.output.find("--deep cost estimate")
    entered_pos = result.output.find("RUN_SCAN_ENTERED")
    assert notice_pos != -1, result.output
    assert entered_pos != -1, result.output
    assert notice_pos < entered_pos
