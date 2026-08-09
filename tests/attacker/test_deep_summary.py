"""Tests for `attacker/summary.py`'s `compute_deep_summary()` (05-09-PLAN.md
Task 1) and the `run_scan()`-level deep-summary/limitations wiring
(Task 2).

Task 1's tests are pure -- no adapter, no module, no attacker stack import
-- constructing `CampaignResult`/`BudgetLedger`/`VariantRecord` by hand so
every reconciliation edge case is exercised without a live campaign.
Task 2's tests drive `api.run_scan()` end to end with the attacker layer
monkeypatched, mirroring `tests/test_api.py`'s existing mocked-adapter/
mocked-module pattern -- no attacker stack import needed there either.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest

import llmsec.api as api_module
from llmsec.attacker.config import AttackerConfig
from llmsec.attacker.runner import CampaignResult
from llmsec.attacker.state import (
    ROLE_NAMES,
    BudgetLedger,
    CampaignState,
    RoleSpend,
    VariantRecord,
)
from llmsec.attacker.summary import DeepSummaryReconciliationError, compute_deep_summary
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, Finding, ScanContext, ScanReport, TargetResponse, TestCase, Verdict

# `pyproject.toml` sets `asyncio_mode = "auto"` — no explicit marker needed;
# `async def` tests below are auto-detected, `def` tests run synchronously.


# --- Task 1: compute_deep_summary() -- pure unit fixtures -------------------


def _role_spend(calls: int, usd: float) -> RoleSpend:
    return RoleSpend(calls=calls, usd=usd, share_ceiling_usd=None)


def _full_per_role(**overrides: RoleSpend) -> dict[str, RoleSpend]:
    base: dict[str, RoleSpend] = {
        "recon": _role_spend(1, 0.001),
        "strategist": _role_spend(2, 0.002),
        "mutator": _role_spend(1, 0.001),
        "analyst": _role_spend(1, 0.001),
        "crescendo": _role_spend(0, 0.0),
    }
    base.update(overrides)
    return base


def _ledger(
    *,
    per_role: dict[str, RoleSpend],
    target_spent_usd: float = 0.0,
    truncated: bool = False,
) -> BudgetLedger:
    attacker_spent = sum(r["usd"] for r in per_role.values())
    agent_calls = sum(r["calls"] for r in per_role.values())
    return BudgetLedger(
        cap_usd=5.0,
        warn_usd=3.75,
        spent_usd=attacker_spent + target_spent_usd,
        attacker_spent_usd=attacker_spent,
        target_spent_usd=target_spent_usd,
        agent_calls=agent_calls,
        agent_call_ceiling=300,
        per_role=per_role,
        truncated=truncated,
        overshoot_rounds=0,
        warn_approved=False,
        unpriced_calls=0,
    )


def _variant_record(
    *, parent_case_id: str, round_: int, contributing_agent: str, index: int = 0
) -> VariantRecord:
    return VariantRecord(
        payload="mutated payload text",
        technique_family="instruction_override",
        parent_case_id=parent_case_id,
        parent_technique_id=parent_case_id,
        round=round_,
        contributing_agent=contributing_agent,
        variant_index=index,
        turns=None,
    )


def _finding(
    case_id: str,
    *,
    severity: str,
    parent_case_id: str | None = None,
    contributing_agent: str | None = None,
    round_: int | None = None,
) -> Finding:
    verdict = Verdict.FULL_COMPROMISE if severity in ("high", "critical") else Verdict.PARTIAL_LEAK
    return Finding(
        case_id=case_id,
        technique_id=case_id,
        verdict=verdict,
        severity=severity,
        owasp_ref="LLM01:2025",
        evidence="evidence text",
        remediation="remediation text",
        parent_case_id=parent_case_id,
        parent_technique_id=parent_case_id,
        round=round_,
        contributing_agent=contributing_agent,
    )


def _static_result(case_id: str, verdict: Verdict, evidence: str = "") -> tuple[str, EvalResult]:
    return (
        "prompt_injection",
        EvalResult(
            case_id=case_id, verdict=verdict, confidence=0.9, evidence=evidence, detection_layer="regex"
        ),
    )


def _campaign_result(
    *,
    lineage: dict[str, VariantRecord],
    ledger: BudgetLedger,
    round_: int = 1,
    termination_reason: str | None = "TECHNIQUES_EXHAUSTED",
    constraint_violations: int = 0,
    abandoned_arcs: int = 0,
    role_structural_failures: int = 0,
    audit_path: Path | None = None,
) -> CampaignResult:
    final_state: CampaignState = CampaignState(
        round=round_,
        budget_ledger=ledger,
        termination_reason=termination_reason,  # type: ignore[typeddict-item]
    )
    return CampaignResult(
        eval_results=[],
        lineage=lineage,
        final_state=final_state,
        limitations=[],
        audit_path=audit_path,
        constraint_violations=constraint_violations,
        abandoned_arcs=abandoned_arcs,
        role_structural_failures=role_structural_failures,
    )


def test_bypass_counted_when_severity_strictly_higher_than_parent() -> None:
    static_results = [_static_result("DIRECT-001", Verdict.BLOCKED)]
    lineage = {
        "DIRECT-001-mut-1": _variant_record(
            parent_case_id="DIRECT-001", round_=1, contributing_agent="mutator"
        )
    }
    findings = [
        _finding(
            "DIRECT-001-mut-1",
            severity="high",
            parent_case_id="DIRECT-001",
            contributing_agent="mutator",
            round_=1,
        )
    ]
    ledger = _ledger(per_role=_full_per_role(), target_spent_usd=0.02)
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    summary = compute_deep_summary(findings, static_results, campaign_result)

    assert summary.bypass_case_ids == ["DIRECT-001-mut-1"]
    assert summary.bypasses_found == 1
    assert summary.bypasses_found == len(summary.bypass_case_ids)
    assert summary.cost_per_bypass_usd == pytest.approx(summary.spend_usd)


def test_same_severity_as_parent_is_not_a_bypass() -> None:
    """A mutated variant that only reproduces the parent's own result must
    never be counted as a bypass (`<behavior>`'s explicit non-bypass case)."""
    static_results = [_static_result("DIRECT-002", Verdict.PARTIAL_LEAK, evidence="partial")]
    lineage = {
        "DIRECT-002-mut-1": _variant_record(
            parent_case_id="DIRECT-002", round_=1, contributing_agent="crescendo"
        )
    }
    findings = [
        _finding(
            "DIRECT-002-mut-1",
            severity="medium",  # same band as PARTIAL_LEAK's parent
            parent_case_id="DIRECT-002",
            contributing_agent="crescendo",
            round_=1,
        )
    ]
    ledger = _ledger(per_role=_full_per_role())
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    summary = compute_deep_summary(findings, static_results, campaign_result)

    assert summary.bypass_case_ids == []
    assert summary.bypasses_found == 0
    assert summary.bypasses_found == len(summary.bypass_case_ids)


def test_zero_bypass_costly_campaign_reports_spend_honestly() -> None:
    """A campaign that spent real money but found no bypass is reported
    honestly, not omitted -- `cost_per_bypass_usd` is `None`, never `0.0`."""
    static_results = [_static_result("DIRECT-002", Verdict.PARTIAL_LEAK)]
    lineage = {
        "DIRECT-002-mut-1": _variant_record(
            parent_case_id="DIRECT-002", round_=1, contributing_agent="mutator"
        )
    }
    findings = [
        _finding(
            "DIRECT-002-mut-1",
            severity="medium",
            parent_case_id="DIRECT-002",
            contributing_agent="mutator",
            round_=1,
        )
    ]
    ledger = _ledger(per_role=_full_per_role(strategist=_role_spend(2, 1.50)), target_spent_usd=0.10)
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    summary = compute_deep_summary(findings, static_results, campaign_result)

    assert summary.bypasses_found == 0
    assert summary.cost_per_bypass_usd is None
    assert summary.spend_usd > 0


@pytest.mark.parametrize(
    "shape",
    ["empty_campaign", "single_bypass", "mixed_bypass_and_reproduction", "all_static_findings"],
)
def test_bypasses_found_equals_len_bypass_case_ids_across_shapes(shape: str) -> None:
    if shape == "empty_campaign":
        static_results: list[tuple[str, EvalResult]] = []
        lineage: dict[str, VariantRecord] = {}
        findings: list[Finding] = []
    elif shape == "single_bypass":
        static_results = [_static_result("DIRECT-001", Verdict.BLOCKED)]
        lineage = {
            "DIRECT-001-mut-1": _variant_record(
                parent_case_id="DIRECT-001", round_=1, contributing_agent="mutator"
            )
        }
        findings = [
            _finding(
                "DIRECT-001-mut-1",
                severity="high",
                parent_case_id="DIRECT-001",
                contributing_agent="mutator",
                round_=1,
            )
        ]
    elif shape == "mixed_bypass_and_reproduction":
        static_results = [
            _static_result("DIRECT-001", Verdict.BLOCKED),
            _static_result("DIRECT-002", Verdict.PARTIAL_LEAK),
        ]
        lineage = {
            "DIRECT-001-mut-1": _variant_record(
                parent_case_id="DIRECT-001", round_=1, contributing_agent="mutator"
            ),
            "DIRECT-002-mut-1": _variant_record(
                parent_case_id="DIRECT-002", round_=1, contributing_agent="crescendo"
            ),
            "DIRECT-002-mut-2": _variant_record(
                parent_case_id="DIRECT-002", round_=1, contributing_agent="mutator", index=1
            ),
        }
        findings = [
            _finding(
                "DIRECT-001-mut-1",
                severity="high",
                parent_case_id="DIRECT-001",
                contributing_agent="mutator",
                round_=1,
            ),
            _finding(
                "DIRECT-002-mut-1",
                severity="medium",  # reproduces parent -- not a bypass
                parent_case_id="DIRECT-002",
                contributing_agent="crescendo",
                round_=1,
            ),
            _finding(
                "DIRECT-002-mut-2",
                severity="high",  # strictly higher than PARTIAL_LEAK's medium -- a bypass
                parent_case_id="DIRECT-002",
                contributing_agent="mutator",
                round_=1,
            ),
        ]
    else:  # all_static_findings -- no deep lineage at all
        static_results = [_static_result("DIRECT-003", Verdict.FULL_COMPROMISE, evidence="leaked")]
        lineage = {}
        findings = [_finding("DIRECT-003", severity="high")]

    ledger = _ledger(per_role=_full_per_role())
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    summary = compute_deep_summary(findings, static_results, campaign_result)

    assert summary.bypasses_found == len(summary.bypass_case_ids)


def test_per_role_attribution_and_reconciliation() -> None:
    static_results = [
        _static_result("DIRECT-001", Verdict.BLOCKED),
        _static_result("DIRECT-002", Verdict.BLOCKED),
    ]
    lineage = {
        "DIRECT-001-mut-1": _variant_record(
            parent_case_id="DIRECT-001", round_=1, contributing_agent="mutator"
        ),
        "DIRECT-002-mut-1": _variant_record(
            parent_case_id="DIRECT-002", round_=1, contributing_agent="crescendo"
        ),
    }
    findings = [
        _finding(
            "DIRECT-001-mut-1",
            severity="high",
            parent_case_id="DIRECT-001",
            contributing_agent="mutator",
            round_=1,
        ),
        _finding(
            "DIRECT-002-mut-1",
            severity="high",
            parent_case_id="DIRECT-002",
            contributing_agent="crescendo",
            round_=1,
        ),
    ]
    ledger = _ledger(per_role=_full_per_role())
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    summary = compute_deep_summary(findings, static_results, campaign_result)

    assert set(summary.per_role_activity) == set(ROLE_NAMES)
    assert summary.per_role_activity["mutator"].bypasses == 1
    assert summary.per_role_activity["crescendo"].bypasses == 1
    assert summary.per_role_activity["analyst"].bypasses == 0

    total_calls = sum(activity.calls for activity in summary.per_role_activity.values())
    total_spend = sum(activity.spend_usd for activity in summary.per_role_activity.values())
    assert total_calls == summary.agent_calls
    assert total_spend == pytest.approx(ledger["attacker_spent_usd"])


def test_corrupted_ledger_call_count_raises_reconciliation_error() -> None:
    static_results = [_static_result("DIRECT-001", Verdict.BLOCKED)]
    lineage = {
        "DIRECT-001-mut-1": _variant_record(
            parent_case_id="DIRECT-001", round_=1, contributing_agent="mutator"
        )
    }
    findings = [
        _finding(
            "DIRECT-001-mut-1",
            severity="high",
            parent_case_id="DIRECT-001",
            contributing_agent="mutator",
            round_=1,
        )
    ]
    ledger = _ledger(per_role=_full_per_role())
    ledger["agent_calls"] = 999  # deliberately corrupted -- no longer matches per-role sum
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    with pytest.raises(DeepSummaryReconciliationError):
        compute_deep_summary(findings, static_results, campaign_result)


def test_corrupted_ledger_attacker_spend_raises_reconciliation_error() -> None:
    static_results = [_static_result("DIRECT-001", Verdict.BLOCKED)]
    lineage: dict[str, VariantRecord] = {}
    findings: list[Finding] = []
    ledger = _ledger(per_role=_full_per_role())
    ledger["attacker_spent_usd"] = 123.45  # deliberately corrupted
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    with pytest.raises(DeepSummaryReconciliationError):
        compute_deep_summary(findings, static_results, campaign_result)


def test_deep_finding_with_unresolvable_parent_is_not_counted() -> None:
    """A deep finding whose `parent_case_id` cannot be traced to a specific
    static case must not be counted -- defensive, should never happen in
    practice, but the function verifies independently rather than trusting
    the caller."""
    static_results: list[tuple[str, EvalResult]] = []  # no static case at all
    lineage = {
        "DIRECT-999-mut-1": _variant_record(
            parent_case_id="DIRECT-999", round_=1, contributing_agent="mutator"
        )
    }
    findings = [
        _finding(
            "DIRECT-999-mut-1",
            severity="high",
            parent_case_id="DIRECT-999",
            contributing_agent="mutator",
            round_=1,
        )
    ]
    ledger = _ledger(per_role=_full_per_role())
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    summary = compute_deep_summary(findings, static_results, campaign_result)

    assert summary.bypass_case_ids == []


def test_summary_surfaces_termination_reason_truncated_and_audit_path(tmp_path: Path) -> None:
    ledger = _ledger(per_role=_full_per_role(), truncated=True)
    audit_path = tmp_path / "scan-attacker-audit.jsonl"
    campaign_result = _campaign_result(
        lineage={},
        ledger=ledger,
        termination_reason="BUDGET_CAP_EXCEEDED",
        constraint_violations=2,
        abandoned_arcs=1,
        role_structural_failures=4,
        audit_path=audit_path,
    )

    summary = compute_deep_summary([], [], campaign_result)

    assert summary.truncated is True
    assert summary.termination_reason == "BUDGET_CAP_EXCEEDED"
    assert summary.constraint_violations == 2
    assert summary.abandoned_arcs == 1
    assert summary.role_structural_failures == 4
    assert summary.audit_log_path == str(audit_path)
    assert summary.cases_attacked == 0
    assert summary.variants_dispatched == 0
    assert summary.bypasses_found == 0


def test_cases_attacked_counts_distinct_parents_not_variants() -> None:
    """IN-01: this is a PURE unit test of `compute_deep_summary()`'s own
    counting logic in isolation (Task 1's whole-module convention -- no
    adapter, no module, no attacker-stack import) -- it hand-constructs an
    already-disambiguated `lineage` (`index=1` passed explicitly for the
    second round's `VariantRecord`) rather than driving `graph.py`'s real
    per-round `enumerate()`-based `variant_index` assignment, so it does
    NOT exercise (and would not have caught) CR-01's `case_id`-collision
    bug on its own. The integration-level regression CR-01 asks for --
    driving the REAL `dispatch_variants_node`/`mutator_node` pipeline
    across two rounds against the same selected case and asserting the
    lineage/`eval_results` counts reconcile against the true dispatch
    count -- lives in
    `test_audit_wiring.py::test_cr01_repeated_parent_case_across_rounds_yields_unique_case_ids_and_reconciled_lineage`,
    which shares this same two-distinct-parent-cases-attacked shape end
    to end. This test remains a hand-built fixture deliberately: it is
    `compute_deep_summary()`'s own `cases_attacked`-vs-`variants_dispatched`
    counting-logic test, not a case_id-generation test.
    """
    static_results = [_static_result("DIRECT-001", Verdict.BLOCKED)]
    lineage = {
        "DIRECT-001-mut-1": _variant_record(
            parent_case_id="DIRECT-001", round_=1, contributing_agent="mutator"
        ),
        "DIRECT-001-mut-2": _variant_record(
            parent_case_id="DIRECT-001", round_=2, contributing_agent="mutator", index=1
        ),
    }
    findings: list[Finding] = []
    ledger = _ledger(per_role=_full_per_role())
    campaign_result = _campaign_result(lineage=lineage, ledger=ledger)

    summary = compute_deep_summary(findings, static_results, campaign_result)

    assert summary.cases_attacked == 1  # one distinct parent case
    assert summary.variants_dispatched == 2  # two dispatched variants


def test_no_case_id_string_parsing_in_summary_module() -> None:
    import llmsec.attacker.summary as summary_module

    source = inspect.getsource(summary_module)
    assert "parent_case_id.split(" not in source
    assert ".case_id.split(" not in source


def test_scan_report_deep_summary_defaults_to_none() -> None:
    field = ScanReport.model_fields["deep_summary"]
    assert field.default is None


# --- Task 2: run_scan() deep-summary / limitations wiring -------------------


class _AttackerMockAdapter:
    def __init__(self) -> None:
        self.send = AsyncMock(side_effect=self._send)
        self.closed = False
        self.supports_system_prompt_override = False
        self.supports_multi_turn = False

    async def _send(self, case: TestCase) -> TargetResponse:
        return TargetResponse(case_id=case.case_id, raw_text=f"response-to-{case.case_id}", latency_ms=1.0)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


class _AttackerMockModule:
    id = "prompt_injection"
    name = "Mock Prompt Injection"
    owasp_ref = "LLM01:2025"
    uses_attacker_llm = True

    def __init__(self, cases: list[tuple[str, Verdict, str]]) -> None:
        self._cases = cases

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        for case_id, _verdict, _evidence in self._cases:
            yield TestCase(case_id=case_id, prompt=f"prompt-{case_id}", technique_id=case_id)

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        verdict, evidence = next((v, e) for cid, v, e in self._cases if cid == case.case_id)
        return EvalResult(
            case_id=case.case_id, verdict=verdict, confidence=0.9, evidence=evidence, detection_layer="regex"
        )


def _attacker_deep_config(tmp_path: Path) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(
            type="http_app",
            method="POST",
            url="http://localhost:8000/chat",
            headers={},
            body_template='{"message": "{{payload}}"}',
            response_path="response",
        ),
        enabled_modules=["prompt_injection"],
        max_concurrency=5,
        output_dir=str(tmp_path / "reports"),
        judge_model="openai/gpt-4o-mini",
        judge_api_key_env=None,
        attacker=AttackerConfig(enabled=True),
    )


def _patch_attacker_fixtures(
    monkeypatch: pytest.MonkeyPatch, module: _AttackerMockModule
) -> list[_AttackerMockAdapter]:
    monkeypatch.setattr(
        api_module.PluginRegistry,
        "load_allowed",
        lambda self, allowlist, module_config=None: {module.id: module},
    )
    created: list[_AttackerMockAdapter] = []

    def _factory(*args: object, **kwargs: object) -> _AttackerMockAdapter:
        instance = _AttackerMockAdapter()
        created.append(instance)
        return instance

    monkeypatch.setattr(api_module, "HttpAppAdapter", _factory)
    return created


def _make_campaign_result_for_deep_run(
    *,
    truncated: bool = False,
    audit_path: Path | None = None,
    lineage: dict[str, VariantRecord] | None = None,
    eval_results: list[tuple[str, EvalResult]] | None = None,
) -> CampaignResult:
    ledger = _ledger(per_role=_full_per_role(), truncated=truncated)
    final_state: CampaignState = CampaignState(
        round=1, budget_ledger=ledger, termination_reason="TECHNIQUES_EXHAUSTED"
    )
    return CampaignResult(
        eval_results=eval_results if eval_results is not None else [],
        lineage=lineage if lineage is not None else {},
        final_state=final_state,
        limitations=[],
        audit_path=audit_path,
        constraint_violations=0,
        abandoned_arcs=0,
        role_structural_failures=0,
    )


def _static_only_config(tmp_path: Path) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(
            type="http_app",
            method="POST",
            url="http://localhost:8000/chat",
            headers={},
            body_template='{"message": "{{payload}}"}',
            response_path="response",
        ),
        enabled_modules=["prompt_injection"],
        max_concurrency=5,
        output_dir=str(tmp_path / "reports"),
        judge_model="openai/gpt-4o-mini",
        judge_api_key_env=None,
        # Explicit `attacker=None` (05-11 Rule 1 fix): omitting this kwarg
        # is NOT equivalent to "no attacker config at all" -- `ScanConfig`'s
        # `settings_customise_sources()` still consults the cwd's
        # `llmsec.config.yaml` for any key absent from `init_settings`, and
        # 05-11 gave that file a real `attacker: {enabled: true, ...}`
        # block for the live smoke gate. Without this explicit override,
        # this "static-only, deep branch never runs" fixture silently
        # inherited that YAML block and the test's own
        # `assert report.deep_summary is None` failed live. Passing
        # `attacker=None` explicitly makes this fixture hermetic --
        # `init_settings` always wins over the YAML source for a key it
        # actually contains, `None` included.
        attacker=None,
    )


async def test_run_scan_static_only_deep_summary_none_and_limitations_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _AttackerMockModule([("c1", Verdict.BLOCKED, "no leak here")])
    module.uses_attacker_llm = False
    _patch_attacker_fixtures(monkeypatch, module)
    config = _static_only_config(tmp_path)

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.deep_summary is None
    assert report.limitations == api_module._scan_limitations(report.module_ids, report.case_log)


async def test_run_scan_deep_run_populates_summary_and_audit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _AttackerMockModule([("c1", Verdict.BLOCKED, "held the line")])
    _patch_attacker_fixtures(monkeypatch, module)
    config = _attacker_deep_config(tmp_path)

    audit_path = tmp_path / "reports" / "scan-attacker-audit.jsonl"
    lineage = {
        "c1-mut-1": _variant_record(parent_case_id="c1", round_=1, contributing_agent="mutator")
    }
    deep_eval_result: tuple[str, EvalResult] = (
        "prompt_injection",
        EvalResult(
            case_id="c1-mut-1",
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.9,
            evidence="mutated bypass evidence",
            detection_layer="regex",
        ),
    )
    campaign_result = _make_campaign_result_for_deep_run(
        audit_path=audit_path, lineage=lineage, eval_results=[deep_eval_result]
    )
    mock_run_campaign = AsyncMock(return_value=campaign_result)
    monkeypatch.setattr(api_module, "run_attacker_campaign", mock_run_campaign)

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.deep_summary is not None
    assert report.deep_summary.audit_log_path == str(audit_path)
    assert report.deep_summary.bypasses_found == 1
    assert report.deep_summary.bypass_case_ids == ["c1-mut-1"]
    mock_run_campaign.assert_awaited_once()


async def test_run_scan_truncated_deep_run_includes_disclosure_even_with_zero_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _AttackerMockModule([("c1", Verdict.BLOCKED, "held the line")])
    _patch_attacker_fixtures(monkeypatch, module)
    config = _attacker_deep_config(tmp_path)

    # Zero-finding deep run (no lineage, no deep eval_results) that still
    # hit its budget cap -- the disclosure is driven by the ledger, not by
    # whether any finding exists (D-83).
    campaign_result = _make_campaign_result_for_deep_run(truncated=True)
    monkeypatch.setattr(api_module, "run_attacker_campaign", AsyncMock(return_value=campaign_result))

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.findings == []
    assert any("hard budget cap" in item for item in report.limitations)
    assert report.deep_summary is not None
    assert report.deep_summary.truncated is True


async def test_run_scan_untruncated_deep_run_has_no_truncation_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _AttackerMockModule([("c1", Verdict.BLOCKED, "held the line")])
    _patch_attacker_fixtures(monkeypatch, module)
    config = _attacker_deep_config(tmp_path)

    campaign_result = _make_campaign_result_for_deep_run(truncated=False)
    monkeypatch.setattr(api_module, "run_attacker_campaign", AsyncMock(return_value=campaign_result))

    report = await api_module.run_scan(config, bypass_flag=True)

    assert not any("hard budget cap" in item for item in report.limitations)
    assert report.deep_summary is not None
    assert report.deep_summary.truncated is False


async def test_run_scan_repeated_deep_runs_produce_byte_identical_limitations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _make_module() -> _AttackerMockModule:
        return _AttackerMockModule([("c1", Verdict.BLOCKED, "held the line")])

    _patch_attacker_fixtures(monkeypatch, _make_module())
    campaign_result_1 = _make_campaign_result_for_deep_run(truncated=True)
    monkeypatch.setattr(api_module, "run_attacker_campaign", AsyncMock(return_value=campaign_result_1))
    config1 = _attacker_deep_config(tmp_path / "run1")
    report1 = await api_module.run_scan(config1, bypass_flag=True)

    _patch_attacker_fixtures(monkeypatch, _make_module())
    campaign_result_2 = _make_campaign_result_for_deep_run(truncated=True)
    monkeypatch.setattr(api_module, "run_attacker_campaign", AsyncMock(return_value=campaign_result_2))
    config2 = _attacker_deep_config(tmp_path / "run2")
    report2 = await api_module.run_scan(config2, bypass_flag=True)

    assert report1.limitations == report2.limitations


async def test_run_scan_deep_mode_raising_still_persists_static_report_with_failure_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _AttackerMockModule([("c1", Verdict.FULL_COMPROMISE, "leaked without deep mode")])
    _patch_attacker_fixtures(monkeypatch, module)
    config = _attacker_deep_config(tmp_path)

    async def _raise(*args: object, **kwargs: object) -> CampaignResult:
        raise RuntimeError("attacker stack boom")

    monkeypatch.setattr(api_module, "run_attacker_campaign", _raise)

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.deep_summary is None
    assert api_module._DEEP_MODE_FAILED_LIMITATION_NOTE in report.limitations
    assert len(report.findings) == 1  # the static c1 finding is still produced
    written_path = tmp_path / "reports" / f"scan_{report.scan_id}.json"
    assert written_path.exists()


def test_score_runs_before_redaction_source_order() -> None:
    source = inspect.getsource(api_module.run_scan)
    assert source.index("score(") < source.index("redact_credential_match(redact_pii_match(")
