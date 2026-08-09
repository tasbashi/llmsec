"""D-94 gate / AT-1 + AT-3: coverage-delta reconciliation and audit
completeness (05-10-PLAN.md Task 3).

AT-1 and AT-3 are the same evidentiary claim from two directions (the
plan's own `<action>`): AT-1 asks "does the reported coverage-delta figure
match the evidence", AT-3 asks "is the evidence itself complete and
ordered". Both are checked here against ONE real, scripted campaign, using
code written independently of `attacker/summary.py`'s `compute_deep_summary()`
-- a reconciliation that called the function it is checking would prove
nothing.

Drives `build_campaign_graph()` directly (rather than
`run_attacker_campaign()`) so this test owns its own `AttackerAuditHandler`
instance -- the only way to read back `captured_events`/`written_lines`
independently of what got persisted to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmsec.attacker.audit import AttackerAuditHandler, AttackerAuditWriter
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles.analyst import ObservedDefence, build_analyst_agent
from llmsec.attacker.roles.crescendo import CrescendoOutput, build_crescendo_agent
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.runner import CampaignResult
from llmsec.attacker.state import QUEUE_ELIGIBLE_VERDICTS, QueuedCase, VariantRecord, new_campaign_state
from llmsec.attacker.summary import compute_deep_summary
from llmsec.models import EvalResult, Finding, TestCase, Verdict
from llmsec.scoring.engine import Severity, score

_STATIC_CASE_ID = "COV-001"
_STATIC_TECHNIQUE_ID = "COV-001"
_MODULE_ID = "cov_module"

# Deliberately mixed outcomes so the reconciliation is meaningful: two
# variants bypass the static BLOCKED parent, one reproduces it (BLOCKED
# again, not a bypass).
_VERDICT_BY_CASE_ID: dict[str, Verdict] = {
    f"{_STATIC_CASE_ID}-mut-1": Verdict.FULL_COMPROMISE,
    f"{_STATIC_CASE_ID}-mut-2": Verdict.BLOCKED,
    f"{_STATIC_CASE_ID}-mut-3": Verdict.FULL_COMPROMISE,
}


class _CovModule:
    id = _MODULE_ID
    name = "Coverage Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    async def generate_cases(self, context):
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="cov parent payload", technique_id=_STATIC_TECHNIQUE_ID
        )

    async def evaluate(self, case: TestCase, response) -> EvalResult:
        verdict = _VERDICT_BY_CASE_ID.get(case.case_id, Verdict.BLOCKED)
        return EvalResult(
            case_id=case.case_id,
            verdict=verdict,
            confidence=0.9,
            evidence=f"evidence for {case.case_id}",
            detection_layer="regex",
        )


def _case_queue() -> list[QueuedCase]:
    return [
        QueuedCase(
            module_id=_MODULE_ID,
            case_id=_STATIC_CASE_ID,
            technique_id=_STATIC_TECHNIQUE_ID,
            prompt="cov parent payload",
            verdict="blocked",
            turns=None,
        )
    ]


def _static_results() -> list[tuple[str, EvalResult]]:
    return [
        (
            _MODULE_ID,
            EvalResult(
                case_id=_STATIC_CASE_ID,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            ),
        )
    ]


def _strategist_output() -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
        ordered_case_ids=[_STATIC_CASE_ID],
        escalate=False,
        reason_code=None,
        rationale="rationale",
    )


def _mutator_output() -> MutatorOutput:
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload=f"variant payload {i}",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale=f"rationale {i}",
            )
            for i in range(3)
        ]
    )


def _neutral_analyst_output() -> ObservedDefence:
    return ObservedDefence(
        refusal_style="unchanged",
        apparent_filter="none",
        what_moved="no change",
        technique_outcome="inconclusive",
        notes="",
    )


def _neutral_crescendo_output() -> CrescendoOutput:
    return CrescendoOutput(
        turns=["turn one", "turn two"], arc_rationale="unused", backtrack_from_turn=None, abort_recommended=False
    )


class _ReconciliationError(AssertionError):
    """Raised by `_reconcile_variants_against_audit()` on a mismatch --
    this test's own independent reconciliation check, never
    `DeepSummaryReconciliationError` (that one belongs to
    `compute_deep_summary()` itself and is exercised by
    `tests/attacker/test_deep_summary.py`, not here)."""


def _reconcile_variants_against_audit(
    lineage: dict[str, VariantRecord], audit_lines: list[dict[str, Any]]
) -> None:
    """The bidirectional orphan check (AT-3): every dispatched variant
    (a `lineage` key) has at least one corresponding `target_dispatch`
    audit line, and every `target_dispatch` audit line's `case_id` is a
    key present in `lineage` -- nothing orphaned in either direction.
    Written independently of `compute_deep_summary()`, which never reads
    the audit file at all."""
    dispatch_case_ids = {line["case_id"] for line in audit_lines if line["event"] == "target_dispatch"}
    lineage_case_ids = set(lineage.keys())
    missing_in_audit = lineage_case_ids - dispatch_case_ids
    orphaned_in_audit = dispatch_case_ids - lineage_case_ids
    if missing_in_audit or orphaned_in_audit:
        raise _ReconciliationError(
            f"reconciliation failed: dispatched variant(s) with no audit line "
            f"{sorted(missing_in_audit)}, audit line(s) with no matching dispatched "
            f"variant {sorted(orphaned_in_audit)}"
        )


async def _run_scripted_campaign(fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path):
    module = _CovModule()
    adapter = fake_target_adapter()
    for case_id in _VERDICT_BY_CASE_ID:
        adapter.queue_response(case_id, mock_target_response(case_id=case_id, raw_text="a reply"))

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=3)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model([_strategist_output()])
    mutator_model = scripted_chat_model([_mutator_output()])
    analyst_model = scripted_chat_model([_neutral_analyst_output()])
    crescendo_model = scripted_chat_model([_neutral_crescendo_output()])

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "analyst": build_analyst_agent(settings, cfg, model=analyst_model),
        "crescendo": build_crescendo_agent(settings, cfg, model=crescendo_model),
    }

    writer = AttackerAuditWriter(tmp_path, "scan-coverage-1")
    handler = AttackerAuditHandler(writer, "scan-coverage-1")
    handler.record_campaign_start(module_order=[_MODULE_ID])
    compiled = build_campaign_graph(
        roles=roles,
        adapter=adapter,
        modules={_MODULE_ID: module},
        max_concurrency=5,
        callbacks=[handler],
    )

    initial_state = new_campaign_state(
        scan_id="scan-coverage-1", settings=settings, module_order=[_MODULE_ID], case_queue=_case_queue()
    )
    initial_state["current_module"] = _MODULE_ID
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)
    writer.close()

    return final_state, writer.path, handler


def _findings_from_dispatch(final_state) -> list[Finding]:
    """Minimal, self-contained replica of `api.py`'s D-90 Finding-population
    loop -- only the fields `compute_deep_summary()` actually reads
    (`case_id`, `parent_case_id`, `severity`, `contributing_agent`), built
    directly from `final_state["dispatch_results"]` rather than by calling
    any attacker-summary code."""
    findings: list[Finding] = []
    for entry in final_state.get("dispatch_results", []):
        eval_result: EvalResult = entry["eval_result"]
        record: VariantRecord = entry["record"]
        severity = score(eval_result.verdict, eval_result.evidence)
        if severity is Severity.NONE:
            continue
        findings.append(
            Finding(
                case_id=entry["case_id"],
                technique_id=record["parent_technique_id"],
                verdict=eval_result.verdict,
                severity=severity.value,
                owasp_ref="LLM00:2025",
                evidence=eval_result.evidence,
                remediation="remediate",
                parent_case_id=record["parent_case_id"],
                parent_technique_id=record["parent_technique_id"],
                round=record["round"],
                contributing_agent=record["contributing_agent"],
            )
        )
    return findings


def _lineage_from_dispatch(final_state) -> dict[str, VariantRecord]:
    return {
        entry["case_id"]: entry["record"]
        for entry in final_state.get("dispatch_results", [])
        if entry.get("eval_result") is not None
    }


def _eval_results_from_dispatch(final_state) -> list[tuple[str, EvalResult]]:
    return [
        (_MODULE_ID, entry["eval_result"])
        for entry in final_state.get("dispatch_results", [])
        if entry.get("eval_result") is not None
    ]


def _read_audit_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def test_every_summary_figure_reconciles_against_audit_and_lineage_independently(
    fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
):
    final_state, audit_path, handler = await _run_scripted_campaign(
        fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
    )

    lineage = _lineage_from_dispatch(final_state)
    findings = _findings_from_dispatch(final_state)
    static_results = _static_results()

    campaign_result = CampaignResult(
        eval_results=_eval_results_from_dispatch(final_state),
        lineage=lineage,
        final_state=final_state,
        limitations=[],
        audit_path=audit_path,
    )

    # The value under test.
    reported = compute_deep_summary(findings, static_results, campaign_result)

    audit_lines = _read_audit_lines(audit_path)

    # --- AT-3: bidirectional orphan check (no dropped line here) ----------
    _reconcile_variants_against_audit(lineage, audit_lines)

    # --- AT-1: variants_dispatched, re-derived by counting audit lines ----
    dispatch_case_ids = {line["case_id"] for line in audit_lines if line["event"] == "target_dispatch"}
    variants_dispatched_independent = len(dispatch_case_ids)
    assert variants_dispatched_independent == reported.variants_dispatched, (
        f"variants_dispatched: reported={reported.variants_dispatched} "
        f"independent={variants_dispatched_independent}"
    )

    # --- AT-1: cases_attacked, re-derived by walking the lineage fields ---
    cases_attacked_independent = len({record["parent_case_id"] for record in lineage.values()})
    assert cases_attacked_independent == reported.cases_attacked, (
        f"cases_attacked: reported={reported.cases_attacked} independent={cases_attacked_independent}"
    )

    # --- AT-1: bypasses, re-derived via the SAME production score()/Severity
    # ordering but a FRESH walk over lineage/eval_results/static_results,
    # never calling compute_deep_summary() itself. -------------------------
    static_by_case_id = {eval_result.case_id: eval_result for _module_id, eval_result in static_results}
    eval_by_case_id = {ev.case_id: ev for _module_id, ev in _eval_results_from_dispatch(final_state)}
    severity_order = list(Severity)
    bypass_case_ids_independent: list[str] = []
    for case_id, record in lineage.items():
        parent_eval = static_by_case_id.get(record["parent_case_id"])
        if parent_eval is None:
            continue
        if parent_eval.verdict not in QUEUE_ELIGIBLE_VERDICTS:
            continue
        child_eval = eval_by_case_id[case_id]
        parent_severity = score(parent_eval.verdict, parent_eval.evidence)
        child_severity = score(child_eval.verdict, child_eval.evidence)
        if severity_order.index(child_severity) > severity_order.index(parent_severity):
            bypass_case_ids_independent.append(case_id)
            # Each reported bypass case id resolves through its lineage to a
            # specific static case present in the case log, and that
            # static case's own verdict was strictly less severe.
            assert record["parent_case_id"] in static_by_case_id
            assert severity_order.index(parent_severity) < severity_order.index(child_severity)

    assert sorted(bypass_case_ids_independent) == sorted(reported.bypass_case_ids), (
        f"bypass_case_ids: reported={sorted(reported.bypass_case_ids)} "
        f"independent={sorted(bypass_case_ids_independent)}"
    )
    assert len(bypass_case_ids_independent) == reported.bypasses_found

    # --- AT-1: spend/calls, re-derived directly from the final ledger -----
    ledger = final_state.get("budget_ledger") or {}
    assert ledger.get("spent_usd", 0.0) == reported.spend_usd
    assert ledger.get("agent_calls", 0) == reported.agent_calls

    # --- AT-3: audit line count equals the callback capture count ---------
    assert handler.capture_failures == 0
    assert handler.captured_events == handler.written_lines
    assert len(audit_lines) == handler.written_lines

    # --- AT-3: round ordering is non-decreasing across the whole file -----
    rounds = [line["round"] for line in audit_lines]
    assert rounds == sorted(rounds)


async def test_dropped_audit_line_fails_the_reconciliation_gate(
    fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
):
    """The negative control (AT-3's own requirement): a deliberately-dropped
    `target_dispatch` audit line makes `_reconcile_variants_against_audit()`
    fail -- proving the gate is not passing merely because it never actually
    inspects the file's contents."""
    final_state, audit_path, _handler = await _run_scripted_campaign(
        fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
    )
    lineage = _lineage_from_dispatch(final_state)
    audit_lines = _read_audit_lines(audit_path)

    # Sanity: the UNMODIFIED file reconciles cleanly first.
    _reconcile_variants_against_audit(lineage, audit_lines)

    dispatch_indices = [
        i for i, line in enumerate(audit_lines) if line["event"] == "target_dispatch"
    ]
    assert dispatch_indices, "expected at least one target_dispatch line to drop"
    corrupted_lines = list(audit_lines)
    del corrupted_lines[dispatch_indices[0]]

    with pytest.raises(_ReconciliationError):
        _reconcile_variants_against_audit(lineage, corrupted_lines)
