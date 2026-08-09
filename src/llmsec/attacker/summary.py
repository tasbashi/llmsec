"""`compute_deep_summary()` -- turns one completed deep-mode campaign into a
`DeepModeSummary` (D-91): every field a COUNTED event read off the
campaign's lineage map, its final `BudgetLedger`, and the report's own
findings/static-results lists -- never an estimate, a rounded value, or a
diff of two report totals (AT-1, `05-AI-SPEC.md` §5).

The PRE-RUN cost estimate (`attacker/budget.py`'s `CostEstimate`/
`estimate_campaign_cost()`) is a DIFFERENT, explicitly-labelled number and
must never be substituted for this module's `spend_usd`/
`cost_per_bypass_usd` -- one is a forecast made before a dollar was spent,
the other is the ledger's own post-run count of what actually was.

This module performs ZERO imports of the `langchain`/`langgraph`/
`deepagents` stack (it only reads `CampaignResult`'s already-resolved
dataclass fields), so it stays importable without the `[deep]` extra
installed, exactly like `attacker/state.py` and `attacker/runner.py`'s own
module-scope imports.
"""

from __future__ import annotations

from llmsec.attacker.runner import CampaignResult
from llmsec.attacker.state import ROLE_NAMES, QUEUE_ELIGIBLE_VERDICTS, BudgetLedger, RoleSpend
from llmsec.models import DeepModeSummary, EvalResult, Finding, RoleActivity
from llmsec.scoring.engine import Severity, score

#: Fixed severity-band ordering, taken directly from `scoring.engine.Severity`'s
#: own declaration order (NONE < LOW < MEDIUM < HIGH < CRITICAL) -- this is
#: the "existing scoring engine's mapping" the plan requires reusing, never a
#: second, independently-maintained ordering.
_SEVERITY_ORDER: list[Severity] = list(Severity)


class DeepSummaryReconciliationError(Exception):
    """Raised by `compute_deep_summary()` when per-role spend/calls do not
    sum to the campaign's own final ledger totals.

    A figure that does not reconcile is exactly the failure this module
    exists to prevent (T-05-09-01/AT-1) -- raised rather than emitting a
    summary whose numbers cannot be trusted. A silently wrong figure is
    worse than a loud failure.
    """


def _severity_rank(severity: Severity) -> int:
    return _SEVERITY_ORDER.index(severity)


def compute_deep_summary(
    findings: list[Finding],
    static_results: list[tuple[str, EvalResult]],
    campaign_result: CampaignResult,
) -> DeepModeSummary:
    """Derive `DeepModeSummary` from `campaign_result`'s lineage map and
    final ledger, plus `findings`/`static_results` for bypass attribution
    -- and nothing else, so no figure can originate from an estimate or
    from diffing two report totals (AT-1).

    `static_results` MUST be the run's ORIGINAL (`--quick`-equivalent)
    results -- captured by the caller (`api.run_scan()`) BEFORE
    `campaign_result.eval_results` was appended to it -- since this
    function has no other way to recover a parent case's own pre-attack
    verdict/evidence (a `blocked` parent never produces a `Finding` at
    all, `Severity.NONE` is filtered out of `findings` upstream, so
    `findings` alone cannot answer "was this parent queue-eligible and
    what was its own severity band").

    A deep `Finding` (one with `parent_case_id` set) counts as a bypass
    only when ALL of the following hold, exactly as `<behavior>` states:
      1. `parent_case_id` resolves to a case actually present in
         `static_results`.
      2. That parent case's own verdict was in `QUEUE_ELIGIBLE_VERDICTS`
         (D-77) -- defensive; every case reaching the deep-mode queue
         already satisfied this, but this function verifies it
         independently rather than trusting the caller.
      3. The deep finding's severity band is STRICTLY higher than the
         parent's own severity band (via `scoring.engine.score()`, never
         a second ordering) -- a mutated variant that only reproduced the
         parent's own result is not a bypass.

    Raises `DeepSummaryReconciliationError` if the per-role spend/call
    sums do not reconcile against the campaign's final ledger totals.
    """
    lineage = campaign_result.lineage
    final_state = campaign_result.final_state
    ledger: BudgetLedger = final_state.get("budget_ledger") or BudgetLedger(
        cap_usd=0.0,
        warn_usd=0.0,
        spent_usd=0.0,
        attacker_spent_usd=0.0,
        target_spent_usd=0.0,
        agent_calls=0,
        agent_call_ceiling=0,
        per_role={},
        truncated=False,
        overshoot_rounds=0,
        warn_approved=False,
        unpriced_calls=0,
    )

    # Parent lookup, built ONCE from the ORIGINAL static results -- the
    # only source of a parent case's own pre-attack verdict/evidence.
    parent_eval_by_case_id: dict[str, EvalResult] = {
        eval_result.case_id: eval_result for _module_id, eval_result in static_results
    }

    # --- The walkable bypass list, built FIRST -- `bypasses_found` is
    # derived from its length below, so the count and the list can never
    # disagree (the same fact expressed twice, never two independently
    # computed numbers). ------------------------------------------------
    bypass_case_ids: list[str] = []
    bypass_contributor_by_case_id: dict[str, str | None] = {}
    for finding in findings:
        if finding.parent_case_id is None:
            continue  # a static finding, not a deep-mode variant at all
        parent_eval = parent_eval_by_case_id.get(finding.parent_case_id)
        if parent_eval is None:
            continue  # cannot resolve to a specific static case -- not counted
        if parent_eval.verdict not in QUEUE_ELIGIBLE_VERDICTS:
            continue  # defensive: only a queue-eligible parent can be bypassed (D-77)
        parent_severity = score(parent_eval.verdict, parent_eval.evidence)
        child_severity = Severity(finding.severity)
        if _severity_rank(child_severity) <= _severity_rank(parent_severity):
            continue  # reproduced (or regressed from) the parent's own result
        bypass_case_ids.append(finding.case_id)
        bypass_contributor_by_case_id[finding.case_id] = finding.contributing_agent

    bypasses_found = len(bypass_case_ids)

    spend_usd = ledger.get("spent_usd", 0.0)
    cost_per_bypass_usd = spend_usd / bypasses_found if bypasses_found > 0 else None

    # --- Per-role activity, reconciled against the ledger's own totals --
    per_role_activity: dict[str, RoleActivity] = {}
    for role in ROLE_NAMES:
        role_spend: RoleSpend = ledger.get("per_role", {}).get(
            role, RoleSpend(calls=0, usd=0.0, share_ceiling_usd=None)
        )
        role_bypasses = sum(
            1 for case_id in bypass_case_ids if bypass_contributor_by_case_id.get(case_id) == role
        )
        per_role_activity[role] = RoleActivity(
            calls=role_spend.get("calls", 0),
            spend_usd=role_spend.get("usd", 0.0),
            bypasses=role_bypasses,
        )

    reconciled_calls = sum(activity.calls for activity in per_role_activity.values())
    reconciled_spend = sum(activity.spend_usd for activity in per_role_activity.values())
    ledger_calls = ledger.get("agent_calls", 0)
    ledger_attacker_spend = ledger.get("attacker_spent_usd", 0.0)
    if reconciled_calls != ledger_calls:
        raise DeepSummaryReconciliationError(
            f"per-role call sum ({reconciled_calls}) does not reconcile against "
            f"the ledger's agent_calls ({ledger_calls}) -- refusing to emit a "
            "summary whose figures cannot be trusted."
        )
    if abs(reconciled_spend - ledger_attacker_spend) > 1e-9:
        raise DeepSummaryReconciliationError(
            f"per-role spend sum (${reconciled_spend:.6f}) does not reconcile against "
            f"the ledger's attacker_spent_usd (${ledger_attacker_spend:.6f}) -- refusing "
            "to emit a summary whose figures cannot be trusted."
        )

    cases_attacked = len({record["parent_case_id"] for record in lineage.values()})
    variants_dispatched = len(lineage)

    return DeepModeSummary(
        cases_attacked=cases_attacked,
        rounds_run=final_state.get("round", 0),
        variants_dispatched=variants_dispatched,
        bypasses_found=bypasses_found,
        bypass_case_ids=bypass_case_ids,
        spend_usd=spend_usd,
        cost_per_bypass_usd=cost_per_bypass_usd,
        agent_calls=ledger_calls,
        per_role_activity=per_role_activity,
        termination_reason=final_state.get("termination_reason"),
        constraint_violations=campaign_result.constraint_violations,
        abandoned_arcs=campaign_result.abandoned_arcs,
        role_structural_failures=campaign_result.role_structural_failures,
        truncated=bool(ledger.get("truncated", False)),
        audit_log_path=str(campaign_result.audit_path) if campaign_result.audit_path else None,
    )
