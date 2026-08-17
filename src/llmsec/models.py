"""Shared data-model vocabulary for llmsec.

This module defines the ONE shared verdict/data contract every later plan
(03-10) imports from: `Verdict`, `TestCase`, `TargetResponse`, `EvalResult`,
`Finding`, `ScanReport`, `ScanContext`. Nothing downstream should redefine
these types or introduce a second ad-hoc verdict scheme.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Multi-tier verdict (D-04) — never a binary pass/fail.

    Values byte-match AI-SPEC §3's `VerdictTier` string values exactly, since
    plan 06/07's judge integration reuses this exact enum.
    """

    BLOCKED = "blocked"
    PARTIAL_LEAK = "partial_leak"
    FULL_COMPROMISE = "full_compromise"
    UNCERTAIN = "uncertain"


# TransportMode records which transport actually produced a `TargetResponse`,
# so no downstream layer (module `evaluate()`, reporting) ever has to infer
# it from context (D-12: "every finding records which mode produced it").
#
# - "single": one ordinary request/response exchange.
# - "multi_turn_real": genuine dialogue — the target actually saw prior
#   turns (e.g. `LLMApiAdapter`'s accumulated `messages` list, or an
#   `HttpAppAdapter` with a configured session round-trip).
# - "multi_turn_concatenated": the degraded fallback where all turns were
#   flattened into a single request because the adapter cannot hold
#   conversation state (the `TargetAdapter.send_conversation()` ABC
#   default).
TransportMode = Literal["single", "multi_turn_real", "multi_turn_concatenated"]


class TestCase(BaseModel):
    """A single adversarial probe to send to a target."""

    case_id: str
    prompt: str
    technique_id: str
    system_prompt_override: str | None = None
    # Ordered user turns for a multi-turn sequence (crescendo, payload
    # splitting). When set, `prompt` still holds the pre-joined fallback
    # text so both real and degraded transport can consume the same
    # instance. `None` for an ordinary single-exchange case.
    turns: list[str] | None = None
    # --- Phase 5 (05-02): deep-mode lineage fields (D-90) ---------------
    # Populated ONLY by the deep-mode attacker layer when it emits a
    # mutated variant of a static payload; every Phase 1-4 module leaves
    # all four at their `None` default. The human-readable `DIRECT-003-mut-1`
    # `case_id` convention is preserved for readability, but no code path
    # may recover lineage by parsing that string -- these fields are the
    # only sanctioned source of truth. Optional-with-`None`-default is the
    # same additive shape `detection_layer` used in Phase 3 (see `EvalResult`
    # below), so no existing `TestCase(...)` construction site migrates.
    parent_case_id: str | None = None
    parent_technique_id: str | None = None
    round: int | None = None
    contributing_agent: str | None = None


class TargetResponse(BaseModel):
    """The raw response captured from a target for a given `TestCase`."""

    case_id: str
    raw_text: str
    status_code: int | None = None
    latency_ms: float
    tokens_used: int | None = None
    # Which transport mode actually produced this response (D-12).
    transport_mode: TransportMode | None = None
    # The target's reply per turn, ascending turn order, truncated at an
    # early abort. `None` for an ordinary single exchange.
    turn_replies: list[str] | None = None


class EvalResult(BaseModel):
    """The outcome of evaluating a `TargetResponse` against a `TestCase`."""

    case_id: str
    verdict: Verdict
    # Bounded to match `JudgeVerdict.confidence` (IN-05, 02-REVIEW.md) — every
    # detection tier (canary, refusal, judge) writes into this field, so an
    # out-of-[0,1]-range value from a third-party module or a future tier
    # should be rejected at the model layer, not merely by convention.
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str
    # Widened (Phase 3, Pitfall 1) from ["regex", "judge"] to also accept the
    # NER and canary-exact-match tiers `pii_exfiltration` introduces. Every
    # existing "regex"/"judge"-only caller continues to validate unchanged.
    # Widened again (Phase 6, 06-01, MOD-06): the new audit member is the
    # standalone-audit tier `BaseModule.run_standalone_audit()` results
    # carry -- no reporter branches on this value (rendered as plain text).
    # Widened again (Phase 7, 07-01, MOD-08/MOD-09): the new threshold member
    # is the deterministic token/latency-comparison tier
    # `unbounded_consumption` uses for both its `evaluate()` and
    # `run_direct_probe()` paths -- like "audit", `reporting/templates/
    # report.md.j2` renders this as plain text and branches on nothing, so
    # this widening needs no reporter change.
    detection_layer: Literal["regex", "judge", "ner", "canary", "audit", "threshold"]
    transport_mode: TransportMode | None = None
    # Module-supplied, technique-specific remediation that overrides the
    # generic verdict-keyed default at report time (D-26). Optional so
    # Phase 1's module continues to fall through to the generic text.
    remediation: str | None = None


class Finding(BaseModel):
    """A scored, reportable vulnerability derived from one or more `EvalResult`s."""

    case_id: str
    technique_id: str
    verdict: Verdict
    severity: str
    owasp_ref: str
    evidence: str
    remediation: str
    transport_mode: TransportMode | None = None
    # NEW (Phase 3, Pitfall 1): optional/defaulted so Phase 1/2's existing
    # `Finding(...)` construction site in `api.py` needs no mandatory-field
    # migration. Threads `EvalResult.detection_layer` through to the
    # persisted/reported artifact (SC#3) — populated for every module's
    # findings, not just `pii_exfiltration`'s, since `eval_result.
    # detection_layer` already exists on every `EvalResult`.
    # Widened (Phase 6, 06-01, MOD-06) to also accept the audit tier -- see
    # `EvalResult.detection_layer`'s comment above; no reporter change
    # needed for this widening.
    # Widened again (Phase 7, 07-01, MOD-08/MOD-09) to also accept the
    # threshold tier -- see `EvalResult.detection_layer`'s comment above.
    detection_layer: Literal["regex", "judge", "ner", "canary", "audit", "threshold"] | None = (
        None
    )
    # --- Phase 5 (05-02): deep-mode lineage fields (D-90) ---------------
    # Same shape/discipline as `TestCase`'s lineage fields above: populated
    # ONLY by the deep-mode attacker layer, every field optional and
    # `None`-defaulted so `api.py`'s existing `Finding(...)` construction
    # site is untouched, and lineage must never be recovered by
    # string-parsing `case_id` -- these fields are the sole source of truth.
    parent_case_id: str | None = None
    parent_technique_id: str | None = None
    round: int | None = None
    contributing_agent: str | None = None


class RoleActivity(BaseModel):
    """One attacker-team role's counted activity within a deep-mode
    campaign (D-91, `DeepModeSummary.per_role_activity`).

    Every field is a counted event read off the campaign's final
    `BudgetLedger`/`Finding` lineage -- never an estimate. `bypasses` is
    attributed via `Finding.contributing_agent`, the sole sanctioned
    lineage field for that purpose (D-90) -- never by parsing a `case_id`.
    """

    calls: int
    spend_usd: float
    bypasses: int


class DeepModeSummary(BaseModel):
    """The deep-mode coverage-delta summary block (D-91), rendered by both
    reporters when a `--deep` campaign ran.

    Every field here is a COUNTED event derived from the campaign's
    lineage map and its final `BudgetLedger` -- `attacker/summary.py`'s
    `compute_deep_summary()` is the ONE place that ever populates one of
    these, and it is documented there as never taking a total from one
    report and subtracting another (AT-1). This is the same
    additive-optional-field convention `EvalResult.detection_layer`
    established in Phase 3: `ScanReport.deep_summary` below defaults to
    `None`, so an older persisted report still loads and a static-only
    run's report omits this block entirely rather than emitting an empty
    one (D-93).

    `bypasses_found` MUST equal `len(bypass_case_ids)` for every instance
    -- the count and its own walkable list are the same fact expressed
    twice, never two independently-computed numbers that could disagree.
    `cost_per_bypass_usd` is `None` exactly when `bypasses_found == 0` --
    never a rendered `0.0` or an infinity -- so a costly campaign that
    found nothing is reported honestly rather than misleadingly cheap.
    """

    cases_attacked: int
    rounds_run: int
    variants_dispatched: int
    bypasses_found: int
    bypass_case_ids: list[str]
    spend_usd: float
    cost_per_bypass_usd: float | None
    agent_calls: int
    per_role_activity: dict[str, RoleActivity]
    termination_reason: str | None
    constraint_violations: int
    abandoned_arcs: int
    # AT-6 (D-94, 05-11 Rule 1/2 fix): the count of genuine structured-
    # output-retry exhaustions across any of the five roles -- distinct
    # from `constraint_violations` (D-95 allowlist refusals only) and from
    # `abandoned_arcs` (a strategic Crescendo abort, not a structural
    # failure). Populated from `CampaignResult.role_structural_failures`.
    role_structural_failures: int
    truncated: bool
    audit_log_path: str | None


class ScanReport(BaseModel):
    """The final artifact produced by a scan run."""

    scan_id: str
    target_summary: str
    module_ids: list[str]
    findings: list[Finding]
    case_log: list[EvalResult]
    started_at: str
    completed_at: str
    # Human-readable statements of what this scan did and did not actually
    # test, carried in both the JSON and Markdown reports.
    limitations: list[str] = Field(default_factory=list)
    # Phase 5 (05-09, D-91): populated ONLY when a `--deep` campaign
    # actually ran and returned (`api.run_scan()`'s deep-mode branch).
    # `None` for every static-only (`--quick`) run, and for every report
    # persisted before this field existed (additive default, never a
    # migration for existing construction sites or persisted JSON).
    deep_summary: DeepModeSummary | None = None


class ScanContext(BaseModel):
    """Shared context threaded through a scan (known system prompt, judge model).

    IN-01 (02-REVIEW.md): only `system_prompt_controllable` is actually read
    back off a `ScanContext` instance today (by
    `PromptInjectionModule.generate_cases()`, D-17's canary planting-mode
    decision). `known_system_prompt`/`judge_model`/`judge_api_key_env` are
    populated here for observability/API-shape reasons, but the values
    modules actually receive are threaded through `PluginRegistry.
    load_allowed()`'s separate `module_config` kwarg path (see
    `api.run_scan()`) — not read off this object. `supports_multi_turn`
    reflects the constructed adapter's real capability (asserted by
    `tests/test_api.py`'s capability-wiring tests) but has no current reader;
    the orchestrator's single-turn-vs-sequence dispatch decision is driven
    purely by whether the generated `TestCase.turns` is non-empty, never by
    this flag. Kept as an honest capability signal for future
    modules/dispatch logic rather than removed, since an adapter's real
    multi-turn capability is exactly the kind of fact a scan context should
    carry -- but it is not yet wired into a decision point.
    """

    known_system_prompt: str | None = None
    judge_model: str
    judge_api_key_env: str
    # Whether the configured target lets the framework set the system
    # prompt, which decides D-17's canary planting mode.
    system_prompt_controllable: bool = False
    # Whether the configured adapter can hold conversation state. Not yet
    # read by any orchestrator/module decision point (IN-01) -- see the
    # class docstring above.
    supports_multi_turn: bool = False
