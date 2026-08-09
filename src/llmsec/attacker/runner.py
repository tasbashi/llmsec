"""`run_attacker_campaign()` -- the deep-mode campaign entry point (D-93
additive layer).

Called by `api.run_scan()` strictly AFTER `ScanOrchestrator.run()` returns
(D-93) -- `require_deep_extra()` is the FIRST thing this coroutine does,
before any spend or import of the `langchain`/`langgraph`/`deepagents`
stack. Every module-scope import in this file is deep-extra-free
(`attacker.config`/`attacker.state`/`models`/`adapters.base`/
`plugins.base`, none of which touch the optional stack); everything that
DOES touch it (`attacker.graph`, `attacker.roles.*`) is imported inside
this function's body, after the gate has already cleared.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import typer

from llmsec.adapters.base import TargetAdapter
from llmsec.attacker import require_deep_extra
from llmsec.attacker.config import DEFAULT_ENABLED_TECHNIQUES, resolve_settings, resolve_role_model
from llmsec.attacker.state import (
    BudgetLedger,
    CampaignState,
    QUEUE_ELIGIBLE_VERDICTS,
    QueuedCase,
    VariantRecord,
    new_campaign_state,
)
from llmsec.config import ScanConfig
from llmsec.models import EvalResult, ScanContext, TestCase
from llmsec.plugins.base import BaseModule

logger = logging.getLogger(__name__)


class UnknownScanIdError(Exception):
    """`--resume` named a `scan_id` with no matching checkpoint (D-75) --
    either no `checkpoint_dir` was ever configured for it, or nothing was
    ever persisted under that id. `cli.py` catches this and prints a clean,
    non-zero-exit message, never a traceback."""


class ConfigFingerprintMismatchError(Exception):
    """`--resume`'s freshly-resolved configuration does not match the
    checkpointed campaign's stored `config_fingerprint` (D-75.2) -- a hard
    refusal, never a soft warning that proceeds anyway. `cli.py` catches
    this and prints a clean, non-zero-exit message, never a traceback."""

#: 05-04/D-82: a non-interactive terminal is NEVER implicit consent for
#: continued spend past the warn threshold -- mirrors `auth_gate.py`'s own
#: D-02 discipline exactly, and lives here (not only in `cli.py`) so a
#: library caller (`import llmsec; await llmsec.run_scan(...)`) gets the
#: identical guarantee a CLI user gets.
def _resolve_budget_approval(interrupt_payloads: Any) -> bool:
    if not sys.stdin.isatty():
        logger.warning(
            "--deep campaign crossed its warn threshold in a non-interactive "
            "session; refusing further spend rather than assuming approval."
        )
        return False
    payload: dict[str, Any] = {}
    if interrupt_payloads:
        payload = getattr(interrupt_payloads[0], "value", {}) or {}
    typer.echo(
        "\n--deep campaign crossed its warn threshold: spent "
        f"${payload.get('spent_usd', 0.0):.2f} of ${payload.get('cap_usd', 0.0):.2f} cap "
        f"(${payload.get('remaining_usd', 0.0):.2f} remaining).",
        err=True,
    )
    return typer.confirm("Continue spending?")


@dataclass
class CampaignResult:
    """The campaign's output, returned to `api.run_scan()`.

    `lineage` is keyed by generated `case_id` (the readable
    `{parent_case_id}-mut-{n}` form) -- the ONLY sanctioned source of D-90
    lineage; no consumer may recover it by parsing that string.
    """

    eval_results: list[tuple[str, EvalResult]] = field(default_factory=list)
    lineage: dict[str, VariantRecord] = field(default_factory=dict)
    final_state: CampaignState = field(default_factory=lambda: CampaignState())
    limitations: list[str] = field(default_factory=list)
    #: 05-06 Task 1 (D-84/D-85): the redacted `{scan_id}-attacker-audit.jsonl`
    #: artifact's path, so `api.py` and the reporters can point an operator
    #: at the file -- `None` only when no campaign graph was ever compiled
    #: (the empty-`case_queue` early-return path below, which invokes no
    #: role and therefore has nothing to audit).
    audit_path: Path | None = None
    #: 05-08/D-95: the count of D-95 allowlist refusals this campaign
    #: recorded (`final_state["constraint_violations"]`'s length) -- exposed
    #: here so it can be reported and trended, never left buried in
    #: `final_state` alone.
    constraint_violations: int = 0
    #: 05-08/D-76: the count of Crescendo arcs the Crescendo Orchestrator
    #: itself recommended aborting, never dispatched
    #: (`final_state["abandoned_arcs"]`'s length).
    abandoned_arcs: int = 0
    #: AT-6 (D-94, 05-11 Rule 1/2 fix): the count of genuine structured-
    #: output-retry exhaustions across ANY of the five roles
    #: (`final_state["role_structural_failures"]`'s length) -- distinct
    #: from `constraint_violations` (D-95 allowlist refusals only) and from
    #: `abandoned_arcs` (a strategic Crescendo abort, not a structural
    #: failure).
    role_structural_failures: int = 0


async def _rebuild_case_by_id(
    module: BaseModule, context: ScanContext
) -> dict[str, TestCase]:
    """Regenerate `module`'s deterministic static case set, keyed by
    `case_id`, so the deep-mode work queue can recover the parent
    `TestCase.prompt` text a queued case needs to be mutated.

    `ScanOrchestrator.run()` (D-93, never touched) returns only
    `(module_id, EvalResult)` pairs -- `EvalResult` carries no prompt text
    at all, only the target's response (`evidence`). Regenerating via
    `generate_cases()` a second time on the SAME module instance is safe:
    both opted-in modules (`prompt_injection`, `pii_exfiltration`) document
    `generate_cases()` as deterministic/idempotent per instance --
    `prompt_injection.py` uses a fixed `CANARY_TOKEN` constant and
    `pii_exfiltration.py` lazily caches its per-instance `CanaryPiiSet` on
    first call, reusing it on every subsequent call -- so this never
    dispatches anything to the target and never diverges from what was
    actually sent during the static batch.
    """
    cases: dict[str, TestCase] = {}
    # mypy misreads BaseModule.generate_cases()'s abstract (no-`yield`)
    # stub as returning a plain Coroutine rather than an async generator --
    # a pre-existing, codebase-wide quirk of the ABC's signature also
    # present at orchestrator.py's own identical `async for case in
    # module.generate_cases(context):` call site (never touched, D-93), not
    # something this plan introduces or can fix without editing
    # `plugins/base.py` (out of scope here).
    async for case in module.generate_cases(context):  # type: ignore[attr-defined]
        cases[case.case_id] = case
    return cases


def _collect_dispatch_results(
    state: CampaignState,
) -> tuple[list[tuple[str, EvalResult]], dict[str, VariantRecord]]:
    """Flatten `state["dispatch_results"]` into the `(module_id,
    EvalResult)` pairs and D-90 lineage map `CampaignResult` carries --
    shared by `run_attacker_campaign()` and `resume_attacker_campaign()`
    (05-06 Task 3) so there is exactly one place this projection is
    written down. `entry.get("eval_result")` is always populated by every
    `dispatch_variants_node` exit branch in normal operation (success,
    adapter-failure-degrade, evaluate-failure-degrade all construct a real
    `EvalResult`); the `is None` guard exists only for a resumed campaign's
    already-at-cap early return, whose `restored_state["dispatch_results"]`
    entries are otherwise identical in shape.
    """
    eval_results: list[tuple[str, EvalResult]] = []
    lineage: dict[str, VariantRecord] = {}
    for entry in state.get("dispatch_results", []):
        if entry.get("eval_result") is None:
            continue
        eval_results.append((entry["module_id"], entry["eval_result"]))
        lineage[entry["case_id"]] = entry["record"]
    return eval_results, lineage


def _count_constraint_violations(state: CampaignState) -> int:
    """05-08/D-95: the number of allowlist refusals recorded in `state`,
    exposed on `CampaignResult` so it can be reported and trended."""
    return len(state.get("constraint_violations") or [])


def _count_abandoned_arcs(state: CampaignState) -> int:
    """05-08/D-76: the number of Crescendo arcs recommended for abort and
    never dispatched, recorded in `state`."""
    return len(state.get("abandoned_arcs") or [])


def _count_role_structural_failures(state: CampaignState) -> int:
    """AT-6 (D-94, 05-11 Rule 1/2 fix): the number of genuine structured-
    output-retry exhaustions across any role, recorded in `state`. Mirrors
    `_count_constraint_violations()`/`_count_abandoned_arcs()`'s own shape
    exactly."""
    return len(state.get("role_structural_failures") or [])


def _bounded_loss_disclosure(state: CampaignState) -> str | None:
    """D-75.4: a bounded-loss disclosure, present exactly when the RESTORED
    ledger's recorded spend is inconsistent with the restored round count.

    05-07 update: every completed round makes THREE attacker-side calls
    in the common case (Strategist + Mutator-or-Crescendo + Analyst --
    05-08's Crescendo Orchestrator REPLACES the Mutator on the escalation
    path rather than adding a fourth call, D-65, so this formula did not
    need revisiting again once it shipped), plus Recon's own ONE call
    amortized once per campaign, before round 1 (D-65). The original 05-06
    formula (`2 * round_count`, Strategist+Mutator only) predates the
    Analyst/Recon roles this plan adds and is corrected here to a base of
    `3 * round_count + 1` -- `ledger["agent_calls"]` is incremented once
    per call via `record_agent_spend()`.

    WR-02 fix: THREE calls per round is only the common case, not a
    guarantee. A round that hit the D-95 allowlist refusal
    (`strategist_node`'s `TechniqueNotAllowed` branch, recorded into
    `state["constraint_violations"]`) still bumps `state["round"]`
    (`dispatch_variants_node` always increments it, even with zero
    variants) but makes only TWO real attacker calls that round
    (Strategist + Analyst -- the Mutator/Crescendo `.ainvoke()` is never
    called on a refused round, see `test_technique_allowlist.py`'s own
    `test_disallowed_technique_refused_zero_mutator_invocations_zero_
    dispatches`). Subtracting one call per recorded refusal from the
    naive 3-per-round baseline tightens the expectation to the true
    round-accurate minimum, so a legitimately refused round never trips
    this disclosure as a false positive. (The Strategist's OWN
    structured-output-retry exhaustion is not accounted for here because
    it terminates the campaign immediately, before `dispatch_variants_node`
    ever bumps `state["round"]` -- that round is never counted in
    `round_count` in the first place, so it cannot make this formula
    overshoot either.)

    If the restored `round` counter (net of refused rounds) implies at
    least this many calls happened, but the restored ledger's
    `agent_calls` is LOWER than that, a crash landed between an LLM call
    returning (and being billed) and the checkpoint write capturing that
    spend -- bounded at one round's worth by construction (the checkpoint
    granularity this phase's topology commits at), and disclosed here
    rather than silently absorbed. Undercount is the safe direction
    (D-75.4): this never claims MORE spend happened than the ledger shows.
    """
    ledger = state.get("budget_ledger")
    round_count = state.get("round", 0)
    if ledger is None or round_count <= 0:
        return None
    refused_rounds = _count_constraint_violations(state)
    expected_min_calls = max(0, 3 * round_count + 1 - refused_rounds)
    if ledger.get("agent_calls", 0) < expected_min_calls:
        return (
            "Resumed campaign: the restored budget ledger's recorded spend is lower "
            "than the spend implied by the restored round count -- up to one round of "
            "spend may be unrecorded (a crash between an LLM call returning and the "
            "checkpoint write capturing it). Undercount is the safe direction; this is "
            "disclosed, never silently absorbed."
        )
    return None


async def run_attacker_campaign(
    *,
    config: ScanConfig,
    adapter: TargetAdapter,
    modules: dict[str, BaseModule],
    static_results: list[tuple[str, EvalResult]],
    scan_id: str,
) -> CampaignResult:
    """Run one deep-mode campaign and return its `CampaignResult`.

    Never handed `config.target.api_key_env`'s resolved value,
    `config.known_system_prompt`'s privileged origin beyond what
    `ScanContext` already exposes to every module, or the raw `ScanConfig`
    object itself, into any role brief or tool closure (D-87) -- only
    `CampaignState` crosses into a role's brief, via `AgentRole.brief()`.
    """
    require_deep_extra()  # D-74: before ANY spend or import of the attacker stack

    # Deferred imports: everything below touches langchain/langgraph/
    # deepagents and must never be imported before the gate above clears.
    from langgraph.types import Command

    from llmsec.attacker.audit import AttackerAuditHandler, AttackerAuditWriter
    from llmsec.attacker.budget import truncation_disclosure
    from llmsec.attacker.checkpoint import build_checkpointer, config_fingerprint
    from llmsec.attacker.graph import build_campaign_graph
    from llmsec.attacker.roles import get_role
    import llmsec.attacker.roles.analyst  # noqa: F401 -- registers "analyst"
    import llmsec.attacker.roles.crescendo  # noqa: F401 -- registers "crescendo"
    import llmsec.attacker.roles.mutator  # noqa: F401 -- registers "mutator"
    import llmsec.attacker.roles.recon  # noqa: F401 -- registers "recon"
    import llmsec.attacker.roles.strategist  # noqa: F401 -- registers "strategist"

    attacker_cfg = config.attacker
    if attacker_cfg is None:  # defensive -- api.py only calls this when set
        raise RuntimeError("run_attacker_campaign() called without config.attacker set")
    settings = resolve_settings(attacker_cfg)

    context = ScanContext(
        known_system_prompt=config.known_system_prompt,
        judge_model=config.judge_model,
        judge_api_key_env=config.judge_api_key_env or "",
        system_prompt_controllable=adapter.supports_system_prompt_override,
        supports_multi_turn=adapter.supports_multi_turn,
    )

    # D-77 + ATK-01: only opted-in modules ever enter the work queue.
    # D-78: fixed `sorted()` order, never set/dict iteration.
    eligible_module_ids = sorted(
        module_id for module_id, module in modules.items() if module.uses_attacker_llm
    )

    case_by_module_and_id: dict[str, dict[str, TestCase]] = {
        module_id: await _rebuild_case_by_id(modules[module_id], context)
        for module_id in eligible_module_ids
    }

    case_queue: list[QueuedCase] = []
    for module_id, eval_result in static_results:
        if module_id not in case_by_module_and_id:
            continue
        if eval_result.verdict not in QUEUE_ELIGIBLE_VERDICTS:  # D-77
            continue
        original_case = case_by_module_and_id[module_id].get(eval_result.case_id)
        if original_case is None:
            continue
        case_queue.append(
            QueuedCase(
                module_id=module_id,
                case_id=original_case.case_id,
                technique_id=original_case.technique_id,
                prompt=original_case.prompt,
                verdict=eval_result.verdict.value,
                turns=list(original_case.turns) if original_case.turns else None,
            )
        )

    # D-81: per-role share ceilings, sourced from AttackerConfig.roles[*]
    # .budget_share -- a role absent here (or with budget_share unset)
    # keeps share_ceiling_usd=None (no per-role ceiling configured).
    role_shares: dict[str, float] = {
        role: override.budget_share
        for role, override in attacker_cfg.roles.items()
        if override.budget_share is not None
    }
    state: CampaignState = new_campaign_state(
        scan_id, settings, eligible_module_ids, case_queue, role_shares
    )
    state["enabled_techniques"] = list(
        attacker_cfg.enabled_techniques
        if attacker_cfg.enabled_techniques is not None
        else DEFAULT_ENABLED_TECHNIQUES
    )
    if eligible_module_ids:
        state["current_module"] = eligible_module_ids[0]
    # 05-06 Task 2/D-75.2: stamped at campaign start (never left unset) so
    # a LATER `--resume` (05-06 Task 3) always has a fingerprint to compare
    # a freshly recomputed one against, refusing hard on any cap/model/
    # module/profile mismatch rather than warning and proceeding.
    state["config_fingerprint"] = config_fingerprint(config, settings)

    if not case_queue:
        # Nothing eligible this campaign -- terminate cleanly without
        # compiling/invoking a graph that has no work to do.
        state["termination_reason"] = "TECHNIQUES_EXHAUSTED"
        return CampaignResult(eval_results=[], lineage={}, final_state=state, limitations=[])

    strategist_role = get_role("strategist")
    mutator_role = get_role("mutator")
    analyst_role = get_role("analyst")
    recon_role = get_role("recon")
    crescendo_role = get_role("crescendo")
    # 05-07: the full three-role-per-round core (Strategist/Mutator-or-
    # Crescendo/Analyst) plus Recon, amortized once per scan -- Recon is
    # the only role needing `adapter=` (its target probe tool closes over
    # it, T-05-07-01/D-87). Every round is now the three-role core the
    # cost model assumes (D-65). 05-08: Crescendo is built unconditionally
    # alongside Mutator, since the escalation flag is a per-round Strategist
    # decision (`_post_strategist_edge`), not something knowable up front.
    roles: dict[str, Any] = {
        "strategist": strategist_role.build(settings, attacker_cfg),
        "mutator": mutator_role.build(settings, attacker_cfg),
        "analyst": analyst_role.build(settings, attacker_cfg),
        "crescendo": crescendo_role.build(settings, attacker_cfg),
        # Recon is the only role whose `.build()` departs from the
        # `AgentRole` protocol's own `(settings, cfg, *, model=None)`
        # shape -- `adapter=` is required (`roles/recon.py`'s own
        # docstring), which the Protocol type itself does not declare.
        "recon": recon_role.build(settings, attacker_cfg, adapter=adapter),  # type: ignore[call-arg]
    }
    # 05-04: a model NAME STRING per role for attacker/budget.py's pricing
    # lookup -- resolved via the SAME `resolve_role_model()` helper
    # `roles/__init__.py`'s `build_role_chat_model()` already uses, so this
    # never re-derives per-role model resolution a second way. Recon's
    # spend is therefore recorded against its OWN `"recon"` role entry in
    # the ledger (`_record_role_call_spend()`, `graph.py`'s `recon_node`),
    # exactly like every other role, never folded into another role's
    # total or left unpriced by omission.
    role_models: dict[str, str] = {
        "strategist": resolve_role_model(attacker_cfg, "strategist"),
        "mutator": resolve_role_model(attacker_cfg, "mutator"),
        "analyst": resolve_role_model(attacker_cfg, "analyst"),
        "recon": resolve_role_model(attacker_cfg, "recon"),
        "crescendo": resolve_role_model(attacker_cfg, "crescendo"),
    }

    thread_config = {"configurable": {"thread_id": scan_id}}

    # 05-06 Task 1 (D-84/D-85/D-86): one `AttackerAuditHandler` instance,
    # shared across the whole campaign, writing to the SAME redacted
    # `{scan_id}-attacker-audit.jsonl` artifact `audit.py` defines --
    # constructed only once we know a graph will actually be compiled
    # (the empty-`case_queue` early return above never gets here, so no
    # audit file is created for a campaign that invoked no role at all).
    # Closed in a `finally` block mirroring `api.run_scan()`'s own
    # `finally: await adapter.close()` discipline, so a mid-campaign
    # exception still leaves a well-formed, fully-flushed file on disk.
    writer = AttackerAuditWriter(Path(config.output_dir), scan_id)
    handler = AttackerAuditHandler(writer, scan_id)
    # D-68: the deterministic module iteration order, stamped as ONE
    # dedicated audit line before any role is invoked -- two runs of the
    # identical configuration must produce the identical recorded value.
    handler.record_campaign_start(module_order=eligible_module_ids)

    # 05-06 Task 2 (D-73 mitigation 2/D-75): `build_checkpointer()` yields
    # either a redacting disk-backed `AsyncSqliteSaver` (when
    # `settings.checkpoint_dir` is configured) or the framework's own
    # `MemorySaver()` -- an async context manager because the disk-backed
    # case owns a live `aiosqlite.Connection` that must be closed on the
    # way out, exactly like `writer`'s own `finally: writer.close()` below.
    try:
        async with build_checkpointer(settings) as checkpointer:
            compiled = build_campaign_graph(
                roles=roles,
                adapter=adapter,
                modules=modules,
                max_concurrency=config.max_concurrency,
                role_models=role_models,
                checkpointer=checkpointer,
                callbacks=[handler],
            )

            final_state: CampaignState = await compiled.ainvoke(state, config=thread_config)
            # `__interrupt__` is a LangGraph-injected key never declared on
            # `CampaignState` itself (it only ever appears in the raw `ainvoke()`
            # return dict when a node paused) -- read via a plain-`dict` view so
            # mypy's TypedDict key-checking does not apply to this one lookup.
            raw_final_state: dict[str, Any] = final_state  # type: ignore[assignment]
            if raw_final_state.get("__interrupt__"):
                # D-82: the warn threshold was crossed. `_resolve_budget_approval()`
                # mirrors `auth_gate.py`'s own D-02 discipline (a non-interactive
                # terminal is NEVER implicit consent) -- resolved here, in the
                # library layer, so `import llmsec; await llmsec.run_scan(...)`
                # gets the identical guarantee a CLI user gets, exactly like
                # `confirm_authorization()` already does for the initial gate.
                approved = _resolve_budget_approval(raw_final_state["__interrupt__"])
                final_state = await compiled.ainvoke(Command(resume=approved), config=thread_config)
    finally:
        writer.close()

    eval_results, lineage = _collect_dispatch_results(final_state)

    limitations: list[str] = []
    ledger = final_state.get("budget_ledger")
    if ledger is not None:
        disclosure = truncation_disclosure(ledger)
        if disclosure is not None:
            limitations.append(disclosure)

    return CampaignResult(
        eval_results=eval_results,
        lineage=lineage,
        final_state=final_state,
        limitations=limitations,
        audit_path=writer.path,
        constraint_violations=_count_constraint_violations(final_state),
        abandoned_arcs=_count_abandoned_arcs(final_state),
        role_structural_failures=_count_role_structural_failures(final_state),
    )


async def resume_attacker_campaign(
    *,
    config: ScanConfig,
    adapter: TargetAdapter,
    modules: dict[str, BaseModule],
    scan_id: str,
    budget_top_up_usd: float | None = None,
    on_prior_spend: Callable[[float, float, float], None] | None = None,
) -> CampaignResult:
    """`--resume`'s own entry point (D-75) -- deliberately a SEPARATE
    coroutine from `run_attacker_campaign()`'s fresh-campaign path, not a
    `resume: bool` branch inside it: a resumed campaign needs none of the
    static-batch/case-queue construction `run_attacker_campaign()` does
    (that state is ALREADY checkpointed under `scan_id`), and needs several
    checks a fresh campaign never performs (fingerprint refusal,
    already-at-cap immediate termination, per-dispatch idempotency).

    `on_prior_spend`, when given, is called EXACTLY ONCE with
    `(prior_spent, original_cap, effective_cap)` -- after the fingerprint
    check passes, before ANY further spend decision (including the
    already-at-cap early-return branch below) -- so D-75.1's "prior spend
    is printed before the new ceiling takes effect" holds by construction
    regardless of what `cli.py` does with the callback, rather than relying
    on caller discipline to sequence two separate calls correctly.

    Raises `UnknownScanIdError` when `scan_id` has no matching checkpoint
    (including when no `checkpoint_dir` was ever configured, so nothing
    could ever have been persisted for it), and
    `ConfigFingerprintMismatchError` when the freshly-resolved
    configuration does not match what was checkpointed (D-75.2) -- both
    caught by `cli.py` and surfaced as a clean, non-zero-exit message.
    """
    require_deep_extra()  # D-74: before ANY spend or import of the attacker stack

    # Deferred imports -- see `run_attacker_campaign()`'s own module
    # docstring: everything below touches langchain/langgraph/deepagents
    # and must never be imported before the gate above clears.
    from langgraph.types import Command

    from llmsec.attacker.audit import AttackerAuditHandler, AttackerAuditWriter
    from llmsec.attacker.budget import truncation_disclosure
    from llmsec.attacker.checkpoint import build_checkpointer, config_fingerprint, idempotency_key
    from llmsec.attacker.graph import build_campaign_graph
    from llmsec.attacker.roles import get_role
    import llmsec.attacker.roles.analyst  # noqa: F401 -- registers "analyst"
    import llmsec.attacker.roles.crescendo  # noqa: F401 -- registers "crescendo"
    import llmsec.attacker.roles.mutator  # noqa: F401 -- registers "mutator"
    import llmsec.attacker.roles.strategist  # noqa: F401 -- registers "strategist"

    # Deliberately NOT `import llmsec.attacker.roles.recon` here: `recon`
    # is reachable ONLY via the graph's `START` edge (`graph.py`'s module
    # docstring), and `.ainvoke(None, ...)` below resumes from wherever the
    # checkpoint left off, never re-entering at `START` -- a resumed
    # campaign can therefore never reach `recon_node` again, and building
    # a Recon agent it will never invoke would be pure waste.

    attacker_cfg = config.attacker
    if attacker_cfg is None:  # defensive -- cli.py only calls this when set
        raise RuntimeError("resume_attacker_campaign() called without config.attacker set")
    settings = resolve_settings(attacker_cfg)

    if not settings.checkpoint_dir:
        raise UnknownScanIdError(
            f"Cannot resume scan_id={scan_id!r}: no checkpoint_dir is configured for "
            "this campaign, so no durable state could ever have been persisted for it."
        )

    thread_config = {"configurable": {"thread_id": scan_id}}

    async with build_checkpointer(settings) as checkpointer:
        checkpoint_tuple = await checkpointer.aget_tuple(thread_config)  # type: ignore[arg-type]
        if checkpoint_tuple is None or not checkpoint_tuple.checkpoint.get("channel_values"):
            raise UnknownScanIdError(
                f"No checkpointed campaign found for scan_id={scan_id!r} under "
                f"{settings.checkpoint_dir!r}."
            )
        restored_state: CampaignState = checkpoint_tuple.checkpoint[  # type: ignore[assignment]
            "channel_values"
        ]

        # D-75.2: a config-fingerprint mismatch is a HARD refusal, never a
        # soft warning that proceeds anyway -- resuming after a cap/model/
        # module/profile change would produce a report claiming a
        # configuration that never ran.
        fresh_fingerprint = config_fingerprint(config, settings)
        restored_fingerprint = restored_state.get("config_fingerprint")
        if restored_fingerprint != fresh_fingerprint:
            raise ConfigFingerprintMismatchError(
                f"Cannot resume scan_id={scan_id!r}: the resolved attacker configuration "
                "(cap, call ceiling, rounds, variants-per-round, model, enabled "
                "techniques, profile, or enabled-module list) has changed since this "
                "campaign was checkpointed. Re-run without --resume, or restore the "
                "original configuration this campaign started with."
            )

        ledger: BudgetLedger = dict(restored_state.get("budget_ledger") or {})  # type: ignore[assignment]
        prior_spent = ledger.get("spent_usd", 0.0)
        original_cap = ledger.get("cap_usd", settings.budget_usd)
        # D-75.1: `--resume` NEVER silently tops up the budget -- the
        # ORIGINAL checkpointed cap is the effective one unless the
        # operator explicitly passed a top-up value.
        effective_cap = budget_top_up_usd if budget_top_up_usd is not None else original_cap

        if on_prior_spend is not None:
            on_prior_spend(prior_spent, original_cap, effective_cap)

        # D-75.4: computed from the RESTORED state, before any further
        # spend this resume might make -- always reflects what the
        # CHECKPOINT showed, never masked by round 2+'s own fresh spend.
        bounded_loss_note = _bounded_loss_disclosure(restored_state)

        writer = AttackerAuditWriter(Path(config.output_dir), scan_id)
        handler = AttackerAuditHandler(writer, scan_id)

        try:
            if prior_spent >= effective_cap:
                # D-75.1/D-83: a campaign that had already spent at or
                # above its (possibly topped-up) cap terminates
                # IMMEDIATELY -- no further spend, no graph invocation at
                # all, so "produces no further spend" holds trivially.
                restored_state["termination_reason"] = "BUDGET_CAP_EXCEEDED"
                eval_results, lineage = _collect_dispatch_results(restored_state)
                limitations: list[str] = []
                disclosure = truncation_disclosure(ledger)
                if disclosure is not None:
                    limitations.append(disclosure)
                if bounded_loss_note is not None:
                    limitations.append(bounded_loss_note)
                return CampaignResult(
                    eval_results=eval_results,
                    lineage=lineage,
                    final_state=restored_state,
                    limitations=limitations,
                    audit_path=writer.path,
                    constraint_violations=_count_constraint_violations(restored_state),
                    abandoned_arcs=_count_abandoned_arcs(restored_state),
                    role_structural_failures=_count_role_structural_failures(restored_state),
                )

            # D-75.3: every `(case, round, variant)` triple already
            # recorded as dispatched in the RESTORED state, looked up by
            # `idempotency_key()` -- `dispatch_variants_node` (graph.py)
            # skips re-dispatching (and re-paying for) any of these.
            already_dispatched = frozenset(
                idempotency_key(
                    entry["record"]["parent_case_id"],
                    entry["record"]["round"],
                    entry["record"]["variant_index"],
                )
                for entry in restored_state.get("dispatch_results", [])
            )

            strategist_role = get_role("strategist")
            mutator_role = get_role("mutator")
            analyst_role = get_role("analyst")
            crescendo_role = get_role("crescendo")
            # No `recon_role` here -- see the module-docstring-adjacent
            # comment above this function's deferred-import block: `recon`
            # can never be reached by a resumed campaign. `crescendo` IS
            # built here (05-08) -- unlike Recon, a resumed campaign can
            # perfectly well continue into a round on the escalation path.
            roles: dict[str, Any] = {
                "strategist": strategist_role.build(settings, attacker_cfg),
                "mutator": mutator_role.build(settings, attacker_cfg),
                "analyst": analyst_role.build(settings, attacker_cfg),
                "crescendo": crescendo_role.build(settings, attacker_cfg),
            }
            role_models: dict[str, str] = {
                "strategist": resolve_role_model(attacker_cfg, "strategist"),
                "mutator": resolve_role_model(attacker_cfg, "mutator"),
                "analyst": resolve_role_model(attacker_cfg, "analyst"),
                "crescendo": resolve_role_model(attacker_cfg, "crescendo"),
            }

            compiled = build_campaign_graph(
                roles=roles,
                adapter=adapter,
                modules=modules,
                max_concurrency=config.max_concurrency,
                role_models=role_models,
                checkpointer=checkpointer,
                callbacks=[handler],
                resume_dispatched_keys=already_dispatched,
            )

            if budget_top_up_usd is not None:
                # Persist the topped-up cap into the checkpointed ledger
                # BEFORE resuming, so `budget_check_node`'s own read of
                # `state["budget_ledger"]["cap_usd"]` sees the new ceiling
                # from the very next superstep onward.
                ledger["cap_usd"] = effective_cap
                await compiled.aupdate_state(thread_config, {"budget_ledger": ledger})

            # `None` as input tells LangGraph "resume from the latest
            # checkpoint for this thread_id" rather than starting a fresh
            # run -- the graph continues from whichever node comes next
            # after the one the restored checkpoint last completed.
            final_state: CampaignState = await compiled.ainvoke(None, config=thread_config)
            raw_final_state: dict[str, Any] = final_state  # type: ignore[assignment]
            if raw_final_state.get("__interrupt__"):
                approved = _resolve_budget_approval(raw_final_state["__interrupt__"])
                final_state = await compiled.ainvoke(Command(resume=approved), config=thread_config)
        finally:
            writer.close()

    eval_results, lineage = _collect_dispatch_results(final_state)
    limitations = []
    final_ledger = final_state.get("budget_ledger")
    if final_ledger is not None:
        disclosure = truncation_disclosure(final_ledger)
        if disclosure is not None:
            limitations.append(disclosure)
    if bounded_loss_note is not None:
        limitations.append(bounded_loss_note)

    return CampaignResult(
        eval_results=eval_results,
        lineage=lineage,
        final_state=final_state,
        limitations=limitations,
        audit_path=writer.path,
        constraint_violations=_count_constraint_violations(final_state),
        abandoned_arcs=_count_abandoned_arcs(final_state),
        role_structural_failures=_count_role_structural_failures(final_state),
    )
