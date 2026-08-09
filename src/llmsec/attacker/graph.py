"""`build_campaign_graph()` -- round topology, conditional termination edges
(D-70, D-73).

Nodes: `recon` (05-07), `strategist`, `mutator`, `crescendo` (05-08),
`dispatch_variants`, `analyst` (05-07), `budget_check`, `budget_approval`
(05-04), and a terminal `finalize`.

`analyst_node`/`recon_node` are backward-compatible NO-OPS (`return {}`)
when the caller's `roles` dict has no `"analyst"`/`"recon"` entry -- every
pre-05-07 call site (and every pre-05-07 test) that only supplies
`strategist`/`mutator` continues to route straight through both unchanged.

`validate_technique` (D-95) is the hard technique-allowlist gate function,
called from EXACTLY ONE place: inside `strategist_node`, immediately after the
Strategist's structured output is read and before either mutation role is
ever constructed or invoked. A refusal is recorded as a constraint
violation and never propagates out of the node -- `_post_strategist_edge`
(below) reads `state["selected_technique"] is None` to route a refused
round straight to `dispatch_variants` with zero variants, bypassing both
`mutator` and `crescendo` entirely, while the round is still genuinely
consumed (T-05-08-05).

Topology:
    START -> recon -> strategist
        -[_post_strategist_edge]-> {
            "mutator": mutator (D-95 gate cleared, not on escalation path),
            "crescendo": crescendo (D-95 gate cleared, escalation path -- D-65
                REPLACES mutator, never runs alongside it),
            "dispatch_variants": dispatch_variants (D-95 gate refused this
                round's technique -- zero mutation-role invocations),
        }
    mutator -> dispatch_variants
    crescendo -> dispatch_variants
    dispatch_variants -> analyst -> budget_check
        -[_post_budget_check_edge]-> {
            "finalize": finalize (hard cap/call-ceiling tripped, D-73 mit. 1),
            "budget_approval": budget_approval (warn threshold crossed, D-82),
            round_cap_edge's own {"strategist": strategist, "finalize": finalize},
        }
    budget_approval -[_post_budget_approval_edge]-> {
        "finalize": finalize (operator refused),
        round_cap_edge's own {"strategist": strategist, "finalize": finalize},
    }
    finalize -> END

`dispatch_variants_node` routes a turn-carrying `VariantRecord` (one whose
`"turns"` field is non-empty -- only ever produced by `crescendo_node`)
through the adapter's `send_conversation()` entry point, and everything
else through `send()` -- the SAME `TestCase.turns`-non-empty rule
`orchestrator.py`'s own dispatch uses (D-93, never re-derived differently
here), with the module's optional sequence-abort hook resolved via the
module-level `_resolve_stop_when()` below, reproducing
`ScanOrchestrator._resolve_stop_when()`'s exact shape without importing
from that file.

`recon` is reachable ONLY via the `START` edge -- no other edge in this
graph ever targets it -- so a single `.ainvoke()` call (fresh, or
`--resume`d since a resume never re-enters at `START`) invokes it at most
once regardless of how many rounds follow (D-65).

`budget_check` sits immediately AFTER `dispatch_variants` -- deliberately,
not before `mutator` -- because D-83's "nothing bought is thrown away"
semantic depends on it: a round's `dispatch_variants_node` has ALWAYS
already run (unconditionally) by the time `budget_check` can see that
round's own spend, so a cap trip discovered here can never discard variants
already generated and dispatched this round; it only ever prevents the
NEXT round's `strategist`/`mutator` calls. This reproduces exactly the
Wave 0 spike's own proven shape (05-RESEARCH `## Code Examples` --
`budget_check_node` setting `over_budget`, consumed by the SAME
`round_cap_edge`-shaped decision that already gates round continuation),
rather than the strategist->mutator insertion point 05-03's placeholder
comment anticipated -- see 05-04-SUMMARY.md's Decisions Made for the full
rationale.

`round_cap_edge` itself is UNCHANGED from 05-03: it is evaluated once per
round, after `dispatch_variants_node` has already bumped `state["round"]`,
and returns `"finalize"` once `state["round"] >= state["max_rounds"]` or
the Strategist emitted a reason code this round (D-70/D-72) -- round
control is topology, never agent discretion. It is no longer wired
DIRECTLY off `dispatch_variants`; `_post_budget_check_edge`/
`_post_budget_approval_edge` call it as a plain function once the budget
layer has cleared, so 05-04 never needed to touch its own logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from llmsec.adapters.base import TargetAdapter
from llmsec.attacker.audit import AttackerAuditHandler, redact_audit_text
from llmsec.attacker.budget import (
    attacker_call_cost,
    budget_approval_node,
    budget_check_node,
    mark_truncated,
    over_budget_edge,
    record_agent_spend,
    record_target_spend,
    target_call_cost,
)
from llmsec.attacker.checkpoint import idempotency_key
from llmsec.attacker.memory import (
    mark_partial_movement,
    mark_technique_dead,
    remember_refusal_signature,
)
from llmsec.attacker.roles import get_role
from llmsec.attacker.roles._structured_retry import (
    MAX_STRUCTURED_OUTPUT_RETRIES,
    StructuredOutputFailure,
    invoke_role_with_retry,
)
from llmsec.attacker.roles.analyst import ObservedDefence
from llmsec.attacker.roles.crescendo import CrescendoOutput
from llmsec.attacker.roles.mutator import MutatorOutput
from llmsec.attacker.roles.recon import ReconOutput
from llmsec.attacker.roles.strategist import StrategistOutput
from llmsec.attacker.state import (
    BudgetLedger,
    CampaignMemory,
    CampaignState,
    QueuedCase,
    RoleSpend,
    VariantRecord,
    new_campaign_memory,
)
from llmsec.models import EvalResult, TestCase, Verdict
from llmsec.payloads.schema import PiiAttackVector, TechniqueFamily
from llmsec.plugins.base import BaseModule

logger = logging.getLogger(__name__)

#: D-95 allowlist gate: the closed vocabulary a Strategist-selected
#: technique must belong to, sourced ONLY from the existing closed
#: `TechniqueFamily`/`PiiAttackVector` enums (`payloads/schema.py`) --
#: never re-declared as a free-form string set, mirroring
#: `roles/mutator.py`'s own `_VALID_TECHNIQUE_FAMILIES` computation
#: exactly (a typo or hallucinated family name can never widen this set,
#: because it is derived from the enums, not hand-maintained).
_CLOSED_TECHNIQUE_VOCABULARY: frozenset[str] = frozenset(
    f.value for f in TechniqueFamily
) | frozenset(v.value for v in PiiAttackVector)


class TechniqueNotAllowed(ValueError):
    """D-95: raised when a Strategist-selected technique is not a member of
    BOTH the closed `TechniqueFamily`/`PiiAttackVector` vocabulary AND the
    campaign's configured `enabled_techniques` set.

    Raised and caught entirely INSIDE `strategist_node` -- it never
    propagates out of a graph node. The gate refuses the dispatch (this
    round routes straight to `dispatch_variants` with zero variants,
    recorded as a constraint violation), it never aborts the campaign
    (05-AI-SPEC AT-2 rubric).
    """


def validate_technique(selected: str, enabled: list[str] | tuple[str, ...]) -> str:
    """The D-95 hard allowlist gate, enforced at the ONE point a technique
    can enter dispatch -- the transition out of the Strategist node, before
    either mutation role (`mutator`/`crescendo`) is ever constructed or
    invoked. Mirrors `plugins/registry.py`'s `load_allowed()`: enforce at
    the boundary, never trust the caller.

    Returns `selected` unchanged when it is a member of BOTH the closed
    `TechniqueFamily`/`PiiAttackVector` vocabulary (`payloads/schema.py`)
    AND the campaign's configured `enabled` set. Checks the closed
    vocabulary FIRST so a typo or hallucinated family name present in a
    misconfigured `enabled_techniques` list can never widen the accepted
    vocabulary beyond the enums that are its source of record. Raises
    `TechniqueNotAllowed` otherwise -- never silently substitutes or
    coerces a near-miss value.
    """
    if selected not in _CLOSED_TECHNIQUE_VOCABULARY:
        raise TechniqueNotAllowed(
            f"technique {selected!r} is not a member of the closed "
            "TechniqueFamily/PiiAttackVector vocabulary (payloads/schema.py)"
        )
    if selected not in set(enabled):
        raise TechniqueNotAllowed(
            f"technique {selected!r} is not in this campaign's enabled_techniques set"
        )
    return selected


def _resolve_stop_when(module: BaseModule | None, case: TestCase) -> Callable[[str], bool] | None:
    """05-08/D-76: resolve `module`'s optional sequence-abort hook by plain
    attribute lookup -- reproducing `orchestrator.py`'s own
    `ScanOrchestrator._resolve_stop_when()` resolution shape EXACTLY
    (duck-typed `getattr`, never an ABC method; the returned callable binds
    `case` and swallows any exception the module's predicate raises,
    treating it as "do not abort"). Copied here rather than imported from
    `orchestrator.py` because that module stays byte-for-byte unchanged
    (D-93) and this file must never import from it. `module=None` (an
    unknown `module_id`) degrades to `stop_when=None`, identical to "the
    hook does not exist".
    """
    if module is None:
        return None
    predicate = getattr(module, "should_abort_sequence", None)
    if predicate is None:
        return None

    def _stop_when(turn_reply: str) -> bool:
        try:
            return bool(predicate(case, turn_reply))
        except Exception:  # noqa: BLE001 -- a misbehaving predicate must never abort dispatch
            return False

    return _stop_when


def _extract_usage_metadata(result: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort extraction of the invocation's final `AIMessage.usage_metadata`
    from `invoke_role_with_retry()`'s `on_success` callback payload
    (`result["messages"]`) -- `None` when absent (a scripted/offline test
    model, or a provider that does not report usage), in which case
    `attacker_call_cost()` already returns `0.0` rather than guessing.
    """
    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return None
    for message in reversed(messages):
        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            return dict(usage_metadata)
    return None


def _audit_handlers(callbacks: list[Any] | None) -> list[AttackerAuditHandler]:
    """Filter `callbacks` (the SAME list threaded into `graph_config`) for
    `AttackerAuditHandler` instances -- 05-06 Task 1's wiring point.

    `callbacks` is deliberately typed `list[Any] | None` at
    `build_campaign_graph()`'s own boundary (any `BaseCallbackHandler` is a
    legal entry, and other future callback consumers may be added), so a
    node that needs to call the audit handler's OWN explicit methods
    (`set_context()`/`record_inter_agent()`/`record_target_dispatch()` --
    none of which are ordinary LangChain callback hooks) must first pick it
    back out by type. Returns `[]` (never raises) when `callbacks` is
    `None`/empty or contains no `AttackerAuditHandler` -- 05-RESEARCH's own
    `<behavior>` requirement that "a campaign run with no audit handler
    supplied still completes" depends on every call site below being a
    no-op over an empty list, never a KeyError/AttributeError.
    """
    if not callbacks:
        return []
    return [cb for cb in callbacks if isinstance(cb, AttackerAuditHandler)]


def _raw_output_text(raw: Any) -> str:
    """Best-effort projection of `invoke_role_with_retry()`'s `on_attempt`
    callback's `raw` payload into a human-readable string for the AT-6
    structural-failure audit line (05-11 Rule 1/2 fix).

    `raw` is `None` for every attempt that raised a real
    `pydantic.ValidationError`/`StructuredOutputValidationError`
    (`_structured_retry.py`'s own except-branch never has a raw payload to
    hand back for that case) -- degrades to `""` rather than `"None"`. When
    `raw` IS the `{"messages": [...]}`-shaped dict `_structured_retry.py`'s
    "structured_response missing" branch supplies, this flattens every
    message's `.content` into one string, mirroring `_flatten_chat_messages()`
    (`audit.py`)'s own tolerant `getattr(message, "content", "")` shape --
    never assumes a specific `BaseMessage` subclass. Any other shape is
    stringified directly. This text is ALWAYS target-influenced-content-shaped
    in the worst case (the model's own raw completion), so every call site
    passes it through `redact_audit_text()` before it reaches an audit line.
    """
    if raw is None:
        return ""
    if isinstance(raw, dict):
        messages = raw.get("messages")
        if messages:
            parts = [str(getattr(message, "content", "") or "") for message in messages]
            return "\n".join(part for part in parts if part)
        return str(raw)
    return str(raw)


def _make_attempt_capture(
    role_name: str, attempts: list[tuple[int, str, Any]]
) -> Callable[[int, str, Any], None]:
    """Factory for a per-node `on_attempt` closure that BOTH logs (mirroring
    every node's pre-existing `logger.warning(...)` call) AND accumulates
    `(attempt, violation, raw)` into the node's own local `attempts` list, so
    the node's `except StructuredOutputFailure` block has the FULL retry
    history to build an AT-6 audit line from, not just `str(exc)` (05-11
    Rule 1/2 fix -- `StructuredOutputFailure` itself carries only the last
    violation text, never the raw output that produced it)."""

    def _capture(attempt: int, violation: str, raw: Any) -> None:
        attempts.append((attempt, violation, raw))
        logger.warning("%s structured-output attempt %d failed: %s", role_name, attempt, violation)

    return _capture


def _record_structural_failure(
    *,
    role_name: str,
    state: CampaignState,
    attempts: list[tuple[int, str, Any]],
    exc: StructuredOutputFailure,
    audit_handlers: list[AttackerAuditHandler],
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    """D-94 AT-6 (05-11 Rule 1/2 fix): record one role's genuine structured-
    output exhaustion into BOTH `state["role_structural_failures"]` (never
    `constraint_violations` -- that field is reserved for D-95 allowlist
    refusals only, see `state.py`) and the audit trail, mirroring the D-95
    refusal block's own `handler.record_inter_agent(...)` calling
    convention exactly (`from_role`, `to_role`, `content`, `round`,
    `module_id`, `case_id`).

    `05-AI-SPEC.md` AT-6's PASS condition: "a 3rd consecutive failure logs
    the attempt count, schema violation, and redacted raw output to the
    audit trail". `attempts` is the node's own locally-accumulated
    `on_attempt` history; the LAST attempt's raw output is what gets
    redacted into the audit line. Defensive fallback (empty `attempts`,
    never expected in practice since `invoke_role_with_retry()` always
    calls `on_attempt` at least once before raising) uses `exc`'s own text
    and `MAX_STRUCTURED_OUTPUT_RETRIES + 1` so this never crashes even if a
    future refactor of the retry loop changes that invariant.

    Returns the UPDATED `role_structural_failures` list -- the caller
    includes it verbatim in its own node-return dict, exactly like every
    other `state[...]`-derived list this file threads through (mirrors
    `constraint_violations`/`abandoned_arcs`'s own read-append-return
    shape).
    """
    attempt_count = len(attempts) if attempts else MAX_STRUCTURED_OUTPUT_RETRIES + 1
    last_violation = attempts[-1][1] if attempts else str(exc)
    raw_text = _raw_output_text(attempts[-1][2]) if attempts else ""

    failures = list(state.get("role_structural_failures", []))
    failures.append(
        {
            "role": role_name,
            "round": state.get("round", 0),
            "attempt_count": attempt_count,
            "reason": last_violation,
        }
    )

    # D-86 defense in depth: explicitly redacted here (the raw-output
    # segment specifically), even though `AttackerAuditHandler._write_line()`
    # ALSO redacts the whole assembled `content` string unconditionally
    # before writing -- two independent passes over the same no-exemption
    # chokepoint, never a single point of failure for a target-influenced
    # raw completion reaching disk unredacted.
    redacted_raw = redact_audit_text(raw_text)
    content = (
        f"STRUCTURAL FAILURE role={role_name} attempts={attempt_count} "
        f"schema_violation={last_violation} raw_output={redacted_raw}"
    )
    for handler in audit_handlers:
        handler.record_inter_agent(
            from_role=role_name,
            to_role="campaign",
            content=content,
            round=state.get("round", 0),
            module_id=state.get("current_module"),
            case_id=case_id,
        )
    return failures


def build_campaign_graph(
    *,
    roles: dict[str, Any],
    adapter: TargetAdapter,
    modules: dict[str, BaseModule],
    max_concurrency: int,
    role_models: dict[str, str] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    callbacks: list[Any] | None = None,
    resume_dispatched_keys: frozenset[str] | None = None,
) -> Any:
    """Compile the campaign `StateGraph`.

    `roles` maps role name -> an ALREADY-`.build()`-called compiled agent
    (never a raw `AgentRole`/factory) -- callers resolve
    `get_role(name).build(settings, cfg)` themselves so this function never
    re-derives model resolution. `role_models` maps role name -> the
    resolved model NAME STRING for that role (05-04's `attacker_call_cost()`
    needs a string to look up its price table by; the compiled agent object
    in `roles` does not reliably retain it once built) -- `None`/a missing
    key degrades to `0.0`-priced spend for that role, never a crash.
    `callbacks`, when given, is threaded into every role invocation's
    `config` -- 05-05's audit handler attaches here without editing this
    function's signature. 05-06 Task 1: every role node ALSO explicitly
    forwards `config=graph_config` into its own `invoke_role_with_retry()`
    call (05-RESEARCH Pitfall 6 -- ambient contextvar callback propagation
    is real but must never be the only path an audit callback reaches a
    nested role invocation) and, via `_audit_handlers(callbacks)`, updates
    the audit handler's current-context fields immediately before invoking
    and records the strategist->mutator handoff (`direction="inter_agent"`)
    and every dispatched variant (`direction="target"`, both on success and
    on the contained-failure path) explicitly -- neither of these is an
    ordinary LangChain callback hook, so both are graph-topology-driven
    calls, not callback dispatch.

    `resume_dispatched_keys` (05-06 Task 3/D-75.3): a set of
    `checkpoint.idempotency_key(parent_case_id, round, variant_index)`
    strings already present in a RESTORED campaign's `dispatch_results` --
    `dispatch_variants_node` consults it to skip re-dispatching (and
    re-paying for) a variant whose triple is already accounted for, the
    defensive guard against 05-RESEARCH Pitfall 3's "a node re-executes
    from the top on resume" gotcha reaching `dispatch_variants_node`
    specifically. `None`/empty (every non-resume call site) is a pure
    no-op -- every variant in `state["variants"]` is dispatched exactly as
    before.
    """
    graph_config: dict[str, Any] | None = {"callbacks": list(callbacks)} if callbacks else None
    resolved_role_models: dict[str, str] = dict(role_models or {})

    def _record_role_call_spend(state: CampaignState, role_name: str, raw_result: dict[str, Any]) -> BudgetLedger:
        """05-04/D-80: fold this role call's attacker-side cost into the
        campaign ledger, returning the MUTATED ledger so the caller can
        include it verbatim in its own node-return dict -- the ledger
        update is therefore a node return value visible in graph state,
        never a side-channel mutation.

        WR-01: `dict(state.get("budget_ledger") or {})` only copies the
        TOP-LEVEL keys -- `ledger["per_role"]` would otherwise still be
        the SAME nested dict object reachable from the previously
        committed `CampaignState["budget_ledger"]` value.
        `record_agent_spend()` (`budget.py`) does
        `per_role.setdefault(role, ...)` then mutates the returned
        `RoleSpend` dict IN PLACE via `+=` -- for every role after its
        first call, that `setdefault` returns the EXISTING dict by
        reference, so without this explicit re-copy the mutation would
        also corrupt an earlier superstep's already-checkpointed ledger
        snapshot (an in-process `get_state_history()`/`MemorySaver`
        reference kept elsewhere would observe LATER rounds' spend
        bleeding into an EARLIER round's `per_role` entry). One level
        deeper than `dict(...)` reaches is exactly as far as this
        function's own mutations go, so a shallow per-entry re-copy of
        `per_role` (never a full deep-copy of the whole ledger) is
        sufficient and keeps this cheap."""
        ledger: BudgetLedger = dict(state.get("budget_ledger") or {})  # type: ignore[assignment]
        ledger["per_role"] = {
            role: RoleSpend(**spend) for role, spend in ledger.get("per_role", {}).items()
        }
        usage_metadata = _extract_usage_metadata(raw_result)
        model = resolved_role_models.get(role_name, "")
        cost = attacker_call_cost(usage_metadata, model, ledger=ledger)
        record_agent_spend(ledger, role=role_name, usd=cost, calls=1)
        return ledger

    async def strategist_node(state: CampaignState) -> dict[str, Any]:
        role = get_role("strategist")
        agent = roles["strategist"]
        brief = role.brief(state)
        audit_handlers = _audit_handlers(callbacks)
        for handler in audit_handlers:
            handler.set_context(
                round=state.get("round", 0), module_id=state.get("current_module"), role="strategist"
            )
        call_result_holder: dict[str, Any] = {}
        attempts: list[tuple[int, str, Any]] = []
        try:
            output: StrategistOutput = await invoke_role_with_retry(
                agent,
                [("user", brief)],
                role="strategist",
                config=graph_config,
                on_attempt=_make_attempt_capture("strategist", attempts),
                on_success=call_result_holder.update,
            )
        except StructuredOutputFailure as exc:
            # A recorded structural failure (D-94 AT-6) -- terminate this
            # campaign cleanly rather than let the exception cancel the
            # whole graph run. 05-11 Rule 1/2 fix: also recorded to the
            # audit trail (`role_structural_failures`, never
            # `constraint_violations`) -- no case was selected yet this
            # round, so no `case_id` is passed.
            logger.error("Strategist failed to produce structured output: %s", exc)
            failures = _record_structural_failure(
                role_name="strategist",
                state=state,
                attempts=attempts,
                exc=exc,
                audit_handlers=audit_handlers,
            )
            return {
                "reason_code": None,
                "termination_reason": "TECHNIQUES_EXHAUSTED",
                "role_structural_failures": failures,
            }

        module_id = state.get("current_module")
        module_cases = [
            case for case in state.get("case_queue", []) if case["module_id"] == module_id
        ]
        cases_by_id: dict[str, QueuedCase] = {case["case_id"]: case for case in module_cases}
        current_case: QueuedCase | None = None
        for case_id in output.ordered_case_ids:
            if case_id in cases_by_id:
                current_case = cases_by_id[case_id]
                break
        if current_case is None and module_cases:
            current_case = module_cases[0]

        selected_case_id = current_case.get("case_id") if current_case else None
        ledger = _record_role_call_spend(state, "strategist", call_result_holder)

        # D-95: the ONE call site for the allowlist gate -- runs BEFORE
        # either mutation role (`mutator`/`crescendo`) is ever constructed
        # or invoked. A refusal never propagates out of this node: it is
        # recorded as a constraint violation and the round proceeds to
        # `dispatch_variants` with zero variants (routed there by
        # `_post_strategist_edge` below reading `selected_technique is
        # None`), so a Strategist stuck emitting disallowed techniques
        # still terminates at the round cap rather than looping
        # (T-05-08-05) -- never abort-the-whole-campaign (AT-2 rubric).
        enabled_techniques = state.get("enabled_techniques", [])
        try:
            validate_technique(output.technique, enabled_techniques)
        except TechniqueNotAllowed as exc:
            violations = list(state.get("constraint_violations", []))
            violations.append(
                {
                    "round": state.get("round", 0),
                    "case_id": selected_case_id,
                    "technique": output.technique,
                    "enabled_techniques": list(enabled_techniques),
                    "reason": str(exc),
                }
            )
            for handler in audit_handlers:
                handler.record_inter_agent(
                    from_role="strategist",
                    to_role="mutator",
                    content=f"REFUSED technique={output.technique!r}: {exc}",
                    round=state.get("round", 0),
                    module_id=module_id,
                    case_id=selected_case_id,
                )
            return {
                "selected_technique": None,
                "reason_code": output.reason_code,
                "current_case": current_case,
                "variants": [],
                "escalation_path": False,
                "constraint_violations": violations,
                "budget_ledger": ledger,
            }

        # D-84: the strategist->mutator (or ->crescendo, D-65/D-76) handoff
        # IS the attack reasoning in this team -- recorded as a dedicated
        # `direction="inter_agent"` line, not just implicit in the pair of
        # model_start/model_end lines the callback hooks already captured
        # for each role.
        for handler in audit_handlers:
            handler.record_inter_agent(
                from_role="strategist",
                to_role="crescendo" if output.escalate else "mutator",
                content=(
                    f"technique={output.technique} reason_code={output.reason_code} "
                    f"escalate={output.escalate} case_id={selected_case_id} "
                    f"rationale={output.rationale}"
                ),
                round=state.get("round", 0),
                module_id=module_id,
                case_id=selected_case_id,
            )

        return {
            "selected_technique": output.technique,
            "reason_code": output.reason_code,
            "current_case": current_case,
            # D-65/D-76: the escalation edge (`_post_strategist_edge`)
            # reads this flag to route to `crescendo` instead of `mutator`
            # -- topology reading state the Strategist set, never an agent
            # invoking a peer directly.
            "escalation_path": bool(output.escalate),
            "budget_ledger": ledger,
        }

    async def mutator_node(state: CampaignState) -> dict[str, Any]:
        role = get_role("mutator")
        agent = roles["mutator"]
        brief = role.brief(state)
        current_case_for_context: QueuedCase | dict[str, Any] = state.get("current_case") or {}
        audit_handlers = _audit_handlers(callbacks)
        for handler in audit_handlers:
            handler.set_context(
                round=state.get("round", 0),
                module_id=state.get("current_module"),
                case_id=current_case_for_context.get("case_id"),
                role="mutator",
            )
        call_result_holder: dict[str, Any] = {}
        attempts: list[tuple[int, str, Any]] = []
        try:
            output: MutatorOutput = await invoke_role_with_retry(
                agent,
                [("user", brief)],
                role="mutator",
                config=graph_config,
                on_attempt=_make_attempt_capture("mutator", attempts),
                on_success=call_result_holder.update,
            )
        except StructuredOutputFailure as exc:
            logger.error("Mutator failed to produce structured output: %s", exc)
            failures = _record_structural_failure(
                role_name="mutator",
                state=state,
                attempts=attempts,
                exc=exc,
                audit_handlers=audit_handlers,
                case_id=current_case_for_context.get("case_id"),
            )
            return {
                "variants": [],
                "role_structural_failures": failures,
                "budget_ledger": _record_role_call_spend(state, "mutator", call_result_holder),
            }

        case: QueuedCase | dict[str, Any] = state.get("current_case") or {}
        round_number = state.get("round", 0) + 1
        records: list[VariantRecord] = [
            VariantRecord(
                payload=variant.payload,
                technique_family=variant.technique_family,
                parent_case_id=case.get("case_id", ""),
                parent_technique_id=case.get("technique_id", ""),
                round=round_number,
                contributing_agent="mutator",
                variant_index=index,
                # 05-08: a Mutator-produced variant is always a single-exchange
                # refinement -- `None` here is what tells `dispatch_variants_node`
                # to route it through `adapter.send()`, never
                # `adapter.send_conversation()`.
                turns=None,
            )
            for index, variant in enumerate(output.variants)
        ]
        return {
            "variants": records,
            "budget_ledger": _record_role_call_spend(state, "mutator", call_result_holder),
        }

    async def crescendo_node(state: CampaignState) -> dict[str, Any]:
        """D-65/D-76: REPLACES `mutator_node` for a case on the escalation
        path (reached only via `_post_strategist_edge` when
        `state["escalation_path"]` is True) -- plans a short ordered
        multi-turn arc instead of independent in-turn refinements. Never
        runs alongside the Mutator for the same round (D-65): the
        conditional edge out of `strategist` picks exactly one of
        `mutator`/`crescendo`.

        Produces at most ONE `VariantRecord` (an arc is one escalation
        sequence, not up to D-79's per-round variant count) carrying
        `turns` -- the field `dispatch_variants_node` reads to route the
        resulting `TestCase` through the adapter's conversation entry
        point rather than the single-exchange one.
        """
        agent = roles["crescendo"]
        role = get_role("crescendo")
        brief = role.brief(state)
        current_case_for_context: QueuedCase | dict[str, Any] = state.get("current_case") or {}
        audit_handlers = _audit_handlers(callbacks)
        for handler in audit_handlers:
            handler.set_context(
                round=state.get("round", 0),
                module_id=state.get("current_module"),
                case_id=current_case_for_context.get("case_id"),
                role="crescendo",
            )
        call_result_holder: dict[str, Any] = {}
        attempts: list[tuple[int, str, Any]] = []
        try:
            output: CrescendoOutput = await invoke_role_with_retry(
                agent,
                [("user", brief)],
                role="crescendo",
                config=graph_config,
                on_attempt=_make_attempt_capture("crescendo", attempts),
                on_success=call_result_holder.update,
            )
        except StructuredOutputFailure as exc:
            logger.error("Crescendo Orchestrator failed to produce structured output: %s", exc)
            failures = _record_structural_failure(
                role_name="crescendo",
                state=state,
                attempts=attempts,
                exc=exc,
                audit_handlers=audit_handlers,
                case_id=current_case_for_context.get("case_id"),
            )
            return {
                "variants": [],
                "role_structural_failures": failures,
                "budget_ledger": _record_role_call_spend(state, "crescendo", call_result_holder),
            }

        ledger = _record_role_call_spend(state, "crescendo", call_result_holder)
        case: QueuedCase | dict[str, Any] = state.get("current_case") or {}
        round_number = state.get("round", 0) + 1

        if output.abort_recommended:
            # D-76: the arc is not dispatched at all, and is recorded as
            # abandoned with a reason -- distinct from a D-95 constraint
            # violation, since aborting a dead arc is a legitimate
            # strategic call, not a policy refusal. `dispatch_variants_node`
            # still runs (with an empty `variants` list for this case) so
            # the round is genuinely consumed, exactly like a D-95 refusal.
            abandoned = list(state.get("abandoned_arcs", []))
            abandoned.append(
                {
                    "round": round_number,
                    "case_id": case.get("case_id"),
                    "reason": output.arc_rationale,
                }
            )
            for handler in audit_handlers:
                handler.record_inter_agent(
                    from_role="crescendo",
                    to_role="strategist",
                    content=(
                        f"ABORTED arc for case_id={case.get('case_id')}: {output.arc_rationale}"
                    ),
                    round=state.get("round", 0),
                    module_id=state.get("current_module"),
                    case_id=case.get("case_id"),
                )
            return {
                "variants": [],
                "abandoned_arcs": abandoned,
                "budget_ledger": ledger,
            }

        turns = list(output.turns)
        record = VariantRecord(
            # D-90: the pre-joined fallback text, mirroring
            # `TargetAdapter.send_conversation()`'s own default degraded
            # substitute so a degraded (flattened) run and a genuine one
            # dispatch the SAME underlying content, differing only in
            # transport.
            payload="\n\n".join(turns),
            technique_family=state.get("selected_technique") or "",
            parent_case_id=case.get("case_id", ""),
            parent_technique_id=case.get("technique_id", ""),
            round=round_number,
            contributing_agent="crescendo",
            variant_index=0,
            turns=turns,
        )
        for handler in audit_handlers:
            handler.record_inter_agent(
                from_role="crescendo",
                to_role="dispatch_variants",
                content=(
                    f"arc_turns={len(turns)} backtrack_from_turn={output.backtrack_from_turn} "
                    f"rationale={output.arc_rationale}"
                ),
                round=state.get("round", 0),
                module_id=state.get("current_module"),
                case_id=case.get("case_id"),
            )
        return {
            "variants": [record],
            "budget_ledger": ledger,
        }

    async def dispatch_variants_node(state: CampaignState) -> dict[str, Any]:
        """D-79's bounded-concurrency parallel dispatch, T-01-18 re-derived
        for this layer (orchestrator.py's own containment pattern, applied
        here since LangGraph's default node-exception behavior is the
        opposite of `asyncio.gather`'s degrade-not-cancel semantics): each
        variant's dispatch+evaluate is wrapped in its own try/except
        degrading to a recorded `UNCERTAIN` result, never cancelling a
        sibling or the whole superstep. Also bumps `state["round"]` --
        the ONE state write `round_cap_edge` (the conditional edge
        immediately following this node) reads."""
        all_variants = state.get("variants", [])
        # 05-06 Task 3/D-75.3: on a `--resume`d campaign, skip any variant
        # whose `(parent_case_id, round, variant_index)` triple already
        # appears in the RESTORED `dispatch_results` -- the defensive guard
        # against 05-RESEARCH Pitfall 3 (a node re-executing from the top
        # on resume) reaching this node specifically. A pure no-op
        # (`variants == all_variants`) whenever `resume_dispatched_keys` is
        # `None`/empty, i.e. every non-resume call site.
        if resume_dispatched_keys:
            variants = [
                record
                for record in all_variants
                if idempotency_key(
                    record["parent_case_id"], record["round"], record["variant_index"]
                )
                not in resume_dispatched_keys
            ]
        else:
            variants = all_variants
        current_case: QueuedCase | dict[str, Any] = state.get("current_case") or {}
        module_id = current_case.get("module_id")
        module = modules.get(module_id) if module_id else None
        semaphore = asyncio.Semaphore(max_concurrency)
        audit_handlers = _audit_handlers(callbacks)

        # CR-01: `record["variant_index"]` (`enumerate()`-based, set by
        # `mutator_node`/`crescendo_node`) is only unique WITHIN the round
        # that produced it -- it resets to 0 on every Mutator/Crescendo
        # invocation, so it cannot alone disambiguate a `case_id` once the
        # Strategist re-selects the same `parent_case_id` in a LATER round
        # (a routine occurrence: the work queue is static for the whole
        # campaign and nothing prevents re-selection). Disambiguate by
        # adding the count of variants ALREADY dispatched for this exact
        # `parent_case_id` in prior rounds: `state["dispatch_results"]` is
        # `Annotated[..., operator.add]` (state.py), so by the time this
        # node runs for round N, it already holds every prior round's own
        # entries (never this round's -- those are only merged into state
        # AFTER this node returns), making this a stable, round-invariant
        # offset for every concurrent `_dispatch_one()` call in the current
        # round. This keeps the id format byte-for-byte unchanged
        # (`{parent_case_id}-mut-{n}`, 1-based) for the common case where a
        # parent case is only ever attacked in one round -- the vast
        # majority of existing fixtures -- while still guaranteeing a
        # campaign-wide-unique id whenever it is not.
        prior_dispatch_counts: dict[str, int] = {}
        for entry in state.get("dispatch_results", []):
            parent_id = entry["record"]["parent_case_id"]
            prior_dispatch_counts[parent_id] = prior_dispatch_counts.get(parent_id, 0) + 1

        def _record_dispatch(
            *, generated_case_id: str, record: VariantRecord, eval_result: EvalResult
        ) -> None:
            """D-84/D-85: one `direction="target"` line per dispatched
            variant, on BOTH the success path and the contained-failure
            path -- a degraded case is still auditable (05-06 Task 1's own
            `<action>` requirement). `round` uses `record["round"]` (the
            round the Mutator stamped onto the variant itself, set BEFORE
            `dispatch_variants_node` bumps `state["round"]` below) so every
            line's `round` reflects the campaign round the variant actually
            belongs to, consistent across all three exit branches."""
            for handler in audit_handlers:
                handler.record_target_dispatch(
                    case_id=generated_case_id,
                    content=(
                        f"payload={record['payload']!r} -> "
                        f"verdict={eval_result.verdict.value} evidence={eval_result.evidence!r}"
                    ),
                    module_id=module_id,
                    round=record["round"],
                )

        async def _dispatch_one(record: VariantRecord) -> dict[str, Any]:
            # D-90/CR-01: human-readable `{parent_case_id}-mut-{n}`
            # (1-based) -- never the sole source of lineage. `record` (a
            # `VariantRecord`) is what actually populates the lineage map
            # in `runner.py`. `n` is offset by `prior_dispatch_counts`
            # (computed once above, from the campaign's own accumulated
            # `dispatch_results`) so a later round re-selecting the SAME
            # `parent_case_id` never regenerates an id a prior round
            # already used.
            prior_count = prior_dispatch_counts.get(record["parent_case_id"], 0)
            generated_case_id = (
                f"{record['parent_case_id']}-mut-{prior_count + record['variant_index'] + 1}"
            )
            turns = record.get("turns")
            test_case = TestCase(
                case_id=generated_case_id,
                prompt=record["payload"],
                technique_id=record["parent_technique_id"],
                parent_case_id=record["parent_case_id"],
                parent_technique_id=record["parent_technique_id"],
                round=record["round"],
                contributing_agent=record["contributing_agent"],
                turns=list(turns) if turns else None,
            )
            async with semaphore:
                try:
                    # 05-08/D-76: a Crescendo-produced record carries
                    # `turns` -- routed through the adapter's CONVERSATION
                    # entry point, never the single-exchange one, exactly
                    # `orchestrator.py`'s own dispatch rule (D-93, never
                    # re-derived differently here: a non-empty `turns` is
                    # what decides the branch, in both places). The
                    # module's optional sequence-abort hook is resolved by
                    # the SAME duck-typed `getattr` shape
                    # `ScanOrchestrator._resolve_stop_when()` uses --
                    # reproduced locally (`_resolve_stop_when` below,
                    # module-level in this file) rather than imported,
                    # since `orchestrator.py` stays byte-for-byte unchanged
                    # (D-93). A module without the hook simply never
                    # aborts early; a hook that raises is treated as "do
                    # not abort" -- identical semantics.
                    if turns:
                        stop_when = _resolve_stop_when(module, test_case)
                        response = await adapter.send_conversation(test_case, stop_when=stop_when)
                    else:
                        response = await adapter.send(test_case)
                except Exception as exc:  # noqa: BLE001 -- T-01-18 containment
                    logger.warning(
                        "Deep-mode variant %r dispatch failed, degrading to UNCERTAIN: %s",
                        generated_case_id,
                        exc,
                    )
                    eval_result = EvalResult(
                        case_id=generated_case_id,
                        verdict=Verdict.UNCERTAIN,
                        confidence=0.0,
                        evidence=f"transport failure: {exc}",
                        detection_layer="regex",
                    )
                    _record_dispatch(
                        generated_case_id=generated_case_id, record=record, eval_result=eval_result
                    )
                    return {
                        "case_id": generated_case_id,
                        "module_id": module_id,
                        "record": record,
                        # No response was ever received -- nothing to price
                        # (05-04's target-side spend recording below reads
                        # this key and degrades to $0.0 when it is `None`).
                        "target_response": None,
                        "eval_result": eval_result,
                    }
                try:
                    if module is None:
                        raise RuntimeError(
                            f"unknown module_id {module_id!r} for deep-mode dispatch"
                        )
                    # D-66: the owning module's own unchanged evaluate() is
                    # the ONLY scoring instrument -- no role in this file
                    # assigns a verdict.
                    eval_result = await module.evaluate(test_case, response)
                except Exception as exc:  # noqa: BLE001 -- T-01-18 containment
                    logger.warning(
                        "Deep-mode variant %r evaluation failed, degrading to UNCERTAIN: %s",
                        generated_case_id,
                        exc,
                    )
                    eval_result = EvalResult(
                        case_id=generated_case_id,
                        verdict=Verdict.UNCERTAIN,
                        confidence=0.0,
                        evidence=f"evaluation failure: {exc}",
                        detection_layer="judge",
                    )
                    _record_dispatch(
                        generated_case_id=generated_case_id, record=record, eval_result=eval_result
                    )
                    return {
                        "case_id": generated_case_id,
                        "module_id": module_id,
                        "record": record,
                        # The response WAS received (evaluation failed
                        # afterward) -- still priceable target-side spend.
                        "target_response": response,
                        "eval_result": eval_result,
                    }
                _record_dispatch(
                    generated_case_id=generated_case_id, record=record, eval_result=eval_result
                )
                return {
                    "case_id": generated_case_id,
                    "module_id": module_id,
                    "record": record,
                    "target_response": response,
                    "eval_result": eval_result,
                }

        # Never fewer than one recorded outcome per dispatched variant --
        # `asyncio.gather` here is safe because every branch above already
        # degrades instead of raising.
        results = await asyncio.gather(*[_dispatch_one(record) for record in variants])

        # 05-04/D-80: fold this round's target-side spend into the SAME
        # ledger every attacker-side call already updates -- one number
        # counting both sides. `getattr(adapter, "model", None)` degrades
        # cleanly to `0.0`-priced spend for an `HttpAppAdapter` (no single
        # per-token-priced "model" concept for an arbitrary HTTP target);
        # D-79's variant-per-round cap plus `max_concurrency` is what
        # structurally bounds target-side call volume regardless.
        ledger: BudgetLedger = dict(state.get("budget_ledger") or {})  # type: ignore[assignment]
        target_model = getattr(adapter, "model", None)
        for entry in results:
            response = entry.get("target_response")
            tokens_used = getattr(response, "tokens_used", None) if response is not None else None
            cost = target_call_cost(target_model, tokens_used, ledger=ledger)
            if cost:
                record_target_spend(ledger, usd=cost)

        return {
            "dispatch_results": list(results),
            "round": state.get("round", 0) + 1,
            "budget_ledger": ledger,
        }

    async def analyst_node(state: CampaignState) -> dict[str, Any]:
        """D-65/D-66: reads this round's dispatched target responses (via
        `analyst.py`'s `brief()`, filtered to `state["round"]`) and reports
        a structured OBSERVATION of the target's defence -- never a
        verdict. Scoring is exclusively `module.evaluate()`'s job, already
        run inside `dispatch_variants_node` above; nothing here constructs
        a `Verdict`/`Finding` or is ever read by `scoring/engine.py`.

        Backward-compatible NO-OP (`return {}`) when the caller's `roles`
        dict has no `"analyst"` entry -- every pre-05-07 call site (and
        every pre-05-07 test) that only supplies `strategist`/`mutator`
        continues to route straight through this node unchanged, exactly
        as the "insertion point" comment it replaces already documented.
        """
        agent = roles.get("analyst")
        if agent is None:
            return {}
        role = get_role("analyst")
        brief = role.brief(state)
        audit_handlers = _audit_handlers(callbacks)
        for handler in audit_handlers:
            handler.set_context(
                round=state.get("round", 0), module_id=state.get("current_module"), role="analyst"
            )
        call_result_holder: dict[str, Any] = {}
        attempts: list[tuple[int, str, Any]] = []
        try:
            output: ObservedDefence = await invoke_role_with_retry(
                agent,
                [("user", brief)],
                role="analyst",
                config=graph_config,
                on_attempt=_make_attempt_capture("analyst", attempts),
                on_success=call_result_holder.update,
            )
        except StructuredOutputFailure as exc:
            # D-94 AT-6: a recorded structural failure, never a silently
            # skipped round -- the round itself already completed
            # (dispatch already ran); only the Analyst's own read of it is
            # missing this round. 05-11 Rule 1/2 fix: also recorded to the
            # audit trail (`role_structural_failures`).
            logger.error("Analyst failed to produce structured output: %s", exc)
            current_case_for_failure: QueuedCase | dict[str, Any] = state.get("current_case") or {}
            failures = _record_structural_failure(
                role_name="analyst",
                state=state,
                attempts=attempts,
                exc=exc,
                audit_handlers=audit_handlers,
                case_id=current_case_for_failure.get("case_id"),
            )
            return {
                "role_structural_failures": failures,
                "budget_ledger": _record_role_call_spend(state, "analyst", call_result_holder),
            }

        memory: CampaignMemory = state.get("bounded_memory") or new_campaign_memory()
        technique = state.get("selected_technique")
        if technique:
            if output.technique_outcome == "dead":
                memory = mark_technique_dead(memory, technique)
            elif output.technique_outcome == "partial_movement":
                memory = mark_partial_movement(memory, technique)
        if output.refusal_style:
            memory = remember_refusal_signature(memory, output.refusal_style)

        # D-84: the analyst->strategist handoff is what closes the
        # coordination loop -- recorded explicitly, mirroring the
        # strategist->mutator handoff `strategist_node` already records.
        for handler in audit_handlers:
            handler.record_inter_agent(
                from_role="analyst",
                to_role="strategist",
                content=(
                    f"technique_outcome={output.technique_outcome} "
                    f"refusal_style={output.refusal_style!r} apparent_filter={output.apparent_filter!r}"
                ),
                round=state.get("round", 0),
                module_id=state.get("current_module"),
            )

        return {
            # D-66: a structured OBSERVATION only, keyed by name so a grep
            # for `Verdict(` / `Finding(` in this function finds neither --
            # never read by score()/Finding/Verdict construction anywhere
            # in this codebase.
            "observed_defence": output.model_dump(),
            "bounded_memory": memory,
            "budget_ledger": _record_role_call_spend(state, "analyst", call_result_holder),
        }

    async def recon_node(state: CampaignState) -> dict[str, Any]:
        """D-65: runs exactly once per campaign -- reachable ONLY via the
        `START` edge (see module docstring), so a single `.ainvoke()` call
        invokes this node at most once regardless of how many rounds or
        modules follow. Seeds campaign memory BEFORE the first
        `strategist_node` invocation, so the Strategist is informed from
        round 1.

        Backward-compatible NO-OP (`return {}`) when the caller's `roles`
        dict has no `"recon"` entry -- mirrors `analyst_node`'s own
        graceful degrade for every pre-05-07 call site/test.

        A Recon failure degrades to an empty seed with a recorded note
        (`constraint_violations`) rather than aborting the campaign --
        the campaign is still worth running with an unseeded Strategist
        (T-05-07-07).
        """
        agent = roles.get("recon")
        if agent is None:
            return {}
        role = get_role("recon")
        brief = role.brief(state)
        audit_handlers = _audit_handlers(callbacks)
        for handler in audit_handlers:
            handler.set_context(round=0, module_id=state.get("current_module"), role="recon")
        call_result_holder: dict[str, Any] = {}
        attempts: list[tuple[int, str, Any]] = []
        try:
            output: ReconOutput = await invoke_role_with_retry(
                agent,
                [("user", brief)],
                role="recon",
                config=graph_config,
                on_attempt=_make_attempt_capture("recon", attempts),
                on_success=call_result_holder.update,
            )
        except StructuredOutputFailure as exc:
            # 05-11 Rule 1/2 fix: recorded into `role_structural_failures`,
            # NEVER `constraint_violations` -- that field is reserved for
            # D-95 allowlist refusals only (`state.py`), and Recon's own
            # structured-output exhaustion mislabeled itself into it before
            # this field existed. No case has been selected yet (Recon runs
            # once, before round 1), so no `case_id` is passed.
            logger.warning("Recon failed to produce structured output, continuing unseeded: %s", exc)
            failures = _record_structural_failure(
                role_name="recon",
                state=state,
                attempts=attempts,
                exc=exc,
                audit_handlers=audit_handlers,
            )
            return {
                "role_structural_failures": failures,
                "budget_ledger": _record_role_call_spend(state, "recon", call_result_holder),
            }

        memory: CampaignMemory = state.get("bounded_memory") or new_campaign_memory()
        if output.initial_refusal_signature:
            memory = remember_refusal_signature(memory, output.initial_refusal_signature)
        for hypothesis in output.initial_technique_hypotheses:
            memory = mark_partial_movement(memory, hypothesis)

        for handler in audit_handlers:
            handler.record_inter_agent(
                from_role="recon",
                to_role="strategist",
                content=(
                    f"initial_refusal_signature={output.initial_refusal_signature!r} "
                    f"initial_technique_hypotheses={output.initial_technique_hypotheses}"
                ),
                round=0,
                module_id=state.get("current_module"),
            )

        return {
            "bounded_memory": memory,
            "budget_ledger": _record_role_call_spend(state, "recon", call_result_holder),
        }

    def _post_strategist_edge(state: CampaignState) -> str:
        """D-95/D-65/D-76: the escalation edge -- the SINGLE conditional
        edge that routes out of `strategist_node`.

        Reads `state["selected_technique"]`, set by `strategist_node`'s
        own return: `None` exactly when the D-95 allowlist gate refused
        this round's selection (`strategist_node`'s own `try`/`except
        TechniqueNotAllowed` block, the one call site for the gate). A
        refusal routes straight to `dispatch_variants` with the
        already-empty `variants` list `strategist_node` set on that
        branch, bypassing BOTH mutation roles entirely -- zero
        mutation-role invocations for a refused round -- while still
        letting the round-bump-and-cap machinery run exactly as it would
        for a normal round (T-05-08-05).

        Otherwise reads `state["escalation_path"]` (the Strategist's own
        `escalate` flag, only ever set True alongside a technique that
        already cleared the allowlist gate) to choose `crescendo` over
        `mutator` -- the Crescendo Orchestrator REPLACES the Mutator on
        the escalation path, it never runs alongside it (D-65).
        """
        if state.get("selected_technique") is None:
            return "dispatch_variants"
        if state.get("escalation_path"):
            return "crescendo"
        return "mutator"

    def round_cap_edge(state: CampaignState) -> str:
        """D-70/D-72: round control lives here, in graph topology -- never
        in an agent's own reasoning. Reads `state["round"]` as bumped by
        `dispatch_variants_node` immediately prior. No longer wired
        directly as `dispatch_variants`'s own conditional edge (05-04) --
        `_post_budget_check_edge`/`_post_budget_approval_edge` call this as
        a plain function once the budget layer (evaluated first) has
        cleared."""
        if state.get("reason_code") is not None:
            return "finalize"
        if state.get("round", 0) >= state.get("max_rounds", 1):
            return "finalize"
        return "strategist"

    def _post_budget_check_edge(state: CampaignState) -> str:
        """Composes `over_budget_edge()`'s three abstract outcomes with
        `round_cap_edge`'s own decision for the `"continue"` case --
        `attacker/budget.py` cannot import `round_cap_edge` itself (that
        would be circular; this module already imports `budget.py`), so
        the composition lives here instead."""
        outcome = over_budget_edge(state)
        if outcome == "continue":
            return round_cap_edge(state)
        return outcome

    def _post_budget_approval_edge(state: CampaignState) -> str:
        """After `budget_approval_node` resumes: a refusal already set
        `termination_reason` (`WARN_APPROVAL_REFUSED`), so route straight
        to `finalize`; an approval falls through to the SAME round-cap
        decision the non-paused path uses."""
        if state.get("termination_reason") is not None:
            return "finalize"
        return round_cap_edge(state)

    async def finalize_node(state: CampaignState) -> dict[str, Any]:
        if state.get("termination_reason") is not None:
            return {}
        if state.get("over_budget"):
            # D-83: the hard cap (or independent call ceiling) tripped.
            # `dispatch_variants_node` has ALREADY run this round (this
            # node is reached only via `budget_check`, which sits AFTER
            # it) -- nothing bought is discarded, and overshoot is bounded
            # at exactly the one round that tripped it.
            updates: dict[str, Any] = {"termination_reason": "BUDGET_CAP_EXCEEDED"}
            ledger = state.get("budget_ledger")
            if ledger is not None:
                new_ledger: BudgetLedger = dict(ledger)  # type: ignore[assignment]
                mark_truncated(new_ledger, overshoot_rounds=1)
                updates["budget_ledger"] = new_ledger
            return updates
        if state.get("reason_code") is not None:
            return {"termination_reason": state["reason_code"]}
        return {"termination_reason": "ROUND_CAP_REACHED"}

    builder: StateGraph = StateGraph(CampaignState)
    builder.add_node("recon", recon_node)
    builder.add_node("strategist", strategist_node)
    builder.add_node("mutator", mutator_node)
    builder.add_node("crescendo", crescendo_node)
    builder.add_node("dispatch_variants", dispatch_variants_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("budget_check", budget_check_node)
    builder.add_node("budget_approval", budget_approval_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "recon")
    builder.add_edge("recon", "strategist")
    builder.add_conditional_edges(
        "strategist",
        _post_strategist_edge,
        {
            "mutator": "mutator",
            "crescendo": "crescendo",
            "dispatch_variants": "dispatch_variants",
        },
    )
    builder.add_edge("mutator", "dispatch_variants")
    builder.add_edge("crescendo", "dispatch_variants")
    builder.add_edge("dispatch_variants", "analyst")
    builder.add_edge("analyst", "budget_check")
    builder.add_conditional_edges(
        "budget_check",
        _post_budget_check_edge,
        {"strategist": "strategist", "finalize": "finalize", "budget_approval": "budget_approval"},
    )
    builder.add_conditional_edges(
        "budget_approval",
        _post_budget_approval_edge,
        {"strategist": "strategist", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
