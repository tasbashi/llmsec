"""`CampaignState` — the deep-mode attacker team's checkpointed graph state.

Every field defined here is a plain `TypedDict` field or JSON-safe scalar --
no Pydantic models, no `datetime` objects, no enum VALUES stored as members
(enum members are stored as their `.value` strings) -- because every field
is serialized by 05-06's checkpointer on every LangGraph superstep.

The budget ledger lives HERE, as an ordinary checkpointed `CampaignState`
field, and never as a runner-local attribute (D-75) -- locked regardless of
whether `--resume` ships, because retrofitting it later would touch every
agent node and every checkpoint write.

This module performs ZERO imports of the `langchain`/`langgraph`/
`deepagents` stack, so it stays importable without the `[deep]` extra
installed.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from llmsec.attacker.config import ResolvedAttackerSettings
from llmsec.models import Verdict

# Fixed-order role tuple (D-68's determinism discipline applies to every
# iteration order in this layer) -- never a set.
ROLE_NAMES: tuple[str, ...] = ("recon", "strategist", "mutator", "analyst", "crescendo")

# D-72: the Strategist may only signal its own early exhaust, never a hard
# stop -- a strict subset of TerminationReason.
StrategistReasonCode = Literal["TECHNIQUES_EXHAUSTED", "TARGET_HARDENED", "LOW_YIELD"]

TerminationReason = Literal[
    "TECHNIQUES_EXHAUSTED",
    "TARGET_HARDENED",
    "LOW_YIELD",
    "ROUND_CAP_REACHED",
    "BUDGET_CAP_EXCEEDED",
    # 05-04/D-82: the operator explicitly declined to approve continued
    # spend past the warn threshold -- a distinct reason from the hard-cap
    # trip above, so a report can tell "operator said stop" apart from
    # "the cap itself stopped it".
    "WARN_APPROVAL_REFUSED",
]


class RoleSpend(TypedDict):
    """One role's spend counters within the shared campaign budget pool."""

    calls: int
    usd: float
    # D-81: this role's spend ceiling, expressed as an absolute dollar value
    # derived from `RoleOverride.budget_share` * the campaign cap. `None`
    # when no per-role share was configured for this role.
    share_ceiling_usd: float | None


class BudgetLedger(TypedDict):
    """The single campaign budget pool (D-80/D-81), checkpointed in-state (D-75)."""

    cap_usd: float
    warn_usd: float
    spent_usd: float
    # D-80: the ledger counts BOTH sides of the campaign's spend.
    attacker_spent_usd: float
    target_spent_usd: float
    agent_calls: int
    agent_call_ceiling: int
    per_role: dict[str, RoleSpend]
    truncated: bool
    overshoot_rounds: int
    warn_approved: bool
    # 05-04/D-80: incremented (never guessed into a dollar figure) every
    # time `attacker_call_cost()`/`target_call_cost()` finds no price entry
    # for the model in play -- the observable signal that this ledger's
    # dollar total is an undercount and the independent `agent_call_ceiling`
    # above is the real backstop for this campaign.
    unpriced_calls: int


class CampaignMemory(TypedDict):
    """Bounded structured campaign memory (D-71) -- NEVER a raw transcript.

    Each list is capped at `MEMORY_CAP` entries. This module owns only the
    shape and the cap; the actual eviction POLICY (which entry is most
    worth keeping when the cap is hit) is 05-07's `memory.py` job -- the
    `add_memory_entry()` helper below performs the simplest correct
    cap-enforcing behavior (oldest-first truncation) so the cap is never
    silently violated in the meantime.
    """

    refusal_signatures: list[str]
    dead_techniques: list[str]
    partial_movement_techniques: list[str]


#: Maximum entries retained per `CampaignMemory` list (D-71).
MEMORY_CAP: int = 12

#: The three `CampaignMemory` list keys, fixed order (D-68).
_MEMORY_KEYS: tuple[str, ...] = (
    "refusal_signatures",
    "dead_techniques",
    "partial_movement_techniques",
)


class QueuedCase(TypedDict):
    """A flattened, JSON-safe projection of a `TestCase` queued for deep-mode
    rework (D-77). References `TestCase`'s field names exactly -- this is a
    projection, never a second case model."""

    module_id: str
    case_id: str
    technique_id: str
    prompt: str
    verdict: str
    turns: list[str] | None


class VariantRecord(TypedDict):
    """One Mutator- or Crescendo-Orchestrator-produced payload variant.

    Field names here are exactly the fields that later populate D-90's
    lineage on the emitted `TestCase`/`Finding` -- so no downstream code
    ever needs to string-parse a `case_id` to recover lineage.

    `turns` (05-08, D-76): `None` for every Mutator-produced record (a
    single-exchange refinement); an ordered list of escalation turns for a
    Crescendo-produced record. `dispatch_variants_node` (`graph.py`) reads
    this field to decide whether to route the resulting `TestCase` through
    the adapter's `send()` (single-exchange) or `send_conversation()`
    (multi-turn) entry point -- the same `TestCase.turns`-non-empty rule
    `orchestrator.py`'s own dispatch already uses (D-93, never re-derived
    differently here). A required key (this `TypedDict` is NOT `total=False`)
    so every construction site states its intent explicitly rather than
    silently defaulting.
    """

    payload: str
    technique_family: str
    parent_case_id: str
    parent_technique_id: str
    round: int
    contributing_agent: str
    variant_index: int
    turns: list[str] | None


class CampaignState(TypedDict, total=False):
    """The single source of truth for one deep-mode campaign run.

    The budget ledger lives HERE (`budget_ledger`), never as a runner-local
    attribute -- locked regardless of resume (D-75). This is the single
    object 05-06's checkpointer persists once per LangGraph superstep.
    """

    scan_id: str
    round: int
    max_rounds: int
    variants_per_round: int
    # Deterministic round-robin ACROSS modules (D-78); Strategist orders
    # cases WITHIN a module.
    module_order: list[str]
    current_module: str | None
    case_queue: list[QueuedCase]
    current_case: QueuedCase | None
    selected_technique: str | None
    reason_code: StrategistReasonCode | None
    escalation_path: bool
    variants: list[VariantRecord]
    # 05-06 Task 3 (Rule 1 deviation -- see `05-06-SUMMARY.md`): `Annotated
    # [..., operator.add]` so LangGraph MERGES (concatenates) each round's
    # `dispatch_variants_node` return into the channel's PRIOR value,
    # rather than the TypedDict-default last-writer-wins overwrite. Without
    # this, a 2+-round campaign's `final_state["dispatch_results"]` held
    # only the LAST round's entries -- every earlier round's findings
    # silently vanished from `CampaignResult.eval_results`/`lineage`
    # (undercounting coverage-delta reporting, D-91) and from the set
    # `--resume`'s idempotency check (D-75.3) needs to reconstruct "already
    # dispatched" triples across the WHOLE campaign, not just its last
    # round. Discovered while implementing Task 3's idempotency guard,
    # which cannot be correct against a state shape that already drops
    # history every round.
    dispatch_results: Annotated[list[dict], operator.add]
    observed_defence: dict | None
    bounded_memory: CampaignMemory
    budget_ledger: BudgetLedger
    over_budget: bool
    # 05-04/D-73 mitigation 1: set by `budget_check_node`'s pure read,
    # never mutated anywhere else -- True once the ledger's spend has
    # crossed `warn_usd` and the operator has not yet approved continuing
    # (`budget_ledger["warn_approved"]`).
    warn_pending: bool
    termination_reason: TerminationReason | None
    constraint_violations: list[dict]
    # 05-08/D-76: one entry per Crescendo arc the Crescendo Orchestrator
    # itself recommended aborting (never dispatched) -- distinct from
    # `constraint_violations` (D-95 allowlist refusals), since an aborted
    # arc is a legitimate strategic call, not a policy violation. Each
    # entry carries `round`/`case_id`/`reason`, mirroring
    # `constraint_violations`'s own shape.
    abandoned_arcs: list[dict]
    # AT-6 (D-94, 05-11 Rule 1/2 fix): one entry per role whose structured
    # output retry (`roles/_structured_retry.py`) genuinely EXHAUSTED all
    # `MAX_STRUCTURED_OUTPUT_RETRIES + 1` attempts -- distinct from
    # `constraint_violations` (D-95 allowlist refusals ONLY, a Strategist
    # policy refusal, not a schema-validation failure) and distinct from
    # `abandoned_arcs` (a Crescendo Orchestrator's own strategic abort of a
    # dead arc, not a structural failure at all). Each entry carries
    # `role`/`round`/`attempt_count`/`reason`, populated by `graph.py`'s
    # five `except StructuredOutputFailure` blocks (strategist, mutator,
    # crescendo, analyst, recon). Before this field existed, Recon's own
    # structural-failure path mistakenly appended here into
    # `constraint_violations` -- see `graph.py`'s `recon_node` for the fix.
    role_structural_failures: list[dict]
    enabled_techniques: list[str]
    config_fingerprint: str | None


# D-77: FULL_COMPROMISE already proved the vector; UNCERTAIN in this
# codebase is overwhelmingly a synthetic degradation marker emitted by
# `orchestrator.py` for retry-exhausted transport failures and `evaluate()`
# exceptions, so queueing it would spend budget re-attacking a broken
# socket.
QUEUE_ELIGIBLE_VERDICTS: frozenset[Verdict] = frozenset({Verdict.BLOCKED, Verdict.PARTIAL_LEAK})


def new_role_spend() -> dict[str, RoleSpend]:
    """Initialize a fresh per-role spend map for every name in `ROLE_NAMES`."""
    return {role: RoleSpend(calls=0, usd=0.0, share_ceiling_usd=None) for role in ROLE_NAMES}


def new_budget_ledger(
    settings: ResolvedAttackerSettings,
    role_shares: dict[str, float] | None = None,
) -> BudgetLedger:
    """Construct a fresh `BudgetLedger` initialized from `settings`.

    `role_shares` (D-81), if given, is a `{role: fraction}` map applied to
    `settings.budget_usd` to populate each role's `share_ceiling_usd`; a
    role absent from `role_shares` keeps `share_ceiling_usd=None` (no
    per-role ceiling configured).
    """
    per_role = new_role_spend()
    if role_shares:
        for role, share in role_shares.items():
            if role in per_role:
                per_role[role]["share_ceiling_usd"] = settings.budget_usd * share
    return BudgetLedger(
        cap_usd=settings.budget_usd,
        warn_usd=settings.warn_threshold_usd,
        spent_usd=0.0,
        attacker_spent_usd=0.0,
        target_spent_usd=0.0,
        agent_calls=0,
        agent_call_ceiling=settings.agent_call_ceiling,
        per_role=per_role,
        truncated=False,
        overshoot_rounds=0,
        warn_approved=False,
        unpriced_calls=0,
    )


def new_campaign_memory() -> CampaignMemory:
    """Construct a fresh, empty `CampaignMemory`."""
    return CampaignMemory(refusal_signatures=[], dead_techniques=[], partial_movement_techniques=[])


def is_full(memory: CampaignMemory) -> bool:
    """True if ANY of `memory`'s three bounded lists is at `MEMORY_CAP`."""
    return any(len(memory[key]) >= MEMORY_CAP for key in _MEMORY_KEYS)  # type: ignore[literal-required]


def add_memory_entry(
    memory: CampaignMemory,
    kind: Literal["refusal_signatures", "dead_techniques", "partial_movement_techniques"],
    entry: str,
) -> None:
    """Append `entry` to `memory[kind]`, keeping the list at <= `MEMORY_CAP`.

    Oldest-first truncation -- the simplest correct cap enforcement. This
    module owns only the shape and the cap; 05-07's `memory.py` owns the
    actual eviction POLICY and may replace this truncation with something
    smarter (e.g. least-useful-first) without changing this shape.
    """
    bucket = memory[kind]
    bucket.append(entry)
    if len(bucket) > MEMORY_CAP:
        del bucket[: len(bucket) - MEMORY_CAP]


def new_campaign_state(
    scan_id: str,
    settings: ResolvedAttackerSettings,
    module_order: list[str],
    case_queue: list[QueuedCase],
    role_shares: dict[str, float] | None = None,
) -> CampaignState:
    """Construct the initial `CampaignState` for a fresh deep-mode campaign.

    `role_shares` (D-81), when given, is forwarded verbatim to
    `new_budget_ledger()` -- a role absent from it keeps
    `share_ceiling_usd=None` (no per-role ceiling configured).
    """
    return CampaignState(
        scan_id=scan_id,
        round=0,
        max_rounds=settings.max_rounds,
        variants_per_round=settings.variants_per_round,
        module_order=list(module_order),
        current_module=None,
        case_queue=list(case_queue),
        current_case=None,
        selected_technique=None,
        reason_code=None,
        escalation_path=False,
        variants=[],
        dispatch_results=[],
        observed_defence=None,
        bounded_memory=new_campaign_memory(),
        budget_ledger=new_budget_ledger(settings, role_shares),
        over_budget=False,
        warn_pending=False,
        termination_reason=None,
        constraint_violations=[],
        abandoned_arcs=[],
        role_structural_failures=[],
        enabled_techniques=[],
        config_fingerprint=None,
    )
