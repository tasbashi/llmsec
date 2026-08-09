"""`attacker/budget.py` -- D-73 mitigation 1: the budget ledger, the
between-round abort, the within-role step cap, cap-trip stop semantics, and
the pre-run labelled cost estimate.

Two independent halves make the budget a structural guarantee rather than a
report (05-RESEARCH `## Wave 0 Spike Results` / `### Per-Mitigation Go/No-Go
Verdicts`, Mitigation 1 -- GO):

1. **Between-round abort**: `budget_check_node` (a pure read) + `over_budget_edge`
   (the conditional-edge target selector) sit immediately after
   `dispatch_variants` in `graph.py`'s topology -- AFTER a round's spend has
   already been incurred and its variants already dispatched (D-83: "nothing
   bought is thrown away"), but BEFORE a new round's `strategist`/`mutator`
   calls could spend anything further. This is what bounds overshoot to at
   most one round (Task 2).
2. **Within-role step cap**: `StepCapMiddleware.awrap_model_call` (and its
   sync mirror `wrap_model_call`) returns a synthetic terminal message
   *instead of* calling the handler once a role's own `max_model_calls` is
   reached -- the short-circuit happens strictly BEFORE the paid model call,
   never after (D-70's per-role hard step cap).

`budget_check_node` performs no spend, no I/O, and no ledger mutation -- it
only reads state and returns flags. This is load-bearing: its downstream
sibling `budget_approval_node` calls `langgraph.types.interrupt()`, and
05-RESEARCH Pitfall 3 proved a node re-executes from the top on resume, so
keeping the check itself side-effect-free means re-execution can never
double-record anything.

Never a bare callback handler: `on_llm_end` raising inside a callback is
frequently swallowed under async execution and cannot stop a superstep
already in flight (05-RESEARCH `### Anti-Patterns to Avoid`). Every abort
path in this module is graph topology (conditional edges) or an in-loop
middleware short-circuit -- never a callback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import litellm
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from llmsec.attacker.config import ResolvedAttackerSettings
from llmsec.attacker.state import BudgetLedger, CampaignState, RoleSpend

logger = logging.getLogger(__name__)


# --- Ledger operations --------------------------------------------------
#
# `BudgetLedger` (state.py) is the single campaign budget pool (D-80/D-81),
# checkpointed in-state (D-75). Every function below mutates the ledger dict
# it is given in place -- callers own whether/when to fold that mutation
# into a node's return value (graph.py does; nothing here writes to
# `CampaignState` directly, keeping this module importable and testable
# without ever touching a graph).


def record_agent_spend(ledger: BudgetLedger, *, role: str, usd: float, calls: int = 1) -> None:
    """Attacker-side spend: increases `attacker_spent_usd`, `agent_calls`,
    and `per_role[role]`. `spent_usd` is re-derived as the sum of both
    sides on every call (D-80 -- one number counting both attacker-side and
    target-side spend, always).
    """
    ledger["attacker_spent_usd"] = ledger.get("attacker_spent_usd", 0.0) + usd
    ledger["agent_calls"] = ledger.get("agent_calls", 0) + calls
    per_role = ledger.setdefault("per_role", {})
    role_spend = per_role.setdefault(role, RoleSpend(calls=0, usd=0.0, share_ceiling_usd=None))
    role_spend["calls"] += calls
    role_spend["usd"] += usd
    ledger["spent_usd"] = ledger.get("attacker_spent_usd", 0.0) + ledger.get("target_spent_usd", 0.0)


def record_target_spend(ledger: BudgetLedger, *, usd: float) -> None:
    """Target-side spend: increases `target_spent_usd` only -- NEVER
    `agent_calls`, which counts only attacker-side model calls (D-80's
    independent call ceiling is scoped to the attacker stack specifically,
    the side that can accumulate zero against a dollar cap when unpriced).
    """
    ledger["target_spent_usd"] = ledger.get("target_spent_usd", 0.0) + usd
    ledger["spent_usd"] = ledger.get("attacker_spent_usd", 0.0) + ledger.get("target_spent_usd", 0.0)


def role_over_share(ledger: BudgetLedger, role: str) -> bool:
    """True once `role`'s own spend exceeds its configured share ceiling
    (D-81); False when the role has no configured share (`share_ceiling_usd`
    is `None`) or is not present in the ledger at all.
    """
    role_spend = ledger.get("per_role", {}).get(role)
    if role_spend is None:
        return False
    ceiling = role_spend.get("share_ceiling_usd")
    if ceiling is None:
        return False
    return role_spend.get("usd", 0.0) > ceiling


def remaining_usd(ledger: BudgetLedger) -> float:
    """Dollars left before the hard cap trips, floored at 0.0."""
    return max(0.0, ledger.get("cap_usd", 0.0) - ledger.get("spent_usd", 0.0))


def remaining_calls(ledger: BudgetLedger) -> int:
    """Attacker-side calls left before the independent call ceiling trips,
    floored at 0 -- the backstop that still bounds an unpriced model.
    """
    return max(0, ledger.get("agent_call_ceiling", 0) - ledger.get("agent_calls", 0))


# --- Pricing --------------------------------------------------------------
#
# D-73 mitigation 3: the attacker-side per-token pricing lookup is NOT the
# target adapter's cost helper. The two LLM stacks stay split even for
# pricing, deliberately, in two separate functions below -- folding them
# into one helper would couple the two stacks the way D-73 mitigation 3
# forbids for the adapter path itself.

#: Attacker-side per-model price table, USD per 1,000,000 tokens
#: (input_price, output_price) -- declared HERE, never sourced from
#: `litellm`'s own pricing lookup (that lookup is `target_call_cost()`'s
#: job, below, and only for the TARGET side).
ATTACKER_MODEL_PRICES_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    # openai:gpt-4o-mini is DEFAULT_ATTACKER_MODEL (attacker/config.py).
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4o": (2.50, 10.00),
}


def attacker_call_cost(
    usage_metadata: dict[str, Any] | None,
    model: str,
    *,
    ledger: BudgetLedger | None = None,
) -> float:
    """Attacker-side per-call cost from LangChain-side `usage_metadata`
    (`{"input_tokens": int, "output_tokens": int, ...}`, the shape
    `AIMessage.usage_metadata` carries).

    Returns a positive figure for a priced model. Returns `0.0` for a model
    with no entry in `ATTACKER_MODEL_PRICES_PER_MILLION_TOKENS` -- NEVER a
    guessed number -- and, when `ledger` is given, increments
    `ledger["unpriced_calls"]`. This exact silent-zero-if-unpriced behaviour
    is why D-80's independent `agent_call_ceiling` exists: a dollar cap
    alone would let an unpriced model spend forever while reporting $0.
    """
    if not usage_metadata:
        return 0.0
    prices = ATTACKER_MODEL_PRICES_PER_MILLION_TOKENS.get(model)
    if prices is None:
        if ledger is not None:
            ledger["unpriced_calls"] = ledger.get("unpriced_calls", 0) + 1
        return 0.0
    input_price, output_price = prices
    input_tokens = usage_metadata.get("input_tokens") or 0
    output_tokens = usage_metadata.get("output_tokens") or 0
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0


def target_call_cost(
    model: str | None,
    tokens_used: int | None,
    *,
    ledger: BudgetLedger | None = None,
) -> float:
    """Target-side per-call cost via `litellm.cost_per_token()` -- the
    TARGET stack's own pricing helper, deliberately a separate function
    from `attacker_call_cost()` above (D-73 mitigation 3). `tokens_used` is
    treated entirely as prompt-side tokens (the only breakdown
    `TargetResponse.tokens_used` gives us); this is a documented
    approximation, not a precise per-direction accounting -- acceptable
    here since this is best-effort dollar tracking, not the hard backstop
    (D-79's variant-per-round cap plus `max_concurrency` already
    structurally bounds target-side call volume).

    Returns `0.0` (never a guessed number) when `model` is unset, when
    `tokens_used` is falsy, or when `litellm` has no price for `model` --
    and, when `ledger` is given, increments `ledger["unpriced_calls"]` in
    the last case, exactly like `attacker_call_cost()`.
    """
    if not model or not tokens_used:
        return 0.0
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=tokens_used, completion_tokens=0
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:  # noqa: BLE001 -- litellm raises on any unknown/unpriced model
        if ledger is not None:
            ledger["unpriced_calls"] = ledger.get("unpriced_calls", 0) + 1
        return 0.0


# --- Between-round abort (Task 1 / Task 2) --------------------------------


def budget_check_node(state: CampaignState) -> dict[str, Any]:
    """Pure read: no spend, no I/O, no ledger mutation -- returns
    `over_budget`/`warn_pending` flags only. Load-bearing for
    `budget_approval_node`'s own pause-and-resume-on-approval semantics
    (05-RESEARCH Pitfall 3): this node's own re-execution can never
    double-record anything because it never records anything.

    `over_budget` is True once `spent_usd >= cap_usd`, OR independently
    once `agent_calls >= agent_call_ceiling` -- the unpriced-model case,
    where `spent_usd` can still read `0.0` (D-80).

    `warn_pending` is True once `spent_usd >= warn_usd` and the operator
    has not already approved continuing (`warn_approved`) -- but only when
    NOT already `over_budget` (the hard cap always wins).
    """
    ledger: dict[str, Any] = dict(state.get("budget_ledger") or {})
    spent_usd = ledger.get("spent_usd", 0.0)
    cap_usd = ledger.get("cap_usd", float("inf"))
    over_budget = spent_usd >= cap_usd
    if not over_budget:
        agent_calls = ledger.get("agent_calls", 0)
        agent_call_ceiling = ledger.get("agent_call_ceiling", float("inf"))
        over_budget = agent_calls >= agent_call_ceiling

    warn_pending = False
    if not over_budget:
        warn_usd = ledger.get("warn_usd", float("inf"))
        warn_pending = spent_usd >= warn_usd and not ledger.get("warn_approved", False)

    return {"over_budget": over_budget, "warn_pending": warn_pending}


def over_budget_edge(state: CampaignState) -> str:
    """The conditional-edge target selector reading `budget_check_node`'s
    flags. Returns one of three abstract outcomes:

    - `"finalize"` -- the hard cap (or call ceiling) has tripped; abort.
    - `"budget_approval"` -- the warn threshold has been crossed and not
      yet approved; pause for operator approval.
    - `"continue"` -- neither; proceed with the round-cap/reason-code
      decision that already exists in `graph.py` (`round_cap_edge`).

    Deliberately never imports `graph.py` (that would be circular --
    `graph.py` already imports THIS module to wire the graph); `graph.py`'s
    own wiring composes `"continue"` with whatever comes next.
    """
    if state.get("over_budget"):
        return "finalize"
    if state.get("warn_pending"):
        return "budget_approval"
    return "continue"


# --- Cap-trip stop semantics (Task 2, D-83) -------------------------------


def mark_truncated(ledger: BudgetLedger, overshoot_rounds: int) -> None:
    """Record that this campaign stopped early on a cap trip.
    `overshoot_rounds` is the count of rounds of target dispatch that ran
    AFTER the trip was detected -- bounded at 1 by `graph.py`'s topology
    (budget_check sits immediately after `dispatch_variants`, so the round
    that trips the cap has already been fully dispatched, and no further
    round begins).
    """
    ledger["truncated"] = True
    ledger["overshoot_rounds"] = overshoot_rounds


#: Shared wording between the pre-run notice (`render_cost_notice()`) and
#: the post-run disclosure (`truncation_disclosure()`) so the approved
#: number and the actual disclosure can never drift apart (Task 3's own
#: requirement: "the same bound stated to the operator before the run").
_OVERSHOOT_BOUND_SENTENCE = (
    "already-generated payloads are still dispatched, so the final spend "
    "may exceed the cap by at most one round of target calls"
)


def truncation_disclosure(ledger: BudgetLedger) -> str | None:
    """The operator-facing truncation sentence for `ScanReport.limitations`
    (mirroring the existing NER-not-installed/deep-mode-failed honest-
    disclosure precedent in `api.py`) -- present exactly when
    `ledger["truncated"]` is True, computed from the ledger alone
    (independent of whether any finding was produced), `None` otherwise.
    """
    if not ledger.get("truncated"):
        return None
    cap_usd = ledger.get("cap_usd", 0.0)
    spent_usd = ledger.get("spent_usd", 0.0)
    return (
        f"The --deep attacker campaign hit its ${cap_usd:.2f} hard budget cap "
        f"(actual spend: ${spent_usd:.2f}) and stopped early -- {_OVERSHOOT_BOUND_SENTENCE}."
    )


# --- Within-role step cap (Task 1, D-70) ----------------------------------

#: Default hard step cap applied to every role's own internal model-call
#: loop (D-70) -- deliberately small: every role in this package passes
#: `tools=[]` and a `ToolStrategy(<Schema>, handle_errors=False)`
#: response_format, so a well-behaved role needs exactly one real call.
#: This bounds the pathological case (a role that keeps calling a bound
#: filesystem/`task` tool -- 05-RESEARCH Pitfall 2 -- instead of ever
#: producing structured output) without needing per-role tuning yet.
DEFAULT_MAX_MODEL_CALLS_PER_ROLE: int = 5


class StepCapMiddleware(AgentMiddleware):
    """D-70's per-role hard step cap. Once `max_model_calls` REAL (paid)
    model calls have been made through this middleware instance, every
    further call within the SAME role invocation is intercepted and
    returns a synthetic terminal `AIMessage` INSTEAD OF calling the
    handler -- the short-circuit happens strictly before the paid call,
    never after (05-RESEARCH `mitigation1b_withinrole_abort.py`: "exactly 3
    real model calls were made before the 4th was intercepted").

    Implements both `awrap_model_call` (the functional path -- every role
    in this codebase is invoked exclusively via `.ainvoke()`) and its sync
    mirror `wrap_model_call` (defense in depth only; LangChain's own
    dispatcher raises `NotImplementedError` for whichever hook a subclass
    does NOT override, if the graph is ever invoked through the OTHER
    execution mode, per `langchain.agents.factory`'s own explicit
    "include middleware with either implementation to ensure
    NotImplementedError is raised when middleware doesn't support the
    execution path" comment -- defining both means neither path breaks).
    """

    def __init__(self, *, role: str, max_model_calls: int) -> None:
        super().__init__()
        self.role = role
        self.max_model_calls = max_model_calls
        self._calls_made = 0

    def _should_short_circuit(self) -> bool:
        return self._calls_made >= self.max_model_calls

    @staticmethod
    def _abort_message() -> AIMessage:
        return AIMessage(content="", additional_kwargs={"step_cap_abort": True})

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        if self._should_short_circuit():
            return self._abort_message()
        self._calls_made += 1
        return await handler(request)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        if self._should_short_circuit():
            return self._abort_message()
        self._calls_made += 1
        return handler(request)


def build_step_cap_middleware(role: str, *, max_model_calls: int | None = None) -> list[Any]:
    """The ONE shared construction point every role factory calls to attach
    `StepCapMiddleware` -- never duplicated per role file (D-70's action
    text). Each role file adds exactly one line
    (`middleware=build_step_cap_middleware("<role>")`) to its own
    `create_deep_agent()` call; the middleware CLASS and its default cap
    live only here.
    """
    cap = max_model_calls if max_model_calls is not None else DEFAULT_MAX_MODEL_CALLS_PER_ROLE
    return [StepCapMiddleware(role=role, max_model_calls=cap)]


# --- Pre-run labelled cost estimate + warn-threshold approval (Task 3) ----

#: D-65's per-round core roles (Strategist + Mutator-or-Crescendo +
#: Analyst) -- the estimate assumes the EVENTUAL full round shape (even
#: though only Strategist/Mutator are wired as of this plan) so it does not
#: need re-deriving once Analyst (05-07)/Crescendo (05-08) ship.
CORE_ROLES_PER_ROUND: int = 3

#: Recon runs once per scan, not once per round (D-65).
RECON_CALLS_PER_CAMPAIGN: int = 1

#: A representative (input, output) token pair for one structured-output
#: role call -- NOT a real per-call measurement (none exists before the
#: campaign runs); an ESTIMATE input only, always rendered with an explicit
#: "estimate" label by `render_cost_notice()`.
_ASSUMED_INPUT_TOKENS_PER_CALL: int = 1500
_ASSUMED_OUTPUT_TOKENS_PER_CALL: int = 512


@dataclass(frozen=True)
class CostEstimate:
    """A labelled typical/worst-case cost range beside the hard cap
    (D-82). `typical_usd`/`worst_case_usd` are `None` exactly when the
    configured attacker model has no entry in
    `ATTACKER_MODEL_PRICES_PER_MILLION_TOKENS` -- rendered as an explicit
    "unavailable" notice by `render_cost_notice()`, never a `0.0` (which
    would read as a real, priced quote of zero).
    """

    typical_usd: float | None
    worst_case_usd: float | None
    cap_usd: float
    assumed_calls_typical: int
    assumed_calls_worst: int


def _avg_attacker_call_cost(model: str) -> float | None:
    """One representative attacker-side call's cost, or `None` if `model`
    has no price entry."""
    prices = ATTACKER_MODEL_PRICES_PER_MILLION_TOKENS.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (
        _ASSUMED_INPUT_TOKENS_PER_CALL * input_price
        + _ASSUMED_OUTPUT_TOKENS_PER_CALL * output_price
    ) / 1_000_000.0


def estimate_campaign_cost(
    settings: ResolvedAttackerSettings, queue_size: int, model: str
) -> CostEstimate:
    """A typical/worst-case range built from the per-round call arithmetic:
    `CORE_ROLES_PER_ROUND` times the round cap, plus the one-off Recon
    call, times `queue_size` (the campaign's work-queue breadth -- the
    number of independently-round-driven work units, e.g. eligible modules
    at the CLI's own call site). Worst case assumes every unit runs to the
    full `settings.max_rounds` cap; typical assumes the configured average
    early-exit behaviour (D-72), approximated as half the round cap
    (floored at 1 round). `worst_case_usd >= typical_usd` always, since
    `typical_rounds <= settings.max_rounds`.
    """
    per_call = _avg_attacker_call_cost(model)
    typical_rounds = max(1, round(settings.max_rounds / 2))
    worst_calls = queue_size * (CORE_ROLES_PER_ROUND * settings.max_rounds + RECON_CALLS_PER_CAMPAIGN)
    typical_calls = queue_size * (CORE_ROLES_PER_ROUND * typical_rounds + RECON_CALLS_PER_CAMPAIGN)
    if per_call is None:
        return CostEstimate(
            typical_usd=None,
            worst_case_usd=None,
            cap_usd=settings.budget_usd,
            assumed_calls_typical=typical_calls,
            assumed_calls_worst=worst_calls,
        )
    return CostEstimate(
        typical_usd=typical_calls * per_call,
        worst_case_usd=worst_calls * per_call,
        cap_usd=settings.budget_usd,
        assumed_calls_typical=typical_calls,
        assumed_calls_worst=worst_calls,
    )


def render_cost_notice(estimate: CostEstimate) -> str:
    """The operator-facing pre-run block (D-82): both the typical and
    worst-case figures ALWAYS shown together, each explicitly labelled an
    estimate, with the configured hard cap on its own line -- three
    distinct numbers, never one (a single number would read as a quote).
    Uses the SAME overshoot-bound wording `truncation_disclosure()` uses,
    so the approved number and the post-run disclosure never drift apart.
    """
    if estimate.typical_usd is None or estimate.worst_case_usd is None:
        return (
            "--deep cost estimate: UNAVAILABLE (the configured attacker model has "
            "no known price entry, so no dollar figure can be estimated).\n"
            f"Hard budget cap: ${estimate.cap_usd:.2f} -- the independent agent-call "
            "ceiling is what bounds this run since no dollar estimate can be computed.\n"
        )
    return (
        "--deep cost estimate (both figures below are ESTIMATES, not quotes):\n"
        f"  Typical estimate:    ~${estimate.typical_usd:.2f} "
        f"(assumes early Strategist exit, ~{estimate.assumed_calls_typical} agent calls)\n"
        f"  Worst-case estimate: ~${estimate.worst_case_usd:.2f} "
        f"(assumes the full round cap, ~{estimate.assumed_calls_worst} agent calls)\n"
        f"  Hard cap:            ${estimate.cap_usd:.2f}\n"
        f"If the hard cap trips mid-campaign, {_OVERSHOOT_BOUND_SENTENCE}.\n"
    )


def budget_approval_node(state: CampaignState) -> dict[str, Any]:
    """Pause for operator approval once `budget_check_node` sets
    `warn_pending`.

    `interrupt()` is the FIRST statement in this function body (verified by
    an AST check, per Task 3's acceptance criteria) -- 05-RESEARCH Pitfall
    3: this node body re-executes from the top on resume, so ANY code
    placed before `interrupt()` runs a second time. All spend-incurring or
    side-effecting work (recording the approval) belongs strictly AFTER
    it, which is exactly where it sits below: a resumed campaign therefore
    records exactly one approval event, never two.
    """
    # `interrupt()` is deliberately the ONLY thing evaluated in this first
    # statement -- even the ledger lookups it needs for its payload are
    # inlined into this SAME statement's expression, never split into a
    # preceding assignment, so nothing whatsoever runs before the pause.
    decision = interrupt(
        {
            "spent_usd": cast(BudgetLedger, state.get("budget_ledger") or {}).get("spent_usd", 0.0),
            "remaining_usd": remaining_usd(cast(BudgetLedger, state.get("budget_ledger") or {})),
            "cap_usd": cast(BudgetLedger, state.get("budget_ledger") or {}).get("cap_usd", 0.0),
        }
    )
    if decision:
        ledger = dict(state.get("budget_ledger") or {})
        ledger["warn_approved"] = True
        return {"budget_ledger": ledger}
    return {"termination_reason": "WARN_APPROVAL_REFUSED"}
