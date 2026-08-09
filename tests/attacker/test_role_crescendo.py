"""05-08-PLAN.md Task 2: the Crescendo Orchestrator role and its multi-turn
arc schema (D-65, D-76).

Drives the real `build_campaign_graph()` topology directly (mirroring
`test_role_analyst.py`/`test_role_recon.py`'s own standalone-graph style),
using a fake adapter with a togglable conversation capability so both the
genuine and the flattened transport paths are exercised, offline, with zero
network access.
"""

from __future__ import annotations

from llmsec.api import _DEGRADED_MULTI_TURN_LIMITATION_NOTE, _scan_limitations
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles.crescendo import CrescendoOutput, build_crescendo_agent
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.state import QueuedCase, new_campaign_state
from llmsec.models import EvalResult, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"


class _StubModule(BaseModule):
    """Propagates `response.transport_mode` onto the returned `EvalResult`,
    exactly like `prompt_injection.py`/`pii_exfiltration.py` do -- so a
    degraded transport label is provably preserved end to end, onto the
    recorded result the report's limitations are computed from."""

    id = "stub_module"
    name = "Stub Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    async def generate_cases(self, context):  # type: ignore[no-untyped-def]
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="stub parent payload", technique_id=_STATIC_TECHNIQUE_ID
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.9,
            evidence=response.raw_text,
            detection_layer="regex",
            transport_mode=response.transport_mode,
        )


class _RaisingAgent:
    """A role-agent double whose `.ainvoke()` unconditionally raises --
    proves the wrapped role is structurally never invoked, rather than
    merely that its output goes unused."""

    async def ainvoke(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("this role must never be invoked on this path")


class _ConversationOnlyAdapter:
    """A `TargetAdapter`-shaped double whose `send()` raises -- proving a
    turn-carrying (Crescendo-produced) case reaches ONLY
    `send_conversation()`, never the single-exchange entry point directly.
    Mirrors `TargetAdapter.send_conversation()`'s own degraded-transport
    default: always labels its response `multi_turn_concatenated`, never
    `multi_turn_real` -- this adapter genuinely cannot hold conversation
    state.
    """

    supports_multi_turn = False
    supports_system_prompt_override = False

    def __init__(self, raw_text: str = "sure, here goes") -> None:
        self._raw_text = raw_text
        self.conversation_cases: list[TestCase] = []

    async def send(self, case: TestCase) -> TargetResponse:  # pragma: no cover -- must never be hit
        raise AssertionError("send() must never be called for a turn-carrying case")

    async def send_conversation(self, case: TestCase, stop_when=None) -> TargetResponse:  # type: ignore[no-untyped-def]
        self.conversation_cases.append(case)
        return TargetResponse(
            case_id=case.case_id,
            raw_text=self._raw_text,
            status_code=200,
            latency_ms=1.0,
            transport_mode="multi_turn_concatenated",
            turn_replies=[self._raw_text],
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "_ConversationOnlyAdapter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def _strategist_output(
    case_ids: list[str], *, escalate: bool, technique: str = "instruction_override"
) -> StrategistOutput:
    return StrategistOutput(
        technique=technique,
        ordered_case_ids=case_ids,
        escalate=escalate,
        reason_code=None,
        rationale="Escalate across turns." if escalate else "Refine in-turn.",
    )


def _mutator_output(n_variants: int = 1) -> MutatorOutput:
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload=f"mutated payload variant {i + 1}",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale=f"rationale {i + 1}",
            )
            for i in range(n_variants)
        ]
    )


def _crescendo_output(
    *, abort_recommended: bool = False, arc_rationale: str = "escalate gradually across two turns"
) -> CrescendoOutput:
    return CrescendoOutput(
        turns=["turn one, benign framing", "turn two, escalating slightly"],
        arc_rationale=arc_rationale,
        backtrack_from_turn=None,
        abort_recommended=abort_recommended,
    )


def _case_queue() -> list[QueuedCase]:
    return [
        QueuedCase(
            module_id="stub_module",
            case_id=_STATIC_CASE_ID,
            technique_id=_STATIC_TECHNIQUE_ID,
            prompt="stub parent payload",
            verdict="blocked",
            turns=None,
        )
    ]


def _initial_state(settings):  # type: ignore[no-untyped-def]
    state = new_campaign_state(
        scan_id="scan-crescendo-1",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    state["current_module"] = "stub_module"
    state["enabled_techniques"] = ["instruction_override"]
    return state


# --- Escalation routing: crescendo REPLACES mutator, never runs alongside --


async def test_escalation_path_dispatches_through_conversation_entry_point_never_send(
    scripted_chat_model,
):
    module = _StubModule()
    modules = {"stub_module": module}
    adapter = _ConversationOnlyAdapter()

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model(
        [_strategist_output([_STATIC_CASE_ID], escalate=True)]
    )
    crescendo_model = scripted_chat_model([_crescendo_output()])

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": _RaisingAgent(),  # must never be invoked on the escalation path
        "crescendo": build_crescendo_agent(settings, cfg, model=crescendo_model),
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    final_state = await compiled.ainvoke(_initial_state(settings))

    assert final_state.get("termination_reason") is not None
    assert len(adapter.conversation_cases) == 1
    dispatched_case = adapter.conversation_cases[0]
    assert dispatched_case.turns == ["turn one, benign framing", "turn two, escalating slightly"]

    dispatch_results = final_state.get("dispatch_results", [])
    assert len(dispatch_results) == 1
    eval_result = dispatch_results[0]["eval_result"]
    # The state-less adapter's degraded label is preserved onto the
    # recorded result, never overwritten with the genuine-conversation one.
    assert eval_result.transport_mode == "multi_turn_concatenated"
    assert eval_result.transport_mode != "multi_turn_real"


async def test_degraded_multi_turn_limitation_fires_for_a_flattened_deep_mode_case():
    """The EXISTING `api.py` limitation, computed from the case log --
    deep-mode results join that same log, so a flattened Crescendo run
    triggers the identical honest-disclosure caveat a Phase-2 flattened
    static multi-turn run would."""
    case_log = [
        EvalResult(
            case_id=f"{_STATIC_CASE_ID}-mut-1",
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.9,
            evidence="sure, here goes",
            detection_layer="regex",
            transport_mode="multi_turn_concatenated",
        )
    ]
    limitations = _scan_limitations(["stub_module"], case_log)
    assert _DEGRADED_MULTI_TURN_LIMITATION_NOTE in limitations


async def test_non_escalation_path_invokes_mutator_never_crescendo(scripted_chat_model):
    module = _StubModule()
    modules = {"stub_module": module}

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model(
        [_strategist_output([_STATIC_CASE_ID], escalate=False)]
    )
    mutator_model = scripted_chat_model([_mutator_output(1)])

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "crescendo": _RaisingAgent(),  # must never be invoked when escalate=False
    }
    compiled = build_campaign_graph(
        roles=roles, adapter=_NonEscalationAdapter(), modules=modules, max_concurrency=5
    )

    final_state = await compiled.ainvoke(_initial_state(settings))

    assert final_state.get("termination_reason") is not None
    dispatch_results = final_state.get("dispatch_results", [])
    assert len(dispatch_results) == 1
    # A Mutator-produced variant carries no turns -- routed through send(),
    # never send_conversation().
    assert dispatch_results[0]["record"]["turns"] is None


class _NonEscalationAdapter:
    """The mirror-image double of `_ConversationOnlyAdapter`: `send_conversation()`
    raises, proving a Mutator-produced (non-turn-carrying) variant reaches
    ONLY the single-exchange entry point."""

    supports_multi_turn = False
    supports_system_prompt_override = False

    def __init__(self) -> None:
        self.sent_cases: list[TestCase] = []

    async def send(self, case: TestCase) -> TargetResponse:
        self.sent_cases.append(case)
        return TargetResponse(
            case_id=case.case_id, raw_text="sure", status_code=200, latency_ms=1.0, transport_mode="single"
        )

    async def send_conversation(self, case: TestCase, stop_when=None):  # type: ignore[no-untyped-def]
        raise AssertionError("send_conversation() must never be called for a non-turn-carrying case")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "_NonEscalationAdapter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


# --- Abort: the arc is not dispatched at all, and the round is still consumed --


async def test_abort_recommended_arc_produces_zero_dispatches_and_recorded_abandoned_case(
    scripted_chat_model,
):
    module = _StubModule()
    modules = {"stub_module": module}
    adapter = _ConversationOnlyAdapter()

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model(
        [_strategist_output([_STATIC_CASE_ID], escalate=True)]
    )
    crescendo_model = scripted_chat_model(
        [_crescendo_output(abort_recommended=True, arc_rationale="the arc is clearly dead")]
    )

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": _RaisingAgent(),
        "crescendo": build_crescendo_agent(settings, cfg, model=crescendo_model),
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    final_state = await compiled.ainvoke(_initial_state(settings))

    # Zero dispatches: the aborted arc never reached the adapter at all.
    assert adapter.conversation_cases == []
    assert final_state.get("dispatch_results", []) == []

    abandoned = final_state.get("abandoned_arcs", [])
    assert len(abandoned) == 1
    assert abandoned[0]["case_id"] == _STATIC_CASE_ID
    assert abandoned[0]["reason"] == "the arc is clearly dead"

    # The round was still genuinely consumed -- `dispatch_variants_node`
    # still ran (with zero variants) and bumped the round counter, so the
    # campaign reaches a normal termination rather than looping.
    assert final_state.get("round", 0) >= 1
    assert final_state.get("termination_reason") is not None
