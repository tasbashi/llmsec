"""05-07-PLAN.md Task 3: the Recon role -- one seeded pass per scan over a
bounded, target-locked probe set (D-65, D-87).
"""

from __future__ import annotations

import inspect

from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles._structured_retry import MAX_STRUCTURED_OUTPUT_RETRIES
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.recon import (
    RECON_PROBE_COUNT,
    RECON_PROBE_SET,
    ReconOutput,
    build_recon_agent,
    build_target_probe_tool,
)
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.state import QueuedCase, new_campaign_state
from llmsec.models import EvalResult, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

from .conftest import ScriptedToolCallChatModel

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"


class _StubModule(BaseModule):
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
            verdict=Verdict.BLOCKED,
            confidence=0.9,
            evidence=response.raw_text,
            detection_layer="regex",
        )


class _AlwaysInvalidAgent:
    """A role-agent double whose `.ainvoke()` always raises a REAL
    `pydantic.ValidationError`, driving `invoke_role_with_retry()`'s
    genuine bounded-retry loop to real exhaustion."""

    def __init__(self) -> None:
        self.call_count = 0

    async def ainvoke(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.call_count += 1
        ReconOutput.model_validate({})


def _strategist_output(case_ids: list[str], technique: str = "instruction_override") -> StrategistOutput:
    return StrategistOutput(
        technique=technique,
        ordered_case_ids=case_ids,
        escalate=False,
        reason_code=None,
        rationale="Try a direct instruction-override refinement.",
    )


def _mutator_output(n_variants: int) -> MutatorOutput:
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


def _recon_output(
    *, hypotheses: list[str] | None = None, refusal_signature: str = "polite decline citing policy"
) -> ReconOutput:
    return ReconOutput(
        initial_refusal_signature=refusal_signature,
        initial_technique_hypotheses=hypotheses if hypotheses is not None else ["instruction_override"],
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


def _recon_scripted_model(outputs: list[ReconOutput], *, n_probe_calls: int) -> ScriptedToolCallChatModel:
    """Build a scripted model that calls `probe_target` `n_probe_calls`
    times, then returns the final `ReconOutput` structured response."""
    script = [{"name": "probe_target", "args": {"probe": p}} for p in RECON_PROBE_SET[:n_probe_calls]]
    script += [{"name": type(out).__name__, "args": out.model_dump(mode="json")} for out in outputs]
    return ScriptedToolCallChatModel(script=script)


# --- RECON_PROBE_COUNT / RECON_PROBE_SET ------------------------------------


def test_recon_probe_count_is_between_three_and_five():
    assert 3 <= RECON_PROBE_COUNT <= 5


def test_recon_probe_set_has_exactly_probe_count_entries():
    assert len(RECON_PROBE_SET) == RECON_PROBE_COUNT


# --- build_target_probe_tool: single-parameter, adapter-only dispatch -----


def test_build_target_probe_tool_signature_is_adapter_only():
    sig = inspect.signature(build_target_probe_tool)
    assert list(sig.parameters) == ["adapter"]


async def test_probe_tool_callable_exposes_exactly_one_parameter(fake_target_adapter):
    adapter = fake_target_adapter()
    tool = build_target_probe_tool(adapter)
    sig = inspect.signature(tool.coroutine)
    assert list(sig.parameters) == ["probe"]


async def test_probe_tool_dispatches_only_through_the_supplied_adapter(
    fake_target_adapter, mock_target_response
):
    adapter = fake_target_adapter()
    tool = build_target_probe_tool(adapter)
    result = await tool.ainvoke({"probe": "hello target"})
    assert len(adapter.sent_cases) == 1
    assert adapter.sent_cases[0].prompt == "hello target"
    assert isinstance(result, str)


async def test_probe_tool_call_count_matches_number_of_invocations(fake_target_adapter):
    adapter = fake_target_adapter()
    tool = build_target_probe_tool(adapter)
    for probe in RECON_PROBE_SET:
        await tool.ainvoke({"probe": probe})
    assert len(adapter.sent_cases) == len(RECON_PROBE_SET)


# --- ReconOutput schema shape ------------------------------------------------


def test_recon_output_has_no_verdict_shaped_field():
    fields = set(ReconOutput.model_fields)
    assert not (fields & {"verdict", "severity", "confidence", "score"})


def test_recon_role_build_requires_adapter_kwarg():
    role = ROLE_REGISTRY["recon"]
    cfg = AttackerConfig(profile="light")
    settings = resolve_settings(cfg)
    try:
        role.build(settings, cfg)
    except ValueError as exc:
        assert "adapter=" in str(exc)
    else:
        raise AssertionError("Recon.build() must require adapter=")


def test_recon_role_is_registered():
    assert "recon" in ROLE_REGISTRY
    role = ROLE_REGISTRY["recon"]
    assert role.name == "recon"
    assert role.output_schema is ReconOutput


# --- Full campaign: Recon runs exactly once, seeds memory before round 1 --


async def test_recon_runs_exactly_once_across_a_multi_round_campaign(
    fake_target_adapter, mock_target_response, scripted_chat_model
):
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1"))

    cfg = AttackerConfig(profile="light", max_rounds=3, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model([_strategist_output([_STATIC_CASE_ID]) for _ in range(3)])
    mutator_model = scripted_chat_model([_mutator_output(1) for _ in range(3)])
    recon_model = _recon_scripted_model([_recon_output()], n_probe_calls=RECON_PROBE_COUNT)

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "recon": build_recon_agent(settings, cfg, model=recon_model, adapter=adapter),
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    initial_state = new_campaign_state(
        scan_id="scan-recon-1",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)

    # Exactly one Recon model invocation-batch for the whole 3-round
    # campaign (the probe calls plus the final structured-output call are
    # all made against the SAME scripted model instance, never re-entered
    # -- `recon_node` is reachable only via START).
    assert len(recon_model.received_message_batches) == RECON_PROBE_COUNT + 1
    assert final_state.get("round") == 3
    assert final_state.get("termination_reason") == "ROUND_CAP_REACHED"

    # At most RECON_PROBE_COUNT probe dispatches were made through the
    # adapter by Recon's own tool calls (round dispatches happen through a
    # different case_id shape, `{parent}-mut-{n}`, so counting
    # `recon-probe-*`-prefixed sent cases isolates Recon's own dispatches).
    recon_probe_dispatches = [c for c in adapter.sent_cases if c.case_id.startswith("recon-probe-")]
    assert len(recon_probe_dispatches) <= RECON_PROBE_COUNT


async def test_recon_seeds_memory_before_the_first_strategist_brief_is_built(
    fake_target_adapter, mock_target_response, scripted_chat_model
):
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1"))

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model([_strategist_output([_STATIC_CASE_ID])])
    mutator_model = scripted_chat_model([_mutator_output(1)])
    recon_model = _recon_scripted_model(
        [_recon_output(hypotheses=["instruction_override"], refusal_signature="firm policy refusal")],
        n_probe_calls=2,
    )

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "recon": build_recon_agent(settings, cfg, model=recon_model, adapter=adapter),
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    initial_state = new_campaign_state(
        scan_id="scan-recon-2",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)

    memory = final_state.get("bounded_memory")
    assert memory is not None
    assert "firm policy refusal" in memory["refusal_signatures"]
    assert "instruction_override" in memory["partial_movement_techniques"]

    # The Strategist's own first brief was built AFTER Recon populated
    # memory -- verified indirectly: the Strategist's received message
    # batch (its brief) mentions the refusal signature Recon seeded.
    assert len(strategist_model.received_message_batches) == 1
    strategist_brief_text = str(strategist_model.received_message_batches[0])
    assert "firm policy refusal" in strategist_brief_text


# --- Recon failure degrades to an empty seed, campaign keeps running ------


async def test_recon_failure_leaves_campaign_running_with_empty_seed_and_recorded_note(
    fake_target_adapter, mock_target_response, scripted_chat_model
):
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1"))

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model([_strategist_output([_STATIC_CASE_ID])])
    mutator_model = scripted_chat_model([_mutator_output(1)])
    bad_recon = _AlwaysInvalidAgent()

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "recon": bad_recon,
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    initial_state = new_campaign_state(
        scan_id="scan-recon-fail-1",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)

    assert bad_recon.call_count == MAX_STRUCTURED_OUTPUT_RETRIES + 1
    # The campaign is still worth running with an unseeded Strategist --
    # never aborted, and the round's real work still completed.
    assert final_state.get("termination_reason") is not None
    assert len(final_state.get("dispatch_results", [])) == 1
    memory = final_state.get("bounded_memory")
    assert memory == {"refusal_signatures": [], "dead_techniques": [], "partial_movement_techniques": []}
    # 05-11 Rule 1/2 fix: Recon's own structured-output exhaustion is
    # recorded in `role_structural_failures`, NEVER `constraint_violations`
    # -- that field is reserved for D-95 allowlist refusals only.
    violations = final_state.get("constraint_violations", [])
    assert not any(v.get("role") == "recon" for v in violations)
    structural_failures = final_state.get("role_structural_failures", [])
    assert any(f.get("role") == "recon" for f in structural_failures)


# --- Backward compatibility: no "recon" entry is a pure no-op -------------


async def test_graph_without_recon_role_routes_straight_through(
    fake_target_adapter, mock_target_response, scripted_chat_model
):
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1"))

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model([_strategist_output([_STATIC_CASE_ID])])
    mutator_model = scripted_chat_model([_mutator_output(1)])

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    initial_state = new_campaign_state(
        scan_id="scan-recon-noop-1",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"

    final_state = await compiled.ainvoke(initial_state)

    assert final_state.get("termination_reason") is not None
    assert final_state.get("bounded_memory") == {
        "refusal_signatures": [],
        "dead_techniques": [],
        "partial_movement_techniques": [],
    }
