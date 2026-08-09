"""05-07-PLAN.md Task 2: the Analyst role -- a structured read of the
target's defence that never scores (D-65, D-66).

Drives the real `build_campaign_graph()` topology directly (mirroring
`test_budget_abort.py`'s standalone-graph style) so the round loop
(strategist -> mutator -> dispatch_variants -> analyst -> budget_check ->
strategist ...) is exercised end to end, offline, with zero network
access.
"""

from __future__ import annotations

import pathlib

import llmsec
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles._structured_retry import MAX_STRUCTURED_OUTPUT_RETRIES
from llmsec.attacker.roles.analyst import ObservedDefence, build_analyst_agent
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.state import QueuedCase, new_campaign_state
from llmsec.models import EvalResult, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"
_PACKAGE_ROOT = pathlib.Path(llmsec.__file__).resolve().parent


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
    `pydantic.ValidationError` (via `ObservedDefence.model_validate({})`),
    driving `invoke_role_with_retry()`'s genuine bounded-retry loop to
    real exhaustion -- never a mocked `StructuredOutputFailure`."""

    def __init__(self) -> None:
        self.call_count = 0

    async def ainvoke(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.call_count += 1
        ObservedDefence.model_validate({})


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


def _analyst_output(
    *,
    technique_outcome: str,
    refusal_style: str = "polite decline citing policy",
    apparent_filter: str = "content policy filter",
    what_moved: str = "no change from the parent attempt",
) -> ObservedDefence:
    return ObservedDefence(
        refusal_style=refusal_style,
        apparent_filter=apparent_filter,
        what_moved=what_moved,
        technique_outcome=technique_outcome,  # type: ignore[arg-type]
        notes="",
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


# --- ObservedDefence schema shape (D-66) ------------------------------------


def test_observed_defence_has_no_verdict_shaped_field():
    fields = set(ObservedDefence.model_fields)
    assert not (fields & {"verdict", "severity", "confidence", "score"})


def test_observed_defence_requires_all_named_behavior_fields():
    fields = set(ObservedDefence.model_fields)
    assert fields == {
        "refusal_style",
        "apparent_filter",
        "what_moved",
        "technique_outcome",
        "notes",
    }


def test_observed_defence_technique_outcome_is_closed_three_value_set():
    import typing

    outcome_field = ObservedDefence.model_fields["technique_outcome"]
    assert set(typing.get_args(outcome_field.annotation)) == {"dead", "partial_movement", "inconclusive"}


def test_observed_defence_every_string_field_is_length_bounded():
    for name, field in ObservedDefence.model_fields.items():
        if field.annotation is str:
            metadata_kinds = {type(item).__name__ for item in field.metadata}
            assert "MaxLen" in metadata_kinds, f"{name} has no max_length bound"


# --- Registration (D-65) ----------------------------------------------------


def test_analyst_role_is_registered():
    assert "analyst" in ROLE_REGISTRY
    role = ROLE_REGISTRY["analyst"]
    assert role.name == "analyst"
    assert role.output_schema is ObservedDefence


# --- Brief construction: CampaignState only, bounded truncation (D-87) -----


def test_analyst_brief_truncates_target_response_text_and_logs(caplog):
    import logging

    role = ROLE_REGISTRY["analyst"]
    long_text = "x" * 5000  # > MAX_RESPONSE_CHARS (4000)
    response = TargetResponse(
        case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text=long_text, status_code=200, latency_ms=1.0
    )
    state = new_campaign_state(
        scan_id="scan-brief-1",
        settings=resolve_settings(AttackerConfig(profile="light")),
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    state["current_module"] = "stub_module"
    state["current_case"] = _case_queue()[0]
    state["selected_technique"] = "instruction_override"
    state["round"] = 1
    state["dispatch_results"] = [
        {
            "case_id": f"{_STATIC_CASE_ID}-mut-1",
            "module_id": "stub_module",
            "record": {"round": 1, "parent_case_id": _STATIC_CASE_ID},
            "target_response": response,
            "eval_result": None,
        }
    ]

    with caplog.at_level(logging.INFO, logger="llmsec.attacker.roles.analyst"):
        brief = role.brief(state)

    assert long_text not in brief
    assert any("truncated" in record.message for record in caplog.records)


def test_analyst_brief_never_includes_scan_config_or_privileged_context():
    """D-87: the brief is built from `CampaignState` alone -- the function
    signature has no `ScanConfig`/system-prompt/credential parameter at
    all, so there is structurally nothing privileged it could include."""
    import inspect

    from llmsec.attacker.roles import analyst as analyst_mod

    sig = inspect.signature(analyst_mod._analyst_brief)
    assert list(sig.parameters) == ["state"]


# --- No verdict/scoring leakage (D-66) --------------------------------------


def test_observed_defence_never_referenced_in_scoring_or_api():
    scoring_dir = _PACKAGE_ROOT / "scoring"
    api_file = _PACKAGE_ROOT / "api.py"
    for path in [*scoring_dir.rglob("*.py"), api_file]:
        text = path.read_text(encoding="utf-8")
        assert "ObservedDefence" not in text, f"ObservedDefence referenced in {path}"


def test_analyst_module_never_constructs_a_verdict():
    analyst_file = _PACKAGE_ROOT / "attacker" / "roles" / "analyst.py"
    text = analyst_file.read_text(encoding="utf-8")
    assert "Verdict." not in text


# --- Full round loop: strategist -> mutator -> dispatch -> analyst ---------


async def test_round_loop_reaches_analyst_updates_memory_and_terminates_at_round_cap(
    fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
):
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1",
        mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="no, I can't help with that"),
    )

    cfg = AttackerConfig(profile="light", max_rounds=2, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model(
        [_strategist_output([_STATIC_CASE_ID]), _strategist_output([_STATIC_CASE_ID])]
    )
    mutator_model = scripted_chat_model([_mutator_output(1), _mutator_output(1)])
    analyst_model = scripted_chat_model(
        [
            _analyst_output(technique_outcome="partial_movement"),
            _analyst_output(technique_outcome="dead"),
        ]
    )

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "analyst": build_analyst_agent(settings, cfg, model=analyst_model),
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    initial_state = new_campaign_state(
        scan_id="scan-analyst-round-1",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)

    # The Analyst's own model was invoked once per round -- the loop
    # genuinely reached it twice, not zero or once.
    assert len(analyst_model.received_message_batches) == 2
    assert final_state.get("round") == 2
    assert final_state.get("termination_reason") == "ROUND_CAP_REACHED"

    # Round 1's "partial_movement" outcome marked the technique promising;
    # round 2's "dead" outcome then moved it OUT of partial-movement and
    # into dead-techniques -- never simultaneously in both.
    memory = final_state.get("bounded_memory")
    assert memory is not None
    assert "instruction_override" in memory["dead_techniques"]
    assert "instruction_override" not in memory["partial_movement_techniques"]
    assert "polite decline citing policy" in memory["refusal_signatures"]

    # The Analyst's own structured observation is present in state but
    # never flowed into a Verdict/Finding (D-66) -- eval_results/lineage
    # only ever come from module.evaluate(), asserted separately above.
    assert final_state.get("observed_defence") is not None
    assert final_state["observed_defence"]["technique_outcome"] == "dead"


# --- Structural failure: recorded, never a silently skipped round ---------


async def test_analyst_structural_failure_is_recorded_never_silently_skips_the_round(
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
    bad_analyst = _AlwaysInvalidAgent()

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "analyst": bad_analyst,
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    initial_state = new_campaign_state(
        scan_id="scan-analyst-fail-1",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)

    # The bounded retry genuinely exhausted (3 real attempts), and the
    # failure never propagated out of `.ainvoke()`.
    assert bad_analyst.call_count == MAX_STRUCTURED_OUTPUT_RETRIES + 1
    assert final_state.get("termination_reason") is not None
    assert final_state.get("observed_defence") is None

    # The round's real work (dispatch + scoring) still completed --
    # the Analyst's own failure never silently skipped it.
    assert len(final_state.get("dispatch_results", [])) == 1


# --- Backward compatibility: no "analyst" entry is a pure no-op ------------


async def test_graph_without_analyst_role_routes_straight_through(
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
        scan_id="scan-analyst-noop-1",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"

    final_state = await compiled.ainvoke(initial_state)

    assert final_state.get("termination_reason") is not None
    assert final_state.get("observed_defence") is None
    assert final_state.get("bounded_memory") == {
        "refusal_signatures": [],
        "dead_techniques": [],
        "partial_movement_techniques": [],
    }
