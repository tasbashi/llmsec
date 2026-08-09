"""D-94 gate / AT-7: bounded termination (05-10-PLAN.md Task 2).

Round control lives in graph topology (`round_cap_edge`), never in an
agent's own reasoning (D-70/D-72) -- every campaign reaches termination
within the round cap plus one graph step (the `finalize` node), and the
terminal state always carries a `termination_reason`. Parametrized over
several round-cap values so the bound is proven at more than one point, not
just the default profile's own cap.

A Strategist early-exhaust reason code (D-72) wins over the hard round
cap: a campaign whose Strategist emits `TECHNIQUES_EXHAUSTED` on round 1
terminates with THAT code well before reaching a much larger configured
cap, proving "the cap is the contract; early exit is the efficiency" holds
in practice, not just in the docstring.
"""

from __future__ import annotations

import pytest

from llmsec.attacker.config import AttackerConfig
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.runner import run_attacker_campaign
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

_STATIC_CASE_ID = "TERM-001"
_STATIC_TECHNIQUE_ID = "TERM-001"


class _StubModule(BaseModule):
    id = "stub_module"
    name = "Stub Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    async def generate_cases(self, context: ScanContext):
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="stub parent payload", technique_id=_STATIC_TECHNIQUE_ID
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        if case.case_id == _STATIC_CASE_ID:
            return EvalResult(
                case_id=case.case_id,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            )
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.9,
            evidence="complied",
            detection_layer="regex",
        )


def _make_config(*, max_rounds: int, output_dir: str) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        output_dir=output_dir,
        attacker=AttackerConfig(enabled=True, profile="light", variants_per_round=1, max_rounds=max_rounds),
    )


def _static_results() -> list[tuple[str, EvalResult]]:
    return [
        (
            "stub_module",
            EvalResult(
                case_id=_STATIC_CASE_ID,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            ),
        )
    ]


def _strategist_output(*, reason_code=None) -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
        ordered_case_ids=[_STATIC_CASE_ID],
        escalate=False,
        reason_code=reason_code,
        rationale="rationale",
    )


def _mutator_output() -> MutatorOutput:
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload="mutated payload",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale="rationale",
            )
        ]
    )


def _patch_roles(monkeypatch, scripted_chat_model, *, strategist_outputs, mutator_outputs):
    strategist_model = scripted_chat_model(strategist_outputs)
    mutator_model = scripted_chat_model(mutator_outputs)
    monkeypatch.setattr(
        ROLE_REGISTRY["strategist"],
        "build",
        lambda settings, cfg, *, model=None: build_strategist_agent(
            settings, cfg, model=strategist_model
        ),
    )
    monkeypatch.setattr(
        ROLE_REGISTRY["mutator"],
        "build",
        lambda settings, cfg, *, model=None: build_mutator_agent(settings, cfg, model=mutator_model),
    )
    return strategist_model, mutator_model


@pytest.mark.parametrize("max_rounds", [1, 2, 3])
async def test_campaign_terminates_within_round_cap_plus_one_step_with_reason(
    max_rounds,
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    module = _StubModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1",
        mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure, here it is"),
    )

    # Never an early-exit reason code -- every round is scripted to run the
    # campaign all the way to the hard round cap, so this test isolates the
    # CAP itself, not the Strategist's own early-exhaust behaviour.
    _patch_roles(
        monkeypatch,
        scripted_chat_model,
        strategist_outputs=[_strategist_output() for _ in range(max_rounds)],
        mutator_outputs=[_mutator_output() for _ in range(max_rounds)],
    )
    patch_analyst_and_recon_roles()

    config = _make_config(max_rounds=max_rounds, output_dir=str(tmp_path))
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules={"stub_module": module},
        static_results=_static_results(),
        scan_id=f"scan-term-{max_rounds}",
    )

    # Never overshoots the round cap -- termination is reached within the
    # cap plus at most the one extra `finalize` graph step, never a second
    # round beyond it.
    assert result.final_state.get("round") == max_rounds
    assert result.final_state.get("round", 0) <= max_rounds

    # The terminal state always carries a non-null termination reason --
    # never a campaign that simply stops with no recorded cause.
    assert result.final_state.get("termination_reason") is not None
    assert result.final_state.get("termination_reason") == "ROUND_CAP_REACHED"


async def test_strategist_early_exhaust_reason_code_wins_over_hard_cap(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """A Strategist reason code emitted on round 1 terminates the campaign
    with THAT code -- never the generic `ROUND_CAP_REACHED` -- well before
    a much larger configured hard cap is ever reached."""
    large_cap = 10
    module = _StubModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1",
        mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure, here it is"),
    )

    _patch_roles(
        monkeypatch,
        scripted_chat_model,
        strategist_outputs=[_strategist_output(reason_code="TECHNIQUES_EXHAUSTED")],
        mutator_outputs=[_mutator_output()],
    )
    patch_analyst_and_recon_roles()

    config = _make_config(max_rounds=large_cap, output_dir=str(tmp_path))
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules={"stub_module": module},
        static_results=_static_results(),
        scan_id="scan-term-early-exhaust",
    )

    assert result.final_state.get("round") == 1
    assert result.final_state.get("round", 0) < large_cap
    assert result.final_state.get("termination_reason") == "TECHNIQUES_EXHAUSTED"
    assert result.final_state.get("termination_reason") != "ROUND_CAP_REACHED"
