"""05-04-PLAN.md Task 2: cap-trip stop semantics -- dispatch what was paid
for, bound the overshoot, record the truncation (D-83).

Drives `run_attacker_campaign()` end to end (offline, scripted roles) with
`agent_call_ceiling` set low enough to trip after exactly one round's
Strategist+Mutator calls -- deterministic and fixture-friendly, unlike a
dollar-cap trip (the scripted models never report real `usage_metadata`,
so attacker-side dollar spend always reads 0.0 against them; D-80's
independent call ceiling is exactly the mechanism that still bounds an
unpriced/never-priced campaign, which is precisely this test's own
scenario).
"""

from __future__ import annotations

from llmsec.attacker.budget import mark_truncated, truncation_disclosure
from llmsec.attacker.config import AttackerConfig
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.runner import run_attacker_campaign
from llmsec.attacker.state import BudgetLedger
from llmsec.config import ScanConfig, TargetConfig
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
        if case.case_id == _STATIC_CASE_ID:
            return EvalResult(
                case_id=case.case_id,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            )
        # Every mutated variant is BLOCKED too (zero findings, per D-83's
        # "produced for a campaign with zero findings" requirement).
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.BLOCKED,
            confidence=0.9,
            evidence="refused",
            detection_layer="regex",
        )


def _make_config(
    *,
    agent_call_ceiling: int | None,
    max_rounds: int = 5,
    variants_per_round: int = 2,
    output_dir: str = "./llmsec_reports",
) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        output_dir=output_dir,
        attacker=AttackerConfig(
            enabled=True,
            profile="light",
            variants_per_round=variants_per_round,
            max_rounds=max_rounds,
            agent_call_ceiling=agent_call_ceiling,
        ),
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


def _strategist_output(case_ids: list[str]) -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
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


def _patch_roles(monkeypatch, *, strategist_outputs: list[StrategistOutput], mutator_outputs: list[MutatorOutput]):
    """Scripts enough responses for MULTIPLE rounds -- if the cap trips
    early (as this file's tests expect), the extra scripted entries are
    simply never consumed."""
    strategist_model = ScriptedToolCallChatModel(
        script=[
            {"name": type(out).__name__, "args": out.model_dump(mode="json")}
            for out in strategist_outputs
        ]
    )
    mutator_model = ScriptedToolCallChatModel(
        script=[
            {"name": type(out).__name__, "args": out.model_dump(mode="json")}
            for out in mutator_outputs
        ]
    )
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


async def test_cap_trip_dispatches_already_generated_variants_and_stops_further_rounds(
    monkeypatch, fake_target_adapter, mock_target_response, patch_analyst_and_recon_roles, tmp_path
):
    """agent_call_ceiling=2 trips immediately after round 1's Strategist
    (call 1) + Mutator (call 2) -- variants ALREADY generated this round
    are still dispatched, and no further attacker-side call is made."""
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-2", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-2")
    )

    strategist_model, mutator_model = _patch_roles(
        monkeypatch,
        strategist_outputs=[_strategist_output([_STATIC_CASE_ID]) for _ in range(5)],
        mutator_outputs=[_mutator_output(2) for _ in range(5)],
    )
    patch_analyst_and_recon_roles()

    config = _make_config(
        agent_call_ceiling=2, max_rounds=5, variants_per_round=2, output_dir=str(tmp_path)
    )
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-stop-1",
    )

    # Both round-1 variants were still dispatched and recorded, despite the
    # cap tripping the moment round 1's Mutator call landed. Recon's own
    # `recon-probe-*`-prefixed dispatches are excluded -- this assertion
    # is about the Mutator's own dispatched variants.
    assert len(result.eval_results) == 2
    variant_case_ids = {
        c.case_id for c in adapter.sent_cases if not c.case_id.startswith("recon-probe-")
    }
    assert variant_case_ids == {
        f"{_STATIC_CASE_ID}-mut-1",
        f"{_STATIC_CASE_ID}-mut-2",
    }

    # No further attacker-side model call happened -- the scripted models
    # each had 5 rounds' worth of responses queued; only 1 round's worth
    # was ever consumed.
    assert len(strategist_model.received_message_batches) == 1
    assert len(mutator_model.received_message_batches) == 1

    ledger = result.final_state.get("budget_ledger")
    assert ledger is not None
    assert ledger["truncated"] is True
    assert ledger["overshoot_rounds"] == 1
    assert ledger["overshoot_rounds"] <= 1
    assert result.final_state.get("termination_reason") == "BUDGET_CAP_EXCEEDED"

    # Exactly zero further rounds began.
    assert result.final_state.get("round") == 1

    # The truncation disclosure is present, even though every case in this
    # scenario is BLOCKED (zero findings would ever be produced downstream).
    assert any("hard budget cap" in note for note in result.limitations)


async def test_campaign_that_never_trips_the_cap_has_no_truncation_disclosure(
    monkeypatch, fake_target_adapter, mock_target_response, patch_analyst_and_recon_roles, tmp_path
):
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-2", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-2")
    )

    _patch_roles(
        monkeypatch,
        strategist_outputs=[_strategist_output([_STATIC_CASE_ID])],
        mutator_outputs=[_mutator_output(2)],
    )
    patch_analyst_and_recon_roles()

    # High ceiling/cap (the light profile's defaults), max_rounds=1 -- the
    # campaign ends via the ordinary round cap, never the budget cap.
    config = _make_config(
        agent_call_ceiling=None, max_rounds=1, variants_per_round=2, output_dir=str(tmp_path)
    )
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-stop-2",
    )

    ledger = result.final_state.get("budget_ledger")
    assert ledger is not None
    assert ledger["truncated"] is False
    assert ledger["overshoot_rounds"] == 0
    assert result.final_state.get("termination_reason") == "ROUND_CAP_REACHED"
    assert result.limitations == []


# --- Pure unit coverage of mark_truncated()/truncation_disclosure() -----


def _fresh_ledger() -> BudgetLedger:
    return BudgetLedger(
        cap_usd=1.0,
        warn_usd=0.75,
        spent_usd=0.0,
        attacker_spent_usd=0.0,
        target_spent_usd=0.0,
        agent_calls=0,
        agent_call_ceiling=100,
        per_role={},
        truncated=False,
        overshoot_rounds=0,
        warn_approved=False,
        unpriced_calls=0,
    )


def test_mark_truncated_sets_flag_and_overshoot_rounds():
    ledger = _fresh_ledger()
    mark_truncated(ledger, overshoot_rounds=1)
    assert ledger["truncated"] is True
    assert ledger["overshoot_rounds"] == 1


def test_truncation_disclosure_none_when_not_truncated():
    ledger = _fresh_ledger()
    assert truncation_disclosure(ledger) is None


def test_truncation_disclosure_present_and_names_cap_and_spend_when_truncated():
    ledger = _fresh_ledger()
    ledger["spent_usd"] = 1.20
    mark_truncated(ledger, overshoot_rounds=1)
    disclosure = truncation_disclosure(ledger)
    assert disclosure is not None
    assert "1.00" in disclosure  # cap_usd
    assert "1.20" in disclosure  # spent_usd
    assert "at most one round" in disclosure


def test_truncation_disclosure_wording_matches_render_cost_notice_overshoot_sentence():
    from llmsec.attacker.budget import (
        CostEstimate,
        _OVERSHOOT_BOUND_SENTENCE,
        render_cost_notice,
    )

    ledger = _fresh_ledger()
    mark_truncated(ledger, overshoot_rounds=1)
    disclosure = truncation_disclosure(ledger)
    assert disclosure is not None
    assert _OVERSHOOT_BOUND_SENTENCE in disclosure

    notice = render_cost_notice(
        CostEstimate(
            typical_usd=0.1, worst_case_usd=0.5, cap_usd=1.0, assumed_calls_typical=4, assumed_calls_worst=10
        )
    )
    assert _OVERSHOOT_BOUND_SENTENCE in notice
