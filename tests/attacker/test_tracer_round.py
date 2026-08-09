"""End-to-end tracer test (05-03-PLAN.md Task 2): one static failure becomes
one scored, lineage-carrying mutated finding, entirely offline.

Drives `run_attacker_campaign()` with the scripted role models and fake
adapter from `conftest.py`, plus a stub module whose static result was
`BLOCKED` and whose `evaluate()` returns `FULL_COMPROMISE` for the mutated
payload.
"""

from __future__ import annotations

from typing import AsyncIterator

from llmsec.attacker.config import AttackerConfig
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import (
    MutatedVariant,
    MutatorOutput,
    build_mutator_agent,
)
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.runner import run_attacker_campaign
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"


class _StubModule(BaseModule):
    """A minimal `uses_attacker_llm=True` module: one static case, and an
    `evaluate()` that scores the static case `BLOCKED` and every mutated
    variant `FULL_COMPROMISE` -- except any case_id matching
    `fail_case_ids`, whose `evaluate()` raises (T-01-18 containment probe).
    """

    id = "stub_module"
    name = "Stub Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    def __init__(self, fail_case_ids: frozenset[str] = frozenset()) -> None:
        self.fail_case_ids = fail_case_ids
        self.evaluated_cases: list[TestCase] = []

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="stub parent payload", technique_id=_STATIC_TECHNIQUE_ID
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        self.evaluated_cases.append(case)
        if case.case_id in self.fail_case_ids:
            raise RuntimeError("simulated evaluate() failure")
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
            confidence=0.95,
            evidence=f"complied: {response.raw_text}",
            detection_layer="regex",
        )


def _make_config(
    *, variants_per_round: int = 2, max_rounds: int = 1, output_dir: str = "./llmsec_reports"
) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        output_dir=output_dir,
        attacker=AttackerConfig(
            enabled=True, profile="light", variants_per_round=variants_per_round, max_rounds=max_rounds
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


def _patch_roles(monkeypatch, scripted_chat_model, *, strategist_output, mutator_output):
    """Inject scripted models into the registered strategist/mutator roles
    via each role's own `model=` test-injection hook, so
    `run_attacker_campaign()` exercises the real production code path
    (registry lookup, `build_*_agent()`, `invoke_role_with_retry()`,
    `graph.py`'s nodes) with zero network access."""
    strategist_model = scripted_chat_model([strategist_output])
    mutator_model = scripted_chat_model([mutator_output])

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


async def test_tracer_round_end_to_end(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    module = _StubModule()
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1",
        mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure, here it is"),
    )
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-2",
        mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-2", raw_text="sure, here it is too"),
    )

    _patch_roles(
        monkeypatch,
        scripted_chat_model,
        strategist_output=_strategist_output([_STATIC_CASE_ID]),
        mutator_output=_mutator_output(2),
    )
    patch_analyst_and_recon_roles()

    config = _make_config(variants_per_round=2, max_rounds=1, output_dir=str(tmp_path))
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-tracer-1",
    )

    # The static case was queued and its technique reached the Mutator --
    # both mutated variants were evaluated (evaluate() is never re-invoked
    # for the static case itself; its BLOCKED verdict came from
    # `static_results`, per D-93's "orchestrator.py stays untouched"
    # design -- see runner.py's module docstring).
    assert {c.case_id for c in module.evaluated_cases} == {
        f"{_STATIC_CASE_ID}-mut-1",
        f"{_STATIC_CASE_ID}-mut-2",
    }

    # Exactly one recorded result per dispatched variant.
    assert len(result.eval_results) == 2
    module_ids = {module_id for module_id, _ in result.eval_results}
    assert module_ids == {"stub_module"}

    verdicts = {case_id: eval_result.verdict for case_id, eval_result in (
        (eval_result.case_id, eval_result) for _, eval_result in result.eval_results
    )}
    assert verdicts[f"{_STATIC_CASE_ID}-mut-1"] == Verdict.FULL_COMPROMISE
    assert verdicts[f"{_STATIC_CASE_ID}-mut-2"] == Verdict.FULL_COMPROMISE

    # The lineage map resolves each generated case id to a record whose
    # parent_case_id is the static case (D-90) -- never recovered by
    # string-parsing the generated case_id.
    assert set(result.lineage.keys()) == {f"{_STATIC_CASE_ID}-mut-1", f"{_STATIC_CASE_ID}-mut-2"}
    for record in result.lineage.values():
        assert record["parent_case_id"] == _STATIC_CASE_ID
        assert record["parent_technique_id"] == _STATIC_TECHNIQUE_ID
        assert record["round"] == 1
        assert record["contributing_agent"] == "mutator"

    # The campaign terminated with a termination_reason (D-70/D-72).
    assert result.final_state.get("termination_reason") is not None

    # Both variants were actually sent through the fake adapter. Recon
    # (05-07) also dispatches its own `recon-probe-*`-prefixed cases
    # through the SAME adapter once per campaign -- excluded here since
    # this assertion is about the Mutator's own dispatched variants.
    sent_case_ids = {
        case.case_id for case in adapter.sent_cases if not case.case_id.startswith("recon-probe-")
    }
    assert sent_case_ids == {f"{_STATIC_CASE_ID}-mut-1", f"{_STATIC_CASE_ID}-mut-2"}


async def test_tracer_round_containment_one_failing_variant_degrades_not_cancels(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """T-01-18 re-derived: one of three variant dispatches raises inside
    evaluate() -- three recorded results, the failing one degraded to
    UNCERTAIN, never cancelling its siblings."""
    failing_case_id = f"{_STATIC_CASE_ID}-mut-2"
    module = _StubModule(fail_case_ids=frozenset({failing_case_id}))
    modules = {"stub_module": module}

    adapter = fake_target_adapter()
    for i in range(1, 4):
        case_id = f"{_STATIC_CASE_ID}-mut-{i}"
        adapter.queue_response(case_id, mock_target_response(case_id=case_id, raw_text="sure"))

    _patch_roles(
        monkeypatch,
        scripted_chat_model,
        strategist_output=_strategist_output([_STATIC_CASE_ID]),
        mutator_output=_mutator_output(3),
    )
    patch_analyst_and_recon_roles()

    config = _make_config(variants_per_round=3, max_rounds=1, output_dir=str(tmp_path))
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-tracer-2",
    )

    assert len(result.eval_results) == 3
    verdict_by_case_id = {
        eval_result.case_id: eval_result.verdict for _, eval_result in result.eval_results
    }
    assert verdict_by_case_id[f"{_STATIC_CASE_ID}-mut-1"] == Verdict.FULL_COMPROMISE
    assert verdict_by_case_id[failing_case_id] == Verdict.UNCERTAIN
    assert verdict_by_case_id[f"{_STATIC_CASE_ID}-mut-3"] == Verdict.FULL_COMPROMISE

    # The degraded result is recorded, not dropped, and lineage still
    # resolves it.
    assert failing_case_id in result.lineage
