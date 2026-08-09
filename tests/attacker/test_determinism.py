"""D-94 gate / AT-8: deterministic reproducibility (05-10-PLAN.md Task 2).

Two independent claims:
  1. Module iteration order is identical across repeated runs of the same
     configuration (D-68's fixed `sorted()` order, never set/dict
     iteration) -- verified both by re-deriving `eligible_module_ids`
     directly and by re-running the full campaign five times and comparing
     the audit trail's own `campaign_start` line (`record_campaign_start()`,
     05-06 Task 1).
  2. No round emits two variants with byte-identical payloads for the same
     case. Since `MutatorOutput` now enforces this at the schema boundary
     (`_validate_distinct_payloads()`, added in this plan -- see
     05-10-SUMMARY.md's Deviations section), a scripted model CAPABLE of
     returning a duplicate is provably rejected and retried rather than
     ever reaching dispatch: the fault-injection half of this gate proves
     the enforcement is real, not merely that this test's own fixtures
     happen not to duplicate.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import (
    MutatedVariant,
    MutatorOutput,
    build_mutator_agent,
)
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.runner import run_attacker_campaign
from llmsec.attacker.state import QueuedCase, new_campaign_state
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

from .conftest import ScriptedToolCallChatModel

_STATIC_CASE_ID = "DET-001"
_STATIC_TECHNIQUE_ID = "DET-001"


class _StubModule(BaseModule):
    def __init__(self, module_id: str) -> None:
        self.id = module_id
        self.name = module_id
        self.owasp_ref = "LLM00:2025"
        self.uses_attacker_llm = True

    async def generate_cases(self, context: ScanContext):
        yield TestCase(
            case_id=f"{self.id}-{_STATIC_CASE_ID}",
            prompt="stub parent payload",
            technique_id=_STATIC_TECHNIQUE_ID,
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.BLOCKED,
            confidence=0.9,
            evidence="refused",
            detection_layer="regex",
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


def _strategist_output() -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
        ordered_case_ids=[_STATIC_CASE_ID],
        escalate=False,
        reason_code=None,
        rationale="rationale",
    )


def _mutator_output(*, distinct: bool) -> MutatorOutput:
    if distinct:
        payloads = ["variant payload one", "variant payload two", "variant payload three"]
    else:
        payloads = ["duplicate payload", "duplicate payload", "duplicate payload"]
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload=payload,
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale=f"rationale {i}",
            )
            for i, payload in enumerate(payloads)
        ]
    )


# --- Claim 1: deterministic module ordering ---------------------------------


def test_eligible_module_order_is_a_fixed_sort_not_dict_iteration():
    """`run_attacker_campaign()`'s own `eligible_module_ids = sorted(...)`
    line (D-78) -- re-derived here directly against a deliberately
    out-of-order/reversed input dict, proving the result is a real sort and
    not incidental dict-insertion order."""
    modules = {
        "zeta_module": _StubModule("zeta_module"),
        "alpha_module": _StubModule("alpha_module"),
        "mid_module": _StubModule("mid_module"),
    }
    eligible = sorted(module_id for module_id, module in modules.items() if module.uses_attacker_llm)
    assert eligible == ["alpha_module", "mid_module", "zeta_module"]

    # Insertion order reversed -- the sorted result must be unchanged.
    reversed_modules = dict(reversed(list(modules.items())))
    eligible_reversed = sorted(
        module_id for module_id, module in reversed_modules.items() if module.uses_attacker_llm
    )
    assert eligible_reversed == eligible


async def test_five_repeated_runs_produce_an_identical_module_order(
    fake_target_adapter, mock_target_response, monkeypatch, scripted_chat_model,
    patch_analyst_and_recon_roles, tmp_path,
):
    modules = {
        "zeta_module": _StubModule("zeta_module"),
        "alpha_module": _StubModule("alpha_module"),
        "mid_module": _StubModule("mid_module"),
    }
    static_results = [
        (
            module_id,
            EvalResult(
                case_id=f"{module_id}-{_STATIC_CASE_ID}",
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            ),
        )
        for module_id in modules
    ]

    recorded_orders: list[list[str]] = []
    for i in range(5):
        adapter = fake_target_adapter()
        for module_id in modules:
            case_id = f"{module_id}-{_STATIC_CASE_ID}-mut-1"
            adapter.queue_response(case_id, mock_target_response(case_id=case_id, raw_text="sure"))

        strategist_model = scripted_chat_model(
            [_strategist_output(), _strategist_output(), _strategist_output()]
        )
        mutator_model = scripted_chat_model(
            [_mutator_output(distinct=True), _mutator_output(distinct=True), _mutator_output(distinct=True)]
        )
        monkeypatch.setattr(
            ROLE_REGISTRY["strategist"],
            "build",
            lambda settings, cfg, *, model=None, _m=strategist_model: build_strategist_agent(
                settings, cfg, model=_m
            ),
        )
        monkeypatch.setattr(
            ROLE_REGISTRY["mutator"],
            "build",
            lambda settings, cfg, *, model=None, _m=mutator_model: build_mutator_agent(settings, cfg, model=_m),
        )
        patch_analyst_and_recon_roles()

        config = ScanConfig(
            target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
            output_dir=str(tmp_path / f"run-{i}"),
            attacker=AttackerConfig(enabled=True, profile="light", variants_per_round=3, max_rounds=1),
        )
        result = await run_attacker_campaign(
            config=config,
            adapter=adapter,
            modules=modules,
            static_results=static_results,
            scan_id=f"scan-det-order-{i}",
        )

        # The `campaign_start` audit line is where `record_campaign_start()`
        # (05-06 Task 1) stamps the deterministic module order -- read back
        # from the persisted file, not from in-memory state, so this
        # assertion covers the SAME artifact an operator would inspect.
        lines = [json.loads(line) for line in result.audit_path.read_text(encoding="utf-8").splitlines()]
        start_line = next(line for line in lines if line["event"] == "campaign_start")
        recorded_orders.append(json.loads(start_line["content"])["module_order"])

    assert len(recorded_orders) == 5
    assert all(order == recorded_orders[0] for order in recorded_orders)
    assert recorded_orders[0] == ["alpha_module", "mid_module", "zeta_module"]


# --- Claim 2: per-round payload uniqueness ----------------------------------


def test_mutator_output_rejects_duplicate_payloads_within_one_response():
    """The fault-injection half of this gate: a scripted model CAPABLE of
    returning duplicate payloads is rejected at schema validation --
    `MutatorOutput`'s own `_validate_distinct_payloads()` -- rather than
    ever reaching dispatch. Proves the rejection is real, not merely that
    this test file's OTHER fixtures happen not to duplicate."""
    with pytest.raises(ValidationError, match="distinct"):
        _mutator_output(distinct=False)


async def test_duplicate_mutator_response_is_rejected_never_dispatched(
    fake_target_adapter, mock_target_response
):
    """End to end: a Mutator whose EVERY scripted attempt returns duplicate
    payloads exhausts the bounded retry and is recorded as a structural
    failure (`mutator_node`'s own `except StructuredOutputFailure` branch)
    -- zero variants dispatched, never the duplicates dispatched anyway."""
    from llmsec.attacker.roles._structured_retry import MAX_STRUCTURED_OUTPUT_RETRIES

    module = _StubModule("stub_module")
    adapter = fake_target_adapter()

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=3)
    settings = resolve_settings(cfg)

    strategist_model = ScriptedToolCallChatModel(
        script=[
            {
                "name": "StrategistOutput",
                "args": _strategist_output().model_dump(mode="json"),
            }
        ]
    )
    # Every attempt's raw tool-call args are duplicate-shaped -- since
    # `ScriptedToolCallChatModel` returns them verbatim (no validation of
    # its own), the SCHEMA is what must reject them, exactly what this test
    # is proving.
    duplicate_args = {
        "variants": [
            {
                "payload": "duplicate payload",
                "technique_family": "instruction_override",
                "parent_technique_id": _STATIC_TECHNIQUE_ID,
                "rationale": f"rationale {i}",
            }
            for i in range(3)
        ]
    }
    mutator_model = ScriptedToolCallChatModel(
        script=[
            {"name": "MutatorOutput", "args": duplicate_args}
            for _ in range(MAX_STRUCTURED_OUTPUT_RETRIES + 1)
        ]
    )

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "crescendo": build_mutator_agent(settings, cfg, model=ScriptedToolCallChatModel(script=[])),
    }
    compiled = build_campaign_graph(
        roles=roles, adapter=adapter, modules={"stub_module": module}, max_concurrency=5
    )

    initial_state = new_campaign_state(
        scan_id="scan-det-dup",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)

    # Every attempt was genuinely made (the bounded retry exhausted for
    # real), and NOTHING was dispatched as a result -- the duplicate-laden
    # response never reached `dispatch_variants_node`.
    assert mutator_model._call_index == MAX_STRUCTURED_OUTPUT_RETRIES + 1  # noqa: SLF001
    assert final_state.get("dispatch_results", []) == []
    assert adapter.sent_cases == []
    assert final_state.get("termination_reason") is not None
