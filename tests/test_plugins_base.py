"""Tests for BaseModule ABC (src/llmsec/plugins/base.py)."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase
from llmsec.modules.data_poisoning import DataPoisoningModule
from llmsec.modules.insecure_output import InsecureOutputModule
from llmsec.modules.pii_exfiltration import PiiExfiltrationModule
from llmsec.modules.prompt_injection import PromptInjectionModule
from llmsec.modules.supply_chain import SupplyChainModule
from llmsec.modules.system_prompt_leakage import SystemPromptLeakageModule
from llmsec.plugins.base import PLUGIN_API_VERSION, BaseModule


class _FullyImplementedModule(BaseModule):
    id = "fully_implemented"
    name = "Fully Implemented Test Module"
    owasp_ref = "LLM07:2025"

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        yield TestCase(case_id="c1", prompt="p", technique_id="t1")

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        return EvalResult(
            case_id=case.case_id,
            verdict="blocked",
            confidence=1.0,
            evidence="",
            detection_layer="regex",
        )


class _PartiallyImplementedModule(BaseModule):
    id = "partially_implemented"
    name = "Partially Implemented Test Module"
    owasp_ref = "LLM07:2025"

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        yield TestCase(case_id="c1", prompt="p", technique_id="t1")

    # `evaluate` intentionally NOT implemented


def test_base_module_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseModule()


def test_fully_implemented_subclass_instantiates():
    module = _FullyImplementedModule()
    assert module.id == "fully_implemented"


def test_partially_implemented_subclass_raises_type_error():
    with pytest.raises(TypeError):
        _PartiallyImplementedModule()


# --- Phase 6 (06-01): run_standalone_audit() ABC-default contract ---------


def test_plugin_api_version_still_1_0():
    """Additive plugin evolution (PROJECT.md): PLUGIN_API_VERSION must stay
    "1.0" across run_standalone_audit()'s introduction -- it is a concrete
    default, never a breaking change."""
    assert PLUGIN_API_VERSION == "1.0"


def test_run_standalone_audit_present_on_base_module():
    assert hasattr(BaseModule, "run_standalone_audit")


def test_run_standalone_audit_not_abstract():
    """run_standalone_audit() must be a concrete default -- never the third
    abstract method -- so every existing module and third-party plugin
    inherits it unedited."""
    assert "run_standalone_audit" not in BaseModule.__abstractmethods__


def test_generate_cases_and_evaluate_still_abstract():
    """generate_cases() and evaluate() remain the ONLY two abstract methods
    -- run_standalone_audit()'s introduction must never quietly promote
    either of them out of, or a third method into, the abstract set."""
    assert "generate_cases" in BaseModule.__abstractmethods__
    assert "evaluate" in BaseModule.__abstractmethods__
    assert len(BaseModule.__abstractmethods__) == 2


async def test_base_module_default_run_standalone_audit_yields_nothing():
    module = _FullyImplementedModule()
    results = [result async for result in module.run_standalone_audit(_context())]
    assert results == []


def _context() -> ScanContext:
    return ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="")


@pytest.mark.parametrize(
    "module_cls",
    [
        pytest.param(SystemPromptLeakageModule, id="system_prompt_leakage"),
        pytest.param(PromptInjectionModule, id="prompt_injection"),
        pytest.param(PiiExfiltrationModule, id="pii_exfiltration"),
        pytest.param(InsecureOutputModule, id="insecure_output"),
    ],
)
async def test_v1_0_modules_instantiate_bare_and_inherit_no_op_audit(module_cls):
    """Every v1.0 module (system_prompt_leakage, prompt_injection,
    pii_exfiltration, insecure_output) instantiates with zero arguments
    (D-10) and yields ZERO items from the inherited
    run_standalone_audit(context) -- proving none of these four files
    needed editing for this phase's ABC-default addition."""
    module = module_cls()
    results = [result async for result in module.run_standalone_audit(_context())]
    assert results == []


# --- Phase 7 (07-03): run_direct_probe() ABC-default contract -------------


class _StubAdapter:
    """Minimal `TargetAdapter`-shaped stub -- unused by every one of the
    six pre-existing modules' inherited `run_direct_probe()` default, but
    required by the method's signature."""

    supports_system_prompt_override = False
    supports_multi_turn = False

    async def send(self, case: TestCase) -> TargetResponse:  # pragma: no cover
        raise AssertionError("the inherited no-op default must never call adapter.send()")

    async def send_conversation(self, case: TestCase, stop_when=None) -> TargetResponse:  # pragma: no cover
        raise AssertionError(
            "the inherited no-op default must never call adapter.send_conversation()"
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class TestRunDirectProbe:
    """07-03-PLAN.md Task 3: mirrors the existing Phase 6
    `run_standalone_audit()` test block shape exactly -- presence,
    not-abstract, exactly-two-abstract-methods, `PLUGIN_API_VERSION`
    unchanged, the base default yields nothing, and a parametrized test
    over all SIX pre-existing module classes."""

    def test_run_direct_probe_present_on_base_module(self):
        assert hasattr(BaseModule, "run_direct_probe")

    def test_run_direct_probe_not_abstract(self):
        """run_direct_probe() must be a concrete default -- never a third
        abstract method -- so every existing module and third-party plugin
        inherits it unedited."""
        assert "run_direct_probe" not in BaseModule.__abstractmethods__

    def test_generate_cases_and_evaluate_remain_the_only_two_abstract_methods(self):
        """run_direct_probe()'s introduction must never quietly promote
        generate_cases()/evaluate() out of, or a third method into, the
        abstract set -- exactly two, unchanged."""
        assert BaseModule.__abstractmethods__ == frozenset({"generate_cases", "evaluate"})

    def test_plugin_api_version_still_1_0(self):
        assert PLUGIN_API_VERSION == "1.0"

    async def test_base_module_default_run_direct_probe_yields_nothing(self):
        module = _FullyImplementedModule()
        results = [
            result async for result in module.run_direct_probe(_context(), _StubAdapter())
        ]
        assert results == []

    @pytest.mark.parametrize(
        "module_cls",
        [
            pytest.param(SystemPromptLeakageModule, id="system_prompt_leakage"),
            pytest.param(PromptInjectionModule, id="prompt_injection"),
            pytest.param(PiiExfiltrationModule, id="pii_exfiltration"),
            pytest.param(InsecureOutputModule, id="insecure_output"),
            pytest.param(SupplyChainModule, id="supply_chain"),
            pytest.param(DataPoisoningModule, id="data_poisoning"),
        ],
    )
    async def test_all_six_pre_existing_modules_instantiate_bare_and_inherit_no_op_direct_probe(
        self, module_cls
    ):
        """Every one of the six pre-existing built-in modules instantiates
        with zero arguments (D-10) and yields ZERO items from the inherited
        run_direct_probe(context, adapter) -- proving none of these six
        files needed editing for this phase's ABC-default addition."""
        module = module_cls()
        results = [
            result async for result in module.run_direct_probe(_context(), _StubAdapter())
        ]
        assert results == []
