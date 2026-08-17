"""Tests for per-module `uses_attacker_llm` opt-in (ATK-01, D-77) and the
additive-plugin-evolution invariant (`PLUGIN_API_VERSION` stays "1.0",
`BaseModule`/`TargetAdapter` abstract-method surface unchanged)."""

from llmsec.adapters.base import TargetAdapter
from llmsec.modules.excessive_agency import ExcessiveAgencyModule
from llmsec.modules.insecure_output import InsecureOutputModule
from llmsec.modules.pii_exfiltration import PiiExfiltrationModule
from llmsec.modules.prompt_injection import PromptInjectionModule
from llmsec.modules.system_prompt_leakage import SystemPromptLeakageModule
from llmsec.modules.vector_embedding_weaknesses import VectorEmbeddingWeaknessesModule
from llmsec.plugins.base import PLUGIN_API_VERSION, BaseModule


def test_prompt_injection_opts_in():
    assert PromptInjectionModule.uses_attacker_llm is True


def test_pii_exfiltration_opts_in():
    assert PiiExfiltrationModule.uses_attacker_llm is True


def test_system_prompt_leakage_stays_opted_out():
    assert SystemPromptLeakageModule.uses_attacker_llm is False


def test_insecure_output_stays_opted_out():
    assert InsecureOutputModule.uses_attacker_llm is False


def test_vector_embedding_weaknesses_opts_in():
    """08-02-PLAN.md Task 3: `vector_embedding_weaknesses` sets
    `uses_attacker_llm = True` (D-03), matching the three-call-site
    allowlist widening `VectorContextTechniqueVector` receives."""
    assert VectorEmbeddingWeaknessesModule.uses_attacker_llm is True


def test_excessive_agency_opts_in():
    """08-04-PLAN.md Task 3: `excessive_agency` sets `uses_attacker_llm =
    True` (D-03), matching the three-call-site allowlist widening
    `AgencyClass` receives (08-02)."""
    assert ExcessiveAgencyModule.uses_attacker_llm is True


def test_plugin_api_version_unchanged():
    assert PLUGIN_API_VERSION == "1.0"


def test_base_module_abstract_method_surface_unchanged():
    assert BaseModule.__abstractmethods__ == frozenset({"generate_cases", "evaluate"})


def test_target_adapter_abstract_method_surface_unchanged():
    assert TargetAdapter.__abstractmethods__ == frozenset({"send", "health_check", "close"})
