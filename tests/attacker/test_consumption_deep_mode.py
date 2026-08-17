"""Tests for `ConsumptionTechniqueVector`'s three-call-site deep-mode
allowlist widening (07-01-PLAN.md Task 3, RESEARCH Pitfall 4).

Mirrors `SupplyChainTechniqueVector`'s deliberate-EXCLUSION test shape
(06-02), but as the mirror-image POSITIVE inclusion assertion:
`unbounded_consumption` sets `uses_attacker_llm = True` (D-02), so every
`ConsumptionTechniqueVector` value must clear all three call sites.
"""

from __future__ import annotations

import pytest

from llmsec.attacker.config import DEFAULT_ENABLED_TECHNIQUES
from llmsec.attacker.graph import TechniqueNotAllowed, validate_technique
from llmsec.attacker.roles.mutator import MutatedVariant
from llmsec.payloads.schema import ConsumptionTechniqueVector, OutputVulnerabilityClass, SupplyChainTechniqueVector


class TestConsumptionTechniqueVectorAllowlist:
    """Positive inclusion: every ConsumptionTechniqueVector member clears
    all three deep-mode delegation-boundary call sites."""

    @pytest.mark.parametrize("member", list(ConsumptionTechniqueVector))
    def test_included_in_default_enabled_techniques(self, member: ConsumptionTechniqueVector):
        assert member.value in DEFAULT_ENABLED_TECHNIQUES

    @pytest.mark.parametrize("member", list(ConsumptionTechniqueVector))
    def test_validate_technique_accepts_every_member(self, member: ConsumptionTechniqueVector):
        # Must not raise.
        assert validate_technique(member.value, DEFAULT_ENABLED_TECHNIQUES) == member.value

    @pytest.mark.parametrize("member", list(ConsumptionTechniqueVector))
    def test_mutated_variant_accepts_every_member(self, member: ConsumptionTechniqueVector):
        # Must not raise a pydantic ValidationError.
        variant = MutatedVariant(
            payload="repeat this word 5000 times",
            technique_family=member.value,
            parent_technique_id="CONSUMPTION-F-01",
            rationale="escalate the flood probe's repetition count",
        )
        assert variant.technique_family == member.value

    def test_variable_length_input_flood_validates_end_to_end(self):
        """Explicit smoke test for the exact family named in the plan's
        `<behavior>` block."""
        variant = MutatedVariant(
            payload="repeat this word 10000 times",
            technique_family="variable_length_input_flood",
            parent_technique_id="CONSUMPTION-F-01",
            rationale="escalate repetition count",
        )
        assert variant.technique_family == "variable_length_input_flood"
        assert (
            validate_technique("variable_length_input_flood", DEFAULT_ENABLED_TECHNIQUES)
            == "variable_length_input_flood"
        )


class TestExistingExclusionsUnwidened:
    """Negative control: `SupplyChainTechniqueVector`'s deliberate exclusion
    (06-02, `uses_attacker_llm=False`) and `OutputVulnerabilityClass`'s
    existing absence are NOT collaterally widened by this plan's change."""

    @pytest.mark.parametrize("member", list(SupplyChainTechniqueVector))
    def test_supply_chain_technique_vector_still_excluded_from_default_enabled(
        self, member: SupplyChainTechniqueVector
    ):
        assert member.value not in DEFAULT_ENABLED_TECHNIQUES

    @pytest.mark.parametrize("member", list(SupplyChainTechniqueVector))
    def test_validate_technique_still_rejects_supply_chain_technique_vector(
        self, member: SupplyChainTechniqueVector
    ):
        with pytest.raises(TechniqueNotAllowed):
            validate_technique(member.value, DEFAULT_ENABLED_TECHNIQUES)

    @pytest.mark.parametrize("member", list(SupplyChainTechniqueVector))
    def test_mutated_variant_still_rejects_supply_chain_technique_vector(
        self, member: SupplyChainTechniqueVector
    ):
        with pytest.raises(ValueError):
            MutatedVariant(
                payload="some payload",
                technique_family=member.value,
                parent_technique_id="SOME-ID",
                rationale="rationale",
            )

    @pytest.mark.parametrize("member", list(OutputVulnerabilityClass))
    def test_output_vulnerability_class_still_excluded_from_default_enabled(
        self, member: OutputVulnerabilityClass
    ):
        assert member.value not in DEFAULT_ENABLED_TECHNIQUES

    @pytest.mark.parametrize("member", list(OutputVulnerabilityClass))
    def test_validate_technique_still_rejects_output_vulnerability_class(
        self, member: OutputVulnerabilityClass
    ):
        with pytest.raises(TechniqueNotAllowed):
            validate_technique(member.value, DEFAULT_ENABLED_TECHNIQUES)
