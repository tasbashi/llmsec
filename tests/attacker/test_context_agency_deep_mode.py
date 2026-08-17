"""Tests for `VectorContextTechniqueVector`'s and `AgencyClass`'s three-call-
site deep-mode allowlist widening (08-02-PLAN.md Task 3, RESEARCH Pitfall 1).

Mirrors `tests/attacker/test_consumption_deep_mode.py`'s positive-inclusion
plus negative-exclusion shape exactly: `vector_embedding_weaknesses` and
`excessive_agency` both set `uses_attacker_llm = True` (D-03), so every
member of both new enums must clear `DEFAULT_ENABLED_TECHNIQUES`,
`_CLOSED_TECHNIQUE_VOCABULARY`, `validate_technique()`, and
`MutatedVariant`. `SupplyChainTechniqueVector`/`OutputVulnerabilityClass`
remain deliberately excluded (`uses_attacker_llm = False` /
static-taxonomy), proving the widening is additive, never a blanket
opening.
"""

from __future__ import annotations

import pytest

from llmsec.attacker.config import DEFAULT_ENABLED_TECHNIQUES
from llmsec.attacker.graph import _CLOSED_TECHNIQUE_VOCABULARY, TechniqueNotAllowed, validate_technique
from llmsec.attacker.roles.mutator import MutatedVariant, _VALID_TECHNIQUE_FAMILIES
from llmsec.payloads.schema import (
    AgencyClass,
    OutputVulnerabilityClass,
    SupplyChainTechniqueVector,
    VectorContextTechniqueVector,
)


class TestVectorContextTechniqueVectorAllowlist:
    """Positive inclusion: every VectorContextTechniqueVector member clears
    all three deep-mode delegation-boundary call sites."""

    @pytest.mark.parametrize("member", list(VectorContextTechniqueVector))
    def test_included_in_default_enabled_techniques(self, member: VectorContextTechniqueVector):
        assert member.value in DEFAULT_ENABLED_TECHNIQUES

    @pytest.mark.parametrize("member", list(VectorContextTechniqueVector))
    def test_included_in_closed_technique_vocabulary(self, member: VectorContextTechniqueVector):
        assert member.value in _CLOSED_TECHNIQUE_VOCABULARY

    @pytest.mark.parametrize("member", list(VectorContextTechniqueVector))
    def test_included_in_valid_technique_families(self, member: VectorContextTechniqueVector):
        assert member.value in _VALID_TECHNIQUE_FAMILIES

    @pytest.mark.parametrize("member", list(VectorContextTechniqueVector))
    def test_validate_technique_accepts_every_member(self, member: VectorContextTechniqueVector):
        # Must not raise.
        assert validate_technique(member.value, DEFAULT_ENABLED_TECHNIQUES) == member.value

    @pytest.mark.parametrize("member", list(VectorContextTechniqueVector))
    def test_mutated_variant_accepts_every_member(self, member: VectorContextTechniqueVector):
        # Must not raise a pydantic ValidationError.
        variant = MutatedVariant(
            payload="a more elaborate fake retrieved document",
            technique_family=member.value,
            parent_technique_id="CTXLEAK-01",
            rationale="escalate the simulated retrieved-context probe",
        )
        assert variant.technique_family == member.value


class TestAgencyClassAllowlist:
    """Positive inclusion: every AgencyClass member clears all three
    deep-mode delegation-boundary call sites."""

    @pytest.mark.parametrize("member", list(AgencyClass))
    def test_included_in_default_enabled_techniques(self, member: AgencyClass):
        assert member.value in DEFAULT_ENABLED_TECHNIQUES

    @pytest.mark.parametrize("member", list(AgencyClass))
    def test_included_in_closed_technique_vocabulary(self, member: AgencyClass):
        assert member.value in _CLOSED_TECHNIQUE_VOCABULARY

    @pytest.mark.parametrize("member", list(AgencyClass))
    def test_included_in_valid_technique_families(self, member: AgencyClass):
        assert member.value in _VALID_TECHNIQUE_FAMILIES

    @pytest.mark.parametrize("member", list(AgencyClass))
    def test_validate_technique_accepts_every_member(self, member: AgencyClass):
        # Must not raise.
        assert validate_technique(member.value, DEFAULT_ENABLED_TECHNIQUES) == member.value

    @pytest.mark.parametrize("member", list(AgencyClass))
    def test_mutated_variant_accepts_every_member(self, member: AgencyClass):
        # Must not raise a pydantic ValidationError.
        variant = MutatedVariant(
            payload="a more indirect capability-boundary question",
            technique_family=member.value,
            parent_technique_id="AGENCY-01",
            rationale="escalate the capability-boundary probe",
        )
        assert variant.technique_family == member.value


class TestExistingExclusionsUnwidened:
    """Negative control: `SupplyChainTechniqueVector`'s and
    `OutputVulnerabilityClass`'s deliberate exclusion is NOT collaterally
    widened by this plan's change -- proving the widening is additive, not
    a blanket opening."""

    @pytest.mark.parametrize("member", list(SupplyChainTechniqueVector))
    def test_supply_chain_technique_vector_still_excluded_from_all_three_sites(
        self, member: SupplyChainTechniqueVector
    ):
        assert member.value not in DEFAULT_ENABLED_TECHNIQUES
        assert member.value not in _CLOSED_TECHNIQUE_VOCABULARY
        assert member.value not in _VALID_TECHNIQUE_FAMILIES

    @pytest.mark.parametrize("member", list(SupplyChainTechniqueVector))
    def test_validate_technique_still_rejects_supply_chain_technique_vector(
        self, member: SupplyChainTechniqueVector
    ):
        with pytest.raises(TechniqueNotAllowed):
            validate_technique(member.value, DEFAULT_ENABLED_TECHNIQUES)

    @pytest.mark.parametrize("member", list(OutputVulnerabilityClass))
    def test_output_vulnerability_class_still_excluded_from_all_three_sites(
        self, member: OutputVulnerabilityClass
    ):
        assert member.value not in DEFAULT_ENABLED_TECHNIQUES
        assert member.value not in _CLOSED_TECHNIQUE_VOCABULARY
        assert member.value not in _VALID_TECHNIQUE_FAMILIES

    @pytest.mark.parametrize("member", list(OutputVulnerabilityClass))
    def test_validate_technique_still_rejects_output_vulnerability_class(
        self, member: OutputVulnerabilityClass
    ):
        with pytest.raises(TechniqueNotAllowed):
            validate_technique(member.value, DEFAULT_ENABLED_TECHNIQUES)
