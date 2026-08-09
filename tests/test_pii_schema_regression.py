"""Regression tests for D-37's `PayloadEntry.technique_family` schema
widening (Pitfall 5) — proving the Union-of-enums extension does not break
`prompt_injection.py`'s existing `TechniqueFamily`-typed comparisons or
loosen validation into an open string.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmsec.payloads.schema import PayloadEntry, PiiAttackVector, TechniqueFamily


def test_technique_family_round_trips_and_compares_equal_for_technique_family_member():
    entry = PayloadEntry(
        id="DIRECT-001",
        technique_family=TechniqueFamily.INSTRUCTION_OVERRIDE,
        description="d",
        prompt="p",
        remediation="r" * 20,
    )
    assert entry.technique_family == TechniqueFamily.INSTRUCTION_OVERRIDE
    restored = PayloadEntry.model_validate_json(entry.model_dump_json())
    assert restored.technique_family == TechniqueFamily.INSTRUCTION_OVERRIDE


def test_technique_family_validates_for_pii_attack_vector_member():
    entry = PayloadEntry(
        id="PII-004",
        technique_family=PiiAttackVector.CREDENTIAL_PROBING,
        description="d",
        prompt="p",
        remediation="r" * 20,
    )
    assert entry.technique_family == PiiAttackVector.CREDENTIAL_PROBING


def test_technique_family_out_of_taxonomy_string_raises_validation_error():
    """The Union stays a CLOSED set of two enums — not a free string that
    happens to validate against nothing."""
    with pytest.raises(ValidationError):
        PayloadEntry(
            id="X-001",
            technique_family="not_a_real_family_in_either_enum",
            description="d",
            prompt="p",
            remediation="r" * 20,
        )


def test_canary_pii_type_defaults_to_none_and_accepts_valid_values():
    entry = PayloadEntry(
        id="PII-004",
        technique_family=PiiAttackVector.CREDENTIAL_PROBING,
        description="d",
        prompt="p",
        remediation="r" * 20,
    )
    assert entry.canary_pii_type is None

    entry_with_canary = PayloadEntry(
        id="PII-006",
        technique_family=PiiAttackVector.CANARY_TRIGGERING,
        description="d",
        prompt="p",
        remediation="r" * 20,
        canary_pii_type="ssn",
    )
    assert entry_with_canary.canary_pii_type == "ssn"


def test_canary_pii_type_rejects_invalid_value():
    with pytest.raises(ValidationError):
        PayloadEntry(
            id="PII-006",
            technique_family=PiiAttackVector.CANARY_TRIGGERING,
            description="d",
            prompt="p",
            remediation="r" * 20,
            canary_pii_type="not_a_valid_pii_type",
        )


def test_pii_attack_vector_has_exactly_eight_members():
    assert {v.value for v in PiiAttackVector} == {
        "training_data_extraction",
        "context_replay",
        "credential_probing",
        "canary_triggering",
        "membership_inference",
        "rag_pii_extraction",
        "url_exfiltration",
        "pii_aggregation",
    }
