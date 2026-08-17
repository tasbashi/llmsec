"""Mutator role -- produces PAIR-style refinements of one parent payload
using the Strategist's selected technique (D-65, D-76, D-79).
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, field_validator, model_validator

from llmsec.attacker.budget import build_step_cap_middleware
from llmsec.attacker.config import AttackerConfig, ResolvedAttackerSettings
from llmsec.attacker.prompts import MUTATOR_SYSTEM_PROMPT
from llmsec.attacker.roles import build_role_chat_model, register_role
from llmsec.attacker.state import CampaignState
from llmsec.payloads.schema import (
    AgencyClass,
    ConsumptionTechniqueVector,
    MisinformationTechniqueVector,
    PiiAttackVector,
    PoisoningTechniqueVector,
    TechniqueFamily,
    VectorContextTechniqueVector,
)

#: The closed technique-family taxonomy a Mutator-returned
#: `technique_family` must belong to (D-95) -- sourced from the existing
#: enums in `payloads/schema.py`, never re-declared, so this allowlist can
#: never silently drift from the corpus taxonomy those enums define.
#: 06-02 (RESEARCH Pitfall #3): `PoisoningTechniqueVector` widened in here
#: MUST land in the same commit as `attacker/config.py`'s
#: `DEFAULT_ENABLED_TECHNIQUES` and `attacker/graph.py`'s
#: `_CLOSED_TECHNIQUE_VOCABULARY` -- omitting any one of the three fails
#: silently (see `attacker/config.py`'s identical comment for the full
#: failure mode).
#: 07-01 (RESEARCH Pitfall #4): `ConsumptionTechniqueVector` widened in here
#: in the SAME plan as the enum's introduction (`unbounded_consumption` sets
#: `uses_attacker_llm = True`, D-02).
#: 08-02 (RESEARCH Pitfall #1): `VectorContextTechniqueVector` and
#: `AgencyClass` widened in here, in `attacker/config.py`'s
#: `DEFAULT_ENABLED_TECHNIQUES`, and in `attacker/graph.py`'s
#: `_CLOSED_TECHNIQUE_VOCABULARY` -- all three, in this SAME commit --
#: because `vector_embedding_weaknesses` and `excessive_agency` both set
#: `uses_attacker_llm = True` (D-03).
#: 09-02 (RESEARCH Pitfall #1): `MisinformationTechniqueVector` widened in
#: here, in `attacker/config.py`'s `DEFAULT_ENABLED_TECHNIQUES`, and in
#: `attacker/graph.py`'s `_CLOSED_TECHNIQUE_VOCABULARY` -- all three, in
#: this SAME commit -- because `misinformation` sets `uses_attacker_llm =
#: True`.
_VALID_TECHNIQUE_FAMILIES: frozenset[str] = (
    frozenset(f.value for f in TechniqueFamily)
    | frozenset(v.value for v in PiiAttackVector)
    | frozenset(p.value for p in PoisoningTechniqueVector)
    | frozenset(c.value for c in ConsumptionTechniqueVector)
    | frozenset(x.value for x in VectorContextTechniqueVector)
    | frozenset(a.value for a in AgencyClass)
    | frozenset(m.value for m in MisinformationTechniqueVector)
)


class MutatedVariant(BaseModel):
    """One Mutator-produced payload variant. Bounded string lengths
    throughout, mirroring `JudgeVerdict`'s shape."""

    payload: str = Field(..., min_length=1, max_length=4000)
    technique_family: str = Field(..., min_length=1, max_length=100)
    parent_technique_id: str = Field(..., min_length=1, max_length=200)
    rationale: str = Field(..., min_length=1, max_length=300)

    @field_validator("technique_family")
    @classmethod
    def _validate_technique_family(cls, value: str) -> str:
        """D-95: never accept a free-form family string into a lineage
        record. A value outside the closed taxonomy fails Pydantic
        validation here, which `roles/_structured_retry.py`'s bounded
        manual retry treats exactly like any other schema-invalid
        structured-output attempt."""
        if value not in _VALID_TECHNIQUE_FAMILIES:
            raise ValueError(
                f"technique_family {value!r} is not a member of the closed "
                "TechniqueFamily/PiiAttackVector/PoisoningTechniqueVector/"
                "ConsumptionTechniqueVector taxonomy (payloads/schema.py)"
            )
        return value


class MutatorOutput(BaseModel):
    """Structured output contract for the Mutator role. D-79's per-round
    variant cap (3) is expressed directly in the schema."""

    variants: list[MutatedVariant] = Field(..., min_length=1, max_length=3)

    @model_validator(mode="after")
    def _validate_distinct_payloads(self) -> "MutatorOutput":
        """05-10/D-94 AT-8: no round may dispatch two byte-identical variant
        payloads for the same case -- a duplicate wastes a target call on
        redundant work and pollutes lineage/coverage stats with a
        meaningless repeat. Enforced HERE, at the schema boundary
        (mirroring `MutatedVariant._validate_technique_family`'s own
        raise-on-violation shape), so a duplicate-producing response is
        treated exactly like any other schema-invalid structured-output
        attempt by `roles/_structured_retry.py`'s bounded manual retry --
        rejected and retried, never silently dispatched twice."""
        payloads = [variant.payload for variant in self.variants]
        if len(payloads) != len(set(payloads)):
            raise ValueError(
                "MutatorOutput.variants contains two or more byte-identical "
                "payloads -- each round's variants must be distinct (D-94 AT-8)"
            )
        return self


def build_mutator_agent(
    settings: ResolvedAttackerSettings,
    cfg: AttackerConfig,
    *,
    model: str | BaseChatModel | None = None,
) -> Any:
    """Build the Mutator's compiled `create_deep_agent()` graph.

    See `strategist.build_strategist_agent()`'s docstring for the
    `ToolStrategy(..., handle_errors=False)` rationale, identical here, and
    for `build_step_cap_middleware()`'s shared-construction-point rationale.
    """
    resolved_model = build_role_chat_model(settings, cfg, "mutator", model)
    return create_deep_agent(
        model=resolved_model,
        tools=[],
        system_prompt=MUTATOR_SYSTEM_PROMPT,
        response_format=ToolStrategy(MutatorOutput, handle_errors=False),
        middleware=build_step_cap_middleware("mutator"),
    )


def _mutator_brief(state: CampaignState) -> str:
    """Build the Mutator's invocation message from `CampaignState` alone
    (D-87) -- never the target's system prompt, credentials, or
    `ScanConfig`."""
    case: dict[str, Any] = dict(state.get("current_case") or {})
    already_tried = [variant["payload"] for variant in state.get("variants", [])]
    return (
        f"TECHNIQUE: {state.get('selected_technique')}\n"
        f"PARENT_CASE_ID: {case.get('case_id')}\n"
        f"PARENT_TECHNIQUE_ID: {case.get('technique_id')}\n"
        f"PARENT_PAYLOAD: {case.get('prompt')}\n"
        f"VARIANTS REQUESTED: {state.get('variants_per_round', 1)}\n"
        f"ALREADY TRIED PAYLOADS: {already_tried}"
    )


class _MutatorRole:
    """`AgentRole` implementation wrapping this module's factory + brief
    builder -- the registry entry D-65 requires."""

    name = "mutator"
    output_schema: type[BaseModel] = MutatorOutput
    system_prompt: str = MUTATOR_SYSTEM_PROMPT

    def build(
        self,
        settings: ResolvedAttackerSettings,
        cfg: AttackerConfig,
        *,
        model: str | BaseChatModel | None = None,
    ) -> Any:
        return build_mutator_agent(settings, cfg, model=model)

    def brief(self, state: CampaignState) -> str:
        return _mutator_brief(state)


register_role(_MutatorRole())
