"""Crescendo Orchestrator role -- plans a short ordered multi-turn escalation
arc, replacing the Mutator for a case on the escalation path (D-65, D-76).
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, field_validator

from llmsec.attacker.budget import build_step_cap_middleware
from llmsec.attacker.config import AttackerConfig, ResolvedAttackerSettings
from llmsec.attacker.prompts import CRESCENDO_SYSTEM_PROMPT
from llmsec.attacker.roles import build_role_chat_model, register_role
from llmsec.attacker.state import CampaignState

#: Bounds on an arc's length (Claude's discretion per 05-08-PLAN.md): at
#: least 2 turns (a single turn is not an escalation), at most 6 (a bounded
#: arc -- unbounded turn count would defeat D-70's per-role step cap and
#: D-79's fixed per-round cost model).
_MIN_ARC_TURNS = 2
_MAX_ARC_TURNS = 6

#: Per-turn text length bound, mirroring `MutatedVariant.payload`'s own
#: `max_length=4000` (`roles/mutator.py`).
_MAX_TURN_CHARS = 2000


class CrescendoOutput(BaseModel):
    """Structured output contract for the Crescendo Orchestrator role
    (D-65/D-76). Every field explicitly bounded; no verdict/severity/
    confidence/score field anywhere -- scoring stays exclusively
    `module.evaluate()`'s job (D-66), exactly like every other role's
    structured output in this package.
    """

    turns: list[str] = Field(
        ...,
        min_length=_MIN_ARC_TURNS,
        max_length=_MAX_ARC_TURNS,
        description=(
            "Ordered escalation turns, each a natural continuation of the "
            "conversation so far -- never restating the objective in every turn."
        ),
    )
    arc_rationale: str = Field(..., min_length=1, max_length=500)
    backtrack_from_turn: int | None = Field(
        default=None,
        ge=0,
        description=(
            "0-based turn index to resume escalation from after a later turn "
            "hit a hard refusal. `None` when no backtrack is recommended."
        ),
    )
    abort_recommended: bool = Field(
        default=False,
        description="True when the arc is clearly dead and should not be dispatched at all.",
    )

    @field_validator("turns")
    @classmethod
    def _validate_turn_lengths(cls, value: list[str]) -> list[str]:
        for turn in value:
            if not (1 <= len(turn) <= _MAX_TURN_CHARS):
                raise ValueError(f"each turn must be 1-{_MAX_TURN_CHARS} characters")
        return value


def build_crescendo_agent(
    settings: ResolvedAttackerSettings,
    cfg: AttackerConfig,
    *,
    model: str | BaseChatModel | None = None,
) -> Any:
    """Build the Crescendo Orchestrator's compiled `create_deep_agent()`
    graph.

    See `strategist.build_strategist_agent()`'s docstring for the
    `ToolStrategy(..., handle_errors=False)` rationale, identical here, and
    for `build_step_cap_middleware()`'s shared-construction-point
    rationale. Passes `tools=[]` -- the Crescendo Orchestrator plans an arc
    from its brief alone (built from `CampaignState`, D-87); it never
    dispatches to the target itself, that is `dispatch_variants_node`'s
    job once this role returns a plan.
    """
    resolved_model = build_role_chat_model(settings, cfg, "crescendo", model)
    return create_deep_agent(
        model=resolved_model,
        tools=[],
        system_prompt=CRESCENDO_SYSTEM_PROMPT,
        response_format=ToolStrategy(CrescendoOutput, handle_errors=False),
        middleware=build_step_cap_middleware("crescendo"),
    )


def _crescendo_brief(state: CampaignState) -> str:
    """Build the Crescendo Orchestrator's invocation message from
    `CampaignState` alone (D-87) -- never the target's system prompt,
    credentials, or `ScanConfig`."""
    case: dict[str, Any] = dict(state.get("current_case") or {})
    bounded_memory = state.get("bounded_memory")
    refusal_signatures = bounded_memory.get("refusal_signatures", []) if bounded_memory else []
    return (
        f"TECHNIQUE: {state.get('selected_technique')}\n"
        f"PARENT_CASE_ID: {case.get('case_id')}\n"
        f"PARENT_TECHNIQUE_ID: {case.get('technique_id')}\n"
        f"PARENT_PAYLOAD: {case.get('prompt')}\n"
        f"KNOWN REFUSAL SIGNATURES: {refusal_signatures}"
    )


class _CrescendoRole:
    """`AgentRole` implementation wrapping this module's factory + brief
    builder -- the registry entry D-65 requires."""

    name = "crescendo"
    output_schema: type[BaseModel] = CrescendoOutput
    system_prompt: str = CRESCENDO_SYSTEM_PROMPT

    def build(
        self,
        settings: ResolvedAttackerSettings,
        cfg: AttackerConfig,
        *,
        model: str | BaseChatModel | None = None,
    ) -> Any:
        return build_crescendo_agent(settings, cfg, model=model)

    def brief(self, state: CampaignState) -> str:
        return _crescendo_brief(state)


register_role(_CrescendoRole())
