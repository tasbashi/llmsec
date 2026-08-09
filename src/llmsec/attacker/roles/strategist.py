"""Strategist role -- selects one technique, orders candidate cases, and
optionally signals early campaign exhaust (D-65, D-70, D-72, D-78).
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from llmsec.attacker.budget import build_step_cap_middleware
from llmsec.attacker.config import AttackerConfig, ResolvedAttackerSettings
from llmsec.attacker.prompts import STRATEGIST_SYSTEM_PROMPT
from llmsec.attacker.roles import build_role_chat_model, register_role
from llmsec.attacker.state import CampaignState, StrategistReasonCode


class StrategistOutput(BaseModel):
    """Structured output contract for the Strategist role.

    Bounded string lengths throughout, mirroring `JudgeVerdict`'s shape
    (`detection/judge.py`) -- never an unconstrained `str`.
    """

    technique: str = Field(..., min_length=1, max_length=100)
    ordered_case_ids: list[str] = Field(default_factory=list, max_length=50)
    escalate: bool = False
    reason_code: StrategistReasonCode | None = None
    rationale: str = Field(..., min_length=1, max_length=500)


def build_strategist_agent(
    settings: ResolvedAttackerSettings,
    cfg: AttackerConfig,
    *,
    model: str | BaseChatModel | None = None,
) -> Any:
    """Build the Strategist's compiled `create_deep_agent()` graph.

    `model`, when given, overrides the resolved production model --
    `roles/__init__.py`'s `build_role_chat_model()`'s sole test-injection
    hook, never required in production use.

    Passes `tools=[]` (the Strategist reads only its brief -- no
    target-facing tools) and `response_format=ToolStrategy(StrategistOutput,
    handle_errors=False)` -- NOT the bare `StrategistOutput` class, whose
    default handling retries internally, unbounded except by LangGraph's
    (much larger) recursion limit rather than this package's own bounded
    2-manual-retry contract (`roles/_structured_retry.py`, D-94 AT-6).
    Never enables DeepAgents' own delegation-to-sub-agent dispatch (D-73)
    -- round control belongs to the caller's `StateGraph`, never to an
    agentic supervisor loop.

    `middleware=build_step_cap_middleware("strategist")` attaches D-70's
    per-role hard step cap (`attacker/budget.py`'s `StepCapMiddleware`) --
    the ONE shared construction point every role factory calls, never a
    per-role reimplementation.
    """
    resolved_model = build_role_chat_model(settings, cfg, "strategist", model)
    return create_deep_agent(
        model=resolved_model,
        tools=[],
        system_prompt=STRATEGIST_SYSTEM_PROMPT,
        response_format=ToolStrategy(StrategistOutput, handle_errors=False),
        middleware=build_step_cap_middleware("strategist"),
    )


def _strategist_brief(state: CampaignState) -> str:
    """Build the Strategist's invocation message from `CampaignState`
    alone (D-87) -- never the target's system prompt, credentials, or
    `ScanConfig`."""
    current_module = state.get("current_module")
    module_cases = [
        case for case in state.get("case_queue", []) if case["module_id"] == current_module
    ]
    if module_cases:
        case_lines = "\n".join(
            f"- {case['case_id']} (verdict={case['verdict']}): {case['prompt'][:200]}"
            for case in module_cases
        )
    else:
        case_lines = "(no candidate cases queued for this module)"
    bounded_memory = state.get("bounded_memory")
    refusal_signatures = bounded_memory.get("refusal_signatures", []) if bounded_memory else []
    dead_techniques = bounded_memory.get("dead_techniques", []) if bounded_memory else []
    partial_movement = bounded_memory.get("partial_movement_techniques", []) if bounded_memory else []
    return (
        f"ROUND: {state.get('round', 0) + 1} of {state.get('max_rounds', 1)}\n"
        f"MODULE: {current_module}\n"
        f"ALLOWED TECHNIQUES: {', '.join(state.get('enabled_techniques', []))}\n"
        f"CANDIDATE CASES (this module):\n{case_lines}\n"
        f"KNOWN REFUSAL SIGNATURES: {refusal_signatures}\n"
        f"KNOWN DEAD TECHNIQUES: {dead_techniques}\n"
        f"KNOWN PARTIAL-MOVEMENT TECHNIQUES: {partial_movement}"
    )


class _StrategistRole:
    """`AgentRole` implementation wrapping this module's factory + brief
    builder -- the registry entry D-65 requires. Adding a future role is
    exactly this shape plus a pinned prompt, never a `graph.py` edit."""

    name = "strategist"
    output_schema: type[BaseModel] = StrategistOutput
    system_prompt: str = STRATEGIST_SYSTEM_PROMPT

    def build(
        self,
        settings: ResolvedAttackerSettings,
        cfg: AttackerConfig,
        *,
        model: str | BaseChatModel | None = None,
    ) -> Any:
        return build_strategist_agent(settings, cfg, model=model)

    def brief(self, state: CampaignState) -> str:
        return _strategist_brief(state)


register_role(_StrategistRole())
