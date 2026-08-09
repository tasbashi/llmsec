"""Shared `TargetAdapter` ABC (CORE-03).

Every concrete adapter (`LLMApiAdapter`, `HttpAppAdapter`) implements this
single interface so the orchestrator and every module's `evaluate()` never
know or care which concrete transport they're talking to. Per D-73, Phase 5's
deep-mode attacker team is deliberately NOT a `TargetAdapter` implementer --
it runs as a separate layer that *calls* a `TargetAdapter` instance to reach
the target, on the pinned `litellm` core dependency (D-73 mitigation 3); the
attacker team's own model stack must never migrate the target transport path
onto it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from llmsec.models import TestCase, TargetResponse


class TargetAdapter(ABC):
    """Abstract transport contract for "the target" of a scan."""

    # Capability flags (D-12). A concrete adapter may override either as a
    # class attribute (a static capability, e.g. `LLMApiAdapter`) or set it
    # on the instance in `__init__` (a configuration-dependent capability —
    # this is how `HttpAppAdapter` upgrades once an operator configures a
    # session round-trip).
    supports_multi_turn: bool = False
    supports_system_prompt_override: bool = False

    @abstractmethod
    async def send(self, case: TestCase) -> TargetResponse: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def send_conversation(
        self, case: TestCase, stop_when: Callable[[str], bool] | None = None
    ) -> TargetResponse:
        """Default, honest *degraded* substitute for real multi-turn dialogue.

        This concatenates `case.turns` (or falls back to `case.prompt` when
        `turns` is unset) into a single blank-line-joined `send()` call, and
        labels the result `transport_mode="multi_turn_concatenated"` — never
        `"multi_turn_real"`. Callers must consult `supports_multi_turn` and
        honor the recorded `transport_mode` rather than assuming a real
        conversation happened (D-12/D-15): a wrongly-True capability flag or
        a mislabeled response converts a degraded run into a report that
        falsely claims genuine crescendo coverage.

        `stop_when` is ignored here and documented as such: a single
        flattened request has no turn boundary at which to abort early.
        """
        turns = case.turns or [case.prompt]
        combined = TestCase(
            case_id=case.case_id,
            prompt="\n\n".join(turns),
            technique_id=case.technique_id,
            system_prompt_override=case.system_prompt_override,
        )
        response = await self.send(combined)
        return response.model_copy(
            update={
                "transport_mode": "multi_turn_concatenated",
                "turn_replies": [response.raw_text],
            }
        )

    async def __aenter__(self) -> "TargetAdapter":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()
