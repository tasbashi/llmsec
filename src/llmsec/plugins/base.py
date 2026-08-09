"""BaseModule ABC — the trust chokepoint contract every test module implements.

Every module (built-in `system_prompt_leakage` in Phase 1, plus Phase 2-4
modules and any third-party plugin registered via the `llmsec.modules`
entry_points group) subclasses `BaseModule` and implements exactly this
contract. Nothing downstream should redefine a second module contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase

PLUGIN_API_VERSION = "1.0"  # bump on breaking BaseModule changes


class BaseModule(ABC):
    """Contract every test module implements.

    Class attributes (set by subclasses):
        id: unique module identifier, must match its `llmsec.modules`
            entry_point name (e.g. "system_prompt_leakage").
        name: human-readable module name.
        owasp_ref: OWASP LLM Top 10 reference (e.g. "LLM07:2025").
        uses_attacker_llm: forward-compat hook for Phase 5's Attacker LLM;
            unused in Phase 1.
    """

    id: str
    name: str
    owasp_ref: str
    uses_attacker_llm: bool = False

    @abstractmethod
    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        """Yield adversarial `TestCase`s to send to the target."""
        ...

    @abstractmethod
    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        """Evaluate a target's response to a `TestCase` and return a verdict."""
        ...
