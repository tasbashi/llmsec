"""BaseModule ABC — the trust chokepoint contract every test module implements.

Every module (built-in `system_prompt_leakage` in Phase 1, plus Phase 2-4
modules and any third-party plugin registered via the `llmsec.modules`
entry_points group) subclasses `BaseModule` and implements exactly this
contract. Nothing downstream should redefine a second module contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator

from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase

if TYPE_CHECKING:
    # Import-cycle avoidance (07-01): `adapters/base.py` never imports from
    # `plugins/base.py`, but keeping this under TYPE_CHECKING mirrors the
    # discipline the rest of the codebase uses for forward references.
    from llmsec.adapters.base import TargetAdapter

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

    # Phase 6 (06-01, MOD-06): concrete ABC default, never the abstract-method
    # decorator used by the two methods above -- additive evolution per
    # PROJECT.md's "PLUGIN_API_VERSION stays 1.0" rule.
    # Every existing module and third-party plugin inherits this no-op
    # default unedited. A module overrides it when its findings come from
    # inspecting operator-declared configuration (e.g. a dependency
    # manifest path) rather than probing a live target through
    # `generate_cases()`/`evaluate()`'s request/response contract.
    #
    # Signature frozen at 06-01-PLAN.md Task 2's checkpoint:decision
    # (async-generator option) -- once a third-party plugin overrides this,
    # changing the signature is a breaking change requiring a
    # PLUGIN_API_VERSION bump.
    async def run_standalone_audit(self, context: ScanContext) -> AsyncIterator[EvalResult]:
        """Concrete default: yields nothing.

        Override for modules whose findings come from inspecting
        operator-declared configuration (e.g. a dependency manifest path)
        rather than probing a live target. `api.py`'s `_run_standalone_audits()`
        gathers this alongside `ScanOrchestrator.run()`.
        """
        return
        yield  # pragma: no cover -- unreachable, forces this to be an async generator

    # Phase 7 (07-01, MOD-09): a second, distinct additive ABC default --
    # never the abstract-method decorator, `PLUGIN_API_VERSION` stays "1.0"
    # -- mirroring `run_standalone_audit()`'s exact shape above but
    # semantically distinct: this is for live-target probing that must
    # bypass `ScanOrchestrator._dispatch_with_retry()` entirely, e.g. a case
    # whose transport-failure signal (429/rate-limit, timeout) IS the
    # desired positive result and must never be retried. Every existing
    # module and third-party plugin inherits this no-op default unedited.
    #
    # Unlike `run_standalone_audit()`, this receives the SAME `TargetAdapter`
    # instance `run_scan()` constructed and shares across every concurrent
    # case (CORE-03) -- the module is responsible for its own dispatch loop,
    # its own concurrency bound if needed, and its own T-01-18 failure
    # containment (never raising past `api.py`'s per-module
    # catch-log-continue wrapper, the same discipline `_run_standalone_
    # audits()` already enforces for `run_standalone_audit()`).
    #
    # Signature frozen the same way `run_standalone_audit()`'s is -- once a
    # third-party plugin overrides this, changing the signature is a
    # breaking change requiring a `PLUGIN_API_VERSION` bump.
    async def run_direct_probe(
        self, context: ScanContext, adapter: "TargetAdapter"
    ) -> AsyncIterator[EvalResult]:
        """Concrete default: yields nothing.

        Override for modules whose findings require bypassing
        `ScanOrchestrator._dispatch_with_retry()` entirely -- e.g. a case
        whose transport-failure signal (429/rate-limit, timeout) IS the
        desired positive result and must never be retried (MOD-09).
        `api.py`'s `_run_direct_probes()` gathers this alongside
        `ScanOrchestrator.run()` and `run_standalone_audit()`.
        """
        return
        yield  # pragma: no cover -- unreachable, forces this to be an async generator
