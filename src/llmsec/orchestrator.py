"""`ScanOrchestrator` — bounded-concurrency fan-out across adapter + module.

Fans every module's `generate_cases()` output through the shared
`TargetAdapter` (one instance, shared across all concurrently-dispatched
`TestCase`s per CORE-03) and then through that module's `evaluate()`,
bounding in-flight `adapter.send()`/`adapter.send_conversation()` calls with
`asyncio.Semaphore` and retrying transient transport failures with
`tenacity`.

Dispatch rule (plan 02-07): a `TestCase` carrying a non-empty `turns` list
is routed through `adapter.send_conversation()`; every other case (including
one with `turns=[]`, per the EDGE:empty rule) goes through `adapter.send()`
exactly as in Phase 1. For the sequence branch, the module's optional
`should_abort_sequence(case, turn_reply) -> bool` hook is resolved by plain
`getattr` duck-typing (never an ABC method) and wired in as
`send_conversation()`'s `stop_when` callable — `BaseModule` and
`PLUGIN_API_VERSION 1.0` are deliberately unchanged (D-12/D-14). A module
without the hook simply never aborts early. Whichever branch runs, a `None`
`transport_mode` on the returned response is normalized to `"single"` so
`evaluate()` always receives a concrete transport label (D-15).

A persistently-failing case (retry-exhausted) is degraded to a recorded
`Verdict.UNCERTAIN` `EvalResult` rather than raised — T-01-18: one
misbehaving test case must never cascade into a full scan failure by
cancelling sibling `asyncio.gather` tasks. This containment guarantee
extends unchanged to multi-turn sequences (T-02-27): a retry-exhausted
sequence re-sends the *whole* conversation from turn 1 on each attempt
(a partially-completed sequence has no resumable state on the target's
side), bounded by the same 3-attempt retry policy as `send()`.
Sequences are serial internally but still run concurrently with each other
under the existing semaphore — concurrency bounds are unchanged.

Everything in this file is `async def`; the event loop is never started
from here (PITFALLS Pitfall 3) — only the CLI entrypoint (plan 09) owns
the event loop via its own top-level call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from llmsec.adapters.base import TargetAdapter
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

logger = logging.getLogger(__name__)

# Transient, transport-layer failure shapes (429/5xx-shaped exceptions per
# the plan's <behavior>) that are worth retrying. `httpx.HTTPError` covers
# `httpx.TimeoutException`/`httpx.TransportError` raised by `HttpAppAdapter`;
# `TimeoutError`/`ConnectionError`/`OSError` cover the same failure class
# from `LLMApiAdapter`'s underlying litellm/network stack. A non-transient
# error (e.g. a bad request shape) is intentionally NOT in this set, so it
# still propagates immediately rather than being retried uselessly.
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.HTTPError,
    TimeoutError,
    ConnectionError,
    OSError,
)


class ScanOrchestrator:
    """Fans `TestCase`s from every loaded module through one shared adapter.

    `adapter` is constructed once by the caller (`api.run_scan()`) and
    shared across every concurrently-dispatched case — `TargetAdapter`
    implementations hold no per-call mutable state, so this is
    concurrency-safe (CORE-03).
    """

    def __init__(
        self,
        adapter: TargetAdapter,
        modules: dict[str, BaseModule],
        max_concurrency: int = 5,
    ) -> None:
        self.adapter = adapter
        self.modules = modules
        self.max_concurrency = max_concurrency

    async def run(self, context: ScanContext) -> list[tuple[str, EvalResult]]:
        """Fan out every module's generated `TestCase`s and return every
        outcome, one `(module_id, EvalResult)` pair per generated case —
        never fewer, even when a case's adapter call ultimately fails after
        retries (that case's outcome is recorded as `UNCERTAIN`, not
        dropped)."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks: list[asyncio.Task[tuple[str, EvalResult]]] = []
        for module_id, module in self.modules.items():
            try:
                async for case in module.generate_cases(context):
                    tasks.append(
                        asyncio.create_task(self._run_case(semaphore, module_id, module, case))
                    )
            except Exception as exc:
                logger.error(
                    "Module %r failed during generate_cases(); skipping remaining cases "
                    "from this module: %s", module_id, exc,
                )
                continue
        return await asyncio.gather(*tasks)

    async def _run_case(
        self,
        semaphore: asyncio.Semaphore,
        module_id: str,
        module: BaseModule,
        case: TestCase,
    ) -> tuple[str, EvalResult]:
        async with semaphore:
            try:
                response = await self._dispatch_with_retry(case, module)
            except Exception as exc:  # retry-exhausted or non-retryable failure
                logger.warning(
                    "Case %r failed after retries, degrading to UNCERTAIN: %s",
                    case.case_id,
                    exc,
                )
                return module_id, EvalResult(
                    case_id=case.case_id,
                    verdict=Verdict.UNCERTAIN,
                    confidence=0.0,
                    # `detection_layer` is a closed Literal["regex", "judge"];
                    # neither tier actually ran here, so "regex" (the
                    # zero-LLM-cost tier) is the least misleading choice for
                    # a synthetic transport-failure result.
                    evidence=f"transport failure after retries: {exc}",
                    detection_layer="regex",
                )
            try:
                eval_result = await module.evaluate(case, response)
            except Exception as exc:  # judge-side/non-schema evaluate() failure
                logger.warning(
                    "Case %r evaluation failed, degrading to UNCERTAIN: %s",
                    case.case_id,
                    exc,
                )
                return module_id, EvalResult(
                    case_id=case.case_id,
                    verdict=Verdict.UNCERTAIN,
                    confidence=0.0,
                    evidence=f"evaluation failure: {exc}",
                    detection_layer="judge",
                )
            return module_id, eval_result

    async def _dispatch_with_retry(self, case: TestCase, module: BaseModule) -> TargetResponse:
        """Dispatch `case` to the adapter, routing multi-turn sequences
        through `send_conversation()` and everything else through `send()`,
        both wrapped in the same bounded exponential-backoff retry (3
        attempts) on transient transport failures. On exhaustion,
        `reraise=True` lets the original exception propagate to
        `_run_case`'s except clause rather than wrapping it in a
        `tenacity.RetryError` — a retried sequence re-sends the whole
        conversation from turn 1, since a partially-completed sequence has
        no resumable state on the target's side.

        `case.turns` is only treated as a sequence when it is truthy and
        non-empty; `turns=[]` falls through to the ordinary single-turn
        path rather than dispatching an empty conversation (EDGE:empty).
        """
        is_sequence = bool(case.turns)
        stop_when = self._resolve_stop_when(module, case) if is_sequence else None

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            wait=wait_exponential(multiplier=0.01, max=1),
            stop=stop_after_attempt(3),
            reraise=True,
        ):
            with attempt:
                if is_sequence:
                    response = await self.adapter.send_conversation(case, stop_when=stop_when)
                else:
                    response = await self.adapter.send(case)
                return self._normalize_transport(response)
        # Unreachable: AsyncRetrying always either returns or raises above.
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _resolve_stop_when(module: BaseModule, case: TestCase) -> Callable[[str], bool] | None:
        """Resolve the module's optional abort hook by attribute lookup.

        `should_abort_sequence(case, turn_reply) -> bool` is a duck-typed
        protocol, not an ABC method — `BaseModule` and `PLUGIN_API_VERSION
        1.0` are deliberately unchanged in this phase (D-12/D-14), so a
        module that does not define the hook simply never aborts early
        (`stop_when=None`). When the hook exists, the returned callable
        binds `case` and swallows any exception the module's predicate
        raises, treating it as "do not abort" (T-02-24) — a misbehaving
        third-party predicate degrades the case like any other failure
        rather than crashing the scan.
        """
        predicate = getattr(module, "should_abort_sequence", None)
        if predicate is None:
            return None

        def _stop_when(turn_reply: str) -> bool:
            try:
                return bool(predicate(case, turn_reply))
            except Exception:
                return False

        return _stop_when

    @staticmethod
    def _normalize_transport(response: TargetResponse) -> TargetResponse:
        """Guarantee every response reaching `evaluate()` carries a
        concrete `transport_mode` (D-15): a `None` mode is filled in with
        `"single"`; a response that already carries a mode (every
        `send_conversation()` result does) is returned untouched."""
        if response.transport_mode is None:
            return response.model_copy(update={"transport_mode": "single"})
        return response
