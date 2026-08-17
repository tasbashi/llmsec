"""Safety-property tests for `UnboundedConsumptionModule` (07-03-PLAN.md
Task 3, MOD-09).

Pins the guarantees that make MOD-09's design trustworthy beyond "a 429
scores blocked once": the cap holds per invocation (never cumulative), the
corpus is never mutated by a probe run, at most one flood request is ever
in flight (even under concurrent orchestrator dispatch), and no retry
primitive is imported by the module at all.
"""

from __future__ import annotations

import asyncio

from llmsec.models import ScanContext, TargetResponse, TestCase
from llmsec.modules.unbounded_consumption import UnboundedConsumptionModule


def _context() -> ScanContext:
    return ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="")


class _CountingStubAdapter:
    """A `TargetAdapter`-shaped stub that always returns a benign, clearly
    within-threshold `TargetResponse` (never raises), and counts total
    `send()`/`send_conversation()` calls."""

    supports_system_prompt_override = False
    supports_multi_turn = False

    def __init__(self) -> None:
        self.call_count = 0
        self.call_counts_by_case: dict[str, int] = {}
        self.closed = False

    async def send(self, case: TestCase) -> TargetResponse:
        self.call_count += 1
        self.call_counts_by_case[case.case_id] = self.call_counts_by_case.get(case.case_id, 0) + 1
        return TargetResponse(case_id=case.case_id, raw_text="ok", latency_ms=1.0, tokens_used=1)

    async def send_conversation(self, case: TestCase, stop_when=None) -> TargetResponse:
        return await self.send(case)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


class _ConcurrencyTrackingAdapter:
    """A `TargetAdapter`-shaped stub whose `send()` increments an in-flight
    counter, yields control (zero-delay sleep) so a concurrently-running
    coroutine gets a chance to overlap, then decrements -- recording the
    maximum observed in-flight value across the whole run."""

    supports_system_prompt_override = False
    supports_multi_turn = False

    def __init__(self) -> None:
        self._in_flight = 0
        self.max_in_flight = 0
        self.call_count = 0
        self.closed = False

    async def send(self, case: TestCase) -> TargetResponse:
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        self.call_count += 1
        try:
            await asyncio.sleep(0)
            return TargetResponse(
                case_id=case.case_id, raw_text="ok", latency_ms=1.0, tokens_used=1
            )
        finally:
            self._in_flight -= 1

    async def send_conversation(self, case: TestCase, stop_when=None) -> TargetResponse:
        return await self.send(case)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


class TestFloodProbeCap:
    """MOD-09/T-07-11: `run_direct_probe()` dispatches exactly
    `min(len(flood_entries), cap)` adapter calls per invocation."""

    async def test_five_entries_cap_three_dispatches_exactly_three(self):
        module = UnboundedConsumptionModule()
        flood_entries = module._flood_entries()
        assert len(flood_entries) == 5, "corpus expected to carry 5 flood-class entries"
        assert module._flood_probe_cap == 3

        adapter = _CountingStubAdapter()
        results = [
            result async for result in module.run_direct_probe(_context(), adapter)
        ]

        assert adapter.call_count == 3
        assert len(results) == 3
        assert len(adapter.call_counts_by_case) == 3

    async def test_flood_entries_shorter_than_cap_dispatches_all_of_them(self):
        """Boundary case: when the flood-entry list is SHORTER than the
        cap, every entry is dispatched -- the cap never pads or requires a
        minimum count."""
        module = UnboundedConsumptionModule()
        module._flood_probe_cap = 100  # artificially raise the cap above corpus size

        adapter = _CountingStubAdapter()
        results = [
            result async for result in module.run_direct_probe(_context(), adapter)
        ]

        flood_entry_count = len(module._flood_entries())
        assert flood_entry_count < module._flood_probe_cap
        assert adapter.call_count == flood_entry_count
        assert len(results) == flood_entry_count


class TestRunDirectProbeIdempotency:
    """MOD-09: invoking `run_direct_probe()` twice on the SAME module
    instance yields the same ordered case-id sequence both times, and each
    invocation contributes exactly the capped number of adapter calls --
    never 3 then 0, never 3 then 5, never cumulative."""

    async def test_two_invocations_yield_same_ordered_case_ids_and_per_call_cap(self):
        module = UnboundedConsumptionModule()
        adapter = _CountingStubAdapter()

        first_results = [
            result async for result in module.run_direct_probe(_context(), adapter)
        ]
        first_case_ids = [result.case_id for result in first_results]
        assert adapter.call_count == 3  # first invocation: exactly the cap

        second_results = [
            result async for result in module.run_direct_probe(_context(), adapter)
        ]
        second_case_ids = [result.case_id for result in second_results]

        assert first_case_ids == second_case_ids
        # Per-invocation, not cumulative: the counter grew by exactly the
        # cap again on the second call -- never 3 then 0 (never dispatched
        # again), never 3 then 5 (cap ignored second time), never 6 read as
        # "cumulative-capped" evidence of a bug either way.
        assert adapter.call_count == 6

    async def test_corpus_cache_unchanged_in_length_and_content_after_two_runs(self):
        """A probe run must never mutate the lazily-cached corpus list."""
        module = UnboundedConsumptionModule()
        adapter = _CountingStubAdapter()

        before = list(module._corpus_entries())
        [result async for result in module.run_direct_probe(_context(), adapter)]
        [result async for result in module.run_direct_probe(_context(), adapter)]
        after = list(module._corpus_entries())

        assert len(before) == len(after)
        assert [entry.id for entry in before] == [entry.id for entry in after]


class TestRunDirectProbeSequentialDispatch:
    """MOD-09/T-07-11: at most ONE flood request is ever in flight, even
    while `run_direct_probe()` runs concurrently alongside a stand-in for
    the orchestrator's own concurrent gather."""

    async def test_max_in_flight_is_exactly_one(self):
        module = UnboundedConsumptionModule()
        adapter = _ConcurrencyTrackingAdapter()

        results = [
            result async for result in module.run_direct_probe(_context(), adapter)
        ]

        assert adapter.max_in_flight == 1
        assert adapter.call_count == 3
        assert len(results) == 3

    async def test_runs_inside_asyncio_gather_alongside_a_stand_in_orchestrator_coroutine(self):
        """`run_direct_probe()` completing cleanly and yielding its full
        capped result set while running concurrently inside the same
        `asyncio.gather` as another coroutine is what makes "the module
        holds no per-call mutable state on the shared adapter" a tested
        guarantee rather than an implementation detail."""
        module = UnboundedConsumptionModule()
        adapter = _ConcurrencyTrackingAdapter()

        async def _collect_direct_probe_results() -> list:
            return [
                result async for result in module.run_direct_probe(_context(), adapter)
            ]

        async def _stand_in_orchestrator_coroutine() -> str:
            await asyncio.sleep(0)
            return "orchestrator-done"

        direct_probe_results, orchestrator_result = await asyncio.gather(
            _collect_direct_probe_results(), _stand_in_orchestrator_coroutine()
        )

        assert orchestrator_result == "orchestrator-done"
        assert len(direct_probe_results) == 3
        assert adapter.max_in_flight == 1


class TestNoRetryPrimitive:
    """MOD-09/T-07-11: the module imports no retry library and constructs
    no retry primitive -- any retry wrapper, even a module-local one, would
    defeat the entire point of the retry-exempt dispatch path."""

    def test_source_contains_no_retry_library_literals(self):
        import inspect

        import llmsec.modules.unbounded_consumption as module_source

        source = inspect.getsource(module_source)
        # Strip comment lines (and pure-docstring content isn't excluded,
        # deliberately -- a future explanatory comment referencing these
        # names in prose must not be able to invalidate this gate by
        # hiding behind a `#`; grepping non-`#`-prefixed lines only strips
        # the trivial case, matching 07-03-PLAN.md's acceptance criterion
        # exactly: `grep -v '^\\s*#' ... | grep -cE '...'` returns 0).
        non_comment_lines = [
            line for line in source.splitlines() if not line.strip().startswith("#")
        ]
        non_comment_source = "\n".join(non_comment_lines)

        for forbidden in ("tenacity", "AsyncRetrying", "retry_if_exception_type"):
            assert forbidden not in non_comment_source, (
                f"found forbidden retry-library literal {forbidden!r} in "
                "unbounded_consumption.py's non-comment source"
            )
