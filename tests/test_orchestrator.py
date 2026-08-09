"""Tests for `llmsec.orchestrator.ScanOrchestrator` (bounded-concurrency
fan-out + retry, plan 08 Task 1).

Uses an inline mock `TargetAdapter` and mock `BaseModule` (both defined
here, no need for a real adapter/module) per the plan's `<action>`.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

import httpx
import pytest

from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.orchestrator import ScanOrchestrator


class _MockAdapter:
    """Minimal `TargetAdapter`-shaped mock exposing what the orchestrator
    uses (`send` and, from plan 02-07, `send_conversation`), plus
    concurrency/call tracking shared across both dispatch paths."""

    def __init__(
        self,
        send_fn: Callable[[TestCase], "asyncio.Future[TargetResponse]"] | None = None,
        send_conversation_fn: (
            Callable[[TestCase, Callable[[str], bool] | None], "asyncio.Future[TargetResponse]"]
            | None
        ) = None,
        delay: float = 0.0,
        track_concurrency: bool = False,
    ) -> None:
        self._send_fn = send_fn
        self._send_conversation_fn = send_conversation_fn
        self._delay = delay
        self._track_concurrency = track_concurrency
        self.in_flight = 0
        self.max_in_flight = 0
        self.call_count = 0
        self.send_conversation_call_count = 0
        # (case, stop_when) per send_conversation() call, in call order.
        self.send_conversation_calls: list[tuple[TestCase, Callable[[str], bool] | None]] = []

    async def send(self, case: TestCase) -> TargetResponse:
        self.call_count += 1
        if self._track_concurrency:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._send_fn is not None:
                return await self._send_fn(case)
            return TargetResponse(case_id=case.case_id, raw_text="ok", latency_ms=1.0)
        finally:
            if self._track_concurrency:
                self.in_flight -= 1

    async def send_conversation(
        self, case: TestCase, stop_when: Callable[[str], bool] | None = None
    ) -> TargetResponse:
        self.send_conversation_call_count += 1
        self.send_conversation_calls.append((case, stop_when))
        if self._track_concurrency:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._send_conversation_fn is not None:
                return await self._send_conversation_fn(case, stop_when)
            return TargetResponse(
                case_id=case.case_id,
                raw_text="ok-multi",
                latency_ms=1.0,
                transport_mode="multi_turn_real",
                turn_replies=["ok-multi"],
            )
        finally:
            if self._track_concurrency:
                self.in_flight -= 1

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _MockModule:
    """Minimal `BaseModule`-shaped mock.

    `turns_by_case` (plan 02-07) maps a `case_id` to the `turns` list its
    generated `TestCase` should carry — `None`/absent means an ordinary
    single-turn case, matching the pre-02-07 default exactly.
    """

    id = "mock_module"
    name = "Mock Module"
    owasp_ref = "LLM07:2025"
    uses_attacker_llm = False

    def __init__(
        self,
        case_ids: list[str],
        turns_by_case: dict[str, list[str] | None] | None = None,
    ) -> None:
        self._case_ids = case_ids
        self._turns_by_case = turns_by_case or {}
        self.evaluated: list[str] = []
        # The exact `TargetResponse` each case's `evaluate()` call received,
        # in call order — lets tests assert on `transport_mode` etc.
        self.evaluated_responses: list[TargetResponse] = []

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        for case_id in self._case_ids:
            yield TestCase(
                case_id=case_id,
                prompt=f"prompt-{case_id}",
                technique_id=case_id,
                turns=self._turns_by_case.get(case_id),
            )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        self.evaluated.append(case.case_id)
        self.evaluated_responses.append(response)
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.BLOCKED,
            confidence=0.9,
            evidence=response.raw_text,
            detection_layer="regex",
        )


class _MockModuleWithAbort(_MockModule):
    """A module defining the optional `should_abort_sequence` hook: aborts
    exactly when the turn reply contains the literal marker `"STOP"`."""

    def should_abort_sequence(self, case: TestCase, turn_reply: str) -> bool:
        return "STOP" in turn_reply


class _MockModuleAbortRaises(_MockModule):
    """A module whose `should_abort_sequence` always raises, proving the
    orchestrator's duck-typed resolution swallows the exception (T-02-24)
    rather than letting it crash the scan."""

    def should_abort_sequence(self, case: TestCase, turn_reply: str) -> bool:
        raise RuntimeError("should_abort_sequence boom")


@pytest.fixture
def context() -> ScanContext:
    return ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="OPENAI_API_KEY")


async def test_run_fans_out_every_generated_case_through_adapter_and_module(context):
    adapter = _MockAdapter()
    module = _MockModule(["c1", "c2", "c3"])
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    results = await orchestrator.run(context)

    assert len(results) == 3
    assert {eval_result.case_id for _, eval_result in results} == {"c1", "c2", "c3"}
    assert adapter.call_count == 3
    assert sorted(module.evaluated) == ["c1", "c2", "c3"]
    assert all(module_id == "mock_module" for module_id, _ in results)


async def test_max_concurrency_bounds_in_flight_send_calls(context):
    adapter = _MockAdapter(delay=0.02, track_concurrency=True)
    module = _MockModule([f"c{i}" for i in range(5)])
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=2)

    await orchestrator.run(context)

    assert adapter.max_in_flight <= 2
    assert adapter.call_count == 5


async def test_transient_failure_is_retried_and_eventually_succeeds(context):
    call_counts = {"n": 0}

    async def flaky_send(case: TestCase) -> TargetResponse:
        call_counts["n"] += 1
        if call_counts["n"] < 2:
            raise httpx.TimeoutException("simulated timeout")
        return TargetResponse(case_id=case.case_id, raw_text="recovered", latency_ms=1.0)

    adapter = _MockAdapter(send_fn=flaky_send)
    module = _MockModule(["c1"])
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    results = await orchestrator.run(context)

    assert len(results) == 1
    module_id, eval_result = results[0]
    assert module_id == "mock_module"
    # module.evaluate() ran normally once the retried send() recovered —
    # not a synthetic UNCERTAIN outcome.
    assert eval_result.verdict == Verdict.BLOCKED
    assert call_counts["n"] == 2
    assert module.evaluated == ["c1"]


async def test_retry_exhausted_case_recorded_as_uncertain_not_dropped(context):
    async def always_fails(case: TestCase) -> TargetResponse:
        raise httpx.TimeoutException("simulated persistent timeout")

    adapter = _MockAdapter(send_fn=always_fails)
    module = _MockModule(["c1", "c2"])
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    results = await orchestrator.run(context)

    # Every generated case's outcome is recorded — none silently dropped.
    assert len(results) == 2
    result_case_ids = {eval_result.case_id for _, eval_result in results}
    assert result_case_ids == {"c1", "c2"}
    for _, eval_result in results:
        assert eval_result.verdict == Verdict.UNCERTAIN
        assert "transport failure" in eval_result.evidence
    # module.evaluate() must NOT run for a case whose adapter call never
    # succeeded — the synthetic UNCERTAIN result bypasses evaluate().
    assert module.evaluated == []
    # Retried up to the bounded attempt count (3) before giving up.
    assert adapter.call_count == 2 * 3


async def test_evaluate_exception_degrades_to_uncertain_not_crash(context):
    class _RaisingModule(_MockModule):
        async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
            self.evaluated.append(case.case_id)
            raise RuntimeError("judge boom")

    adapter = _MockAdapter()
    module = _RaisingModule(["c1", "c2"])
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    results = await orchestrator.run(context)

    # run() completes and returns one outcome per generated case — the
    # evaluate() exception never crashes the scan or drops the case.
    assert len(results) == 2
    result_case_ids = {eval_result.case_id for _, eval_result in results}
    assert result_case_ids == {"c1", "c2"}
    for _, eval_result in results:
        assert eval_result.verdict == Verdict.UNCERTAIN
        assert "evaluation failure" in eval_result.evidence


async def test_single_turn_case_dispatches_send_and_never_send_conversation(context):
    """A `turns=None` case calls `send()` exclusively; `evaluate()` sees a
    concrete `transport_mode == "single"` (D-15)."""
    adapter = _MockAdapter()
    module = _MockModule(["c1"])
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    await orchestrator.run(context)

    assert adapter.call_count == 1
    assert adapter.send_conversation_call_count == 0
    assert module.evaluated_responses[0].transport_mode == "single"


async def test_multi_turn_case_dispatches_send_conversation_and_never_send(context):
    """A case with a non-empty `turns` list calls `send_conversation()`
    exclusively."""
    adapter = _MockAdapter()
    module = _MockModule(["c1"], turns_by_case={"c1": ["turn 1", "turn 2"]})
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    await orchestrator.run(context)

    assert adapter.send_conversation_call_count == 1
    assert adapter.call_count == 0


async def test_empty_turns_list_treated_as_single_turn(context):
    """[EDGE:empty] `turns=[]` is treated as a single-turn case, not an
    empty conversation."""
    adapter = _MockAdapter()
    module = _MockModule(["c1"], turns_by_case={"c1": []})
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    await orchestrator.run(context)

    assert adapter.call_count == 1
    assert adapter.send_conversation_call_count == 0


async def test_stop_when_is_none_for_module_without_abort_hook(context):
    """A module that never defines `should_abort_sequence` degrades to
    `stop_when=None` rather than erroring."""
    adapter = _MockAdapter()
    module = _MockModule(["c1"], turns_by_case={"c1": ["turn 1", "turn 2"]})
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    await orchestrator.run(context)

    _case, stop_when = adapter.send_conversation_calls[0]
    assert stop_when is None


async def test_stop_when_is_callable_and_forwards_to_module_hook(context):
    """A module defining `should_abort_sequence` yields a one-argument
    `stop_when` callable that forwards to it with the case bound."""
    adapter = _MockAdapter()
    module = _MockModuleWithAbort(["c1"], turns_by_case={"c1": ["turn 1", "turn 2"]})
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    await orchestrator.run(context)

    _case, stop_when = adapter.send_conversation_calls[0]
    assert callable(stop_when)
    assert stop_when("contains the STOP marker") is True
    assert stop_when("a clean reply") is False


async def test_multi_turn_response_existing_transport_mode_is_preserved(context):
    """A response that already carries a `transport_mode` (every
    `send_conversation()` result does) is returned untouched."""
    adapter = _MockAdapter()  # default send_conversation() labels multi_turn_real
    module = _MockModule(["c1"], turns_by_case={"c1": ["turn 1", "turn 2"]})
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    await orchestrator.run(context)

    assert module.evaluated_responses[0].transport_mode == "multi_turn_real"


async def test_multi_turn_transient_failure_retried_under_same_policy_and_recovers(context):
    """A multi-turn dispatch that raises a retryable exception is retried
    under the same `stop_after_attempt(3)` policy as `send()`."""
    call_counts = {"n": 0}

    async def flaky_send_conversation(case: TestCase, stop_when):
        call_counts["n"] += 1
        if call_counts["n"] < 2:
            raise httpx.TimeoutException("simulated timeout")
        return TargetResponse(
            case_id=case.case_id,
            raw_text="recovered",
            latency_ms=1.0,
            transport_mode="multi_turn_real",
            turn_replies=["recovered"],
        )

    adapter = _MockAdapter(send_conversation_fn=flaky_send_conversation)
    module = _MockModule(["c1"], turns_by_case={"c1": ["turn 1", "turn 2"]})
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    results = await orchestrator.run(context)

    assert len(results) == 1
    _module_id, eval_result = results[0]
    assert eval_result.verdict == Verdict.BLOCKED
    assert call_counts["n"] == 2


async def test_multi_turn_retry_exhausted_degrades_to_uncertain_sibling_completes(context):
    """A multi-turn dispatch that exhausts retries produces a recorded
    `UNCERTAIN` result with `detection_layer="regex"`, and a sibling
    single-turn case in the same run still completes (T-02-27)."""

    async def always_fails(case: TestCase, stop_when):
        raise httpx.TimeoutException("simulated persistent timeout")

    adapter = _MockAdapter(send_conversation_fn=always_fails)
    module = _MockModule(
        ["seq1", "single1"],
        turns_by_case={"seq1": ["turn 1", "turn 2"], "single1": None},
    )
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    results = await orchestrator.run(context)

    assert len(results) == 2
    by_case = {eval_result.case_id: eval_result for _, eval_result in results}
    assert by_case["seq1"].verdict == Verdict.UNCERTAIN
    assert by_case["seq1"].detection_layer == "regex"
    assert "transport failure" in by_case["seq1"].evidence
    assert by_case["single1"].verdict == Verdict.BLOCKED
    # seq1 retried up to the bounded attempt count (3); single1's send()
    # never touched send_conversation() and succeeded on its own first try.
    assert adapter.send_conversation_call_count == 3
    assert adapter.call_count == 1


async def test_raising_should_abort_sequence_does_not_crash_scan(context):
    """A module whose `should_abort_sequence` itself raises does not crash
    the scan — the resolved `stop_when` swallows the exception and treats
    it as 'do not abort' (T-02-24)."""

    async def invoking_send_conversation(case: TestCase, stop_when):
        # Exercise the resolved predicate the way a real adapter would —
        # a raising module predicate must not propagate through here.
        aborted = stop_when("some turn reply") if stop_when is not None else False
        assert aborted is False
        return TargetResponse(
            case_id=case.case_id,
            raw_text="ok",
            latency_ms=1.0,
            transport_mode="multi_turn_real",
            turn_replies=["ok"],
        )

    adapter = _MockAdapter(send_conversation_fn=invoking_send_conversation)
    module = _MockModuleAbortRaises(["c1"], turns_by_case={"c1": ["turn 1", "turn 2"]})
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    results = await orchestrator.run(context)

    assert len(results) == 1
    assert results[0][1].verdict == Verdict.BLOCKED


async def test_mixed_batch_single_and_multi_turn_yields_one_result_per_case(context):
    """A mixed batch of single-turn and multi-turn cases from one module
    produces exactly one `EvalResult` per generated case — never more,
    never fewer."""
    adapter = _MockAdapter()
    module = _MockModule(
        ["single1", "single2", "seq1", "seq2"],
        turns_by_case={
            "seq1": ["turn 1", "turn 2"],
            "seq2": ["turn 1", "turn 2", "turn 3"],
        },
    )
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=5)

    results = await orchestrator.run(context)

    assert len(results) == 4
    assert {eval_result.case_id for _, eval_result in results} == {
        "single1",
        "single2",
        "seq1",
        "seq2",
    }
    assert adapter.call_count == 2
    assert adapter.send_conversation_call_count == 2


async def test_max_concurrency_bounds_mixed_dispatch_types(context):
    """The total number of concurrently in-flight dispatches is bounded by
    `max_concurrency` across both `send()` and `send_conversation()`."""
    adapter = _MockAdapter(delay=0.02, track_concurrency=True)
    module = _MockModule(
        ["s1", "s2", "s3", "m1", "m2"],
        turns_by_case={"m1": ["turn 1", "turn 2"], "m2": ["turn 1", "turn 2"]},
    )
    orchestrator = ScanOrchestrator(adapter, {"mock_module": module}, max_concurrency=2)

    await orchestrator.run(context)

    assert adapter.max_in_flight <= 2
    assert adapter.call_count == 3
    assert adapter.send_conversation_call_count == 2


def test_orchestrator_never_calls_asyncio_run():
    import inspect

    import llmsec.orchestrator as orchestrator_module

    source = inspect.getsource(orchestrator_module)
    assert "asyncio.run(" not in source
