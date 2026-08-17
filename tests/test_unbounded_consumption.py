"""Tests for `UnboundedConsumptionModule` (OWASP LLM10:2025, MOD-08/MOD-09,
07-01-PLAN.md Task 2).

Covers the deterministic threshold-scoring tier (MOD-08), the retry-exempt
`run_direct_probe()` dispatch path (MOD-09), and an end-to-end tracer proving
both dispatch paths through `run_scan()`.
"""

from __future__ import annotations

from collections import Counter

import httpx
import litellm

import llmsec.api as api_module
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import ScanContext, TargetResponse, TestCase, Verdict
from llmsec.modules.unbounded_consumption import (
    _DEFAULT_LATENCY_THRESHOLD_MS,
    _DEFAULT_TOKEN_THRESHOLD,
    UnboundedConsumptionModule,
    _score_by_threshold,
)
from llmsec.payloads import load_corpus
from llmsec.payloads.schema import ConsumptionTechniqueVector
from llmsec.plugins.registry import PluginRegistry


def _response(
    case_id: str = "case-1",
    tokens_used: int | None = 100,
    latency_ms: float = 100.0,
) -> TargetResponse:
    return TargetResponse(case_id=case_id, raw_text="reply", latency_ms=latency_ms, tokens_used=tokens_used)


def _context() -> ScanContext:
    return ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="")


class TestCorpusCoverage:
    """07-02-PLAN.md Task 1: the full 15-entry corpus covers every
    `ConsumptionTechniqueVector` family in both baseline and flood shape.

    This exact-count assertion is load-bearing: `load_corpus()` skips
    individually-invalid entries and logs rather than raising, so a schema
    mistake in the YAML would otherwise show up as a silently smaller
    corpus rather than a failure.
    """

    def test_corpus_loads_exactly_fifteen_entries(self):
        entries = load_corpus("unbounded_consumption")
        assert len(entries) == 15

    def test_ten_baseline_five_flood(self):
        entries = load_corpus("unbounded_consumption")
        baseline = [e for e in entries if not e.consumption_flood_class]
        flood = [e for e in entries if e.consumption_flood_class]
        assert len(baseline) == 10
        assert len(flood) == 5

    def test_every_family_has_two_baseline_and_one_flood(self):
        entries = load_corpus("unbounded_consumption")
        baseline_counts = Counter(
            e.technique_family for e in entries if not e.consumption_flood_class
        )
        flood_counts = Counter(
            e.technique_family for e in entries if e.consumption_flood_class
        )
        for member in ConsumptionTechniqueVector:
            assert baseline_counts[member] == 2, member
            assert flood_counts[member] == 1, member

    def test_every_entry_id_is_unique(self):
        entries = load_corpus("unbounded_consumption")
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    def test_every_remediation_is_non_empty(self):
        entries = load_corpus("unbounded_consumption")
        assert all(e.remediation.strip() for e in entries)

    def test_every_entry_sets_exactly_one_of_prompt_or_turns(self):
        entries = load_corpus("unbounded_consumption")
        for entry in entries:
            has_prompt = entry.prompt is not None
            has_turns = entry.turns is not None
            assert has_prompt != has_turns, entry.id

    def test_tracer_entries_kept_as_is(self):
        """07-01's two tracer entries (CONSUMPTION-B-01, CONSUMPTION-F-01)
        must still be present, unrenumbered, so no downstream test that
        names them needs updating."""
        entries = load_corpus("unbounded_consumption")
        ids = {e.id for e in entries}
        assert "CONSUMPTION-B-01" in ids
        assert "CONSUMPTION-F-01" in ids


class TestScoreByThreshold:
    def test_both_exceeded_is_full_compromise(self):
        response = _response(tokens_used=5000, latency_ms=20000.0)
        verdict, confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.FULL_COMPROMISE
        assert confidence == 0.8
        assert "token count 5000 exceeded threshold 2000" in evidence
        assert "latency 20000ms exceeded threshold 15000ms" in evidence

    def test_only_tokens_exceeded_is_partial_leak(self):
        response = _response(tokens_used=5000, latency_ms=100.0)
        verdict, confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.PARTIAL_LEAK
        assert confidence == 0.6
        assert "token count 5000 exceeded threshold 2000" in evidence
        assert "latency" not in evidence

    def test_only_latency_exceeded_is_partial_leak(self):
        response = _response(tokens_used=100, latency_ms=20000.0)
        verdict, confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.PARTIAL_LEAK
        assert confidence == 0.6
        assert "latency 20000ms exceeded threshold 15000ms" in evidence
        assert "token count" not in evidence

    def test_neither_exceeded_is_blocked(self):
        response = _response(tokens_used=100, latency_ms=100.0)
        verdict, confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.BLOCKED
        assert confidence == 0.8
        assert "tokens=100" in evidence
        assert "latency=100ms" in evidence

    def test_tokens_used_none_is_unavailable_not_zero_not_pass(self):
        """A `None` tokens_used must never be treated as 0 (which would
        trivially pass) or as a signal firing -- it is UNAVAILABLE, and the
        latency signal alone decides the verdict."""
        response = _response(tokens_used=None, latency_ms=20000.0)
        verdict, confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.PARTIAL_LEAK
        assert "latency 20000ms exceeded threshold 15000ms" in evidence

        response_clean = _response(tokens_used=None, latency_ms=100.0)
        verdict2, _confidence2, evidence2 = _score_by_threshold(response_clean, 2000, 15000.0)
        assert verdict2 is Verdict.BLOCKED
        assert "unavailable" in evidence2

    def test_strict_greater_than_no_rounding_in_comparison(self):
        """Values exactly AT the threshold must not fire -- strict `>`,
        never `>=`, and no pre-rounding before the comparison."""
        response = _response(tokens_used=2000, latency_ms=15000.0)
        verdict, _confidence, _evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.BLOCKED


class TestThresholdBoundaries:
    """07-02-PLAN.md Task 2: every boundary/precision case from `<behavior>`,
    named explicitly so a future edit to `_score_by_threshold()` can't
    silently drift `>=` back in."""

    def test_tokens_exactly_at_threshold_not_exceeded(self):
        response = _response(tokens_used=2000, latency_ms=100.0)
        verdict, _confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.BLOCKED
        assert "exceeded" not in evidence

    def test_tokens_one_over_threshold_is_exceeded(self):
        response = _response(tokens_used=2001, latency_ms=100.0)
        verdict, _confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.PARTIAL_LEAK
        assert "token count 2001 exceeded threshold 2000" in evidence

    def test_latency_exactly_at_threshold_not_exceeded(self):
        response = _response(tokens_used=100, latency_ms=15000.0)
        verdict, _confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.BLOCKED
        assert "exceeded" not in evidence

    def test_latency_fractional_amount_over_threshold_is_exceeded(self):
        """`15000.0000001` against a `15000.0` threshold IS exceeded --
        the comparison runs on raw floats, never pre-rounded."""
        response = _response(tokens_used=100, latency_ms=15000.0000001)
        verdict, _confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.PARTIAL_LEAK
        assert "exceeded" in evidence

    def test_both_exceeded_confidence_is_zero_point_eight(self):
        response = _response(tokens_used=2001, latency_ms=15000.0000001)
        verdict, confidence, _evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.FULL_COMPROMISE
        assert confidence == 0.8

    def test_exactly_one_exceeded_confidence_is_zero_point_six(self):
        response = _response(tokens_used=2001, latency_ms=100.0)
        verdict, confidence, _evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.PARTIAL_LEAK
        assert confidence == 0.6

    def test_neither_exceeded_confidence_is_zero_point_eight(self):
        response = _response(tokens_used=100, latency_ms=100.0)
        verdict, confidence, _evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.BLOCKED
        assert confidence == 0.8

    async def test_evaluate_at_exact_boundary_routes_through_threshold_layer(self):
        """Drives the exact-boundary case through the real `evaluate()`
        path (not just `_score_by_threshold()` directly)."""
        module = UnboundedConsumptionModule()
        baseline_entry = module._baseline_entries()[0]
        case = TestCase(
            case_id=baseline_entry.id, prompt=baseline_entry.prompt or "", technique_id=baseline_entry.id
        )
        response = TargetResponse(
            case_id=baseline_entry.id, raw_text="reply", latency_ms=15000.0, tokens_used=2000
        )
        result = await module.evaluate(case, response)
        assert result.verdict is Verdict.BLOCKED
        assert result.detection_layer == "threshold"
        assert result.remediation == baseline_entry.remediation


class TestUnavailableTokenSignal:
    """07-02-PLAN.md Task 2: `tokens_used is None` is UNAVAILABLE -- never
    zero, never a pass, and `FULL_COMPROMISE` must be structurally
    unreachable since only one signal (latency) could be measured."""

    def test_tokens_none_latency_under_threshold_is_blocked_states_unavailable(self):
        response = _response(tokens_used=None, latency_ms=100.0)
        verdict, confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.BLOCKED
        assert confidence == 0.8
        assert "unavailable" in evidence

    def test_tokens_none_latency_over_threshold_is_partial_leak_never_full_compromise(self):
        response = _response(tokens_used=None, latency_ms=20000.0)
        verdict, confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert verdict is Verdict.PARTIAL_LEAK
        assert verdict is not Verdict.FULL_COMPROMISE
        assert confidence == 0.6
        assert "unavailable" in evidence

    def test_full_compromise_structurally_unreachable_when_tokens_none(self):
        for latency_ms in (0.0, 100.0, 15000.0, 15000.0001, 1_000_000.0):
            response = _response(tokens_used=None, latency_ms=latency_ms)
            verdict, _confidence, _evidence = _score_by_threshold(response, 2000, 15000.0)
            assert verdict is not Verdict.FULL_COMPROMISE

    async def test_run_direct_probe_success_branch_with_unavailable_tokens(self):
        """Drives the unavailable-token-signal edge through the real
        `run_direct_probe()` dispatch path, not just
        `_score_by_threshold()` directly."""
        module = UnboundedConsumptionModule()
        capped_entries = module._flood_entries()[: module._flood_probe_cap]
        target_case_id = capped_entries[0].id
        behaviors: dict[str, object] = {
            target_case_id: TargetResponse(
                case_id=target_case_id, raw_text="ok", latency_ms=20000.0, tokens_used=None
            )
        }
        for entry in capped_entries[1:]:
            behaviors[entry.id] = TargetResponse(
                case_id=entry.id, raw_text="ok", latency_ms=1.0, tokens_used=1
            )
        adapter = _CountingStubAdapter(behaviors)
        results = [result async for result in module.run_direct_probe(_context(), adapter)]
        target_result = next(r for r in results if r.case_id == target_case_id)
        assert target_result.verdict is Verdict.PARTIAL_LEAK
        assert "unavailable" in target_result.evidence


class TestEvidenceNamesSignal:
    """07-02-PLAN.md Task 2 / D-06 / ROADMAP SC#4: every firing signal's
    evidence names the signal, the observed value, AND the threshold it was
    compared against -- including the clean (BLOCKED) case."""

    def test_both_signals_evidence_contains_observed_and_threshold(self):
        response = _response(tokens_used=5000, latency_ms=20000.0)
        _verdict, _confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert "5000" in evidence and "2000" in evidence
        assert "20000" in evidence and "15000" in evidence

    def test_blocked_evidence_also_names_both_observed_values_and_thresholds(self):
        """Even a clean (BLOCKED) verdict's evidence names BOTH observed
        values AND BOTH thresholds they were compared against -- not just
        that nothing exceeded."""
        response = _response(tokens_used=100, latency_ms=100.0)
        _verdict, _confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert "100" in evidence
        assert "2000" in evidence  # token threshold
        assert "15000" in evidence  # latency threshold

    def test_unavailable_tokens_evidence_still_names_latency_and_its_threshold(self):
        response = _response(tokens_used=None, latency_ms=100.0)
        _verdict, _confidence, evidence = _score_by_threshold(response, 2000, 15000.0)
        assert "unavailable" in evidence
        assert "100" in evidence
        assert "15000" in evidence

    async def test_evaluate_threads_evidence_detection_layer_and_remediation(self):
        """Drives the evidence-naming contract through the real
        `evaluate()` path, confirming `detection_layer == "threshold"` and
        `remediation` threading alongside the evidence text."""
        module = UnboundedConsumptionModule()
        baseline_entry = module._baseline_entries()[0]
        case = TestCase(
            case_id=baseline_entry.id, prompt=baseline_entry.prompt or "", technique_id=baseline_entry.id
        )
        response = TargetResponse(
            case_id=baseline_entry.id, raw_text="reply", latency_ms=20000.0, tokens_used=5000
        )
        result = await module.evaluate(case, response)
        assert result.verdict is Verdict.FULL_COMPROMISE
        assert result.detection_layer == "threshold"
        assert result.remediation == baseline_entry.remediation
        assert "5000" in result.evidence and "20000" in result.evidence


class _CountingStubAdapter:
    """A `TargetAdapter`-shaped stub whose `send()`/`send_conversation()`
    raise a configured exception (or return a configured response) keyed by
    `case_id`, and whose call counter proves `run_direct_probe()` makes
    EXACTLY one attempt per case (no retry)."""

    supports_system_prompt_override = False
    supports_multi_turn = False

    def __init__(self, behaviors: dict[str, object]) -> None:
        # behaviors[case_id] is either an Exception instance to raise, or a
        # TargetResponse to return.
        self._behaviors = behaviors
        self.call_counts: dict[str, int] = {}
        self.closed = False

    async def send(self, case: TestCase) -> TargetResponse:
        self.call_counts[case.case_id] = self.call_counts.get(case.case_id, 0) + 1
        behavior = self._behaviors[case.case_id]
        if isinstance(behavior, Exception):
            raise behavior
        return behavior

    async def send_conversation(self, case: TestCase, stop_when=None) -> TargetResponse:
        return await self.send(case)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://target.test/chat")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"status {status_code}", request=request, response=response)


class TestRunDirectProbeRetryExemption:
    """MOD-09: a flood-class case whose adapter call raises a 429/timeout
    signal scores `blocked` on the FIRST response, with exactly one adapter
    call -- never retried."""

    async def _run_flood_probe(self, exc_or_response):
        """Dispatches the real, capped `run_direct_probe()` loop with the
        FIRST capped flood entry behaving per `exc_or_response`, and every
        OTHER capped entry given a benign, clearly-within-threshold
        response -- so the 07-02 corpus expansion (five flood entries, cap
        of three) doesn't change what these tests exercise (the first
        case's outcome), while still driving the real multi-entry loop."""
        module = UnboundedConsumptionModule()
        capped_entries = module._flood_entries()[: module._flood_probe_cap]
        assert capped_entries, "expected at least one flood-class corpus entry within the cap"
        target_case_id = capped_entries[0].id
        behaviors: dict[str, object] = {target_case_id: exc_or_response}
        for entry in capped_entries[1:]:
            behaviors[entry.id] = TargetResponse(
                case_id=entry.id, raw_text="ok", latency_ms=1.0, tokens_used=1
            )
        adapter = _CountingStubAdapter(behaviors)
        results = [
            result async for result in module.run_direct_probe(_context(), adapter)
        ]
        target_result = next(r for r in results if r.case_id == target_case_id)
        return target_result, results, adapter, target_case_id

    async def test_429_scores_blocked_exactly_one_call(self):
        target_result, results, adapter, case_id = await self._run_flood_probe(
            _http_status_error(429)
        )
        assert target_result.verdict is Verdict.BLOCKED
        assert target_result.detection_layer == "threshold"
        assert adapter.call_counts[case_id] == 1
        assert len(results) == len(adapter.call_counts)  # one result per dispatched case

    async def test_timeout_scores_blocked_exactly_one_call(self):
        target_result, _results, adapter, case_id = await self._run_flood_probe(
            httpx.TimeoutException("timed out")
        )
        assert target_result.verdict is Verdict.BLOCKED
        assert adapter.call_counts[case_id] == 1

    async def test_litellm_rate_limit_error_scores_blocked_exactly_one_call(self):
        exc = litellm.RateLimitError(message="rate limited", llm_provider="openai", model="gpt-4o-mini")
        target_result, _results, adapter, case_id = await self._run_flood_probe(exc)
        assert target_result.verdict is Verdict.BLOCKED
        assert adapter.call_counts[case_id] == 1

    async def test_litellm_timeout_scores_blocked_exactly_one_call(self):
        exc = litellm.Timeout(message="timed out", model="gpt-4o-mini", llm_provider="openai")
        target_result, _results, adapter, case_id = await self._run_flood_probe(exc)
        assert target_result.verdict is Verdict.BLOCKED
        assert adapter.call_counts[case_id] == 1

    async def test_503_scores_uncertain_exactly_one_call_no_retry(self):
        """A non-429 HTTP status (including 5xx) is NOT a MOD-09 blocked
        signal -- it degrades to UNCERTAIN, but is still never retried."""
        target_result, _results, adapter, case_id = await self._run_flood_probe(
            _http_status_error(503)
        )
        assert target_result.verdict is Verdict.UNCERTAIN
        assert adapter.call_counts[case_id] == 1

    async def test_success_response_scores_through_shared_threshold_helper(self):
        """A flood-class case whose adapter call returns normally (no
        429/timeout) is scored by the SAME `_score_by_threshold()` helper
        `evaluate()` uses."""
        response = TargetResponse(case_id="placeholder", raw_text="ok", latency_ms=1.0, tokens_used=1)
        target_result, _results, adapter, case_id = await self._run_flood_probe(response)
        assert target_result.verdict is Verdict.BLOCKED
        assert target_result.detection_layer == "threshold"
        assert adapter.call_counts[case_id] == 1


class TestFloodProbeCap:
    def test_dispatch_never_exceeds_cap(self):
        """Cap enforcement is on the SLICE `run_direct_probe()` applies,
        not on corpus size -- assert the slice math directly."""
        module = UnboundedConsumptionModule()
        capped = module._flood_entries()[: module._flood_probe_cap]
        assert len(capped) == min(len(module._flood_entries()), module._flood_probe_cap)


class TestConsumptionConfigRoundTrip:
    def test_none_resolves_to_module_defaults(self):
        module = UnboundedConsumptionModule()
        assert module._token_threshold == _DEFAULT_TOKEN_THRESHOLD
        assert module._latency_threshold_ms == _DEFAULT_LATENCY_THRESHOLD_MS

    def test_explicit_values_override_defaults(self):
        module = UnboundedConsumptionModule(
            consumption_token_threshold=999, consumption_latency_threshold_ms=1234.0
        )
        assert module._token_threshold == 999
        assert module._latency_threshold_ms == 1234.0

    async def test_config_driven_threshold_flip_via_load_allowed(self):
        """07-02-PLAN.md Task 3 / ROADMAP SC#3: the SAME `TargetResponse`
        scores differently depending ONLY on the thresholds threaded
        through `PluginRegistry().load_allowed()`'s `module_config` (the
        real `accepted_params` filter path a scan actually uses), with no
        code change between the two runs -- a user who lowers
        `consumption_token_threshold` in `llmsec.config.yaml` and re-runs
        the same scan sees the same response flip verdict."""
        response = TargetResponse(case_id="fixed-case", raw_text="reply", latency_ms=100.0, tokens_used=100)

        high_modules = PluginRegistry().load_allowed(
            ["unbounded_consumption"],
            module_config={
                "unbounded_consumption": {
                    "consumption_token_threshold": 5000,
                    "consumption_latency_threshold_ms": 20000.0,
                }
            },
        )
        high_module = high_modules["unbounded_consumption"]
        assert high_module._token_threshold == 5000
        assert high_module._latency_threshold_ms == 20000.0
        case = TestCase(case_id="fixed-case", prompt="p", technique_id="fixed-case")
        high_result = await high_module.evaluate(case, response)

        low_modules = PluginRegistry().load_allowed(
            ["unbounded_consumption"],
            module_config={
                "unbounded_consumption": {
                    "consumption_token_threshold": 10,
                    "consumption_latency_threshold_ms": 1.0,
                }
            },
        )
        low_module = low_modules["unbounded_consumption"]
        assert low_module._token_threshold == 10
        assert low_module._latency_threshold_ms == 1.0
        low_result = await low_module.evaluate(case, response)

        assert high_result.verdict is Verdict.BLOCKED
        assert low_result.verdict is Verdict.FULL_COMPROMISE
        assert high_result.verdict != low_result.verdict


class TestGenerateCasesExcludesFloodClass:
    async def test_generate_cases_yields_only_baseline_entries(self):
        module = UnboundedConsumptionModule()
        cases = [case async for case in module.generate_cases(_context())]
        flood_ids = {entry.id for entry in module._flood_entries()}
        assert all(case.case_id not in flood_ids for case in cases)
        assert len(cases) == len(module._baseline_entries())


def _consumption_http_config(tmp_path) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(
            type="http_app",
            method="POST",
            url="http://target.test/chat",
            headers={},
            body_template='{"message": "{{payload}}"}',
            response_path="response",
        ),
        enabled_modules=["unbounded_consumption"],
        max_concurrency=5,
        output_dir=str(tmp_path / "reports"),
        judge_model="openai/gpt-4o-mini",
        judge_api_key_env=None,
    )


class TestTracerEndToEnd:
    """Drives the real `UnboundedConsumptionModule` through the real
    `run_scan()` pipeline end-to-end: one baseline probe through the
    ordinary orchestrator path, one flood probe through `run_direct_probe()`,
    proving both dispatch paths are wired correctly (Task 2's `<done>`)."""

    async def test_run_scan_produces_baseline_and_flood_results(self, tmp_path, monkeypatch):
        # Discover the real corpus entry ids up front so the stub adapter's
        # behavior map is keyed correctly regardless of exact ids.
        probe_module = UnboundedConsumptionModule()
        baseline_id = probe_module._baseline_entries()[0].id
        flood_id = probe_module._flood_entries()[0].id

        behaviors = {
            baseline_id: TargetResponse(
                case_id=baseline_id, raw_text="a short reply", latency_ms=10.0, tokens_used=5
            ),
            flood_id: _http_status_error(429),
        }
        created: list[_CountingStubAdapter] = []

        def _factory(*args, **kwargs) -> _CountingStubAdapter:
            instance = _CountingStubAdapter(behaviors)
            created.append(instance)
            return instance

        monkeypatch.setattr(api_module, "HttpAppAdapter", _factory)

        config = _consumption_http_config(tmp_path)
        report = await api_module.run_scan(config, bypass_flag=True)

        case_ids = {result.case_id: result for result in report.case_log}
        assert baseline_id in case_ids
        assert flood_id in case_ids
        assert case_ids[baseline_id].detection_layer == "threshold"
        assert case_ids[flood_id].verdict is Verdict.BLOCKED
        assert created[0].call_counts[flood_id] == 1
