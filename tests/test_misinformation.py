"""Tests for `MisinformationModule` (src/llmsec/modules/misinformation.py) --
Phase 9, 09-01-PLAN.md.

Task 2 (`TestGenerateCases`/`TestEvaluate`): `generate_cases()`'s shape and
`evaluate()`'s full four-tier verdict mapping, proven under a mocked judge --
`judge_misinformation` is always monkeypatched at
`llmsec.modules.misinformation.judge_misinformation` (the module-level name
the module actually calls, not the definition site -- patching the
definition site after the module has already bound the name has no effect,
the same singleton-binding trap `tests/detection/conftest.py` documents for
`judge_client`). No test in this file makes a live network call.

Task 3 (added by 09-01-PLAN.md Task 3, below `TestEvaluate`): the three
T-01-18 never-raising degrade paths, the outer-wrapper containment
guarantee, sibling-task isolation under `asyncio.gather`, and the identity
surface Task 1 established.

09-02-PLAN.md Task 1 (`TestCorpusIntegrity`, below `TestIdentitySurface`):
corpus-density (9 entries, 3 per family, both sub-shapes per family) and
cross-module attribution-boundary assertions over the grown corpus.
"""

from __future__ import annotations

import logging
from collections import Counter
from unittest.mock import AsyncMock

import pytest

from llmsec.detection.judge import JudgeVerdict
from llmsec.models import EvalResult, TargetResponse, TestCase, Verdict
from llmsec.modules.misinformation import (
    _MISSING_GROUND_TRUTH_NOTE,
    _UNRESOLVED_ENTRY_NOTE,
    MisinformationModule,
)
from llmsec.payloads import load_corpus
from llmsec.payloads.schema import PayloadEntry
from llmsec.plugins.registry import PluginRegistry


def _response(case_id: str, raw_text: str = "anything") -> TargetResponse:
    return TargetResponse(
        case_id=case_id,
        raw_text=raw_text,
        latency_ms=1.0,
        transport_mode="single",  # type: ignore[arg-type]
    )


def _case(case_id: str, prompt: str = "unused") -> TestCase:
    return TestCase(case_id=case_id, prompt=prompt, technique_id=case_id, turns=None)


# ============================================================================
# Task 2: generate_cases()
# ============================================================================


class TestGenerateCases:
    async def test_generate_cases_yields_one_case_per_entry(self):
        module = MisinformationModule()
        cases = [case async for case in module.generate_cases(None)]
        entries = load_corpus("misinformation")
        assert len(cases) == len(entries)
        assert len(cases) >= 3

    async def test_generate_cases_case_id_technique_id_turns_and_prompt_shape(self):
        module = MisinformationModule()
        cases = [case async for case in module.generate_cases(None)]
        entries_by_id = {e.id: e for e in load_corpus("misinformation")}
        for case in cases:
            assert case.case_id in entries_by_id
            assert case.technique_id == case.case_id
            assert case.turns is None
            assert case.prompt is not None and case.prompt

    async def test_generate_cases_skips_entry_with_no_prompt_and_logs(self, caplog):
        module = MisinformationModule()
        module._corpus_entries()  # populate cache first

        bad_entry = PayloadEntry(
            id="MISINFO-BAD",
            technique_family="fictional_statistic",
            description="no prompt set",
            turns=["only one turn"],
            remediation="n/a",
        )
        module._corpus = [bad_entry]
        module._entries_by_id = {bad_entry.id: bad_entry}

        with caplog.at_level(logging.WARNING):
            cases = [case async for case in module.generate_cases(None)]

        assert cases == []
        assert any("MISINFO-BAD" in record.message for record in caplog.records)

    async def test_generate_cases_never_raises_on_empty_corpus(self, monkeypatch):
        module = MisinformationModule()
        monkeypatch.setattr(module, "_corpus_entries", lambda: [])
        cases = [case async for case in module.generate_cases(None)]
        assert cases == []


# ============================================================================
# Task 2: evaluate() -- the four-tier verdict mapping
# ============================================================================


class TestEvaluate:
    def _case_and_entry(self):
        module = MisinformationModule()
        module._corpus_entries()
        entry = module._entries_by_id["MISINFO-001"]
        return module, entry

    @pytest.mark.parametrize(
        "verdict,confidence",
        [
            (Verdict.FULL_COMPROMISE, 0.95),
            (Verdict.PARTIAL_LEAK, 0.6),
            (Verdict.BLOCKED, 0.85),
            (Verdict.UNCERTAIN, 0.3),
        ],
    )
    async def test_evaluate_passes_through_verdict_and_confidence_with_no_cap(
        self, monkeypatch, verdict, confidence
    ):
        module, entry = self._case_and_entry()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=verdict, confidence=confidence, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        result = await module.evaluate(_case(entry.id), _response(entry.id))

        assert result.verdict == verdict
        assert result.confidence == confidence
        assert result.detection_layer == "judge"

    async def test_evaluate_detection_layer_is_always_judge(self, monkeypatch):
        module, entry = self._case_and_entry()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        result = await module.evaluate(_case(entry.id), _response(entry.id))

        assert result.detection_layer == "judge"

    async def test_evaluate_propagates_transport_mode(self, monkeypatch):
        module, entry = self._case_and_entry()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )
        response = TargetResponse(
            case_id=entry.id,
            raw_text="anything",
            latency_ms=1.0,
            transport_mode="multi_turn_real",  # type: ignore[arg-type]
        )

        result = await module.evaluate(_case(entry.id), response)

        assert result.transport_mode == "multi_turn_real"

    async def test_evaluate_awaits_judge_exactly_once_per_case(self, monkeypatch):
        module, entry = self._case_and_entry()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        await module.evaluate(_case(entry.id), _response(entry.id))

        assert mock_judge.await_count == 1

    async def test_evaluate_ground_truth_fact_kwarg_matches_entry_verbatim(self, monkeypatch):
        """D-66: the exact regression a future re-derive-from-prompt refactor
        would catch -- mutating `case.prompt` before `evaluate()` (simulating
        a deep-mode mutated variant) leaves the `ground_truth_fact` kwarg
        unchanged."""
        module, entry = self._case_and_entry()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )
        mutated_case = _case(entry.id, prompt="completely unrelated mutated prompt text")

        await module.evaluate(mutated_case, _response(entry.id))

        kwargs = mock_judge.call_args.kwargs
        assert kwargs["ground_truth_fact"] == entry.ground_truth_fact
        assert kwargs["probe_shape"] == entry.misinformation_probe_shape

    async def test_evaluate_remediation_reaches_eval_result(self, monkeypatch):
        module, entry = self._case_and_entry()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        result = await module.evaluate(_case(entry.id), _response(entry.id))

        assert result.remediation == entry.remediation
        assert result.remediation

    async def test_evaluate_full_slice_through_registered_plugin(self, monkeypatch):
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(
                verdict=Verdict.FULL_COMPROMISE, confidence=0.9, reasoning="fabricated a detail"
            )
        )
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        modules = PluginRegistry().load_allowed(["misinformation"])
        module = modules["misinformation"]
        cases = [case async for case in module.generate_cases(None)]
        assert len(cases) >= 3

        results = [await module.evaluate(case, _response(case.case_id)) for case in cases]
        assert len(results) == len(cases)
        assert all(isinstance(r, EvalResult) for r in results)
        assert all(r.verdict == Verdict.FULL_COMPROMISE for r in results)


# ============================================================================
# Task 3: T-01-18 containment -- three never-raising degrade paths
# ============================================================================


class TestDegradePaths:
    async def test_unresolvable_case_id_degrades_to_uncertain_without_calling_judge(
        self, monkeypatch
    ):
        module = MisinformationModule()
        module._corpus_entries()
        mock_judge = AsyncMock()
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        result = await module.evaluate(_case("NOT-IN-CORPUS"), _response("NOT-IN-CORPUS"))

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0
        assert result.detection_layer == "judge"
        assert result.evidence == _UNRESOLVED_ENTRY_NOTE
        mock_judge.assert_not_awaited()

    async def test_missing_ground_truth_fact_degrades_to_uncertain_without_calling_judge(
        self, monkeypatch
    ):
        module = MisinformationModule()
        module._corpus_entries()

        bad_entry = PayloadEntry(
            id="MISINFO-BAD",
            technique_family="fictional_statistic",
            description="no ground_truth_fact set",
            prompt="A fact is established. What detail was never stated?",
            remediation="n/a",
        )
        module._entries_by_id["MISINFO-BAD"] = bad_entry
        mock_judge = AsyncMock()
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        result = await module.evaluate(_case("MISINFO-BAD"), _response("MISINFO-BAD"))

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0
        assert result.evidence == _MISSING_GROUND_TRUTH_NOTE
        mock_judge.assert_not_awaited()

    async def test_raising_judge_call_degrades_to_uncertain_without_propagating(
        self, monkeypatch
    ):
        module = MisinformationModule()
        module._corpus_entries()
        mock_judge = AsyncMock(side_effect=RuntimeError("judge unavailable"))
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        result = await module.evaluate(_case("MISINFO-001"), _response("MISINFO-001"))

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0
        assert result.detection_layer == "judge"
        assert "judge unavailable" in result.evidence

    async def test_raising_instructor_retry_exception_degrades_to_uncertain(self, monkeypatch):
        import instructor.exceptions

        module = MisinformationModule()
        module._corpus_entries()
        exc = instructor.exceptions.InstructorRetryException(
            "schema validation failed",
            n_attempts=2,
            total_usage=0,
            messages=[],
            last_completion=None,
        )
        mock_judge = AsyncMock(side_effect=exc)
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        result = await module.evaluate(_case("MISINFO-001"), _response("MISINFO-001"))

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0


class TestOuterWrapperContainment:
    async def test_corpus_entries_raising_still_returns_uncertain(self, monkeypatch):
        module = MisinformationModule()

        def _raise():
            raise RuntimeError("corpus load exploded")

        monkeypatch.setattr(module, "_corpus_entries", _raise)

        result = await module.evaluate(_case("MISINFO-001"), _response("MISINFO-001"))

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0


class TestSiblingIsolation:
    async def test_one_raising_case_never_cancels_sibling_gather_tasks(self, monkeypatch):
        """The assertion that would actually catch a regression where a raise
        cancels sibling `asyncio.gather` tasks (T-01-18)."""
        import asyncio

        module = MisinformationModule()
        module._corpus_entries()

        async def _judge_side_effect(*, ground_truth_fact, **kwargs):
            if ground_truth_fact == "boom-trigger":
                raise RuntimeError("boom")
            return JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")

        mock_judge = AsyncMock(side_effect=_judge_side_effect)
        monkeypatch.setattr(
            "llmsec.modules.misinformation.judge_misinformation", mock_judge
        )

        entries = load_corpus("misinformation")
        assert len(entries) >= 3
        # Force exactly one entry's ground_truth_fact to trigger the raise.
        boom_id = entries[0].id
        module._entries_by_id[boom_id] = entries[0].model_copy(
            update={"ground_truth_fact": "boom-trigger"}
        )

        cases = [_case(e.id) for e in entries[:3]]
        responses = [_response(e.id) for e in entries[:3]]

        results = await asyncio.gather(
            *(module.evaluate(case, response) for case, response in zip(cases, responses))
        )

        assert len(results) == 3
        assert all(isinstance(r, EvalResult) for r in results)
        uncertain = [r for r in results if r.verdict == Verdict.UNCERTAIN]
        scored = [r for r in results if r.verdict != Verdict.UNCERTAIN]
        assert len(uncertain) == 1
        assert len(scored) == 2


class TestIdentitySurface:
    def test_uses_attacker_llm_is_true(self):
        assert MisinformationModule.uses_attacker_llm is True

    def test_owasp_ref_is_llm09(self):
        assert MisinformationModule.owasp_ref == "LLM09:2025"

    def test_no_other_discovered_module_claims_llm09(self):
        discovered = PluginRegistry().discover_all()
        refs = [cls.owasp_ref for cls in discovered.values() if cls.owasp_ref == "LLM09:2025"]
        assert refs == ["LLM09:2025"]


# ============================================================================
# 09-02 Task 1: corpus-integrity and cross-module attribution-boundary checks
# ============================================================================


class TestCorpusIntegrity:
    """MOD-12 (09-02-PLAN.md Task 1): the grown nine-entry corpus's density,
    sub-shape balance, schema-shape, and threaded-fact invariants."""

    def test_nine_entries(self):
        entries = load_corpus("misinformation")
        assert len(entries) == 9

    def test_ids_unique(self):
        entries = load_corpus("misinformation")
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    def test_every_entry_sets_prompt_never_turns(self):
        for entry in load_corpus("misinformation"):
            assert entry.prompt is not None and entry.prompt
            assert entry.turns is None

    def test_every_entry_has_required_misinformation_fields(self):
        for entry in load_corpus("misinformation"):
            assert entry.ground_truth_fact
            assert entry.misinformation_probe_shape in ("extrapolation", "restatement")
            assert entry.remediation

    def test_each_family_has_exactly_three_entries(self):
        entries = load_corpus("misinformation")
        counts = Counter(e.technique_family.value for e in entries)
        assert len(counts) == 3
        assert set(counts.values()) == {3}

    def test_both_sub_shapes_appear_in_all_three_families(self):
        entries = load_corpus("misinformation")
        by_family: dict[str, set[str]] = {}
        for e in entries:
            by_family.setdefault(e.technique_family.value, set()).add(
                e.misinformation_probe_shape
            )
        assert len(by_family) == 3
        for shapes in by_family.values():
            assert shapes == {"extrapolation", "restatement"}

    def test_ground_truth_fact_appears_verbatim_in_own_prompt(self):
        for entry in load_corpus("misinformation"):
            normalized_fact = " ".join(entry.ground_truth_fact.split())
            normalized_prompt = " ".join(entry.prompt.split())
            assert normalized_fact in normalized_prompt, entry.id

    def test_no_id_collides_with_any_other_shipped_corpus(self):
        from llmsec.plugins.registry import BUILTIN_MODULE_IDS

        misinfo_ids = {e.id for e in load_corpus("misinformation")}
        for module_id in BUILTIN_MODULE_IDS:
            if module_id == "misinformation":
                continue
            try:
                other_ids = {e.id for e in load_corpus(module_id)}
            except Exception:
                continue
            assert not (misinfo_ids & other_ids), (module_id, misinfo_ids & other_ids)


class TestAttributionBoundary:
    """The no-imperative attribution boundary (09-02-PLAN.md Task 1): a
    disjointness assertion against `prompt_injection`'s
    `indirect_data_as_instruction` family, never a hard-coded phrase
    blocklist."""

    def test_no_prompt_substantially_duplicated_from_indirect_injection(self):
        misinfo_prompts = {" ".join(e.prompt.split()) for e in load_corpus("misinformation")}
        injection_prompts = {
            " ".join((e.prompt or " ".join(e.turns or [])).split())
            for e in load_corpus("prompt_injection")
        }
        assert not (misinfo_prompts & injection_prompts)

    def test_every_misinformation_prompt_ends_in_a_question(self):
        for entry in load_corpus("misinformation"):
            assert entry.prompt.rstrip().endswith("?"), entry.id
