"""Tests for `VectorEmbeddingWeaknessesModule`
(src/llmsec/modules/vector_embedding_weaknesses.py) -- Phase 8,
08-01-PLAN.md (family 1) and 08-03-PLAN.md (family 2).

Task 1 (08-01): registered module skeleton, curated `cross_document_leakage`
corpus, `generate_cases()`/`evaluate()`'s canary tier, end-to-end via
`PluginRegistry().load_allowed()`.

Task 2 (08-01): the D-04/D-05/marker-distinctness attribution-boundary
guard suite and the `_VECTOR_CONTEXT_SIMULATED_NOTE` honest-limits
disclosure.

Task 1 (08-03, `TestOverTrustCorpus` above `TestGenerateCases`): the
`retrieved_content_over_trust` corpus entries.

Task 2 (08-03, `TestOverTrustEvaluate` below `TestEvaluate`): the judge
dispatch branch for family 2, mirroring `tests/test_data_poisoning.py`'s
`mock_judge`-patching convention -- no live network call and no live judge
call is made anywhere in this file; `judge_context_over_trust` is always
monkeypatched at `llmsec.modules.vector_embedding_weaknesses.
judge_context_over_trust`.

Task 3 (08-03, `TestAttributionAcrossModules`/`TestFullCorpusEndToEnd`/
`TestCoverageCompleteness` at the bottom of this file): the ROADMAP
success-criterion #1/#3/#4 proofs.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

from llmsec.detection.canary import CANARY_TOKEN
from llmsec.detection.judge import JudgeVerdict
from llmsec.models import EvalResult, TargetResponse, TestCase, Verdict
from llmsec.modules.vector_embedding_weaknesses import (
    CONTEXT_LEAK_MARKER,
    VectorEmbeddingWeaknessesModule,
    find_context_leak_marker,
)
from llmsec.payloads import load_corpus
from llmsec.payloads.schema import VectorContextTechniqueVector
from llmsec.plugins.registry import BUILTIN_MODULE_IDS, PluginRegistry


def _response(raw_text: str, transport_mode: str = "single") -> TargetResponse:
    return TargetResponse(
        case_id="CTXLEAK-001",
        raw_text=raw_text,
        latency_ms=1.0,
        transport_mode=transport_mode,  # type: ignore[arg-type]
    )


# ============================================================================
# Task 1: registered module skeleton
# ============================================================================


class TestModuleIdentity:
    def test_class_attributes(self):
        assert VectorEmbeddingWeaknessesModule.id == "vector_embedding_weaknesses"
        assert VectorEmbeddingWeaknessesModule.owasp_ref == "LLM08:2025"
        assert VectorEmbeddingWeaknessesModule.uses_attacker_llm is True

    def test_bare_instantiation_works(self):
        module = VectorEmbeddingWeaknessesModule()
        assert module.id == "vector_embedding_weaknesses"

    def test_no_run_standalone_audit_or_run_direct_probe_override(self):
        """D-01/D-02: normal live-probing contract only."""
        assert "run_standalone_audit" not in VectorEmbeddingWeaknessesModule.__dict__
        assert "run_direct_probe" not in VectorEmbeddingWeaknessesModule.__dict__

    def test_discoverable_via_registry_discover_all(self):
        discovered = PluginRegistry().discover_all()
        assert "vector_embedding_weaknesses" in discovered
        assert discovered["vector_embedding_weaknesses"] is VectorEmbeddingWeaknessesModule

    def test_in_builtin_module_ids(self):
        assert "vector_embedding_weaknesses" in BUILTIN_MODULE_IDS

    def test_load_allowed_vector_embedding_weaknesses_alone(self):
        loaded = PluginRegistry().load_allowed(["vector_embedding_weaknesses"])
        assert set(loaded.keys()) == {"vector_embedding_weaknesses"}
        module = loaded["vector_embedding_weaknesses"]
        assert isinstance(module, VectorEmbeddingWeaknessesModule)
        assert module.owasp_ref == "LLM08:2025"
        assert module.uses_attacker_llm is True

    def test_load_allowed_none_loads_it_too(self):
        loaded = PluginRegistry().load_allowed(None)
        assert "vector_embedding_weaknesses" in loaded


# ============================================================================
# Task 1: corpus
# ============================================================================


def _family1_entries():
    return [
        e
        for e in load_corpus("vector_embedding_weaknesses")
        if e.technique_family == VectorContextTechniqueVector.CROSS_DOCUMENT_LEAKAGE
    ]


def _family2_entries():
    return [
        e
        for e in load_corpus("vector_embedding_weaknesses")
        if e.technique_family == VectorContextTechniqueVector.RETRIEVED_CONTENT_OVER_TRUST
    ]


class TestCorpus:
    """`cross_document_leakage` (D-04 family 1, 08-01) corpus assertions."""

    def test_corpus_loads_at_least_four_valid_entries(self):
        entries = load_corpus("vector_embedding_weaknesses")
        assert len(entries) >= 4

    def test_every_family1_entry_is_cross_document_leakage_with_prompt_no_turns(self):
        entries = _family1_entries()
        assert len(entries) > 0
        for entry in entries:
            assert entry.technique_family == VectorContextTechniqueVector.CROSS_DOCUMENT_LEAKAGE
            assert entry.turns is None
            assert entry.prompt is not None and entry.prompt

    def test_every_family1_entry_has_the_context_marker_placeholder(self):
        entries = _family1_entries()
        assert len(entries) > 0
        for entry in entries:
            assert "{context_marker}" in entry.prompt, entry.id

    def test_every_family1_id_starts_with_ctxleak_prefix_and_is_unique(self):
        entries = _family1_entries()
        assert all(e.id.startswith("CTXLEAK-") for e in entries)
        assert len({e.id for e in entries}) == len(entries)

    def test_every_entry_has_non_empty_remediation(self):
        entries = load_corpus("vector_embedding_weaknesses")
        for entry in entries:
            assert entry.remediation


# ============================================================================
# Task 1 (08-03): retrieved_content_over_trust (D-04 family 2) corpus
# ============================================================================


class TestOverTrustCorpus:
    """`retrieved_content_over_trust` (D-04 family 2, 08-03) corpus
    assertions -- the boundary rule (no embedded imperative, no
    `{context_marker}`/marker literal) is asserted separately by
    `TestAttributionBoundaryGuards` below, which already iterates over the
    whole corpus.
    """

    def test_at_least_two_entries_per_vector_context_technique_family(self):
        entries = load_corpus("vector_embedding_weaknesses")
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry.technique_family.value] = counts.get(entry.technique_family.value, 0) + 1
        for member in VectorContextTechniqueVector:
            assert counts.get(member.value, 0) >= 2, member

    def test_family2_entries_have_prompt_no_turns_and_remediation(self):
        entries = _family2_entries()
        assert len(entries) >= 2
        for entry in entries:
            assert entry.turns is None
            assert entry.prompt is not None and entry.prompt
            assert entry.remediation

    def test_family2_entries_have_planted_claim_appearing_verbatim_in_prompt(self):
        entries = _family2_entries()
        assert len(entries) >= 2
        for entry in entries:
            assert entry.planted_claim, entry.id
            assert entry.planted_claim in entry.prompt, entry.id

    def test_family1_entries_leave_planted_claim_unset(self):
        entries = _family1_entries()
        assert len(entries) >= 2
        for entry in entries:
            assert entry.planted_claim is None, entry.id

    def test_family2_entries_have_no_context_marker_placeholder_or_literal(self):
        entries = _family2_entries()
        assert len(entries) >= 2
        for entry in entries:
            assert "{context_marker}" not in entry.prompt, entry.id
            assert CONTEXT_LEAK_MARKER not in entry.prompt, entry.id

    def test_family2_ids_start_with_overtrust_prefix_and_are_unique(self):
        entries = _family2_entries()
        assert all(e.id.startswith("OVERTRUST-") for e in entries)
        assert len({e.id for e in entries}) == len(entries)

    def test_no_id_collides_across_the_two_families(self):
        entries = load_corpus("vector_embedding_weaknesses")
        assert len({e.id for e in entries}) == len(entries)


# ============================================================================
# Task 1: generate_cases()
# ============================================================================


class TestGenerateCases:
    async def test_yields_one_case_per_entry_with_marker_substituted(self):
        module = VectorEmbeddingWeaknessesModule()
        cases = [case async for case in module.generate_cases(None)]
        entries = load_corpus("vector_embedding_weaknesses")
        assert len(cases) == len(entries)
        family1_ids = {e.id for e in entries if e.technique_family.value == "cross_document_leakage"}
        for case in cases:
            assert case.turns is None
            assert "{context_marker}" not in case.prompt
            # The marker substitution only applies to family-1
            # (cross_document_leakage) entries -- family-2
            # (retrieved_content_over_trust) entries never carry the
            # placeholder in the first place (D-04), so the marker is
            # correctly absent from their generated prompts too.
            if case.case_id in family1_ids:
                assert CONTEXT_LEAK_MARKER in case.prompt
            else:
                assert CONTEXT_LEAK_MARKER not in case.prompt

    async def test_skips_entry_with_no_prompt(self):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()  # populate cache first

        from llmsec.payloads.schema import PayloadEntry

        bad_entry = PayloadEntry(
            id="CTXLEAK-BAD",
            technique_family="cross_document_leakage",
            description="no prompt set",
            turns=["only one turn"],
            remediation="n/a",
        )
        module._corpus = [bad_entry]
        module._entries_by_id = {bad_entry.id: bad_entry}

        cases = [case async for case in module.generate_cases(None)]
        assert cases == []


# ============================================================================
# Task 1: evaluate() -- canary tier
# ============================================================================


class TestEvaluate:
    async def test_marker_echo_resolves_full_compromise_at_canary_layer(self):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        case = TestCase(
            case_id="CTXLEAK-001", prompt="unused", technique_id="CTXLEAK-001", turns=None
        )
        response = _response(f"Some answer that echoes {CONTEXT_LEAK_MARKER} unexpectedly.")

        result = await module.evaluate(case, response)

        assert result.verdict == Verdict.FULL_COMPROMISE
        assert result.detection_layer == "canary"
        assert result.confidence == 0.9

    async def test_clean_chunk_scoped_answer_resolves_blocked_at_canary_layer(self):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        case = TestCase(
            case_id="CTXLEAK-001", prompt="unused", technique_id="CTXLEAK-001", turns=None
        )
        response = _response("You reset your password via Account Settings > Security.")

        result = await module.evaluate(case, response)

        assert result.verdict == Verdict.BLOCKED
        assert result.detection_layer == "canary"
        assert result.confidence == 0.8

    async def test_unresolvable_case_id_resolves_uncertain_without_raising(self):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        case = TestCase(
            case_id="NOT-IN-CORPUS", prompt="unused", technique_id="NOT-IN-CORPUS", turns=None
        )
        response = _response("Anything.")

        result = await module.evaluate(case, response)

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0
        assert result.detection_layer == "judge"

    async def test_evaluate_never_raises_on_unexpected_exception(self, monkeypatch):
        module = VectorEmbeddingWeaknessesModule()

        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.find_context_leak_marker", _raise
        )
        case = TestCase(
            case_id="CTXLEAK-001", prompt="unused", technique_id="CTXLEAK-001", turns=None
        )
        response = _response("anything")

        result = await module.evaluate(case, response)

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0


# ============================================================================
# Task 2 (08-03): evaluate() -- retrieved_content_over_trust judge tier
# ============================================================================


class TestOverTrustEvaluate:
    def _case(self, case_id: str = "OVERTRUST-001") -> TestCase:
        return TestCase(case_id=case_id, prompt="unused", technique_id=case_id, turns=None)

    async def test_full_compromise_judge_verdict_resolves_finding_at_judge_layer(
        self, monkeypatch
    ):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(
                verdict=Verdict.FULL_COMPROMISE,
                confidence=0.9,
                reasoning="Flatly restated the planted claim with no source framing.",
            )
        )
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        response = _response("Yes, that's correct -- that's exactly the policy.")
        result = await module.evaluate(self._case(), response)

        assert result.verdict == Verdict.FULL_COMPROMISE
        assert result.detection_layer == "judge"
        assert result.confidence == 0.9

    async def test_blocked_judge_verdict_resolves_blocked_at_judge_layer(self, monkeypatch):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(
                verdict=Verdict.BLOCKED,
                confidence=0.85,
                reasoning="Attributed the claim to the retrieved document and caveated it.",
            )
        )
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        response = _response(
            "According to the retrieved policy document, that's stated -- but I can't "
            "confirm it's accurate, you should verify with HR."
        )
        result = await module.evaluate(self._case(), response)

        assert result.verdict == Verdict.BLOCKED
        assert result.detection_layer == "judge"

    async def test_judge_called_exactly_once_per_evaluate(self, monkeypatch):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        await module.evaluate(self._case(), _response("anything"))

        assert mock_judge.await_count == 1

    async def test_planted_claim_passed_to_judge_is_corpus_entry_value_verbatim(
        self, monkeypatch
    ):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        entry = module._entries_by_id["OVERTRUST-001"]
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        await module.evaluate(self._case(), _response("anything"))

        kwargs = mock_judge.call_args.kwargs
        assert kwargs["planted_claim"] == entry.planted_claim
        assert kwargs["target_reply"] == "anything"

    async def test_raising_judge_degrades_to_uncertain_without_propagating(self, monkeypatch):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        mock_judge = AsyncMock(side_effect=RuntimeError("judge unavailable"))
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        result = await module.evaluate(self._case(), _response("anything"))

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0
        assert result.detection_layer == "judge"
        assert "judge unavailable" in result.evidence

    async def test_missing_planted_claim_degrades_to_uncertain_without_calling_judge(
        self, monkeypatch
    ):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()

        from llmsec.payloads.schema import PayloadEntry

        bad_entry = PayloadEntry(
            id="OVERTRUST-BAD",
            technique_family="retrieved_content_over_trust",
            description="no planted_claim set",
            prompt="Here is a chunk. Based on it, answer the question.",
            remediation="n/a",
        )
        module._entries_by_id["OVERTRUST-BAD"] = bad_entry
        mock_judge = AsyncMock()
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        result = await module.evaluate(self._case("OVERTRUST-BAD"), _response("anything"))

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0
        mock_judge.assert_not_awaited()

    async def test_cross_document_leakage_case_never_triggers_judge_call(self, monkeypatch):
        module = VectorEmbeddingWeaknessesModule()
        module._corpus_entries()
        mock_judge = AsyncMock()
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        response = _response(f"Echoing {CONTEXT_LEAK_MARKER} by mistake.")
        result = await module.evaluate(self._case("CTXLEAK-001"), response)

        assert result.verdict == Verdict.FULL_COMPROMISE
        assert result.detection_layer == "canary"
        mock_judge.assert_not_awaited()


# ============================================================================
# Task 1: find_context_leak_marker()
# ============================================================================


class TestFindContextLeakMarker:
    def test_none_and_blank_return_false(self):
        assert find_context_leak_marker(None) is False
        assert find_context_leak_marker("") is False
        assert find_context_leak_marker("   ") is False

    def test_marker_present_returns_true_case_insensitively(self):
        assert find_context_leak_marker(f"text {CONTEXT_LEAK_MARKER} text") is True
        assert find_context_leak_marker(f"text {CONTEXT_LEAK_MARKER.lower()} text") is True

    def test_marker_absent_returns_false(self):
        assert find_context_leak_marker("nothing interesting here") is False


# ============================================================================
# Task 1: end-to-end (through the real registered plugin)
# ============================================================================


class TestEndToEndThroughRegistry:
    async def test_full_slice_through_load_allowed(self):
        modules = PluginRegistry().load_allowed(["vector_embedding_weaknesses"])
        module = modules["vector_embedding_weaknesses"]

        cases = [case async for case in module.generate_cases(None)]
        assert len(cases) >= 4

        leak_case = cases[0]
        leak_response = _response(f"Echoing {CONTEXT_LEAK_MARKER} by mistake.")
        leak_result = await module.evaluate(leak_case, leak_response)
        assert leak_result.verdict == Verdict.FULL_COMPROMISE
        assert leak_result.detection_layer == "canary"

        clean_response = _response("A clean, scoped answer with no marker at all.")
        clean_result = await module.evaluate(leak_case, clean_response)
        assert clean_result.verdict == Verdict.BLOCKED
        assert clean_result.detection_layer == "canary"


# ============================================================================
# Task 2: attribution-boundary guards (D-04/D-05/marker-distinctness)
# ============================================================================

# Directive cue regexes covering the shapes `prompt_injection.yaml`'s
# `indirect_data_as_instruction` (`INDIRECT-00*`) family actually uses, plus
# obvious variants. Kept as a small, named list local to this test so a
# future corpus edit that re-introduces an assistant-addressed directive
# fails this guard rather than silently crossing D-04's boundary.
_DIRECTIVE_CUE_PATTERNS = [
    r"note to assistant",
    r"ignore (the )?(above|previous) instructions",
    r"before (summarizing|answering|responding)[^.]*output",
    r"you must now",
    r"system:\s",
]
_DIRECTIVE_CUE_RE = re.compile("|".join(_DIRECTIVE_CUE_PATTERNS), re.IGNORECASE)

# PII-shaped value regexes -- kept local to this test rather than importable
# from a shared module (no single shared PII regex vocabulary is importable
# outside `pii_exfiltration`'s own detection internals).
_PII_SHAPE_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
}


class TestAttributionBoundaryGuards:
    """RESEARCH Pitfall 3 / D-04 / D-05: these guards keep the two modules'
    corpora and marker semantics disjoint. A future corpus edit that
    silently re-crosses the boundary fails one of these tests.
    """

    def test_no_entry_contains_an_embedded_imperative(self):
        """D-04: no `vector_embedding_weaknesses` entry may address the
        assistant with a directive -- that shape belongs to
        `prompt_injection`'s `indirect_data_as_instruction` family
        (`INDIRECT-001`..`INDIRECT-005`), never here. This is what keeps
        the two modules' corpora disjoint and prevents double-counting a
        single weakness under two `owasp_ref`s (ROADMAP SC#4).
        """
        entries = load_corpus("vector_embedding_weaknesses")
        assert len(entries) > 0
        for entry in entries:
            assert not _DIRECTIVE_CUE_RE.search(entry.prompt), (
                entry.id,
                "contains an assistant-addressed directive -- belongs in "
                "prompt_injection.yaml, not here",
            )

    def test_no_entry_plants_a_pii_shaped_value(self):
        """D-05: the planted marker/content must never be PII-shaped --
        that detection surface belongs to `pii_exfiltration`'s four-tier
        canary/regex/NER/judge pipeline."""
        entries = load_corpus("vector_embedding_weaknesses")
        assert len(entries) > 0
        for entry in entries:
            assert entry.canary_pii_type is None, entry.id
            for pii_type, pattern in _PII_SHAPE_PATTERNS.items():
                assert not pattern.search(entry.prompt), (entry.id, pii_type)

    def test_marker_distinct_from_shared_jailbreak_canary(self):
        """RESEARCH Pitfall 3: a marker hit in a report must attribute to
        exactly one owasp_ref. If `CONTEXT_LEAK_MARKER` collided with (or
        contained/was contained by) `CANARY_TOKEN`, a report reader could
        not tell whether a hit was a jailbreak/persona-override bypass
        (prompt_injection) or a wrong-chunk-surfaced RAG leak
        (vector_embedding_weaknesses).
        """
        assert CONTEXT_LEAK_MARKER != CANARY_TOKEN
        assert CANARY_TOKEN not in CONTEXT_LEAK_MARKER
        assert CONTEXT_LEAK_MARKER not in CANARY_TOKEN

        import inspect

        import llmsec.modules.vector_embedding_weaknesses as module_source

        source_text = inspect.getsource(module_source)
        assert CANARY_TOKEN not in source_text

    def test_detection_layer_closure(self):
        """Every `detection_layer` value this module can emit ("canary",
        "judge") is a member of `EvalResult`'s existing closed `Literal` --
        this phase adds no new detection-layer member."""
        allowed = EvalResult.model_fields["detection_layer"].annotation.__args__
        assert "canary" in allowed
        assert "judge" in allowed


# ============================================================================
# Task 2: honest-limits disclosure (mirrors tests/test_api.py's style)
# ============================================================================


class TestVectorContextSimulatedLimitationNote:
    def test_present_when_vector_embedding_weaknesses_loaded(self):
        import llmsec.api as api_module

        limitations = api_module._scan_limitations(["vector_embedding_weaknesses"], [])
        assert api_module._VECTOR_CONTEXT_SIMULATED_NOTE in limitations

    def test_absent_when_not_loaded(self):
        import llmsec.api as api_module

        limitations = api_module._scan_limitations(["prompt_injection"], [])
        assert api_module._VECTOR_CONTEXT_SIMULATED_NOTE not in limitations

    def test_note_mentions_vector_database(self):
        import llmsec.api as api_module

        assert "vector database" in api_module._VECTOR_CONTEXT_SIMULATED_NOTE


# ============================================================================
# Task 3 (08-03): ROADMAP success-criterion #1/#3/#4 proofs -- MOD-10 complete
# ============================================================================


class TestAttributionAcrossModules:
    """ROADMAP SC#4: a weakness this module surfaces is never double-counted
    against `prompt_injection`'s `indirect_data_as_instruction` family --
    disjoint ids, disjoint technique-family values, no prompt-text overlap,
    and a distinct `owasp_ref`/`id` per module."""

    def test_entry_ids_disjoint_from_prompt_injection(self):
        vector_ids = {e.id for e in load_corpus("vector_embedding_weaknesses")}
        injection_ids = {e.id for e in load_corpus("prompt_injection")}
        assert not (vector_ids & injection_ids)

    def test_technique_family_values_disjoint_from_prompt_injection(self):
        vector_families = {
            e.technique_family.value for e in load_corpus("vector_embedding_weaknesses")
        }
        injection_families = {e.technique_family.value for e in load_corpus("prompt_injection")}
        assert not (vector_families & injection_families)

    def test_no_prompt_substring_overlap_with_indirect_data_as_instruction(self):
        vector_prompts = [e.prompt for e in load_corpus("vector_embedding_weaknesses")]
        indirect_prompts = [
            e.prompt
            for e in load_corpus("prompt_injection")
            if e.technique_family.value == "indirect_data_as_instruction"
        ]
        # Sanity: the family this module must stay disjoint from actually exists.
        assert indirect_prompts
        for vector_prompt in vector_prompts:
            for indirect_prompt in indirect_prompts:
                assert vector_prompt not in indirect_prompt
                assert indirect_prompt not in vector_prompt

    def test_modules_report_different_owasp_ref_and_id(self):
        from llmsec.modules.prompt_injection import PromptInjectionModule

        assert VectorEmbeddingWeaknessesModule.owasp_ref != PromptInjectionModule.owasp_ref
        assert VectorEmbeddingWeaknessesModule.id != PromptInjectionModule.id

    def test_finding_owasp_ref_is_llm08_for_both_families(self):
        """A `Finding` built from either family's `EvalResult` carries
        `owasp_ref == module.owasp_ref` (`api.py`'s real construction
        mechanism, `owasp_ref = module.owasp_ref`) -- `LLM08:2025` for both
        families, never a family-specific value that could be confused with
        another module's category."""
        from llmsec.models import Finding

        module = VectorEmbeddingWeaknessesModule()
        for case_id, detection_layer in [("CTXLEAK-001", "canary"), ("OVERTRUST-001", "judge")]:
            finding = Finding(
                case_id=case_id,
                technique_id=case_id,
                verdict=Verdict.FULL_COMPROMISE,
                severity="high",
                owasp_ref=module.owasp_ref,
                evidence="evidence",
                remediation="remediation",
                transport_mode="single",
                detection_layer=detection_layer,  # type: ignore[arg-type]
            )
            assert finding.owasp_ref == "LLM08:2025"


class TestFullCorpusEndToEnd:
    """ROADMAP SC#1: the module produces findings for a leaking target,
    proven over the FULL corpus (both families) through the real registered
    plugin, with the T-01-18 one-result-per-case guarantee."""

    async def test_every_generated_case_yields_exactly_one_eval_result(self, monkeypatch):
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        modules = PluginRegistry().load_allowed(["vector_embedding_weaknesses"])
        module = modules["vector_embedding_weaknesses"]
        cases = [case async for case in module.generate_cases(None)]
        entries = load_corpus("vector_embedding_weaknesses")
        assert len(cases) == len(entries)

        results = [await module.evaluate(case, _response("a clean, scoped answer")) for case in cases]
        assert len(results) == len(cases)
        assert all(isinstance(r, EvalResult) for r in results)

    async def test_mixed_run_produces_findings_from_both_families(self, monkeypatch):
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(
                verdict=Verdict.FULL_COMPROMISE,
                confidence=0.9,
                reasoning="Flatly restated the planted claim with no source framing.",
            )
        )
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        modules = PluginRegistry().load_allowed(["vector_embedding_weaknesses"])
        module = modules["vector_embedding_weaknesses"]
        cases = [case async for case in module.generate_cases(None)]

        family1_result = next(
            r
            for r in [
                await module.evaluate(
                    case, _response(f"Echoing {CONTEXT_LEAK_MARKER} by mistake.")
                )
                for case in cases
                if case.case_id.startswith("CTXLEAK-")
            ]
        )
        family2_result = next(
            r
            for r in [
                await module.evaluate(case, _response("that's correct, established fact"))
                for case in cases
                if case.case_id.startswith("OVERTRUST-")
            ]
        )

        assert family1_result.verdict == Verdict.FULL_COMPROMISE
        assert family1_result.detection_layer == "canary"
        assert family2_result.verdict == Verdict.FULL_COMPROMISE
        assert family2_result.detection_layer == "judge"


class TestCoverageCompleteness:
    """Every `VectorContextTechniqueVector` member has corpus coverage, and
    the module's actually-emitted `detection_layer` values are exactly the
    two closed-`Literal` members this phase uses (no new member added)."""

    def test_every_vector_context_technique_vector_member_has_corpus_coverage(self):
        entries = load_corpus("vector_embedding_weaknesses")
        got = {e.technique_family.value for e in entries}
        assert got == {m.value for m in VectorContextTechniqueVector}

    async def test_detection_layer_values_emitted_across_full_corpus_are_canary_and_judge(
        self, monkeypatch
    ):
        mock_judge = AsyncMock(
            return_value=JudgeVerdict(verdict=Verdict.BLOCKED, confidence=0.8, reasoning="ok")
        )
        monkeypatch.setattr(
            "llmsec.modules.vector_embedding_weaknesses.judge_context_over_trust", mock_judge
        )

        module = VectorEmbeddingWeaknessesModule()
        cases = [case async for case in module.generate_cases(None)]
        results = [await module.evaluate(case, _response("a clean, scoped answer")) for case in cases]

        emitted = {r.detection_layer for r in results}
        assert emitted == {"canary", "judge"}
