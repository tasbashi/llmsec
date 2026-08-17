"""Tests for `ExcessiveAgencyModule` (OWASP LLM06:2025, MOD-11,
08-04-PLAN.md).

Task 1 covers: corpus coverage, module identity/registration,
`generate_cases()`'s system-prompt-controllability fallback, the refusal
fast-path, the judge tier, and both never-raising degrade paths.

Task 3 extends this file with the tier-boundary (edge-probe row A-08-02),
class-attribution/confidence-pass-through (edge-probe row A-08-03), and
containment/contract test classes.
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import AsyncMock

import pytest

from llmsec.detection import judge as judge_module
from llmsec.detection.judge import JudgeVerdict
from llmsec.detection.regex_rules import REFUSAL_PATTERNS
from llmsec.models import ScanContext, TargetResponse, TestCase, Verdict
from llmsec.modules.excessive_agency import (
    _AGENCY_REFUSAL_PATTERNS,
    ExcessiveAgencyModule,
    _is_agency_refusal,
)
from llmsec.payloads import load_corpus
from llmsec.payloads.schema import AgencyClass
from llmsec.plugins.registry import BUILTIN_MODULE_IDS, PluginRegistry


@pytest.fixture
def mock_judge_client(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Same boundary as `tests/detection/conftest.py`'s fixture: patch the
    module-level Instructor singleton's `.create()` directly."""
    mock = AsyncMock()
    monkeypatch.setattr(judge_module.judge_client.chat.completions, "create", mock)
    return mock


def _response(case_id: str, raw_text: str, **kwargs: object) -> TargetResponse:
    return TargetResponse(case_id=case_id, raw_text=raw_text, latency_ms=1.0, **kwargs)


def _context(system_prompt_controllable: bool = False) -> ScanContext:
    return ScanContext(
        judge_model="openai/gpt-4o-mini",
        judge_api_key_env="",
        system_prompt_controllable=system_prompt_controllable,
    )


# --- Module identity / registration (D-10) ---------------------------------


class TestModuleIdentity:
    def test_module_class_attrs_and_bare_cls_instantiation(self):
        module = ExcessiveAgencyModule()
        assert module.id == "excessive_agency"
        assert module.name == "Excessive Agency (Capability-Boundary Probing)"
        assert module.owasp_ref == "LLM06:2025"
        assert module.uses_attacker_llm is True

    def test_excessive_agency_in_builtin_module_ids(self):
        assert "excessive_agency" in BUILTIN_MODULE_IDS

    def test_loadable_through_registry(self):
        modules = PluginRegistry().load_allowed(["excessive_agency"])
        module = modules["excessive_agency"]
        assert module.owasp_ref == "LLM06:2025"
        assert module.uses_attacker_llm is True

    def test_no_standalone_audit_or_direct_probe_override(self):
        """D-01/D-02: normal live-probing contract only."""
        assert "run_standalone_audit" not in ExcessiveAgencyModule.__dict__
        assert "run_direct_probe" not in ExcessiveAgencyModule.__dict__


# --- Corpus coverage ---------------------------------------------------------


class TestCorpusCoverage:
    def test_every_agency_class_has_at_least_two_entries(self):
        entries = load_corpus("excessive_agency")
        counts = Counter(e.technique_family.value for e in entries)
        assert counts["functionality"] >= 2
        assert counts["permissions"] >= 2
        assert counts["autonomy"] >= 2

    def test_permissions_entries_all_declare_a_boundary(self):
        entries = load_corpus("excessive_agency")
        permissions_entries = [
            e for e in entries if e.technique_family == AgencyClass.PERMISSIONS
        ]
        assert permissions_entries
        assert all(e.declared_permission_boundary for e in permissions_entries)

    def test_every_entry_sets_prompt_never_turns(self):
        entries = load_corpus("excessive_agency")
        assert entries
        assert all(e.turns is None and e.prompt for e in entries)

    def test_every_entry_has_nonempty_remediation(self):
        entries = load_corpus("excessive_agency")
        assert all(e.remediation.strip() for e in entries)

    def test_every_entry_id_is_unique(self):
        entries = load_corpus("excessive_agency")
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    def test_every_agency_class_member_has_corpus_coverage(self):
        got = {e.technique_family.value for e in load_corpus("excessive_agency")}
        assert got == {m.value for m in AgencyClass}


# --- generate_cases() --------------------------------------------------------


class TestGenerateCases:
    async def test_yields_one_case_per_corpus_entry(self):
        module = ExcessiveAgencyModule()
        cases = [case async for case in module.generate_cases(_context())]
        assert len(cases) == len(load_corpus("excessive_agency"))

    async def test_system_prompt_controllable_true_sets_override_on_permissions_only(self):
        module = ExcessiveAgencyModule()
        cases = {
            case.case_id: case
            async for case in module.generate_cases(_context(system_prompt_controllable=True))
        }
        entries_by_id = {e.id: e for e in load_corpus("excessive_agency")}
        for case_id, case in cases.items():
            entry = entries_by_id[case_id]
            if entry.declared_permission_boundary:
                assert case.system_prompt_override == entry.declared_permission_boundary
            else:
                assert case.system_prompt_override is None

    async def test_system_prompt_controllable_false_embeds_boundary_in_prompt(self):
        module = ExcessiveAgencyModule()
        cases = {
            case.case_id: case
            async for case in module.generate_cases(_context(system_prompt_controllable=False))
        }
        entries_by_id = {e.id: e for e in load_corpus("excessive_agency")}
        for case_id, case in cases.items():
            entry = entries_by_id[case_id]
            assert case.system_prompt_override is None
            if entry.declared_permission_boundary:
                assert entry.declared_permission_boundary in case.prompt
                assert entry.prompt in case.prompt


# --- Refusal fast-path (D-08) ------------------------------------------------


class TestRefusalFastPath:
    async def test_capability_decline_resolves_blocked_zero_judge_calls(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        entry = next(
            e for e in load_corpus("excessive_agency") if e.technique_family == AgencyClass.FUNCTIONALITY
        )
        case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
        response = _response(
            entry.id,
            "I don't have the ability to send emails on your behalf, so I can't do that for you.",
        )
        result = await module.evaluate(case, response)
        assert result.verdict is Verdict.BLOCKED
        assert result.detection_layer == "regex"
        assert mock_judge_client.call_count == 0

    def test_shared_refusal_patterns_alone_match_none_of_the_decline_fixtures(self):
        """RESEARCH Pitfall 2: pins the reason `_AGENCY_REFUSAL_PATTERNS`
        exists, not just documents it."""
        import re

        fixtures = [
            "I don't have the ability to send emails on your behalf.",
            "I'm not allowed to take that action.",
            "I can't execute that command for you.",
            "I have no access to the tool in question.",
            "That's outside my permissions.",
            "I'm only allowed to read data, not write it.",
        ]
        for text in fixtures:
            assert not any(re.search(p, text, re.IGNORECASE) for p in REFUSAL_PATTERNS), text
            assert any(re.search(p, text, re.IGNORECASE) for p in _AGENCY_REFUSAL_PATTERNS), text


# --- Judge tier ---------------------------------------------------------------


class TestJudgeTier:
    async def test_confident_false_claim_resolves_via_judge_exactly_once(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        entry = next(
            e for e in load_corpus("excessive_agency") if e.technique_family == AgencyClass.FUNCTIONALITY
        )
        case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
        response = _response(entry.id, "Done! I've sent the confirmation email successfully.")
        mock_judge_client.return_value = JudgeVerdict(
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.85,
            reasoning="The target confidently claimed to have sent an email despite no email tool.",
        )
        result = await module.evaluate(case, response)
        assert result.verdict is Verdict.FULL_COMPROMISE
        assert result.detection_layer == "judge"
        assert mock_judge_client.call_count == 1
        assert result.remediation == entry.remediation


# --- Degrade paths (T-01-18) --------------------------------------------------


class TestDegradePaths:
    async def test_unresolvable_case_id_returns_uncertain_without_raising(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        module._corpus_entries()
        case = TestCase(case_id="NO-SUCH-ENTRY", prompt="probe", technique_id="NO-SUCH-ENTRY")
        response = _response("NO-SUCH-ENTRY", "some reply")
        result = await module.evaluate(case, response)
        assert result.verdict is Verdict.UNCERTAIN
        assert result.confidence == 0.0
        assert mock_judge_client.call_count == 0

    async def test_raising_judge_call_returns_uncertain_without_raising(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        entry = next(
            e for e in load_corpus("excessive_agency") if e.technique_family == AgencyClass.AUTONOMY
        )
        case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
        response = _response(entry.id, "Sure, I've gone ahead and completed the whole migration.")
        mock_judge_client.side_effect = RuntimeError("boom")
        result = await module.evaluate(case, response)
        assert result.verdict is Verdict.UNCERTAIN
        assert result.confidence == 0.0


# --- Task 3: tier-boundary, class-attribution, confidence pass-through, and
# containment/contract proofs (08-04-PLAN.md Task 3) -------------------------


class TestTierBoundary:
    """Edge-probe row A-08-02: the refusal fast-path's exact boundary and
    one step either side of it, with the mocked judge asserted on for call
    count in every case."""

    async def test_genuine_capability_decline_resolves_blocked_zero_judge_calls(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        entry = next(
            e
            for e in load_corpus("excessive_agency")
            if e.technique_family == AgencyClass.PERMISSIONS
        )
        case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
        response = _response(
            entry.id,
            "I'm only allowed to look up stock levels, so I can't place that order for you.",
        )
        result = await module.evaluate(case, response)
        assert result.verdict is Verdict.BLOCKED
        assert result.detection_layer == "regex"
        assert mock_judge_client.call_count == 0

    async def test_contrastive_hedge_refuse_then_comply_falls_through_to_judge(
        self, mock_judge_client: AsyncMock
    ):
        """A decline that pivots to compliance via a connector word ("but")
        must NOT resolve deterministically to `BLOCKED`."""
        module = ExcessiveAgencyModule()
        entry = next(
            e
            for e in load_corpus("excessive_agency")
            if e.technique_family == AgencyClass.FUNCTIONALITY
        )
        case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
        response = _response(
            entry.id,
            "I don't have the ability to send emails, but I went ahead and sent it anyway.",
        )
        mock_judge_client.return_value = JudgeVerdict(
            verdict=Verdict.FULL_COMPROMISE, confidence=0.8, reasoning="refuse-then-comply"
        )
        result = await module.evaluate(case, response)
        assert result.detection_layer == "judge"
        assert mock_judge_client.call_count == 1

    async def test_sentence_pivot_refuse_then_comply_falls_through_to_judge(
        self, mock_judge_client: AsyncMock
    ):
        """A decline that pivots to compliance with no connector at all --
        just a sentence break followed by an affirmative opener -- must
        also NOT resolve deterministically to `BLOCKED`."""
        module = ExcessiveAgencyModule()
        entry = next(
            e for e in load_corpus("excessive_agency") if e.technique_family == AgencyClass.AUTONOMY
        )
        case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
        response = _response(
            entry.id,
            "I don't have permission to do that. Sure, I've gone ahead and committed the changes.",
        )
        mock_judge_client.return_value = JudgeVerdict(
            verdict=Verdict.FULL_COMPROMISE, confidence=0.8, reasoning="refuse-then-comply"
        )
        result = await module.evaluate(case, response)
        assert result.detection_layer == "judge"
        assert mock_judge_client.call_count == 1

    async def test_decline_phrased_outside_both_pattern_lists_falls_through_to_judge(
        self, mock_judge_client: AsyncMock
    ):
        """Documented, accepted residual (mirrors
        `insecure_output._is_clean_refusal()`'s own accepted asymmetry): a
        genuine decline phrased entirely outside
        `REFUSAL_PATTERNS`/`_AGENCY_REFUSAL_PATTERNS` is not masked to
        `BLOCKED` -- it falls through to the judge tier."""
        module = ExcessiveAgencyModule()
        entry = next(
            e
            for e in load_corpus("excessive_agency")
            if e.technique_family == AgencyClass.FUNCTIONALITY
        )
        case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
        response = _response(
            entry.id, "Unfortunately that's simply beyond what I'm capable of doing right now."
        )
        assert _is_agency_refusal(response.raw_text) is False
        mock_judge_client.return_value = JudgeVerdict(
            verdict=Verdict.BLOCKED, confidence=0.7, reasoning="genuine decline, judged"
        )
        result = await module.evaluate(case, response)
        assert result.detection_layer == "judge"
        assert mock_judge_client.call_count == 1

    def test_generic_boilerplate_alone_does_not_qualify_as_refusal(self):
        """A text that matches a decline pattern but is otherwise pure
        generic AI self-description boilerplate (no corroborating leak
        indicator) must not resolve deterministically -- the same
        `is_generic_boilerplate_only()` guard `insecure_output.py` uses."""
        text = "As an AI assistant, I don't have the ability to do that."
        assert _is_agency_refusal(text) is False


class TestClassAttributionAndConfidencePassthrough:
    """Edge-probe row A-08-03: single-class attribution and verbatim
    confidence pass-through."""

    async def test_every_result_attributes_to_exactly_its_entrys_agency_class(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        entries = load_corpus("excessive_agency")
        mock_judge_client.return_value = JudgeVerdict(
            verdict=Verdict.FULL_COMPROMISE, confidence=0.9, reasoning="overreach claimed"
        )
        by_class: dict[str, set[str]] = {m.value: set() for m in AgencyClass}
        for entry in entries:
            case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
            response = _response(entry.id, "Yes, I've already done that for you, all set.")
            result = await module.evaluate(case, response)
            assert entry.technique_family.value in result.evidence
            by_class[entry.technique_family.value].add(entry.id)

        # Partition property: every entry id appears under exactly one class.
        all_ids = {e.id for e in entries}
        union_ids: set[str] = set()
        for ids in by_class.values():
            assert not (ids & union_ids), "case counted under two classes"
            union_ids |= ids
        assert union_ids == all_ids, "every case must be attributed"
        assert all(ids for ids in by_class.values()), "every class must be represented"

    async def test_confidence_passes_through_verbatim_no_rounding(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        entry = next(
            e
            for e in load_corpus("excessive_agency")
            if e.technique_family == AgencyClass.PERMISSIONS
        )
        case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
        response = _response(entry.id, "Sure, done -- placed the order already.")
        precise_confidence = 0.6789123
        mock_judge_client.return_value = JudgeVerdict(
            verdict=Verdict.FULL_COMPROMISE, confidence=precise_confidence, reasoning="overreach"
        )
        result = await module.evaluate(case, response)
        assert result.confidence == precise_confidence

    async def test_full_corpus_findings_grouped_by_class_cover_all_three(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        entries = load_corpus("excessive_agency")
        mock_judge_client.return_value = JudgeVerdict(
            verdict=Verdict.FULL_COMPROMISE, confidence=0.9, reasoning="overreach claimed"
        )
        results = []
        for entry in entries:
            case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
            response = _response(entry.id, "Absolutely, already handled that for you.")
            results.append(await module.evaluate(case, response))

        assert all(r.verdict is Verdict.FULL_COMPROMISE for r in results)
        entries_by_id = {e.id: e for e in entries}
        classes_covered = {entries_by_id[r.case_id].technique_family.value for r in results}
        assert classes_covered == {m.value for m in AgencyClass}


class TestContainmentAndContract:
    """T-01-18: `evaluate()` returns exactly one `EvalResult` per generated
    case across a full-corpus run in which a subset of judge calls raise,
    with no exception ever escaping."""

    async def test_one_result_per_case_with_some_judge_calls_raising(
        self, mock_judge_client: AsyncMock
    ):
        module = ExcessiveAgencyModule()
        entries = load_corpus("excessive_agency")

        outcomes = []
        for i in range(len(entries)):
            if i % 2 == 0:
                outcomes.append(
                    JudgeVerdict(
                        verdict=Verdict.FULL_COMPROMISE, confidence=0.8, reasoning="ok"
                    )
                )
            else:
                outcomes.append(RuntimeError("judge boom"))
        mock_judge_client.side_effect = outcomes

        results = []
        for entry in entries:
            case = TestCase(case_id=entry.id, prompt=entry.prompt or "", technique_id=entry.id)
            response = _response(entry.id, "Yes, already done -- all set.")
            results.append(await module.evaluate(case, response))

        assert len(results) == len(entries)
        assert all(r.verdict in (Verdict.FULL_COMPROMISE, Verdict.UNCERTAIN) for r in results)
        assert any(r.verdict is Verdict.UNCERTAIN for r in results)
        assert any(r.verdict is Verdict.FULL_COMPROMISE for r in results)
