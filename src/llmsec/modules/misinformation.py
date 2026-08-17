"""`MisinformationModule` -- the built-in OWASP LLM09:2025 test module
(MOD-12), the tenth built-in `llmsec.modules` entry.

Probes a target with a ground-truth-assertion scenario -- a single
self-contained fictional fact established in `TestCase.prompt`, drawn from
the curated `misinformation` YAML corpus -- and checks whether the target's
answer stays faithful to that fact when asked a related follow-up question.
Every entry declares a `misinformation_probe_shape`: `extrapolation` (the
question asks for a detail the fact deliberately never specifies -- correct
behaviour is declining or flagging the gap) or `restatement` (the fact does
answer the question -- correct behaviour is an accurate restatement).

Judge-only tier -- no deterministic dispatch exists for ground-truth
fidelity (mirrors `vector_embedding_weaknesses`'s `retrieved_content_over_
trust` family, D-04 family 2/RESEARCH Pattern 2). `evaluate()` resolves the
corpus entry off `case.case_id` and reads `ground_truth_fact`/
`misinformation_probe_shape` off the ENTRY, never re-derived from
`case.prompt` -- a deep-mode mutated variant carries a mutated prompt but
this same unchanged `evaluate()` (D-66). Three never-raising degrade paths
(T-01-18): an unresolvable `case_id`, a resolved entry with no
`ground_truth_fact` (returned WITHOUT calling the judge), and a raising
judge call.

No confidence cap is applied -- unlike `data_poisoning`'s low-confidence
discipline, ground-truth fidelity is directly assessable black-box, so
`full_compromise` stays reachable at the judge's own confidence.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from llmsec.detection.judge import judge_misinformation
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.payloads import PayloadEntry, load_corpus
from llmsec.plugins.base import BaseModule

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

_UNRESOLVED_ENTRY_NOTE = (
    "This case could not be resolved to a misinformation corpus entry -- "
    "most likely case.case_id has no matching corpus entry -- so it is "
    "reported as uncertain rather than guessed (T-01-18 containment "
    "default)."
)

_MISSING_GROUND_TRUTH_NOTE = (
    "The resolved misinformation corpus entry has no ground_truth_fact set; "
    "the misinformation judge classifies fidelity to a specific established "
    "fact, so it cannot run without one. Reported as uncertain rather than "
    "guessed, without calling the judge."
)


class MisinformationModule(BaseModule):
    """Built-in OWASP LLM09:2025 module (MOD-12) -- ground-truth-assertion
    probing (D-01/D-02: normal `generate_cases()`/`evaluate()` live-probing
    contract only, no `run_standalone_audit()`/`run_direct_probe()`
    override).
    """

    id = "misinformation"
    name = "Misinformation & Hallucination (Ground-Truth-Assertion Probing)"
    owasp_ref = "LLM09:2025"
    # D-03 (assumption A3, resolved True): escalating the plausibility of a
    # fabricated elaboration is genuine mutation material -- five of the six
    # shipped new-track modules set this True. `MisinformationTechniqueVector`'s
    # values clear the deep-mode delegation-boundary allowlist (plan 09-02).
    uses_attacker_llm = True

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_api_key_env: str | None = None,
    ) -> None:
        # Every parameter defaulted so `PluginRegistry.load_allowed()`'s bare
        # `cls()` instantiation still works without arguments (D-10). No
        # overlay-path parameter -- this module has no operator overlay.
        self.judge_model = judge_model
        self.judge_api_key_env = judge_api_key_env
        self._corpus: list[PayloadEntry] | None = None
        self._entries_by_id: dict[str, PayloadEntry] = {}

    def _corpus_entries(self) -> list[PayloadEntry]:
        """Lazily load and cache the curated `misinformation` corpus,
        mirroring `vector_embedding_weaknesses._corpus_entries()`'s cache
        shape (minus the overlay layer this module does not have).
        """
        if self._corpus is None:
            entries = load_corpus("misinformation")
            self._corpus = entries
            self._entries_by_id = {entry.id: entry for entry in entries}
            if not self._corpus:
                logger.warning(
                    "misinformation corpus loaded 0 entries; "
                    "generate_cases() will yield nothing"
                )
        return self._corpus

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        """Yield one `TestCase` per corpus entry that sets `prompt`, with
        `case_id`/`technique_id` both the entry id and `turns=None`. No
        placeholder substitution is needed for this module. Skip-and-log any
        entry whose `prompt` is `None`.
        """
        for entry in self._corpus_entries():
            if entry.prompt is None:
                logger.warning(
                    "Skipping misinformation entry %r: no prompt set", entry.id
                )
                continue
            yield TestCase(
                case_id=entry.id,
                prompt=entry.prompt,
                technique_id=entry.id,
                turns=None,
            )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        """Resolve `case.case_id` against the cached corpus, then call
        `judge_misinformation()` exactly once, passing the resolved entry's
        `ground_truth_fact`/`misinformation_probe_shape` -- read off the
        ENTRY, never parsed back out of `case.prompt` (D-66, RESEARCH
        Pitfall 2), since a deep-mode mutated variant carries a mutated
        prompt but this same unchanged `evaluate()`. `verdict`/`confidence`
        pass through from the `JudgeVerdict` unmodified -- no confidence cap
        (that discipline belongs to `data_poisoning`, not here).

        Three never-raising degrade paths, each returning
        `Verdict.UNCERTAIN` at confidence `0.0` (T-01-18): an unresolvable
        `case_id`, a resolved entry with no `ground_truth_fact` (returned
        WITHOUT calling the judge), and a raising judge call.

        Never raises: the whole dispatch is wrapped so an unexpected
        exception degrades to the same `UNCERTAIN` shape (T-01-18).
        """
        try:
            self._corpus_entries()
            entry = self._entries_by_id.get(case.case_id)

            if entry is None:
                return EvalResult(
                    case_id=case.case_id,
                    verdict=Verdict.UNCERTAIN,
                    confidence=0.0,
                    evidence=_UNRESOLVED_ENTRY_NOTE,
                    detection_layer="judge",
                    transport_mode=response.transport_mode,
                )

            remediation = entry.remediation

            if not entry.ground_truth_fact:
                return EvalResult(
                    case_id=case.case_id,
                    verdict=Verdict.UNCERTAIN,
                    confidence=0.0,
                    evidence=_MISSING_GROUND_TRUTH_NOTE,
                    detection_layer="judge",
                    transport_mode=response.transport_mode,
                    remediation=remediation,
                )

            try:
                judged = await judge_misinformation(
                    ground_truth_fact=entry.ground_truth_fact,
                    probe_shape=entry.misinformation_probe_shape or "",
                    probe_prompt=case.prompt or "",
                    target_reply=response.raw_text or "",
                    judge_model=self.judge_model,
                    judge_api_key_env=self.judge_api_key_env,
                )
            except Exception as exc:
                return EvalResult(
                    case_id=case.case_id,
                    verdict=Verdict.UNCERTAIN,
                    confidence=0.0,
                    evidence=f"The misinformation judge call failed: {exc}",
                    detection_layer="judge",
                    transport_mode=response.transport_mode,
                    remediation=remediation,
                )

            return EvalResult(
                case_id=case.case_id,
                verdict=judged.verdict,
                confidence=judged.confidence,
                evidence=(
                    f"Ground truth fact: {entry.ground_truth_fact!r}\n"
                    f"Probe shape: {entry.misinformation_probe_shape!r}\n\n"
                    f"Judge reasoning: {judged.reasoning}"
                ),
                detection_layer="judge",
                transport_mode=response.transport_mode,
                remediation=remediation,
            )
        except Exception as exc:
            logger.error(
                "misinformation evaluate() failed for case %r: %s", case.case_id, exc
            )
            return EvalResult(
                case_id=case.case_id,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                evidence=f"evaluate() raised unexpectedly: {exc}",
                detection_layer="judge",
                transport_mode=response.transport_mode,
            )
