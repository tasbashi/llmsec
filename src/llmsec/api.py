"""`run_scan()` — the library entrypoint (CLI-01's engine).

Wires config, authorization, plugin loading, orchestration, scoring, and
persistence into one coroutine: `import llmsec; await llmsec.run_scan(config)`
works identically to the eventual `llmsec scan` command (plan 09), which
merely wraps this same coroutine at its own top-level event-loop entrypoint.

`confirm_authorization()` (D-01/D-02) is called here — in `run_scan()`
itself, not only from `cli.py` — so a library caller who imports `llmsec`
directly gets the identical authorization guarantee a CLI user gets
(T-01-16, Architectural Responsibility Map).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llmsec.adapters.base import TargetAdapter
from llmsec.adapters.http_app import HttpAppAdapter
from llmsec.adapters.llm_api import LLMApiAdapter
from llmsec.attacker.runner import CampaignResult, run_attacker_campaign
from llmsec.attacker.state import VariantRecord
from llmsec.attacker.summary import compute_deep_summary
from llmsec.auth_gate import confirm_authorization
from llmsec.config import ScanConfig
from llmsec.detection.canary import CANARY_LIMITATION_NOTE
from llmsec.detection.pii_ner import ner_available
from llmsec.models import DeepModeSummary, EvalResult, Finding, ScanContext, ScanReport, Verdict
from llmsec.modules.supply_chain import (
    AUDIT_CASE_ID_EXTRA_MISSING,
    AUDIT_CASE_ID_MANIFEST_MISSING,
    AUDIT_CASE_ID_OSV_UNREACHABLE,
    SLOPSQUATTING_CASE_ID_PREFIX,
    index_snapshot_limitation_note,
)
# Phase 7 (07-03): imported (never re-typed) so `_CONSUMPTION_FLOOD_CAP_NOTE`
# below can never drift from the module's own cap constant. `api.py`
# already imports from `llmsec.modules.supply_chain` at top level with no
# cycle, and `unbounded_consumption.py` has no back-import of `api` --
# a top-level import here is safe for the same reason.
from llmsec.modules.unbounded_consumption import _DEFAULT_FLOOD_PROBE_CAP
from llmsec.orchestrator import ScanOrchestrator
from llmsec.plugins.base import BaseModule
from llmsec.plugins.registry import BUILTIN_MODULE_IDS, PluginRegistry
from llmsec.reporting.json_reporter import JsonReporter
from llmsec.scoring.engine import (
    Severity,
    redact_all_protecting_literals,
    redact_credential_match,
    redact_pii_match,
    score,
)

logger = logging.getLogger(__name__)

# D-24: the indirect-injection honesty caveat attached to the report's
# limitations block whenever `prompt_injection` is among the loaded
# modules. Kept here (rather than imported from the module) since it is a
# report-level statement about what THIS RUN tested, not a per-evidence
# note like `CANARY_LIMITATION_NOTE`.
_INDIRECT_INJECTION_LIMITATION_NOTE = (
    "Indirect-injection cases in this run were simulated by embedding "
    "poisoned content directly in the prompt. This tests whether the "
    "target honors instructions found in content-context — it does NOT "
    "test whether the operator's real retrieval pipeline can be poisoned."
)

# D-15: the report-level honesty caveat for a run where at least one
# multi-turn sequence was flattened into a single concatenated request
# because the configured target cannot hold conversation state.
_DEGRADED_MULTI_TURN_LIMITATION_NOTE = (
    "One or more multi-turn sequences in this run were flattened into a "
    "single concatenated request because the configured target cannot "
    "hold conversation state, so those results do not demonstrate "
    "resistance to genuine multi-turn escalation. Configure "
    "target.session_id_path (plus session_id_header or a {session_id} "
    "body token) to test crescendo sequences properly."
)

# CR-02/D-28: the report-level honesty caveat for a run where the
# `pii_exfiltration` module was loaded but the optional `[pii-ner]` extra
# is not installed. Without this, a well-behaved target that gets BLOCKED
# on every case never produces a `Finding` (severity NONE is filtered out
# in `run_scan()` below) and `report.case_log` — the only place the
# per-case `_NER_SKIP_CAVEAT` evidence text survives — is never rendered
# by the Markdown report template, so an entire detection tier can
# silently never execute for a scan that otherwise reads as a clean,
# fully-covered "No findings." result. Surfacing it unconditionally in
# `ScanReport.limitations` (rendered by both the Markdown and JSON
# reporters) closes that gap regardless of any individual case's score.
_NER_NOT_INSTALLED_LIMITATION_NOTE = (
    "The optional NER (Named Entity Recognition) detection tier for "
    "pii_exfiltration did not run because the `[pii-ner]` extra is not "
    "installed. Findings from this run reflect only the canary, regex/Luhn, "
    "and judge tiers; install `[pii-ner]` for unstructured-PII coverage."
)

# Phase 6 (06-01, D-06): honest-degradation notes for supply_chain's
# standalone CVE/SBOM audit, in the same voice as
# `_NER_NOT_INSTALLED_LIMITATION_NOTE` above. Each pairs with a stable
# `case_id` sentinel `supply_chain.run_standalone_audit()` yields --
# `_scan_limitations()` below keys off those exact case_ids.
_SUPPLY_CHAIN_MANIFEST_MISSING_NOTE = (
    "The CVE/SBOM dependency audit for supply_chain did not run because no "
    "`supply_chain_manifest_path` was configured (or the configured path "
    "could not be read). This scan evaluated no dependency manifest; "
    "configure `supply_chain_manifest_path` in llmsec.config.yaml to enable it."
)
_SUPPLY_CHAIN_EXTRA_NOT_INSTALLED_NOTE = (
    "The CVE/SBOM dependency audit for supply_chain did not run because the "
    "optional `[supply-chain]` extra is not installed. Install it with "
    '`pip install ".[supply-chain]"` for CVE/SBOM coverage.'
)
_SUPPLY_CHAIN_OSV_UNREACHABLE_NOTE = (
    "The CVE/SBOM dependency audit for supply_chain could not reach the "
    "OSV.dev vulnerability database. Findings from this run do not reflect "
    "any dependency-vulnerability check; rerun once connectivity is restored."
)
# Phase 6 (06-01, D-11): the structural low-confidence/heuristic caveat for
# data_poisoning (MOD-07). Placed here now so `_scan_limitations()` can
# reference it from its first commit, even though data_poisoning itself
# ships in a later plan this phase.
_POISONING_HEURISTIC_ONLY_NOTE = (
    "data_poisoning cannot detect an untriggered backdoor at all -- it can "
    "only observe a behavioral shift when a specific trigger phrase is "
    "actually probed. Its findings are heuristic behavioral indicators, "
    "capped below full-compromise confidence; a clean result from this "
    "module is NOT evidence the target model is unpoisoned."
)
# Phase 7 (07-03, D-08/T-07-12): the honest-degradation caveat for
# unbounded_consumption's flood-class dispatch path. The cap value is
# imported from the module's own constant (above), never hand-copied, so
# the two can never drift. A clean consumption report must never read as
# an exhaustively-probed one -- only a capped subset of flood-class probes
# was ever dispatched (D-08's scan-itself-never-becomes-the-attack
# constraint), and the token/latency thresholds applied are documented
# starting heuristics, not calibrated universal values.
_CONSUMPTION_FLOOD_CAP_NOTE = (
    f"unbounded_consumption dispatched at most {_DEFAULT_FLOOD_PROBE_CAP} "
    "flood-class consumption probes this run, capped so the scan itself "
    "does not become a denial-of-wallet event against the scanned target "
    "(D-08). The token and latency thresholds applied are documented "
    "starting heuristics, not calibrated universal values -- tune them via "
    "consumption_token_threshold/consumption_latency_threshold_ms in "
    "llmsec.config.yaml."
)

# Phase 8 (08-01, D-04/D-07 simulated-context framing): the honest-degradation
# caveat for vector_embedding_weaknesses (MOD-10). No real vector database,
# embedding index, or retrieval pipeline is ever exercised -- every
# "retrieved chunk" this module probes with is simulated inline in the
# prompt. Mirrors _CONSUMPTION_FLOOD_CAP_NOTE's declarative voice and
# module_ids-keyed shape exactly (T-02-33 fixed-order guarantee).
_VECTOR_CONTEXT_SIMULATED_NOTE = (
    "vector_embedding_weaknesses did not exercise a real vector database, "
    "embedding index, or retrieval pipeline -- every retrieved chunk in "
    "this run's probes was simulated inline in the prompt. Index-level "
    "exposures (poisoned or adversarially-similar embeddings, cross-tenant "
    "index access, write access to the vector store) were therefore not "
    "tested; a clean result from this module is not a clearance of the "
    "retrieval stack."
)

# Phase 8 (08-04, D-07/T-08-18): the honest-degradation caveat for
# excessive_agency (MOD-11). No real tool execution and no real permission
# enforcement is exercised anywhere -- every finding describes what the
# target SAID it could do, had done, or would do, never what it actually
# did. Mirrors _VECTOR_CONTEXT_SIMULATED_NOTE's declarative voice and
# module_ids-keyed shape exactly (T-02-33 fixed-order guarantee). Also
# discloses the generate_cases() system-prompt-controllability fallback
# (Task 1): when the configured target does not let the framework set the
# system prompt, a permissions-class probe's declared boundary is stated
# in the conversation instead, and a target may weight that differently
# from a boundary stated in its own system prompt.
_EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE = (
    "excessive_agency did not exercise any real tool execution or real "
    "permission enforcement -- every finding describes what the target "
    "SAID it could do, had done, or would do, never what it actually did. "
    "A clean result from this module is not evidence that access control "
    "is enforced anywhere in the integration. Where the configured target "
    "does not let the framework set the system prompt, a permissions-class "
    "probe's declared boundary was stated in the conversation instead of "
    "in the system prompt, and a target may weight an in-conversation "
    "boundary differently from one stated in its system prompt."
)

# Phase 9 (09-02, D-02/honest-degradation): the honest-degradation caveat for
# misinformation (MOD-12). Every ground truth this module probes with is a
# self-contained fiction established inside each probe prompt, deliberately
# so that findings never depend on the judge's own knowledge cutoff and the
# corpus never seeds real-world misinformation. Mirrors
# _EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE's declarative voice and
# module_ids-keyed shape exactly (T-02-33 fixed-order guarantee).
_MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE = (
    "misinformation tested the target's fidelity to a self-contained "
    "fictional ground truth established inside each probe prompt, not the "
    "target's real-world factual accuracy. No real, identifiable person, "
    "company, product, organisation, or event was ever asserted about, so "
    "no finding from this module is evidence about any real entity. A "
    "clean result from this module is not a clearance of the target's "
    "factual reliability on real-world questions."
)

# D-93/honest-degradation: the report-level caveat for a `--deep` run whose
# attacker-team campaign raised before completing. The static (`--quick`
# path) results and report are still produced/persisted unaffected -- this
# note discloses only that the deep-mode layer itself did not complete, the
# same honest-degradation instinct the NER-not-installed note above
# establishes.
_DEEP_MODE_FAILED_LIMITATION_NOTE = (
    "The --deep attacker-team campaign failed to complete and was skipped for "
    "this run. Findings reflect only the static (--quick-equivalent) payload "
    "batch; re-run to retry deep mode."
)

# Severity-descending rank used for the SCORE-01 deterministic ordering
# guarantee: lower rank sorts first. NONE-severity results never reach the
# findings list (see `run_scan()` below) but the entry is kept for
# completeness/exhaustiveness of the `Severity` enum.
_SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.NONE: 4,
}

# Verdict-keyed remediation guidance attached to every emitted `Finding`.
# BLOCKED never reaches a `Finding` (severity NONE is filtered out below);
# kept here for completeness rather than special-cased away.
_REMEDIATION_BY_VERDICT: dict[Verdict, str] = {
    Verdict.FULL_COMPROMISE: (
        "Strengthen system-prompt confidentiality: add explicit non-disclosure "
        "instructions and validate/filter model output before returning it to the user."
    ),
    Verdict.PARTIAL_LEAK: (
        "Tighten refusal instructions for prompt-extraction probes and add output "
        "filtering to catch partial instruction disclosure."
    ),
    Verdict.UNCERTAIN: "Manually review this case — the automated verdict was inconclusive.",
    Verdict.BLOCKED: "No action needed.",
}


def _build_adapter(config: ScanConfig) -> TargetAdapter:
    """Construct the `TargetAdapter` matching `config.target.type` (D-07).

    Referenced module-level so tests can monkeypatch `llmsec.api.LLMApiAdapter`
    / `llmsec.api.HttpAppAdapter` directly.
    """
    target = config.target
    if target.type == "raw_llm":
        if target.model is None or target.api_key_env is None:
            raise ValueError(
                "target.type == 'raw_llm' requires both 'model' and 'api_key_env' in config"
            )
        return LLMApiAdapter(model=target.model, api_key_env=target.api_key_env)
    if target.type == "http_app":
        if target.method is None or target.url is None or target.body_template is None:
            raise ValueError(
                "target.type == 'http_app' requires 'method', 'url', and 'body_template' in config"
            )
        return HttpAppAdapter(
            method=target.method,
            url=target.url,
            headers=target.headers,
            body_template=target.body_template,
            response_path=target.response_path or "",
            session_id_path=target.session_id_path,
            session_id_header=target.session_id_header,
        )
    raise ValueError(f"Unknown target.type: {target.type!r}")  # pragma: no cover — Literal-guarded


def _scan_limitations(
    module_ids: list[str],
    case_log: list[EvalResult],
    *,
    deep_mode_failed: bool = False,
    deep_truncation_note: str | None = None,
) -> list[str]:
    """Assemble this run's honest-labeling statements, in a fixed order.

    Order is fixed and condition-driven (never derived from findings, which
    can be empty or vary run-to-run) so two runs over the same target/module
    set produce byte-identical limitations text (T-02-33):

    1. Canary caveat, when `prompt_injection` is loaded — reuses
       `CANARY_LIMITATION_NOTE` verbatim so this text can never drift from
       the mechanism it describes.
    2. Indirect-injection simulation caveat, when `prompt_injection` is
       loaded.
    3. Degraded-multi-turn caveat, when ANY case-log entry ran over
       `multi_turn_concatenated` transport — computed from the case log
       rather than the findings list, since a degraded run whose results
       were all capped to `uncertain` may produce few or no findings while
       still needing this caveat (D-15).
    4. NER-not-installed caveat (CR-02/D-28), when `pii_exfiltration` is
       loaded and `ner_available()` is False — computed independently of
       both the findings list and the case log, since a well-behaved
       target can produce zero findings and a case log whose per-case
       NER-skip caveat text is never rendered by the Markdown report.
    5. Deep-mode budget-truncation caveat (D-83), when `deep_truncation_note`
       is given — the caller computes this by calling
       `attacker.budget.truncation_disclosure()` against the campaign's
       final `BudgetLedger`, independently of both the findings list and
       the case log, for the same reason as #4: a truncated run can still
       have produced zero findings and must not read as an untruncated
       clean scan.
    6. Deep-mode-failed caveat, when `deep_mode_failed` is True — the
       `run_scan()` deep-mode branch's own honest-degradation disclosure
       (D-93), computed independently of both the findings list and the
       case log for the same reason as #4.
    7. Supply-chain manifest-missing caveat (D-06), when any case-log
       entry's `case_id` equals `AUDIT_CASE_ID_MANIFEST_MISSING`.
    8. Supply-chain extra-not-installed caveat (D-06), when any case-log
       entry's `case_id` equals `AUDIT_CASE_ID_EXTRA_MISSING`.
    9. Supply-chain OSV-unreachable caveat (D-06), when any case-log
       entry's `case_id` equals `AUDIT_CASE_ID_OSV_UNREACHABLE`.
    10. Slopsquatting index-staleness caveat, when
        `index_snapshot_limitation_note()` is not `None` AND at least one
        case-log entry's `case_id` starts with `SLOPSQUATTING_CASE_ID_PREFIX`.
    11. Poisoning heuristic-only caveat (D-11), when `"data_poisoning"` is
        in `module_ids`.
    12. Consumption flood-probe-cap/heuristic-thresholds caveat (D-08/
        T-07-12), when `"unbounded_consumption"` is in `module_ids` --
        mirrors condition 11's `module_ids`-keyed shape exactly.
    13. Vector-context simulated-retrieval caveat (`_VECTOR_CONTEXT_SIMULATED_NOTE`,
        D-04/D-07, ROADMAP SC#3), when `"vector_embedding_weaknesses"` is in
        `module_ids` -- mirrors condition 11's/12's `module_ids`-keyed shape
        exactly.
    14. Excessive-agency no-real-enforcement caveat
        (`_EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE`, D-07, ROADMAP SC#3), when
        `"excessive_agency"` is in `module_ids` -- mirrors condition 11's/
        12's/13's `module_ids`-keyed shape exactly.
    15. Misinformation fictional-ground-truth caveat
        (`_MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE`, D-02, RESEARCH Open
        Question #2), when `"misinformation"` is in `module_ids` -- mirrors
        condition 11's/12's/13's/14's `module_ids`-keyed shape exactly.

    Every parameter here is a value the caller has already computed and
    passes explicitly — this function reads no module-level state.

    De-duplicated while preserving order (defensive; each condition can
    currently only fire once).
    """
    limitations: list[str] = []

    if "prompt_injection" in module_ids:
        limitations.append(CANARY_LIMITATION_NOTE)
        limitations.append(_INDIRECT_INJECTION_LIMITATION_NOTE)

    if any(result.transport_mode == "multi_turn_concatenated" for result in case_log):
        limitations.append(_DEGRADED_MULTI_TURN_LIMITATION_NOTE)

    if "pii_exfiltration" in module_ids and not ner_available():
        limitations.append(_NER_NOT_INSTALLED_LIMITATION_NOTE)

    if deep_truncation_note is not None:
        limitations.append(deep_truncation_note)

    if deep_mode_failed:
        limitations.append(_DEEP_MODE_FAILED_LIMITATION_NOTE)

    if any(result.case_id == AUDIT_CASE_ID_MANIFEST_MISSING for result in case_log):
        limitations.append(_SUPPLY_CHAIN_MANIFEST_MISSING_NOTE)

    if any(result.case_id == AUDIT_CASE_ID_EXTRA_MISSING for result in case_log):
        limitations.append(_SUPPLY_CHAIN_EXTRA_NOT_INSTALLED_NOTE)

    if any(result.case_id == AUDIT_CASE_ID_OSV_UNREACHABLE for result in case_log):
        limitations.append(_SUPPLY_CHAIN_OSV_UNREACHABLE_NOTE)

    snapshot_note = index_snapshot_limitation_note()
    if snapshot_note is not None and any(
        result.case_id.startswith(SLOPSQUATTING_CASE_ID_PREFIX) for result in case_log
    ):
        limitations.append(snapshot_note)

    if "data_poisoning" in module_ids:
        limitations.append(_POISONING_HEURISTIC_ONLY_NOTE)

    if "unbounded_consumption" in module_ids:
        limitations.append(_CONSUMPTION_FLOOD_CAP_NOTE)

    if "vector_embedding_weaknesses" in module_ids:
        limitations.append(_VECTOR_CONTEXT_SIMULATED_NOTE)

    if "excessive_agency" in module_ids:
        limitations.append(_EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE)

    if "misinformation" in module_ids:
        limitations.append(_MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE)

    # De-duplicate while preserving first-seen order.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in limitations:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


async def _run_standalone_audits(
    modules: dict[str, BaseModule], context: ScanContext
) -> list[tuple[str, EvalResult]]:
    """Gather every loaded module's `run_standalone_audit()` results (Phase
    6, 06-01, MOD-06's standalone-audit path).

    `asyncio.gather()`-ed alongside (not inside) `ScanOrchestrator.run()` in
    `run_scan()` below, since the two result sources are independent — no
    module's standalone-audit output depends on any other module's
    request/response results, or vice versa.

    Per-module catch-log-continue containment, identical in shape to
    `ScanOrchestrator.run()`'s own `generate_cases()` failure handling
    (T-01-18): one module's `run_standalone_audit()` raising must never
    cancel a sibling module's audit, and must never propagate out of this
    helper and cancel the orchestrator's own `asyncio.gather` running
    concurrently.
    """
    results: list[tuple[str, EvalResult]] = []
    for module_id, module in modules.items():
        try:
            async for eval_result in module.run_standalone_audit(context):
                results.append((module_id, eval_result))
        except Exception as exc:
            logger.error(
                "run_standalone_audit() failed for module %r; skipping remaining "
                "audit results from this module: %s", module_id, exc,
            )
            continue
    return results


async def _run_direct_probes(
    modules: dict[str, BaseModule], context: ScanContext, adapter: TargetAdapter
) -> list[tuple[str, EvalResult]]:
    """Gather every loaded module's `run_direct_probe()` results (Phase 7,
    07-01, MOD-09's retry-exempt dispatch path).

    `asyncio.gather()`-ed alongside `ScanOrchestrator.run()` and
    `_run_standalone_audits()` in `run_scan()` below -- all three result
    sources are independent.

    Same per-module catch-log-continue containment as
    `_run_standalone_audits()` (T-01-18): one module's `run_direct_probe()`
    raising must never cancel a sibling module's probes or the
    orchestrator's own concurrent gather.
    """
    results: list[tuple[str, EvalResult]] = []
    for module_id, module in modules.items():
        try:
            async for eval_result in module.run_direct_probe(context, adapter):
                results.append((module_id, eval_result))
        except Exception as exc:
            logger.error(
                "run_direct_probe() failed for module %r; skipping remaining "
                "probes from this module: %s", module_id, exc,
            )
            continue
    return results


def _redact_protecting_canary_values(evidence_text: str, canary_values: tuple[str, ...]) -> str:
    """WR-05/CR-02: run both redaction primitives over `evidence_text`
    while protecting the known canary-PII literal(s) so ONLY they survive
    redaction unmasked — D-32's actual scope ("canary-PII values are...
    shown VERBATIM") is per-literal, not per-finding.

    Before WR-05, a canary-tier `Finding` skipped redaction entirely for
    its WHOLE evidence string (since the canary tier short-circuits
    `_classify_pii_tier()` before the regex/Luhn tier ever runs on that
    turn's text). If a target's response happened to echo the planted
    canary value AND independently leak a real secret in the same
    excerpt, that real secret was shipped completely raw into the report.

    CR-02: WR-05's first implementation swapped each canary literal for a
    `\\x00`-delimited sentinel *before* running the redaction passes, then
    restored it afterward. That broke whenever a real secret's own match
    span *included* the canary literal's character range (e.g. a
    JWT-shaped `header.CANARY.footer` string, where the canary fills the
    middle dot-delimited segment): the sentinel bytes broke the
    surrounding pattern's structure, so nothing in that whole span got
    redacted — the real secret fragments on either side of the literal
    shipped completely unmasked. `redact_all_protecting_literals()` fixes
    this by matching against the ORIGINAL, unmodified text (so a pattern
    spanning the literal's position still matches normally) and only
    masking the non-literal sub-run(s) of any match that overlaps a
    protected literal — never swapping the literal's characters out at
    all, so it is provably never altered by any pass, and everything
    genuinely outside it is still fully subject to redaction.
    """
    return redact_all_protecting_literals(evidence_text, canary_values)


def _target_summary(config: ScanConfig) -> str:
    target = config.target
    if target.type == "raw_llm":
        return f"raw_llm:{target.model}"
    return f"http_app:{target.method} {target.url}"


async def run_scan(config: ScanConfig, bypass_flag: bool = False) -> ScanReport:
    """Run a full scan and return the resulting `ScanReport`.

    Pipeline: authorization gate -> adapter construction -> allowlisted
    plugin loading -> bounded-concurrency orchestration -> scoring +
    credential redaction -> deterministically-sorted, persisted report.

    `confirm_authorization()` runs FIRST, before any adapter is
    constructed or any request is sent (T-01-16). This coroutine never
    starts its own event loop — a library caller drives that themselves.
    """
    confirm_authorization(bypass_flag)

    started_at = datetime.now(timezone.utc).isoformat()
    # Generated here (rather than at ScanReport construction, as before) so
    # the deep-mode branch further below can pass this SAME id into the
    # attacker-team campaign call -- one scan, one id, threaded through
    # both the static and deep-mode layers.
    scan_id = uuid.uuid4().hex
    adapter = _build_adapter(config)
    try:
        # Thread operator config (CORE-02/MOD-02/CORE-04) to every
        # allowlisted module id via `load_allowed()`'s config-hook — the
        # values are only ever applied AFTER a module clears the allowlist
        # gate (D-10), never a second load path. Built per-call so
        # concurrent run_scan() invocations never share a module_config
        # dict or module instance.
        effective_module_ids = config.enabled_modules or list(BUILTIN_MODULE_IDS)
        module_config = {
            module_id: {
                "known_system_prompt": config.known_system_prompt,
                "judge_model": config.judge_model,
                # WR-03: previously threaded onto ScanContext but never
                # consumed by either built-in module's judge call — now
                # forwarded so a configured judge_api_key_env is actually
                # used to resolve the judge's API key.
                "judge_api_key_env": config.judge_api_key_env,
                # Phase 6 (06-01): both new keys are threaded into the
                # shared dict for every module id -- `load_allowed()`'s
                # `accepted_params` filter drops whichever key a given
                # module's `__init__` doesn't declare, so this stays safe
                # even though only one module each will actually accept it.
                "supply_chain_manifest_path": config.supply_chain_manifest_path,
                "poisoning_trigger_overlay_path": config.poisoning_trigger_overlay_path,
                # Phase 7 (07-01, D-05): same threading discipline as the two
                # keys above -- `accepted_params` drops these for every
                # module except `unbounded_consumption`.
                "consumption_token_threshold": config.consumption_token_threshold,
                "consumption_latency_threshold_ms": config.consumption_latency_threshold_ms,
            }
            for module_id in effective_module_ids
        }
        modules = PluginRegistry().load_allowed(config.enabled_modules, module_config=module_config)
        context = ScanContext(
            known_system_prompt=config.known_system_prompt,
            judge_model=config.judge_model,
            judge_api_key_env=config.judge_api_key_env or "",
            # D-17: derived from the already-constructed adapter's REAL
            # capability flags, never guessed from config alone — a
            # raw-LLM target gets the strong system-prompt canary
            # planting, an HTTP-app target with no session config gets
            # the honestly-weaker turn-based fallback.
            system_prompt_controllable=adapter.supports_system_prompt_override,
            supports_multi_turn=adapter.supports_multi_turn,
        )
        orchestrator = ScanOrchestrator(adapter, modules, max_concurrency=config.max_concurrency)
        # Phase 6 (06-01): the standalone-audit path (`_run_standalone_audits()`)
        # is gathered ALONGSIDE (not inside) `orchestrator.run()` -- the two
        # result sources are independent, so this concurrent gather changes
        # nothing about `orchestrator.run()`'s own behavior or timing.
        # Phase 7 (07-01): the direct-probe path (`_run_direct_probes()`) is
        # gathered as a third, equally independent member.
        # `orchestrator.py` itself is byte-for-byte unchanged (D-73).
        results, audit_results, direct_probe_results = await asyncio.gather(
            orchestrator.run(context),
            _run_standalone_audits(modules, context),
            _run_direct_probes(modules, context, adapter),
        )

        # D-93: the deep-mode attacker team is a NEW ADDITIVE LAYER that
        # runs strictly AFTER ScanOrchestrator.run() returns -- never
        # interleaved into the same asyncio.gather -- so `--quick` (the
        # branch above) is provably unaffected regardless of what happens
        # here. `lineage` stays empty (every Finding's D-90 fields default
        # to None below) unless the branch below populates it.
        lineage: dict[str, VariantRecord] = {}
        deep_mode_failed = False
        deep_truncation_note: str | None = None
        campaign_result: CampaignResult | None = None
        # D-91/Task 2: the ORIGINAL (`--quick`-equivalent) results, captured
        # BEFORE any deep-mode eval_results are appended below --
        # `compute_deep_summary()` needs this exact pre-merge list to look
        # up a bypassed case's own parent verdict/evidence (a `blocked`
        # parent never produces a `Finding`, so `findings` alone cannot
        # answer that).
        static_results = results
        if config.attacker is not None and config.attacker.enabled:
            try:
                campaign_result = await run_attacker_campaign(
                    config=config,
                    adapter=adapter,
                    modules=modules,
                    static_results=results,
                    scan_id=scan_id,
                )
                results = results + campaign_result.eval_results
                lineage = campaign_result.lineage
                # Deferred import (D-74): `require_deep_extra()` already ran
                # as the FIRST thing inside `run_attacker_campaign()` above,
                # so the `[deep]` extra is confirmed installed by this
                # point — importing `attacker.budget` here, rather than at
                # this module's top level, means a `--quick` run never pays
                # for (or requires) the optional langchain/langgraph stack.
                from llmsec.attacker.budget import truncation_disclosure

                ledger = campaign_result.final_state.get("budget_ledger")
                if ledger is not None:
                    deep_truncation_note = truncation_disclosure(ledger)
            except Exception as exc:
                # Honest degradation (matches the NER-not-installed
                # precedent below): the static results/report are still
                # produced intact, disclosed via _scan_limitations().
                logger.warning(
                    "Deep-mode attacker campaign failed, continuing with "
                    "static results only: %s",
                    exc,
                )
                deep_mode_failed = True
                campaign_result = None

        # Phase 6 (06-01): standalone-audit results merge in AFTER the
        # entire deep-mode block, immediately BEFORE the findings loop --
        # `static_results` (above) and `run_attacker_campaign(static_results=...)`
        # both keep receiving exactly the orchestrator's own output,
        # untouched by this merge, preserving D-93's guarantee.
        # Phase 7 (07-01): direct-probe (flood-class) results merge in at
        # the SAME point, for the SAME reason -- a flood-class case_id must
        # never enter the deep-mode Strategist's mutation-selection pool
        # (Pitfall 3, 07-RESEARCH.md).
        results = results + audit_results + direct_probe_results

        findings: list[Finding] = []
        case_log: list[EvalResult] = []
        for module_id, eval_result in results:
            # Every generated case's outcome is logged, regardless of
            # verdict — closes the CLI-01/SCORE-01 no-silent-drop
            # prohibition: a `blocked` case produces no `Finding` but is
            # never missing from the audit trail.
            case_log.append(eval_result)

            severity = score(eval_result.verdict, eval_result.evidence)
            if severity is Severity.NONE:
                continue

            module = modules.get(module_id)
            owasp_ref = module.owasp_ref if module is not None else ""
            # D-26: prefer the module-supplied, per-technique remediation
            # when the eval result carries one; fall back to the generic
            # verdict-keyed table when it does not, so Phase 1's module
            # (which supplies no per-case remediation) keeps working
            # exactly as it does today.
            remediation = eval_result.remediation or _REMEDIATION_BY_VERDICT[eval_result.verdict]
            # D-32: canary-PII values are fake by construction and shown
            # VERBATIM in report evidence to prove the echo occurred — the
            # one deliberate exception to D-34's redact-everything rule.
            # Every other detection layer (including this plan's "regex")
            # is redacted through both primitives, in order, AFTER score()
            # has already run against the raw evidence above (Pattern 3 —
            # never invert this ordering).
            if eval_result.detection_layer == "canary":
                # WR-05: the D-32 exemption is scoped to the specific
                # canary literal(s), never the whole evidence string — a
                # real secret co-located in the same excerpt as an echoed
                # canary value must still be redacted. `detection_layer ==
                # "canary"` is only ever emitted by `pii_exfiltration.py`
                # (`prompt_injection.py`'s canary tier uses the fixed
                # `CANARY_TOKEN` but always reports detection_layer
                # "regex"/"judge"), so `module.canary_pii_values()` is
                # always available here; the `getattr` fallback exists
                # only so a future module using this layer name without
                # that accessor degrades to the old (safe-if-imprecise)
                # verbatim behavior instead of crashing the scan.
                canary_pii_values = getattr(module, "canary_pii_values", None)
                if canary_pii_values is not None:
                    evidence = _redact_protecting_canary_values(
                        eval_result.evidence, canary_pii_values()
                    )
                else:
                    evidence = eval_result.evidence
            else:
                # CR-01: `redact_pii_match()` MUST run first. Its patterns
                # (e.g. `_JWT_RE`) are structurally precise (dot-delimited
                # segments, anchored shapes); `redact_credential_match()`'s
                # generic 32+-char catch-all has no `\b` boundary and no
                # dot in its character class, so if it ran first it could
                # partially consume one segment of a larger dot-structured
                # secret (a JWT) and insert a `***REDACTED***` marker whose
                # `*` characters break the dot-structure `_JWT_RE` needs to
                # match the remaining segments — leaving real payload/
                # signature content unredacted. Running the structurally
                # precise pass first, then the generic catch-all second,
                # ensures the generic pass can only ever mask what the
                # precise pass left behind, never destroy its ability to
                # match. Never invert this order (see 03-REVIEW.md CR-01).
                evidence = redact_credential_match(redact_pii_match(eval_result.evidence))
            # D-90: populate lineage from the runner's lineage map keyed by
            # case_id -- the ONLY sanctioned source of truth. Absent for
            # every static finding (empty `lineage` dict), which is why
            # every field below defaults to None rather than raising.
            lineage_record = lineage.get(eval_result.case_id)
            findings.append(
                Finding(
                    case_id=eval_result.case_id,
                    # `EvalResult` carries no separate `technique_id` field;
                    # every built-in module (system_prompt_leakage) sets
                    # `case_id == technique_id` in `generate_cases()`, so
                    # `case_id` doubles as the technique identifier here.
                    technique_id=eval_result.case_id,
                    verdict=eval_result.verdict,
                    severity=severity.value,
                    owasp_ref=owasp_ref,
                    evidence=evidence,
                    remediation=remediation,
                    transport_mode=eval_result.transport_mode,
                    detection_layer=eval_result.detection_layer,
                    parent_case_id=lineage_record["parent_case_id"] if lineage_record else None,
                    parent_technique_id=(
                        lineage_record["parent_technique_id"] if lineage_record else None
                    ),
                    round=lineage_record["round"] if lineage_record else None,
                    contributing_agent=(
                        lineage_record["contributing_agent"] if lineage_record else None
                    ),
                )
            )

        # SCORE-01: severity descending, then case_id ascending — deterministic
        # across runs against an unchanged target/response set.
        findings.sort(key=lambda f: (_SEVERITY_RANK[Severity(f.severity)], f.case_id))

        # D-91/Task 2: populated ONLY when the deep branch actually ran and
        # returned a `CampaignResult` (never on a static-only run, never
        # when the attacker layer raised) — `findings`/`static_results` are
        # both post-merge-decision values by this point, exactly what
        # `compute_deep_summary()`'s own docstring requires.
        deep_summary: DeepModeSummary | None = None
        if campaign_result is not None:
            deep_summary = compute_deep_summary(findings, static_results, campaign_result)

        completed_at = datetime.now(timezone.utc).isoformat()
        module_ids = list(modules.keys())
        report = ScanReport(
            scan_id=scan_id,
            target_summary=_target_summary(config),
            module_ids=module_ids,
            findings=findings,
            case_log=case_log,
            started_at=started_at,
            completed_at=completed_at,
            limitations=_scan_limitations(
                module_ids,
                case_log,
                deep_mode_failed=deep_mode_failed,
                deep_truncation_note=deep_truncation_note,
            ),
            deep_summary=deep_summary,
        )
        await JsonReporter().write(report, Path(config.output_dir))
        return report
    finally:
        await adapter.close()
