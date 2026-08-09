"""Tests for `llmsec.api.run_scan()` — full pipeline wiring + library
entrypoint (plan 08 Task 2).

Reuses the mocked-adapter/mocked-module pattern established in
`tests/test_orchestrator.py`; no live network calls.
"""

from __future__ import annotations

import inspect
import json
from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest

import llmsec.api as api_module
from llmsec.auth_gate import AuthorizationDeclined
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict


class _MockAdapter:
    """Minimal `TargetAdapter`-shaped mock. `send` is an `AsyncMock` so
    tests can assert call/no-call directly.

    Capability flags default to the `TargetAdapter` ABC defaults (both
    False, matching an `HttpAppAdapter` with no session config) and can be
    overridden per-instance to simulate a capable adapter (e.g.
    `LLMApiAdapter`, or an `HttpAppAdapter` with a configured session
    round-trip) without needing a real adapter subclass.
    """

    def __init__(
        self,
        supports_system_prompt_override: bool = False,
        supports_multi_turn: bool = False,
    ) -> None:
        self.send = AsyncMock(side_effect=self._send)
        self.closed = False
        self.supports_system_prompt_override = supports_system_prompt_override
        self.supports_multi_turn = supports_multi_turn

    async def _send(self, case: TestCase) -> TargetResponse:
        return TargetResponse(case_id=case.case_id, raw_text=f"response-to-{case.case_id}", latency_ms=1.0)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


class _MockModule:
    """Minimal `BaseModule`-shaped mock yielding one `TestCase` per
    `(case_id, verdict, evidence)` tuple and returning that verdict/evidence
    verbatim from `evaluate()` — no real detection logic."""

    id = "mock_module"
    name = "Mock Module"
    owasp_ref = "LLM07:2025"
    uses_attacker_llm = False

    def __init__(self, cases: list[tuple[str, Verdict, str]]) -> None:
        self._cases = cases

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        for case_id, _verdict, _evidence in self._cases:
            yield TestCase(case_id=case_id, prompt=f"prompt-{case_id}", technique_id=case_id)

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        verdict, evidence = next(
            (v, e) for cid, v, e in self._cases if cid == case.case_id
        )
        return EvalResult(
            case_id=case.case_id,
            verdict=verdict,
            confidence=0.9,
            evidence=evidence,
            detection_layer="regex",
        )


class _RaisingModule(_MockModule):
    """`_MockModule` variant whose `evaluate()` always raises, simulating a
    judge-side (rate limit/auth/non-schema) failure."""

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        raise RuntimeError("judge boom")


def _http_app_config(tmp_path, enabled_modules: list[str]) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(
            type="http_app",
            method="POST",
            url="http://localhost:8000/chat",
            headers={},
            body_template='{"message": "{{payload}}"}',
            response_path="response",
        ),
        enabled_modules=enabled_modules,
        max_concurrency=5,
        output_dir=str(tmp_path / "reports"),
        judge_model="openai/gpt-4o-mini",
        judge_api_key_env=None,
    )


def _patch_http_adapter(
    monkeypatch: pytest.MonkeyPatch,
    supports_system_prompt_override: bool = False,
    supports_multi_turn: bool = False,
) -> list[_MockAdapter]:
    """Patch `llmsec.api.HttpAppAdapter` to a factory returning
    `_MockAdapter` instances, and return the list of created instances.

    Defaults to both capability flags False, matching a real
    `HttpAppAdapter` with no session config configured.
    """
    created: list[_MockAdapter] = []

    def _factory(*args, **kwargs) -> _MockAdapter:
        instance = _MockAdapter(
            supports_system_prompt_override=supports_system_prompt_override,
            supports_multi_turn=supports_multi_turn,
        )
        created.append(instance)
        return instance

    monkeypatch.setattr(api_module, "HttpAppAdapter", _factory)
    return created


def _patch_raw_llm_adapter(monkeypatch: pytest.MonkeyPatch) -> list[_MockAdapter]:
    """Patch `llmsec.api.LLMApiAdapter` to a factory returning `_MockAdapter`
    instances with both capability flags True, matching the real
    `LLMApiAdapter`'s class-level `supports_multi_turn = supports_system_prompt_override
    = True`."""
    created: list[_MockAdapter] = []

    def _factory(*args, **kwargs) -> _MockAdapter:
        instance = _MockAdapter(supports_system_prompt_override=True, supports_multi_turn=True)
        created.append(instance)
        return instance

    monkeypatch.setattr(api_module, "LLMApiAdapter", _factory)
    return created


def _raw_llm_config(tmp_path, enabled_modules: list[str]) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="OPENAI_API_KEY"),
        enabled_modules=enabled_modules,
        max_concurrency=5,
        output_dir=str(tmp_path / "reports"),
        judge_model="openai/gpt-4o-mini",
        judge_api_key_env=None,
    )


class _MockModuleAdvanced:
    """`_MockModule`-shaped mock module whose cases carry an optional
    `remediation` and `transport_mode`, for testing D-26 remediation
    override and transport-mode round-trip through `run_scan()`."""

    id = "mock_module"
    name = "Mock Module"
    owasp_ref = "LLM07:2025"
    uses_attacker_llm = False

    def __init__(self, cases: list[dict]) -> None:
        self._cases = cases

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        for case in self._cases:
            yield TestCase(case_id=case["case_id"], prompt=f"prompt-{case['case_id']}", technique_id=case["case_id"])

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        spec = next(c for c in self._cases if c["case_id"] == case.case_id)
        return EvalResult(
            case_id=case.case_id,
            verdict=spec["verdict"],
            confidence=0.9,
            evidence=spec.get("evidence", "evidence"),
            detection_layer="regex",
            transport_mode=spec.get("transport_mode"),
            remediation=spec.get("remediation"),
        )


def _patch_modules(monkeypatch: pytest.MonkeyPatch, module: _MockModule) -> None:
    monkeypatch.setattr(
        api_module.PluginRegistry,
        "load_allowed",
        lambda self, allowlist, module_config=None: {module.id: module},
    )


async def test_run_scan_full_pipeline_returns_populated_report_and_writes_json(tmp_path, monkeypatch):
    module = _MockModule(
        [
            ("c1", Verdict.BLOCKED, "no leak here"),
            ("c2", Verdict.FULL_COMPROMISE, "revealed the whole system prompt"),
        ]
    )
    _patch_modules(monkeypatch, module)
    created_adapters = _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert len(report.case_log) == 2
    assert len(report.findings) == 1  # only the FULL_COMPROMISE case scores above NONE
    assert report.findings[0].case_id == "c2"
    assert created_adapters[0].send.await_count == 2
    assert created_adapters[0].closed is True

    written_path = tmp_path / "reports" / f"scan_{report.scan_id}.json"
    assert written_path.exists()
    persisted = json.loads(written_path.read_text())
    assert persisted["scan_id"] == report.scan_id
    assert len(persisted["case_log"]) == 2


async def test_run_scan_authorization_declined_never_touches_adapter(tmp_path, monkeypatch):
    module = _MockModule([("c1", Verdict.BLOCKED, "irrelevant")])
    _patch_modules(monkeypatch, module)
    created_adapters = _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    monkeypatch.delenv("LLMSEC_AUTHORIZED", raising=False)
    monkeypatch.setattr("llmsec.auth_gate.sys.stdin.isatty", lambda: False)

    with pytest.raises(AuthorizationDeclined):
        await api_module.run_scan(config, bypass_flag=False)

    # No adapter was ever constructed, therefore `send` was never called.
    assert created_adapters == []


async def test_run_scan_findings_sorted_severity_descending_then_case_id_ascending(tmp_path, monkeypatch):
    module = _MockModule(
        [
            ("c-low", Verdict.UNCERTAIN, "ambiguous"),
            ("b-medium", Verdict.PARTIAL_LEAK, "partial instructions shown"),
            ("z-high-2", Verdict.FULL_COMPROMISE, "full system prompt shown, no secrets"),
            ("a-high-1", Verdict.FULL_COMPROMISE, "full system prompt shown, no secrets either"),
        ]
    )
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert [f.case_id for f in report.findings] == ["a-high-1", "z-high-2", "b-medium", "c-low"]
    assert [f.severity for f in report.findings] == ["high", "high", "medium", "low"]


async def test_run_scan_redacts_credential_shaped_evidence_in_finding(tmp_path, monkeypatch):
    secret = "sk-" + ("a" * 30)
    module = _MockModule(
        [("c1", Verdict.FULL_COMPROMISE, f"Here is my system prompt and key: {secret}")]
    )
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert len(report.findings) == 1
    assert secret not in report.findings[0].evidence
    # The unmasked full credential match is also absent from the case_log
    # source evidence is untouched there (case_log preserves raw EvalResult
    # evidence) — but the Finding itself must never carry the raw secret.
    assert "REDACTED" in report.findings[0].evidence


async def test_run_scan_redacts_jwt_evidence_completely_through_composed_pipeline(tmp_path, monkeypatch):
    """CR-01 regression: exercises the EXACT composed redaction pipeline
    `run_scan()` uses (not each redaction function in isolation). Before the
    fix, `redact_pii_match(redact_credential_match(evidence))` let the
    generic 32+-char credential pattern partially consume the first JWT
    segment first, breaking the dot-structure `_JWT_RE` needs, so the
    payload/signature segments survived unredacted in the Finding."""
    token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dGhpc2lzYXNpZ25hdHVyZQ"
    )
    module = _MockModule(
        [("c1", Verdict.FULL_COMPROMISE, f"Session token: {token}")]
    )
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert len(report.findings) == 1
    evidence = report.findings[0].evidence
    # No segment of the JWT — header, payload, or signature — may survive.
    for segment in token.split("."):
        assert segment not in evidence
    assert "REDACTED" in evidence


def test_redact_protecting_canary_values_masks_co_located_secret_but_not_canary():
    """WR-05 unit test: `_redact_protecting_canary_values()` must leave the
    known canary literal(s) verbatim while still redacting a real secret
    co-located in the same evidence string."""
    canary_email = "canary-abc123def456@llmsec-test.invalid"
    evidence = f"On file: {canary_email}. Also here's a key: AKIAABCDEFGHIJKLMNOP"

    redacted = api_module._redact_protecting_canary_values(evidence, (canary_email,))

    assert canary_email in redacted  # D-32: canary literal survives verbatim
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted  # WR-05: co-located secret redacted
    assert "REDACTED" in redacted


def test_redact_protecting_canary_values_empty_evidence_returns_unchanged():
    assert api_module._redact_protecting_canary_values("", ("x",)) == ""


def test_redact_protecting_canary_values_ignores_values_absent_from_evidence():
    """A canary value that never appears in the evidence text must be a
    no-op for the sentinel-protection step -- only redaction of whatever
    IS present matters."""
    evidence = "Here's a key: AKIAABCDEFGHIJKLMNOP"
    redacted = api_module._redact_protecting_canary_values(
        evidence, ("never-appears@llmsec-test.invalid",)
    )
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "REDACTED" in redacted


def test_redact_protecting_canary_values_redacts_secret_spanning_canary_jwt_shape():
    """CR-02 regression: a real secret that forms one contiguous
    JWT-shaped `header.CANARY.footer` match TOGETHER WITH the canary
    literal must still have its real (non-canary) segments redacted, even
    though the canary literal itself sits inside the same pattern match.

    Before the CR-02 fix, `_redact_protecting_canary_values()` swapped the
    canary literal for a `\\x00`-delimited sentinel before matching, which
    broke `_JWT_RE`'s ability to match the whole dot-structured span at
    all -- so NOTHING in that span was redacted and `header`/`footer`
    (real secret fragments) shipped completely unmasked. This exact
    scenario is the one 03-REVIEW.md CR-02 reproduced against the shipped
    code."""
    canary_api_key = "llmsec-canary-abcdef0123456789abcdef0123456789"
    header = "REALSECRETHEAD"
    footer = "REALSECRETFOOT"
    evidence = f"{header}.{canary_api_key}.{footer}"

    redacted = api_module._redact_protecting_canary_values(evidence, (canary_api_key,))

    assert canary_api_key in redacted  # D-32: canary literal survives verbatim
    assert header not in redacted  # CR-02: real secret fragment before canary is redacted
    assert footer not in redacted  # CR-02: real secret fragment after canary is redacted
    assert "REDACTED" in redacted


class _MockModuleWithCanary(_MockModule):
    """`_MockModule` variant whose `evaluate()` always reports
    `detection_layer="canary"` and exposes `canary_pii_values()`, so
    `run_scan()`'s canary-tier redaction-scoping branch (WR-05) can be
    exercised end to end without needing the real `PiiExfiltrationModule`
    corpus/multi-turn machinery."""

    def __init__(self, cases: list[tuple[str, Verdict, str]], canary_values: tuple[str, ...]) -> None:
        super().__init__(cases)
        self._canary_values = canary_values

    def canary_pii_values(self) -> tuple[str, ...]:
        return self._canary_values

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        result = await super().evaluate(case, response)
        return result.model_copy(update={"detection_layer": "canary"})


async def test_run_scan_canary_finding_redacts_co_located_secret_not_canary_literal(
    tmp_path, monkeypatch
):
    """WR-05 end-to-end regression: a canary-tier `Finding` must redact a
    real secret co-located in the same evidence excerpt as the echoed
    canary value, while the canary literal itself still survives verbatim
    (D-32) -- not the old coarse whole-evidence exemption."""
    canary_value = "canary-abc123def456@llmsec-test.invalid"
    module = _MockModuleWithCanary(
        [
            (
                "c1",
                Verdict.FULL_COMPROMISE,
                f"On file: {canary_value}. Also here's a key: AKIAABCDEFGHIJKLMNOP",
            )
        ],
        canary_values=(canary_value,),
    )
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert len(report.findings) == 1
    evidence = report.findings[0].evidence
    assert report.findings[0].detection_layer == "canary"
    assert canary_value in evidence
    assert "AKIAABCDEFGHIJKLMNOP" not in evidence
    assert "REDACTED" in evidence


async def test_run_scan_canary_finding_falls_back_to_verbatim_when_module_lacks_canary_accessor(
    tmp_path, monkeypatch
):
    """A module reporting `detection_layer="canary"` but with no
    `canary_pii_values()` accessor degrades to the old (safe-if-imprecise)
    verbatim passthrough rather than crashing the scan."""

    class _CanaryModuleNoAccessor(_MockModule):
        async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
            result = await super().evaluate(case, response)
            return result.model_copy(update={"detection_layer": "canary"})

    module = _CanaryModuleNoAccessor([("c1", Verdict.FULL_COMPROMISE, "planted-value echoed back")])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.findings[0].evidence == "planted-value echoed back"


async def test_run_scan_persists_report_when_evaluate_raises(tmp_path, monkeypatch):
    module = _RaisingModule([("c1", Verdict.BLOCKED, "irrelevant")])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    written_path = tmp_path / "reports" / f"scan_{report.scan_id}.json"
    assert written_path.exists()
    assert len(report.case_log) == 1
    assert report.case_log[0].case_id == "c1"
    assert report.case_log[0].verdict == Verdict.UNCERTAIN


async def test_run_scan_forwards_module_config_from_config(tmp_path, monkeypatch):
    captured: dict = {}

    def _capturing_load_allowed(self, allowlist, module_config=None):
        captured["allowlist"] = allowlist
        captured["module_config"] = module_config
        module = _MockModule([("c1", Verdict.BLOCKED, "irrelevant")])
        return {module.id: module}

    monkeypatch.setattr(api_module.PluginRegistry, "load_allowed", _capturing_load_allowed)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])
    config.known_system_prompt = "GROUND TRUTH"

    await api_module.run_scan(config, bypass_flag=True)

    assert captured["module_config"] is not None
    assert captured["module_config"]["mock_module"]["known_system_prompt"] == "GROUND TRUTH"
    assert captured["module_config"]["mock_module"]["judge_model"] == config.judge_model
    # WR-03: judge_api_key_env must also be forwarded into module_config so
    # a built-in module can actually consume it — previously dropped here,
    # leaving the field a no-op regardless of what the operator configured.
    assert captured["module_config"]["mock_module"]["judge_api_key_env"] == config.judge_api_key_env


def test_run_scan_never_calls_asyncio_run():
    source = inspect.getsource(api_module)
    assert "asyncio.run(" not in source


# --- Task 1 (plan 09): capability wiring, remediation override, limitations ---


async def test_run_scan_context_capability_flags_true_for_raw_llm_target(tmp_path, monkeypatch):
    """`ScanContext` capability flags mirror the constructed adapter's real
    flags — a raw-LLM target (LLMApiAdapter: both True) gets the strong
    system-prompt canary planting (D-17)."""
    captured: dict = {}

    class _CapturingOrchestrator:
        def __init__(self, adapter, modules, max_concurrency):
            self.adapter = adapter
            self.modules = modules

        async def run(self, context: ScanContext):
            captured["context"] = context
            return []

    monkeypatch.setattr(api_module, "ScanOrchestrator", _CapturingOrchestrator)
    module = _MockModule([])
    _patch_modules(monkeypatch, module)
    _patch_raw_llm_adapter(monkeypatch)
    config = _raw_llm_config(tmp_path, enabled_modules=["mock_module"])

    await api_module.run_scan(config, bypass_flag=True)

    assert captured["context"].system_prompt_controllable is True
    assert captured["context"].supports_multi_turn is True


async def test_run_scan_context_capability_flags_false_for_http_app_no_session(tmp_path, monkeypatch):
    """An HTTP-app target with no session config (both flags False on the
    constructed adapter) gets the honestly-weaker turn-based fallback, never
    guessed True."""
    captured: dict = {}

    class _CapturingOrchestrator:
        def __init__(self, adapter, modules, max_concurrency):
            self.adapter = adapter
            self.modules = modules

        async def run(self, context: ScanContext):
            captured["context"] = context
            return []

    monkeypatch.setattr(api_module, "ScanOrchestrator", _CapturingOrchestrator)
    module = _MockModule([])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)  # defaults: both False
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    await api_module.run_scan(config, bypass_flag=True)

    assert captured["context"].system_prompt_controllable is False
    assert captured["context"].supports_multi_turn is False


async def test_run_scan_finding_uses_eval_result_remediation_when_set(tmp_path, monkeypatch):
    module = _MockModuleAdvanced(
        [
            {
                "case_id": "c1",
                "verdict": Verdict.FULL_COMPROMISE,
                "evidence": "revealed everything",
                "remediation": "Rotate the leaked credential and rewrite the system prompt.",
            }
        ]
    )
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert len(report.findings) == 1
    assert report.findings[0].remediation == "Rotate the leaked credential and rewrite the system prompt."


async def test_run_scan_finding_falls_back_to_verdict_keyed_remediation_when_none(tmp_path, monkeypatch):
    module = _MockModuleAdvanced(
        [
            {
                "case_id": "c1",
                "verdict": Verdict.FULL_COMPROMISE,
                "evidence": "revealed everything",
                "remediation": None,
            }
        ]
    )
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert len(report.findings) == 1
    assert report.findings[0].remediation == api_module._REMEDIATION_BY_VERDICT[Verdict.FULL_COMPROMISE]


async def test_run_scan_finding_transport_mode_round_trips_from_eval_result(tmp_path, monkeypatch):
    module = _MockModuleAdvanced(
        [
            {
                "case_id": "c1",
                "verdict": Verdict.FULL_COMPROMISE,
                "evidence": "revealed everything",
                "transport_mode": "multi_turn_concatenated",
            },
            {
                "case_id": "c2",
                "verdict": Verdict.PARTIAL_LEAK,
                "evidence": "partial",
                "transport_mode": None,
            },
        ]
    )
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    by_case = {f.case_id: f for f in report.findings}
    assert by_case["c1"].transport_mode == "multi_turn_concatenated"
    assert by_case["c2"].transport_mode is None


async def test_run_scan_limitations_empty_when_only_non_injection_module_loaded(tmp_path, monkeypatch):
    module = _MockModule([("c1", Verdict.BLOCKED, "no leak here")])
    module.id = "system_prompt_leakage"
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["system_prompt_leakage"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.limitations == []


async def test_run_scan_limitations_include_canary_and_indirect_caveats_when_prompt_injection_loaded(
    tmp_path, monkeypatch
):
    module = _MockModule([("c1", Verdict.BLOCKED, "no leak here")])
    module.id = "prompt_injection"
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["prompt_injection"])

    report = await api_module.run_scan(config, bypass_flag=True)

    from llmsec.detection.canary import CANARY_LIMITATION_NOTE

    assert CANARY_LIMITATION_NOTE in report.limitations
    assert any("retrieval pipeline" in item for item in report.limitations)
    # Fixed order: canary caveat before indirect-simulation caveat.
    assert report.limitations.index(CANARY_LIMITATION_NOTE) < len(report.limitations) - 1


async def test_run_scan_limitations_include_degraded_caveat_even_when_case_produces_no_finding(
    tmp_path, monkeypatch
):
    """A degraded-transport case that produces NO finding (e.g. a `blocked`
    verdict) must still surface the degraded-multi-turn caveat — computed
    from the case log, not the findings list (D-15)."""
    module = _MockModuleAdvanced(
        [
            {
                "case_id": "c1",
                "verdict": Verdict.BLOCKED,
                "evidence": "held the line",
                "transport_mode": "multi_turn_concatenated",
            }
        ]
    )
    module.id = "system_prompt_leakage"  # not prompt_injection — isolates the degraded-transport condition
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["system_prompt_leakage"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.findings == []
    assert len(report.limitations) == 1
    assert "concatenated" in report.limitations[0]


async def test_run_scan_no_degraded_caveat_when_no_case_ran_over_concatenated_transport(tmp_path, monkeypatch):
    module = _MockModuleAdvanced(
        [{"case_id": "c1", "verdict": Verdict.BLOCKED, "evidence": "fine", "transport_mode": "single"}]
    )
    module.id = "system_prompt_leakage"
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["system_prompt_leakage"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.limitations == []


async def test_run_scan_limitations_include_ner_not_installed_caveat_when_pii_exfiltration_loaded_and_ner_unavailable(
    tmp_path, monkeypatch
):
    """CR-02 regression: a well-behaved target (every case BLOCKED, severity
    NONE, so NO Finding is ever created) must still surface the
    NER-not-installed caveat in `report.limitations` — the only report
    surface `report.md.j2` and the JSON reporter actually render. Before
    the fix, this caveat only ever reached `EvalResult.evidence` inside
    `case_log`, which the Markdown report never displays, so a scan run
    without `[pii-ner]` installed against a clean target looked like a
    fully-covered "No findings." result."""
    monkeypatch.setattr(api_module, "ner_available", lambda: False)
    module = _MockModule([("c1", Verdict.BLOCKED, "held the line, NER tier skipped")])
    module.id = "pii_exfiltration"
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["pii_exfiltration"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.findings == []
    assert api_module._NER_NOT_INSTALLED_LIMITATION_NOTE in report.limitations


async def test_run_scan_no_ner_caveat_when_ner_available(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "ner_available", lambda: True)
    module = _MockModule([("c1", Verdict.BLOCKED, "held the line")])
    module.id = "pii_exfiltration"
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["pii_exfiltration"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert api_module._NER_NOT_INSTALLED_LIMITATION_NOTE not in report.limitations


async def test_run_scan_no_ner_caveat_when_pii_exfiltration_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "ner_available", lambda: False)
    module = _MockModule([("c1", Verdict.BLOCKED, "held the line")])
    module.id = "system_prompt_leakage"
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["system_prompt_leakage"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert api_module._NER_NOT_INSTALLED_LIMITATION_NOTE not in report.limitations


async def test_run_scan_repeated_runs_produce_identical_limitations(tmp_path, monkeypatch):
    def _make_module():
        module = _MockModuleAdvanced(
            [
                {
                    "case_id": "c1",
                    "verdict": Verdict.BLOCKED,
                    "evidence": "held",
                    "transport_mode": "multi_turn_concatenated",
                }
            ]
        )
        module.id = "prompt_injection"
        return module

    _patch_modules(monkeypatch, _make_module())
    _patch_http_adapter(monkeypatch)
    config1 = _http_app_config(tmp_path / "run1", enabled_modules=["prompt_injection"])
    report1 = await api_module.run_scan(config1, bypass_flag=True)

    _patch_modules(monkeypatch, _make_module())
    _patch_http_adapter(monkeypatch)
    config2 = _http_app_config(tmp_path / "run2", enabled_modules=["prompt_injection"])
    report2 = await api_module.run_scan(config2, bypass_flag=True)

    assert report1.limitations == report2.limitations


async def test_run_scan_no_modules_producing_findings_returns_well_formed_report(tmp_path, monkeypatch):
    module = _MockModule([])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    config = _http_app_config(tmp_path, enabled_modules=["mock_module"])

    report = await api_module.run_scan(config, bypass_flag=True)

    assert report.findings == []
    assert report.case_log == []
    assert report.limitations == []


def test_scan_limitations_reuses_canary_limitation_note_not_restated():
    """Acceptance criterion: the caveat text is imported/reused, never
    hand-copied — so it can never drift from `canary.CANARY_LIMITATION_NOTE`."""
    source = inspect.getsource(api_module)
    assert "CANARY_LIMITATION_NOTE" in source


def test_build_adapter_forwards_session_id_fields_to_real_http_app_adapter():
    """Regression test (CR-01): `_build_adapter()` must forward
    `target.session_id_path`/`target.session_id_header` through to a REAL
    `HttpAppAdapter` (not the `_MockAdapter` factory used elsewhere in this
    file), so `supports_multi_turn` genuinely activates end-to-end from
    config/CLI. Previously these fields were silently dropped, leaving
    every config-driven HTTP-app scan permanently degraded to
    `multi_turn_concatenated` transport regardless of what the operator
    configured."""
    config = ScanConfig(
        target=TargetConfig(
            type="http_app",
            method="POST",
            url="http://localhost:8000/chat",
            headers={},
            body_template='{"message": "{{payload}}", "session_id": "{{session_id}}"}',
            response_path="response",
            session_id_path="session_id",
            session_id_header=None,
        ),
        enabled_modules=["prompt_injection"],
        max_concurrency=5,
        judge_model="openai/gpt-4o-mini",
        judge_api_key_env=None,
    )

    adapter = api_module._build_adapter(config)

    assert isinstance(adapter, api_module.HttpAppAdapter)
    assert adapter.session_id_path == "session_id"
    assert adapter.supports_multi_turn is True
