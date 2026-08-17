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


# --- Phase 6 (06-01): _run_standalone_audits() containment + determinism ---


class _RaisingAuditModule:
    """`run_standalone_audit()` raises immediately -- simulates one
    module's audit failing (T-01-18)."""

    id = "raising_audit_module"

    async def run_standalone_audit(self, context):
        raise RuntimeError("audit boom")
        yield  # pragma: no cover — unreachable, keeps this an async generator


class _YieldingAuditModule:
    """`run_standalone_audit()` yields one real result."""

    id = "yielding_audit_module"

    async def run_standalone_audit(self, context):
        yield EvalResult(
            case_id="AUDIT-OK",
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            evidence="audit ran fine",
            detection_layer="audit",
        )


async def test_run_standalone_audits_contains_one_module_failure_sibling_still_returns():
    """T-01-18: one module's run_standalone_audit() raising must never
    cancel a sibling module's audit results, and _run_standalone_audits()
    itself must never raise."""
    modules = {
        "raising_audit_module": _RaisingAuditModule(),
        "yielding_audit_module": _YieldingAuditModule(),
    }
    context = ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="")

    results = await api_module._run_standalone_audits(modules, context)

    assert len(results) == 1
    module_id, eval_result = results[0]
    assert module_id == "yielding_audit_module"
    assert eval_result.case_id == "AUDIT-OK"


def test_scan_limitations_determinism_including_supply_chain_conditions():
    """T-02-33: two `_scan_limitations()` calls over the same
    module_ids/case_log produce byte-identical lists, including when
    conditions 7-11 (supply-chain/poisoning) fire."""
    from llmsec.modules.supply_chain import (
        AUDIT_CASE_ID_EXTRA_MISSING,
        AUDIT_CASE_ID_MANIFEST_MISSING,
        AUDIT_CASE_ID_OSV_UNREACHABLE,
    )

    module_ids = ["supply_chain", "data_poisoning"]
    case_log = [
        EvalResult(
            case_id=AUDIT_CASE_ID_MANIFEST_MISSING,
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            evidence="no manifest",
            detection_layer="audit",
        ),
        EvalResult(
            case_id=AUDIT_CASE_ID_EXTRA_MISSING,
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            evidence="no extra",
            detection_layer="audit",
        ),
        EvalResult(
            case_id=AUDIT_CASE_ID_OSV_UNREACHABLE,
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            evidence="osv unreachable",
            detection_layer="audit",
        ),
    ]

    first = api_module._scan_limitations(module_ids, case_log)
    second = api_module._scan_limitations(module_ids, case_log)

    assert first == second
    assert api_module._SUPPLY_CHAIN_MANIFEST_MISSING_NOTE in first
    assert api_module._SUPPLY_CHAIN_EXTRA_NOT_INSTALLED_NOTE in first
    assert api_module._SUPPLY_CHAIN_OSV_UNREACHABLE_NOTE in first
    assert api_module._POISONING_HEURISTIC_ONLY_NOTE in first


def test_audit_detection_layer_finding_round_trips_json_and_renders_markdown(tmp_path):
    """A `Finding` carrying `detection_layer="audit"` round-trips through
    `JsonReporter` and renders in the Markdown template without any
    reporter change -- proves the additive `Literal` widening needed no
    downstream branching."""
    import asyncio

    from llmsec.models import Finding, ScanReport
    from llmsec.reporting.json_reporter import JsonReporter, load_report
    from llmsec.reporting.markdown_reporter import MarkdownReporter

    finding = Finding(
        case_id="SUPPLY-CHAIN-AUDIT-MANIFEST-MISSING",
        technique_id="SUPPLY-CHAIN-AUDIT-MANIFEST-MISSING",
        verdict=Verdict.UNCERTAIN,
        severity="low",
        owasp_ref="LLM03:2025",
        evidence="no manifest configured",
        remediation="configure supply_chain_manifest_path",
        detection_layer="audit",
    )
    report = ScanReport(
        scan_id="audit-round-trip",
        target_summary="raw_llm target, model=gpt-4o-mini",
        module_ids=["supply_chain"],
        findings=[finding],
        case_log=[],
        started_at="2026-08-13T00:00:00Z",
        completed_at="2026-08-13T00:00:05Z",
    )

    json_path = asyncio.run(JsonReporter().write(report, tmp_path / "json"))
    reloaded = load_report(json_path)
    assert reloaded.findings[0].detection_layer == "audit"

    md_path = asyncio.run(MarkdownReporter().write(report, tmp_path / "md"))
    assert "audit" in md_path.read_text()


# --- Phase 7 (07-03): direct-probe deep-mode exclusion + containment ----------


class TestDirectProbeResultsExcludedFromDeepMode:
    """MOD-09/T-07-09: a flood-class case id can never reach the deep-mode
    Strategist's mutation-selection pool, via either of two independent
    barriers -- structural (`generate_cases()` never yields one, so
    `attacker/runner.py::_rebuild_case_by_id()` cannot build one into the
    mutation pool) and ordering (`direct_probe_results` merges into
    `results` only after `static_results` is captured)."""

    async def test_generate_cases_disjoint_from_flood_class_ids(self):
        """Barrier one (structural, primary)."""
        from llmsec.modules.unbounded_consumption import UnboundedConsumptionModule

        module = UnboundedConsumptionModule()
        context = ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="")
        case_ids = {case.case_id async for case in module.generate_cases(context)}
        flood_ids = {entry.id for entry in module._flood_entries()}

        assert case_ids.isdisjoint(flood_ids)

    async def test_deep_mode_static_results_exclude_flood_class_ids_but_case_log_includes_them(
        self, tmp_path, monkeypatch
    ):
        """Barrier two (ordering): `run_attacker_campaign` is monkeypatched
        to record the `static_results` it actually received. No flood-class
        case id may appear there, while the final `report.case_log` DOES
        contain the flood-class results -- that asserted asymmetry is the
        merge-ordering guarantee."""
        from llmsec.attacker.config import AttackerConfig
        from llmsec.attacker.runner import CampaignResult
        from llmsec.modules.unbounded_consumption import UnboundedConsumptionModule

        module = UnboundedConsumptionModule()
        flood_ids = {entry.id for entry in module._flood_entries()}

        monkeypatch.setattr(
            api_module.PluginRegistry,
            "load_allowed",
            lambda self, allowlist, module_config=None: {module.id: module},
        )

        class _AnyCaseAdapter:
            supports_system_prompt_override = False
            supports_multi_turn = False

            async def send(self, case):
                return TargetResponse(
                    case_id=case.case_id, raw_text="ok", latency_ms=1.0, tokens_used=1
                )

            async def send_conversation(self, case, stop_when=None):
                return await self.send(case)

            async def health_check(self):
                return True

            async def close(self):
                pass

        monkeypatch.setattr(api_module, "HttpAppAdapter", lambda *a, **k: _AnyCaseAdapter())

        captured: dict = {}

        async def _capturing_campaign(*, config, adapter, modules, static_results, scan_id):
            captured["static_results"] = static_results
            return CampaignResult(eval_results=[], lineage={})

        monkeypatch.setattr(api_module, "run_attacker_campaign", _capturing_campaign)

        config = _http_app_config(tmp_path, enabled_modules=["unbounded_consumption"])
        config.attacker = AttackerConfig(enabled=True, profile="light")

        report = await api_module.run_scan(config, bypass_flag=True)

        captured_case_ids = {
            eval_result.case_id for _module_id, eval_result in captured["static_results"]
        }
        assert captured_case_ids.isdisjoint(flood_ids)

        report_case_ids = {result.case_id for result in report.case_log}
        assert report_case_ids & flood_ids  # flood-class results ARE present post-merge

    def test_static_results_capture_precedes_direct_probe_merge_in_source(self):
        """Catches a future behavior-preserving reorder of the two lines
        that would otherwise be invisible to a mocked test (Pitfall 3,
        07-RESEARCH.md)."""
        source = inspect.getsource(api_module)
        static_idx = source.index("static_results = results")
        merge_idx = source.index("results = results + audit_results + direct_probe_results")
        assert static_idx < merge_idx


class _RaisingDirectProbeModule:
    """`run_direct_probe()` raises immediately -- simulates one module's
    direct-probe path failing (T-01-18)."""

    id = "raising_direct_probe_module"

    async def run_direct_probe(self, context, adapter):
        raise RuntimeError("probe boom")
        yield  # pragma: no cover — unreachable, keeps this an async generator


class _YieldingDirectProbeModule:
    """`run_direct_probe()` yields two real results."""

    id = "yielding_direct_probe_module"

    async def run_direct_probe(self, context, adapter):
        yield EvalResult(
            case_id="PROBE-OK-1",
            verdict=Verdict.BLOCKED,
            confidence=0.8,
            evidence="probe ran fine 1",
            detection_layer="threshold",
        )
        yield EvalResult(
            case_id="PROBE-OK-2",
            verdict=Verdict.BLOCKED,
            confidence=0.8,
            evidence="probe ran fine 2",
            detection_layer="threshold",
        )


class _PartialThenRaisingDirectProbeModule:
    """`run_direct_probe()` yields one result, THEN raises -- the
    already-yielded result must survive."""

    id = "partial_then_raising_direct_probe_module"

    async def run_direct_probe(self, context, adapter):
        yield EvalResult(
            case_id="PROBE-PARTIAL-1",
            verdict=Verdict.BLOCKED,
            confidence=0.8,
            evidence="partial result before raise",
            detection_layer="threshold",
        )
        raise RuntimeError("boom after one yield")


class TestRunDirectProbesContainment:
    """MOD-09/T-07-10: mirrors the existing `_run_standalone_audits()`
    containment discipline (T-01-18) for `_run_direct_probes()`."""

    async def test_one_module_raising_immediately_sibling_still_returns(self):
        modules = {
            "raising_direct_probe_module": _RaisingDirectProbeModule(),
            "yielding_direct_probe_module": _YieldingDirectProbeModule(),
        }
        context = ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="")
        adapter = _MockAdapter()

        results = await api_module._run_direct_probes(modules, context, adapter)

        assert len(results) == 2
        result_case_ids = {eval_result.case_id for _module_id, eval_result in results}
        assert result_case_ids == {"PROBE-OK-1", "PROBE-OK-2"}

    async def test_one_module_yields_then_raises_partial_result_survives(self):
        modules = {
            "partial_then_raising_direct_probe_module": _PartialThenRaisingDirectProbeModule(),
        }
        context = ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="")
        adapter = _MockAdapter()

        results = await api_module._run_direct_probes(modules, context, adapter)

        assert len(results) == 1
        assert results[0][1].case_id == "PROBE-PARTIAL-1"

    async def test_run_scan_survives_run_direct_probe_raising_and_returns_orchestrator_results(
        self, tmp_path, monkeypatch
    ):
        """Drives a full `run_scan()` where the loaded `unbounded_consumption`
        module's `run_direct_probe()` is monkeypatched to raise, and asserts
        `run_scan()` still returns a `ScanReport` whose `case_log` contains
        the orchestrator-path results."""
        from llmsec.modules.unbounded_consumption import UnboundedConsumptionModule

        module = UnboundedConsumptionModule()
        baseline_id = module._baseline_entries()[0].id

        async def _raising_run_direct_probe(self, context, adapter):
            raise RuntimeError("probe boom")
            yield  # pragma: no cover — unreachable, keeps this an async generator

        monkeypatch.setattr(
            UnboundedConsumptionModule, "run_direct_probe", _raising_run_direct_probe
        )
        monkeypatch.setattr(
            api_module.PluginRegistry,
            "load_allowed",
            lambda self, allowlist, module_config=None: {module.id: module},
        )

        class _BenignAdapter:
            supports_system_prompt_override = False
            supports_multi_turn = False

            async def send(self, case):
                return TargetResponse(
                    case_id=case.case_id, raw_text="ok", latency_ms=1.0, tokens_used=1
                )

            async def send_conversation(self, case, stop_when=None):
                return await self.send(case)

            async def health_check(self):
                return True

            async def close(self):
                pass

        monkeypatch.setattr(api_module, "HttpAppAdapter", lambda *a, **k: _BenignAdapter())

        config = _http_app_config(tmp_path, enabled_modules=["unbounded_consumption"])
        report = await api_module.run_scan(config, bypass_flag=True)

        case_ids = {result.case_id for result in report.case_log}
        assert baseline_id in case_ids


# --- Phase 7 (07-03): consumption flood-probe-cap limitation note -------------


class TestConsumptionLimitationNote:
    """MOD-08/T-07-12 (07-03-PLAN.md Task 2): the flood-probe-cap /
    heuristic-thresholds caveat -- present exactly when
    `unbounded_consumption` is loaded, byte-identical across runs, at a
    fixed position, and rendered in the Markdown report."""

    def test_present_when_unbounded_consumption_loaded(self):
        limitations = api_module._scan_limitations(["unbounded_consumption"], [])
        assert api_module._CONSUMPTION_FLOOD_CAP_NOTE in limitations

    def test_absent_when_unbounded_consumption_not_loaded(self):
        limitations = api_module._scan_limitations(["prompt_injection"], [])
        assert api_module._CONSUMPTION_FLOOD_CAP_NOTE not in limitations

    def test_note_carries_real_cap_value_not_hand_copied(self):
        from llmsec.modules.unbounded_consumption import _DEFAULT_FLOOD_PROBE_CAP

        assert str(_DEFAULT_FLOOD_PROBE_CAP) in api_module._CONSUMPTION_FLOOD_CAP_NOTE

    def test_byte_identical_across_repeated_calls_with_note_at_fixed_position(self):
        """T-02-33: two `_scan_limitations()` calls over the same
        module_ids produce byte-identical lists, with the new note at a
        fixed position (last, after the poisoning heuristic-only note)."""
        module_ids = ["unbounded_consumption", "prompt_injection", "data_poisoning"]
        first = api_module._scan_limitations(module_ids, [])
        second = api_module._scan_limitations(module_ids, [])

        assert first == second
        assert first[-1] == api_module._CONSUMPTION_FLOOD_CAP_NOTE

    async def test_run_scan_with_unbounded_consumption_loaded_includes_note(
        self, tmp_path, monkeypatch
    ):
        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "unbounded_consumption"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["unbounded_consumption"])

        report = await api_module.run_scan(config, bypass_flag=True)

        assert api_module._CONSUMPTION_FLOOD_CAP_NOTE in report.limitations

    async def test_run_scan_without_unbounded_consumption_omits_note(
        self, tmp_path, monkeypatch
    ):
        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "system_prompt_leakage"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["system_prompt_leakage"])

        report = await api_module.run_scan(config, bypass_flag=True)

        assert api_module._CONSUMPTION_FLOOD_CAP_NOTE not in report.limitations

    async def test_rendered_markdown_report_contains_flood_cap_note_text(
        self, tmp_path, monkeypatch
    ):
        from llmsec.reporting.markdown_reporter import MarkdownReporter

        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "unbounded_consumption"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["unbounded_consumption"])

        report = await api_module.run_scan(config, bypass_flag=True)

        md_path = await MarkdownReporter().write(report, tmp_path / "md")
        assert api_module._CONSUMPTION_FLOOD_CAP_NOTE in md_path.read_text()


# --- Phase 8 (08-01): vector-context simulated-retrieval limitation note ------


class TestVectorContextLimitationNote:
    """MOD-10/D-04/D-07 (08-01-PLAN.md Task 2): the simulated-retrieval
    caveat -- present exactly when `vector_embedding_weaknesses` is loaded,
    byte-identical across runs, at a fixed position, and rendered in the
    Markdown report."""

    def test_present_when_vector_embedding_weaknesses_loaded(self):
        limitations = api_module._scan_limitations(["vector_embedding_weaknesses"], [])
        assert api_module._VECTOR_CONTEXT_SIMULATED_NOTE in limitations

    def test_absent_when_vector_embedding_weaknesses_not_loaded(self):
        limitations = api_module._scan_limitations(["prompt_injection"], [])
        assert api_module._VECTOR_CONTEXT_SIMULATED_NOTE not in limitations

    def test_byte_identical_across_repeated_calls_with_note_at_fixed_position(self):
        """T-02-33: two `_scan_limitations()` calls over the same
        module_ids produce byte-identical lists, with the new note at a
        fixed position (last, after the consumption flood-cap note) and the
        twelve pre-existing conditions' relative order unperturbed."""
        module_ids = [
            "vector_embedding_weaknesses",
            "unbounded_consumption",
            "prompt_injection",
            "data_poisoning",
        ]
        first = api_module._scan_limitations(module_ids, [])
        second = api_module._scan_limitations(module_ids, [])

        assert first == second
        assert first[-1] == api_module._VECTOR_CONTEXT_SIMULATED_NOTE
        assert first.index(api_module._CONSUMPTION_FLOOD_CAP_NOTE) < first.index(
            api_module._VECTOR_CONTEXT_SIMULATED_NOTE
        )
        assert first.index(api_module._POISONING_HEURISTIC_ONLY_NOTE) < first.index(
            api_module._CONSUMPTION_FLOOD_CAP_NOTE
        )

    async def test_run_scan_with_vector_embedding_weaknesses_loaded_includes_note(
        self, tmp_path, monkeypatch
    ):
        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "vector_embedding_weaknesses"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["vector_embedding_weaknesses"])

        report = await api_module.run_scan(config, bypass_flag=True)

        assert api_module._VECTOR_CONTEXT_SIMULATED_NOTE in report.limitations

    async def test_run_scan_without_vector_embedding_weaknesses_omits_note(
        self, tmp_path, monkeypatch
    ):
        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "system_prompt_leakage"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["system_prompt_leakage"])

        report = await api_module.run_scan(config, bypass_flag=True)

        assert api_module._VECTOR_CONTEXT_SIMULATED_NOTE not in report.limitations

    async def test_rendered_markdown_report_contains_simulated_note_text(
        self, tmp_path, monkeypatch
    ):
        from llmsec.reporting.markdown_reporter import MarkdownReporter

        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "vector_embedding_weaknesses"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["vector_embedding_weaknesses"])

        report = await api_module.run_scan(config, bypass_flag=True)

        md_path = await MarkdownReporter().write(report, tmp_path / "md")
        assert api_module._VECTOR_CONTEXT_SIMULATED_NOTE in md_path.read_text()


# --- Phase 8 (08-04): excessive-agency no-real-enforcement limitation note -----


class TestExcessiveAgencyLimitationNote:
    """MOD-11/D-07 (08-04-PLAN.md Task 2): the no-real-enforcement caveat --
    present exactly when `excessive_agency` is loaded, present even for a
    zero-finding (all-clean) run, byte-identical across runs, at a fixed
    position (after the vector-context note), and rendered in the Markdown
    report."""

    def test_present_when_excessive_agency_loaded(self):
        limitations = api_module._scan_limitations(["excessive_agency"], [])
        assert api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE in limitations

    def test_absent_when_excessive_agency_not_loaded(self):
        limitations = api_module._scan_limitations(["prompt_injection"], [])
        assert api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE not in limitations

    def test_present_even_with_a_completely_clean_case_log(self):
        """A zero-finding run must still be distinguishable from an
        untested one -- the note is keyed on `module_ids`, never on
        whether any finding was produced."""
        clean_case_log = [
            EvalResult(
                case_id="AGENCY-F01",
                verdict=Verdict.BLOCKED,
                confidence=0.8,
                evidence="clean refusal",
                detection_layer="regex",
            )
        ]
        limitations = api_module._scan_limitations(["excessive_agency"], clean_case_log)
        assert api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE in limitations

    def test_byte_identical_across_repeated_calls_with_note_at_fixed_position(self):
        """T-02-33: two `_scan_limitations()` calls over the same
        module_ids produce byte-identical lists, with the new note at a
        fixed position (last, after the vector-context note), the relative
        order of the thirteen pre-existing conditions unperturbed, and
        condition 13 strictly before condition 14."""
        module_ids = [
            "excessive_agency",
            "vector_embedding_weaknesses",
            "unbounded_consumption",
            "prompt_injection",
            "data_poisoning",
        ]
        first = api_module._scan_limitations(module_ids, [])
        second = api_module._scan_limitations(module_ids, [])

        assert first == second
        assert first[-1] == api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE
        assert first.index(api_module._VECTOR_CONTEXT_SIMULATED_NOTE) < first.index(
            api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE
        )
        assert first.index(api_module._CONSUMPTION_FLOOD_CAP_NOTE) < first.index(
            api_module._VECTOR_CONTEXT_SIMULATED_NOTE
        )
        assert first.index(api_module._POISONING_HEURISTIC_ONLY_NOTE) < first.index(
            api_module._CONSUMPTION_FLOOD_CAP_NOTE
        )

    def test_both_new_phase_8_modules_together_emit_both_notes_in_fixed_order(self):
        module_ids = ["excessive_agency", "vector_embedding_weaknesses"]
        limitations = api_module._scan_limitations(module_ids, [])
        assert limitations.index(api_module._VECTOR_CONTEXT_SIMULATED_NOTE) < limitations.index(
            api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE
        )

    async def test_run_scan_with_excessive_agency_loaded_includes_note(
        self, tmp_path, monkeypatch
    ):
        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "excessive_agency"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["excessive_agency"])

        report = await api_module.run_scan(config, bypass_flag=True)

        assert api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE in report.limitations

    async def test_run_scan_without_excessive_agency_omits_note(
        self, tmp_path, monkeypatch
    ):
        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "system_prompt_leakage"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["system_prompt_leakage"])

        report = await api_module.run_scan(config, bypass_flag=True)

        assert api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE not in report.limitations

    async def test_rendered_markdown_report_contains_no_enforcement_note_text(
        self, tmp_path, monkeypatch
    ):
        from llmsec.reporting.markdown_reporter import MarkdownReporter

        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "excessive_agency"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["excessive_agency"])

        report = await api_module.run_scan(config, bypass_flag=True)

        md_path = await MarkdownReporter().write(report, tmp_path / "md")
        assert api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE in md_path.read_text()


# --- Phase 9 (09-02): misinformation fictional-ground-truth limitation note ----


class TestMisinformationLimitationNote:
    """MOD-12/D-02 (09-02-PLAN.md Task 3): the fictional-ground-truth caveat
    -- present exactly when `misinformation` is loaded, present even for a
    zero-finding (all-clean) run, byte-identical across runs, at a fixed
    position (after the excessive-agency note), and rendered in the
    Markdown report."""

    def test_present_when_misinformation_loaded(self):
        limitations = api_module._scan_limitations(["misinformation"], [])
        assert api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE in limitations

    def test_absent_when_misinformation_not_loaded(self):
        limitations = api_module._scan_limitations(["prompt_injection"], [])
        assert api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE not in limitations

    def test_present_even_with_a_completely_clean_case_log(self):
        """A zero-finding run must still be distinguishable from an
        untested one -- the note is keyed on `module_ids`, never on
        whether any finding was produced."""
        clean_case_log = [
            EvalResult(
                case_id="MISINFO-F01",
                verdict=Verdict.BLOCKED,
                confidence=0.8,
                evidence="clean refusal",
                detection_layer="judge",
            )
        ]
        limitations = api_module._scan_limitations(["misinformation"], clean_case_log)
        assert api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE in limitations

    def test_byte_identical_across_repeated_calls_with_note_at_fixed_position(self):
        """T-02-33: two `_scan_limitations()` calls over the same
        module_ids produce byte-identical lists, with the new note at a
        fixed position (last, after the excessive-agency note)."""
        module_ids = [
            "misinformation",
            "excessive_agency",
            "vector_embedding_weaknesses",
            "unbounded_consumption",
            "prompt_injection",
            "data_poisoning",
        ]
        first = api_module._scan_limitations(module_ids, [])
        second = api_module._scan_limitations(module_ids, [])

        assert first == second
        assert first[-1] == api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE
        assert first.index(api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE) < first.index(
            api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE
        )

    def test_both_new_modules_together_emit_both_notes_in_fixed_order(self):
        module_ids = ["misinformation", "excessive_agency"]
        limitations = api_module._scan_limitations(module_ids, [])
        assert limitations.index(api_module._EXCESSIVE_AGENCY_NO_ENFORCEMENT_NOTE) < limitations.index(
            api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE
        )

    async def test_run_scan_with_misinformation_loaded_includes_note(self, tmp_path, monkeypatch):
        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "misinformation"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["misinformation"])

        report = await api_module.run_scan(config, bypass_flag=True)

        assert api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE in report.limitations

    async def test_run_scan_without_misinformation_omits_note(self, tmp_path, monkeypatch):
        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "system_prompt_leakage"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["system_prompt_leakage"])

        report = await api_module.run_scan(config, bypass_flag=True)

        assert api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE not in report.limitations

    async def test_rendered_markdown_report_contains_fictional_ground_truth_note_text(
        self, tmp_path, monkeypatch
    ):
        from llmsec.reporting.markdown_reporter import MarkdownReporter

        module = _MockModule([("c1", Verdict.BLOCKED, "clean")])
        module.id = "misinformation"
        _patch_modules(monkeypatch, module)
        _patch_http_adapter(monkeypatch)
        config = _http_app_config(tmp_path, enabled_modules=["misinformation"])

        report = await api_module.run_scan(config, bypass_flag=True)

        md_path = await MarkdownReporter().write(report, tmp_path / "md")
        assert api_module._MISINFORMATION_FICTIONAL_GROUND_TRUTH_NOTE in md_path.read_text()
