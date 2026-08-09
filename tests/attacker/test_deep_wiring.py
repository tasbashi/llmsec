"""`--quick`/`--deep`/`--deep-profile` CLI wiring + `api.run_scan()`'s
deep-mode additive branch (05-03-PLAN.md Task 3).

CLI cases use `typer.testing.CliRunner`, mirroring `tests/test_cli.py`'s
`monkeypatch("llmsec.cli.api.run_scan", ...)` pattern. `run_scan()` cases
monkeypatch `llmsec.api.run_attacker_campaign` directly (the module-level
name `api.py` imports it under) so no attacker stack import/credential is
ever required to exercise this wiring.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import yaml
from typer.testing import CliRunner

import llmsec.api as api_module
from llmsec.attacker import AttackerExtraNotInstalled
from llmsec.attacker.runner import CampaignResult
from llmsec.attacker.state import VariantRecord
from llmsec.cli import app
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, ScanReport, TargetResponse, TestCase, Verdict

runner = CliRunner()


# --- CLI wiring ---------------------------------------------------------------


def _scan_yaml(tmp_path, **overrides):
    data = {
        "target": {
            "type": "http_app",
            "method": "POST",
            "url": "http://localhost:8000/chat",
            "headers": {},
            "body_template": '{"message": "{{payload}}"}',
            "response_path": "response",
        },
        "enabled_modules": ["system_prompt_leakage"],
        "max_concurrency": 5,
    }
    data.update(overrides)
    config_path = tmp_path / "llmsec.config.yaml"
    config_path.write_text(yaml.safe_dump(data))
    return config_path


def _fake_report() -> ScanReport:
    return ScanReport(
        scan_id="fake-scan-1",
        target_summary="http_app:POST http://localhost:8000/chat",
        module_ids=["system_prompt_leakage"],
        findings=[],
        case_log=[],
        started_at="2026-07-21T00:00:00Z",
        completed_at="2026-07-21T00:00:01Z",
    )


def test_scan_help_lists_deep_flags():
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "--deep" in result.output
    assert "--quick" in result.output
    assert "--deep-profile" in result.output


def test_scan_no_flags_attacker_absent_from_config(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)

    result = runner.invoke(app, ["scan", "--config", str(yaml_path), "--yes-i-am-authorized"])

    assert result.exit_code == 0, result.output
    called_config = mock_run_scan.await_args.args[0]
    assert called_config.attacker is None


def test_scan_quick_behaves_identically_to_no_flag(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)

    result = runner.invoke(
        app, ["scan", "--config", str(yaml_path), "--quick", "--yes-i-am-authorized"]
    )

    assert result.exit_code == 0, result.output
    called_config = mock_run_scan.await_args.args[0]
    # Behavioral equivalence, not object identity: an explicitly-passed
    # --quick now materializes AttackerConfig(enabled=False, ...) rather
    # than leaving `attacker` as `None` (see the CORE-02/D-93 fix below) --
    # both are "attacker did not run", which is the actual contract this
    # test name promises.
    assert called_config.attacker is None or called_config.attacker.enabled is False


def test_scan_quick_overrides_yaml_configured_attacker_enabled(tmp_path, monkeypatch):
    """CORE-02/D-93 regression: an explicitly-passed --quick must win over a
    YAML-configured `attacker.enabled: true`, exactly like --deep wins in the
    other direction. Found live in 05-11's operator verification -- --quick
    was validated against --deep but never actually forced `enabled: False`,
    so a configured `attacker:` block silently survived --quick."""
    yaml_path = _scan_yaml(
        tmp_path,
        attacker={"enabled": True, "model": "openai:gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    )
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)

    result = runner.invoke(
        app, ["scan", "--config", str(yaml_path), "--quick", "--yes-i-am-authorized"]
    )

    assert result.exit_code == 0, result.output
    called_config = mock_run_scan.await_args.args[0]
    assert called_config.attacker is None or called_config.attacker.enabled is False


def test_scan_deep_and_quick_together_exits_nonzero_before_load_config(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)
    mock_load_config = AsyncMock()
    monkeypatch.setattr("llmsec.cli.load_config", mock_load_config)

    result = runner.invoke(
        app, ["scan", "--config", str(yaml_path), "--deep", "--quick", "--yes-i-am-authorized"]
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "--deep" in result.output and "--quick" in result.output
    mock_load_config.assert_not_called()
    mock_run_scan.assert_not_awaited()


def test_scan_deep_profile_without_deep_exits_nonzero(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)

    result = runner.invoke(
        app,
        [
            "scan",
            "--config",
            str(yaml_path),
            "--deep-profile",
            "thorough",
            "--yes-i-am-authorized",
        ],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "deep mode" in result.output.lower()
    mock_run_scan.assert_not_awaited()


def test_scan_deep_without_extra_exits_with_install_hint_no_target_request(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)

    def _raise(*args, **kwargs):
        raise AttackerExtraNotInstalled(
            "--deep mode requires the 'langchain' module, which is not installed. "
            'Deep mode is unavailable and the scan did NOT fall back to static-only -- '
            'install the attacker stack with `pip install ".[deep]"`, then rerun with --deep.'
        )

    monkeypatch.setattr("llmsec.cli.require_deep_extra", _raise)

    result = runner.invoke(app, ["scan", "--config", str(yaml_path), "--deep", "--yes-i-am-authorized"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert 'pip install ".[deep]"' in result.output
    mock_run_scan.assert_not_awaited()


def test_scan_deep_sets_attacker_enabled_and_deep_profile_sets_profile(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)
    monkeypatch.setattr("llmsec.cli.require_deep_extra", lambda: None)

    result = runner.invoke(
        app,
        [
            "scan",
            "--config",
            str(yaml_path),
            "--deep",
            "--deep-profile",
            "thorough",
            "--yes-i-am-authorized",
        ],
    )

    assert result.exit_code == 0, result.output
    called_config = mock_run_scan.await_args.args[0]
    assert called_config.attacker.enabled is True
    assert called_config.attacker.profile == "thorough"


def test_scan_deep_without_deep_profile_preserves_yaml_profile(tmp_path, monkeypatch):
    """CORE-02: `--deep-profile` not passed must never clobber a
    YAML-configured `attacker.profile` back to the schema default."""
    yaml_path = _scan_yaml(
        tmp_path, attacker={"enabled": False, "profile": "thorough", "budget_usd": 3.0}
    )
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)
    monkeypatch.setattr("llmsec.cli.require_deep_extra", lambda: None)

    result = runner.invoke(app, ["scan", "--config", str(yaml_path), "--deep", "--yes-i-am-authorized"])

    assert result.exit_code == 0, result.output
    called_config = mock_run_scan.await_args.args[0]
    assert called_config.attacker.enabled is True
    assert called_config.attacker.profile == "thorough"
    assert called_config.attacker.budget_usd == 3.0


# --- api.run_scan() deep-mode branch -----------------------------------------


class _MockAdapter:
    def __init__(self) -> None:
        self.send = AsyncMock(side_effect=self._send)
        self.closed = False
        self.supports_system_prompt_override = False
        self.supports_multi_turn = False

    async def _send(self, case: TestCase) -> TargetResponse:
        return TargetResponse(case_id=case.case_id, raw_text=f"response-to-{case.case_id}", latency_ms=1.0)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


class _DeepModule:
    """`BaseModule`-shaped mock with `uses_attacker_llm=True`."""

    id = "deep_module"
    name = "Deep Module"
    owasp_ref = "LLM01:2025"
    uses_attacker_llm = True

    def __init__(self, cases: list[tuple[str, Verdict, str]]) -> None:
        self._cases = cases

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        for case_id, _verdict, _evidence in self._cases:
            yield TestCase(case_id=case_id, prompt=f"prompt-{case_id}", technique_id=case_id)

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        verdict, evidence = next((v, e) for cid, v, e in self._cases if cid == case.case_id)
        return EvalResult(
            case_id=case.case_id, verdict=verdict, confidence=0.9, evidence=evidence, detection_layer="regex"
        )


def _patch_http_adapter(monkeypatch: pytest.MonkeyPatch) -> list[_MockAdapter]:
    created: list[_MockAdapter] = []

    def _factory(*args, **kwargs) -> _MockAdapter:
        adapter = _MockAdapter()
        created.append(adapter)
        return adapter

    monkeypatch.setattr(api_module, "HttpAppAdapter", _factory)
    return created


def _patch_modules(monkeypatch: pytest.MonkeyPatch, module) -> None:
    monkeypatch.setattr(
        api_module.PluginRegistry,
        "load_allowed",
        lambda self, allowlist, module_config=None: {module.id: module},
    )


def _http_app_config(tmp_path, *, attacker=None) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(
            type="http_app",
            method="POST",
            url="http://localhost:8000/chat",
            headers={},
            body_template='{"message": "{{payload}}"}',
            response_path="response",
        ),
        enabled_modules=["deep_module"],
        max_concurrency=5,
        output_dir=str(tmp_path / "reports"),
        judge_model="openai/gpt-4o-mini",
        judge_api_key_env=None,
        attacker=attacker,
    )


async def test_run_scan_attacker_absent_never_calls_run_attacker_campaign(tmp_path, monkeypatch):
    module = _DeepModule([("c1", Verdict.BLOCKED, "refused")])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    mock_campaign = AsyncMock()
    monkeypatch.setattr(api_module, "run_attacker_campaign", mock_campaign)

    config = _http_app_config(tmp_path, attacker=None)
    await api_module.run_scan(config, bypass_flag=True)

    mock_campaign.assert_not_awaited()


async def test_run_scan_attacker_disabled_never_calls_run_attacker_campaign(tmp_path, monkeypatch):
    from llmsec.attacker.config import AttackerConfig

    module = _DeepModule([("c1", Verdict.BLOCKED, "refused")])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)
    mock_campaign = AsyncMock()
    monkeypatch.setattr(api_module, "run_attacker_campaign", mock_campaign)

    config = _http_app_config(tmp_path, attacker=AttackerConfig(enabled=False))
    await api_module.run_scan(config, bypass_flag=True)

    mock_campaign.assert_not_awaited()


async def test_run_scan_deep_enabled_calls_campaign_once_after_orchestrator_run(tmp_path, monkeypatch):
    from llmsec.attacker.config import AttackerConfig

    module = _DeepModule([("c1", Verdict.BLOCKED, "refused")])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)

    mutated_result = EvalResult(
        case_id="c1-mut-1", verdict=Verdict.FULL_COMPROMISE, confidence=0.9, evidence="complied", detection_layer="regex"
    )
    lineage_record = VariantRecord(
        payload="mutated",
        technique_family="instruction_override",
        parent_case_id="c1",
        parent_technique_id="c1",
        round=1,
        contributing_agent="mutator",
        variant_index=0,
    )
    mock_campaign = AsyncMock(
        return_value=CampaignResult(
            eval_results=[("deep_module", mutated_result)],
            lineage={"c1-mut-1": lineage_record},
        )
    )
    monkeypatch.setattr(api_module, "run_attacker_campaign", mock_campaign)

    config = _http_app_config(tmp_path, attacker=AttackerConfig(enabled=True, profile="light"))
    report = await api_module.run_scan(config, bypass_flag=True)

    mock_campaign.assert_awaited_once()
    assert report.case_log[-1].case_id == "c1-mut-1"


async def test_run_scan_deep_mode_findings_carry_lineage_fields(tmp_path, monkeypatch):
    from llmsec.attacker.config import AttackerConfig

    module = _DeepModule([("c1", Verdict.BLOCKED, "refused")])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)

    mutated_result = EvalResult(
        case_id="c1-mut-1", verdict=Verdict.FULL_COMPROMISE, confidence=0.9, evidence="complied", detection_layer="regex"
    )
    lineage_record = VariantRecord(
        payload="mutated",
        technique_family="instruction_override",
        parent_case_id="c1",
        parent_technique_id="c1",
        round=1,
        contributing_agent="mutator",
        variant_index=0,
    )
    mock_campaign = AsyncMock(
        return_value=CampaignResult(
            eval_results=[("deep_module", mutated_result)],
            lineage={"c1-mut-1": lineage_record},
        )
    )
    monkeypatch.setattr(api_module, "run_attacker_campaign", mock_campaign)

    config = _http_app_config(tmp_path, attacker=AttackerConfig(enabled=True, profile="light"))
    report = await api_module.run_scan(config, bypass_flag=True)

    deep_finding = next(f for f in report.findings if f.case_id == "c1-mut-1")
    assert deep_finding.parent_case_id == "c1"
    assert deep_finding.parent_technique_id == "c1"
    assert deep_finding.round == 1
    assert deep_finding.contributing_agent == "mutator"

    # A static finding (if it produced one) carries no lineage.
    static_findings = [f for f in report.findings if f.case_id == "c1"]
    for finding in static_findings:
        assert finding.parent_case_id is None


async def test_run_scan_deep_mode_exception_falls_back_to_static_report_with_limitation(
    tmp_path, monkeypatch
):
    from llmsec.attacker.config import AttackerConfig

    module = _DeepModule([("c1", Verdict.FULL_COMPROMISE, "leaked it all")])
    _patch_modules(monkeypatch, module)
    _patch_http_adapter(monkeypatch)

    mock_campaign = AsyncMock(side_effect=RuntimeError("attacker stack exploded"))
    monkeypatch.setattr(api_module, "run_attacker_campaign", mock_campaign)

    config = _http_app_config(tmp_path, attacker=AttackerConfig(enabled=True, profile="light"))
    report = await api_module.run_scan(config, bypass_flag=True)

    # Static results are still intact.
    assert len(report.case_log) == 1
    assert report.findings[0].case_id == "c1"
    # A limitation entry discloses the deep-mode failure.
    assert any("deep" in note.lower() and "fail" in note.lower() for note in report.limitations)
