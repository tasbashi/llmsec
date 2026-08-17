"""Tests for the `llmsec` Typer CLI (`scan`, `report`, `list-modules`) — CLI-01/02/03.

Uses `typer.testing.CliRunner` for in-process invocation of `llmsec.cli.app`.
`llmsec.cli.api.run_scan` is monkeypatched at the usage site (as imported
into `cli.py`, not the original `llmsec.api.run_scan` reference) per standard
`unittest.mock` patch-at-usage-site practice.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import yaml
from typer.testing import CliRunner

from llmsec.cli import app
from llmsec.models import ScanReport
from llmsec.plugins.registry import BUILTIN_MODULE_IDS, PluginRegistry

runner = CliRunner()

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_REPORT_FIXTURE = FIXTURES_DIR / "scan_report_sample.json"
SAMPLE_SCAN_ID = "sample-scan-001"


def _copy_sample_report(tmp_path: Path) -> Path:
    """Copy the sample fixture into a tmp_path-based output_dir following
    the `scan_<id>.json` naming convention `report_cmd` expects."""
    output_dir = tmp_path / "llmsec_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE_REPORT_FIXTURE, output_dir / f"scan_{SAMPLE_SCAN_ID}.json")
    return output_dir


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    config_path = tmp_path / "llmsec.config.yaml"
    config_path.write_text(yaml.safe_dump(data))
    return config_path


def _scan_yaml(tmp_path: Path, **overrides) -> Path:
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
    return _write_yaml(tmp_path, data)


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


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "scan" in result.output
    assert "report" in result.output
    assert "list-modules" in result.output


def test_scan_calls_run_scan_with_config_reflecting_yaml(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)

    result = runner.invoke(app, ["scan", "--config", str(yaml_path), "--yes-i-am-authorized"])

    assert result.exit_code == 0, result.output
    mock_run_scan.assert_awaited_once()
    called_config = mock_run_scan.await_args.args[0]
    assert called_config.target.type == "http_app"
    assert called_config.max_concurrency == 5
    assert mock_run_scan.await_args.kwargs["bypass_flag"] is True


def test_scan_max_concurrency_override_wins_over_yaml(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    mock_run_scan = AsyncMock(return_value=_fake_report())
    monkeypatch.setattr("llmsec.cli.api.run_scan", mock_run_scan)

    result = runner.invoke(
        app,
        ["scan", "--config", str(yaml_path), "--max-concurrency", "9", "--yes-i-am-authorized"],
    )

    assert result.exit_code == 0, result.output
    called_config = mock_run_scan.await_args.args[0]
    assert called_config.max_concurrency == 9


def test_scan_authorization_declined_is_clean_error_not_traceback(tmp_path, monkeypatch):
    yaml_path = _scan_yaml(tmp_path)
    # Non-interactive CliRunner stdin + no --yes-i-am-authorized + no env var
    # bypass means the REAL auth gate (inside the real api.run_scan) declines
    # before any adapter is ever constructed — nothing needs mocking here.
    monkeypatch.delenv("LLMSEC_AUTHORIZED", raising=False)

    result = runner.invoke(app, ["scan", "--config", str(yaml_path)])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "Refusing to run" in result.output or "authorization" in result.output.lower()


def test_report_markdown_regenerates_without_rescan(tmp_path):
    output_dir = _copy_sample_report(tmp_path)

    result = runner.invoke(
        app, ["report", SAMPLE_SCAN_ID, "--output-dir", str(output_dir), "--format", "markdown"]
    )

    assert result.exit_code == 0, result.output
    written_md = output_dir / f"scan_{SAMPLE_SCAN_ID}.md"
    assert written_md.exists()
    assert str(written_md) in result.output


def test_report_json_round_trip_regeneration_no_rescan(tmp_path):
    output_dir = _copy_sample_report(tmp_path)
    json_path = output_dir / f"scan_{SAMPLE_SCAN_ID}.json"
    original_bytes = json_path.read_bytes()

    result = runner.invoke(
        app, ["report", SAMPLE_SCAN_ID, "--output-dir", str(output_dir), "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    assert json_path.exists()
    # Re-written JSON round-trips the same scan_id and finding count — a
    # regeneration, not a re-scan (no adapter/target call occurs).
    reloaded = json.loads(json_path.read_text())
    original = json.loads(original_bytes)
    assert reloaded["scan_id"] == original["scan_id"] == SAMPLE_SCAN_ID
    assert len(reloaded["findings"]) == len(original["findings"])


def test_report_invalid_format_is_clean_error_not_unhandled_keyerror(tmp_path):
    """Regression test (WR-02): an unrecognized `--format` value must be
    rejected by Click's choice validation with a clean, non-zero exit —
    never an unhandled `KeyError` from indexing the format->reporter dict."""
    output_dir = _copy_sample_report(tmp_path)

    result = runner.invoke(
        app, ["report", SAMPLE_SCAN_ID, "--output-dir", str(output_dir), "--format", "yaml"]
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, KeyError)


def test_report_nonexistent_scan_id_is_clean_not_found_error(tmp_path):
    output_dir = _copy_sample_report(tmp_path)

    result = runner.invoke(app, ["report", "nonexistent-scan-id", "--output-dir", str(output_dir)])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "not found" in result.output.lower() or "no persisted scan" in result.output.lower()


def test_list_modules_shows_real_builtin_module_as_loaded(tmp_path, monkeypatch):
    # No --config passed and no llmsec.config.yaml in cwd -> config load
    # fails -> falls back to BUILTIN_MODULE_IDS, same default load_allowed()
    # would use, per plan intent.
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "system_prompt_leakage" in result.output
    assert "[loaded]" in result.output


def test_list_modules_shows_both_builtins_loaded(tmp_path, monkeypatch):
    """ROADMAP SC#4 user-visible half: `llmsec list-modules` surfaces every
    built-in module via real entry-point discovery (no monkeypatching),
    each independently marked `[loaded]` with no allowlist configured.
    The Prompt Injection module's `LLM01:2025` OWASP reference is asserted
    at the registry level in test_plugin_registry.py::
    test_prompt_injection_discoverable — `list_modules_cmd` itself does
    not print owasp_ref in its current output format (cli.py is out of
    this plan's files_modified scope).

    Count updated from 2 to 3 in 03-01 (Rule 1 auto-fix): registering the
    `pii_exfiltration` built-in (D-39) grows `BUILTIN_MODULE_IDS`, so a run
    with no allowlist configured now loads all three built-ins. Updated
    again from 3 to 4 in 04-01 (Rule 1 auto-fix): registering
    `insecure_output` (D-42) grows it further. Updated again from 4 to 5 in
    06-01 (Rule 1 auto-fix): registering `supply_chain` grows it further.
    Updated again from 5 to 6 in 06-05 (Rule 1 auto-fix): registering
    `data_poisoning` grows it further. Updated again from 6 to 7 in 07-01
    (Rule 1 auto-fix): registering `unbounded_consumption` grows it further.
    Updated again from 7 to 8 in 08-01 (Rule 1 auto-fix): registering
    `vector_embedding_weaknesses` grows it further. Updated again from 8 to
    9 in 08-04 (Rule 1 auto-fix): registering `excessive_agency` grows it
    further. Updated again from 9 to 10 in 09-01 (Rule 1 auto-fix):
    registering `misinformation` grows it further."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "system_prompt_leakage" in result.output
    assert "prompt_injection" in result.output
    assert "pii_exfiltration" in result.output
    assert "insecure_output" in result.output
    assert "supply_chain" in result.output
    assert "data_poisoning" in result.output
    assert "unbounded_consumption" in result.output
    assert "misinformation" in result.output
    assert result.output.count("[loaded]") == 10


def test_list_modules_empty_registry_prints_message_without_raising(monkeypatch):
    monkeypatch.setattr(PluginRegistry, "discover_all", lambda self: {})

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "No modules discovered." in result.output


def test_list_modules_deterministic_alphabetical_order_across_invocations(monkeypatch):
    fake_modules = {"zzz_module": object, "aaa_module": object, "mmm_module": object}
    monkeypatch.setattr(PluginRegistry, "discover_all", lambda self: fake_modules)

    result1 = runner.invoke(app, ["list-modules"])
    result2 = runner.invoke(app, ["list-modules"])

    assert result1.exit_code == 0, result1.output
    assert result2.exit_code == 0, result2.output

    def _order(output: str) -> list[str]:
        return [
            line.split()[0]
            for line in output.splitlines()
            if line.split() and line.split()[0] in fake_modules
        ]

    order1, order2 = _order(result1.output), _order(result2.output)
    assert order1 == ["aaa_module", "mmm_module", "zzz_module"]
    assert order1 == order2


# --- Phase 3 (03-01): pii_exfiltration discoverability + SC#4 selectability ---


def test_list_modules_shows_pii_exfiltration_discoverable_and_loaded(tmp_path, monkeypatch):
    """SC#4: `pii_exfiltration` is discoverable via the real installed
    entry points and, with no allowlist configured, shows as `[loaded]`
    alongside the other built-ins (eight total as of 08-01,
    vector_embedding_weaknesses)."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "pii_exfiltration" in result.output
    assert result.output.count("[loaded]") == 10


def test_list_modules_config_selecting_only_pii_exfiltration_loads_it_alone(tmp_path):
    """A config selecting only `pii_exfiltration` marks it `[loaded]` while
    the other two built-ins show `[not allowlisted]` — independent
    selectability (SC#4)."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["pii_exfiltration"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[loaded]" in lines["pii_exfiltration"]
    assert "[not allowlisted]" in lines["prompt_injection"]
    assert "[not allowlisted]" in lines["system_prompt_leakage"]


def test_list_modules_config_excluding_pii_exfiltration_omits_it(tmp_path):
    """Excluding `pii_exfiltration` from `enabled_modules` marks it
    `[not allowlisted]` while the configured modules load — independent
    exclusion (SC#4)."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["system_prompt_leakage", "prompt_injection"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[not allowlisted]" in lines["pii_exfiltration"]
    assert "[loaded]" in lines["prompt_injection"]
    assert "[loaded]" in lines["system_prompt_leakage"]


# --- Phase 4 (04-04): insecure_output discoverability + SC#4 selectability ---


def test_list_modules_shows_insecure_output_discoverable_and_loaded(tmp_path, monkeypatch):
    """SC#4: `insecure_output` is discoverable via the real installed entry
    points and, with no allowlist configured, shows as `[loaded]` alongside
    the other built-ins (eight total as of 08-01, vector_embedding_weaknesses)."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "insecure_output" in result.output
    assert result.output.count("[loaded]") == 10


def test_list_modules_config_selecting_only_insecure_output_loads_it_alone(tmp_path):
    """A config selecting only `insecure_output` marks it `[loaded]` while
    the other three built-ins show `[not allowlisted]` -- independent
    selectability (SC#4)."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["insecure_output"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[loaded]" in lines["insecure_output"]
    assert "[not allowlisted]" in lines["prompt_injection"]
    assert "[not allowlisted]" in lines["system_prompt_leakage"]
    assert "[not allowlisted]" in lines["pii_exfiltration"]


def test_list_modules_config_excluding_insecure_output_omits_it(tmp_path):
    """Excluding `insecure_output` from `enabled_modules` marks it
    `[not allowlisted]` while the configured modules load -- independent
    exclusion (SC#4)."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["system_prompt_leakage", "prompt_injection"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[not allowlisted]" in lines["insecure_output"]
    assert "[loaded]" in lines["prompt_injection"]
    assert "[loaded]" in lines["system_prompt_leakage"]


# --- Phase 6 (06-01): supply_chain discoverability + SC#4 selectability ------


def test_list_modules_shows_supply_chain_discoverable_and_loaded(tmp_path, monkeypatch):
    """SC#4: `supply_chain` is discoverable via the real installed entry
    points and, with no allowlist configured, shows as `[loaded]` alongside
    the other built-ins (eight total as of 08-01, vector_embedding_weaknesses).
    Its `LLM03:2025` OWASP reference is asserted at the registry level in
    test_plugin_registry.py::test_supply_chain_discoverable --
    `list_modules_cmd` does not print owasp_ref in its current output
    format (cli.py is out of this plan's files_modified scope, same
    established boundary as prompt_injection/pii_exfiltration/
    insecure_output's discoverability tests above)."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "supply_chain" in result.output
    assert result.output.count("[loaded]") == 10


def test_list_modules_config_selecting_only_supply_chain_loads_it_alone(tmp_path):
    """A config selecting only `supply_chain` marks it `[loaded]` while
    every other built-in shows `[not allowlisted]` -- independent
    selectability (SC#4)."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["supply_chain"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[loaded]" in lines["supply_chain"]
    assert "[not allowlisted]" in lines["pii_exfiltration"]
    assert "[not allowlisted]" in lines["insecure_output"]


# --- Phase 6 (06-05): data_poisoning discoverability + SC#4 selectability ---


def test_list_modules_shows_data_poisoning_discoverable_and_loaded(tmp_path, monkeypatch):
    """SC#4/MOD-07: `data_poisoning` is discoverable via the real installed
    entry points and, with no allowlist configured, shows as `[loaded]`
    alongside the other built-ins (eight total as of 08-01,
    vector_embedding_weaknesses). Its `LLM04:2025` OWASP reference is asserted at
    the registry level in
    test_plugin_registry.py::test_data_poisoning_discoverable --
    `list_modules_cmd` prints only `{module_id} {status}` (see
    `test_list_modules_shows_supply_chain_discoverable_and_loaded`'s
    docstring for this established boundary)."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "data_poisoning" in result.output
    assert result.output.count("[loaded]") == 10


def test_list_modules_config_selecting_only_data_poisoning_loads_it_alone(tmp_path):
    """A config selecting only `data_poisoning` marks it `[loaded]` while
    every other built-in shows `[not allowlisted]` -- independent
    selectability (SC#4)."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["data_poisoning"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[loaded]" in lines["data_poisoning"]
    assert "[not allowlisted]" in lines["supply_chain"]
    assert "[not allowlisted]" in lines["pii_exfiltration"]


# --- Phase 9 (09-03): misinformation discoverability + CLI-05 full ------
# selection matrix (all ten OWASP categories, alone/mixed/excluded) -------


def test_list_modules_shows_all_ten_owasp_categories(tmp_path, monkeypatch):
    """ROADMAP SC#3: with no config present (falls back to
    `BUILTIN_MODULE_IDS`), `list-modules` shows all ten OWASP LLM Top 10
    module ids. Iterates `BUILTIN_MODULE_IDS` itself rather than
    hard-coding ten literals, so the test cannot drift from the registry."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    missing = [module_id for module_id in BUILTIN_MODULE_IDS if module_id not in result.output]
    assert not missing, missing
    assert len(BUILTIN_MODULE_IDS) == 10
    assert result.output.count("[loaded]") == 10


def test_list_modules_shows_misinformation_discoverable_and_loaded(tmp_path, monkeypatch):
    """CLI-05/MOD-12: `misinformation` is discoverable via the real
    installed entry points and, with no allowlist configured, shows as
    `[loaded]` alongside the other nine built-ins. Its `LLM09:2025` OWASP
    reference is asserted at the registry level in
    test_plugin_registry.py::test_misinformation_discoverable --
    `list_modules_cmd` prints only `{module_id} {status}` (see
    `test_list_modules_shows_supply_chain_discoverable_and_loaded`'s
    docstring for this established boundary)."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "misinformation" in result.output
    assert result.output.count("[loaded]") == 10


def test_list_modules_config_selecting_only_misinformation_loads_it_alone(tmp_path):
    """A config selecting only `misinformation` marks it `[loaded]` while
    every other built-in shows `[not allowlisted]` -- independent
    selectability, the CLI-visible form of ROADMAP SC#2's "alone" case."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["misinformation"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[loaded]" in lines["misinformation"]
    assert "[not allowlisted]" in lines["prompt_injection"]
    assert "[not allowlisted]" in lines["system_prompt_leakage"]
    assert "[not allowlisted]" in lines["data_poisoning"]


def test_list_modules_config_mixing_misinformation_with_a_v1_module(tmp_path):
    """A config selecting `misinformation` plus the v1.0 `prompt_injection`
    module marks both `[loaded]` while the remaining v1.0 modules stay
    `[not allowlisted]` -- the CLI-visible form of ROADMAP SC#2's "mixed"
    case."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["misinformation", "prompt_injection"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[loaded]" in lines["misinformation"]
    assert "[loaded]" in lines["prompt_injection"]
    assert "[not allowlisted]" in lines["system_prompt_leakage"]
    assert "[not allowlisted]" in lines["pii_exfiltration"]


def test_list_modules_config_excluding_misinformation_omits_it(tmp_path):
    """Excluding `misinformation` from `enabled_modules` marks it
    `[not allowlisted]` while the configured modules load -- independent
    exclusion."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "target": {
                "type": "http_app",
                "method": "POST",
                "url": "http://localhost:8000/chat",
                "headers": {},
                "body_template": '{"message": "{{payload}}"}',
                "response_path": "response",
            },
            "enabled_modules": ["system_prompt_leakage", "prompt_injection"],
            "max_concurrency": 5,
        },
    )

    result = runner.invoke(app, ["list-modules", "--config", str(yaml_path)])

    assert result.exit_code == 0, result.output
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.split()}
    assert "[not allowlisted]" in lines["misinformation"]
    assert "[loaded]" in lines["prompt_injection"]
    assert "[loaded]" in lines["system_prompt_leakage"]


def test_list_modules_never_instantiates_a_module(tmp_path, monkeypatch):
    """D-10/T-01-19: `list_modules_cmd` calls ONLY `discover_all()`, never
    `load_allowed()` -- no module is ever instantiated to render the
    listing. Patching `load_allowed` to raise proves the command still
    succeeds, since it must never be called."""
    monkeypatch.chdir(tmp_path)

    def _raise(*args, **kwargs):
        raise AssertionError("load_allowed() must never be called by list-modules")

    monkeypatch.setattr(PluginRegistry, "load_allowed", _raise)

    result = runner.invoke(app, ["list-modules"])

    assert result.exit_code == 0, result.output
    assert "supply_chain" in result.output


def test_installed_help_smoke():
    """CORE-01 closing smoke test: the real, installed `llmsec` console
    script (declared in plan 01's pyproject.toml, unimplemented until this
    plan) now resolves and runs end to end as a subprocess."""
    result = subprocess.run(["llmsec", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
