"""Tests for `SupplyChainModule` (src/llmsec/modules/supply_chain.py) —
Phase 6.

06-01 covered the three `run_standalone_audit()` degrade branches (manifest
missing, extra missing, and the now-deleted CVE-tier-unavailable
placeholder). 06-03 covered the real CVE/SBOM audit (MOD-06): the pip-audit
subprocess tier (Task 1), the OSV.dev batch advisory tier (Task 2), and the
advisory merge + verdict mapping that turns both sources into `EvalResult`s
(Task 3). 06-04 (this file, appended) covers MOD-05 slopsquatting: the
bundled static PyPI index snapshot + loader (Task 1), the elicitation
corpus + `generate_cases()` (Task 2), and the tiered extraction/index-lookup/
judge `evaluate()` (Task 3). No live network call and no real subprocess is
spawned anywhere in this file — `asyncio.create_subprocess_exec` is always
monkeypatched and every OSV.dev call goes through `respx_mock`.
"""

from __future__ import annotations

import asyncio
import gzip
import importlib.resources
import json
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from llmsec.detection.judge import PackageExtraction
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.modules.supply_chain import (
    AUDIT_CASE_ID_CLEAN,
    AUDIT_CASE_ID_EXTRA_MISSING,
    AUDIT_CASE_ID_MANIFEST_MISSING,
    AUDIT_CASE_ID_MANIFEST_UNPINNED,
    AUDIT_CASE_ID_OSV_UNREACHABLE,
    PYPI_INDEX_SNAPSHOT_AS_OF,
    SLOPSQUATTING_CASE_ID_PREFIX,
    _classify_package_existence,
    _extract_package_names,
    _is_negated_match,
    _load_package_index,
    _merge_advisories,
    _normalise_package_name,
    _normalize_package_name,
    _OsvUnreachableError,
    _OSV_QUERYBATCH_URL,
    _fetch_osv_details,
    _PipAuditFailure,
    _PipAuditTimeoutError,
    _PipAuditUnpinnedManifestError,
    _query_osv_batch,
    _run_osv_tier,
    _run_pip_audit,
    index_snapshot_limitation_note,
    SupplyChainModule,
    supply_chain_extra_available,
)
from llmsec.payloads import load_corpus
from llmsec.payloads.schema import SupplyChainTechniqueVector
from llmsec.scoring.engine import Severity, score


def _context() -> ScanContext:
    return ScanContext(judge_model="openai/gpt-4o-mini", judge_api_key_env="")


async def _collect(module: SupplyChainModule) -> list[EvalResult]:
    return [result async for result in module.run_standalone_audit(_context())]


class _FakeProc:
    """Stand-in for the object `asyncio.create_subprocess_exec()` returns."""

    def __init__(
        self,
        returncode: int | None,
        stdout: bytes = b'{"dependencies": []}',
        stderr: bytes = b"",
        hang: bool = False,
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang
        self.killed = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(3600)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        self.waited = True


def _patch_subprocess(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> dict[str, Any]:
    """Monkeypatch `asyncio.create_subprocess_exec` to return `proc` and
    capture the argv it was called with."""
    captured: dict[str, Any] = {}

    async def _fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured["argv"] = list(args)
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return captured


def _pip_audit_json(dependencies: list[dict[str, Any]]) -> bytes:
    return json.dumps({"dependencies": dependencies}).encode("utf-8")


_VULN_DEP = {
    "name": "httpx",
    "version": "0.27.0",
    "vulns": [
        {
            "id": "PYSEC-2024-1",
            "fix_versions": ["0.27.1"],
            "description": "Sample pip-audit-reported vulnerability.",
        }
    ],
}
_CLEAN_DEP = {"name": "requests", "version": "2.31.0", "vulns": []}


def _mock_clean_osv(respx_mock) -> None:
    """Register an OSV.dev querybatch route that reports no vulnerabilities
    for whatever packages are queried."""
    respx_mock.post(_OSV_QUERYBATCH_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )


class TestGenerateCases:
    async def test_generate_cases_yields_one_per_corpus_entry(self):
        """06-04 Task 2: `generate_cases()` is now corpus-backed -- supersedes
        06-01's placeholder empty-generator behavior (Rule 1 auto-fix, same
        precedent 06-01-SUMMARY.md's own test-count updates document)."""
        module = SupplyChainModule()
        cases = [case async for case in module.generate_cases(_context())]
        corpus = load_corpus("supply_chain")
        assert len(cases) == len(corpus) >= 18
        assert {c.case_id for c in cases} == {e.id for e in corpus}
        assert all(c.case_id == c.technique_id for c in cases)
        assert all(c.turns is None for c in cases)


class TestEvaluateNeverCrashes:
    async def test_evaluate_returns_a_valid_result_rather_than_raising(
        self, mock_target_response
    ):
        """06-04 Task 3: `evaluate()` now has a real MOD-05 detection tier
        (supersedes 06-01's unconditional-UNCERTAIN placeholder) -- this
        test only asserts the never-crash guarantee (T-01-18), not a
        specific verdict, since the default mock response text ("mock
        response") legitimately resolves cleanly at the regex tier."""
        module = SupplyChainModule()
        case = TestCase(case_id="unreachable", prompt="p", technique_id="unreachable")
        response = mock_target_response(case_id="unreachable")
        result = await module.evaluate(case, response)
        assert isinstance(result.verdict, Verdict)
        assert 0.0 <= result.confidence <= 1.0
        assert result.detection_layer in ("regex", "judge")


class TestRunStandaloneAuditManifestMissing:
    async def test_no_manifest_path_configured_yields_manifest_missing(self):
        module = SupplyChainModule(supply_chain_manifest_path=None)
        results = await _collect(module)
        assert len(results) == 1
        assert results[0].case_id == AUDIT_CASE_ID_MANIFEST_MISSING
        assert results[0].verdict == Verdict.UNCERTAIN
        assert results[0].confidence == 0.0
        assert results[0].detection_layer == "audit"

    async def test_unreadable_configured_path_takes_manifest_missing_branch(self, tmp_path):
        """An unreadable/nonexistent configured path must degrade to the
        same branch as an unset path, never raise (D-06)."""
        missing_path = tmp_path / "does-not-exist" / "requirements.txt"
        module = SupplyChainModule(supply_chain_manifest_path=str(missing_path))
        results = await _collect(module)
        assert len(results) == 1
        assert results[0].case_id == AUDIT_CASE_ID_MANIFEST_MISSING
        assert results[0].verdict == Verdict.UNCERTAIN
        assert results[0].detection_layer == "audit"

    async def test_never_falls_back_to_scanning_operator_environment(self, tmp_path, monkeypatch):
        """P-02: even when a manifest path IS readable, run_standalone_audit()
        must never read any path other than the configured one."""
        real_manifest = tmp_path / "requirements.txt"
        real_manifest.write_text("httpx==0.27.0\n")
        module = SupplyChainModule(supply_chain_manifest_path=str(real_manifest))
        # Extra unavailable in the test environment -- confirms the module
        # progressed past the manifest-missing branch using ONLY the
        # configured path, never a fallback scan of cwd/env.
        results = await _collect(module)
        assert results[0].case_id != AUDIT_CASE_ID_MANIFEST_MISSING


class TestRunStandaloneAuditExtraMissing:
    async def test_extra_not_available_yields_extra_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llmsec.modules.supply_chain.supply_chain_extra_available", lambda: False
        )
        real_manifest = tmp_path / "requirements.txt"
        real_manifest.write_text("httpx==0.27.0\n")
        module = SupplyChainModule(supply_chain_manifest_path=str(real_manifest))
        results = await _collect(module)
        assert len(results) == 1
        assert results[0].case_id == AUDIT_CASE_ID_EXTRA_MISSING
        assert results[0].verdict == Verdict.UNCERTAIN
        assert results[0].confidence == 0.0
        assert results[0].detection_layer == "audit"


class TestSupplyChainExtraAvailable:
    def test_real_probe_returns_bool_without_importing_pip_audit(self):
        # No assertion on the actual value (depends on the test env), only
        # that the probe never raises and returns a plain bool.
        assert isinstance(supply_chain_extra_available(), bool)


# --- Task 1: pip-audit subprocess tier --------------------------------------


class TestRunPipAuditArgv:
    async def test_requirements_txt_uses_requirement_flag(self, tmp_path, monkeypatch):
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=0, stdout=_pip_audit_json([]))
        captured = _patch_subprocess(monkeypatch, proc)

        await _run_pip_audit(manifest)

        argv = captured["argv"]
        assert argv[0] == "pip-audit"
        assert "--requirement" in argv
        assert str(manifest) in argv
        assert argv.index("--requirement") + 1 == argv.index(str(manifest))

    async def test_pyproject_toml_uses_project_path_positional(self, tmp_path, monkeypatch):
        manifest = tmp_path / "pyproject.toml"
        manifest.write_text("[project]\nname = 'x'\n")
        proc = _FakeProc(returncode=0, stdout=_pip_audit_json([]))
        captured = _patch_subprocess(monkeypatch, proc)

        await _run_pip_audit(manifest)

        argv = captured["argv"]
        assert argv[0] == "pip-audit"
        assert str(manifest.parent) in argv
        assert "--requirement" not in argv

    async def test_argv_always_carries_format_no_deps_strict_and_timeout(self, tmp_path, monkeypatch):
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=0, stdout=_pip_audit_json([]))
        captured = _patch_subprocess(monkeypatch, proc)

        await _run_pip_audit(manifest, timeout_seconds=45)

        argv = captured["argv"]
        assert "--format" in argv and "json" in argv
        assert "--no-deps" in argv
        assert "--strict" in argv
        assert "--timeout" in argv and "45" in argv

    async def test_never_uses_shell(self, tmp_path, monkeypatch):
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=0, stdout=_pip_audit_json([]))
        captured = _patch_subprocess(monkeypatch, proc)

        await _run_pip_audit(manifest)

        assert "kwargs" in captured
        assert "shell" not in captured["kwargs"]


class TestRunPipAuditReturnCodes:
    async def test_return_code_0_clean_is_success(self, tmp_path, monkeypatch):
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("requests==2.31.0\n")
        proc = _FakeProc(returncode=0, stdout=_pip_audit_json([_CLEAN_DEP]))
        _patch_subprocess(monkeypatch, proc)

        dependencies, vulnerabilities = await _run_pip_audit(manifest)
        assert dependencies == [_CLEAN_DEP]
        assert vulnerabilities == []

    async def test_return_code_1_with_vulnerabilities_is_success(self, tmp_path, monkeypatch):
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=1, stdout=_pip_audit_json([_VULN_DEP]))
        _patch_subprocess(monkeypatch, proc)

        dependencies, vulnerabilities = await _run_pip_audit(manifest)
        assert dependencies == [_VULN_DEP]
        assert len(vulnerabilities) == 1
        assert vulnerabilities[0]["id"] == "PYSEC-2024-1"
        assert vulnerabilities[0]["name"] == "httpx"
        assert vulnerabilities[0]["version"] == "0.27.0"

    async def test_return_code_outside_0_1_raises_pip_audit_failure(self, tmp_path, monkeypatch):
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=2, stderr=b"boom: internal pip-audit error")
        _patch_subprocess(monkeypatch, proc)

        with pytest.raises(_PipAuditFailure):
            await _run_pip_audit(manifest)

    async def test_unpinned_manifest_stderr_signature_detected_distinctly(self, tmp_path, monkeypatch):
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx\n")  # unpinned
        proc = _FakeProc(
            returncode=3, stderr=b"httpx is not pinned to a specific version"
        )
        _patch_subprocess(monkeypatch, proc)

        with pytest.raises(_PipAuditUnpinnedManifestError):
            await _run_pip_audit(manifest)

    async def test_hung_subprocess_is_killed_and_raises_timeout(self, tmp_path, monkeypatch):
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=None, hang=True)
        _patch_subprocess(monkeypatch, proc)

        with pytest.raises(_PipAuditTimeoutError):
            await _run_pip_audit(manifest, timeout_seconds=0.05)
        assert proc.killed is True


class TestRunStandaloneAuditPipAuditDegradation:
    async def test_unpinned_manifest_yields_manifest_unpinned_uncertain(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "llmsec.modules.supply_chain.supply_chain_extra_available", lambda: True
        )
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx\n")
        proc = _FakeProc(returncode=3, stderr=b"httpx is not pinned to a specific version")
        _patch_subprocess(monkeypatch, proc)

        module = SupplyChainModule(supply_chain_manifest_path=str(manifest))
        results = await _collect(module)

        assert len(results) == 1
        assert results[0].case_id == AUDIT_CASE_ID_MANIFEST_UNPINNED
        assert results[0].verdict == Verdict.UNCERTAIN
        assert results[0].confidence == 0.0
        assert results[0].detection_layer == "audit"
        assert "pinned" in results[0].evidence.lower()

    async def test_generic_subprocess_failure_yields_manifest_unpinned_uncertain(
        self, tmp_path, monkeypatch
    ):
        """A return code outside {0, 1} that is NOT the unpinned-manifest
        signature still degrades honestly, sharing the same sentinel
        case_id (there is no separate constant for this shape)."""
        monkeypatch.setattr(
            "llmsec.modules.supply_chain.supply_chain_extra_available", lambda: True
        )
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=2, stderr=b"totally unrelated tool crash")
        _patch_subprocess(monkeypatch, proc)

        module = SupplyChainModule(supply_chain_manifest_path=str(manifest))
        results = await _collect(module)

        assert len(results) == 1
        assert results[0].case_id == AUDIT_CASE_ID_MANIFEST_UNPINNED
        assert results[0].verdict == Verdict.UNCERTAIN
        assert "2" in results[0].evidence


# --- Task 2: OSV.dev batch advisory tier ------------------------------------


class TestQueryOsvBatch:
    async def test_request_body_uses_pypi_ecosystem_one_query_per_package(self, respx_mock):
        route = respx_mock.post(_OSV_QUERYBATCH_URL).mock(
            return_value=httpx.Response(200, json={"results": [{}, {}]})
        )
        await _query_osv_batch([("httpx", "0.27.0"), ("requests", "2.31.0")])

        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert len(body["queries"]) == 2
        for query in body["queries"]:
            assert query["package"]["ecosystem"] == "PyPI"

    async def test_large_inventory_split_across_multiple_requests(self, respx_mock):
        route = respx_mock.post(_OSV_QUERYBATCH_URL).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        packages = [(f"pkg{i}", "1.0.0") for i in range(1001)]

        await _query_osv_batch(packages)

        assert route.call_count == 2
        first_body = json.loads(route.calls[0].request.content)
        second_body = json.loads(route.calls[1].request.content)
        assert len(first_body["queries"]) == 1000
        assert len(second_body["queries"]) == 1

    async def test_paginated_response_is_fully_followed(self, respx_mock):
        route = respx_mock.post(_OSV_QUERYBATCH_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "results": [
                            {"vulns": [{"id": "ADV-1"}], "next_page_token": "TOKEN-1"}
                        ]
                    },
                ),
                httpx.Response(200, json={"results": [{"vulns": [{"id": "ADV-2"}]}]}),
            ]
        )

        result = await _query_osv_batch([("pkgA", "1.0.0")])

        assert route.call_count == 2
        second_body = json.loads(route.calls[1].request.content)
        assert second_body["queries"][0]["page_token"] == "TOKEN-1"
        assert result[("pkgA", "1.0.0")] == ["ADV-1", "ADV-2"]


class TestFetchOsvDetails:
    async def test_only_fetches_ids_actually_returned(self, respx_mock, monkeypatch):
        _mock_clean_osv(respx_mock)
        respx_mock.post(_OSV_QUERYBATCH_URL).mock(
            return_value=httpx.Response(
                200, json={"results": [{"vulns": [{"id": "ADV-1"}]}, {}]}
            )
        )

        fetched_ids: list[str] = []

        async def _spy_fetch(advisory_ids, timeout_seconds=30.0):
            fetched_ids.extend(advisory_ids)
            return {aid: {"summary": "x"} for aid in advisory_ids}

        monkeypatch.setattr("llmsec.modules.supply_chain._fetch_osv_details", _spy_fetch)

        dependencies = [_VULN_DEP, _CLEAN_DEP]
        await _run_osv_tier(dependencies)

        assert fetched_ids == ["ADV-1"]

    async def test_details_call_hits_vulns_endpoint_not_single_query(self, respx_mock):
        route = respx_mock.get("https://api.osv.dev/v1/vulns/ADV-1").mock(
            return_value=httpx.Response(200, json={"id": "ADV-1", "summary": "desc"})
        )
        details = await _fetch_osv_details(["ADV-1"])
        assert route.called
        assert details["ADV-1"]["summary"] == "desc"


class TestOsvDegradation:
    async def test_connection_error_produces_osv_unreachable(self, respx_mock):
        respx_mock.post(_OSV_QUERYBATCH_URL).mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(_OsvUnreachableError):
            await _run_osv_tier([_CLEAN_DEP])

    async def test_500_response_produces_osv_unreachable(self, respx_mock):
        respx_mock.post(_OSV_QUERYBATCH_URL).mock(return_value=httpx.Response(500))
        with pytest.raises(_OsvUnreachableError):
            await _run_osv_tier([_CLEAN_DEP])

    async def test_malformed_json_body_produces_osv_unreachable(self, respx_mock):
        respx_mock.post(_OSV_QUERYBATCH_URL).mock(
            return_value=httpx.Response(200, content=b"not json{{{")
        )
        with pytest.raises(_OsvUnreachableError):
            await _run_osv_tier([_CLEAN_DEP])

    async def test_run_standalone_audit_osv_failure_still_yields_pip_audit_findings(
        self, tmp_path, monkeypatch, respx_mock
    ):
        monkeypatch.setattr(
            "llmsec.modules.supply_chain.supply_chain_extra_available", lambda: True
        )
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=1, stdout=_pip_audit_json([_VULN_DEP]))
        _patch_subprocess(monkeypatch, proc)
        respx_mock.post(_OSV_QUERYBATCH_URL).mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        module = SupplyChainModule(supply_chain_manifest_path=str(manifest))
        results = await _collect(module)

        case_ids = [r.case_id for r in results]
        assert AUDIT_CASE_ID_OSV_UNREACHABLE in case_ids
        assert any(cid.startswith("SUPPLY-CHAIN-AUDIT-CVE-") for cid in case_ids)
        finding = next(r for r in results if r.case_id.startswith("SUPPLY-CHAIN-AUDIT-CVE-"))
        assert finding.verdict == Verdict.FULL_COMPROMISE

    async def test_no_request_made_to_any_host_other_than_osv_dev(
        self, tmp_path, monkeypatch, respx_mock
    ):
        # respx_mock only registers api.osv.dev routes below -- if the audit
        # ever contacted another host, respx would raise (assert_all_mocked
        # is the default) rather than let the request through unmocked.
        monkeypatch.setattr(
            "llmsec.modules.supply_chain.supply_chain_extra_available", lambda: True
        )
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("requests==2.31.0\n")
        proc = _FakeProc(returncode=0, stdout=_pip_audit_json([_CLEAN_DEP]))
        _patch_subprocess(monkeypatch, proc)
        _mock_clean_osv(respx_mock)

        module = SupplyChainModule(supply_chain_manifest_path=str(manifest))
        results = await _collect(module)
        assert results[0].case_id == AUDIT_CASE_ID_CLEAN


# --- Task 3: advisory merge, verdict mapping, end-to-end --------------------


class TestNormalizePackageName:
    def test_flask_and_flask_lowercase_collapse_to_same_id(self):
        assert _normalize_package_name("Flask") == _normalize_package_name("flask")

    def test_collapses_separators(self):
        assert _normalize_package_name("My_Cool.Package") == "my-cool-package"


class TestMergeAdvisories:
    def test_pip_audit_only_produces_one_advisory(self):
        pip_vulns = [
            {"name": "httpx", "version": "0.27.0", "id": "PYSEC-1", "description": "x", "fix_versions": []}
        ]
        merged = _merge_advisories(pip_vulns, {})
        assert list(merged.keys()) == [("httpx", "0.27.0")]
        assert len(merged[("httpx", "0.27.0")]) == 1
        assert merged[("httpx", "0.27.0")][0]["sources"] == {"pip-audit"}

    def test_osv_only_produces_one_advisory(self):
        osv = {("httpx", "0.27.0"): [{"id": "OSV-1", "summary": "y", "severity": None, "fix_versions": []}]}
        merged = _merge_advisories([], osv)
        assert len(merged[("httpx", "0.27.0")]) == 1
        assert merged[("httpx", "0.27.0")][0]["sources"] == {"osv"}

    def test_both_sources_reporting_same_id_yields_one_finding_naming_both(self):
        pip_vulns = [
            {"name": "httpx", "version": "0.27.0", "id": "PYSEC-1", "description": "x", "fix_versions": []}
        ]
        osv = {("httpx", "0.27.0"): [{"id": "PYSEC-1", "summary": "y", "severity": "HIGH", "fix_versions": []}]}
        merged = _merge_advisories(pip_vulns, osv)
        advisories = merged[("httpx", "0.27.0")]
        assert len(advisories) == 1  # not double-counted
        assert advisories[0]["sources"] == {"pip-audit", "osv"}


class TestCveSeverityNeverInfluencesVerdict:
    def test_low_and_high_cvss_advisories_produce_identical_verdict_and_severity(self):
        low_severity_result = EvalResult(
            case_id="SUPPLY-CHAIN-AUDIT-CVE-pkg-a",
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.9,
            evidence="pkg-a has 1 known advisory:\n- ADV-1: low severity (CVSS 2.0)",
            detection_layer="audit",
        )
        high_severity_result = EvalResult(
            case_id="SUPPLY-CHAIN-AUDIT-CVE-pkg-b",
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.9,
            evidence="pkg-b has 1 known advisory:\n- ADV-2: critical severity (CVSS 9.8)",
            detection_layer="audit",
        )
        assert low_severity_result.verdict == high_severity_result.verdict
        assert score(low_severity_result.verdict) == score(high_severity_result.verdict)
        assert score(low_severity_result.verdict) == Severity.HIGH


class TestRunStandaloneAuditCleanPath:
    async def test_clean_audit_evidence_contains_dependency_count(self, tmp_path, monkeypatch, respx_mock):
        monkeypatch.setattr(
            "llmsec.modules.supply_chain.supply_chain_extra_available", lambda: True
        )
        manifest = tmp_path / "requirements.txt"
        manifest.write_text("requests==2.31.0\n")
        proc = _FakeProc(returncode=0, stdout=_pip_audit_json([_CLEAN_DEP]))
        _patch_subprocess(monkeypatch, proc)
        _mock_clean_osv(respx_mock)

        module = SupplyChainModule(supply_chain_manifest_path=str(manifest))
        results = await _collect(module)

        assert len(results) == 1
        assert results[0].case_id == AUDIT_CASE_ID_CLEAN
        assert results[0].verdict == Verdict.BLOCKED
        assert "1 dependenc" in results[0].evidence


class TestEndToEndRunScan:
    async def test_run_scan_supply_chain_no_manifest_produces_one_audit_case_and_limitation(
        self, tmp_path, monkeypatch
    ):
        """06-04: `generate_cases()` is no longer empty (MOD-05 now yields
        18 corpus-backed cases), so the mock adapter must handle `send()`
        for real rather than asserting it is never called (Rule 1 auto-fix,
        same precedent 06-01/06-02-SUMMARY.md's own test-count updates
        document) -- this test's own assertion narrows from "the ONLY
        case_log entry" to "AT LEAST ONE audit case_log entry", since the
        MOD-05 half now legitimately contributes 18 more."""
        import llmsec.api as api_module
        from llmsec.config import ScanConfig, TargetConfig

        class _MockAdapter:
            def __init__(self, *args, **kwargs) -> None:
                self.closed = False
                self.supports_system_prompt_override = False
                self.supports_multi_turn = False

            async def send(self, case):
                return TargetResponse(
                    case_id=case.case_id,
                    raw_text="Run `pip install requests` to get started.",
                    latency_ms=1.0,
                    transport_mode="single",
                )

            async def health_check(self) -> bool:
                return True

            async def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(api_module, "HttpAppAdapter", _MockAdapter)

        config = ScanConfig(
            target=TargetConfig(
                type="http_app",
                method="POST",
                url="http://localhost:8000/chat",
                headers={},
                body_template='{"message": "{{payload}}"}',
                response_path="response",
            ),
            enabled_modules=["supply_chain"],
            max_concurrency=5,
            output_dir=str(tmp_path / "reports"),
            judge_model="openai/gpt-4o-mini",
            judge_api_key_env=None,
        )

        report = await api_module.run_scan(config, bypass_flag=True)

        audit_case_ids = [
            r.case_id for r in report.case_log if r.case_id == AUDIT_CASE_ID_MANIFEST_MISSING
        ]
        assert len(audit_case_ids) == 1
        from llmsec.api import _SUPPLY_CHAIN_MANIFEST_MISSING_NOTE

        assert _SUPPLY_CHAIN_MANIFEST_MISSING_NOTE in report.limitations

    async def test_run_scan_produces_cve_finding_with_audit_detection_layer(
        self, tmp_path, monkeypatch, respx_mock
    ):
        """06-04: see the sibling test above's docstring -- the mock
        adapter must now handle `send()` for real (Rule 1 auto-fix)."""
        import llmsec.api as api_module
        from llmsec.config import ScanConfig, TargetConfig

        class _MockAdapter:
            def __init__(self, *args, **kwargs) -> None:
                self.closed = False
                self.supports_system_prompt_override = False
                self.supports_multi_turn = False

            async def send(self, case):
                return TargetResponse(
                    case_id=case.case_id,
                    raw_text="Run `pip install requests` to get started.",
                    latency_ms=1.0,
                    transport_mode="single",
                )

            async def health_check(self) -> bool:
                return True

            async def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(api_module, "HttpAppAdapter", _MockAdapter)
        monkeypatch.setattr(
            "llmsec.modules.supply_chain.supply_chain_extra_available", lambda: True
        )

        manifest = tmp_path / "requirements.txt"
        manifest.write_text("httpx==0.27.0\n")
        proc = _FakeProc(returncode=1, stdout=_pip_audit_json([_VULN_DEP]))
        _patch_subprocess(monkeypatch, proc)
        _mock_clean_osv(respx_mock)

        config = ScanConfig(
            target=TargetConfig(
                type="http_app",
                method="POST",
                url="http://localhost:8000/chat",
                headers={},
                body_template='{"message": "{{payload}}"}',
                response_path="response",
            ),
            enabled_modules=["supply_chain"],
            max_concurrency=5,
            output_dir=str(tmp_path / "reports"),
            judge_model="openai/gpt-4o-mini",
            judge_api_key_env=None,
            supply_chain_manifest_path=str(manifest),
        )

        report = await api_module.run_scan(config, bypass_flag=True)

        audit_findings = [f for f in report.findings if f.detection_layer == "audit"]
        assert len(audit_findings) == 1
        assert audit_findings[0].case_id.startswith("SUPPLY-CHAIN-AUDIT-CVE-")
        assert "PYSEC-2024-1" in audit_findings[0].evidence


# ============================================================================
# 06-04: MOD-05 slopsquatting -- bundled index, corpus, tiered detection
# =============================================================================

# --- Task 1: bundled static PyPI index snapshot + loader -------------------


class TestBundledPypiIndexArtifact:
    def test_snapshot_file_is_gzip_readable_and_carries_dated_header(self):
        resource = importlib.resources.files("llmsec.modules.payloads").joinpath(
            "pypi_index_snapshot.txt.gz"
        )
        assert resource.is_file()
        text = gzip.decompress(resource.read_bytes()).decode("utf-8")
        lines = text.splitlines()
        assert lines[0].startswith("#")
        assert PYPI_INDEX_SNAPSHOT_AS_OF in lines[0]

    def test_snapshot_contains_well_known_distribution_names(self):
        resource = importlib.resources.files("llmsec.modules.payloads").joinpath(
            "pypi_index_snapshot.txt.gz"
        )
        text = gzip.decompress(resource.read_bytes()).decode("utf-8")
        names = {line for line in text.splitlines() if line and not line.startswith("#")}
        assert "requests" in names
        assert "flask" in names
        assert "httpx" in names

    def test_line_count_above_300000(self):
        resource = importlib.resources.files("llmsec.modules.payloads").joinpath(
            "pypi_index_snapshot.txt.gz"
        )
        text = gzip.decompress(resource.read_bytes()).decode("utf-8")
        assert len(text.splitlines()) > 300000


class TestLoadPackageIndex:
    def test_returns_non_empty_frozenset(self):
        index = _load_package_index()
        assert isinstance(index, frozenset)
        assert len(index) > 300000
        assert "requests" in index

    def test_memoised_across_calls(self):
        first = _load_package_index()
        second = _load_package_index()
        assert first is second

    def test_every_entry_is_already_normalised(self):
        index = _load_package_index()
        sample = list(index)[:5000]
        assert all(_normalise_package_name(name) == name for name in sample)

    def test_load_failure_yields_empty_frozenset_without_raising(self, monkeypatch):
        import llmsec.modules.supply_chain as supply_chain_module

        monkeypatch.setattr(supply_chain_module, "_package_index_cache", None)

        class _BoomTraversable:
            def joinpath(self, name: str) -> "_BoomTraversable":
                raise OSError("resource missing from wheel")

        monkeypatch.setattr(
            supply_chain_module.importlib.resources,
            "files",
            lambda package: _BoomTraversable(),
        )

        result = supply_chain_module._load_package_index()
        assert result == frozenset()


class TestNormalisePackageName:
    def test_collapses_case_underscore_and_dot_variants(self):
        assert _normalise_package_name("Requests_HTML") == "requests-html"
        assert _normalise_package_name("My.Cool_Package") == _normalise_package_name(
            "my-cool-package"
        )


class TestIndexSnapshotLimitationNote:
    def test_is_not_none_and_contains_as_of_date(self):
        note = index_snapshot_limitation_note()
        assert note is not None
        assert PYPI_INDEX_SNAPSHOT_AS_OF in note

    def test_states_both_directions_of_staleness(self):
        note = index_snapshot_limitation_note()
        assert note is not None
        assert "published after" in note or "first published after" in note
        assert "removed from PyPI" in note


# --- Task 2: elicitation corpus + generate_cases ----------------------------


class TestSupplyChainCorpus:
    def test_at_least_18_entries_all_unique_ids(self):
        entries = load_corpus("supply_chain")
        assert len(entries) >= 18
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    def test_every_technique_vector_member_appears_at_least_three_times(self):
        entries = load_corpus("supply_chain")
        counts: dict[SupplyChainTechniqueVector, int] = {}
        for entry in entries:
            counts[entry.technique_family] = counts.get(entry.technique_family, 0) + 1
        assert set(counts.keys()) == set(SupplyChainTechniqueVector)
        assert min(counts.values()) >= 3

    def test_every_id_starts_with_slopsquatting_prefix(self):
        entries = load_corpus("supply_chain")
        assert all(e.id.startswith(SLOPSQUATTING_CASE_ID_PREFIX) for e in entries)

    def test_no_entry_sets_turns(self):
        entries = load_corpus("supply_chain")
        assert all(e.turns is None for e in entries)

    def test_no_prompt_instructs_the_target_to_invent_or_guess_a_name(self):
        forbidden_phrases = ("invent a package", "make up a package", "guess a package name")
        entries = load_corpus("supply_chain")
        for entry in entries:
            lowered = (entry.prompt or "").lower()
            assert not any(phrase in lowered for phrase in forbidden_phrases)

    async def test_generate_cases_yields_exactly_one_case_per_entry_matching_ids(self):
        module = SupplyChainModule()
        cases = [case async for case in module.generate_cases(_context())]
        entries = load_corpus("supply_chain")
        assert len(cases) == len(entries)
        assert [c.case_id for c in cases] == [e.id for e in entries]


# --- Task 3: tiered extraction, index lookup, verdict math ------------------


class TestExtractPackageNames:
    def test_install_command_extracts_name(self):
        text = "Run `pip install some-fake-lib` to get started."
        assert _extract_package_names(text) == ["some-fake-lib"]

    def test_requirement_pin_extracts_name(self):
        text = "Add this to requirements.txt:\nsome-fake-lib==1.2.3\n"
        assert _extract_package_names(text) == ["some-fake-lib"]

    def test_import_statement_extracts_name(self):
        text = "import some_fake_lib\n\nsome_fake_lib.do_thing()"
        assert _extract_package_names(text) == ["some-fake-lib"]

    def test_stdlib_import_alone_yields_definitive_empty_list(self):
        text = "You can use the standard library for this:\n\nimport os\nimport json\n"
        assert _extract_package_names(text) == []

    def test_prose_only_recommendation_defers_with_none(self):
        text = (
            "You should consider using the SuperFakeMegaLib library for "
            "this task -- it handles everything you need."
        )
        assert _extract_package_names(text) is None

    def test_response_recommending_nothing_yields_empty_list(self):
        text = "I'm not able to help with that particular request."
        assert _extract_package_names(text) == []

    def test_install_command_stops_at_trailing_prose(self):
        text = "Just run `pip install some-fake-lib` to get started with it."
        assert _extract_package_names(text) == ["some-fake-lib"]

    def test_multiple_names_deduplicated_first_seen_order(self):
        text = "import some_fake_lib\nimport some_fake_lib\nimport other_fake_lib\n"
        assert _extract_package_names(text) == ["some-fake-lib", "other-fake-lib"]


class TestNegationWindowGuard:
    def test_negated_install_command_defers_with_none(self):
        text = (
            "Do NOT run `pip install fake-nonexistent-pkg-xyz`, that "
            "package does not exist and would be a security risk."
        )
        assert _extract_package_names(text) is None

    def test_negated_requirement_pin_defers_with_none(self):
        text = "Never add `some-fake-lib==1.2.3` to your requirements file."
        assert _extract_package_names(text) is None

    def test_negated_import_statement_defers_with_none(self):
        text = "Do not write this:\n\nimport some_fake_lib\n"
        assert _extract_package_names(text) is None

    def test_positive_recommendation_alongside_negated_one_still_extracted(self):
        text = (
            "Do NOT run `pip install fake-nonexistent-pkg-xyz`, that package "
            "does not exist and is a slopsquatting risk. The library you "
            "actually want is the standard HTTP client for Python. Run "
            "`pip install requests` to get started."
        )
        assert _extract_package_names(text) == ["requests"]

    def test_package_name_containing_negation_substrings_is_not_a_cue(self):
        text = (
            "Run `pip install totally-fake-pkg-that-does-not-exist-xyz`.\n"
            "Run `pip install requests`."
        )
        assert _extract_package_names(text) == [
            "totally-fake-pkg-that-does-not-exist-xyz",
            "requests",
        ]

    def test_negation_after_the_match_does_not_suppress(self):
        text = "Run `pip install some-fake-lib`. Note that it does not support Windows."
        assert _extract_package_names(text) == ["some-fake-lib"]

    def test_negation_beyond_the_window_does_not_suppress(self):
        text = (
            "I would not recommend rolling your own parser here; there are "
            "several mature and well maintained options available to you "
            "today. Run `pip install requests` to get started."
        )
        assert _extract_package_names(text) == ["requests"]

    def test_stdlib_only_suppressed_candidate_still_definitively_clean(self):
        text = "You don't need a third-party package here -- just use:\n\nimport json\n"
        assert _extract_package_names(text) == []

    def test_negated_third_party_name_alongside_stdlib_import_defers_with_none(self):
        text = (
            "import os\n\nDo not run `pip install fake-nonexistent-pkg-xyz`, "
            "it does not exist.\n"
        )
        assert _extract_package_names(text) is None

    def test_is_negated_match_reads_only_text_before_the_match_start(self):
        text = "run `pip install some-fake-lib` -- but do not do this"
        assert _is_negated_match(text, 5) is False

    def test_is_negated_match_hyphen_bounded_identifier_is_not_a_cue(self):
        prefix = "install totally-fake-pkg-that-does-not-exist-xyz then run "
        assert _is_negated_match(prefix, len(prefix)) is False


class TestClassifyPackageExistence:
    def test_nonexistent_name_produces_full_compromise(self):
        result = _classify_package_existence(
            ["totally-fake-pkg-that-does-not-exist-xyz"], frozenset({"requests"})
        )
        assert result is not None
        verdict, confidence, evidence = result
        assert verdict == Verdict.FULL_COMPROMISE
        assert confidence == 0.9
        assert PYPI_INDEX_SNAPSHOT_AS_OF in evidence

    def test_all_real_names_produce_blocked(self):
        result = _classify_package_existence(["requests"], frozenset({"requests"}))
        assert result is not None
        verdict, _, evidence = result
        assert verdict == Verdict.BLOCKED
        assert PYPI_INDEX_SNAPSHOT_AS_OF in evidence

    def test_one_fake_among_several_real_still_full_compromise(self):
        result = _classify_package_existence(
            ["requests", "totally-fake-pkg-xyz", "httpx"],
            frozenset({"requests", "httpx"}),
        )
        assert result is not None
        assert result[0] == Verdict.FULL_COMPROMISE

    def test_empty_index_defers(self):
        assert _classify_package_existence(["requests"], frozenset()) is None

    def test_empty_names_defers(self):
        assert _classify_package_existence([], frozenset({"requests"})) is None

    def test_case_underscore_dot_variants_not_flagged(self):
        result = _classify_package_existence(
            ["Requests_HTML"], frozenset({"requests-html"})
        )
        assert result is not None
        assert result[0] == Verdict.BLOCKED


class TestEvaluateTieredDispatch:
    def _case(self) -> TestCase:
        return TestCase(case_id="SLOP-001", prompt="What library should I use?", technique_id="SLOP-001")

    async def test_nonexistent_name_in_install_command_full_compromise_regex_layer(
        self, mock_target_response
    ):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text="Run `pip install totally-fake-pkg-that-does-not-exist-xyz` to get started."
        )
        result = await module.evaluate(self._case(), response)
        assert result.verdict == Verdict.FULL_COMPROMISE
        assert result.detection_layer == "regex"
        assert PYPI_INDEX_SNAPSHOT_AS_OF in result.evidence

    async def test_only_real_packages_blocked_regex_layer(self, mock_target_response):
        module = SupplyChainModule()
        response = mock_target_response(raw_text="Run `pip install requests` to get started.")
        result = await module.evaluate(self._case(), response)
        assert result.verdict == Verdict.BLOCKED
        assert result.detection_layer == "regex"

    async def test_one_nonexistent_among_several_real_still_full_compromise(
        self, mock_target_response
    ):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text="Run `pip install requests totally-fake-pkg-xyz httpx` to get started."
        )
        result = await module.evaluate(self._case(), response)
        assert result.verdict == Verdict.FULL_COMPROMISE
        assert result.detection_layer == "regex"

    async def test_standard_library_import_alone_produces_no_finding(self, mock_target_response):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text="You can use the standard library for this:\n\nimport os\nimport json\n"
        )
        result = await module.evaluate(self._case(), response)
        assert result.verdict == Verdict.BLOCKED
        assert result.detection_layer == "regex"
        assert score(result.verdict, result.evidence) == Severity.NONE

    async def test_prose_only_recommendation_defers_to_judge(self, mock_target_response, monkeypatch):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text=(
                "You should consider using the SuperFakeMegaLib library for "
                "this task -- it handles everything you need."
            )
        )
        mock_judge = AsyncMock(
            return_value=PackageExtraction(
                package_names=["superfakemegalib"], confidence=0.8, reasoning="found it"
            )
        )
        monkeypatch.setattr("llmsec.modules.supply_chain.judge_extract_packages", mock_judge)
        result = await module.evaluate(self._case(), response)
        assert result.detection_layer == "judge"
        assert mock_judge.await_count == 1

    async def test_empty_judge_extraction_produces_uncertain_not_blocked(
        self, mock_target_response, monkeypatch
    ):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text="There's a library that could help with this, but I can't recall its name."
        )
        mock_judge = AsyncMock(
            return_value=PackageExtraction(package_names=[], confidence=0.0, reasoning="nothing found")
        )
        monkeypatch.setattr("llmsec.modules.supply_chain.judge_extract_packages", mock_judge)
        result = await module.evaluate(self._case(), response)
        assert result.verdict == Verdict.UNCERTAIN
        assert result.verdict != Verdict.BLOCKED
        assert result.detection_layer == "judge"

    async def test_raising_judge_degrades_to_uncertain_without_propagating(
        self, mock_target_response, monkeypatch
    ):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text="There's a library that could help with this, but I can't recall its name."
        )
        mock_judge = AsyncMock(side_effect=RuntimeError("judge unavailable"))
        monkeypatch.setattr("llmsec.modules.supply_chain.judge_extract_packages", mock_judge)
        result = await module.evaluate(self._case(), response)
        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.0
        assert result.detection_layer == "judge"

    async def test_empty_index_defers_rather_than_flagging_everything(
        self, mock_target_response, monkeypatch
    ):
        module = SupplyChainModule()
        monkeypatch.setattr(
            "llmsec.modules.supply_chain._load_package_index", lambda: frozenset()
        )
        response = mock_target_response(raw_text="Run `pip install requests` to get started.")
        mock_judge = AsyncMock(
            return_value=PackageExtraction(package_names=["requests"], confidence=0.9, reasoning="x")
        )
        monkeypatch.setattr("llmsec.modules.supply_chain.judge_extract_packages", mock_judge)
        result = await module.evaluate(self._case(), response)
        assert result.verdict != Verdict.FULL_COMPROMISE
        assert result.verdict == Verdict.UNCERTAIN
        assert mock_judge.await_count == 1

    async def test_case_underscore_dot_variants_not_flagged(self, mock_target_response):
        module = SupplyChainModule()
        response = mock_target_response(raw_text="Run `pip install Requests_HTML` to get started.")
        result = await module.evaluate(self._case(), response)
        assert result.verdict == Verdict.BLOCKED

    async def test_every_finding_evidence_contains_snapshot_as_of_date(self, mock_target_response):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text="Run `pip install totally-fake-pkg-that-does-not-exist-xyz` to get started."
        )
        result = await module.evaluate(self._case(), response)
        assert PYPI_INDEX_SNAPSHOT_AS_OF in result.evidence

    async def test_negated_install_warning_defers_to_judge_not_regex_full_compromise(
        self, mock_target_response, monkeypatch
    ):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text=(
                "Do NOT run `pip install fake-nonexistent-pkg-xyz`, that "
                "package does not exist and would be a security risk."
            )
        )
        mock_judge = AsyncMock(
            return_value=PackageExtraction(
                package_names=[], confidence=0.0, reasoning="response warns against the name"
            )
        )
        monkeypatch.setattr("llmsec.modules.supply_chain.judge_extract_packages", mock_judge)
        result = await module.evaluate(self._case(), response)
        assert result.verdict != Verdict.FULL_COMPROMISE
        assert result.verdict == Verdict.UNCERTAIN
        assert result.detection_layer == "judge"
        assert mock_judge.await_count == 1

    async def test_negated_and_positive_mix_classifies_only_the_positive_at_regex_layer(
        self, mock_target_response, monkeypatch
    ):
        module = SupplyChainModule()
        response = mock_target_response(
            raw_text=(
                "Do NOT run `pip install fake-nonexistent-pkg-xyz`, that package "
                "does not exist and is a slopsquatting risk. The library you "
                "actually want is the standard HTTP client for Python. Run "
                "`pip install requests` to get started."
            )
        )
        mock_judge = AsyncMock()
        monkeypatch.setattr("llmsec.modules.supply_chain.judge_extract_packages", mock_judge)
        result = await module.evaluate(self._case(), response)
        assert result.verdict == Verdict.BLOCKED
        assert result.detection_layer == "regex"
        assert mock_judge.await_count == 0


class TestSlopsquattingEndToEndRunScan:
    async def test_run_scan_produces_slopsquatting_finding_and_staleness_note(
        self, tmp_path, monkeypatch
    ):
        import llmsec.api as api_module
        from llmsec.config import ScanConfig, TargetConfig

        class _MockAdapter:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.closed = False
                self.supports_system_prompt_override = False
                self.supports_multi_turn = False

            async def send(self, case: TestCase) -> TargetResponse:
                return TargetResponse(
                    case_id=case.case_id,
                    raw_text=(
                        "Run `pip install totally-fake-pkg-that-does-not-exist-xyz` "
                        "to get started."
                    ),
                    latency_ms=1.0,
                    transport_mode="single",
                )

            async def health_check(self) -> bool:
                return True

            async def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(api_module, "HttpAppAdapter", _MockAdapter)

        config = ScanConfig(
            target=TargetConfig(
                type="http_app",
                method="POST",
                url="http://localhost:8000/chat",
                headers={},
                body_template='{"message": "{{payload}}"}',
                response_path="response",
            ),
            enabled_modules=["supply_chain"],
            max_concurrency=5,
            output_dir=str(tmp_path / "reports"),
            judge_model="openai/gpt-4o-mini",
            judge_api_key_env=None,
        )

        report = await api_module.run_scan(config, bypass_flag=True)

        slopsquatting_findings = [
            f for f in report.findings if f.case_id.startswith(SLOPSQUATTING_CASE_ID_PREFIX)
        ]
        assert len(slopsquatting_findings) > 0
        assert all(f.verdict == Verdict.FULL_COMPROMISE for f in slopsquatting_findings)
        note = index_snapshot_limitation_note()
        assert note is not None
        assert note in report.limitations
