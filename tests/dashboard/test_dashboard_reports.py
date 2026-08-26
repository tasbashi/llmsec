"""DASH-04/DASH-05 -- the complete read-only report API.

Task 1 covers the filter/aggregation layer in `reports.py` (unit-level, no
`TestClient` needed). Task 2 adds the four `/api` routes (integration-level,
`TestClient`-driven). Task 3 closes the whole-surface guarantees that only
become checkable once all four routes exist.
"""

from __future__ import annotations

import logging
import uuid

import pytest

pytest.importorskip("fastapi", reason='dashboard suite requires pip install -e ".[dashboard,dev]"')

from llmsec.dashboard.reports import (
    FilterOptions,
    ReportSummary,
    distinct_filter_values,
    list_summaries,
    matches_filters,
    max_severity,
)
from llmsec.models import ScanReport
from llmsec.reporting.json_reporter import load_report
from llmsec.scoring.engine import Severity

#: At least seven hostile `scan_id` shapes (RESEARCH.md Pitfall #5 / T-10-01):
#: raw dot-segments, single- and double-percent-encoded dot-segments, an
#: uppercase-hex id, a 31- and a 33-character hex id, and a leading-slash
#: absolute path. Every one of these must return 404 -- never 200, never 500,
#: and never touch the filesystem outside `output_dir`.
HOSTILE_SCAN_IDS = [
    "../../pyproject",
    "..%2f..%2fpyproject",
    "%2e%2e%2f%2e%2e%2fpyproject",
    uuid.uuid4().hex.upper(),
    uuid.uuid4().hex[:31],
    uuid.uuid4().hex + "a",
    "%2Fetc%2Fpasswd",
]


def _summary(**overrides: object) -> ReportSummary:
    """Build a `ReportSummary` directly -- `matches_filters()`'s unit tests
    don't need a real report file on disk."""
    defaults: dict[str, object] = {
        "scan_id": "a" * 32,
        "target_summary": "http_app:POST http://127.0.0.1:8000/chat",
        "module_ids": ["a", "b"],
        "max_severity": "high",
        "finding_count": 1,
        "started_at": "2026-08-19T00:00:00Z",
        "completed_at": "2026-08-19T00:05:00Z",
    }
    defaults.update(overrides)
    return ReportSummary(**defaults)  # type: ignore[arg-type]


class TestMaxSeverity:
    """D-11: the report's MAXIMUM finding severity, never a containment test."""

    def test_empty_findings_is_none(self, corrupted_reports_dir) -> None:
        healthy_path = (
            corrupted_reports_dir.output_dir / f"scan_{corrupted_reports_dir.healthy_scan_id}.json"
        )
        # None of the three `reports_dir` reports has empty findings, so
        # build one locally rather than depending on fixture internals.
        report = load_report(healthy_path).model_copy(update={"findings": []})
        assert max_severity(report) == Severity.NONE.value

    def test_max_of_low_and_critical_is_critical(self, filter_probe_reports_dir) -> None:
        path = (
            filter_probe_reports_dir.output_dir
            / f"scan_{filter_probe_reports_dir.shared_target_scan_ids[1]}.json"
        )
        report = load_report(path)
        assert max_severity(report) == "critical"

    def test_max_of_low_and_medium_is_medium(self, filter_probe_reports_dir) -> None:
        path = (
            filter_probe_reports_dir.output_dir / f"scan_{filter_probe_reports_dir.tie_scan_id_second}.json"
        )
        report = load_report(path)
        assert max_severity(report) == "medium"


class TestMatchesFilters:
    """D-12's AND/intersection semantics, and A-01's superset module reading."""

    def test_no_filters_matches_everything(self) -> None:
        assert matches_filters(_summary(), target=None, modules=None, severity=None) is True

    def test_target_exact_match(self) -> None:
        s = _summary(target_summary="raw_llm target A")
        assert matches_filters(s, target="raw_llm target A") is True

    def test_target_mismatch(self) -> None:
        s = _summary(target_summary="raw_llm target A")
        assert matches_filters(s, target="raw_llm target B") is False

    def test_target_is_never_a_substring_match(self) -> None:
        s = _summary(target_summary="raw_llm target A")
        assert matches_filters(s, target="raw_llm target") is False

    def test_modules_subset_present_matches(self) -> None:
        s = _summary(module_ids=["a", "b"])
        assert matches_filters(s, modules=["a"]) is True

    def test_modules_superset_semantics_excludes_when_not_all_present(self) -> None:
        s = _summary(module_ids=["a", "b"])
        assert matches_filters(s, modules=["a", "c"]) is False

    def test_severity_none_matches_empty_findings_summary(self) -> None:
        s = _summary(max_severity="none", finding_count=0)
        assert matches_filters(s, severity="none") is True

    def test_and_semantics_target_matches_severity_does_not(self) -> None:
        s = _summary(target_summary="raw_llm target A", max_severity="high")
        assert matches_filters(s, target="raw_llm target A", severity="critical") is False


class TestListSummaries:
    """D-10's ordering (newest-first, `scan_id`-ascending tie-break) and
    Pitfall #6's per-file containment."""

    def test_newest_first_by_completed_at(self, reports_dir) -> None:
        summaries = list_summaries(reports_dir)
        completed_ats = [s.completed_at for s in summaries]
        assert completed_ats == sorted(completed_ats, reverse=True)

    def test_tied_completed_at_breaks_by_scan_id_ascending_and_is_stable(
        self, filter_probe_reports_dir
    ) -> None:
        first_call = list_summaries(filter_probe_reports_dir.output_dir)
        second_call = list_summaries(filter_probe_reports_dir.output_dir)

        def _tie_pair(summaries):
            tied = [
                s
                for s in summaries
                if s.scan_id
                in (
                    filter_probe_reports_dir.tie_scan_id_first,
                    filter_probe_reports_dir.tie_scan_id_second,
                )
            ]
            return [s.scan_id for s in tied]

        expected = [
            filter_probe_reports_dir.tie_scan_id_first,
            filter_probe_reports_dir.tie_scan_id_second,
        ]
        assert _tie_pair(first_call) == expected
        assert _tie_pair(second_call) == expected

    def test_empty_directory_returns_empty_list(self, tmp_path) -> None:
        empty_dir = tmp_path / "empty_reports"
        empty_dir.mkdir()
        assert list_summaries(empty_dir) == []

    def test_nonexistent_directory_returns_empty_list(self, tmp_path) -> None:
        assert list_summaries(tmp_path / "does-not-exist") == []

    def test_one_valid_one_truncated_returns_only_the_valid_one(
        self, corrupted_reports_dir
    ) -> None:
        summaries = list_summaries(corrupted_reports_dir.output_dir)
        scan_ids = {s.scan_id for s in summaries}
        assert corrupted_reports_dir.truncated_scan_id not in scan_ids
        assert len(summaries) == 3  # the three healthy reports_dir reports

    def test_skipped_unreadable_file_is_logged_with_warning(
        self, corrupted_reports_dir, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="llmsec.dashboard.reports"):
            list_summaries(corrupted_reports_dir.output_dir)
        assert any(
            corrupted_reports_dir.truncated_scan_id in record.getMessage()
            or corrupted_reports_dir.truncated_scan_id in str(record.exc_info)
            for record in caplog.records
        )


class TestDistinctFilterValues:
    """D-09: server-computed, deduplicated filter option lists."""

    def test_shared_target_appears_once(self, filter_probe_reports_dir) -> None:
        options = distinct_filter_values(filter_probe_reports_dir.output_dir)
        assert options.targets.count(filter_probe_reports_dir.shared_target) == 1

    def test_module_union_across_reports(self, filter_probe_reports_dir) -> None:
        options = distinct_filter_values(filter_probe_reports_dir.output_dir)
        assert "a" in options.modules
        assert "b" in options.modules

    def test_empty_directory_returns_empty_filter_options_not_none(self, tmp_path) -> None:
        empty_dir = tmp_path / "empty_reports"
        empty_dir.mkdir()
        options = distinct_filter_values(empty_dir)
        assert options == FilterOptions(targets=[], modules=[], severities=[])

    def test_severities_are_enum_members_in_declaration_order_reversed(
        self, filter_probe_reports_dir
    ) -> None:
        options = distinct_filter_values(filter_probe_reports_dir.output_dir)
        assert all(s in {member.value for member in Severity} for s in options.severities)
        expected_order = [
            member.value for member in reversed(list(Severity)) if member.value in options.severities
        ]
        assert options.severities == expected_order

    def test_none_band_is_reachable_when_a_report_has_no_findings(
        self, filter_probe_reports_dir
    ) -> None:
        options = distinct_filter_values(filter_probe_reports_dir.output_dir)
        assert Severity.NONE.value in options.severities

    def test_pre_v11_report_with_absent_deep_summary_key_loads_with_null(
        self, pre_v11_reports_dir
    ) -> None:
        path = pre_v11_reports_dir.output_dir / f"scan_{pre_v11_reports_dir.scan_id}.json"
        report: ScanReport = load_report(path)
        assert report.deep_summary is None


class TestListReportsRoute:
    """`GET /api/reports` with the three combinable filter query params."""

    def test_valid_token_returns_newest_first(self, client, auth_headers) -> None:
        res = client.get("/api/reports", headers=auth_headers)
        assert res.status_code == 200
        completed_ats = [entry["completed_at"] for entry in res.json()]
        assert completed_ats == sorted(completed_ats, reverse=True)

    def test_target_filter_returns_only_exact_matches(
        self, filter_probe_client, filter_probe_reports_dir, auth_headers
    ) -> None:
        res = filter_probe_client.get(
            "/api/reports",
            params={"target": filter_probe_reports_dir.shared_target},
            headers=auth_headers,
        )
        assert res.status_code == 200
        body = res.json()
        assert len(body) == 2
        assert {entry["target_summary"] for entry in body} == {filter_probe_reports_dir.shared_target}

    def test_repeated_module_param_applies_superset_filter(
        self, filter_probe_client, auth_headers
    ) -> None:
        res = filter_probe_client.get(
            "/api/reports",
            params=[("module", "a"), ("module", "b")],
            headers=auth_headers,
        )
        assert res.status_code == 200
        body = res.json()
        assert len(body) == 1
        assert set(body[0]["module_ids"]) >= {"a", "b"}

    def test_severity_none_returns_the_zero_findings_report(
        self, filter_probe_client, filter_probe_reports_dir, auth_headers
    ) -> None:
        res = filter_probe_client.get(
            "/api/reports", params={"severity": "none"}, headers=auth_headers
        )
        assert res.status_code == 200
        body = res.json()
        assert [entry["scan_id"] for entry in body] == [filter_probe_reports_dir.empty_findings_scan_id]

    def test_matching_target_with_nonmatching_severity_returns_empty_200(
        self, filter_probe_client, filter_probe_reports_dir, auth_headers
    ) -> None:
        res = filter_probe_client.get(
            "/api/reports",
            params={"target": filter_probe_reports_dir.shared_target, "severity": "none"},
            headers=auth_headers,
        )
        assert res.status_code == 200
        assert res.json() == []

    def test_one_corrupted_file_leaves_list_at_200_with_only_healthy_entries(
        self, corrupted_client, auth_headers
    ) -> None:
        res = corrupted_client.get("/api/reports", headers=auth_headers)
        assert res.status_code == 200
        assert len(res.json()) == 3

    def test_unauthenticated_is_401(self, client, dashboard_token) -> None:
        res = client.get("/api/reports")
        assert res.status_code == 401


class TestFiltersRoute:
    """`GET /api/filters` -- server-computed, deduplicated dropdown options."""

    def test_returns_deduplicated_options(
        self, filter_probe_client, filter_probe_reports_dir, auth_headers
    ) -> None:
        res = filter_probe_client.get("/api/filters", headers=auth_headers)
        assert res.status_code == 200
        body = res.json()
        assert set(body.keys()) == {"targets", "modules", "severities"}
        assert body["targets"].count(filter_probe_reports_dir.shared_target) == 1

    def test_unauthenticated_is_401(self, client, dashboard_token) -> None:
        res = client.get("/api/filters")
        assert res.status_code == 401


class TestReportDetailRoute:
    """`GET /api/reports/{scan_id}` -- full `ScanReport` parity (D-13)."""

    def test_valid_id_returns_full_report_body(self, client, reports_dir, auth_headers) -> None:
        list_res = client.get("/api/reports", headers=auth_headers)
        scan_id = list_res.json()[0]["scan_id"]

        res = client.get(f"/api/reports/{scan_id}", headers=auth_headers)
        assert res.status_code == 200
        body = res.json()
        for field in ("findings", "case_log", "limitations", "deep_summary"):
            assert field in body

    def test_pre_v11_report_serves_with_deep_summary_null(
        self, pre_v11_reports_dir, auth_headers
    ) -> None:
        from fastapi.testclient import TestClient

        from llmsec.dashboard.app import create_app

        client = TestClient(create_app(pre_v11_reports_dir.output_dir))
        res = client.get(f"/api/reports/{pre_v11_reports_dir.scan_id}", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["deep_summary"] is None

    def test_wellformed_id_with_no_file_is_404_json(self, client, auth_headers) -> None:
        res = client.get(f"/api/reports/{uuid.uuid4().hex}", headers=auth_headers)
        assert res.status_code == 404
        assert "detail" in res.json()

    @pytest.mark.parametrize("hostile_id", HOSTILE_SCAN_IDS)
    def test_hostile_scan_id_is_404_never_200_never_500(
        self, client, auth_headers, hostile_id: str
    ) -> None:
        res = client.get(f"/api/reports/{hostile_id}", headers=auth_headers)
        assert res.status_code == 404
        assert "pyproject" not in res.text
        assert "/home/" not in res.text

    def test_corrupted_file_id_is_500_not_a_fabricated_report(
        self, corrupted_client, corrupted_reports_dir, auth_headers
    ) -> None:
        res = corrupted_client.get(
            f"/api/reports/{corrupted_reports_dir.truncated_scan_id}", headers=auth_headers
        )
        assert res.status_code == 500
        assert "findings" not in res.text

    def test_unauthenticated_is_401(self, client, dashboard_token) -> None:
        res = client.get(f"/api/reports/{uuid.uuid4().hex}")
        assert res.status_code == 401


class TestReportDownloadRoute:
    """`GET /api/reports/{scan_id}/download` -- raw on-disk bytes (A-02)."""

    def test_download_body_matches_bytes_on_disk(self, client, reports_dir, auth_headers) -> None:
        list_res = client.get("/api/reports", headers=auth_headers)
        scan_id = list_res.json()[0]["scan_id"]
        on_disk = (reports_dir / f"scan_{scan_id}.json").read_bytes()

        res = client.get(f"/api/reports/{scan_id}/download", headers=auth_headers)
        assert res.status_code == 200
        assert "attachment" in res.headers["content-disposition"]
        assert f"scan_{scan_id}.json" in res.headers["content-disposition"]
        assert res.content == on_disk

    def test_download_serves_the_corrupted_file_as_the_escape_hatch(
        self, corrupted_client, corrupted_reports_dir, auth_headers
    ) -> None:
        on_disk = (
            corrupted_reports_dir.output_dir
            / f"scan_{corrupted_reports_dir.truncated_scan_id}.json"
        ).read_bytes()
        res = corrupted_client.get(
            f"/api/reports/{corrupted_reports_dir.truncated_scan_id}/download", headers=auth_headers
        )
        assert res.status_code == 200
        assert res.content == on_disk

    def test_wellformed_id_with_no_file_is_404(self, client, auth_headers) -> None:
        res = client.get(f"/api/reports/{uuid.uuid4().hex}/download", headers=auth_headers)
        assert res.status_code == 404

    @pytest.mark.parametrize("hostile_id", HOSTILE_SCAN_IDS)
    def test_hostile_scan_id_is_404(self, client, auth_headers, hostile_id: str) -> None:
        res = client.get(f"/api/reports/{hostile_id}/download", headers=auth_headers)
        assert res.status_code == 404

    def test_unauthenticated_is_401(self, client, reports_dir, auth_headers) -> None:
        list_res = client.get("/api/reports", headers=auth_headers)
        scan_id = list_res.json()[0]["scan_id"]
        res = client.get(f"/api/reports/{scan_id}/download")
        assert res.status_code == 401


class TestWholeSurfaceGating:
    """No `/api/*` route added by this plan may be born ungated -- reuses
    `test_dashboard_auth.py`'s enumeration mechanism rather than a
    hand-written per-route case."""

    def test_enumerated_api_route_count_is_at_least_four(self, client, dashboard_token) -> None:
        from tests.dashboard.test_dashboard_auth import _api_routes

        routes = _api_routes(client.app)
        assert len(routes) >= 4

    def test_filters_route_rejects_unauthenticated_caller(self, client, dashboard_token) -> None:
        res = client.get("/api/filters")
        assert res.status_code == 401

    def test_detail_route_rejects_unauthenticated_caller(self, client, dashboard_token) -> None:
        res = client.get(f"/api/reports/{uuid.uuid4().hex}")
        assert res.status_code == 401

    def test_download_route_rejects_unauthenticated_caller(self, client, dashboard_token) -> None:
        res = client.get(f"/api/reports/{uuid.uuid4().hex}/download")
        assert res.status_code == 401


class TestContainmentAsymmetry:
    """One corrupted file degrades exactly one endpoint, never without a
    trace (P-03)."""

    def test_list_and_filters_degrade_while_detail_stays_split_by_id(
        self,
        single_corrupted_client,
        single_corrupted_reports_dir,
        dashboard_token,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        headers = {"Authorization": f"Bearer {dashboard_token}"}

        with caplog.at_level(logging.WARNING, logger="llmsec.dashboard.reports"):
            list_res = single_corrupted_client.get("/api/reports", headers=headers)
            filters_res = single_corrupted_client.get("/api/filters", headers=headers)
            healthy_res = single_corrupted_client.get(
                f"/api/reports/{single_corrupted_reports_dir.healthy_scan_id}", headers=headers
            )
            corrupted_res = single_corrupted_client.get(
                f"/api/reports/{single_corrupted_reports_dir.truncated_scan_id}", headers=headers
            )

        assert list_res.status_code == 200
        assert len(list_res.json()) == 1
        assert filters_res.status_code == 200
        assert healthy_res.status_code == 200
        assert corrupted_res.status_code == 500

        assert any(
            single_corrupted_reports_dir.truncated_scan_id in record.getMessage()
            for record in caplog.records
        )


class TestReportsDirUnreachableExceptViaGuard:
    """ROADMAP success criterion 4: no accidental second read path to
    `output_dir` exists beside the four intended `/api/*` routes."""

    @pytest.mark.parametrize(
        "path_template",
        [
            "/scan_{scan_id}.json",
            "/llmsec_reports/scan_{scan_id}.json",
            "/static/scan_{scan_id}.json",
            "/api/scan_{scan_id}.json",
        ],
    )
    def test_filesystem_shaped_url_with_valid_token_never_returns_report_content(
        self, client, reports_dir, auth_headers, path_template: str
    ) -> None:
        list_res = client.get("/api/reports", headers=auth_headers)
        scan_id = list_res.json()[0]["scan_id"]
        on_disk_content = (reports_dir / f"scan_{scan_id}.json").read_text()

        path = path_template.format(scan_id=scan_id)
        res = client.get(path, headers=auth_headers)
        assert not (res.status_code == 200 and on_disk_content in res.text)


class TestNoLeakOnNewRoutes:
    """No 404/500 response body may leak the reports-directory path or the
    configured token value -- and a 500 must never carry a real-filesystem
    traceback into the response body."""

    def test_404_from_missing_report_leaks_nothing(
        self, client, reports_dir, auth_headers, dashboard_token
    ) -> None:
        res = client.get(f"/api/reports/{uuid.uuid4().hex}", headers=auth_headers)
        assert res.status_code == 404
        assert str(reports_dir) not in res.text
        assert dashboard_token not in res.text

    def test_404_from_traversal_payload_leaks_nothing(
        self, client, reports_dir, auth_headers, dashboard_token
    ) -> None:
        res = client.get("/api/reports/..%2f..%2fpyproject", headers=auth_headers)
        assert res.status_code == 404
        assert str(reports_dir) not in res.text
        assert dashboard_token not in res.text

    def test_500_from_corrupted_file_leaks_nothing(
        self, corrupted_client, corrupted_reports_dir, auth_headers, dashboard_token
    ) -> None:
        res = corrupted_client.get(
            f"/api/reports/{corrupted_reports_dir.truncated_scan_id}", headers=auth_headers
        )
        assert res.status_code == 500
        assert str(corrupted_reports_dir.output_dir) not in res.text
        assert dashboard_token not in res.text
        # No traceback with real filesystem paths -- FastAPI's default 500
        # body carries no traceback, and no handler here adds one.
        assert "Traceback" not in res.text
        assert __file__ not in res.text


class TestDetailParityWithScanReportFields:
    """D-13 / ROADMAP SC#3: the detail payload has parity with the Markdown
    report BY CONSTRUCTION -- derived from `ScanReport.model_fields`, never
    a hand-maintained field list, so a future field addition is
    automatically required to appear here too."""

    def test_detail_payload_contains_every_scanreport_field(
        self, client, reports_dir, auth_headers
    ) -> None:
        list_res = client.get("/api/reports", headers=auth_headers)
        scan_id = list_res.json()[0]["scan_id"]

        res = client.get(f"/api/reports/{scan_id}", headers=auth_headers)
        assert res.status_code == 200
        body_keys = set(res.json().keys())
        expected_keys = set(ScanReport.model_fields.keys())
        assert expected_keys <= body_keys
