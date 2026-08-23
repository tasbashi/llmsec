"""The tracer's own end-to-end assertions (Task 2) -- DASH-03/04's proven
happy path plus the two structural guarantees this plan's `<verify>` block
checks directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason='dashboard suite requires pip install -e ".[dashboard,dev]"')


def test_list_reports_without_token_is_401(client, dashboard_token) -> None:
    res = client.get("/api/reports")
    assert res.status_code == 401


def test_list_reports_with_token_returns_summaries(client, auth_headers) -> None:
    res = client.get("/api/reports", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 3

    # Newest completed_at first (D-10).
    completed_ats = [entry["completed_at"] for entry in body]
    assert completed_ats == sorted(completed_ats, reverse=True)

    for entry in body:
        assert set(entry.keys()) == {
            "scan_id",
            "target_summary",
            "module_ids",
            "max_severity",
            "finding_count",
            "started_at",
            "completed_at",
        }
        assert "findings" not in entry
        assert "case_log" not in entry


def test_list_reports_includes_empty_findings_report_as_none_severity(client, auth_headers) -> None:
    res = client.get("/api/reports", headers=auth_headers)
    body = res.json()
    severities = {entry["max_severity"] for entry in body}
    assert "none" in severities  # Pitfall #3: the findings=[] report is reachable


def test_create_app_constructs_without_a_built_static_dir(tmp_path: Path) -> None:
    """`create_app()` must succeed even when the frontend has never been
    built -- the backend test suite never depends on a Node toolchain."""
    from llmsec.dashboard.app import create_app

    missing_static = tmp_path / "does-not-exist"
    app = create_app(tmp_path / "reports", static_dir=missing_static)
    assert app is not None
