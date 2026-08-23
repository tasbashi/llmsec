"""The tracer's own end-to-end assertions (Task 2, 11-01-PLAN.md).

Proves one authorized `http_app` scan can be launched over HTTP and watched
from `queued` through `complete` (or `failed`) via polling, that the
returned `scan_id` resolves against the existing reports endpoint (D-13),
and that a non-attested request never reaches `dispatch_scan`.
"""

from __future__ import annotations

import re
import time

import pytest

pytest.importorskip("fastapi", reason='dashboard suite requires pip install -e ".[dashboard,dev]"')

_SCAN_ID_RE = re.compile(r"^[0-9a-f]{32}$")

#: Poll budget for waiting on a background scan task to progress past a
#: released gate event -- generous enough to absorb scheduler jitter
#: without making a genuinely stuck test hang indefinitely.
_POLL_ATTEMPTS = 200
_POLL_INTERVAL_S = 0.01


def _poll_scan_status(client, scan_id: str, headers: dict, until_phase: str) -> dict:
    """Poll `GET /api/scans/{scan_id}/status` until `phase == until_phase` or
    the poll budget is exhausted, returning the last observed body.

    A tiny sleep between polls is required: the fake's gate event is
    released synchronously, but the background task actually resumes on
    the portal's event loop asynchronously -- an immediate re-poll with no
    yield can observe the pre-release phase every single time.
    """
    body: dict = {}
    for _ in range(_POLL_ATTEMPTS):
        res = client.get(f"/api/scans/{scan_id}/status", headers=headers)
        body = res.json()
        if body["phase"] == until_phase:
            break
        time.sleep(_POLL_INTERVAL_S)
    return body


@pytest.fixture(autouse=True)
def _permit_all_targets(monkeypatch: pytest.MonkeyPatch):
    """Default `_resolve_ips` fake for every test in this file.

    This file exercises scan-lifecycle behavior (Task 2), not the SSRF gate
    itself (Task 3 covers that in `test_dashboard_ssrf.py`). Resolving
    every hostname to a public address keeps the gate (added in Task 3,
    after these tests were written) from interfering here. A test that
    DOES want to exercise the gate's rejection path
    (`test_ssrf_rejected_target_never_constructs_adapter`) overrides this
    patch with its own `monkeypatch.setattr` call on the same target.
    """
    from ipaddress import ip_address

    from llmsec.dashboard import ssrf_guard

    async def _fake_resolve(hostname: str):
        return [ip_address("93.184.216.34")]

    monkeypatch.setattr(ssrf_guard, "_resolve_ips", _fake_resolve)


def _http_app_body(url: str = "http://example.test/chat", *, attested: bool = True) -> dict:
    return {
        "target": {
            "type": "http_app",
            "method": "POST",
            "url": url,
            "body_template": '{"message": "{{prompt}}"}',
        },
        "authorization_attested": attested,
    }


def test_create_scan_returns_202_and_scan_id_immediately(
    scans_client, auth_headers, controllable_run_scan
) -> None:
    res = scans_client.post("/api/scans", json=_http_app_body(), headers=auth_headers)

    assert res.status_code == 202
    body = res.json()
    assert _SCAN_ID_RE.fullmatch(body["scan_id"])
    # The POST returned while the fake is still gated behind its first
    # event -- proof the dispatch never blocked on the scan itself
    # (RESEARCH.md Pitfall 3).
    assert not controllable_run_scan.running_event.is_set()


def test_create_scan_rejects_max_concurrency_above_50(
    scans_client, auth_headers, controllable_run_scan
) -> None:
    """11-REVIEW.md IN-02: `ScanRequest.max_concurrency` gained an upper
    bound (`le=50`) once this phase exposed it to a plain browser number
    input -- a bearer-token holder must not be able to set an arbitrarily
    large value.
    """
    body = _http_app_body()
    body["max_concurrency"] = 51
    res = scans_client.post("/api/scans", json=body, headers=auth_headers)
    assert res.status_code == 422


def test_create_scan_requires_attestation(
    scans_client, auth_headers, controllable_run_scan
) -> None:
    # Field omitted entirely -> 422.
    body_missing = _http_app_body()
    del body_missing["authorization_attested"]
    res_missing = scans_client.post("/api/scans", json=body_missing, headers=auth_headers)
    assert res_missing.status_code == 422

    # Field explicitly false -> 400 with the fixed detail string.
    res_false = scans_client.post(
        "/api/scans", json=_http_app_body(attested=False), headers=auth_headers
    )
    assert res_false.status_code == 400
    assert (
        res_false.json()["detail"]
        == "You must confirm authorization before starting a scan."
    )

    # Non-bool value -> 422.
    body_nonbool = _http_app_body()
    body_nonbool["authorization_attested"] = "yes"
    res_nonbool = scans_client.post("/api/scans", json=body_nonbool, headers=auth_headers)
    assert res_nonbool.status_code == 422

    assert controllable_run_scan.calls == []


def test_concurrent_requests_do_not_share_attestation(
    scans_client, auth_headers, controllable_run_scan
) -> None:
    res_attested = scans_client.post(
        "/api/scans",
        json=_http_app_body(url="http://example.test/attested-target", attested=True),
        headers=auth_headers,
    )
    res_unattested = scans_client.post(
        "/api/scans",
        json=_http_app_body(url="http://example.test/other-target", attested=False),
        headers=auth_headers,
    )

    assert res_attested.status_code == 202
    assert res_unattested.status_code == 400
    assert len(controllable_run_scan.calls) == 1


def test_scan_status_transitions(scans_client, auth_headers, controllable_run_scan) -> None:
    res = scans_client.post("/api/scans", json=_http_app_body(), headers=auth_headers)
    scan_id = res.json()["scan_id"]

    observed_phases: list[str] = []

    status_res = scans_client.get(f"/api/scans/{scan_id}/status", headers=auth_headers)
    observed_phases.append(status_res.json()["phase"])

    controllable_run_scan.running_event.set()
    body = _poll_scan_status(scans_client, scan_id, auth_headers, until_phase="scoring")
    observed_phases.append(body["phase"])

    controllable_run_scan.scoring_event.set()
    body = _poll_scan_status(scans_client, scan_id, auth_headers, until_phase="complete")
    observed_phases.append(body["phase"])

    valid_phases = {"queued", "running", "scoring", "complete", "failed"}
    assert set(observed_phases).issubset(valid_phases)
    assert "running" in observed_phases
    assert "scoring" in observed_phases
    assert observed_phases[-1] == "complete"
    assert observed_phases.index("running") < observed_phases.index("scoring")


def test_scan_status_records_failure_without_raising(
    scans_client, auth_headers, controllable_run_scan
) -> None:
    res = scans_client.post("/api/scans", json=_http_app_body(), headers=auth_headers)
    assert res.status_code == 202
    scan_id = res.json()["scan_id"]

    controllable_run_scan.raise_error = RuntimeError("simulated target failure")
    controllable_run_scan.running_event.set()

    body = _poll_scan_status(scans_client, scan_id, auth_headers, until_phase="failed")
    assert body["phase"] == "failed"
    assert body["error"]


def test_scan_status_unknown_and_malformed_share_one_detail(scans_client, auth_headers) -> None:
    unknown_res = scans_client.get(
        "/api/scans/00000000000000000000000000000000/status", headers=auth_headers
    )
    malformed_res = scans_client.get("/api/scans/not-a-scan-id/status", headers=auth_headers)

    assert unknown_res.status_code == 404
    assert malformed_res.status_code == 404
    assert unknown_res.json()["detail"] == malformed_res.json()["detail"]


def test_dashboard_scan_config_ignores_ambient_yaml(tmp_path, monkeypatch) -> None:
    from llmsec.config import TargetConfig
    from llmsec.dashboard.routes import ScanRequest
    from llmsec.dashboard.scans import build_dashboard_scan_config

    yaml_path = tmp_path / "llmsec.config.yaml"
    yaml_path.write_text(
        "target:\n"
        "  type: raw_llm\n"
        "  model: openai/gpt-4o-mini\n"
        "  api_key_env: SOME_KEY\n"
        "attacker:\n"
        "  enabled: true\n"
        "output_dir: ./decoy_reports\n"
    )
    monkeypatch.chdir(tmp_path)

    body = ScanRequest(
        target=TargetConfig(
            type="http_app",
            method="POST",
            url="http://example.test/chat",
            body_template='{"message": "{{prompt}}"}',
        ),
        authorization_attested=True,
    )
    dashboard_dir = tmp_path / "dashboard_reports"

    config = build_dashboard_scan_config(body, str(dashboard_dir))

    assert config.attacker is None
    assert config.output_dir == str(dashboard_dir)


def test_completed_scan_id_is_readable_through_the_reports_endpoint(
    scans_client, auth_headers, controllable_run_scan
) -> None:
    res = scans_client.post("/api/scans", json=_http_app_body(), headers=auth_headers)
    scan_id = res.json()["scan_id"]

    controllable_run_scan.running_event.set()
    controllable_run_scan.scoring_event.set()

    report_res = None
    for _ in range(_POLL_ATTEMPTS):
        report_res = scans_client.get(f"/api/reports/{scan_id}", headers=auth_headers)
        if report_res.status_code == 200:
            break
        time.sleep(_POLL_INTERVAL_S)
    assert report_res is not None
    assert report_res.status_code == 200
    assert report_res.json()["scan_id"] == scan_id


def test_ssrf_rejected_target_never_constructs_adapter(
    scans_client, auth_headers, controllable_run_scan, monkeypatch
) -> None:
    """A target that fails the SSRF gate (Task 3, DASH-07) must never reach
    `dispatch_scan` -- proven here by asserting the recording fake was
    never invoked, nothing downstream of the gate ran.
    """
    from ipaddress import ip_address

    from llmsec.dashboard import ssrf_guard

    async def _fake_resolve_loopback(hostname: str):
        return [ip_address("127.0.0.1")]

    monkeypatch.setattr(ssrf_guard, "_resolve_ips", _fake_resolve_loopback)

    url = "http://internal.local/secret/path?token=abc"
    res = scans_client.post(
        "/api/scans", json=_http_app_body(url=url), headers=auth_headers
    )

    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "internal.local" in detail
    assert url not in detail
    assert controllable_run_scan.calls == []
