"""Shared fixtures for `tests/dashboard/`.

NO module-scope import of `fastapi` or of any `llmsec.dashboard` module that
imports it -- every such import goes inside a fixture body, which pytest
only evaluates for tests that actually request the fixture. The
extra-absent guard is a per-module skip-if-fastapi-missing call at the top
of each fastapi-dependent test module (NOT here) -- a conftest-level skip
would also skip `test_dashboard_gate.py`, whose entire job is covering the
extra-absent path and which must run in every environment.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest

from llmsec.models import EvalResult, Finding, ScanReport, Verdict
from llmsec.reporting.json_reporter import JsonReporter

#: Fixed value used across the auth test matrix -- distinctive enough that a
#: substring search for it is meaningful (the no-leak matrix asserts this
#: exact string never appears in a response body/header/reason phrase).
TEST_DASHBOARD_TOKEN = "unit-test-dashboard-secret-llmsec-9f3c"


def _finding(case_id: str, severity: str, verdict: Verdict) -> Finding:
    return Finding(
        case_id=case_id,
        technique_id="direct_ask",
        verdict=verdict,
        severity=severity,
        owasp_ref="LLM07",
        evidence=f"evidence for {case_id}",
        remediation="Add output filtering.",
    )


def _eval_result(case_id: str, verdict: Verdict, evidence: str) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        verdict=verdict,
        confidence=0.9,
        evidence=evidence,
        detection_layer="regex",
    )


@pytest.fixture
def reports_dir(tmp_path: Path) -> Iterator[Path]:
    """Build a `tmp_path` directory holding three hand-constructed reports,
    written through `JsonReporter().write()` -- never a hand-written JSON
    string.

    - one report with a HIGH finding, short target, two overlapping modules
    - one report with a CRITICAL finding, a long URL-shaped target, one
      disjoint module
    - one report with `findings=[]` (every case BLOCKED) -- Pitfall #3's
      empty-findings edge case
    """
    output_dir = tmp_path / "llmsec_reports"
    reporter = JsonReporter()

    scan_id_1 = uuid.uuid4().hex
    scan_id_2 = uuid.uuid4().hex
    scan_id_3 = uuid.uuid4().hex

    report_1 = ScanReport(
        scan_id=scan_id_1,
        target_summary="raw_llm target, model=gpt-4o-mini",
        module_ids=["system_prompt_leakage", "prompt_injection"],
        findings=[_finding("case-1", "high", Verdict.FULL_COMPROMISE)],
        case_log=[_eval_result("case-1", Verdict.FULL_COMPROMISE, "leaked system prompt")],
        started_at="2026-08-17T00:00:00Z",
        completed_at="2026-08-17T00:05:00Z",
    )
    report_2 = ScanReport(
        scan_id=scan_id_2,
        target_summary=(
            "http_app:POST http://127.0.0.1:8000/very/long/path/that/keeps/going/"
            "chat?session=abcdef0123456789&trace=true&debug=verbose-mode-enabled"
        ),
        module_ids=["pii_exfiltration"],
        findings=[_finding("case-2", "critical", Verdict.FULL_COMPROMISE)],
        case_log=[_eval_result("case-2", Verdict.FULL_COMPROMISE, "leaked a credential-shaped secret")],
        started_at="2026-08-18T00:00:00Z",
        completed_at="2026-08-18T00:05:00Z",
    )
    report_3 = ScanReport(
        scan_id=scan_id_3,
        target_summary="http_app:POST http://127.0.0.1:8000/chat",
        module_ids=["insecure_output"],
        findings=[],
        case_log=[_eval_result("case-3", Verdict.BLOCKED, "refused cleanly")],
        started_at="2026-08-16T00:00:00Z",
        completed_at="2026-08-16T00:05:00Z",
    )

    async def _write_all() -> None:
        for report in (report_1, report_2, report_3):
            await reporter.write(report, output_dir)

    asyncio.run(_write_all())

    yield output_dir


class FilterProbeFixture(NamedTuple):
    """Handles into `filter_probe_reports_dir`'s deliberately-constructed
    corpus, so tests can assert against known scan_ids/targets instead of
    re-deriving them from filenames.
    """

    output_dir: Path
    tie_scan_id_first: str
    tie_scan_id_second: str
    shared_target: str
    shared_target_scan_ids: tuple[str, str]
    empty_findings_scan_id: str


@pytest.fixture
def filter_probe_reports_dir(tmp_path: Path) -> FilterProbeFixture:
    """A directory purpose-built for DASH-04's filter/ordering/adjacency edge
    probes (10-02-PLAN.md Task 1 `<behavior>`):

    - two reports sharing one `completed_at` value but different `scan_id`s
      (D-10's ordering tie-break, broken by `scan_id` ascending)
    - two reports sharing one `target_summary` value (the adjacency probe --
      two distinct rows, one deduplicated filter option)
    - one report with `findings=[]` (Pitfall #3's empty-severity edge)
    - `module_ids` overlapping partially, so superset filtering (A-01) is
      distinguishable from any-overlap filtering
    - findings spanning `low`, `medium`, `high`, `critical`, plus the
      empty-findings report's derived `none` -- five distinct bands
    """
    output_dir = tmp_path / "filter_probe_reports"
    reporter = JsonReporter()

    shared_target = "http_app:POST http://127.0.0.1:8000/chat"
    tied_completed_at = "2026-08-19T12:00:00Z"

    tie_scan_id_first, tie_scan_id_second = sorted(uuid.uuid4().hex for _ in range(2))
    shared_scan_id_1 = uuid.uuid4().hex
    shared_scan_id_2 = uuid.uuid4().hex
    empty_scan_id = uuid.uuid4().hex

    reports = [
        ScanReport(
            scan_id=tie_scan_id_first,
            target_summary="raw_llm target A (tie probe)",
            module_ids=["system_prompt_leakage", "prompt_injection"],
            findings=[_finding("case-tie-a", "low", Verdict.UNCERTAIN)],
            case_log=[_eval_result("case-tie-a", Verdict.UNCERTAIN, "uncertain evidence")],
            started_at="2026-08-19T11:55:00Z",
            completed_at=tied_completed_at,
        ),
        ScanReport(
            scan_id=tie_scan_id_second,
            target_summary="raw_llm target B (tie probe)",
            module_ids=["prompt_injection", "pii_exfiltration"],
            findings=[_finding("case-tie-b", "medium", Verdict.PARTIAL_LEAK)],
            case_log=[_eval_result("case-tie-b", Verdict.PARTIAL_LEAK, "partial leak evidence")],
            started_at="2026-08-19T11:55:00Z",
            completed_at=tied_completed_at,
        ),
        ScanReport(
            scan_id=shared_scan_id_1,
            target_summary=shared_target,
            module_ids=["a", "b"],
            findings=[_finding("case-shared-1", "high", Verdict.FULL_COMPROMISE)],
            case_log=[_eval_result("case-shared-1", Verdict.FULL_COMPROMISE, "leak evidence")],
            started_at="2026-08-19T09:00:00Z",
            completed_at="2026-08-19T09:05:00Z",
        ),
        ScanReport(
            scan_id=shared_scan_id_2,
            target_summary=shared_target,
            module_ids=["a"],
            findings=[_finding("case-shared-2", "critical", Verdict.FULL_COMPROMISE)],
            case_log=[_eval_result("case-shared-2", Verdict.FULL_COMPROMISE, "credential-shaped leak")],
            started_at="2026-08-19T10:00:00Z",
            completed_at="2026-08-19T10:05:00Z",
        ),
        ScanReport(
            scan_id=empty_scan_id,
            target_summary="http_app:POST http://127.0.0.1:8000/clean",
            module_ids=["insecure_output"],
            findings=[],
            case_log=[_eval_result("case-empty", Verdict.BLOCKED, "refused cleanly")],
            started_at="2026-08-19T08:00:00Z",
            completed_at="2026-08-19T08:05:00Z",
        ),
    ]

    async def _write_all() -> None:
        for report in reports:
            await reporter.write(report, output_dir)

    asyncio.run(_write_all())

    return FilterProbeFixture(
        output_dir=output_dir,
        tie_scan_id_first=tie_scan_id_first,
        tie_scan_id_second=tie_scan_id_second,
        shared_target=shared_target,
        shared_target_scan_ids=(shared_scan_id_1, shared_scan_id_2),
        empty_findings_scan_id=empty_scan_id,
    )


class CorruptedReportsFixture(NamedTuple):
    """Handles into `corrupted_reports_dir`'s copy-plus-one-bad-file corpus."""

    output_dir: Path
    healthy_scan_id: str
    truncated_scan_id: str


@pytest.fixture
def corrupted_reports_dir(tmp_path: Path, reports_dir: Path) -> CorruptedReportsFixture:
    """A copy of `reports_dir` (three valid reports) plus one deliberately
    truncated `scan_<32-hex>.json` -- a valid filename shape with an invalid
    JSON body (RESEARCH.md Pitfall #6). Pins the containment asymmetry: the
    list/filters endpoints must degrade to omit this file (with a WARNING
    logged, P-03), while the detail/download endpoints for its `scan_id`
    stay reachable -- loudly (500) for detail, literally (raw bytes) for
    download.
    """
    copy_dir = tmp_path / "corrupted_reports"
    shutil.copytree(reports_dir, copy_dir)

    healthy_paths = sorted(copy_dir.glob("scan_*.json"))
    healthy_scan_id = healthy_paths[0].stem.removeprefix("scan_")

    truncated_scan_id = uuid.uuid4().hex
    (copy_dir / f"scan_{truncated_scan_id}.json").write_text('{"scan_id": "trun')

    return CorruptedReportsFixture(
        output_dir=copy_dir,
        healthy_scan_id=healthy_scan_id,
        truncated_scan_id=truncated_scan_id,
    )


class SingleCorruptedReportsFixture(NamedTuple):
    """Handles into `single_corrupted_reports_dir`'s minimal 1-healthy +
    1-truncated corpus."""

    output_dir: Path
    healthy_scan_id: str
    truncated_scan_id: str


@pytest.fixture
def single_corrupted_reports_dir(tmp_path: Path) -> SingleCorruptedReportsFixture:
    """EXACTLY one healthy report plus one truncated file -- the minimal
    corpus 10-02-PLAN.md Task 3's containment-asymmetry test needs (list
    endpoint returns 200 with exactly ONE entry, not incidentally three).
    """
    output_dir = tmp_path / "single_corrupted_reports"
    reporter = JsonReporter()
    healthy_scan_id = uuid.uuid4().hex
    report = ScanReport(
        scan_id=healthy_scan_id,
        target_summary="raw_llm target, single-corrupted probe",
        module_ids=["system_prompt_leakage"],
        findings=[_finding("case-single", "high", Verdict.FULL_COMPROMISE)],
        case_log=[_eval_result("case-single", Verdict.FULL_COMPROMISE, "leak evidence")],
        started_at="2026-08-19T00:00:00Z",
        completed_at="2026-08-19T00:05:00Z",
    )
    asyncio.run(reporter.write(report, output_dir))

    truncated_scan_id = uuid.uuid4().hex
    (output_dir / f"scan_{truncated_scan_id}.json").write_text('{"scan_id": "trun')

    return SingleCorruptedReportsFixture(
        output_dir=output_dir,
        healthy_scan_id=healthy_scan_id,
        truncated_scan_id=truncated_scan_id,
    )


@pytest.fixture
def single_corrupted_client(single_corrupted_reports_dir: SingleCorruptedReportsFixture):
    """A `TestClient` wired to `single_corrupted_reports_dir`, with
    `raise_server_exceptions=False` for the same reason `corrupted_client`
    needs it -- the detail endpoint's loud `ValidationError` must surface as
    a real 500 response, not re-raise in the test process.
    """
    from fastapi.testclient import TestClient

    from llmsec.dashboard.app import create_app

    return TestClient(
        create_app(single_corrupted_reports_dir.output_dir), raise_server_exceptions=False
    )


class PreV11ReportFixture(NamedTuple):
    """Handles into `pre_v11_reports_dir`'s single pre-v1.1-shaped report."""

    output_dir: Path
    scan_id: str


@pytest.fixture
def pre_v11_reports_dir(tmp_path: Path) -> PreV11ReportFixture:
    """A directory holding one report written with the `deep_summary` key
    entirely ABSENT from the persisted JSON (not an explicit `null`) --
    reproduces the exact on-disk shape of a report written before v1.1
    introduced the field, so ROADMAP SC#5's "no migration step" guarantee is
    tested against a file that genuinely predates the field.
    """
    output_dir = tmp_path / "pre_v11_reports"
    reporter = JsonReporter()
    scan_id = uuid.uuid4().hex
    report = ScanReport(
        scan_id=scan_id,
        target_summary="raw_llm target, pre-v1.1 shape",
        module_ids=["system_prompt_leakage"],
        findings=[_finding("case-pre", "medium", Verdict.PARTIAL_LEAK)],
        case_log=[_eval_result("case-pre", Verdict.PARTIAL_LEAK, "partial leak evidence")],
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:05:00Z",
    )
    asyncio.run(reporter.write(report, output_dir))

    path = output_dir / f"scan_{scan_id}.json"
    data = json.loads(path.read_text())
    del data["deep_summary"]
    path.write_text(json.dumps(data))

    return PreV11ReportFixture(output_dir=output_dir, scan_id=scan_id)


@pytest.fixture
def dashboard_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Monkeypatch `LLMSEC_DASHBOARD_TOKEN` to a fixed test value, returning it."""
    from llmsec.dashboard.auth import DASHBOARD_TOKEN_ENV_VAR

    monkeypatch.setenv(DASHBOARD_TOKEN_ENV_VAR, TEST_DASHBOARD_TOKEN)
    return TEST_DASHBOARD_TOKEN


@pytest.fixture
def client(reports_dir: Path):
    """A `TestClient` wired to `create_app(reports_dir)`."""
    from fastapi.testclient import TestClient

    from llmsec.dashboard.app import create_app

    return TestClient(create_app(reports_dir))


@pytest.fixture
def filter_probe_client(filter_probe_reports_dir: FilterProbeFixture):
    """A `TestClient` wired to `create_app(filter_probe_reports_dir.output_dir)`."""
    from fastapi.testclient import TestClient

    from llmsec.dashboard.app import create_app

    return TestClient(create_app(filter_probe_reports_dir.output_dir))


@pytest.fixture
def corrupted_client(corrupted_reports_dir: CorruptedReportsFixture):
    """A `TestClient` wired to `create_app(corrupted_reports_dir.output_dir)`.

    `raise_server_exceptions=False`: an unhandled `ValidationError` from the
    detail endpoint's deliberately-unwrapped `load_report()` call (T-01-11)
    must surface as a real 500 HTTP response -- matching what a real ASGI
    server (uvicorn) does in production -- rather than re-raising in the
    test process, which is `TestClient`'s debugging-oriented default.
    """
    from fastapi.testclient import TestClient

    from llmsec.dashboard.app import create_app

    return TestClient(create_app(corrupted_reports_dir.output_dir), raise_server_exceptions=False)


@pytest.fixture
def auth_headers(dashboard_token: str) -> dict[str, str]:
    """The bearer `Authorization` header dict for `dashboard_token`."""
    return {"Authorization": f"Bearer {dashboard_token}"}


class RecordedRunScanCall(NamedTuple):
    """One recorded invocation of `controllable_run_scan`'s fake."""

    scan_id: str | None
    config: object


class ControllableRunScan:
    """A controllable async fake for `llmsec.dashboard.scans.run_scan` (Task 2).

    Deliberately gated on two release signals the TEST controls, never a
    fixed-duration `asyncio.sleep()` -- a sleep-based fake makes
    phase-transition assertions flaky (11-VALIDATION.md Wave 0). Uses
    `threading.Event` (not `asyncio.Event`) because `TestClient` runs the
    ASGI app on a background-thread event loop (an anyio blocking portal)
    that outlives any single request/response call -- the test's own
    `.set()` call happens on the MAIN thread, and `asyncio.Event.set()` is
    not safe to call cross-thread against a `Future` owned by a different
    thread's running loop. `threading.Event` is thread-safe by design; the
    fake awaits it via `run_in_executor` so the event loop itself is never
    blocked while waiting.

    The test drives the fake's progress explicitly:

    1. `create_scan` dispatches -> fake records the call, calls
       `on_phase_change("running")`, then awaits `running_event`.
    2. Test releases `running_event` -> fake (unless `raise_error` is set)
       calls `on_phase_change("scoring")`, then awaits `scoring_event`.
    3. Test releases `scoring_event` -> fake writes and returns a minimal
       `ScanReport` under `config.output_dir` (so a completed scan's report
       is readable through the real reports endpoint, D-13), unless
       `raise_error` is set, in which case it raises instead.

    Setting `raise_error` before releasing either event makes the fake
    raise at that point instead of proceeding -- the mechanism
    `test_scan_status_records_failure_without_raising` uses.
    """

    def __init__(self) -> None:
        self.running_event: threading.Event = threading.Event()
        self.scoring_event: threading.Event = threading.Event()
        self.calls: list[RecordedRunScanCall] = []
        self.raise_error: Exception | None = None

    async def __call__(
        self,
        config,
        bypass_flag: bool = False,
        *,
        on_phase_change=None,
        scan_id: str | None = None,
    ) -> ScanReport:
        from llmsec.reporting.json_reporter import JsonReporter

        self.calls.append(RecordedRunScanCall(scan_id=scan_id, config=config))

        loop = asyncio.get_running_loop()

        if on_phase_change is not None:
            on_phase_change("running")
        await loop.run_in_executor(None, self.running_event.wait)
        if self.raise_error is not None:
            raise self.raise_error

        if on_phase_change is not None:
            on_phase_change("scoring")
        await loop.run_in_executor(None, self.scoring_event.wait)
        if self.raise_error is not None:
            raise self.raise_error

        report = ScanReport(
            scan_id=scan_id or uuid.uuid4().hex,
            target_summary="http_app:POST http://example.test/chat",
            module_ids=[],
            findings=[],
            case_log=[],
            started_at="2026-08-19T00:00:00Z",
            completed_at="2026-08-19T00:05:00Z",
        )
        await JsonReporter().write(report, Path(config.output_dir))
        return report


@pytest.fixture
def scans_client(reports_dir: Path):
    """A `TestClient` wired to `create_app(reports_dir)` for the scan-trigger
    tests -- `reports_dir` gives the client a real, inspectable temp
    directory for `output_dir`.

    Used as a context manager (`with TestClient(...) as client:`)
    DELIBERATELY -- a bare `TestClient(app)` gives every individual request
    its OWN throwaway `anyio` portal (and event loop), torn down right
    after that request completes. A background `asyncio.create_task()`
    scheduled inside a request handler lives on THAT throwaway loop, so it
    would be abandoned (and the portal's teardown would hang waiting on it)
    the moment the triggering request returns -- before a later polling
    request could ever observe its progress. Entering `TestClient` as a
    context manager makes it open ONE portal/event loop that persists
    across every request made through this client instance, which is what
    lets a dispatched scan's background task actually run to completion
    while `GET /api/scans/{id}/status` polls it from subsequent requests.
    """
    from fastapi.testclient import TestClient

    from llmsec.dashboard.app import create_app

    with TestClient(create_app(reports_dir)) as client:
        yield client


@pytest.fixture
def controllable_run_scan(
    monkeypatch: pytest.MonkeyPatch, scans_client
) -> Iterator[ControllableRunScan]:
    """Monkeypatch `llmsec.dashboard.scans.run_scan` with a controllable fake.

    Patches the name INSIDE `llmsec.dashboard.scans` (not `llmsec.api`) --
    that module's own `from llmsec.api import run_scan` binding is what
    `dispatch_scan()`'s inner coroutine actually calls.

    Deliberately depends on `scans_client` (even though its body never uses
    the client) purely to fix fixture teardown ORDER: pytest tears fixtures
    down in reverse-setup order, so declaring this dependency guarantees
    `scans_client` is set up first and torn down LAST. On teardown, this
    fixture force-releases both of the fake's gate events BEFORE
    `scans_client`'s own teardown runs -- `scans_client`'s `with
    TestClient(...)` exit closes its persistent portal, which will not
    complete until every task scheduled on that portal's loop (including a
    scan a test left mid-flight, deliberately gated behind an unreleased
    event, such as `test_create_scan_returns_202_and_scan_id_immediately`)
    has finished. Without this ordering the portal close would hang
    forever waiting on a task nothing will ever unblock.
    """
    from llmsec.dashboard import scans as scans_module

    fake = ControllableRunScan()
    monkeypatch.setattr(scans_module, "run_scan", fake)
    yield fake
    fake.running_event.set()
    fake.scoring_event.set()
