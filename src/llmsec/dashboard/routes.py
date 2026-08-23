"""The token-gated `/api/*` router (DASH-03/DASH-04/DASH-05/DASH-01/DASH-02).

The bearer-token guard is declared ONCE, at the `APIRouter` level via
`dependencies=[Depends(verify_bearer_token)]` -- never per-endpoint, so a
route added later (this plan or a future one) cannot be born ungated
(ROADMAP success criterion 4, threat T-10-06/T-11-07).

`scan_id` is treated as untrusted at every endpoint that accepts it:
`is_valid_scan_id()` runs before any `Path` is built (T-10-01), and both the
detail and download handlers raise the SAME fixed 404 detail string for
"malformed id" and "well-formed id, no file" -- indistinguishable to the
client, so the endpoint cannot be used to enumerate which 32-hex ids exist
(T-10-11). `GET /scans/{scan_id}/status` (Phase 11) reuses the identical
discipline via its own `_SCAN_NOT_FOUND_DETAIL`.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from llmsec.api import _target_summary
from llmsec.config import TargetConfig
from llmsec.dashboard.auth import verify_bearer_token
from llmsec.dashboard.reports import (
    FilterOptions,
    ReportSummary,
    distinct_filter_values,
    is_valid_scan_id,
    list_summaries,
    report_path,
)
from llmsec.dashboard.scans import (
    ScanStateStore,
    build_dashboard_scan_config,
    dispatch_scan,
)
from llmsec.dashboard.ssrf_guard import TargetRejected, validate_target_not_internal
from llmsec.detection.judge import DEFAULT_JUDGE_MODEL
from llmsec.models import ScanReport
from llmsec.reporting.json_reporter import load_report

#: Fixed detail string for every scan_id-shaped 404, whether the id is
#: malformed or well-formed with no corresponding file -- both branches use
#: this exact string so the response cannot be used to distinguish "invalid
#: shape" from "no such report" (T-10-11).
_REPORT_NOT_FOUND_DETAIL = "This report doesn't exist or its scan id is invalid."

#: Fixed detail string for `GET /scans/{scan_id}/status` -- shared by a
#: malformed `scan_id` and a well-formed id with no store record (including
#: one lost to a dashboard restart, D-04's accepted limitation, assumption
#: A-02) so the endpoint cannot be used to enumerate which scan ids exist
#: (same T-10-11 discipline as `_REPORT_NOT_FOUND_DETAIL` above).
_SCAN_NOT_FOUND_DETAIL = "This scan doesn't exist or its scan id is invalid."

#: Fixed, non-interpolating detail string for a POST whose
#: `authorization_attested` field is present but not `True` (D-06). Never
#: echoes any part of the submitted body.
_AUTHORIZATION_NOT_ATTESTED_DETAIL = "You must confirm authorization before starting a scan."

logger = logging.getLogger(__name__)


class ScanRequest(BaseModel):
    """The `POST /scans` request body (DASH-01/DASH-02).

    Reuses `TargetConfig` verbatim (DASH-01's "reuses the config schema")
    rather than a parallel, dashboard-local target shape. `authorization_
    attested` carries NO default -- a missing or wrong-typed value produces
    FastAPI's automatic 422, so a client that forgets the field fails
    loudly rather than silently defaulting to non-attestation (which would
    be indistinguishable from an operator who deliberately declined).
    """

    target: TargetConfig
    enabled_modules: list[str] = Field(default_factory=list)
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_api_key_env: str | None = None
    known_system_prompt: str | None = None
    # `le=50` (11-REVIEW.md IN-02): `ScanConfig.max_concurrency` itself has
    # no upper bound (pre-existing, mirrored by design for YAML/CLI use),
    # but this phase newly exposes it to a plain number input in a
    # browser form -- a bearer-token holder could otherwise set an
    # arbitrarily large value and launch a scan opening far more
    # simultaneous connections against the target (and the judge API)
    # than intended. 50 is a generous, documented ceiling, not a
    # calibrated universal value.
    max_concurrency: int = Field(default=5, ge=1, le=50)
    # `strict=True`: Pydantic's default (lax) bool validation coerces
    # strings like "yes"/"true"/"1" into `True` -- exactly the
    # non-bool-value case this field must reject with a 422, not silently
    # accept as attestation (D-06).
    authorization_attested: bool = Field(strict=True)


class ScanCreateResponse(BaseModel):
    scan_id: str


class ScanStatusResponse(BaseModel):
    scan_id: str
    phase: str
    error: str | None = None


def build_api_router(output_dir: Path) -> APIRouter:
    """Build the `/api` router bound to `output_dir` (captured by closure --
    never taken from a request parameter).
    """
    router = APIRouter(prefix="/api", dependencies=[Depends(verify_bearer_token)])
    # Instantiated inside `build_api_router()` and captured by closure,
    # exactly like `output_dir` -- NOT a module-level singleton -- so two
    # `create_app()` calls (the test suite does this per fixture) never
    # share scan state.
    scan_state_store = ScanStateStore()

    @router.get("/reports", response_model=list[ReportSummary])
    def list_reports(
        target: str | None = Query(default=None),
        module: list[str] | None = Query(default=None),
        severity: str | None = Query(default=None),
    ) -> list[ReportSummary]:
        # Filter values are never validated against an enum or rejected as
        # "unknown" -- they never reach the filesystem, so an unrecognised
        # value's worst outcome is an empty result set (D-12's own text,
        # RESEARCH.md Security Domain V5).
        return list_summaries(output_dir, target=target, modules=module, severity=severity)

    @router.get("/filters", response_model=FilterOptions)
    def get_filters() -> FilterOptions:
        return distinct_filter_values(output_dir)

    @router.get("/reports/{scan_id}", response_model=ScanReport)
    def get_report(scan_id: str) -> ScanReport:
        # Validation runs BEFORE any Path is constructed -- a traversal
        # payload never reaches the filesystem layer at all (T-10-01).
        if not is_valid_scan_id(scan_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_REPORT_NOT_FOUND_DETAIL)
        path = report_path(output_dir, scan_id)
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_REPORT_NOT_FOUND_DETAIL)
        # Deliberately UNWRAPPED -- a ValidationError from a corrupted file
        # propagates and surfaces as a 500 (T-01-11). This is the exact
        # asymmetry against the list endpoint's per-file containment: one
        # bad file must never fabricate a valid-looking empty report.
        return load_report(path)

    @router.get("/reports/{scan_id}/download")
    def download_report(scan_id: str) -> FileResponse:
        """Serve the raw on-disk bytes of `scan_<id>.json`.

        Deliberately does NOT call `load_report()` (assumption A-02,
        10-02-PLAN.md `<planner_assumptions>`): this is the escape hatch for
        inspecting a file the detail endpoint just refused to parse, and
        handing over the exact on-disk bytes substitutes nothing -- T-01-11
        forbids fabricating a valid-looking report, not handing over the
        real bytes. `scan_id` is still validated before any path is built,
        with the same validation order as `get_report()` above.
        """
        if not is_valid_scan_id(scan_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_REPORT_NOT_FOUND_DETAIL)
        path = report_path(output_dir, scan_id)
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_REPORT_NOT_FOUND_DETAIL)
        return FileResponse(
            path,
            media_type="application/json",
            filename=f"scan_{scan_id}.json",
        )

    @router.post("/scans", response_model=ScanCreateResponse, status_code=status.HTTP_202_ACCEPTED)
    async def create_scan(body: ScanRequest) -> ScanCreateResponse:
        """Launch an authorized scan and return its `scan_id` immediately.

        Ordering is the security property (DASH-01/DASH-02/DASH-07) --
        implemented in exactly this sequence:
        1. Reject with 400 unless `body.authorization_attested` is exactly
           `True`. Read fresh from THIS request's own body only -- no
           cached/shared attestation state across concurrent requests.
        2. SSRF destination gate via `validate_target_not_internal` for an
           `http_app` target -- for a `raw_llm` target, no call is made at
           all; the branch is structurally absent, not a no-op (D-08).
        3. Build the config via `build_dashboard_scan_config`.
        4. Generate `scan_id`.
        5. `dispatch_scan` -- schedules the background task and returns.
        6. Return `{"scan_id": scan_id}`.

        Nothing before step 3 may construct an adapter, touch the
        filesystem, or create a task.
        """
        if body.authorization_attested is not True:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, detail=_AUTHORIZATION_NOT_ATTESTED_DETAIL
            )

        if body.target.type == "http_app":
            try:
                await validate_target_not_internal(body.target.url or "")
            except TargetRejected as exc:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # `raw_llm` targets never reach `validate_target_not_internal` at
        # all -- `litellm` resolves a provider-qualified model string
        # through its own routing, never an operator-supplied host (D-08).

        config = build_dashboard_scan_config(body, str(output_dir))
        scan_id = uuid.uuid4().hex
        # Audit trail for the one action on this surface capable of sending
        # live traffic to an external target (11-REVIEW.md WR-04):
        # auth_gate.py's confirm_authorization() bypass branch explicitly
        # defers logging to the caller ("explicit bypass -- logged by
        # caller, not silently accepted"), which this dashboard path is --
        # dispatch_scan() below calls run_scan(..., bypass_flag=True, ...)
        # for every request, substituting the attestation gate above for
        # the CLI's interactive prompt. _target_summary() reuses the exact
        # format already used in ScanReport.target_summary, never a raw
        # dict/body dump that could leak a header value or body_template.
        logger.info(
            "Dashboard-authorized scan %s dispatched against %s",
            scan_id,
            _target_summary(config),
        )
        dispatch_scan(scan_id, config, scan_state_store)
        return ScanCreateResponse(scan_id=scan_id)

    @router.get("/scans/{scan_id}/status", response_model=ScanStatusResponse)
    def get_scan_status(scan_id: str) -> ScanStatusResponse:
        # Validation runs BEFORE any store lookup -- a malformed id and a
        # well-formed-but-unrecorded id (including one lost to a dashboard
        # restart, D-04) share the identical 404 detail (T-10-11 discipline).
        if not is_valid_scan_id(scan_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_SCAN_NOT_FOUND_DETAIL)
        state = scan_state_store.get(scan_id)
        if state is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_SCAN_NOT_FOUND_DETAIL)
        return ScanStatusResponse(scan_id=state.scan_id, phase=state.phase, error=state.error)

    return router
