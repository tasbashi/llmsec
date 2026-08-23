"""Report store reads for the dashboard -- list, derive severity, load one.

Every report read funnels through `llmsec.reporting.json_reporter.load_report()`
(D-04) -- this module never re-implements JSON parsing, and never restates
the severity band list (imports `llmsec.scoring.engine.Severity` instead).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Sequence

from pydantic import BaseModel

from llmsec.models import ScanReport
from llmsec.reporting.json_reporter import load_report
from llmsec.scoring.engine import Severity

logger = logging.getLogger(__name__)

# Exactly the shape `uuid.uuid4().hex` produces at `api.py`'s scan-id
# generation site -- 32 lowercase hex characters.
SCAN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def is_valid_scan_id(scan_id: str) -> bool:
    """Return whether `scan_id` matches the exact shape `uuid.uuid4().hex` produces."""
    return bool(SCAN_ID_PATTERN.fullmatch(scan_id))


def report_path(output_dir: Path, scan_id: str) -> Path:
    """Build the on-disk path for `scan_id`, validating it first.

    Raises `ValueError` if `scan_id` is not `is_valid_scan_id()`-shaped --
    so the validation cannot be forgotten at a future call site, unlike
    `cli.py`'s `report_cmd`, whose identical join is safe only because its
    `scan_id` is a trusted local CLI argument rather than untrusted HTTP
    input (RESEARCH.md Pitfall #5).
    """
    if not is_valid_scan_id(scan_id):
        raise ValueError(f"Invalid scan_id: {scan_id!r}")
    return output_dir / f"scan_{scan_id}.json"


def iter_reports(output_dir: Path) -> Iterator[ScanReport]:
    """Yield every readable `ScanReport` under `output_dir`, newest-file-glob order.

    Each `load_report()` call is wrapped in a try/except that logs at
    WARNING with `exc_info=True`, names the skipped path, and continues --
    the list surface degrades per-file (RESEARCH.md Pitfall #6) while the
    single-report surface (`report_path()` + `load_report()` used directly)
    keeps `load_report()`'s loud failure (T-01-11). A non-existent
    `output_dir` yields nothing rather than raising.
    """
    if not output_dir.is_dir():
        return
    for path in sorted(output_dir.glob("scan_*.json")):
        try:
            yield load_report(path)
        except Exception:
            logger.warning("Skipping unreadable report %s", path, exc_info=True)


def max_severity(report: ScanReport) -> str:
    """Return the highest severity band among `report.findings` (D-11).

    `Severity.NONE.value` for an empty `findings` list (RESEARCH.md
    Pitfall #3 -- a scan where every case was `BLOCKED` never appends a
    `Finding`, so `findings == []` is a real, reachable state, not an
    error). The ranking comes from `Severity`'s own declaration order via
    `list(Severity).index(...)`, never a restated list.
    """
    if not report.findings:
        return Severity.NONE.value
    order = list(Severity)
    return max(report.findings, key=lambda f: order.index(Severity(f.severity))).severity


class ReportSummary(BaseModel):
    """The list-row projection of a `ScanReport`.

    Deliberately omits `findings`, `case_log`, `limitations` and
    `deep_summary` so a list request never ships every report's full
    evidence text to the browser.
    """

    scan_id: str
    target_summary: str
    module_ids: list[str]
    max_severity: str
    finding_count: int
    started_at: str
    completed_at: str


def summarize(report: ScanReport) -> ReportSummary:
    """Build a `ReportSummary` from a full `ScanReport`."""
    return ReportSummary(
        scan_id=report.scan_id,
        target_summary=report.target_summary,
        module_ids=report.module_ids,
        max_severity=max_severity(report),
        finding_count=len(report.findings),
        started_at=report.started_at,
        completed_at=report.completed_at,
    )


class FilterOptions(BaseModel):
    """The distinct filter values actually present across stored reports.

    D-09: these are server-computed from the reports that exist on disk, so
    the client renders dropdowns rather than deriving options itself or
    offering free-text search against a free-form string.
    """

    targets: list[str]
    modules: list[str]
    severities: list[str]


def matches_filters(
    summary: ReportSummary,
    target: str | None = None,
    modules: Sequence[str] | None = None,
    severity: str | None = None,
) -> bool:
    """Return whether `summary` satisfies every supplied filter.

    Target (single-select), module (multi-select) and severity
    (single-select) combine with AND/intersection semantics (D-12): a
    supplied filter must hold, an omitted or empty one is a no-op.

    - `target` is exact string equality against `summary.target_summary` --
      never a substring or case-insensitive match. D-09 populates the
      dropdown from these exact values, so any looser comparison could only
      ever produce a surprise.
    - `modules` is a superset test: every id in `modules` must be present in
      `summary.module_ids` (assumption A-01, 10-02-PLAN.md
      `<planner_assumptions>` -- selecting two modules narrows the result
      set, it never widens it).
    - `severity` is exact equality against `summary.max_severity`, which is
      the report's MAXIMUM band per D-11, never a containment test.
    """
    if target and summary.target_summary != target:
        return False
    if modules and not set(modules).issubset(summary.module_ids):
        return False
    if severity and summary.max_severity != severity:
        return False
    return True


def list_summaries(
    output_dir: Path,
    *,
    target: str | None = None,
    modules: Sequence[str] | None = None,
    severity: str | None = None,
) -> list[ReportSummary]:
    """List every readable report under `output_dir` that satisfies every
    supplied filter, newest-first by `completed_at`.

    Two reports sharing an identical `completed_at` are ordered by
    `scan_id` ascending -- a repeatable, deterministic tie-break (D-10 plus
    the DASH-04 ordering edge), implemented as two successive STABLE sorts
    (scan_id ascending first, then completed_at descending) rather than one
    `reverse=True` tuple key, which would also flip the tie-break direction.

    A non-existent `output_dir` yields `[]` through `iter_reports()`, never
    an exception.
    """
    summaries = [
        summary
        for summary in (summarize(report) for report in iter_reports(output_dir))
        if matches_filters(summary, target, modules, severity)
    ]
    summaries.sort(key=lambda s: s.scan_id)
    summaries.sort(key=lambda s: s.completed_at, reverse=True)
    return summaries


def distinct_filter_values(output_dir: Path) -> FilterOptions:
    """Compute the distinct `target_summary`, `module_ids` and derived
    `max_severity` values actually present across every readable report
    under `output_dir` -- one pass over `iter_reports()`.

    Severities are ordered by the `Severity` enum's OWN declaration order,
    most-severe-first (never restated as a literal list), so the `none`
    band (Pitfall #3, RESEARCH.md) is always a selectable, reachable option
    whenever any stored report has no findings.
    """
    targets: set[str] = set()
    modules: set[str] = set()
    severities: set[str] = set()
    for report in iter_reports(output_dir):
        targets.add(report.target_summary)
        modules.update(report.module_ids)
        severities.add(max_severity(report))

    severity_order = list(Severity)
    return FilterOptions(
        targets=sorted(targets),
        modules=sorted(modules),
        severities=sorted(
            severities,
            key=lambda value: severity_order.index(Severity(value)),
            reverse=True,
        ),
    )
