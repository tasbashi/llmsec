"""Pins the SPECIFICATION `SeverityChart.jsx` implements, not its runtime.

`SeverityChart.jsx` is specified to compute one count per `Severity` band,
in the enum's own declaration order, over `report.findings` -- structurally
identical to `llmsec.reporting.markdown_reporter._count_by_severity()`
(threat T-10-22: the browser must never become a second, divergent severity
aggregator, ROADMAP success criterion 3). This module reimplements that
band-counting loop in pure Python (no JSX execution, no browser) and
asserts it agrees with `_count_by_severity()` for several fixture reports,
including the zero-findings edge (Pitfall #3) and a report whose findings
occupy only one band.

The runtime side -- does the actual React component render this correctly
-- is out of scope here and is covered instead by the 10-06 blocking
`checkpoint:human-verify` (10-04-PLAN.md assumption A-04's compensating
controls: no JS test runner this phase).
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from llmsec.models import Finding, ScanReport, Verdict
from llmsec.reporting.markdown_reporter import _count_by_severity
from llmsec.scoring.engine import Severity


def _finding(case_id: str, severity: str) -> Finding:
    return Finding(
        case_id=case_id,
        technique_id="direct_ask",
        verdict=Verdict.FULL_COMPROMISE,
        severity=severity,
        owasp_ref="LLM07",
        evidence="evidence",
        remediation="remediation",
    )


def _report(scan_id: str, findings: list[Finding]) -> ScanReport:
    return ScanReport(
        scan_id=scan_id,
        target_summary="raw_llm target",
        module_ids=["system_prompt_leakage"],
        findings=findings,
        case_log=[],
        started_at="2026-08-18T00:00:00Z",
        completed_at="2026-08-18T00:05:00Z",
    )


def _chart_specified_counts(report: ScanReport) -> OrderedDict[str, int]:
    """Re-derive the band-count mapping `SeverityChart.jsx` is specified to
    display: one entry per `Severity` band, in the enum's declaration
    order, incremented once per matching finding. Deliberately
    reimplemented rather than delegated to `_count_by_severity()` so this
    test would actually fail if the two ever diverged.
    """
    counts: OrderedDict[str, int] = OrderedDict((severity.value, 0) for severity in Severity)
    for finding in report.findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


FIXTURE_REPORTS = {
    # Pitfall #3: every case BLOCKED -- findings == []. The chart must
    # still show all five bands at zero, never an absent/degenerate chart.
    "zero_findings": _report("scan-zero-findings", []),
    # Findings occupy exactly one band -- the other four must still read 0,
    # not be omitted from the mapping.
    "single_band": _report(
        "scan-single-band",
        [_finding("case-1", "high"), _finding("case-2", "high")],
    ),
    "multi_band": _report(
        "scan-multi-band",
        [
            _finding("case-1", "low"),
            _finding("case-2", "medium"),
            _finding("case-3", "high"),
            _finding("case-4", "critical"),
            _finding("case-5", "critical"),
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(FIXTURE_REPORTS))
def test_chart_specified_counts_agree_with_markdown_reporter(name: str) -> None:
    report = FIXTURE_REPORTS[name]
    assert _chart_specified_counts(report) == _count_by_severity(report)


def test_band_order_matches_severity_enum_declaration_order() -> None:
    """The chart's fixed band order is not an independent choice -- it must
    be exactly the `Severity` enum's declaration order, the same order
    `_count_by_severity()` iterates, so the two are aligned band-for-band
    rather than merely equal as unordered mappings.
    """
    assert list(_count_by_severity(FIXTURE_REPORTS["zero_findings"]).keys()) == [
        severity.value for severity in Severity
    ]
