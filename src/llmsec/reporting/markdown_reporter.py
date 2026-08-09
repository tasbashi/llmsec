"""Jinja2-templated human-readable Markdown reporter (REPORT-02).

Mirrors `JsonReporter`'s `write(report, output_dir) -> Path` shape so
callers (plan 08's `run_scan()`, plan 09's `report_cmd`) can treat both
reporters interchangeably via `BaseReporter`.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import jinja2

from llmsec.models import DeepModeSummary, Finding, ScanReport
from llmsec.reporting.base import BaseReporter
from llmsec.scoring.engine import Severity

# ASVS V6 / PITFALLS P10-D: same restrictive output_dir permissions as
# JsonReporter, since Markdown output may also contain leaked
# system-prompt content.
_OUTPUT_DIR_MODE = 0o700

_TEMPLATE_NAME = "report.md.j2"


class MarkdownReporter(BaseReporter):
    """Renders a `ScanReport` as Markdown via a Jinja2 template."""

    def __init__(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.PackageLoader("llmsec.reporting", "templates"),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def write(self, report: ScanReport, output_dir: Path) -> Path:
        # WR-05: `Path.mkdir(mode=...)` only applies `mode` when the
        # directory is actually created — if `output_dir` already exists
        # (pre-created by an operator, or left looser by a prior run's
        # umask), `exist_ok=True` silently skips tightening its
        # permissions. Explicitly `chmod` afterward so the restrictive
        # mode invariant (ASVS V6 / PITFALLS P10-D: Markdown output may
        # contain leaked system-prompt content) holds unconditionally.
        output_dir.mkdir(parents=True, exist_ok=True, mode=_OUTPUT_DIR_MODE)
        os.chmod(output_dir, _OUTPUT_DIR_MODE)
        template = self._env.get_template(_TEMPLATE_NAME)
        severity_counts = _count_by_severity(report)
        # T-02-31: pre-fence every finding's evidence in Python (not inside
        # the template) so target-controlled text containing a heading
        # marker or a fenced-code delimiter cannot escape its own bullet
        # and corrupt the surrounding document structure. Parallel list,
        # indexed by `loop.index0` against `report.findings` in the
        # template.
        evidence_blocks = [_safe_evidence_block(finding.evidence) for finding in report.findings]
        # IN-03 (02-REVIEW.md): `finding.remediation` gets the same
        # heading/fence-neutralizing treatment as `finding.evidence` above.
        # `remediation` is only ever framework-authored today (the YAML
        # corpus or `_REMEDIATION_BY_VERDICT`), but `EvalResult.remediation`
        # is a free-form `str | None` a third-party module could set to
        # arbitrary text, so this must not rely on that being true forever.
        remediation_blocks = [_safe_evidence_block(finding.remediation) for finding in report.findings]
        # D-91 (05-09): computed in Python, never in the template, matching
        # the existing `_count_by_severity()`/`_safe_evidence_block()`
        # convention above. `None` exactly when `report.deep_summary` is
        # `None` (a static-only run) -- the template's own `{% if deep_mode
        # %}` guard then renders nothing at all, keeping a static-only
        # report's Markdown byte-identical to its pre-phase-5 shape (D-93).
        findings_by_case_id = {finding.case_id: finding for finding in report.findings}
        deep_mode = _deep_mode_display(report.deep_summary, findings_by_case_id)
        rendered = template.render(
            report=report,
            severity_counts=severity_counts,
            evidence_blocks=evidence_blocks,
            remediation_blocks=remediation_blocks,
            deep_mode=deep_mode,
        )
        path = output_dir / f"scan_{report.scan_id}.md"
        path.write_text(rendered)
        return path


def _count_by_severity(report: ScanReport) -> OrderedDict[str, int]:
    """Compute finding counts per Severity band, in fixed band order.

    Counts are computed in Python (not inside the Jinja template) per the
    plan's action spec.
    """
    counts: OrderedDict[str, int] = OrderedDict((severity.value, 0) for severity in Severity)
    for finding in report.findings:
        severity_value = finding.severity
        counts[severity_value] = counts.get(severity_value, 0) + 1
    return counts


def _longest_backtick_run(text: str) -> int:
    """Return the length of the longest consecutive run of backticks in `text`."""
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _safe_evidence_block(evidence: str) -> str:
    """Wrap `evidence` in a fenced code block whose delimiter is longer than
    any backtick run present in the text (T-02-31).

    `autoescape=False` is correct for a Markdown reporter, but
    target-controlled evidence can otherwise contain heading markers
    (`#...`) or fence delimiters (```` ``` ````) that break the surrounding
    document. A fenced code block absorbs every line up to its own closing
    fence — including lines that would otherwise be interpreted as
    headings — so the only remaining risk is evidence containing a run of
    backticks long enough to prematurely close the block; the fence length
    here always exceeds the longest such run, with a floor of 3 backticks.
    """
    fence = "`" * max(3, _longest_backtick_run(evidence) + 1)
    return f"{fence}\n{evidence}\n{fence}"


def _deep_mode_display(
    summary: DeepModeSummary | None, findings_by_case_id: dict[str, Finding]
) -> dict[str, Any] | None:
    """Precompute every Deep Mode section display value in Python (05-09,
    D-91) -- `None` exactly when `summary` is `None`, so the template's own
    `{% if deep_mode %}` guard omits the whole section for a static-only
    report rather than rendering an empty one (D-93).

    None of `DeepModeSummary`'s own fields carry target-influenced free
    text (every figure is a count, a role name, a fixed `TerminationReason`
    literal, a `case_id`, or a filesystem path -- never target response
    prose), so this section needs no `_safe_evidence_block()`-style
    fence-neutralizing treatment; that discipline stays scoped to
    `finding.evidence`/`finding.remediation` above, which genuinely are
    target-influenced.
    """
    if summary is None:
        return None

    cost_per_bypass_display = (
        f"${summary.cost_per_bypass_usd:.4f}"
        if summary.cost_per_bypass_usd is not None
        else "N/A -- no bypasses found this run"
    )
    audit_log_display = (
        summary.audit_log_path
        if summary.audit_log_path
        else "No audit log written (no attacker-eligible work this run)"
    )
    termination_reason_display = summary.termination_reason or "unknown"

    per_role_rows = [
        {
            "role": role,
            "calls": activity.calls,
            "spend_display": f"${activity.spend_usd:.4f}",
            "bypasses": activity.bypasses,
        }
        for role, activity in summary.per_role_activity.items()
    ]

    # Each bypass paired with its parent case id (05-09 Task 3's own
    # requirement: "pass the bypass list pre-paired with parent case
    # ids") -- looked up from `report.findings`, the one place a bypass
    # Finding's D-90 `parent_case_id` lineage field actually lives;
    # `DeepModeSummary` itself carries only the flat `bypass_case_ids` list.
    bypass_rows = []
    for case_id in summary.bypass_case_ids:
        finding = findings_by_case_id.get(case_id)
        parent_case_id = finding.parent_case_id if finding is not None else "unknown"
        bypass_rows.append({"case_id": case_id, "parent_case_id": parent_case_id})

    return {
        "cases_attacked": summary.cases_attacked,
        "rounds_run": summary.rounds_run,
        "variants_dispatched": summary.variants_dispatched,
        "bypasses_found": summary.bypasses_found,
        # Labelled distinctly from the pre-run estimate wording
        # (`attacker.budget.render_cost_notice()`'s "ESTIMATES, not
        # quotes") so the two numbers can never be confused (D-82/D-91).
        "spend_display": f"${summary.spend_usd:.4f} (counted actual spend, not an estimate)",
        "cost_per_bypass_display": cost_per_bypass_display,
        "agent_calls": summary.agent_calls,
        "termination_reason_display": termination_reason_display,
        "constraint_violations": summary.constraint_violations,
        "abandoned_arcs": summary.abandoned_arcs,
        "role_structural_failures": summary.role_structural_failures,
        "audit_log_display": audit_log_display,
        "per_role_rows": per_role_rows,
        "bypass_rows": bypass_rows,
    }
