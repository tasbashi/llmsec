import { useState } from "react";
import { api, ApiError } from "../api.js";
import SeverityBadge from "./SeverityBadge.jsx";
import SeverityChart from "./SeverityChart.jsx";
import RawCases from "./RawCases.jsx";

// Same fixed band order the Markdown reporter's `_count_by_severity()` and
// the `Severity` enum declare, reversed -- "worst first" for the findings
// table (D-13's severity-descending sort), matching the report-list's own
// max-severity-descending intuition.
const SEVERITY_SORT_ORDER = ["critical", "high", "medium", "low", "none"];

function severityRank(severity) {
  const index = SEVERITY_SORT_ORDER.indexOf(severity);
  return index === -1 ? SEVERITY_SORT_ORDER.length : index;
}

function findingsHeading(count) {
  return count === 1 ? "1 finding" : `${count} findings`;
}

// App.jsx owns the "← Back to reports" control (rendered above this
// component in every state) -- this component owns everything else: the
// three render branches on `state`, then the full detail layout.
export default function ReportDetail({ report, state, errorStatus, scanId, onUnauthorized }) {
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState(false);

  async function handleDownload() {
    setDownloading(true);
    setDownloadError(false);
    try {
      await api.downloadReport(scanId);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        onUnauthorized();
        return;
      }
      setDownloadError(true);
    } finally {
      setDownloading(false);
    }
  }

  // loading: one treatment for the whole page -- the findings table, the
  // chart and the raw-cases toggle all gate on this single fetch, so there
  // is exactly one skeleton here, not three.
  if (state === "loading") {
    return (
      <div className="px-xl py-lg flex flex-col gap-lg">
        <div className="h-8 w-96 rounded bg-surface-secondary animate-pulse" />
        <div className="h-64 w-full rounded bg-surface-secondary animate-pulse" />
      </div>
    );
  }

  // error: two structurally separate branches/literals, never one
  // parameterized message -- a 404 and a 500 mean different things
  // (D-07's precedent) and must stay distinguishable to the reader.
  if (state === "error") {
    let message;
    if (errorStatus === 404) {
      message = "This report doesn't exist or its scan id is invalid.";
    } else if (errorStatus === 500) {
      message = `This report's file is corrupted and can't be read. Check the server logs for scan_${scanId}.json.`;
    } else {
      message = "Couldn't load this report. Check your connection and try again.";
    }
    return (
      <div className="px-xl py-lg">
        <p className="text-body">{message}</p>
      </div>
    );
  }

  if (!report) {
    // Unreachable in practice ("ready" always ships with a report), but
    // keeps this component defensive rather than crashing on a null deref.
    return null;
  }

  const sortedFindings = [...report.findings].sort(
    (a, b) => severityRank(a.severity) - severityRank(b.severity)
  );

  const findingsByCaseId = Object.fromEntries(report.findings.map((f) => [f.case_id, f]));
  const deep = report.deep_summary;

  return (
    <div className="px-xl py-lg flex flex-col gap-2xl">
      {/* Header: target_summary at display size, metadata beneath it,
          Download JSON action on the right. */}
      <header className="flex items-start justify-between gap-md flex-wrap">
        <div className="flex flex-col gap-xs">
          <h1 className="text-display">{report.target_summary}</h1>
          <div className="flex flex-col gap-xs text-label text-body">
            <span className="font-mono">Scan ID: {report.scan_id}</span>
            <span>Modules: {report.module_ids.join(", ")}</span>
            <span className="font-mono">Started: {report.started_at}</span>
            <span className="font-mono">Completed: {report.completed_at}</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-xs">
          <button
            type="button"
            onClick={handleDownload}
            disabled={downloading}
            className="bg-accent text-body rounded px-md py-sm disabled:opacity-50"
          >
            {downloading ? "Downloading…" : "Download JSON"}
          </button>
          {downloadError && (
            <p className="text-body text-destructive">
              Couldn't download this report. Try again.
            </p>
          )}
        </div>
      </header>

      {/* Findings: the focal element, full width, immediately below the
          header (UI-SPEC visual hierarchy). */}
      <section className="flex flex-col gap-md">
        <div className="flex items-baseline gap-md">
          <h2 className="text-heading">Findings</h2>
          <span className="text-label uppercase tracking-wide">
            {findingsHeading(sortedFindings.length)}
          </span>
        </div>
        {sortedFindings.length === 0 ? (
          <p className="text-body">No findings — every test case in this scan was blocked.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-body table-fixed">
              <thead>
                <tr className="text-label uppercase tracking-wide text-left">
                  <th className="px-md py-sm w-32">Severity</th>
                  <th className="px-md py-sm w-40">Technique</th>
                  <th className="px-md py-sm w-24">OWASP Ref</th>
                  <th className="px-md py-sm w-32">Detection Layer</th>
                  <th className="px-md py-sm">Evidence</th>
                  <th className="px-md py-sm">Remediation</th>
                </tr>
              </thead>
              <tbody>
                {sortedFindings.map((finding) => (
                  <tr key={finding.case_id} className="border-t border-surface-secondary align-top">
                    <td className="px-md py-sm">
                      <SeverityBadge severity={finding.severity} />
                    </td>
                    <td className="px-md py-sm font-mono truncate" title={finding.technique_id}>
                      {finding.technique_id}
                    </td>
                    <td className="px-md py-sm font-mono">{finding.owasp_ref}</td>
                    <td className="px-md py-sm">{finding.detection_layer ?? "—"}</td>
                    {/* Long free-form target-derived strings (threat
                        T-10-09): wrap in place, never widen the table, and
                        reach the DOM only as a plain JSX text child. */}
                    <td className="px-md py-sm whitespace-pre-wrap break-words">
                      {finding.evidence}
                    </td>
                    <td className="px-md py-sm whitespace-pre-wrap break-words">
                      {finding.remediation}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Severity breakdown: secondary to the findings table (UI-SPEC
          visual hierarchy), separated by the 2xl spacing token above. */}
      <section className="flex flex-col gap-md">
        <h2 className="text-heading">Severity Breakdown</h2>
        <SeverityChart findings={report.findings} />
      </section>

      {/* Scan limitations: honest-degradation disclosures. Omitted
          entirely when empty; never truncated/clamped at any length
          (prohibition P-04) -- no height cap, no line-count cap, no
          overflow-hiding utility, no ellipsis affordance on this
          container or its items. */}
      {report.limitations.length > 0 && (
        <section className="flex flex-col gap-md" data-testid="scan-limitations">
          <h2 className="text-heading">Scan Limitations</h2>
          <ul className="flex flex-col gap-sm list-disc pl-lg">
            {report.limitations.map((item, index) => (
              <li key={index} className="text-body">
                {item}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Deep mode summary: rendered only when deep_summary is non-null --
          omitted entirely (no heading, no empty box) for a pre-v1.1 or
          `--quick` report. Same render-in-full rule as limitations above
          applies to this prose (prohibition P-04). */}
      {deep && (
        <section className="flex flex-col gap-md" data-testid="deep-summary">
          <h2 className="text-heading">Deep Mode Summary</h2>
          <div className="grid grid-cols-2 gap-sm text-body">
            <p>Cases attacked: {deep.cases_attacked}</p>
            <p>Rounds run: {deep.rounds_run}</p>
            <p>Variants dispatched: {deep.variants_dispatched}</p>
            <p>Bypasses found: {deep.bypasses_found}</p>
            <p>Spend: ${deep.spend_usd.toFixed(4)}</p>
            <p>
              Cost per bypass:{" "}
              {deep.cost_per_bypass_usd !== null
                ? `$${deep.cost_per_bypass_usd.toFixed(4)}`
                : "N/A — no bypasses found this run"}
            </p>
            <p>Agent calls: {deep.agent_calls}</p>
            <p>Termination reason: {deep.termination_reason ?? "unknown"}</p>
            <p>Constraint violations: {deep.constraint_violations}</p>
            <p>Abandoned arcs: {deep.abandoned_arcs}</p>
            <p>Role structural failures: {deep.role_structural_failures}</p>
            <p>Audit log: {deep.audit_log_path ?? "No audit log written"}</p>
          </div>

          <h3 className="text-body font-semibold">Per-Role Activity</h3>
          <table className="w-full text-body">
            <thead>
              <tr className="text-label uppercase tracking-wide text-left">
                <th className="px-md py-sm">Role</th>
                <th className="px-md py-sm">Calls</th>
                <th className="px-md py-sm">Spend (USD)</th>
                <th className="px-md py-sm">Bypasses</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(deep.per_role_activity).map(([role, activity]) => (
                <tr key={role} className="border-t border-surface-secondary">
                  <td className="px-md py-sm">{role}</td>
                  <td className="px-md py-sm">{activity.calls}</td>
                  <td className="px-md py-sm">${activity.spend_usd.toFixed(4)}</td>
                  <td className="px-md py-sm">{activity.bypasses}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3 className="text-body font-semibold">Bypasses</h3>
          {deep.bypass_case_ids.length === 0 ? (
            <p className="text-body">No bypasses found this run.</p>
          ) : (
            <ul className="flex flex-col gap-xs list-disc pl-lg font-mono text-body">
              {deep.bypass_case_ids.map((caseId) => (
                <li key={caseId}>
                  {caseId} (parent: {findingsByCaseId[caseId]?.parent_case_id ?? "unknown"})
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {/* Raw cases: last, collapsed by default (D-14). Same detail payload
          -- no second fetch. */}
      <RawCases caseLog={report.case_log} />
    </div>
  );
}
