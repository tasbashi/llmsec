import { useState } from "react";
import { api, ApiError } from "../api.js";
import SeverityBadge from "./SeverityBadge.jsx";
import SeverityChart from "./SeverityChart.jsx";
import RawCases from "./RawCases.jsx";

// Same fixed band order the Markdown reporter's `_count_by_severity()` and
// the `Severity` enum declare, reversed -- "worst first" for the findings
// list (D-13's severity-descending sort), matching the report-list's own
// max-severity-descending intuition.
const SEVERITY_SORT_ORDER = ["critical", "high", "medium", "low", "none"];

function severityRank(severity) {
  const index = SEVERITY_SORT_ORDER.indexOf(severity);
  return index === -1 ? SEVERITY_SORT_ORDER.length : index;
}

function findingsHeading(count) {
  return count === 1 ? "1 finding" : `${count} findings`;
}

// A collapsed one-line mono row that expands in place to reveal evidence +
// remediation in the sans body face -- mono is for identifiers, never for
// prose (report-and-findings-display.md), so this is the one place inside
// an otherwise-terminal screen that deliberately switches face. Owns only
// its own local expanded/collapsed state; the list above never re-sorts or
// re-filters -- it always renders `sortedFindings` in place.
function FindingRow({ finding }) {
  const [expanded, setExpanded] = useState(false);

  function toggle() {
    setExpanded((prev) => !prev);
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={toggle}
        onKeyDown={handleKeyDown}
        className="flex items-center gap-sm px-sm py-xs font-mono text-body cursor-pointer hover:bg-surface-raised"
      >
        <span className="w-4 shrink-0 text-text-faint" aria-hidden="true">
          {expanded ? "▾" : "▸"}
        </span>
        <SeverityBadge severity={finding.severity} />
        <span className="flex-1 truncate" title={finding.technique_id}>
          {finding.technique_id}
        </span>
        <span className="w-24 shrink-0 text-text-faint">{finding.owasp_ref}</span>
        <span className="w-28 shrink-0 text-right text-text-faint">
          {finding.detection_layer ?? "—"}
        </span>
      </div>
      {expanded && (
        <div className="flex flex-col gap-xs px-sm py-sm pl-xl font-sans text-body text-text-muted">
          <p>
            <span className="text-text-faint">evidence</span> — {finding.evidence}
          </p>
          <p>
            <span className="text-text-faint">remediation</span> — {finding.remediation}
          </p>
        </div>
      )}
    </div>
  );
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

  // loading: one treatment for the whole page -- the findings list, the
  // histogram and the raw-cases toggle all gate on this single fetch, so
  // there is exactly one skeleton here, not three.
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
      {/* Header: mono llmsec://report/{target} title mirroring the shell's
          own brand language, a sans metadata line beneath it (the values
          themselves stay mono since they're identifiers), the download
          action on the right. */}
      <header className="flex items-start justify-between gap-md flex-wrap">
        <div className="flex flex-col gap-xs">
          <h1 className="font-mono text-display break-all">
            <span className="text-text-faint">llmsec://report/</span>
            <span className="text-text">{report.target_summary}</span>
          </h1>
          <div className="font-sans text-label text-text-faint">
            <span className="font-mono">{report.scan_id}</span>
            {" · started "}
            <span className="font-mono">{report.started_at}</span>
            {" · completed "}
            <span className="font-mono">{report.completed_at}</span>
            {" · modules: "}
            <span className="font-mono">{report.module_ids.join(", ")}</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-xs">
          <button
            type="button"
            onClick={handleDownload}
            disabled={downloading}
            className="bg-accent text-body font-mono rounded-sm px-md py-sm hover:bg-accent-hover disabled:opacity-50"
          >
            {downloading ? "Downloading…" : "↓ download.json"}
          </button>
          {downloadError && (
            <p className="text-body text-destructive">
              Couldn't download this report. Try again.
            </p>
          )}
        </div>
      </header>

      {/* Findings: the focal element, full width, immediately below the
          header (UI-SPEC visual hierarchy). Collapsed one-line mono rows
          that expand in place -- sort order unchanged, still driven by the
          single SEVERITY_SORT_ORDER constant above. */}
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
          <div className="flex flex-col divide-y divide-border">
            {sortedFindings.map((finding) => (
              <FindingRow key={finding.case_id} finding={finding} />
            ))}
          </div>
        )}
      </section>

      {/* Severity breakdown: secondary to the findings list (UI-SPEC visual
          hierarchy), separated by the 2xl spacing token above. The card
          chrome lives here so the zero-findings message SeverityChart
          returns renders inside it, in both the populated and empty case. */}
      <section className="flex flex-col gap-md">
        <div className="font-mono text-label text-text-faint">SEVERITY BREAKDOWN</div>
        <div className="bg-surface-secondary border border-border rounded-md px-lg py-md">
          <SeverityChart findings={report.findings} />
        </div>
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
