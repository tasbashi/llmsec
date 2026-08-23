import SeverityBadge from "./SeverityBadge.jsx";

const SKELETON_ROW_COUNT = 4;

function TableShell({ children }) {
  return (
    <table className="w-full text-body">
      <thead>
        <tr className="text-label uppercase tracking-wide text-left">
          <th className="px-md py-sm">Target</th>
          <th className="px-md py-sm">Modules</th>
          <th className="px-md py-sm">Severity</th>
          <th className="px-md py-sm">Completed</th>
        </tr>
      </thead>
      <tbody>{children}</tbody>
    </table>
  );
}

function SkeletonRow({ index }) {
  return (
    <tr className="border-t border-surface-secondary" aria-hidden="true">
      <td className="px-md py-sm">
        <div className="h-4 w-40 rounded bg-surface-secondary animate-pulse" />
      </td>
      <td className="px-md py-sm">
        <div className="h-4 w-28 rounded bg-surface-secondary animate-pulse" />
      </td>
      <td className="px-md py-sm">
        <div className="h-4 w-20 rounded bg-surface-secondary animate-pulse" />
      </td>
      <td className="px-md py-sm">
        <div className="h-4 w-32 rounded bg-surface-secondary animate-pulse" />
      </td>
    </tr>
  );
}

// One component, five mutually exclusive render branches driven by `state`
// and `reports`/`hasAnyFilterActive` -- each branch is its own JSX return
// expression with its own verbatim UI-SPEC copy, never a single generic
// empty-state parameterized by a message string (that one edit away from
// collapsing the two distinct empty states the UI-SPEC forbids merging).
export default function ReportList({ reports, state, hasAnyFilterActive, onRetry, onSelect }) {
  // loading: same row height/column widths/vertical rhythm as a real row so
  // replacing skeletons with real rows causes no layout shift.
  if (state === "loading") {
    return (
      <TableShell>
        {Array.from({ length: SKELETON_ROW_COUNT }, (_, index) => (
          <SkeletonRow key={index} index={index} />
        ))}
      </TableShell>
    );
  }

  // error: retry-able, never a generic app-wide error.
  if (state === "error") {
    return (
      <div className="p-xl text-center flex flex-col items-center gap-md">
        <p className="text-body">Couldn't load reports. Check your connection and try again.</p>
        <button
          type="button"
          onClick={onRetry}
          className="text-body px-md py-sm rounded border border-surface-secondary hover:border-accent"
        >
          Retry
        </button>
      </div>
    );
  }

  // ready, no rows, no active filter: this directory genuinely has no
  // reports yet -- a separate branch/literal from the filters-excluded case
  // below, per the UI-SPEC's explicit "never collapse into one generic
  // message" requirement.
  if (reports.length === 0 && !hasAnyFilterActive) {
    return (
      <div className="p-xl text-center">
        <h2 className="text-heading">No scan reports yet</h2>
        <p className="text-body">
          Run `llmsec scan` from the CLI to generate your first report, then refresh this page.
        </p>
      </div>
    );
  }

  // ready, no rows, an active filter excludes everything: a structurally
  // separate branch and separate literal from the case above.
  if (reports.length === 0 && hasAnyFilterActive) {
    return (
      <div className="p-xl text-center">
        <h2 className="text-heading">No reports match these filters</h2>
        <p className="text-body">Try clearing one or more filters to see more results.</p>
      </div>
    );
  }

  // ready, populated.
  const countLabel = reports.length === 1 ? "1 report" : `${reports.length} reports`;

  function handleRowKeyDown(event, scanId) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(scanId);
    }
  }

  return (
    <div className="flex flex-col gap-sm">
      <p className="text-label uppercase tracking-wide">{countLabel}</p>
      <TableShell>
        {reports.map((report) => (
          <tr
            key={report.scan_id}
            role="button"
            tabIndex={0}
            onClick={() => onSelect(report.scan_id)}
            onKeyDown={(event) => handleRowKeyDown(event, report.scan_id)}
            className="group border-t border-surface-secondary cursor-pointer hover:bg-surface-secondary focus:bg-surface-secondary focus:outline-none"
          >
            {/* Fixed max width + truncate + title: a long free-form
                target_summary (D-09 calls this out as unstructured text)
                cannot widen the table or wrap the row. */}
            <td className="px-md py-sm max-w-[280px] truncate" title={report.target_summary}>
              {report.target_summary}
            </td>
            <td
              className="px-md py-sm max-w-[200px] truncate"
              title={report.module_ids.join(", ")}
            >
              {report.module_ids.join(", ")}
            </td>
            <td className="px-md py-sm">
              <SeverityBadge severity={report.max_severity} />
            </td>
            <td className="px-md py-sm font-mono whitespace-nowrap">
              <div className="flex items-center justify-between gap-sm">
                <span>{report.completed_at}</span>
                <span
                  aria-hidden="true"
                  className="text-accent opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100"
                >
                  View →
                </span>
              </div>
            </td>
          </tr>
        ))}
      </TableShell>
    </div>
  );
}
