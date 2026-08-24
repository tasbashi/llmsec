import SeverityBadge from "./SeverityBadge.jsx";

const SKELETON_ROW_COUNT = 4;

// Shared row-shaped rhythm so a skeleton placeholder and a real row occupy
// identical height/padding/gap -- swapping one for the other causes no
// layout shift (shell-and-navigation.md's pinned row CSS).
const ROW_CLASS = "flex items-center gap-md px-sm py-xs rounded-sm";

function SkeletonRow() {
  return (
    <div className={ROW_CLASS} aria-hidden="true">
      <div className="w-4 shrink-0" />
      <div className="h-4 w-20 rounded bg-surface-secondary animate-pulse" />
      <div className="h-4 flex-1 rounded bg-surface-secondary animate-pulse" />
      <div className="h-4 w-32 rounded bg-surface-secondary animate-pulse" />
      <div className="h-4 w-16 shrink-0 rounded bg-surface-secondary animate-pulse" />
    </div>
  );
}

function ReportRow({ report, isFirst, onSelect, onKeyDown }) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onSelect(report.scan_id)}
      onKeyDown={(event) => onKeyDown(event, report.scan_id)}
      className={
        ROW_CLASS +
        " font-mono text-body cursor-pointer hover:bg-surface-secondary focus:bg-surface-secondary focus:outline-none"
      }
    >
      {/* "You are here" cursor cue -- the top row only, never every row. */}
      <span aria-hidden="true" className="w-4 shrink-0 text-text-faint">
        {isFirst ? "❯" : ""}
      </span>
      <SeverityBadge severity={report.max_severity} />
      {/* Fixed max width + truncate + title: a long free-form
          target_summary (D-09 calls this out as unstructured text) cannot
          widen the row or wrap it. */}
      <span className="flex-1 min-w-0 truncate" title={report.target_summary}>
        {report.target_summary}
      </span>
      <span
        className="w-48 shrink-0 truncate text-text-faint"
        title={report.module_ids.join(", ")}
      >
        {report.module_ids.join(", ")}
      </span>
      <span className="w-24 shrink-0 text-right text-text-faint">{report.completed_at}</span>
    </div>
  );
}

// One component, five mutually exclusive render branches driven by `state`
// and `reports`/`hasAnyFilterActive` -- each branch is its own JSX return
// expression with its own verbatim UI-SPEC copy, never a single generic
// empty-state parameterized by a message string (that's one edit away from
// collapsing the two distinct empty states the UI-SPEC forbids merging).
export default function ReportList({ reports, state, hasAnyFilterActive, onRetry, onSelect }) {
  // loading: same row height/rhythm as a real row so swapping skeletons for
  // real rows causes no layout shift.
  if (state === "loading") {
    return (
      <div className="flex flex-col gap-xs">
        {Array.from({ length: SKELETON_ROW_COUNT }, (_, index) => (
          <SkeletonRow key={index} />
        ))}
      </div>
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
          className="text-body px-md py-sm rounded border border-border hover:border-accent"
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
  function handleRowKeyDown(event, scanId) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(scanId);
    }
  }

  return (
    <div className="flex flex-col gap-xs">
      {reports.map((report, index) => (
        <ReportRow
          key={report.scan_id}
          report={report}
          isFirst={index === 0}
          onSelect={onSelect}
          onKeyDown={handleRowKeyDown}
        />
      ))}
    </div>
  );
}
