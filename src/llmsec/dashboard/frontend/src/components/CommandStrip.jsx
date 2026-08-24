import FilterBar from "./FilterBar.jsx";

// Icon-only ghost button, no icon-library dependency (UI-SPEC Icon library
// row) -- inline SVG glyph styled with the project's existing Tailwind
// utility discipline (mirrors RawCases.jsx's ChevronIcon).
function LogoutIcon() {
  return (
    <svg
      className="w-4 h-4 stroke-current"
      viewBox="0 0 24 24"
      fill="none"
      strokeWidth="2"
      aria-hidden="true"
    >
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M16 17l5-5-5-5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M21 12H9" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// The list view's top bar -- presentational only, owns no state and no
// fetching of its own. `reports` is the same already-fetched array App.jsx
// passes to ReportList; the summary counts below are derived from it, never
// a second request. `filterOptions`/`filtersState`/`filters`/
// `onFiltersChange` pass straight through to FilterBar, whose own
// select/chip controls and onChange contract are unchanged by this plan --
// only their surrounding chrome moved into this strip.
export default function CommandStrip({
  reports,
  filterOptions,
  filtersState,
  filters,
  onFiltersChange,
  onStartNewScan,
  onLogout,
}) {
  const reportCountLabel = reports.length === 1 ? "1 report" : `${reports.length} reports`;
  // D-12-08: the critical count always renders, including zero -- "no
  // critical reports in this filtered set" must never be visually
  // indistinguishable from "the count was not computed" (same
  // never-drop-an-empty-signal reasoning as D-12-02).
  const criticalCount = reports.filter((report) => report.max_severity === "critical").length;

  return (
    <div className="flex items-center gap-md bg-surface-sunken border-b border-border px-lg py-sm">
      <span className="font-mono text-body font-semibold whitespace-nowrap">
        <span className="text-accent">llmsec</span>
        <span className="text-text-faint">://</span>
        <span className="text-text-muted">reports</span>
      </span>

      <FilterBar
        options={filterOptions}
        state={filtersState}
        value={filters}
        onChange={onFiltersChange}
      />

      <div className="flex-1" />

      <span className="font-mono text-label text-text-faint whitespace-nowrap">
        {reportCountLabel} · {criticalCount} critical
      </span>

      {/* shadow-glow-accent on hover only -- that glow and the critical
          severity dot are its only two sanctioned uses (foundations.md). */}
      <button
        type="button"
        onClick={onStartNewScan}
        className="bg-accent text-body font-mono rounded-sm px-md py-xs whitespace-nowrap hover:bg-accent-hover hover:shadow-glow-accent"
      >
        + new_scan
      </button>

      <button
        type="button"
        onClick={onLogout}
        aria-label="Log out"
        className="w-8 h-8 shrink-0 flex items-center justify-center rounded-sm border border-border text-text-muted hover:text-text hover:border-border-hover"
      >
        <LogoutIcon />
      </button>
    </div>
  );
}
