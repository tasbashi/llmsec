// A controlled, presentational component: it owns no fetching and no
// filtering logic of its own. `options` (FilterOptions | null), `state`
// ("idle" | "loading" | "error" | "ready"), `value` ({ target, modules,
// severity }) and `onChange` (replaces the whole filters object) are all
// owned by App.jsx.

// D-11/RESEARCH Pitfall #3: the "none" band's label is the human-readable
// clean/no-findings wording -- bare "none" alone reads like "no filter".
const SEVERITY_LABEL = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
  none: "None — clean / no findings",
};

// Tailwind's class scanner only sees literal, whole class-name strings in
// source -- an interpolated template literal would never be extracted, so
// every combination is spelled out here (mirrors ReportList.jsx's
// SEVERITY_*_CLASS discipline). The accent token is never used for any
// severity element -- severity has its own five-band ramp (UI-SPEC Color
// section).
const SEVERITY_OPTION_CLASS = {
  critical: "text-severity-critical",
  high: "text-severity-high",
  medium: "text-severity-medium",
  low: "text-severity-low",
  none: "text-severity-none",
};

// D-12-07: builds the "active filter summary" that replaces the grep hint
// once at least one filter is set -- pure display text derived from the
// already-owned `value`, never a second source of truth for what is
// actually being requested.
function activeFilterSummary(value) {
  const parts = [];
  if (value.target) parts.push(value.target);
  if (value.modules.length > 0) parts.push(value.modules.join(","));
  if (value.severity) parts.push(value.severity);
  return parts.join(" · ");
}

export default function FilterBar({ options, state, value, onChange }) {
  const disabled = state === "loading" || state === "error";
  const loadingPlaceholder = state === "loading";
  const hasAnyFilterActive =
    Boolean(value.target) || Boolean(value.severity) || value.modules.length > 0;

  function setTarget(target) {
    onChange({ ...value, target });
  }

  function setSeverity(severity) {
    onChange({ ...value, severity });
  }

  function toggleModule(moduleId) {
    const modules = value.modules.includes(moduleId)
      ? value.modules.filter((id) => id !== moduleId)
      : [...value.modules, moduleId];
    onChange({ ...value, modules });
  }

  return (
    <div className="flex flex-wrap items-center gap-sm font-mono text-label bg-surface-primary border border-border rounded-sm px-sm py-xs">
      {/* D-12-07: chrome only, a non-interactive hint / active-filter
          summary -- never a free-text input. A substring search over the
          free-form target_summary is forbidden by D-09, and an input that
          cannot filter would be a control that lies about what it does.
          The three real filter controls below are the unchanged
          selects/chips this bar already had; only their surrounding chrome
          moved into this grep-styled cluster. */}
      <span className="text-text-faint whitespace-nowrap">
        {hasAnyFilterActive ? activeFilterSummary(value) : "grep target, module, severity…"}
      </span>

      {/* Target -- single-select over the server-computed distinct values
          (D-09: never free-text, a substring search against a free-form
          field is a typo dead-end). Truncated with the full value in a
          title attribute so a long target_summary cannot widen the bar. */}
      <select
        aria-label="Filter by target"
        disabled={disabled}
        value={value.target}
        title={value.target || undefined}
        onChange={(e) => setTarget(e.target.value)}
        className="max-w-[180px] truncate rounded-sm bg-surface-sunken border border-border px-sm py-xs text-label disabled:opacity-50"
      >
        {loadingPlaceholder ? (
          <option value="">Loading filters…</option>
        ) : (
          <>
            <option value="">All targets</option>
            {(options?.targets ?? []).map((target) => (
              <option key={target} value={target} title={target}>
                {target}
              </option>
            ))}
          </>
        )}
      </select>

      {/* Modules -- a row of toggle chips (clearer than a native multi-select
          for a small, known set). Superset semantics (10-02 assumption
          A-01): selecting several narrows the result set. */}
      <div
        className="flex flex-wrap items-center gap-xs"
        role="group"
        aria-label="Filter by module"
        title="Selecting more than one module shows only scans that ran all of them."
      >
        {loadingPlaceholder ? (
          <span className="text-text-faint">Loading filters…</span>
        ) : (
          (options?.modules ?? []).map((moduleId) => {
            const active = value.modules.includes(moduleId);
            return (
              <button
                key={moduleId}
                type="button"
                disabled={disabled}
                aria-pressed={active}
                onClick={() => toggleModule(moduleId)}
                className={
                  (active
                    ? "bg-accent-dim text-accent border border-accent"
                    : "bg-surface-sunken border border-border text-text-muted") +
                  " px-sm py-xs rounded-sm disabled:opacity-50"
                }
              >
                {moduleId}
              </button>
            );
          })
        )}
      </div>

      {/* Severity -- single-select. The "none" band is always present when
          any stored report has zero findings (D-11/Pitfall #3), never
          collapsed out of the option list. */}
      <select
        aria-label="Filter by severity"
        disabled={disabled}
        value={value.severity}
        onChange={(e) => setSeverity(e.target.value)}
        className="rounded-sm bg-surface-sunken border border-border px-sm py-xs text-label disabled:opacity-50"
      >
        <option value="">Any severity</option>
        {(options?.severities ?? []).map((severity) => (
          <option
            key={severity}
            value={severity}
            className={SEVERITY_OPTION_CLASS[severity] ?? SEVERITY_OPTION_CLASS.none}
          >
            {SEVERITY_LABEL[severity] ?? severity}
          </option>
        ))}
      </select>

      {state === "error" && <span className="text-destructive">Filters unavailable</span>}
    </div>
  );
}
