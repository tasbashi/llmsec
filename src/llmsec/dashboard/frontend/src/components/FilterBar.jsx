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

export default function FilterBar({ options, state, value, onChange }) {
  const disabled = state === "loading" || state === "error";
  const loadingPlaceholder = state === "loading";

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
    <div className="flex flex-wrap items-center gap-sm bg-surface-secondary p-lg rounded text-body">
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
        className="max-w-[220px] truncate rounded bg-surface-primary border border-surface-secondary px-sm py-xs text-body disabled:opacity-50"
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
      <div className="flex flex-wrap items-center gap-sm" role="group" aria-label="Filter by module">
        {loadingPlaceholder ? (
          <span className="text-label">Loading filters…</span>
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
                    ? "bg-accent text-body"
                    : "bg-surface-secondary border border-surface-secondary text-body") +
                  " text-label px-sm py-xs rounded disabled:opacity-50"
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
        className="rounded bg-surface-primary border border-surface-secondary px-sm py-xs text-body disabled:opacity-50"
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

      <p className="text-label basis-full text-body">
        Selecting more than one module shows only scans that ran all of them.
      </p>

      {state === "error" && (
        <p className="text-label basis-full text-destructive">Filters unavailable</p>
      )}
    </div>
  );
}
