// Extracted from ReportList.jsx's 10-01 inline helper so a future component
// (e.g. the detail view's findings table, 10-05) can render the identical
// pill without restating the severity->label/class maps.

export const SEVERITY_LABEL = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
  none: "None",
};

// Tailwind's class scanner only sees literal, whole class-name strings in
// source -- an interpolated template literal like `bg-severity-${severity}`
// would never be extracted, so every combination is spelled out here.
const SEVERITY_DOT_CLASS = {
  critical: "inline-block w-2 h-2 rounded-full bg-severity-critical",
  high: "inline-block w-2 h-2 rounded-full bg-severity-high",
  medium: "inline-block w-2 h-2 rounded-full bg-severity-medium",
  low: "inline-block w-2 h-2 rounded-full bg-severity-low",
  none: "inline-block w-2 h-2 rounded-full bg-severity-none",
};

const SEVERITY_TEXT_CLASS = {
  critical: "text-label px-sm py-xs rounded inline-flex items-center gap-xs text-severity-critical",
  high: "text-label px-sm py-xs rounded inline-flex items-center gap-xs text-severity-high",
  medium: "text-label px-sm py-xs rounded inline-flex items-center gap-xs text-severity-medium",
  low: "text-label px-sm py-xs rounded inline-flex items-center gap-xs text-severity-low",
  none: "text-label px-sm py-xs rounded inline-flex items-center gap-xs text-severity-none",
};

export default function SeverityBadge({ severity }) {
  const label = SEVERITY_LABEL[severity] ?? severity;
  return (
    <span className={SEVERITY_TEXT_CLASS[severity] ?? SEVERITY_TEXT_CLASS.none}>
      <span className={SEVERITY_DOT_CLASS[severity] ?? SEVERITY_DOT_CLASS.none} aria-hidden="true" />
      {label}
    </span>
  );
}
