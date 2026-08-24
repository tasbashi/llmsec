// The shared severity tag: a fixed-width uppercase mono label with a
// coloured dot, consumed by ReportList.jsx (this plan) and ReportDetail.jsx
// (12-03, whose only prop is `severity` -- unchanged by this rewrite, so a
// later plan never has to edit this file). Colour is never the sole signal:
// the band name always renders as text next to the dot, and the dot itself
// is `aria-hidden`.
//
// Only the `critical` band's dot carries a glow (foundations.md: "only
// critical gets a glow -- applying it to every severity dilutes the one
// 'this matters' signal"). The five hex values behind these tokens live in
// tailwind.config.js, never here -- this component only ever consumes named
// utility classes.

// D-12-*/foundations.md's critical-only-glow rule, declared as its own
// isolated constant well clear of the class maps below (defined further
// down this file, after the component itself -- a `const` module binding is
// fully evaluated before the component is ever rendered, so declaration
// order here does not affect behaviour) so it reads in source exactly as it
// reads on screen: the one deliberate exception, never one interchangeable
// line inside a shared five-entry table.
const CRITICAL_DOT_GLOW_CLASS =
  "inline-block w-2 h-2 rounded-full bg-severity-critical shadow-glow-critical";

export default function SeverityBadge({ severity }) {
  const label = SEVERITY_LABEL[severity] ?? severity;
  return (
    <span className={SEVERITY_TEXT_CLASS[severity] ?? SEVERITY_TEXT_CLASS.none}>
      <span
        className={SEVERITY_DOT_CLASS[severity] ?? SEVERITY_DOT_CLASS.none}
        aria-hidden="true"
      />
      {label}
    </span>
  );
}

export const SEVERITY_LABEL = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
  none: "None",
};

// Tailwind's class scanner only sees literal, whole class-name strings in
// source -- a template literal that interpolates the band name into a
// class name would never be extracted, so every combination below is
// spelled out as its own complete literal string, one entry per band
// (mirrors this file's own pre-existing discipline).
const SEVERITY_DOT_CLASS = {
  critical: CRITICAL_DOT_GLOW_CLASS,
  high: "inline-block w-2 h-2 rounded-full bg-severity-high",
  medium: "inline-block w-2 h-2 rounded-full bg-severity-medium",
  low: "inline-block w-2 h-2 rounded-full bg-severity-low",
  none: "inline-block w-2 h-2 rounded-full bg-severity-none",
};

const SEVERITY_TEXT_CLASS = {
  critical:
    "inline-flex items-center gap-xs w-20 font-mono text-label uppercase text-severity-critical",
  high: "inline-flex items-center gap-xs w-20 font-mono text-label uppercase text-severity-high",
  medium:
    "inline-flex items-center gap-xs w-20 font-mono text-label uppercase text-severity-medium",
  low: "inline-flex items-center gap-xs w-20 font-mono text-label uppercase text-severity-low",
  none: "inline-flex items-center gap-xs w-20 font-mono text-label uppercase text-severity-none",
};
