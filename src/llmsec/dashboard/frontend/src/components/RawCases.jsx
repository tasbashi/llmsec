import { useState } from "react";

// Inline chevron -- no icon-library dependency for a single glyph
// (UI-SPEC Icon library row). Styled with the project's Tailwind utility
// discipline, not a hand-written CSS rule.
function ChevronIcon({ expanded }) {
  return (
    <svg
      className={
        "w-4 h-4 stroke-current transition-transform" + (expanded ? " rotate-180" : "")
      }
      viewBox="0 0 24 24"
      fill="none"
      strokeWidth="2"
      aria-hidden="true"
    >
      <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Collapsed by default (D-14) -- driven purely by local `useState`. There is
// no fetch of its own: `case_log` already arrived on the same
// `GET /api/reports/{scan_id}` payload that populates the findings list
// and severity histogram, which is exactly why the UI-SPEC dismissed a
// separate loading/error state for this component.
export default function RawCases({ caseLog }) {
  const [expanded, setExpanded] = useState(false);
  const count = caseLog.length;

  return (
    <section className="flex flex-col gap-md">
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((prev) => !prev)}
        className="flex items-center gap-sm font-mono text-body text-accent self-start"
      >
        <ChevronIcon expanded={expanded} />
        {expanded ? "Hide raw cases" : `Show raw cases (${count})`}
      </button>
      {expanded && (
        <div className="flex flex-col gap-sm">
          {count === 0 ? (
            <p className="text-body">No case log recorded</p>
          ) : (
            caseLog.map((entry) => (
              <div
                key={entry.case_id}
                className="bg-surface-secondary rounded-md p-md flex flex-col gap-xs"
              >
                <p className="font-mono text-label">{entry.case_id}</p>
                <p className="font-mono text-label text-text-faint">Verdict: {entry.verdict}</p>
                <p className="font-mono text-label text-text-faint">
                  Confidence: {entry.confidence}
                </p>
                <p className="font-mono text-label text-text-faint">
                  Detection layer: {entry.detection_layer}
                </p>
                {entry.transport_mode && (
                  <p className="font-mono text-label text-text-faint">
                    Transport: {entry.transport_mode}
                  </p>
                )}
                {/* Target-derived text (threat T-10-09) -- a plain JSX
                    text child, never a raw-HTML injection attribute. */}
                <p className="text-body whitespace-pre-wrap break-words">
                  Evidence: {entry.evidence}
                </p>
              </div>
            ))
          )}
        </div>
      )}
    </section>
  );
}
