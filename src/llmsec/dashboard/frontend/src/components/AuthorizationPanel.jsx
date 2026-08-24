// Purely presentational controlled child (D-06/D-07, T-11-02): owns no
// state of its own. `checked` comes exclusively from the parent's prop --
// no `defaultChecked`, no `useState`, no read from `localStorage` or any
// other storage API -- so a remount (e.g. ScanForm re-mounting after "Try
// Again") always yields an unchecked box. `targetSummary` is a live
// derivation the parent recomputes from current form values on every
// render, never a snapshot taken at check time.

// Inline warning-triangle glyph -- no icon-library dependency (UI-SPEC Icon
// library row), styled with the project's `w-4 h-4 stroke-current`
// convention (mirrors RawCases.jsx's ChevronIcon).
function WarningTriangleIcon() {
  return (
    <svg
      className="w-4 h-4 stroke-current shrink-0"
      viewBox="0 0 24 24"
      fill="none"
      strokeWidth="2"
      aria-hidden="true"
    >
      <path
        d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <line x1="12" y1="9" x2="12" y2="13" strokeLinecap="round" />
      <line x1="12" y1="17" x2="12.01" y2="17" strokeLinecap="round" />
    </svg>
  );
}

// D-07: this exact wording is the web adaptation of `auth_gate.DISCLAIMER`,
// pinned verbatim by 11-UI-SPEC.md's Copywriting Contract (the 11-03
// contract test pins it byte-for-byte) -- never paraphrased. The UI-SPEC's
// own markdown-table ``**...**`` is that document's bold notation, not a
// literal string this component renders (mirrors this plan's own
// acceptance criteria for the env-var helper text, which strips the same
// markdown markers) -- rendered here as a real `<strong>` instead.
const DISCLAIMER_PREFIX =
  "llmsec is about to send live requests to the target below. This tool is intended for ";
const DISCLAIMER_EMPHASIS = "authorized security testing only";
const DISCLAIMER_SUFFIX =
  ". Scanning systems without explicit written permission from the owner may violate computer fraud laws (e.g. CFAA) and provider Terms of Service. The authors assume no liability for unauthorized use.";

export default function AuthorizationPanel({ targetSummary, checked, onChange }) {
  return (
    <div className="border-l-4 border-severity-medium bg-surface-secondary rounded-lg p-lg flex flex-col gap-md">
      <div className="flex items-center gap-sm text-severity-medium">
        <WarningTriangleIcon />
        <h2 className="text-heading">Authorization Required</h2>
      </div>

      <p className="text-body">
        {DISCLAIMER_PREFIX}
        <strong>{DISCLAIMER_EMPHASIS}</strong>
        {DISCLAIMER_SUFFIX}
      </p>

      <div>
        <p className="text-label uppercase tracking-wide mb-xs">You are about to scan:</p>
        <div className="border border-border rounded-md p-md font-mono text-body break-words bg-surface-primary">
          {targetSummary}
        </div>
      </div>

      <label className="flex items-start gap-sm text-body">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="mt-xs"
        />
        I confirm I have explicit authorization to test this target.
      </label>
    </div>
  );
}
