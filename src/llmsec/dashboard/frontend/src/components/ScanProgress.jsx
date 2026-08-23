// Purely presentational -- declares no effect, issues no network call, and
// schedules no timer. App.jsx owns every poll and passes the parsed status
// body (or the lost-scan sentinel) down as `status`. The stepper's fill is
// a direct function of the phase string alone: no percentage, no
// elapsed/remaining estimate, no advance between polls (DASH-06's
// never-fabricate-progress contract).

// Same inline-glyph discipline as RawCases.jsx/AuthorizationPanel.jsx --
// no icon-library dependency.
function CheckmarkIcon() {
  return (
    <svg
      className="w-4 h-4 stroke-current text-surface-primary"
      viewBox="0 0 24 24"
      fill="none"
      strokeWidth="3"
      aria-hidden="true"
    >
      <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Spinner glyph -- an inline SVG circle with `animate-spin`, no new
// dependency (UI-SPEC Icon library row).
function SpinnerIcon() {
  return (
    <svg
      className="w-4 h-4 animate-spin text-surface-primary"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeDasharray="40 60"
      />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg className="w-4 h-4 stroke-current shrink-0" viewBox="0 0 24 24" fill="none" strokeWidth="2" aria-hidden="true">
      <circle cx="12" cy="12" r="9" />
      <line x1="12" y1="8" x2="12" y2="13" strokeLinecap="round" />
      <line x1="12" y1="16" x2="12.01" y2="16" strokeLinecap="round" />
    </svg>
  );
}

// Exact step labels + per-phase subtext, in server-phase order (never
// derived/computed -- a phase the server has not reported yet must not be
// shown as begun).
const STEPS = [
  { phase: "queued", label: "Queued", subtext: "Waiting to start…" },
  { phase: "running", label: "Running", subtext: "Sending test payloads to the target…" },
  { phase: "scoring", label: "Scoring", subtext: "Scoring results and preparing the report…" },
  { phase: "complete", label: "Complete", subtext: "Scan complete." },
];

const CIRCLE_BASE = "w-10 h-10 rounded-full flex items-center justify-center shrink-0";

// Shared by the complete-state and in-progress-state `if` blocks below --
// each is still its own structurally separate conditional/return (never
// one shared parameterized block deciding which branch to take); only the
// stepper's own markup is factored out to avoid literal duplication.
function Stepper({ currentIndex, allDone }) {
  return (
    <div className="flex items-center w-full max-w-2xl">
      {STEPS.map((step, index) => {
        const state = allDone || index < currentIndex ? "done" : index === currentIndex ? "active" : "future";
        return (
          <div key={step.phase} className="flex items-center flex-1 last:flex-none">
            <div className="flex flex-col items-center gap-xs">
              <div
                className={
                  CIRCLE_BASE + " " + (state === "future" ? "bg-surface-secondary" : "bg-accent")
                }
              >
                {state === "done" && <CheckmarkIcon />}
                {state === "active" && <SpinnerIcon />}
              </div>
              <span className={"text-label" + (state === "future" ? " opacity-50" : "")}>
                {step.label}
              </span>
            </div>
            {index < STEPS.length - 1 && (
              <div
                className={
                  "flex-1 h-px mx-sm " + (state === "done" ? "bg-accent" : "bg-surface-secondary")
                }
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function ScanProgress({ scanId, status, onViewReport, onBackToReports, onTryAgain }) {
  // Lost-scan state (D-04/A-02): a status 404 for a scan_id this dashboard
  // process no longer knows about. Its own branch -- never a generic 404,
  // never a failure, never continued polling (App.jsx already cleared the
  // interval before setting this sentinel).
  if (status?.lost) {
    return (
      <div className="min-h-screen bg-surface-primary px-xl py-3xl flex flex-col items-center gap-lg">
        <div className="max-w-2xl w-full flex flex-col items-center gap-md text-center">
          <h1 className="text-heading">Lost track of this scan.</h1>
          <p className="text-body">
            The dashboard may have restarted while this scan was running. Check the report list — if the scan finished, it will appear there.
          </p>
          <button
            type="button"
            onClick={onBackToReports}
            className="text-body px-md py-sm rounded border border-surface-secondary hover:border-accent"
          >
            Back to Reports
          </button>
        </div>
      </div>
    );
  }

  // Failed state: replaces the stepper entirely (never a partially-coloured
  // stepper, which would imply a phase attribution the status endpoint
  // never made).
  if (status?.phase === "failed") {
    return (
      <div className="min-h-screen bg-surface-primary px-xl py-3xl flex flex-col items-center gap-lg">
        <div className="max-w-2xl w-full border-l-4 border-destructive bg-surface-secondary rounded p-lg flex flex-col gap-md">
          <div className="flex items-center gap-sm text-destructive">
            <AlertIcon />
            <h1 className="text-heading">Scan Failed</h1>
          </div>
          {/* Never a stack trace -- the one-line exception summary the
              store recorded. Never clamped/truncated (same
              never-truncate-a-diagnostic rule as ReportDetail.jsx's
              `scan-limitations`/`deep-summary` containers) -- this
              element carries no truncate/clamp/overflow-hiding/max-h
              utility. */}
          <p data-testid="scan-failure-detail" className="text-body">
            {"Scan failed: "}
            {status.error}
            {". Run `llmsec scan` from the CLI with the same configuration for full error detail."}
          </p>
          <div className="flex items-center gap-md">
            <button
              type="button"
              onClick={onBackToReports}
              className="text-body px-md py-sm rounded border border-surface-secondary hover:border-accent"
            >
              Back to Reports
            </button>
            <button
              type="button"
              onClick={onTryAgain}
              className="text-body px-md py-sm rounded border border-surface-secondary hover:border-accent"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  const currentIndex = STEPS.findIndex((step) => step.phase === status?.phase);
  const activeStep = STEPS[currentIndex] ?? STEPS[0];

  // Complete state: its own structurally separate conditional/return
  // (never merely a flag branching inside the in-progress render) -- all
  // four steps filled (checkmark, never a spinner) plus the success panel
  // handing off to Phase 10's existing report-detail view.
  if (status?.phase === "complete") {
    return (
      <div className="min-h-screen bg-surface-primary px-xl py-3xl flex flex-col items-center gap-xl">
        <h1 className="text-heading">Scan in Progress</h1>
        <p className="text-label text-body">
          Scan ID: <span className="font-mono">{scanId}</span>
        </p>
        <Stepper currentIndex={currentIndex} allDone />
        <p className="text-body">{activeStep.subtext}</p>
        <button
          type="button"
          onClick={onViewReport}
          className="bg-accent text-body px-md py-sm rounded"
        >
          View Report
        </button>
      </div>
    );
  }

  // Happy path: queued / running / scoring.
  return (
    <div className="min-h-screen bg-surface-primary px-xl py-3xl flex flex-col items-center gap-xl">
      <h1 className="text-heading">Scan in Progress</h1>
      <p className="text-label text-body">
        Scan ID: <span className="font-mono">{scanId}</span>
      </p>
      <Stepper currentIndex={currentIndex} allDone={false} />
      <p className="text-body">{activeStep.subtext}</p>
    </div>
  );
}
