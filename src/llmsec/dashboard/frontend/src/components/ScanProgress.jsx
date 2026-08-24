// Purely presentational -- declares no effect, issues no network call, and
// schedules no timer. App.jsx owns every poll and passes the parsed status
// body (or the lost-scan sentinel) down as `status`. The phase tag and the
// log panel are both driven by a single `currentIndex` derivation (one
// `STEPS.findIndex(` call, D-12-03): no percentage, no elapsed/remaining
// estimate, no advance between polls, no line for a phase the server has
// not reported yet (DASH-06's never-fabricate-progress contract).

// Same inline-glyph discipline as RawCases.jsx/AuthorizationPanel.jsx --
// no icon-library dependency.
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

// Mono uppercase pill carrying the current step's label -- replaces the
// deleted stepper widget. `running` and `complete` read as the accent
// "this is the active/finished signal" tone; `queued`/`scoring` stay muted
// since they are waiting states, not the state carrying the live spinner.
function PhaseTag({ phase, label }) {
  const toneClass =
    phase === "queued" || phase === "scoring"
      ? "text-text-muted bg-surface-secondary"
      : "text-accent bg-accent-dim";
  return (
    <span
      className={
        "font-mono text-label uppercase tracking-wide px-sm py-xs rounded-sm shrink-0 " + toneClass
      }
    >
      {label}
    </span>
  );
}

// The observed-phase log panel (D-12-03): its content is a pure function of
// `currentIndex` -- one line for every STEPS entry at an index less than or
// equal to currentIndex, each line built from that entry's OWN `phase` and
// `subtext` fields. This is exactly the fill rule the deleted stepper
// already used (`index <= currentIndex` was "filled"), re-expressed as
// lines instead of circles -- it never synthesises per-request detail and
// never renders a line for a phase the index has not reached.
function LogPanel({ currentIndex, showCursor }) {
  return (
    <div className="w-full h-[260px] overflow-y-auto bg-surface-sunken border border-border rounded-md px-md py-sm font-mono text-label leading-relaxed text-text-muted">
      {STEPS.map((step, index) =>
        index <= currentIndex ? (
          <div key={step.phase}>
            [{step.phase}] {step.subtext}
          </div>
        ) : null
      )}
      {showCursor && (
        <span
          className="inline-block w-[7px] h-[14px] bg-accent animate-blink align-text-bottom"
          aria-hidden="true"
        />
      )}
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
            className="text-body px-md py-sm rounded border border-border hover:border-accent"
          >
            Back to Reports
          </button>
        </div>
      </div>
    );
  }

  // Failed state: replaces the phase tag + log panel entirely (never a
  // partially-filled log, which would imply a phase-reached claim the
  // status store never made -- `set_failed()` discards the last-known
  // phase, so no history is ever synthesised here).
  if (status?.phase === "failed") {
    return (
      <div className="min-h-screen bg-surface-primary px-xl py-3xl flex flex-col items-center gap-lg">
        <div className="max-w-2xl w-full border-l-4 border-destructive bg-surface-secondary rounded-lg p-lg flex flex-col gap-md">
          <div className="flex items-center gap-sm text-destructive">
            <AlertIcon />
            <h1 className="text-heading">Scan Failed</h1>
          </div>
          {/* Never a stack trace -- the one-line exception summary the
              store recorded, rendered as a mono destructive log line
              (never a generic message). Never clamped/truncated (same
              never-truncate-a-diagnostic rule as ReportDetail.jsx's
              `scan-limitations`/`deep-summary` containers) -- this
              element carries no truncate/clamp/overflow-hiding/max-h
              utility. */}
          <p data-testid="scan-failure-detail" className="font-mono text-body text-destructive">
            {"Scan failed: "}
            {status.error}
            {". Run `llmsec scan` from the CLI with the same configuration for full error detail."}
          </p>
          <div className="flex items-center gap-md">
            <button
              type="button"
              onClick={onBackToReports}
              className="text-body px-md py-sm rounded border border-border hover:border-accent"
            >
              Back to Reports
            </button>
            <button
              type="button"
              onClick={onTryAgain}
              className="text-body px-md py-sm rounded border border-border hover:border-accent"
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
  // (never merely a flag branching inside the in-progress render) -- the
  // full log shown at once with a blinking cursor, no spinner, plus the
  // success panel handing off to Phase 10's existing report-detail view.
  if (status?.phase === "complete") {
    return (
      <div className="min-h-screen bg-surface-primary px-xl py-3xl flex flex-col items-center gap-xl">
        <h1 className="text-heading">Scan in Progress</h1>
        <p className="text-label text-body">
          Scan ID: <span className="font-mono">{scanId}</span>
        </p>
        <div className="w-full max-w-2xl flex flex-col gap-md">
          <div className="flex items-center gap-sm">
            <PhaseTag phase={status?.phase} label={activeStep.label} />
            <span className="text-body text-text-muted">{activeStep.subtext}</span>
          </div>
          <LogPanel currentIndex={currentIndex} showCursor />
        </div>
        <button
          type="button"
          onClick={onViewReport}
          className="bg-accent text-body font-mono rounded px-md py-sm hover:bg-accent-hover hover:shadow-glow-accent"
        >
          View Report
        </button>
      </div>
    );
  }

  // Happy path: queued / running / scoring. The `◐` glyph is text in a
  // span with Tailwind's built-in spin animation -- CSS-rotated, no timer,
  // no new dependency.
  return (
    <div className="min-h-screen bg-surface-primary px-xl py-3xl flex flex-col items-center gap-xl">
      <h1 className="text-heading">Scan in Progress</h1>
      <p className="text-label text-body">
        Scan ID: <span className="font-mono">{scanId}</span>
      </p>
      <div className="w-full max-w-2xl flex flex-col gap-md">
        <div className="flex items-center gap-sm">
          <PhaseTag phase={status?.phase} label={activeStep.label} />
          <span className="text-body text-text-muted">{activeStep.subtext}</span>
          <span className="ml-auto text-accent animate-spin" aria-hidden="true">
            ◐
          </span>
        </div>
        <LogPanel currentIndex={currentIndex} showCursor={false} />
      </div>
    </div>
  );
}
