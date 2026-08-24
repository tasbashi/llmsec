// D-12-01: an ASCII block-character histogram, not a Recharts bar chart.
// Fixed band order matches `llmsec.reporting.markdown_reporter.
// _count_by_severity()`'s own order (the `Severity` enum's declaration
// order -- none, low, medium, high, critical) so
// `tests/dashboard/test_dashboard_severity_parity.py` pins this exact
// sequence against the Markdown reporter's own aggregation (D-12-02). Every
// band is always present in the data series, even at count 0 -- a chart
// that silently drops an empty band makes "no critical findings" and
// "critical was not measured" look identical (T-10-22). The five hex values
// stay here beside their band names since this file has always carried raw
// hex values for exactly this reason -- the colour-to-band pin
// (`test_severity_chart_uses_the_full_ui_spec_ramp_with_matching_band_names`)
// requires it (D-12-04).
const SEVERITY_BANDS = [
  { key: "none", label: "None", color: "#6b7280" },
  { key: "low", label: "Low", color: "#22c55e" },
  { key: "medium", label: "Medium", color: "#eab308" },
  { key: "high", label: "High", color: "#f97316" },
  { key: "critical", label: "Critical", color: "#ef4444" },
];

// The bar itself is a fixed-width run of block characters -- a terminal
// histogram, so the string IS the bar, never a div width
// (report-and-findings-display.md).
const BAR_WIDTH = 24;

export default function SeverityChart({ findings }) {
  // Shares the detail view's single fetch (App.jsx) -- no loading/error
  // branch of its own. Empty findings gets the same message the findings
  // list shows, rendered inside the histogram card's body (ReportDetail.jsx
  // owns the card chrome) rather than in place of a chart.
  if (findings.length === 0) {
    return (
      <div className="flex items-center justify-center py-lg text-body">
        No findings — every test case in this scan was blocked.
      </div>
    );
  }

  const counts = Object.fromEntries(SEVERITY_BANDS.map((band) => [band.key, 0]));
  for (const finding of findings) {
    if (finding.severity in counts) {
      counts[finding.severity] += 1;
    }
  }

  // Filled length scales to the worst-populated band across all five
  // (never only the nonzero ones), so the worst band always reads full
  // width. Divisor guarded against a zero maximum -- an all-zero report
  // (a valid state: every band can be legitimately empty even when
  // `findings` isn't, if severities outside SEVERITY_BANDS ever occurred)
  // renders five all-░ bars, not a division-by-zero blank chart.
  const maxCount = Math.max(...SEVERITY_BANDS.map((band) => counts[band.key]));

  return (
    <div className="flex flex-col gap-sm font-mono text-label">
      {SEVERITY_BANDS.map((band) => {
        const count = counts[band.key];
        const filled = maxCount === 0 ? 0 : Math.round((count / maxCount) * BAR_WIDTH);
        const bar = "█".repeat(filled) + "░".repeat(BAR_WIDTH - filled);
        // Colour the whole row (label, bar, count) from the band's own hex
        // via one inline style -- colour is never the sole signal since the
        // band name and the numeric count both always render as text on
        // every row regardless of colour perception.
        return (
          <div key={band.key} className="flex items-center gap-sm" style={{ color: band.color }}>
            <span className="w-20 shrink-0 font-semibold uppercase">{band.label}</span>
            <span>{bar}</span>
            <span>{count}</span>
          </div>
        );
      })}
    </div>
  );
}
