import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// D-16: a chart, not numeric badges. Fixed band order matches
// `llmsec.reporting.markdown_reporter._count_by_severity()`'s own order
// (the `Severity` enum's declaration order -- none, low, medium, high,
// critical) so `tests/dashboard/test_dashboard_severity_parity.py` pins
// this exact sequence against the Markdown reporter's own aggregation.
// Every band is always present in the data series, even at count 0 --
// a chart that silently drops an empty band makes "no critical findings"
// and "critical was not measured" look identical (T-10-22).
const SEVERITY_BANDS = [
  { key: "none", label: "None", color: "#6b7280" },
  { key: "low", label: "Low", color: "#22c55e" },
  { key: "medium", label: "Medium", color: "#eab308" },
  { key: "high", label: "High", color: "#f97316" },
  { key: "critical", label: "Critical", color: "#ef4444" },
];

export default function SeverityChart({ findings }) {
  // Shares the detail view's single fetch (App.jsx) -- no loading/error
  // branch of its own. Empty findings gets the same message the findings
  // table shows, in place of a degenerate all-zero chart.
  if (findings.length === 0) {
    return (
      <div className="h-40 flex items-center justify-center text-body">
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
  const data = SEVERITY_BANDS.map((band) => ({
    key: band.key,
    // The axis tick carries the band name directly, so colour is never the
    // sole signal (UI-SPEC accessibility note) -- pairs each bar's colour
    // with its text label without a separate legend component.
    name: band.label,
    count: counts[band.key],
  }));

  // Bounded height, no full-width dominance -- secondary to the findings
  // table per the UI-SPEC's visual hierarchy.
  return (
    <div className="h-48 max-w-2xl">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#171b22" />
          <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} axisLine={{ stroke: "#171b22" }} />
          <YAxis
            allowDecimals={false}
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            axisLine={{ stroke: "#171b22" }}
          />
          <Tooltip
            contentStyle={{ background: "#171b22", border: "1px solid #0f1115", color: "#e5e7eb" }}
          />
          <Bar dataKey="count">
            {data.map((entry) => (
              <Cell key={entry.key} fill={SEVERITY_BANDS.find((band) => band.key === entry.key).color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
