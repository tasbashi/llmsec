// tailwind.config.js -- extend only, never override Tailwind's base scale
// wholesale. Token values copied verbatim from
// .planning/phases/10-dashboard-report-browsing/10-UI-SPEC.md's
// "Tailwind theme extension" block.
module.exports = {
  // The UI-SPEC's content glob is written repo-root-relative; this config
  // file lives inside the frontend directory itself, so the paths below are
  // relative to this file, not the repo root.
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        surface: { primary: "#0f1115", secondary: "#171b22" },
        accent: "#4f8cff",
        destructive: "#ef4444",
        severity: {
          critical: "#ef4444",
          high: "#f97316",
          medium: "#eab308",
          low: "#22c55e",
          none: "#6b7280",
        },
      },
      spacing: {
        xs: "4px",
        sm: "8px",
        md: "16px",
        lg: "24px",
        xl: "32px",
        "2xl": "48px",
        "3xl": "64px",
      },
      fontSize: {
        label: ["12px", { lineHeight: "1.4", fontWeight: "600" }],
        body: ["14px", { lineHeight: "1.5", fontWeight: "400" }],
        heading: ["20px", { lineHeight: "1.2", fontWeight: "600" }],
        display: ["28px", { lineHeight: "1.2", fontWeight: "600" }],
      },
      fontFamily: {
        sans: ["-apple-system", '"Segoe UI"', "Roboto", "system-ui", "sans-serif"],
        mono: ["ui-monospace", '"SFMono-Regular"', "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
