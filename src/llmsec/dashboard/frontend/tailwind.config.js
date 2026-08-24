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
        surface: {
          primary: "#0f1115",
          secondary: "#171b22",
          // NEW (D-12-*, Phase 12): one step up from secondary --
          // card/row hover states.
          raised: "#1d222b",
          // NEW: one step darker than secondary -- the command-strip and
          // scan-progress log-panel background pinned by
          // shell-and-navigation.md / scan-trigger-and-progress.md. Missing
          // from the UI-SPEC's extension block; added here from the pinned
          // sketch references (see 12-01-SUMMARY.md).
          sunken: "#0a0c10",
        },
        // NEW: dedicated border token, replaces ad-hoc
        // border-surface-secondary/border-surface-primary usage.
        border: { DEFAULT: "#262c37", hover: "#333b48" },
        // NEW: three-tier text hierarchy. `text.DEFAULT` makes the
        // previously-hardcoded inline body color (index.css) a real token.
        text: { DEFAULT: "#e5e7eb", muted: "#8b93a1", faint: "#5b6472" },
        accent: "#4f8cff",
        // NEW: hover state for accent, from the pinned theme file and
        // foundations.md's button pattern -- not in the UI-SPEC's extension
        // block, copied from the pinned sources rather than invented (see
        // 12-01-SUMMARY.md).
        "accent-hover": "#6c9fff",
        // NEW: active-state background wash (nav item, chip, tab,
        // phase-tag).
        "accent-dim": "rgba(79, 140, 255, 0.12)",
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
        // CHANGED (Phase 12, D-12-06): Inter leads the existing system
        // stack as the primary sans face.
        sans: ["Inter", "-apple-system", '"Segoe UI"', "Roboto", "system-ui", "sans-serif"],
        // CHANGED: JetBrains Mono leads the existing mono stack.
        mono: ["\"JetBrains Mono\"", "ui-monospace", '"SFMono-Regular"', "Consolas", "monospace"],
        // NEW: Sora display face -- headings, brand wordmark.
        display: ["Sora", "system-ui", "sans-serif"],
      },
      // NEW (Phase 12): border radius scale from the pinned theme file.
      borderRadius: { sm: "6px", md: "10px", lg: "14px", full: "9999px" },
      // NEW (Phase 12): elevation + glow scale. glow-accent is reserved for
      // primary-button hover only; glow-critical is reserved for the
      // critical-severity dot only (foundations.md).
      boxShadow: {
        sm: "0 1px 2px rgba(0,0,0,0.4)",
        md: "0 4px 16px rgba(0,0,0,0.45)",
        lg: "0 12px 32px rgba(0,0,0,0.5)",
        "glow-accent": "0 0 0 1px rgba(79,140,255,0.3), 0 0 24px rgba(79,140,255,0.15)",
        "glow-critical": "0 0 8px rgba(239,68,68,0.7)",
      },
      // NEW (Phase 12, D-12-03): CSS-driven blinking cursor -- no timer API,
      // matches scan-trigger-and-progress.md's pinned cursor CSS. Lives here
      // (not in plan 12-04) because this plan is the only one that owns
      // tailwind.config.js.
      keyframes: {
        blink: {
          "0%, 50%": { opacity: 1 },
          "51%, 100%": { opacity: 0 },
        },
      },
      animation: {
        blink: "blink 1s step-start infinite",
      },
    },
  },
  plugins: [],
};
