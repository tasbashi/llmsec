import { useState } from "react";
import { api, ApiError, clearToken, setToken } from "../api.js";

const COPY = {
  401: "Invalid token. Double-check `LLMSEC_DASHBOARD_TOKEN` and try again.",
  503: "Dashboard isn't configured yet. Ask your admin to set `LLMSEC_DASHBOARD_TOKEN` and restart it.",
  fallback: "Couldn't load reports. Check your connection and try again.",
};

// D-07: an invalid/missing token surfaces as an inline error on THIS form
// only -- never a generic app-wide error page. The submitted token value is
// never rendered back into the error text.
function errorCopyFor(err) {
  if (err instanceof ApiError && err.status === 401) return COPY[401];
  if (err instanceof ApiError && err.status === 503) return COPY[503];
  return COPY.fallback;
}

export default function LoginForm({ onLoginSuccess }) {
  const [candidate, setCandidate] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!candidate || loading) return;
    setLoading(true);
    setError(null);
    // Reuses GET /api/reports as the implicit "is this token valid" check
    // (10-RESEARCH.md Open Question #2's resolution) rather than a
    // dedicated /api/auth/verify endpoint.
    setToken(candidate);
    try {
      const reports = await api.listReports();
      onLoginSuccess(reports);
    } catch (err) {
      clearToken();
      setError(errorCopyFor(err));
    } finally {
      setLoading(false);
    }
  }

  const disabled = loading || candidate.length === 0;

  return (
    <div className="min-h-screen flex items-center justify-center pt-3xl bg-surface-primary">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-md bg-surface-secondary border border-border rounded-lg shadow-lg p-lg flex flex-col gap-md"
      >
        <div className="flex flex-col gap-xs">
          <h1 className="text-display font-display text-text">llmsec</h1>
          <p className="text-body font-mono text-text-muted">❯ authenticate</p>
        </div>
        <div className="flex flex-col gap-xs">
          <label
            className="text-label font-mono uppercase tracking-wide text-text-muted"
            htmlFor="dashboard-token"
          >
            DASHBOARD TOKEN
          </label>
          <div className="flex items-center gap-xs w-full px-md py-sm rounded bg-surface-primary border border-border focus-within:border-border-hover">
            <span className="text-body font-mono text-accent">❯</span>
            <input
              id="dashboard-token"
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={candidate}
              onChange={(e) => setCandidate(e.target.value)}
              className="w-full text-body font-mono bg-transparent focus:outline-none overflow-x-auto whitespace-nowrap text-text"
              placeholder="Paste your LLMSEC_DASHBOARD_TOKEN"
            />
          </div>
        </div>
        {error && (
          <p className="text-body font-mono text-destructive">[error] {error}</p>
        )}
        <button
          type="submit"
          disabled={disabled}
          className="w-full bg-accent text-body font-mono rounded px-md py-sm hover:bg-accent-hover hover:shadow-glow-accent disabled:opacity-50 disabled:hover:shadow-none"
        >
          {loading ? "Unlocking…" : "$ unlock"}
        </button>
      </form>
    </div>
  );
}
