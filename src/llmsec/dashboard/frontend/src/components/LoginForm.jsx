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
        className="w-full max-w-md bg-surface-secondary rounded-lg p-lg flex flex-col gap-md"
      >
        <h1 className="text-display">llmsec Dashboard</h1>
        <label className="text-label uppercase tracking-wide" htmlFor="dashboard-token">
          Dashboard token
        </label>
        <input
          id="dashboard-token"
          type="password"
          autoComplete="off"
          spellCheck={false}
          value={candidate}
          onChange={(e) => setCandidate(e.target.value)}
          className="w-full text-body px-md py-sm rounded bg-surface-primary border border-surface-secondary focus:outline-none focus:ring-2 focus:ring-accent overflow-x-auto whitespace-nowrap"
          placeholder="Paste your LLMSEC_DASHBOARD_TOKEN"
        />
        {error && <p className="text-body text-destructive">{error}</p>}
        <button
          type="submit"
          disabled={disabled}
          className="bg-accent text-body rounded px-md py-sm disabled:opacity-50"
        >
          {loading ? "Unlocking…" : "Unlock Dashboard"}
        </button>
      </form>
    </div>
  );
}
