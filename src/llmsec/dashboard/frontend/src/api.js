// Fetch wrapper attaching `Authorization: Bearer <token>` from localStorage.
// Adapted from demo-app/frontend/src/api.js's getBaseUrl/setBaseUrl shape --
// this SPA has no base URL (same-origin, served by the same process) and
// instead persists the dashboard bearer token.

const TOKEN_STORAGE_KEY = "llmsec_dashboard_token";

// D-05: localStorage is the user's explicit choice over the originally
// recommended per-tab-lifetime storage API -- persistence across browser
// restarts was preferred. This is the ONLY value this module ever persists
// (prohibition P-01) -- no report content is ever written to localStorage
// or any other browser storage mechanism.
export function getToken() {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setToken(token) {
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// Shared by `request()` and `downloadReport()` below -- one place decides
// how the stored token becomes an `Authorization` header, so the
// bearer-guarded download fetch (which cannot use `request()`, since it
// needs a `Blob` body, not JSON/text) still authenticates identically.
function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...authHeaders(),
    ...(options.headers || {}),
  };
  const res = await fetch(path, { ...options, headers });
  const contentType = res.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await res.json()
    : await res.text();
  if (!res.ok) {
    const message =
      typeof body === "string" ? body : body?.detail || JSON.stringify(body);
    throw new ApiError(message || `Request failed (${res.status})`, res.status);
  }
  return body;
}

export const api = {
  // `filters` is `{ target, modules, severity }` -- an omitted/empty value
  // produces no query parameter at all (never an empty-valued one), so "no
  // filter" and "filter for the empty string" stay distinguishable on the
  // wire. `modules` appends one repeated `module` entry per selected id,
  // matching FastAPI's `list[str]` query binding in routes.py.
  listReports: (filters = {}) => {
    const params = new URLSearchParams();
    if (filters.target) {
      params.append("target", filters.target);
    }
    for (const moduleId of filters.modules || []) {
      params.append("module", moduleId);
    }
    if (filters.severity) {
      params.append("severity", filters.severity);
    }
    const query = params.toString();
    return request(query ? `/api/reports?${query}` : "/api/reports");
  },
  getFilters: () => request("/api/filters"),

  // Full ScanReport parity payload (D-13) -- goes through the same
  // token-attaching `request()` every other call uses, so a 404 (bad
  // scan_id) and a 500 (corrupted file, T-01-11's loud-failure path) both
  // arrive as `ApiError` with the real numeric status for the detail view
  // to branch on.
  getReport: (scanId) => request(`/api/reports/${scanId}`),

  // The download route is behind the same bearer guard as every other
  // `/api` route (T-10-23) -- a plain anchor `href` navigation carries no
  // `Authorization` header, so this must be an explicit authenticated
  // `fetch` instead. The blob and its object URL are transient: the URL is
  // revoked in a `finally` regardless of whether the click path throws,
  // and nothing here is written to any storage API (prohibition P-01).
  // 11-01's two new routes -- both routed through the same `request()`
  // helper as every existing call, so they inherit the same `Authorization`
  // header attachment, the same JSON/text content-type branching, and the
  // same `ApiError` shape carrying a real numeric status. No new fetch
  // primitive, no base URL (this SPA is same-origin). Neither call writes
  // to `localStorage` or any other storage API (prohibition P-01 still
  // holds -- the submitted scan config and the polled status are never
  // persisted).
  startScan: (payload) =>
    request("/api/scans", { method: "POST", body: JSON.stringify(payload) }),

  getScanStatus: (scanId) => request(`/api/scans/${scanId}/status`),

  downloadReport: async (scanId) => {
    const res = await fetch(`/api/reports/${scanId}/download`, {
      headers: authHeaders(),
    });
    if (!res.ok) {
      let message = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        message = body?.detail || message;
      } catch {
        // Non-JSON error body (e.g. a plain-text FastAPI default) -- fall
        // back to the generic message above rather than throwing here.
      }
      throw new ApiError(message, res.status);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    try {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `scan_${scanId}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    } finally {
      URL.revokeObjectURL(url);
    }
  },
};
