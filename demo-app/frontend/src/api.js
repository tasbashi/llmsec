const BASE_URL_KEY = "dvla_api_base_url";

export function getBaseUrl() {
  return localStorage.getItem(BASE_URL_KEY) || "http://127.0.0.1:8000";
}

export function setBaseUrl(url) {
  localStorage.setItem(BASE_URL_KEY, url.replace(/\/+$/, ""));
}

async function request(path, options = {}) {
  const res = await fetch(`${getBaseUrl()}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const contentType = res.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await res.json()
    : await res.text();
  if (!res.ok) {
    const message =
      typeof body === "string" ? body : body?.detail || JSON.stringify(body);
    throw new Error(message || `Request failed (${res.status})`);
  }
  return body;
}

export const api = {
  getConfig: () => request("/admin/api/config"),

  setSystemPrompt: (prompt) =>
    request("/admin/api/system-prompt", {
      method: "PUT",
      body: JSON.stringify({ prompt }),
    }),

  addPii: (label, value) =>
    request("/admin/api/pii", {
      method: "POST",
      body: JSON.stringify({ label, value }),
    }),

  deletePii: (id) => request(`/admin/api/pii/${id}`, { method: "DELETE" }),

  toggleTool: (name, enabled) =>
    request(`/admin/api/tools/${name}`, {
      method: "PUT",
      body: JSON.stringify({ enabled }),
    }),

  setOutputMode: (mode) =>
    request("/admin/api/output-mode", {
      method: "PUT",
      body: JSON.stringify({ mode }),
    }),

  setLlmModel: (model) =>
    request("/admin/api/llm-model", {
      method: "PUT",
      body: JSON.stringify({ model }),
    }),

  reset: () => request("/admin/api/reset", { method: "POST" }),

  // /chat can come back as JSON ({response}) in "safe" mode, or raw
  // text/html in "raw" mode -- the wrapper above already normalizes that
  // into either an object or a string, so callers check the type.
  chat: (message) =>
    request("/chat", { method: "POST", body: JSON.stringify({ message }) }),
};
