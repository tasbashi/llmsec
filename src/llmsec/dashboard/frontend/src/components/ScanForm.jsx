import { useEffect, useState } from "react";
import { api, ApiError } from "../api.js";
import AuthorizationPanel from "./AuthorizationPanel.jsx";

// Ordered LLM01 -> LLM10 (11-UI-SPEC.md "populated | module-checklist"
// row). Ids match `llmsec.plugins.registry.BUILTIN_MODULE_IDS` exactly --
// verified against `pyproject.toml`'s
// `[project.entry-points."llmsec.modules"]` and each module's own
// `owasp_ref` class attribute (D-01).
const MODULES = [
  { id: "prompt_injection", owasp: "LLM01:2025", name: "Prompt Injection & Jailbreaking" },
  { id: "pii_exfiltration", owasp: "LLM02:2025", name: "Data Exfiltration & PII Leak" },
  { id: "supply_chain", owasp: "LLM03:2025", name: "Supply Chain Vulnerabilities" },
  { id: "data_poisoning", owasp: "LLM04:2025", name: "Data and Model Poisoning" },
  { id: "insecure_output", owasp: "LLM05:2025", name: "Insecure Output Handling" },
  { id: "excessive_agency", owasp: "LLM06:2025", name: "Excessive Agency" },
  { id: "system_prompt_leakage", owasp: "LLM07:2025", name: "System Prompt Leakage" },
  { id: "vector_embedding_weaknesses", owasp: "LLM08:2025", name: "Vector & Embedding Weaknesses" },
  { id: "misinformation", owasp: "LLM09:2025", name: "Misinformation" },
  { id: "unbounded_consumption", owasp: "LLM10:2025", name: "Unbounded Consumption" },
];

// Matches `detection/judge.py::DEFAULT_JUDGE_MODEL` / `routes.py`'s own
// `ScanRequest.judge_model` default. No shared frontend/backend constant
// seam exists (same discretion already exercised by every other prefilled
// default on this form).
const DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini";

// Fixed, non-interpolating string `routes.py::_AUTHORIZATION_NOT_ATTESTED_
// DETAIL` returns for a missing/false `authorization_attested`. Matched by
// exact equality (never a substring heuristic) -- the ONLY other 400
// source on `POST /api/scans` is `TargetRejected`'s three possible
// messages (all from the SSRF gate, all sharing the "can't be scanned"
// treatment below), so this binary split is exhaustive.
const AUTHORIZATION_NOT_ATTESTED_DETAIL = "You must confirm authorization before starting a scan.";

const REQUIRED_MESSAGE = "This field is required.";
const URL_MESSAGE = "Enter a valid URL, e.g. `https://api.example.com/chat`.";

function requiredError(value) {
  return value.trim().length === 0 ? REQUIRED_MESSAGE : null;
}

function urlError(value) {
  if (!value.trim()) return REQUIRED_MESSAGE;
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return URL_MESSAGE;
    return null;
  } catch {
    return URL_MESSAGE;
  }
}

// Same collapsed-by-default chevron shape as RawCases.jsx (D-14 precedent).
function ChevronIcon({ expanded }) {
  return (
    <svg
      className={"w-4 h-4 stroke-current transition-transform" + (expanded ? " rotate-180" : "")}
      viewBox="0 0 24 24"
      fill="none"
      strokeWidth="2"
      aria-hidden="true"
    >
      <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const INPUT_CLASS =
  "w-full text-body px-md py-sm rounded bg-surface-primary border border-surface-secondary focus:outline-none focus:ring-2 focus:ring-accent";
const LABEL_CLASS = "text-label uppercase tracking-wide";
const HELPER_CLASS = "text-body";
const ERROR_CLASS = "text-body text-destructive";

// Env-var-name-only helper (D-08) -- shared verbatim by both credential-
// adjacent fields (raw_llm.api_key_env, judge_api_key_env). Neither field
// is ever a masked/password input or bound to a value that could be a
// literal key.
const ENV_VAR_HELPER_TEXT =
  "Environment variable name only — never paste a literal API key here.";

export default function ScanForm({ onScanStarted, onCancel, onUnauthorized }) {
  const [targetType, setTargetType] = useState("http_app");

  // Both target-type groups keep their own state object, never a single
  // shared object -- toggling between them hides the inactive group's
  // fields without discarding whatever was already typed into it.
  const [httpApp, setHttpApp] = useState({
    method: "POST",
    url: "",
    headers: [],
    body_template: "",
    response_path: "",
    session_id_path: "",
    session_id_header: "",
  });
  const [rawLlm, setRawLlm] = useState({ model: "", api_key_env: "" });

  // Initialised to all ten ids explicitly checked (never an omitted field,
  // never an empty start) -- the empty-selection guard below is what keeps
  // an intentional "select none" from ever reaching the wire as an empty
  // array (which the server would read as "run everything").
  const [selectedModules, setSelectedModules] = useState(() => MODULES.map((m) => m.id));

  const [judgeModel, setJudgeModel] = useState(DEFAULT_JUDGE_MODEL);
  const [judgeApiKeyEnv, setJudgeApiKeyEnv] = useState("");
  const [knownSystemPrompt, setKnownSystemPrompt] = useState("");
  const [maxConcurrency, setMaxConcurrency] = useState("5");

  const [sessionOpen, setSessionOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [authChecked, setAuthChecked] = useState(false);
  const [touched, setTouched] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  function markTouched(field) {
    setTouched((prev) => ({ ...prev, [field]: true }));
  }

  function updateHttpApp(field, value) {
    setHttpApp((prev) => ({ ...prev, [field]: value }));
  }

  function updateRawLlm(field, value) {
    setRawLlm((prev) => ({ ...prev, [field]: value }));
  }

  function addHeaderRow() {
    setHttpApp((prev) => ({ ...prev, headers: [...prev.headers, { key: "", value: "" }] }));
  }

  function updateHeaderRow(index, field, value) {
    setHttpApp((prev) => ({
      ...prev,
      headers: prev.headers.map((row, i) => (i === index ? { ...row, [field]: value } : row)),
    }));
  }

  function removeHeaderRow(index) {
    setHttpApp((prev) => ({ ...prev, headers: prev.headers.filter((_, i) => i !== index) }));
  }

  function toggleModule(moduleId) {
    setSelectedModules((prev) =>
      prev.includes(moduleId) ? prev.filter((id) => id !== moduleId) : [...prev, moduleId]
    );
  }

  const allModulesSelected = selectedModules.length === MODULES.length;
  function toggleSelectAll() {
    setSelectedModules(allModulesSelected ? [] : MODULES.map((m) => m.id));
  }

  // D-07/T-11-02: the live-derived target summary the authorization panel
  // reads -- never a snapshot taken at check-time.
  const targetSummary =
    targetType === "http_app" ? `${httpApp.method || "POST"} ${httpApp.url}`.trim() : rawLlm.model;

  // Any edit to a field feeding `targetSummary` unchecks the attestation --
  // the user always re-confirms against the value actually about to be
  // submitted (T-11-02's mitigation: the real gate is server-side, but the
  // client-side box must never imply confirmation of a stale value).
  useEffect(() => {
    setAuthChecked(false);
  }, [targetSummary]);

  function handleTargetTypeChange(nextType) {
    setTargetType(nextType);
    setSubmitError(null);
  }

  function isFormValid() {
    if (selectedModules.length === 0) return false;
    if (targetType === "http_app") {
      if (urlError(httpApp.url)) return false;
    } else {
      // _build_adapter() in api.py raises unless BOTH model and
      // api_key_env are set for a raw_llm target (11-REVIEW.md WR-02) --
      // client-side validation must require both, not just model, or a
      // submission with a blank env-var field passes 422 validation, runs
      // to "running", and only then fails inside the background task.
      if (requiredError(rawLlm.model)) return false;
      if (requiredError(rawLlm.api_key_env)) return false;
    }
    if (requiredError(judgeModel)) return false;
    const concurrency = Number(maxConcurrency);
    // Mirrors routes.py::ScanRequest.max_concurrency's Field(ge=1, le=50)
    // (11-REVIEW.md IN-02) -- rejecting client-side too so a value above
    // the server's ceiling gets an inline error instead of a 422 surfaced
    // only after submit.
    if (!Number.isFinite(concurrency) || concurrency < 1 || concurrency > 50) return false;
    return true;
  }

  // Field names match `routes.py::ScanRequest` exactly -- a rename on
  // either side silently breaks DASH-01 with a 422. Omits the inactive
  // target group's fields entirely (never sends them as null).
  function buildPayload() {
    const target =
      targetType === "http_app"
        ? {
            type: "http_app",
            method: httpApp.method || "POST",
            url: httpApp.url,
            headers: Object.fromEntries(
              httpApp.headers
                .filter((row) => row.key.trim())
                .map((row) => [row.key.trim(), row.value])
            ),
            body_template: httpApp.body_template || null,
            response_path: httpApp.response_path || null,
            session_id_path: httpApp.session_id_path || null,
            session_id_header: httpApp.session_id_header || null,
          }
        : {
            type: "raw_llm",
            model: rawLlm.model,
            api_key_env: rawLlm.api_key_env || null,
          };
    return {
      target,
      enabled_modules: selectedModules,
      judge_model: judgeModel,
      judge_api_key_env: judgeApiKeyEnv || null,
      known_system_prompt: knownSystemPrompt || null,
      max_concurrency: Number(maxConcurrency),
      authorization_attested: authChecked,
    };
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!authChecked || !isFormValid() || submitting) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const body = await api.startScan(buildPayload());
      onScanStarted(body.scan_id);
      return;
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        onUnauthorized?.();
        return;
      }
      if (err instanceof ApiError && err.status === 400) {
        if (err.message === AUTHORIZATION_NOT_ATTESTED_DETAIL) {
          setSubmitError({ kind: "not-attested" });
        } else {
          setSubmitError({ kind: "target-rejected", detail: err.message });
        }
      } else {
        setSubmitError({ kind: "generic" });
      }
    }
    setSubmitting(false);
  }

  const submitDisabled = !authChecked || !isFormValid() || submitting;

  return (
    <div className="min-h-screen bg-surface-primary px-xl py-lg flex flex-col gap-xl">
      <div className="flex flex-col gap-md">
        <button
          type="button"
          onClick={onCancel}
          className="text-body text-accent hover:underline self-start"
        >
          ← Back to reports
        </button>
        <h1 className="text-display">New Scan</h1>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-2xl max-w-3xl">
        {/* Target section -- the largest, first-encountered field group
            (UI-SPEC Visual Hierarchy). */}
        <section className="flex flex-col gap-md bg-surface-secondary rounded p-lg">
          <h2 className="text-heading">Target</h2>

          <div className="flex items-center gap-lg">
            <label className="flex items-center gap-sm text-body">
              <input
                type="radio"
                name="target-type"
                checked={targetType === "http_app"}
                onChange={() => handleTargetTypeChange("http_app")}
              />
              HTTP Application
            </label>
            <label className="flex items-center gap-sm text-body">
              <input
                type="radio"
                name="target-type"
                checked={targetType === "raw_llm"}
                onChange={() => handleTargetTypeChange("raw_llm")}
              />
              Raw LLM API
            </label>
          </div>

          {targetType === "http_app" ? (
            <div className="flex flex-col gap-md">
              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS} htmlFor="target-method">
                  Method
                </label>
                <input
                  id="target-method"
                  value={httpApp.method}
                  onChange={(e) => updateHttpApp("method", e.target.value)}
                  className={INPUT_CLASS}
                />
              </div>

              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS} htmlFor="target-url">
                  Target URL
                </label>
                <input
                  id="target-url"
                  value={httpApp.url}
                  onChange={(e) => {
                    updateHttpApp("url", e.target.value);
                    setSubmitError(null);
                  }}
                  onBlur={() => markTouched("url")}
                  className={INPUT_CLASS}
                />
                {touched.url && urlError(httpApp.url) && (
                  <p className={ERROR_CLASS}>{urlError(httpApp.url)}</p>
                )}
              </div>

              {/* Zero rows is the valid default (matches TargetConfig.
                  headers's empty-dict default) -- no placeholder row is
                  forced. Scrolls inside a fixed-height container past ~5
                  rows rather than growing the page unboundedly. */}
              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS}>Headers</label>
                {httpApp.headers.length > 0 && (
                  <div className="max-h-40 overflow-y-auto flex flex-col gap-sm">
                    {httpApp.headers.map((row, index) => (
                      <div key={index} className="flex items-center gap-sm">
                        <input
                          value={row.key}
                          onChange={(e) => updateHeaderRow(index, "key", e.target.value)}
                          placeholder="Header name"
                          className={INPUT_CLASS}
                        />
                        <input
                          value={row.value}
                          onChange={(e) => updateHeaderRow(index, "value", e.target.value)}
                          placeholder="Header value"
                          className={INPUT_CLASS}
                        />
                        <button
                          type="button"
                          onClick={() => removeHeaderRow(index)}
                          aria-label="Remove header"
                          className="w-6 h-6 shrink-0 flex items-center justify-center rounded border border-surface-secondary text-body"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <button
                  type="button"
                  onClick={addHeaderRow}
                  className="text-body text-accent self-start"
                >
                  Add header
                </button>
              </div>

              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS} htmlFor="target-body-template">
                  Body Template
                </label>
                <textarea
                  id="target-body-template"
                  value={httpApp.body_template}
                  onChange={(e) => updateHttpApp("body_template", e.target.value)}
                  className={INPUT_CLASS + " font-mono h-40 overflow-y-auto"}
                />
              </div>

              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS} htmlFor="target-response-path">
                  Response Path
                </label>
                <input
                  id="target-response-path"
                  value={httpApp.response_path}
                  onChange={(e) => updateHttpApp("response_path", e.target.value)}
                  className={INPUT_CLASS}
                />
              </div>

              {/* Collapsed by default (mirrors RawCases.jsx's D-14
                  precedent). */}
              <div className="flex flex-col gap-sm">
                <button
                  type="button"
                  aria-expanded={sessionOpen}
                  onClick={() => setSessionOpen((prev) => !prev)}
                  className="flex items-center gap-sm text-body text-accent self-start"
                >
                  <ChevronIcon expanded={sessionOpen} />
                  Multi-turn session (optional)
                </button>
                {sessionOpen && (
                  <div className="flex flex-col gap-md">
                    <div className="flex flex-col gap-xs">
                      <label className={LABEL_CLASS} htmlFor="session-id-path">
                        Session ID Path
                      </label>
                      <input
                        id="session-id-path"
                        value={httpApp.session_id_path}
                        onChange={(e) => updateHttpApp("session_id_path", e.target.value)}
                        className={INPUT_CLASS}
                      />
                    </div>
                    <div className="flex flex-col gap-xs">
                      <label className={LABEL_CLASS} htmlFor="session-id-header">
                        Session ID Header
                      </label>
                      <input
                        id="session-id-header"
                        value={httpApp.session_id_header}
                        onChange={(e) => updateHttpApp("session_id_header", e.target.value)}
                        className={INPUT_CLASS}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-md">
              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS} htmlFor="raw-llm-model">
                  Model
                </label>
                <input
                  id="raw-llm-model"
                  value={rawLlm.model}
                  onChange={(e) => {
                    updateRawLlm("model", e.target.value);
                    setSubmitError(null);
                  }}
                  onBlur={() => markTouched("model")}
                  placeholder="openai/gpt-4o-mini"
                  className={INPUT_CLASS}
                />
                {touched.model && requiredError(rawLlm.model) && (
                  <p className={ERROR_CLASS}>{requiredError(rawLlm.model)}</p>
                )}
              </div>
              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS} htmlFor="raw-llm-api-key-env">
                  API Key Env Var
                </label>
                <input
                  id="raw-llm-api-key-env"
                  type="text"
                  value={rawLlm.api_key_env}
                  onChange={(e) => {
                    updateRawLlm("api_key_env", e.target.value);
                    setSubmitError(null);
                  }}
                  onBlur={() => markTouched("api_key_env")}
                  placeholder="OPENAI_API_KEY"
                  className={INPUT_CLASS}
                />
                <p className={HELPER_CLASS}>{ENV_VAR_HELPER_TEXT}</p>
                {touched.api_key_env && requiredError(rawLlm.api_key_env) && (
                  <p className={ERROR_CLASS}>{requiredError(rawLlm.api_key_env)}</p>
                )}
              </div>
            </div>
          )}
        </section>

        {/* Enabled Modules -- mid-form secondary focal point. */}
        <section className="flex flex-col gap-md bg-surface-secondary rounded p-lg">
          <div className="flex items-center justify-between">
            <h2 className="text-heading">Enabled Modules</h2>
            <button type="button" onClick={toggleSelectAll} className="text-body text-accent">
              {allModulesSelected ? "Select none" : "Select all"}
            </button>
          </div>
          <p className={HELPER_CLASS}>
            Select which of the 10 OWASP LLM Top 10 test modules to run. All modules are selected by default.
          </p>
          <div className="flex flex-col gap-sm">
            {MODULES.map((mod) => (
              <label key={mod.id} className="flex items-center gap-sm text-body">
                <input
                  type="checkbox"
                  checked={selectedModules.includes(mod.id)}
                  onChange={() => toggleModule(mod.id)}
                />
                {mod.name}
                <span className="text-label px-xs py-xs rounded bg-surface-primary">
                  {mod.owasp}
                </span>
              </label>
            ))}
          </div>
          {selectedModules.length === 0 && (
            <p className={ERROR_CLASS}>Select at least one module to run.</p>
          )}
        </section>

        {/* Judge Model */}
        <section className="flex flex-col gap-md bg-surface-secondary rounded p-lg">
          <h2 className="text-heading">Judge Model</h2>
          <div className="flex flex-col gap-xs">
            <label className={LABEL_CLASS} htmlFor="judge-model">
              Judge Model
            </label>
            <input
              id="judge-model"
              value={judgeModel}
              onChange={(e) => setJudgeModel(e.target.value)}
              className={INPUT_CLASS}
            />
          </div>
          <div className="flex flex-col gap-xs">
            <label className={LABEL_CLASS} htmlFor="judge-api-key-env">
              Judge API Key Env Var
            </label>
            <input
              id="judge-api-key-env"
              type="text"
              value={judgeApiKeyEnv}
              onChange={(e) => setJudgeApiKeyEnv(e.target.value)}
              className={INPUT_CLASS}
            />
            <p className={HELPER_CLASS}>{ENV_VAR_HELPER_TEXT}</p>
          </div>
        </section>

        {/* Advanced options -- collapsed by default (D-14 precedent). */}
        <section className="flex flex-col gap-sm">
          <button
            type="button"
            aria-expanded={advancedOpen}
            onClick={() => setAdvancedOpen((prev) => !prev)}
            className="flex items-center gap-sm text-body text-accent self-start"
          >
            <ChevronIcon expanded={advancedOpen} />
            Advanced options
          </button>
          {advancedOpen && (
            <div className="flex flex-col gap-md bg-surface-secondary rounded p-lg">
              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS} htmlFor="known-system-prompt">
                  Known System Prompt (optional)
                </label>
                <textarea
                  id="known-system-prompt"
                  value={knownSystemPrompt}
                  onChange={(e) => setKnownSystemPrompt(e.target.value)}
                  className={INPUT_CLASS + " h-40 overflow-y-auto"}
                />
                <p className={HELPER_CLASS}>
                  If you know the target's system prompt, paste it here to sharpen leakage detection.
                </p>
              </div>
              <div className="flex flex-col gap-xs">
                <label className={LABEL_CLASS} htmlFor="max-concurrency">
                  Max Concurrency
                </label>
                <input
                  id="max-concurrency"
                  type="number"
                  min="1"
                  max="50"
                  value={maxConcurrency}
                  onChange={(e) => setMaxConcurrency(e.target.value)}
                  className={INPUT_CLASS}
                />
              </div>
            </div>
          )}
        </section>

        {/* Separated from the form body by the largest section break
            (2xl spacing token, UI-SPEC Spacing Scale). */}
        <AuthorizationPanel
          targetSummary={targetSummary}
          checked={authChecked}
          onChange={setAuthChecked}
        />

        {/* Three structurally separate submission-failure treatments,
            never one flattened message. */}
        {submitError?.kind === "target-rejected" && (
          <div className="flex flex-col gap-xs">
            <p className="text-heading text-destructive">This target can't be scanned.</p>
            <p className={ERROR_CLASS}>{submitError.detail}</p>
            <p className={ERROR_CLASS}>
              llmsec refuses to scan private, loopback, link-local, or cloud-metadata addresses. Enter a public target URL.
            </p>
          </div>
        )}
        {submitError?.kind === "not-attested" && (
          <p className={ERROR_CLASS}>{AUTHORIZATION_NOT_ATTESTED_DETAIL}</p>
        )}
        {submitError?.kind === "generic" && (
          <p className={ERROR_CLASS}>
            Couldn't start the scan. Check your connection and try again.
          </p>
        )}

        <button
          type="submit"
          disabled={submitDisabled}
          className={
            "px-md py-sm rounded text-body self-start " +
            (submitDisabled ? "bg-surface-secondary opacity-50 cursor-not-allowed" : "bg-accent")
          }
        >
          {submitting ? "Starting…" : "Start Scan"}
        </button>
      </form>
    </div>
  );
}
