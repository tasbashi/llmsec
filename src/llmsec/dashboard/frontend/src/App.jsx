import { useEffect, useState } from "react";
import { api, ApiError, clearToken, getToken } from "./api.js";
import LoginForm from "./components/LoginForm.jsx";
import FilterBar from "./components/FilterBar.jsx";
import ReportList from "./components/ReportList.jsx";
import ReportDetail from "./components/ReportDetail.jsx";
import ScanForm from "./components/ScanForm.jsx";
import ScanProgress from "./components/ScanProgress.jsx";

const EMPTY_FILTERS = { target: "", modules: [], severity: "" };

// 11-02 Task 1: the progress view polls GET /api/scans/{scan_id}/status on
// this interval (Claude's Discretion in 11-CONTEXT.md, inside RESEARCH's
// recommended 1-2s band) -- declared exactly once as a module-scope
// constant, never inlined at each call site.
const SCAN_POLL_INTERVAL_MS = 2000;

// Sentinel `scanStatus` value for the D-04/A-02 restart-loses-state case: a
// status 404 for a scan_id this dashboard process no longer knows about.
// Distinguished from a real status body (which always carries `phase`) by
// its own `lost` flag, so ScanProgress can render its own honestly-worded
// branch instead of a generic 404 or infinite silent retry.
const LOST_SCAN_STATUS = { lost: true };

// No react-router-dom -- a plain useState-driven view switch mirrors
// demo-app/frontend/src/App.jsx's own tab-state pattern (D-10's "simpler is
// sufficient" philosophy; 10-RESEARCH.md Alternatives Considered).
export default function App() {
  const [view, setView] = useState("login");
  const [checkingStoredToken, setCheckingStoredToken] = useState(true);

  // Report list state machine.
  const [reports, setReports] = useState([]);
  const [reportsState, setReportsState] = useState("idle");
  const [reportsError, setReportsError] = useState(null);
  const [filters, setFilters] = useState(EMPTY_FILTERS);

  // Filter bar state machine -- independent of the report list's, so a
  // filter-source failure degrades only the filter bar, never the table
  // (UI-SPEC filter-bar error row).
  const [filterOptions, setFilterOptions] = useState(null);
  const [filtersState, setFiltersState] = useState("idle");

  // The scan_id the user clicked, and the detail-view fetch it drives.
  // `report` is never written to any storage API (prohibition P-01) --
  // it lives only in this in-memory state for the lifetime of the view.
  const [selectedScanId, setSelectedScanId] = useState(null);
  const [report, setReport] = useState(null);
  const [reportState, setReportState] = useState("idle");
  const [reportError, setReportError] = useState(null);

  // 11-02 Task 1: the scan_id and phase record that drive the new-scan /
  // progress views. Neither is ever written to any storage API (P-01).
  const [activeScanId, setActiveScanId] = useState(null);
  const [scanStatus, setScanStatus] = useState(null);

  function returnToLoginOnExpiredToken() {
    clearToken();
    setView("login");
  }

  useEffect(() => {
    async function checkStoredToken() {
      if (!getToken()) {
        setCheckingStoredToken(false);
        return;
      }
      try {
        const fetched = await api.listReports();
        setReports(fetched);
        setReportsState("ready");
        setView("list");
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          clearToken();
        }
        // Any other failure (503, network) also leaves the operator on the
        // login form -- there is nothing else to show without a report list.
        setView("login");
      } finally {
        setCheckingStoredToken(false);
      }
    }
    checkStoredToken();
  }, []);

  // Fetches the filter dropdown/chip options once on entering the list
  // view. Deliberately does NOT clear or block `reports` on failure -- a
  // filter-source failure degrades the filter bar alone (UI-SPEC filter-bar
  // error row).
  useEffect(() => {
    if (view !== "list") return;
    let active = true;
    setFiltersState("loading");
    api
      .getFilters()
      .then((options) => {
        if (!active) return;
        setFilterOptions(options);
        setFiltersState("ready");
      })
      .catch((err) => {
        if (!active) return;
        if (err instanceof ApiError && err.status === 401) {
          returnToLoginOnExpiredToken();
          return;
        }
        setFiltersState("error");
      });
    return () => {
      active = false;
    };
  }, [view]);

  // Re-issues the report list request whenever `filters` changes (including
  // the `retry` callback below, which replaces `filters` with an
  // equivalent-but-new object reference to force a refetch).
  useEffect(() => {
    if (view !== "list") return;
    let active = true;
    setReportsState("loading");
    api
      .listReports(filters)
      .then((fetched) => {
        if (!active) return;
        setReports(fetched);
        setReportsState("ready");
        setReportsError(null);
      })
      .catch((err) => {
        if (!active) return;
        if (err instanceof ApiError && err.status === 401) {
          returnToLoginOnExpiredToken();
          return;
        }
        setReportsState("error");
        setReportsError(err.message);
      });
    return () => {
      active = false;
    };
  }, [view, filters]);

  // Fetches the full report exactly once per selected scan_id -- the same
  // single fetch drives the findings table, the severity chart and the
  // raw-cases toggle (one loading treatment, UI-SPEC's "populated /
  // findings-table" / "loading / severity-chart" rows). The failing
  // numeric status is stored separately so ReportDetail can pick between
  // the 404 and the corrupted-file (500) copy rather than one flattened
  // message.
  useEffect(() => {
    if (view !== "detail" || !selectedScanId) return;
    let active = true;
    setReportState("loading");
    setReportError(null);
    api
      .getReport(selectedScanId)
      .then((fetched) => {
        if (!active) return;
        setReport(fetched);
        setReportState("ready");
      })
      .catch((err) => {
        if (!active) return;
        if (err instanceof ApiError && err.status === 401) {
          returnToLoginOnExpiredToken();
          return;
        }
        setReport(null);
        setReportState("error");
        setReportError(err instanceof ApiError ? err.status : null);
      });
    return () => {
      active = false;
    };
  }, [view, selectedScanId]);

  // 11-02 Task 1: polls GET /api/scans/{scan_id}/status on a fixed interval
  // while the progress view is showing an active scan. Three, and only
  // three, stop conditions clear the interval: reaching a terminal phase
  // (`complete`/`failed`), a 404 (the scan_id is unknown -- D-04's
  // restart-loses-state case, rendered as its own honest branch rather than
  // retried forever), and the effect's own cleanup on unmount/view change.
  // A transient network blip (any other error) leaves the last known phase
  // in place and keeps polling -- a single hiccup must never be reported as
  // a scan failure.
  useEffect(() => {
    if (view !== "progress" || !activeScanId) return;
    let active = true;
    let intervalId = null;

    function poll() {
      api
        .getScanStatus(activeScanId)
        .then((body) => {
          if (!active) return;
          setScanStatus(body);
          if (body.phase === "complete" || body.phase === "failed") {
            clearInterval(intervalId);
          }
        })
        .catch((err) => {
          if (!active) return;
          if (err instanceof ApiError && err.status === 401) {
            returnToLoginOnExpiredToken();
            clearInterval(intervalId);
            return;
          }
          if (err instanceof ApiError && err.status === 404) {
            setScanStatus(LOST_SCAN_STATUS);
            clearInterval(intervalId);
            return;
          }
          // Transient network/5xx blip -- keep the last known phase and
          // keep polling, do not surface as a scan failure.
        });
    }

    poll();
    intervalId = setInterval(poll, SCAN_POLL_INTERVAL_MS);

    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, [view, activeScanId]);

  // Derived, never stored: the two distinct empty conditions ReportList must
  // tell apart. "empty because no reports exist" vs. "empty because filters
  // exclude everything" (UI-SPEC's two separate empty-state copy blocks).
  const hasAnyFilterActive =
    Boolean(filters.target) || Boolean(filters.severity) || filters.modules.length > 0;

  function handleLoginSuccess(fetchedReports) {
    setReports(fetchedReports);
    setReportsState("ready");
    setView("list");
  }

  function handleLogout() {
    clearToken();
    setReports([]);
    setReportsState("idle");
    setFilterOptions(null);
    setFiltersState("idle");
    setFilters(EMPTY_FILTERS);
    setSelectedScanId(null);
    setReport(null);
    setReportState("idle");
    setReportError(null);
    setActiveScanId(null);
    setScanStatus(null);
    setView("login");
  }

  function handleStartNewScan() {
    setActiveScanId(null);
    setScanStatus(null);
    setView("new-scan");
  }

  // Seeds `scanStatus` with a synthetic `queued` phase so the progress view
  // always has something to render on its first paint -- the status
  // endpoint's own first poll response has not landed yet, and this phase
  // is never reachable as a genuinely empty state (UI-SPEC "empty /
  // scan-progress-view" row).
  function handleScanStarted(scanId) {
    setActiveScanId(scanId);
    setScanStatus({ scan_id: scanId, phase: "queued", error: null });
    setView("progress");
  }

  // Mirrors handleBackToReports() -- deliberately does NOT touch `filters`.
  function handleBackToReportsFromScan() {
    setActiveScanId(null);
    setScanStatus(null);
    setView("list");
  }

  function handleViewScanReport() {
    setSelectedScanId(activeScanId);
    setView("detail");
  }

  function handleRetryReports() {
    // A new object with the same values forces the filters-dependent effect
    // to re-run without changing what is actually being requested.
    setFilters((prev) => ({ ...prev }));
  }

  function handleSelectReport(scanId) {
    setSelectedScanId(scanId);
    setView("detail");
  }

  // Deliberately does NOT touch `filters` -- going back must land on the
  // same filtered list the user left, not a reset one.
  function handleBackToReports() {
    setSelectedScanId(null);
    setReport(null);
    setReportState("idle");
    setReportError(null);
    setView("list");
  }

  if (checkingStoredToken) {
    return null;
  }

  if (view === "login") {
    return <LoginForm onLoginSuccess={handleLoginSuccess} />;
  }

  if (view === "detail") {
    // The back control is owned by App.jsx (it drives the view-state
    // transition and is the reason `filters` survives the round trip),
    // rendered here so it is always present above the detail content
    // regardless of that content's loading/error/ready state.
    return (
      <div className="min-h-screen bg-surface-primary">
        <div className="px-xl pt-lg">
          <button
            type="button"
            onClick={handleBackToReports}
            className="text-body text-accent hover:underline"
          >
            ← Back to reports
          </button>
        </div>
        <ReportDetail
          report={report}
          state={reportState}
          errorStatus={reportError}
          scanId={selectedScanId}
          onUnauthorized={returnToLoginOnExpiredToken}
        />
      </div>
    );
  }

  // 11-02 Task 1: placeholder bodies -- Task 2 (ScanForm) and Task 3
  // (ScanProgress) replace these with the real components, keeping every
  // intermediate commit's `npm run build` green (mirrors 10-04 Task 1's own
  // precedent of landing the state machine one commit before the component
  // it renders).
  if (view === "new-scan") {
    return (
      <ScanForm
        onScanStarted={handleScanStarted}
        onCancel={handleBackToReportsFromScan}
        onUnauthorized={returnToLoginOnExpiredToken}
      />
    );
  }

  if (view === "progress") {
    return (
      <ScanProgress
        scanId={activeScanId}
        status={scanStatus}
        onViewReport={handleViewScanReport}
        onBackToReports={handleBackToReportsFromScan}
        onTryAgain={handleStartNewScan}
      />
    );
  }

  return (
    <div className="min-h-screen bg-surface-primary">
      <header className="flex items-center justify-between px-xl py-lg">
        <h1 className="text-display">Scan Reports</h1>
        <div className="flex items-center gap-md">
          <button
            onClick={handleStartNewScan}
            className="bg-accent text-body px-md py-sm rounded"
          >
            New Scan
          </button>
          <button
            onClick={handleLogout}
            className="text-body px-md py-sm rounded border border-surface-secondary hover:border-accent"
          >
            Log Out
          </button>
        </div>
      </header>
      <main className="px-xl flex flex-col gap-lg">
        <FilterBar
          options={filterOptions}
          state={filtersState}
          value={filters}
          onChange={setFilters}
        />
        <ReportList
          reports={reports}
          state={reportsState}
          error={reportsError}
          hasAnyFilterActive={hasAnyFilterActive}
          onRetry={handleRetryReports}
          onSelect={handleSelectReport}
        />
      </main>
    </div>
  );
}
