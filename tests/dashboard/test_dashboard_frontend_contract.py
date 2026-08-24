"""Source-contract test pinning every UI-SPEC copy string to the frontend
source file that must contain it verbatim, plus negative structural
assertions over the whole frontend source tree.

This is a pure-filesystem test module: it reads source files and asserts on
their text. It imports nothing from the optional `[dashboard]` web stack and
carries no skip-if-missing availability guard of any kind (no conditional
skip keyed on the `fastapi` package, no `fastapi`/`uvicorn` import) -- it
must run, and can fail, in every environment, including one where the
`[dashboard]` extra was never installed.

This file is the automated half of 10-04-PLAN.md assumption A-04's
compensating controls: this project has no JS test runner this phase, so
frontend correctness is gated by three mechanisms instead of a real
component/behavioral test suite -- `npm run build` (fails on any syntax or
unresolved-import error), this Python-side contract test (copy strings +
negative structural assertions), and the blocking `checkpoint:human-verify`
in 10-06 that walks every enumerated UI state in a real browser.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from llmsec.plugins.registry import BUILTIN_MODULE_IDS

_REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = _REPO_ROOT / "src" / "llmsec" / "dashboard" / "frontend" / "src"
PACKAGE_JSON = FRONTEND_SRC.parent / "package.json"

# Each entry: (copy string, path relative to FRONTEND_SRC that must contain
# it verbatim). Login form's four (10-01), filter bar's four, report list's
# five empty/error strings (10-04) -- every literal quoted verbatim from
# 10-UI-SPEC.md's Copywriting Contract table.
COPY_STRINGS: list[tuple[str, str]] = [
    # LoginForm.jsx (10-01, terminal-framed by 12-01 per D-12-05 -- the
    # submit label changed to `$ unlock`; the 401/503/fallback strings stay
    # byte-for-byte).
    ("$ unlock", "components/LoginForm.jsx"),
    ("❯ authenticate", "components/LoginForm.jsx"),
    ("DASHBOARD TOKEN", "components/LoginForm.jsx"),
    (
        "Invalid token. Double-check `LLMSEC_DASHBOARD_TOKEN` and try again.",
        "components/LoginForm.jsx",
    ),
    (
        "Dashboard isn't configured yet. Ask your admin to set `LLMSEC_DASHBOARD_TOKEN` and restart it.",
        "components/LoginForm.jsx",
    ),
    ("Couldn't load reports. Check your connection and try again.", "components/LoginForm.jsx"),
    # CommandStrip.jsx (12-02 Task 2) -- shell brand + primary CTA.
    ("://", "components/CommandStrip.jsx"),
    ("+ new_scan", "components/CommandStrip.jsx"),
    # FilterBar.jsx (10-04 Task 2, chrome reframed by 12-02 Task 2 per D-12-07 --
    # the select/chip controls and their copy below are unchanged)
    ("Loading filters…", "components/FilterBar.jsx"),
    ("Filters unavailable", "components/FilterBar.jsx"),
    ("All targets", "components/FilterBar.jsx"),
    ("Any severity", "components/FilterBar.jsx"),
    ("grep target, module, severity…", "components/FilterBar.jsx"),
    # ReportList.jsx (10-04 Task 3)
    ("No scan reports yet", "components/ReportList.jsx"),
    ("Run `llmsec scan` from the CLI", "components/ReportList.jsx"),
    ("No reports match these filters", "components/ReportList.jsx"),
    ("Try clearing one or more filters", "components/ReportList.jsx"),
    ("Couldn't load reports.", "components/ReportList.jsx"),
    # ReportDetail.jsx (10-05 Task 2, terminal-framed by 12-03 Task 2 per
    # D-12-*: the download action's label changed to `↓ download.json`; two
    # new pins below (`llmsec://report/`, `SEVERITY BREAKDOWN`) cover the
    # new mono header/section-label copy this same task introduces)
    ("↓ download.json", "components/ReportDetail.jsx"),
    ("Downloading", "components/ReportDetail.jsx"),
    ("llmsec://report/", "components/ReportDetail.jsx"),
    ("SEVERITY BREAKDOWN", "components/ReportDetail.jsx"),
    ("Couldn't download this report. Try again.", "components/ReportDetail.jsx"),
    (
        "This report doesn't exist or its scan id is invalid.",
        "components/ReportDetail.jsx",
    ),
    (
        "This report's file is corrupted and can't be read.",
        "components/ReportDetail.jsx",
    ),
    (
        "No findings — every test case in this scan was blocked.",
        "components/ReportDetail.jsx",
    ),
    # RawCases.jsx (10-05 Task 2)
    ("Show raw cases (", "components/RawCases.jsx"),
    ("Hide raw cases", "components/RawCases.jsx"),
    ("No case log recorded", "components/RawCases.jsx"),
    # AuthorizationPanel.jsx (11-03 Task 1) -- 11-UI-SPEC.md Copywriting
    # Contract. The disclaimer body sentence is split across three literal
    # constants in source (prefix / emphasised phrase / suffix) because the
    # middle portion is rendered inside a real <strong> tag rather than
    # markdown-style asterisks -- pinned as three separate entries so the
    # full sentence is covered without asserting a string that never exists
    # contiguously in the raw file text.
    ("Authorization Required", "components/AuthorizationPanel.jsx"),
    (
        "llmsec is about to send live requests to the target below. This tool is intended for ",
        "components/AuthorizationPanel.jsx",
    ),
    ("authorized security testing only", "components/AuthorizationPanel.jsx"),
    (
        (
            ". Scanning systems without explicit written permission from the owner may violate "
            "computer fraud laws (e.g. CFAA) and provider Terms of Service. The authors assume no "
            "liability for unauthorized use."
        ),
        "components/AuthorizationPanel.jsx",
    ),
    ("You are about to scan:", "components/AuthorizationPanel.jsx"),
    (
        "I confirm I have explicit authorization to test this target.",
        "components/AuthorizationPanel.jsx",
    ),
    # ScanForm.jsx (11-03 Task 1) -- 11-UI-SPEC.md Copywriting Contract.
    ("New Scan", "components/ScanForm.jsx"),
    ("MODULES — ", "components/ScanForm.jsx"),
    ("/10 SELECTED", "components/ScanForm.jsx"),
    ("Select at least one module to run.", "components/ScanForm.jsx"),
    (
        "Environment variable name only — never paste a literal API key here.",
        "components/ScanForm.jsx",
    ),
    (
        "If you know the target's system prompt, paste it here to sharpen leakage detection.",
        "components/ScanForm.jsx",
    ),
    ("Advanced options", "components/ScanForm.jsx"),
    ("Add header", "components/ScanForm.jsx"),
    ("$ start_scan", "components/ScanForm.jsx"),
    ("Starting…", "components/ScanForm.jsx"),
    ("This field is required.", "components/ScanForm.jsx"),
    ("This target can't be scanned.", "components/ScanForm.jsx"),
    (
        "llmsec refuses to scan private, loopback, link-local, or cloud-metadata addresses. Enter a public target URL.",
        "components/ScanForm.jsx",
    ),
    (
        "Couldn't start the scan. Check your connection and try again.",
        "components/ScanForm.jsx",
    ),
    # 11-REVIEW.md WR-05: this string is load-bearing for a branching UI
    # decision in ScanForm.jsx (exact-equality match against
    # routes.py::_AUTHORIZATION_NOT_ATTESTED_DETAIL), not pure display
    # copy -- pinned here like every other string in this table, plus a
    # same-string cross-file assertion below (test_authorization_not_
    # attested_detail_matches_backend).
    ("You must confirm authorization before starting a scan.", "components/ScanForm.jsx"),
    # ScanProgress.jsx (11-03 Task 1) -- 11-UI-SPEC.md Copywriting Contract.
    ("Scan in Progress", "components/ScanProgress.jsx"),
    ("Queued", "components/ScanProgress.jsx"),
    ("Running", "components/ScanProgress.jsx"),
    ("Scoring", "components/ScanProgress.jsx"),
    ("Complete", "components/ScanProgress.jsx"),
    ("Waiting to start…", "components/ScanProgress.jsx"),
    ("Sending test payloads to the target…", "components/ScanProgress.jsx"),
    ("Scoring results and preparing the report…", "components/ScanProgress.jsx"),
    ("Scan complete.", "components/ScanProgress.jsx"),
    ("View Report", "components/ScanProgress.jsx"),
    ("Scan Failed", "components/ScanProgress.jsx"),
    (
        ". Run `llmsec scan` from the CLI with the same configuration for full error detail.",
        "components/ScanProgress.jsx",
    ),
    ("Back to Reports", "components/ScanProgress.jsx"),
    ("Try Again", "components/ScanProgress.jsx"),
    ("Lost track of this scan.", "components/ScanProgress.jsx"),
    (
        (
            "The dashboard may have restarted while this scan was running. Check the report list — "
            "if the scan finished, it will appear there."
        ),
        "components/ScanProgress.jsx",
    ),
]


def _read(relative_path: str) -> str:
    return (FRONTEND_SRC / relative_path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "copy_string,relative_path",
    COPY_STRINGS,
    ids=[f"{path}::{copy[:24]!r}" for copy, path in COPY_STRINGS],
)
def test_copy_string_present_verbatim(copy_string: str, relative_path: str) -> None:
    source = _read(relative_path)
    assert copy_string in source, (
        f"Expected literal copy string {copy_string!r} not found in {relative_path} "
        "-- a UI-SPEC copywriting-contract string was edited away."
    )


#: P-04: a Tailwind utility on either honest-degradation container that
#: would visually cap or abbreviate its content -- checked over the
#: `className` attribute VALUE only (never the surrounding source text),
#: so a comment/docstring mentioning one of these words cannot trip or
#: satisfy this check.
_FORBIDDEN_CLAMP_SUBSTRINGS = ("line-clamp", "truncate", "text-ellipsis", "overflow-hidden")


def _extract_class_name_for_testid(source: str, testid: str) -> str:
    """Return the `className` attribute value on the JSX element that
    carries `data-testid="{testid}"` -- found by locating that element's
    opening tag (nearest `<` before the marker, nearest `>` after it) and
    extracting `className="..."` from within that tag's text only.
    """
    marker = f'data-testid="{testid}"'
    marker_index = source.index(marker)
    tag_start = source.rindex("<", 0, marker_index)
    tag_end = source.index(">", marker_index)
    tag_text = source[tag_start : tag_end + 1]
    match = re.search(r'className="([^"]*)"', tag_text)
    assert match is not None, f"No className attribute found on the {testid!r} element's tag"
    return match.group(1)


@pytest.mark.parametrize("testid", ["scan-limitations", "deep-summary"])
def test_honest_degradation_containers_are_never_clamped(testid: str) -> None:
    """P-04 / T-10-21: neither `limitations` nor `deep_summary`'s container
    may carry a clamp/truncate/overflow-hiding utility or a `max-h-` cap on
    the SAME element as its `data-testid` marker -- an honest-degradation
    disclosure that does not fit must make the layout give way, never the
    disclosure.
    """
    source = _read("components/ReportDetail.jsx")
    class_name = _extract_class_name_for_testid(source, testid)
    for forbidden in _FORBIDDEN_CLAMP_SUBSTRINGS:
        assert forbidden not in class_name, (
            f"{testid!r} element's className contains forbidden utility "
            f"{forbidden!r}: {class_name!r}"
        )
    assert not re.search(r"\bmax-h-\S+", class_name), (
        f"{testid!r} element's className contains a max-h- cap: {class_name!r}"
    )


def test_report_list_has_two_structurally_separate_empty_state_branches() -> None:
    """The two empty-state headings must appear as separate `if` branches,
    not two values of one shared parameterized component -- guards against
    exactly the collapse the UI-SPEC forbids.
    """
    source = _read("components/ReportList.jsx")
    no_reports_index = source.index("No scan reports yet")
    filtered_index = source.index("No reports match these filters")
    between = source[min(no_reports_index, filtered_index) : max(no_reports_index, filtered_index)]
    # Each branch is guarded by its own `if (...)` -- there must be at least
    # one more `if (` between the two literals than a single shared block
    # would have, i.e. they are not both inside the same conditional.
    assert between.count("if (") >= 1


#: (hex value, band name) pairs from the UI-SPEC's severity semantic ramp
#: -- checked together so a colour cannot be silently reassigned to the
#: wrong band (T-10-22).
SEVERITY_RAMP: list[tuple[str, str]] = [
    ("#ef4444", "critical"),
    ("#f97316", "high"),
    ("#eab308", "medium"),
    ("#22c55e", "low"),
    ("#6b7280", "none"),
]


def test_severity_chart_uses_the_full_ui_spec_ramp_with_matching_band_names() -> None:
    """Every one of the five severity hex values must appear in
    `SeverityChart.jsx`, each alongside a reference to its own band name
    (never another band's) -- pins the colour-to-band mapping so a future
    edit cannot silently reassign a colour to the wrong severity.
    """
    source = _read("components/SeverityChart.jsx")
    for hex_value, band_name in SEVERITY_RAMP:
        assert hex_value in source, f"Severity ramp colour {hex_value!r} ({band_name}) not found"
        # The colour and its band key/label must appear close enough
        # together to be "alongside" each other, not merely somewhere in
        # the same file -- look within a fixed-size window around the hex
        # value's own occurrence(s) for the band key.
        for match in re.finditer(re.escape(hex_value), source):
            window = source[max(0, match.start() - 200) : match.start() + 200]
            if band_name in window:
                break
        else:
            raise AssertionError(
                f"Severity ramp colour {hex_value!r} has no nearby reference to its "
                f"band name {band_name!r} -- possible colour/band mismatch"
            )


def _all_source_files() -> list[Path]:
    return sorted(p for p in FRONTEND_SRC.rglob("*") if p.suffix in {".js", ".jsx"})


def test_no_raw_html_injection_attribute() -> None:
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in _all_source_files()
        if "dangerouslySetInnerHTML" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"Raw-HTML injection attribute found in: {offenders}"


def test_no_markdown_renderer_import() -> None:
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in _all_source_files()
        if "react-markdown" in path.read_text(encoding="utf-8")
        or "markdown-to-jsx" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"Markdown-renderer import found in: {offenders}"


def test_package_json_has_no_markdown_or_html_rendering_dependency() -> None:
    package = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    suspicious_terms = ("markdown", "html-react-parser", "dompurify-react")
    all_deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
    offenders = [name for name in all_deps if any(term in name.lower() for term in suspicious_terms)]
    assert not offenders, f"Suspicious markdown/HTML-rendering dependency in package.json: {offenders}"


def test_every_component_is_imported_by_at_least_one_other_source_file() -> None:
    """A component cannot be written, committed, and silently never
    rendered -- every `*.jsx` under `src/components/` must be imported by at
    least one other source file.
    """
    components_dir = FRONTEND_SRC / "components"
    component_files = sorted(components_dir.glob("*.jsx"))
    assert component_files, "expected at least one component under src/components/"

    all_sources = {path: path.read_text(encoding="utf-8") for path in _all_source_files()}

    for component_path in component_files:
        component_name = component_path.stem
        import_pattern = re.compile(
            rf"""from\s+["'][^"']*{re.escape(component_name)}(\.jsx)?["']"""
        )
        imported = any(
            import_pattern.search(text)
            for source_path, text in all_sources.items()
            if source_path != component_path
        )
        assert imported, f"{component_path.name} is not imported by any other source file"


# --- 11-03 Task 1: this phase's minted structural prohibitions ------------


def _strip_js_comments(source: str) -> str:
    """Strip `//` line comments and `/* ... */` block comments before a
    negative structural assertion, so a forbidden token that only appears
    inside explanatory prose (this file's own header comments mention
    "useState"/"elapsed" while documenting exactly why the component must
    NOT use them) can never false-positive-fail a real-code check.
    """
    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    without_line = re.sub(r"//[^\n]*", "", without_block)
    return without_line


def test_authorization_checkbox_is_never_default_checked() -> None:
    """D-07 / this plan's P-06 prohibition (T-11-02): the attestation
    checkbox's `checked` state must come exclusively from the parent's
    live-derived prop -- no `defaultChecked`, no internal `useState`, and no
    read from any browser-storage API that could restore a prior
    attestation across a remount (e.g. after "Try Again").
    """
    source = _read("components/AuthorizationPanel.jsx")
    marker = 'type="checkbox"'
    marker_index = source.index(marker)
    tag_start = source.rindex("<", 0, marker_index)
    tag_end = source.index(">", marker_index)
    tag_text = source[tag_start : tag_end + 1]

    assert re.search(r"checked=\{[^}]+\}", tag_text), (
        "authorization checkbox's opening tag carries no `checked={...}` binding: "
        f"{tag_text!r}"
    )
    assert "defaultChecked" not in tag_text, (
        f"authorization checkbox carries a defaultChecked attribute: {tag_text!r}"
    )

    code = _strip_js_comments(source)
    for forbidden in ("useState", "localStorage", "sessionStorage"):
        assert forbidden not in code, (
            f"AuthorizationPanel.jsx references {forbidden!r} -- the checkbox must be a "
            "purely presentational controlled child with no internal or persisted state"
        )


def test_authorization_not_attested_detail_matches_backend() -> None:
    """11-REVIEW.md WR-05: `ScanForm.jsx` branches its submit-error UI on
    exact equality against the string it hard-codes for a 400 response with
    no authorization attestation, but `routes.py::_AUTHORIZATION_NOT_
    ATTESTED_DETAIL` is the single source of truth server-side. Unlike pure
    display copy, a drift between the two silently breaks the
    not-attested branch (the error renders as the generic target-rejected
    copy instead), so this asserts the literal appears, unchanged, in both
    files -- not just pinned once in COPY_STRINGS.
    """
    detail = "You must confirm authorization before starting a scan."
    frontend_source = _read("components/ScanForm.jsx")
    backend_source = (
        _REPO_ROOT / "src" / "llmsec" / "dashboard" / "routes.py"
    ).read_text(encoding="utf-8")

    assert detail in frontend_source, (
        f"{detail!r} not found in ScanForm.jsx -- the not-attested branch's exact-"
        "equality check would never match"
    )
    assert f'_AUTHORIZATION_NOT_ATTESTED_DETAIL = "{detail}"' in backend_source, (
        f"routes.py::_AUTHORIZATION_NOT_ATTESTED_DETAIL no longer equals {detail!r} -- "
        "it has drifted from ScanForm.jsx's hard-coded copy"
    )


def test_scan_form_never_submits_an_empty_module_selection() -> None:
    """DASH-01: every id ScanForm.jsx offers must come from the real
    module registry (never a hand-typed list that could drift), and the
    zero-selected validation copy must live in the same source file as the
    checklist it guards. The submit-blocking check (inside `isFormValid()`)
    and the visible zero-selected error message (rendered independently)
    must be two structurally separate sites, not one shared condition --
    reuses the structural-separation technique from
    `test_report_list_has_two_structurally_separate_empty_state_branches`.
    """
    source = _read("components/ScanForm.jsx")

    for module_id in BUILTIN_MODULE_IDS:
        assert module_id in source, (
            f"BUILTIN_MODULE_IDS entry {module_id!r} not found in ScanForm.jsx -- the "
            "module checklist has drifted from the real plugin registry"
        )

    assert "Select at least one module to run." in source

    # The zero-selected guard must appear at least twice: once as the
    # submit-blocking condition inside isFormValid(), and once as the
    # independent condition that renders the visible error message --
    # never collapsed into a single shared check.
    guard_count = source.count("selectedModules.length === 0")
    assert guard_count >= 2, (
        "expected the zero-selected guard to appear at least twice (submit-blocking "
        f"check + visible-error-message check), found {guard_count} -- the two are not "
        "structurally separate"
    )


def test_scan_progress_failure_panel_is_never_clamped() -> None:
    """Same never-truncate-a-diagnostic-message rule as
    `test_honest_degradation_containers_are_never_clamped`, applied to the
    Scan Failed panel's recorded exception message -- a failure summary
    that does not fit must make the layout give way, never the message.
    """
    source = _read("components/ScanProgress.jsx")
    class_name = _extract_class_name_for_testid(source, "scan-failure-detail")
    for forbidden in _FORBIDDEN_CLAMP_SUBSTRINGS:
        assert forbidden not in class_name, (
            f"scan-failure-detail element's className contains forbidden utility "
            f"{forbidden!r}: {class_name!r}"
        )
    assert not re.search(r"\bmax-h-\S+", class_name), (
        f"scan-failure-detail element's className contains a max-h- cap: {class_name!r}"
    )


def test_scan_progress_reports_no_fabricated_progress() -> None:
    """DASH-06 / this plan's P-07 prohibition: the stepper's fill is a
    direct function of the server-reported phase string alone -- no
    percentage, no numeric-driven width computation, and no
    elapsed/remaining time estimate may be fabricated client-side.
    """
    code = _strip_js_comments(_read("components/ScanProgress.jsx"))
    assert "%" not in code, "found a percent sign -- possible fabricated progress percentage"
    assert "style=" not in code, "found an inline style -- possible numeric width computation"
    for forbidden in ("elapsed", "remaining", "Math.round", "Math.floor", "setInterval", "setTimeout"):
        assert forbidden not in code, (
            f"found {forbidden!r} in ScanProgress.jsx -- possible fabricated progress signal "
            "(ScanProgress must stay purely presentational, driven only by the phase string prop)"
        )


def test_scan_form_has_no_literal_secret_input() -> None:
    """D-08: config holds env-var *names* only, never literal secrets.
    Neither credential-adjacent field may be a masked/password input, and
    both must be bound to state named for an environment variable, never
    for a key value.
    """
    source = _read("components/ScanForm.jsx")
    assert 'type="password"' not in source
    assert 'type="secret"' not in source
    assert "api_key_env" in source
    assert "judge_api_key_env" in source


def test_lost_scan_state_is_structurally_distinct_from_failure() -> None:
    """D-04 / T-11-... : the restart-loses-state (lost-scan) case must
    never collapse into the generic "Scan Failed" branch -- both headings
    must be guarded by separate `if` blocks, the same structural-separation
    technique used for ReportList.jsx's two empty states.
    """
    source = _read("components/ScanProgress.jsx")
    lost_index = source.index("Lost track of this scan.")
    failed_index = source.index("Scan Failed")
    between = source[min(lost_index, failed_index) : max(lost_index, failed_index)]
    assert between.count("if (") >= 1
