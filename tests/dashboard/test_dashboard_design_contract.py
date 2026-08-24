"""Design-token contract for the Phase 12 terminal/CLI-native dashboard
redesign (D-12-01 through D-12-08).

This is a pure-filesystem test module in the exact style of
`test_dashboard_frontend_contract.py` -- it imports nothing from the
optional `[dashboard]` extra and carries no skip guard, so it runs in every
environment. Later plans in this phase append their own gates here rather
than opening a second design-contract module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from llmsec.scoring.engine import Severity

_REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = _REPO_ROOT / "src" / "llmsec" / "dashboard" / "frontend"
FRONTEND_SRC = FRONTEND / "src"
TAILWIND_CONFIG = FRONTEND / "tailwind.config.js"
INDEX_CSS = FRONTEND_SRC / "index.css"
MAIN_JSX = FRONTEND_SRC / "main.jsx"
PACKAGE_JSON = FRONTEND / "package.json"
SEVERITY_CHART = FRONTEND_SRC / "components" / "SeverityChart.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- Test 1: NEW token values, verbatim, one case per value ---------------

NEW_TOKEN_VALUES: list[str] = [
    "#1d222b",  # surface.raised
    "#0a0c10",  # surface.sunken
    "#262c37",  # border.DEFAULT
    "#333b48",  # border.hover
    "#e5e7eb",  # text.DEFAULT
    "#8b93a1",  # text.muted
    "#5b6472",  # text.faint
    "#6c9fff",  # accent-hover
    "rgba(79, 140, 255, 0.12)",  # accent-dim
]


@pytest.mark.parametrize("token_value", NEW_TOKEN_VALUES, ids=NEW_TOKEN_VALUES)
def test_tailwind_config_contains_new_token_value(token_value: str) -> None:
    source = _read(TAILWIND_CONFIG)
    assert token_value in source, (
        f"Expected NEW token value {token_value!r} not found verbatim in "
        "tailwind.config.js"
    )


# --- Test 2: no drift on the load-bearing pre-existing five ---------------

PRE_EXISTING_TOKEN_VALUES: list[str] = [
    "#0f1115",  # surface.primary
    "#171b22",  # surface.secondary
    "#4f8cff",  # accent
    "#ef4444",  # destructive / severity.critical
    "#f97316",  # severity.high
    "#eab308",  # severity.medium
    "#22c55e",  # severity.low
    "#6b7280",  # severity.none
]


@pytest.mark.parametrize(
    "token_value", PRE_EXISTING_TOKEN_VALUES, ids=PRE_EXISTING_TOKEN_VALUES
)
def test_tailwind_config_pre_existing_tokens_unchanged(token_value: str) -> None:
    source = _read(TAILWIND_CONFIG)
    assert token_value in source, (
        f"Pre-existing token value {token_value!r} missing from "
        "tailwind.config.js -- a redesign must never restate a pre-existing "
        "token at a drifted value"
    )


# --- Test 3: elevation + glow scale ----------------------------------------

SHADOW_VALUES: list[str] = [
    "0 1px 2px rgba(0,0,0,0.4)",
    "0 4px 16px rgba(0,0,0,0.45)",
    "0 12px 32px rgba(0,0,0,0.5)",
    "0 0 0 1px rgba(79,140,255,0.3), 0 0 24px rgba(79,140,255,0.15)",
    "0 0 8px rgba(239,68,68,0.7)",
]


@pytest.mark.parametrize("shadow_value", SHADOW_VALUES, ids=SHADOW_VALUES)
def test_tailwind_config_contains_shadow_value(shadow_value: str) -> None:
    source = _read(TAILWIND_CONFIG)
    assert shadow_value in source, (
        f"Expected shadow/glow value {shadow_value!r} not found verbatim in "
        "tailwind.config.js"
    )


# --- Test 4: single source of truth for body colour ------------------------

_HEX_LITERAL_RE = re.compile(r"#[0-9a-fA-F]{6}")


def test_index_css_has_no_hex_literal() -> None:
    source = _read(INDEX_CSS)
    matches = _HEX_LITERAL_RE.findall(source)
    assert not matches, (
        f"index.css contains six-digit hex literal(s) {matches} -- the body "
        "colour must come from the text.DEFAULT Tailwind token, not a "
        "hardcoded hex"
    )


# --- Test 5: three faces declared in fontFamily -----------------------------


@pytest.mark.parametrize("face", ["Sora", "Inter", "JetBrains Mono"], ids=["Sora", "Inter", "JetBrainsMono"])
def test_tailwind_config_names_font_face(face: str) -> None:
    source = _read(TAILWIND_CONFIG)
    font_family_match = re.search(r"fontFamily:\s*\{.*?\}", source, re.DOTALL)
    assert font_family_match is not None, "no fontFamily block found in tailwind.config.js"
    assert face in font_family_match.group(0), (
        f"{face!r} not named in tailwind.config.js's fontFamily block"
    )


# --- Test 6: self-hosted, not CDN -------------------------------------------

_FONT_CDN_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")


def _all_frontend_src_files() -> list[Path]:
    return sorted(p for p in FRONTEND_SRC.rglob("*") if p.is_file())


def test_no_font_cdn_reference_anywhere_in_src() -> None:
    offenders = []
    for path in _all_frontend_src_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(host in text for host in _FONT_CDN_HOSTS):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, f"Font CDN reference found in: {offenders}"


def test_no_font_cdn_reference_in_index_css() -> None:
    source = _read(INDEX_CSS)
    for host in _FONT_CDN_HOSTS:
        assert host not in source, f"index.css references font CDN host {host!r}"


@pytest.mark.parametrize(
    "package_name",
    ["@fontsource/sora", "@fontsource/inter", "@fontsource/jetbrains-mono"],
)
def test_main_jsx_imports_css_from_each_fontsource_package(package_name: str) -> None:
    source = _read(MAIN_JSX)
    pattern = re.compile(re.escape(package_name) + r"/[^\"'\s]+\.css")
    assert pattern.search(source), (
        f"main.jsx does not import a CSS entrypoint from {package_name!r}"
    )


@pytest.mark.parametrize(
    "package_name",
    ["@fontsource/sora", "@fontsource/inter", "@fontsource/jetbrains-mono"],
)
def test_package_json_lists_fontsource_dependency(package_name: str) -> None:
    package = json.loads(_read(PACKAGE_JSON))
    deps = package.get("dependencies", {})
    assert package_name in deps, f"{package_name!r} missing from package.json dependencies"


# --- 12-02 Task 1: SeverityBadge.jsx becomes the shared severity tag -------

SEVERITY_BADGE = FRONTEND_SRC / "components" / "SeverityBadge.jsx"


def _strip_js_comments(source: str) -> str:
    """Copied from test_dashboard_frontend_contract.py's own helper (not
    imported across test modules, per this plan's own instruction) -- strips
    `//` line comments and `/* ... */` block comments before a negative
    structural assertion.
    """
    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    without_line = re.sub(r"//[^\n]*", "", without_block)
    return without_line


def test_severity_badge_critical_dot_is_the_only_glow() -> None:
    """foundations.md's explicit rule: applying the critical-glow shadow
    utility to every severity band dilutes the one "this matters" signal --
    the critical-glow shadow utility must occur exactly once in the file,
    and that occurrence must sit inside the `critical` entry of a class
    map, never near `high`/`medium`/`low`.
    """
    source = _strip_js_comments(_read(SEVERITY_BADGE))
    occurrences = [m.start() for m in re.finditer(re.escape("shadow-glow-critical"), source)]
    assert len(occurrences) == 1, (
        f"expected exactly one shadow-glow-critical occurrence in SeverityBadge.jsx, "
        f"found {len(occurrences)}"
    )
    index = occurrences[0]
    window = source[max(0, index - 200) : index + 200]
    assert "critical" in window, (
        "shadow-glow-critical occurrence's window does not mention 'critical' -- "
        "the glow must sit inside the critical band's own class entry"
    )
    # Word-boundary match, not a bare substring check: "shadow-glow-critical"
    # itself contains the substring "low" (inside "g-low"), so a plain `in`
    # check would always false-fail on the glow's own occurrence.
    for other_band in ("high", "medium", "low"):
        assert not re.search(rf"\b{other_band}\b", window), (
            f"shadow-glow-critical occurrence's window unexpectedly mentions "
            f"{other_band!r} as a whole word -- the glow may be bleeding into a "
            "non-critical band"
        )


def test_severity_badge_names_all_five_bands_with_no_interpolated_class() -> None:
    """Every one of the five severity bands must be named literally in
    SeverityBadge.jsx, and no Tailwind class name may be built via an
    interpolated template literal (`severity-${...}`) -- Tailwind's class
    scanner only sees literal, whole class-name strings in source, so an
    interpolated one would never be extracted.
    """
    source = _read(SEVERITY_BADGE)
    for band in ("critical", "high", "medium", "low", "none"):
        assert band in source, f"SeverityBadge.jsx does not name the {band!r} band"
    assert not re.search(r"severity-\$\{", source), (
        "SeverityBadge.jsx contains an interpolated Tailwind class template "
        "literal (severity-${...}) -- Tailwind's scanner would never extract it"
    )


# --- 12-03 Task 1: SeverityChart.jsx becomes an ASCII histogram, recharts removed ---


def test_severity_chart_band_order_matches_severity_enum_declaration_order() -> None:
    """D-12-02: the histogram's band order is the backend-parity contract,
    not a stylistic choice. Derive the expected sequence from `Severity`'s
    own declaration order rather than re-typing it, so this gate tracks the
    enum if it ever changes.
    """
    source = _read(SEVERITY_CHART)
    match = re.search(r"SEVERITY_BANDS\s*=\s*\[(.*?)\];", source, re.DOTALL)
    assert match is not None, "no SEVERITY_BANDS array literal found in SeverityChart.jsx"
    array_source = match.group(1)

    expected_order = [severity.value for severity in Severity]
    positions = []
    for key in expected_order:
        found = re.search(rf'key:\s*"{re.escape(key)}"', array_source)
        assert found is not None, f"SEVERITY_BANDS is missing a {key!r} band entry"
        positions.append(found.start())
    assert positions == sorted(positions), (
        f"SEVERITY_BANDS band order does not match Severity's declaration order "
        f"{expected_order} (D-12-02)"
    )


def test_severity_chart_never_filters_the_band_list() -> None:
    """T-10-22's anti-goal, restated for the ASCII treatment: a silently
    dropped empty band makes "no critical findings" indistinguishable from
    "critical was not measured". `SEVERITY_BANDS` must map directly to rows,
    never be filtered by count first.
    """
    source = _strip_js_comments(_read(SEVERITY_CHART))
    assert "SEVERITY_BANDS.map(" in source, (
        "SeverityChart.jsx does not map SEVERITY_BANDS directly to rows"
    )
    assert ".filter(" not in source, (
        "SeverityChart.jsx applies a .filter() call -- every band must render "
        "unconditionally, including a zero-count band (D-12-02)"
    )


def test_severity_chart_ships_ascii_treatment_and_drops_charting_dependency() -> None:
    """D-12-01: the block-drawing characters must appear in
    SeverityChart.jsx, no frontend source file may still reference
    `recharts`, and `package.json`'s merged dependency map must carry no
    `recharts` key.
    """
    source = _read(SEVERITY_CHART)
    assert "█" in source, "SeverityChart.jsx contains no filled block character"
    assert "░" in source, "SeverityChart.jsx contains no empty block character"

    offenders = []
    for path in _all_frontend_src_files():
        if path.suffix not in {".js", ".jsx"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "recharts" in text:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, f"recharts still referenced in: {offenders}"

    package = json.loads(_read(PACKAGE_JSON))
    all_deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
    assert "recharts" not in all_deps, "recharts still present in package.json"


# --- 12-04 Task 2: ScanProgress.jsx phase tag, spinner, observed-phase log panel ---

SCAN_PROGRESS = FRONTEND_SRC / "components" / "ScanProgress.jsx"

_STEPS_SUBTEXTS: list[str] = [
    "Waiting to start…",
    "Sending test payloads to the target…",
    "Scoring results and preparing the report…",
    "Scan complete.",
]


@pytest.mark.parametrize("subtext", _STEPS_SUBTEXTS, ids=_STEPS_SUBTEXTS)
def test_scan_progress_log_copy_is_never_re_typed(subtext: str) -> None:
    """D-12-03: the log panel must map over `STEPS`, never re-type its
    subtext copy a second time -- a second literal occurrence would mean the
    panel drifted from what the phase tag claims.
    """
    source = _strip_js_comments(_read(SCAN_PROGRESS))
    occurrences = source.count(subtext)
    assert occurrences == 1, (
        f"expected exactly one occurrence of {subtext!r} in ScanProgress.jsx, found "
        f"{occurrences} -- the log panel may be re-typing STEPS copy instead of "
        "mapping the array"
    )


def test_scan_progress_has_one_source_of_truth_for_current_phase() -> None:
    """D-12-03: the phase tag and the log panel must both derive from a
    single index computation, so they can never disagree about which phase
    the view is in.
    """
    source = _strip_js_comments(_read(SCAN_PROGRESS))
    occurrences = source.count("STEPS.findIndex(")
    assert occurrences == 1, (
        f"expected exactly one 'STEPS.findIndex(' occurrence in ScanProgress.jsx, "
        f"found {occurrences}"
    )
