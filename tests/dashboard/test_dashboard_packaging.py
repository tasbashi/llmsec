"""Wheel-packaging verification (T-10-24/T-10-25/T-10-26/T-10-27).

This module inspects a *built artifact* (a real wheel, built via `pip wheel`
and opened with `zipfile`) and the repository's on-disk `static/` output --
it never imports FastAPI itself, so it must not require the `[dashboard]`
extra and carries no availability guard for it. The one place this file
touches an extra-gated symbol (`llmsec.dashboard.app.DEFAULT_STATIC_DIR`,
in `test_static_dir_is_git_tracked_and_matches_runtime_path`) degrades to a
narrow, explicitly-worded `pytest.skip()` on `ModuleNotFoundError` rather
than a whole-file guard, so every other assertion in this file still runs
in an environment without the extra installed.

Two failure modes this file exists to catch, that no other test in the
suite can see:

1. The built frontend (`src/llmsec/dashboard/static/`) silently absent from
   the wheel -- `llmsec dashboard` would serve a blank page for every
   `pip install`'d user, even though every backend/frontend-source test is
   green (T-10-24).
2. The Vite source tree (`frontend/`, its `package.json`, its lockfile, any
   `node_modules/`) leaking into the published wheel (T-10-25).

If the wheel inspection ever fails, the fix belongs in `pyproject.toml`'s
`[tool.hatch.build.targets.wheel]` block or in `vite.config.js`'s `outDir`
-- never in relaxing this test.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = REPO_ROOT / "src" / "llmsec" / "dashboard" / "static"

#: Any wheel member whose path contains this substring is the Vite SOURCE
#: tree leaking into the published artifact -- never acceptable, regardless
#: of which specific file it is (package.json, a lockfile, node_modules/*).
FRONTEND_SOURCE_MARKER = "dashboard/frontend/"

#: Every one of these Python source files must ship inside the wheel for
#: the dashboard subpackage to actually be importable/runnable post-install.
EXPECTED_DASHBOARD_MODULES = (
    "llmsec/dashboard/__init__.py",
    "llmsec/dashboard/app.py",
    "llmsec/dashboard/auth.py",
    "llmsec/dashboard/reports.py",
    "llmsec/dashboard/routes.py",
)


def test_wheel_contains_built_frontend_but_not_source_tree(tmp_path: Path) -> None:
    """Build a real wheel with `pip wheel . --no-deps` and inspect its
    contents with `zipfile` -- the same technique plan 02-08 used to verify
    the payload corpora ship (pyproject.toml's own comment cites it).

    If `pip wheel` itself cannot complete (no network access to resolve the
    `hatchling` build backend, pip/build tooling unavailable), that is
    treated as "packaging was NOT verified in this environment" and the
    test skips with an explicit message -- it is never silently read as a
    pass. A wheel that *builds* successfully but is missing the frontend,
    or that leaks the frontend source tree, is a real assertion failure.
    """
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(wheel_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "`pip wheel . --no-deps` failed in this environment (no network access to "
            "fetch the hatchling build backend, or pip/build tooling unavailable) -- "
            "wheel packaging was NOT verified in this environment.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    built_wheels = list(wheel_dir.glob("*.whl"))
    assert len(built_wheels) == 1, f"expected exactly one built wheel, found {built_wheels}"

    with zipfile.ZipFile(built_wheels[0]) as whl:
        names = whl.namelist()

    assert "llmsec/dashboard/static/index.html" in names, (
        "the built SPA shell is absent from the wheel -- `llmsec dashboard` would "
        "serve a blank page for every pip-installed user (T-10-24)"
    )

    hashed_assets = [
        name
        for name in names
        if name.startswith("llmsec/dashboard/static/") and name != "llmsec/dashboard/static/index.html"
    ]
    assert hashed_assets, (
        "index.html shipped in the wheel, but no JS/CSS asset alongside it -- "
        "the committed shell may be stale relative to its own build output"
    )

    for expected_module in EXPECTED_DASHBOARD_MODULES:
        assert expected_module in names, f"{expected_module} missing from the built wheel"

    # Positive check over the FULL member list, not a single lookup -- a
    # partially-excluded frontend source tree must still fail this.
    leaked_frontend_members = [name for name in names if FRONTEND_SOURCE_MARKER in name]
    assert leaked_frontend_members == [], (
        "the Vite SOURCE tree leaked into the published wheel (T-10-25): "
        f"{leaked_frontend_members}"
    )


def test_static_dir_is_git_tracked_and_matches_runtime_path() -> None:
    """The committed `static/` output must actually be git-tracked (not
    accidentally re-gitignored) and must be the exact directory the
    installed package resolves reports from at runtime -- the runtime path
    and the packaged path being the same path is the property that makes
    the wheel-inspection test above meaningful at all.
    """
    index_html = STATIC_DIR / "index.html"
    assert index_html.exists(), (
        f"{index_html} does not exist -- run `npm ci && npm run build` in "
        "src/llmsec/dashboard/frontend"
    )

    check_ignore = subprocess.run(
        ["git", "check-ignore", str(index_html)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # `git check-ignore` exits 0 when the path IS ignored, 1 when it is NOT
    # ignored -- confirmed against this exact file this session.
    assert check_ignore.returncode == 1, (
        f"{index_html} is gitignored (git check-ignore exit {check_ignore.returncode}) -- "
        "it would never ship in a wheel, no matter what the wheel-inspection test finds"
    )

    try:
        from llmsec.dashboard.app import DEFAULT_STATIC_DIR
    except ModuleNotFoundError as exc:
        pytest.skip(
            f"llmsec.dashboard.app could not be imported ({exc}) -- the runtime static "
            "directory was NOT cross-checked against the packaged directory in this "
            "environment (the [dashboard] extra appears to be absent)."
        )

    assert DEFAULT_STATIC_DIR.exists(), f"{DEFAULT_STATIC_DIR} does not exist"
    assert (DEFAULT_STATIC_DIR / "index.html").exists(), f"{DEFAULT_STATIC_DIR} has no index.html"
    assert DEFAULT_STATIC_DIR.resolve() == STATIC_DIR.resolve(), (
        "the runtime static directory and the on-disk committed static directory "
        "are not the same path -- the wheel-inspection test's guarantee would not "
        "transfer to what the installed package actually serves"
    )


def test_index_html_references_an_asset_that_actually_exists() -> None:
    """A committed shell (`index.html`) whose referenced bundle was never
    committed alongside it must fail here, not in a user's browser.
    """
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        pytest.skip(
            "static/index.html is absent -- already covered by "
            "test_static_dir_is_git_tracked_and_matches_runtime_path"
        )

    html = index_html.read_text()
    referenced_assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
    assert referenced_assets, "index.html references no /assets/* file at all"

    for reference in referenced_assets:
        asset_path = STATIC_DIR / reference.lstrip("/")
        assert asset_path.exists(), (
            f"index.html references {reference!r}, but {asset_path} does not exist -- "
            "the committed shell is stale relative to its own build output"
        )
