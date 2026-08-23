"""DASH-03 hardening -- the full auth matrix, one class per property.

Every documented way the bearer gate can fail open is pinned here: unset
env var, empty env var, empty credential, non-ASCII credential, an ungated
`/api` route, a leaked token in an error body, a permissive CORS header, an
unauthenticated path to the reports directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason='dashboard suite requires pip install -e ".[dashboard,dev]"')

from starlette.staticfiles import StaticFiles  # noqa: E402

from tests.dashboard.conftest import TEST_DASHBOARD_TOKEN  # noqa: E402


def _api_routes(app):
    """Recursively collect every registered route whose path starts with
    `/api`, drilling through FastAPI's `_IncludedRouter` wrapper so a route
    added by ANY future plan is automatically covered -- never a
    hard-coded list.
    """
    found = []

    def _walk(candidates):
        for candidate in candidates:
            path = getattr(candidate, "path", None)
            if path is not None and path.startswith("/api"):
                found.append(candidate)
                continue
            original_router = getattr(candidate, "original_router", None)
            if original_router is not None:
                _walk(original_router.routes)
                continue
            sub_routes = getattr(candidate, "routes", None)
            if sub_routes:
                _walk(sub_routes)

    _walk(app.routes)
    return found


def _static_files_directories(app):
    """Recursively collect every resolved directory any `StaticFiles`
    instance reachable from `app` would serve from -- including
    `app.frontend()`'s low-priority frontend route, which is NOT reachable
    via `app.routes` at all (FastAPI stores it separately, checked only
    after every normal route fails to match).
    """
    directories: list[Path] = []
    seen: set[int] = set()

    def _walk(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, StaticFiles):
            directories.append(Path(obj.directory).resolve())
            return
        for attr in ("app", "original_router"):
            _walk(getattr(obj, attr, None))
        for attr in ("routes",):
            sub = getattr(obj, attr, None)
            if sub:
                for item in sub:
                    _walk(item)

    _walk(app)
    low_priority = getattr(app.router, "_low_priority_routes", None)
    if low_priority:
        for item in low_priority:
            _walk(item)
    return directories


class TestCredentialMatrix:
    """Against a client whose `LLMSEC_DASHBOARD_TOKEN` is set."""

    def test_no_authorization_header_is_401(self, client, dashboard_token) -> None:
        res = client.get("/api/reports")
        assert res.status_code == 401

    def test_empty_bearer_credential_is_401(self, client, dashboard_token) -> None:
        res = client.get("/api/reports", headers={"Authorization": "Bearer "})
        assert res.status_code == 401

    def test_wrong_token_is_401(self, client, dashboard_token) -> None:
        res = client.get("/api/reports", headers={"Authorization": "Bearer wrong-token"})
        assert res.status_code == 401

    def test_prefix_of_real_token_is_401(self, client, dashboard_token) -> None:
        prefix = dashboard_token[:-1]
        res = client.get("/api/reports", headers={"Authorization": f"Bearer {prefix}"})
        assert res.status_code == 401

    def test_real_token_plus_trailing_whitespace_is_200(self, client, dashboard_token) -> None:
        # Documents a verified FastAPI/Starlette behavior, not a dashboard
        # decision: `fastapi.security.utils.get_authorization_scheme_param()`
        # calls `.strip()` on the credential portion of the Authorization
        # header BEFORE `verify_bearer_token()` ever sees it -- so "Bearer
        # <token> " (trailing space) and "Bearer <token>" are
        # indistinguishable at the application layer by construction. This
        # is benign (accepting a token with incidental leading/trailing
        # whitespace is not a security weakening -- it's still the exact
        # secret value) and NOT bypassable without abandoning the
        # `HTTPBearer`/`Depends()` pattern RESEARCH.md Pattern 2 specifies.
        res = client.get(
            "/api/reports", headers={"Authorization": f"Bearer {dashboard_token} "}
        )
        assert res.status_code == 200

    def test_real_token_with_internal_corruption_is_401(self, client, dashboard_token) -> None:
        # A near-miss that is NOT reducible to leading/trailing whitespace
        # (which FastAPI's own header parsing strips before this dependency
        # ever runs, see the test above) -- an internal character swap must
        # still be rejected.
        corrupted = dashboard_token[:-1] + ("x" if dashboard_token[-1] != "x" else "y")
        res = client.get("/api/reports", headers={"Authorization": f"Bearer {corrupted}"})
        assert res.status_code == 401

    def test_non_ascii_token_is_401_not_500(self, client, dashboard_token) -> None:
        # Regression test: comparing raw `str` (rather than utf-8-encoded
        # `bytes`) with `secrets.compare_digest` raises `TypeError` on
        # non-ASCII input -- a token pasted from a password manager can
        # contain any character. The header value is passed as
        # already-UTF-8-encoded `bytes` (not `str`) because `httpx`'s own
        # header normalization ASCII-encodes plain `str` header values
        # client-side and would raise `UnicodeEncodeError` before the
        # request ever reaches the server -- passing bytes bypasses that
        # client-side guard so the non-ASCII bytes actually reach
        # `verify_bearer_token()`, which is the code path under test.
        res = client.get(
            "/api/reports",
            headers={"Authorization": "Bearer café-töken-éè".encode("utf-8")},
        )
        assert res.status_code == 401

    def test_exact_token_is_200(self, client, dashboard_token) -> None:
        res = client.get(
            "/api/reports", headers={"Authorization": f"Bearer {dashboard_token}"}
        )
        assert res.status_code == 200


class TestFailClosedMatrix:
    """An unset/empty server-side token must never be treated as auth-disabled."""

    def test_missing_env_var_is_503_with_no_header(self, client, monkeypatch) -> None:
        from llmsec.dashboard.auth import DASHBOARD_TOKEN_ENV_VAR

        monkeypatch.delenv(DASHBOARD_TOKEN_ENV_VAR, raising=False)
        res = client.get("/api/reports")
        assert res.status_code == 503

    def test_missing_env_var_is_503_with_empty_bearer_credential(self, client, monkeypatch) -> None:
        # The pair that distinguishes a correct fail-closed implementation
        # from one that compares "" against "".
        from llmsec.dashboard.auth import DASHBOARD_TOKEN_ENV_VAR

        monkeypatch.delenv(DASHBOARD_TOKEN_ENV_VAR, raising=False)
        res = client.get("/api/reports", headers={"Authorization": "Bearer "})
        assert res.status_code == 503

    def test_empty_string_env_var_is_503(self, client, monkeypatch) -> None:
        from llmsec.dashboard.auth import DASHBOARD_TOKEN_ENV_VAR

        monkeypatch.setenv(DASHBOARD_TOKEN_ENV_VAR, "")
        res = client.get("/api/reports")
        assert res.status_code == 503

    def test_401_and_503_details_are_distinct_strings(self, client, monkeypatch) -> None:
        from llmsec.dashboard.auth import DASHBOARD_TOKEN_ENV_VAR

        monkeypatch.delenv(DASHBOARD_TOKEN_ENV_VAR, raising=False)
        unconfigured = client.get("/api/reports").json()["detail"]

        monkeypatch.setenv(DASHBOARD_TOKEN_ENV_VAR, TEST_DASHBOARD_TOKEN)
        unauthorized = client.get("/api/reports").json()["detail"]

        assert unconfigured != unauthorized


class TestNoLeakMatrix:
    """The configured token value must never appear in a response."""

    def test_401_response_never_contains_token(self, client, dashboard_token) -> None:
        res = client.get("/api/reports", headers={"Authorization": "Bearer wrong"})
        assert TEST_DASHBOARD_TOKEN not in res.text
        assert TEST_DASHBOARD_TOKEN not in str(res.headers)
        assert TEST_DASHBOARD_TOKEN not in res.reason_phrase

    def test_503_response_never_contains_token(self, client, monkeypatch) -> None:
        from llmsec.dashboard.auth import DASHBOARD_TOKEN_ENV_VAR

        monkeypatch.setenv(DASHBOARD_TOKEN_ENV_VAR, TEST_DASHBOARD_TOKEN)
        # 503 is reached only via the unset/empty branch, but the assertion
        # itself must hold regardless of which branch produced the response.
        monkeypatch.delenv(DASHBOARD_TOKEN_ENV_VAR, raising=False)
        res = client.get("/api/reports")
        assert res.status_code == 503
        assert TEST_DASHBOARD_TOKEN not in res.text
        assert TEST_DASHBOARD_TOKEN not in str(res.headers)
        assert TEST_DASHBOARD_TOKEN not in res.reason_phrase

    def test_200_response_headers_never_contain_token(self, client, auth_headers) -> None:
        res = client.get("/api/reports", headers=auth_headers)
        assert res.status_code == 200
        assert TEST_DASHBOARD_TOKEN not in str(res.headers)


class TestWholeSurfaceGating:
    """No `/api/*` route may be born ungated -- enumerated, never hard-coded."""

    def test_every_api_route_rejects_an_unauthenticated_caller(self, client, dashboard_token) -> None:
        routes = _api_routes(client.app)
        # A bug that empties the route table must not make this test
        # vacuously pass.
        assert len(routes) > 0

        for route in routes:
            methods = getattr(route, "methods", None) or {"GET"}
            method = "GET" if "GET" in methods else next(iter(methods))
            path = route.path
            res = client.request(method, path)
            assert res.status_code == 401, f"{method} {path} did not reject an unauthenticated caller"


class TestShellReachabilityAndFallbackIsolation:
    def test_shell_is_reachable_with_no_token(self, tmp_path: Path) -> None:
        from llmsec.dashboard.app import create_app
        from fastapi.testclient import TestClient

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html>shell</html>")

        app = create_app(tmp_path / "reports", static_dir=static_dir)
        no_token_client = TestClient(app)

        res = no_token_client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]

    def test_unmatched_api_shaped_path_returns_json_never_the_shell(
        self, tmp_path: Path
    ) -> None:
        from llmsec.dashboard.app import create_app
        from fastapi.testclient import TestClient

        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html>shell</html>")

        app = create_app(tmp_path / "reports", static_dir=static_dir)
        no_token_client = TestClient(app)

        res = no_token_client.get("/api/reports/deadbeefdeadbeefdeadbeefdeadbeef")
        assert "application/json" in res.headers["content-type"]
        assert "<html" not in res.text.lower()


class TestNoCorsNoStaticExposure:
    def test_no_access_control_allow_origin_header(self, client, auth_headers) -> None:
        res = client.get(
            "/api/reports",
            headers={**auth_headers, "Origin": "https://evil.example.com"},
        )
        assert "access-control-allow-origin" not in res.headers

    def test_no_static_files_mount_ever_resolves_to_the_reports_output_dir(
        self, client, reports_dir: Path
    ) -> None:
        directories = _static_files_directories(client.app)
        assert reports_dir.resolve() not in directories

    def test_filesystem_shaped_paths_do_not_return_report_content(
        self, client, reports_dir: Path, dashboard_token
    ) -> None:
        # Find a real scan_id by asking the (authenticated) list endpoint,
        # then confirm the SAME file is unreachable by a filesystem-shaped
        # path with no token.
        auth_res = client.get(
            "/api/reports", headers={"Authorization": f"Bearer {TEST_DASHBOARD_TOKEN}"}
        )
        scan_id = auth_res.json()[0]["scan_id"]
        on_disk_content = (reports_dir / f"scan_{scan_id}.json").read_text()

        for path in (
            f"/llmsec_reports/scan_{scan_id}.json",
            f"/scan_{scan_id}.json",
        ):
            res = client.get(path)
            assert on_disk_content not in res.text
