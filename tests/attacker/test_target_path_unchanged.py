"""D-94 gate 4 / AT-9: target-path non-regression (05-10-PLAN.md Task 1).

`src/llmsec/orchestrator.py` and `src/llmsec/adapters/llm_api.py` are the two
target-path modules D-93 requires stay byte-for-byte unchanged for the whole
of Phase 5 -- the `--quick` static scan path is provably unaffected by the
attacker-team layer's existence, extra-installed or not.

Three independent checks, each catching a different way this guarantee could
silently break:
  1. An import-statement scan (AST-based, not a substring grep) proving
     neither module imports any of the three attacker-stack distributions,
     at module scope or inside a function.
  2. A content-digest pin against each module's own SHA-256, captured from
     its state at the start of this phase (confirmed via `git log` to be
     unchanged since Phase 2 -- neither file has been touched since
     `bc1cb80`, well before Phase 5 began).
  3. A report-equality test: two static-only `run_scan()` calls produce a
     byte-identical report (every field but `scan_id`/timestamps) even when
     the second run has `langchain`/`langgraph`/`deepagents` forced
     unimportable via `sys.modules` poisoning -- simulating the extra being
     entirely absent, without needing to actually uninstall anything.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from typing import AsyncIterator

import pytest

import llmsec.api as api_module
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORCHESTRATOR_PATH = _REPO_ROOT / "src" / "llmsec" / "orchestrator.py"
_LLM_API_ADAPTER_PATH = _REPO_ROOT / "src" / "llmsec" / "adapters" / "llm_api.py"

#: The three attacker-stack distributions declared in the `[deep]` optional
#: extra (`pyproject.toml`) -- neither target-path module may import any of
#: these, at module scope or inside a function (D-73 mitigation 3).
_ATTACKER_STACK_MODULE_ROOTS = frozenset({"langchain", "langgraph", "deepagents"})

#: Content digests captured from each target-path module's state at the
#: start of this phase (05-10-PLAN.md Task 1's own `<action>` requirement).
#: `git log --oneline -- <path>` confirms neither file has been touched
#: since commit `bc1cb80` (Phase 2, "dispatch multi-turn cases through
#: send_conversation with duck-typed abort hook") -- well before Phase 5
#: (`05-01-PLAN.md`) began, so these digests are stable across the whole of
#: D-93's "orchestrator.py stays byte-for-byte unchanged" guarantee, not
#: just this one plan.
_EXPECTED_DIGESTS: dict[Path, str] = {
    _ORCHESTRATOR_PATH: "4927a4022a6c670f54a687cc930feafa518591aba5369ade08fc47068d5528f6",
    _LLM_API_ADAPTER_PATH: "0627ff6e13754ce7273eb41cd8034ec1d23c46607542b2508aae59251de1ed03",
}


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _imported_module_roots(source: str) -> set[str]:
    """AST-walk `source`, returning the top-level root name of every module
    named in an `import`/`from ... import` statement anywhere in the file --
    at module scope AND nested inside any function/class body, since
    `ast.walk()` visits every node in the tree regardless of nesting depth.
    Only the first dotted-path segment is kept (`"langchain.chat_models"` ->
    `"langchain"`) so a match against `_ATTACKER_STACK_MODULE_ROOTS` catches
    every submodule import, not just a bare top-level one."""
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


# --- Check 1: import-statement scan -----------------------------------------


@pytest.mark.parametrize("path", [_ORCHESTRATOR_PATH, _LLM_API_ADAPTER_PATH])
def test_target_path_module_never_imports_attacker_stack(path: Path):
    source = path.read_text(encoding="utf-8")
    imported_roots = _imported_module_roots(source)
    overlap = imported_roots & _ATTACKER_STACK_MODULE_ROOTS
    assert not overlap, (
        f"{path} imports attacker-stack module(s) {overlap} -- the target adapter "
        "path must stay on litellm only (D-73 mitigation 3)"
    )


def test_grep_negative_control_style_check_matches_ast_result():
    """A plain substring check (mirroring the plan's own acceptance-criteria
    grep) as an independent cross-check against the AST-based scan above --
    two different mechanisms agreeing is stronger evidence than either
    alone."""
    for path in (_ORCHESTRATOR_PATH, _LLM_API_ADAPTER_PATH):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import langchain") or stripped.startswith("import langgraph"):
                pytest.fail(f"{path} contains a bare attacker-stack import: {stripped!r}")
            if stripped.startswith("from langchain") or stripped.startswith("from langgraph"):
                pytest.fail(f"{path} contains a bare attacker-stack import: {stripped!r}")
            if stripped.startswith("import deepagents") or stripped.startswith("from deepagents"):
                pytest.fail(f"{path} contains a bare attacker-stack import: {stripped!r}")


# --- Check 2: content-digest pin --------------------------------------------


@pytest.mark.parametrize("path", [_ORCHESTRATOR_PATH, _LLM_API_ADAPTER_PATH])
def test_target_path_module_content_digest_unchanged(path: Path):
    actual = _sha256_of(path)
    expected = _EXPECTED_DIGESTS[path]
    assert actual == expected, (
        f"{path} content digest changed ({actual}) from its recorded phase-start "
        f"value ({expected}) -- D-93 requires this module stay byte-for-byte "
        "unchanged; the fix belongs in the attacker layer, never here"
    )


# --- Check 3: report-equality with the extra simulated absent --------------


class _StaticOnlyModule(BaseModule):
    """A deterministic, non-attacker module -- `uses_attacker_llm` left at
    its `BaseModule` default (`False`) -- so this test exercises exactly the
    `--quick` code path and never touches `run_attacker_campaign()` at all."""

    id = "static_only_module"
    name = "Static Only Module"
    owasp_ref = "LLM00:2025"

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        yield TestCase(case_id="STATIC-001", prompt="static probe", technique_id="STATIC-001")

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.BLOCKED,
            confidence=0.9,
            evidence="refused",
            detection_layer="regex",
        )


class _StaticOnlyAdapter:
    supports_system_prompt_override = False
    supports_multi_turn = False

    async def send(self, case: TestCase) -> TargetResponse:
        return TargetResponse(case_id=case.case_id, raw_text="a static reply", latency_ms=1.0)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def _patch_static_only(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _StaticOnlyModule()
    monkeypatch.setattr(
        api_module.PluginRegistry,
        "load_allowed",
        lambda self, allowlist, module_config=None: {module.id: module},
    )
    monkeypatch.setattr(api_module, "HttpAppAdapter", lambda *a, **kw: _StaticOnlyAdapter())


def _static_config(tmp_path) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(
            type="http_app",
            method="POST",
            url="http://localhost:8000/chat",
            headers={},
            body_template='{"message": "{{payload}}"}',
            response_path="response",
        ),
        enabled_modules=["static_only_module"],
        max_concurrency=5,
        output_dir=str(tmp_path),
        attacker=None,
    )


async def test_static_scan_report_identical_with_extra_simulated_absent(tmp_path, monkeypatch):
    _patch_static_only(monkeypatch)
    config_a = _static_config(tmp_path / "run-a")
    report_a = await api_module.run_scan(config_a, bypass_flag=True)

    # Simulate the `[deep]` extra being entirely absent for the DURATION of
    # this second run -- `sys.modules[name] = None` forces `ImportError` on
    # any bare `import <name>`/`from <name> import ...`, the standard
    # mechanism for this, without needing to actually uninstall anything.
    for name in _ATTACKER_STACK_MODULE_ROOTS:
        monkeypatch.setitem(sys.modules, name, None)

    config_b = _static_config(tmp_path / "run-b")
    report_b = await api_module.run_scan(config_b, bypass_flag=True)

    excluded = {"scan_id", "started_at", "completed_at"}
    dump_a = report_a.model_dump(exclude=excluded)
    dump_b = report_b.model_dump(exclude=excluded)
    assert dump_a == dump_b, (
        "a static-only scan's report must be identical (bar scan_id/timestamps) "
        "whether or not the [deep] extra is importable"
    )
