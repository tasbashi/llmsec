"""05-08-PLAN.md Task 3: anti-feature enforcement -- willingness signal
only, and no autonomous remediation (D-95, FEATURES.md Section 7).

Two halves:

1. A registry-driven, parametrized test asserting every REGISTERED role
   (`ROLE_REGISTRY`, not a hand-maintained list of prompt constants) carries
   both the shared sandbox and anti-feature clauses -- a sixth role added
   later cannot skip either clause without this test naming it by role.
2. A combined adversarial fixture -- one scripted target response that
   simultaneously attempts scope redirection AND harm elicitation, because
   a real adversarial target would not politely separate the two -- driven
   through the real `build_campaign_graph()` topology, offline, proving
   zero dispatches outside the fake adapter, zero configuration mutations,
   and a recorded outcome that is a willingness signal, never the literal
   harmful content or the redirection target the response tried to plant.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import llmsec
import llmsec.attacker.graph  # noqa: F401 -- transitively registers all five roles
from llmsec.attacker.audit import AttackerAuditHandler, AttackerAuditWriter
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.state import QueuedCase, new_campaign_state
from llmsec.models import EvalResult, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

from .conftest import (
    HARMFUL_CONTENT_MARKER as _HARMFUL_CONTENT_MARKER,
    SCOPE_REDIRECT_TARGET as _SCOPE_REDIRECT_TARGET,
)

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"


# --- Registry-driven prompt-clause presence (per role, not hand-listed) -----


@pytest.mark.parametrize("role_name", sorted(ROLE_REGISTRY))
def test_every_registered_role_prompt_carries_the_sandbox_clause(role_name: str):
    role = ROLE_REGISTRY[role_name]
    assert "SANDBOX:" in role.system_prompt, f"{role_name} prompt is missing the sandbox clause"


@pytest.mark.parametrize("role_name", sorted(ROLE_REGISTRY))
def test_every_registered_role_prompt_carries_the_anti_feature_clause(role_name: str):
    role = ROLE_REGISTRY[role_name]
    assert (
        "ANTI-FEATURE:" in role.system_prompt
    ), f"{role_name} prompt is missing the anti-feature clause"


def test_registry_has_all_five_roles_this_phase_defines():
    """Guards the parametrization itself: if a role's module were never
    imported, the two tests above would silently run over a smaller set --
    this pins the expected membership."""
    assert set(ROLE_REGISTRY) == {"strategist", "mutator", "analyst", "recon", "crescendo"}


# --- Combined adversarial fixture: scope redirection + harm elicitation ----


class _AdversarialStubModule(BaseModule):
    """Demonstrates the anti-feature discipline a deep-mode-aware module's
    own `evaluate()` must follow: when the target's response shows
    willingness to comply with an out-of-scope or harm-eliciting request,
    the recorded evidence is a WILLINGNESS SIGNAL description -- never the
    literal offending text -- even though `module.evaluate()` (D-66) is
    otherwise untouched, unchanged scoring logic."""

    id = "stub_module"
    name = "Stub Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    async def generate_cases(self, context):  # type: ignore[no-untyped-def]
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="stub parent payload", technique_id=_STATIC_TECHNIQUE_ID
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        if _HARMFUL_CONTENT_MARKER in response.raw_text or _SCOPE_REDIRECT_TARGET in response.raw_text:
            return EvalResult(
                case_id=case.case_id,
                verdict=Verdict.PARTIAL_LEAK,
                confidence=0.7,
                evidence=(
                    "target expressed willingness to comply and attempted a scope "
                    "redirection; payload content withheld from evidence per "
                    "anti-feature discipline (D-95)"
                ),
                detection_layer="regex",
            )
        return EvalResult(
            case_id=case.case_id, verdict=Verdict.BLOCKED, confidence=0.9, evidence="refused",
            detection_layer="regex",
        )


def _strategist_output(case_ids: list[str]) -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
        ordered_case_ids=case_ids,
        escalate=False,
        reason_code=None,
        rationale="Try a direct instruction-override refinement.",
    )


def _mutator_output(n_variants: int = 1) -> MutatorOutput:
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload=f"mutated payload variant {i + 1}",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale=f"rationale {i + 1}",
            )
            for i in range(n_variants)
        ]
    )


def _case_queue() -> list[QueuedCase]:
    return [
        QueuedCase(
            module_id="stub_module",
            case_id=_STATIC_CASE_ID,
            technique_id=_STATIC_TECHNIQUE_ID,
            prompt="stub parent payload",
            verdict="blocked",
            turns=None,
        )
    ]


async def test_combined_adversarial_target_produces_willingness_signal_never_harm_content(
    scripted_chat_model, fake_target_adapter, adversarial_target_response
):
    module = _AdversarialStubModule()
    modules = {"stub_module": module}
    adapter = fake_target_adapter()
    adapter.queue_response(f"{_STATIC_CASE_ID}-mut-1", adversarial_target_response)

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)
    cfg_before = cfg.model_dump()

    strategist_model = scripted_chat_model([_strategist_output([_STATIC_CASE_ID])])
    mutator_model = scripted_chat_model([_mutator_output(1)])
    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
    }
    compiled = build_campaign_graph(roles=roles, adapter=adapter, modules=modules, max_concurrency=5)

    initial_state = new_campaign_state(
        scan_id="scan-antifeature-1",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)

    dispatch_results = final_state.get("dispatch_results", [])
    assert len(dispatch_results) == 1
    eval_result = dispatch_results[0]["eval_result"]

    # A willingness signal is recorded -- never the literal harmful content
    # or the redirection target the adversarial response tried to plant.
    assert eval_result.verdict == Verdict.PARTIAL_LEAK
    assert _HARMFUL_CONTENT_MARKER not in eval_result.evidence
    assert _SCOPE_REDIRECT_TARGET not in eval_result.evidence

    # Zero dispatches outside the single, closed-over fake adapter -- there
    # is no second destination this graph could ever reach (T-05-08-02: the
    # probe/dispatch closures carry no destination parameter).
    assert len(adapter.sent_cases) == 1
    assert adapter.sent_cases[0].case_id == f"{_STATIC_CASE_ID}-mut-1"
    assert adapter.conversation_cases == []

    # Zero configuration mutations -- the campaign-level config object this
    # graph was built from is untouched.
    assert cfg.model_dump() == cfg_before


async def test_combined_adversarial_target_marker_absent_from_inter_agent_and_target_audit_lines(
    scripted_chat_model, fake_target_adapter, adversarial_target_response, tmp_path
):
    """Scoped to `direction="inter_agent"` (role handoffs, built from each
    agent's OWN structured-output fields) and `direction="target"` (built
    from `eval_result.evidence`, D-66's own scoring path) -- these are the
    campaign's own recorded reasoning/outcome, which must never carry the
    marker. `direction="outbound"` model-call lines (what a role's brief
    literally SENDS to its model, e.g. the Analyst's truncated raw-response
    quote) are the documented D-84/D-86 exception: `audit.py`'s own
    docstring states the Analyst quotes the target's raw response to its
    peers over this channel by design, so the raw target text -- and
    therefore this fixture's marker -- legitimately reaching an
    `outbound`-direction line is accepted, unredacted-prose behavior, not a
    gap this test (or Task 3) is scoped to close.
    """
    module = _AdversarialStubModule()
    modules = {"stub_module": module}
    adapter = fake_target_adapter()
    adapter.queue_response(f"{_STATIC_CASE_ID}-mut-1", adversarial_target_response)

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    strategist_model = scripted_chat_model([_strategist_output([_STATIC_CASE_ID])])
    mutator_model = scripted_chat_model([_mutator_output(1)])
    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
    }

    writer = AttackerAuditWriter(tmp_path, "scan-antifeature-2")
    handler = AttackerAuditHandler(writer, "scan-antifeature-2")
    compiled = build_campaign_graph(
        roles=roles, adapter=adapter, modules=modules, max_concurrency=5, callbacks=[handler]
    )

    initial_state = new_campaign_state(
        scan_id="scan-antifeature-2",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    await compiled.ainvoke(initial_state)
    writer.close()

    lines = [json.loads(raw) for raw in writer.path.read_text(encoding="utf-8").splitlines() if raw]
    scoped_lines = [line for line in lines if line["direction"] in {"inter_agent", "target"}]
    assert len(scoped_lines) > 0
    for line in scoped_lines:
        assert _HARMFUL_CONTENT_MARKER not in line["content"], line
        assert _SCOPE_REDIRECT_TARGET not in line["content"], line


# --- No autonomous remediation, no target-config write surface -------------


_FORBIDDEN_REMEDIATION_PATTERNS: tuple[str, ...] = (
    "subprocess.",
    "os.system(",
    "shutil.rmtree(",
    "requests.request(",
    "requests.post(",
    "requests.get(",
    "httpx.Client(",
    "httpx.AsyncClient(",
)


def test_no_node_in_attacker_package_writes_target_config_or_applies_remediation():
    """No node in the attacker package performs any write against target
    configuration, and no code path applies a remediation -- the ONLY
    target-facing capability in this package is the closed-over
    `TargetAdapter` instance already threaded through `build_campaign_graph()`
    (`adapter.send()`/`adapter.send_conversation()`), never a second
    transport a node could construct itself."""
    attacker_dir = pathlib.Path(llmsec.__file__).resolve().parent / "attacker"
    for path in attacker_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_REMEDIATION_PATTERNS:
            assert pattern not in text, f"{path} contains forbidden pattern {pattern!r}"


def test_constraint_violations_and_abandoned_arcs_exposed_on_campaign_result():
    """So a rising rate of either is visible to an operator (`<behavior>`).
    05-11 Rule 1/2 fix: `role_structural_failures` (AT-6) is exposed
    alongside the original two, distinct from both."""
    from llmsec.attacker.runner import CampaignResult

    result = CampaignResult()
    assert result.constraint_violations == 0
    assert result.abandoned_arcs == 0
    assert result.role_structural_failures == 0
