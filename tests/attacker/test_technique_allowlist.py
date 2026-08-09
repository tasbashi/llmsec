"""05-08-PLAN.md Task 1: the technique allowlist gate at the delegation
boundary (D-95).

Two halves:

1. `validate_technique()` exercised directly as a pure function -- the
   closed-vocabulary-first, enabled-set-second ordering, and the exact
   source-assertion the plan's own acceptance criteria checks (exactly one
   definition + one call site in `graph.py`, T-05-08-07).
2. A fault-injected Strategist driven through the real
   `run_attacker_campaign()` entry point end to end, offline -- proving the
   gate's structural guarantee: zero mutation-role invocations on a
   refusal, zero dispatches to the fake adapter for that round, a
   constraint-violation entry (and audit line) recorded, and the campaign
   continuing to a normal `termination_reason` rather than aborting
   (05-AI-SPEC AT-2 rubric).
"""

from __future__ import annotations

import inspect
import json

import pytest

import llmsec.attacker.graph as graph_module
from llmsec.attacker.audit import audit_path_for
from llmsec.attacker.config import AttackerConfig
from llmsec.attacker.graph import TechniqueNotAllowed, validate_technique
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.runner import run_attacker_campaign
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.payloads.schema import PiiAttackVector, TechniqueFamily
from llmsec.plugins.base import BaseModule

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"

#: A technique name that is not a member of ANY closed enum in
#: `payloads/schema.py` -- the exact shape of a hallucinated/stale
#: Strategist selection this gate exists to refuse.
_HALLUCINATED_TECHNIQUE = "PAIR_REFINEMENT"


# --- validate_technique(): pure-function unit tests --------------------------


def test_validate_technique_returns_selected_when_in_both_closed_vocab_and_enabled():
    assert (
        validate_technique("instruction_override", ["instruction_override"])
        == "instruction_override"
    )


def test_validate_technique_accepts_every_closed_enum_member():
    """Both `TechniqueFamily` and `PiiAttackVector` members are accepted --
    the vocabulary is the union of both closed enums, never one alone."""
    for family in TechniqueFamily:
        assert validate_technique(family.value, [family.value]) == family.value
    for vector in PiiAttackVector:
        assert validate_technique(vector.value, [vector.value]) == vector.value


def test_validate_technique_refuses_when_not_in_enabled_set():
    with pytest.raises(TechniqueNotAllowed):
        validate_technique("instruction_override", ["persona_jailbreak"])


def test_validate_technique_refuses_hallucinated_family_even_when_present_in_enabled_list():
    """A typo/hallucinated family name present in a misconfigured
    `enabled_techniques` list must never widen the accepted vocabulary
    beyond the closed enums that are its source of record."""
    with pytest.raises(TechniqueNotAllowed):
        validate_technique(
            _HALLUCINATED_TECHNIQUE, [_HALLUCINATED_TECHNIQUE, "instruction_override"]
        )


def test_validate_technique_checks_closed_vocabulary_before_enabled_set():
    with pytest.raises(TechniqueNotAllowed, match="closed"):
        validate_technique("not_a_real_technique_at_all", ["not_a_real_technique_at_all"])


def test_validate_technique_is_the_one_call_site_in_graph_py():
    """The plan's own acceptance criteria source assertion: exactly one
    `def validate_technique(` and one call site (`strategist_node`'s own
    `try:` block) -- a second entry point into dispatch would defeat the
    whole gate, exactly as a second plugin-load path would defeat the
    registry's (T-05-08-07)."""
    source = inspect.getsource(graph_module)
    assert source.count("validate_technique(") == 2


# --- Integration: a fault-injected Strategist output is refused, not dispatched --


class _StubModule(BaseModule):
    id = "stub_module"
    name = "Stub Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    async def generate_cases(self, context: ScanContext):  # type: ignore[no-untyped-def]
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="stub parent payload", technique_id=_STATIC_TECHNIQUE_ID
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.95,
            evidence=f"complied: {response.raw_text}",
            detection_layer="regex",
        )


def _make_config(*, output_dir, max_rounds: int = 2, variants_per_round: int = 2) -> ScanConfig:
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        output_dir=str(output_dir),
        attacker=AttackerConfig(
            enabled=True,
            profile="light",
            max_rounds=max_rounds,
            variants_per_round=variants_per_round,
            # Only "instruction_override" is enabled -- the fault-injected
            # Strategist below selects something else entirely, so the gate
            # must refuse it regardless of round.
            enabled_techniques=["instruction_override"],
        ),
    )


def _static_results() -> list[tuple[str, EvalResult]]:
    return [
        (
            "stub_module",
            EvalResult(
                case_id=_STATIC_CASE_ID,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            ),
        )
    ]


def _disallowed_strategist_output(case_ids: list[str]) -> StrategistOutput:
    return StrategistOutput(
        technique=_HALLUCINATED_TECHNIQUE,
        ordered_case_ids=case_ids,
        escalate=False,
        reason_code=None,
        rationale="Attempt a technique that was never enabled for this campaign.",
    )


class _CountingAgent:
    """Wraps a real compiled role agent, counting `.ainvoke()` calls --
    proves the mutation ROLE ITSELF is never invoked on a refusal (not
    merely that its output is discarded downstream)."""

    def __init__(self, inner) -> None:  # type: ignore[no-untyped-def]
        self._inner = inner
        self.call_count = 0

    async def ainvoke(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.call_count += 1
        return await self._inner.ainvoke(*args, **kwargs)


async def test_disallowed_technique_refused_zero_mutator_invocations_zero_dispatches(
    monkeypatch, fake_target_adapter, scripted_chat_model, patch_analyst_and_recon_roles, tmp_path
):
    module = _StubModule()
    modules = {"stub_module": module}
    adapter = fake_target_adapter()

    strategist_model = scripted_chat_model([_disallowed_strategist_output([_STATIC_CASE_ID])])
    monkeypatch.setattr(
        ROLE_REGISTRY["strategist"],
        "build",
        lambda settings, cfg, *, model=None: build_strategist_agent(
            settings, cfg, model=strategist_model
        ),
    )

    mutator_calls: list[_CountingAgent] = []

    def _counting_mutator_build(settings, cfg, *, model=None):  # type: ignore[no-untyped-def]
        real_agent = build_mutator_agent(settings, cfg, model=scripted_chat_model([]))
        counter = _CountingAgent(real_agent)
        mutator_calls.append(counter)
        return counter

    monkeypatch.setattr(ROLE_REGISTRY["mutator"], "build", _counting_mutator_build)
    patch_analyst_and_recon_roles()

    config = _make_config(output_dir=tmp_path, max_rounds=1, variants_per_round=1)
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-allowlist-1",
    )

    # The Mutator role was constructed (unconditionally, alongside
    # Crescendo) but NEVER invoked -- the gate refuses before either
    # mutation role's `.ainvoke()` is ever called.
    assert len(mutator_calls) == 1
    assert mutator_calls[0].call_count == 0

    # The fake adapter received zero MUTATION-VARIANT dispatches for the
    # refused round -- Recon's own probe-tool calls (run once per campaign,
    # unrelated to this round's gate) are the only sends on the adapter.
    mutation_dispatches = [c for c in adapter.sent_cases if "-mut-" in c.case_id]
    assert mutation_dispatches == []
    assert adapter.conversation_cases == []

    # The refusal is recorded as a constraint violation, exposed on the
    # campaign result (T-05-08-06).
    assert result.constraint_violations == 1
    violations = result.final_state.get("constraint_violations", [])
    assert len(violations) == 1
    assert violations[0]["technique"] == _HALLUCINATED_TECHNIQUE
    assert violations[0]["enabled_techniques"] == ["instruction_override"]


async def test_campaign_continues_after_refusal_and_terminates_normally(
    monkeypatch, fake_target_adapter, scripted_chat_model, patch_analyst_and_recon_roles, tmp_path
):
    """T-05-08-05: a refusal consumes its round -- the campaign proceeds to
    its next round or its reason-code exit, never aborting outright."""
    module = _StubModule()
    modules = {"stub_module": module}
    adapter = fake_target_adapter()

    strategist_model = scripted_chat_model(
        [_disallowed_strategist_output([_STATIC_CASE_ID]) for _ in range(2)]
    )
    monkeypatch.setattr(
        ROLE_REGISTRY["strategist"],
        "build",
        lambda settings, cfg, *, model=None: build_strategist_agent(
            settings, cfg, model=strategist_model
        ),
    )
    monkeypatch.setattr(
        ROLE_REGISTRY["mutator"],
        "build",
        lambda settings, cfg, *, model=None: build_mutator_agent(
            settings, cfg, model=scripted_chat_model([])
        ),
    )
    patch_analyst_and_recon_roles()

    config = _make_config(output_dir=tmp_path, max_rounds=2, variants_per_round=1)
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-allowlist-2",
    )

    # Two refused rounds -- a Strategist stuck emitting a disallowed
    # technique still terminates at the round cap rather than looping.
    assert result.constraint_violations == 2
    assert result.final_state.get("termination_reason") == "ROUND_CAP_REACHED"


async def test_constraint_violation_audit_line_written_for_each_refusal(
    monkeypatch, fake_target_adapter, scripted_chat_model, patch_analyst_and_recon_roles, tmp_path
):
    module = _StubModule()
    modules = {"stub_module": module}
    adapter = fake_target_adapter()

    strategist_model = scripted_chat_model([_disallowed_strategist_output([_STATIC_CASE_ID])])
    monkeypatch.setattr(
        ROLE_REGISTRY["strategist"],
        "build",
        lambda settings, cfg, *, model=None: build_strategist_agent(
            settings, cfg, model=strategist_model
        ),
    )
    monkeypatch.setattr(
        ROLE_REGISTRY["mutator"],
        "build",
        lambda settings, cfg, *, model=None: build_mutator_agent(
            settings, cfg, model=scripted_chat_model([])
        ),
    )
    patch_analyst_and_recon_roles()

    config = _make_config(output_dir=tmp_path, max_rounds=1, variants_per_round=1)
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-allowlist-3",
    )

    assert result.audit_path is not None
    lines = [
        json.loads(raw_line)
        for raw_line in audit_path_for(tmp_path, "scan-allowlist-3").read_text().splitlines()
        if raw_line
    ]
    refusal_lines = [
        line
        for line in lines
        if line["event"] == "inter_agent_handoff"
        and line["agent"] == "strategist"
        and "REFUSED" in line["content"]
    ]
    assert len(refusal_lines) == 1
    assert _HALLUCINATED_TECHNIQUE in refusal_lines[0]["content"]
    assert refusal_lines[0]["recipient"] == "mutator"


async def test_technique_outside_every_closed_enum_refused_even_when_in_configured_enabled_list(
    monkeypatch, fake_target_adapter, scripted_chat_model, patch_analyst_and_recon_roles, tmp_path
):
    """A misconfigured `enabled_techniques` list containing a hallucinated
    family name can never widen the accepted vocabulary -- the closed
    enums in `payloads/schema.py` are the vocabulary of record."""
    module = _StubModule()
    modules = {"stub_module": module}
    adapter = fake_target_adapter()

    strategist_model = scripted_chat_model([_disallowed_strategist_output([_STATIC_CASE_ID])])
    monkeypatch.setattr(
        ROLE_REGISTRY["strategist"],
        "build",
        lambda settings, cfg, *, model=None: build_strategist_agent(
            settings, cfg, model=strategist_model
        ),
    )
    monkeypatch.setattr(
        ROLE_REGISTRY["mutator"],
        "build",
        lambda settings, cfg, *, model=None: build_mutator_agent(
            settings, cfg, model=scripted_chat_model([])
        ),
    )
    patch_analyst_and_recon_roles()

    config = ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        output_dir=str(tmp_path),
        attacker=AttackerConfig(
            enabled=True,
            profile="light",
            max_rounds=1,
            variants_per_round=1,
            # The hallucinated technique IS present here -- a misconfigured
            # allowlist -- but it is still refused because it is not a
            # member of the closed TechniqueFamily/PiiAttackVector enums.
            enabled_techniques=[_HALLUCINATED_TECHNIQUE],
        ),
    )
    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules=modules,
        static_results=_static_results(),
        scan_id="scan-allowlist-4",
    )

    assert result.constraint_violations == 1
    mutation_dispatches = [c for c in adapter.sent_cases if "-mut-" in c.case_id]
    assert mutation_dispatches == []
