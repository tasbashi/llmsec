"""D-94 gate 3 / AT-4: sandbox integrity (05-10-PLAN.md Task 1).

No role ever receives the target's privileged context (system prompt,
credentials, `ScanConfig`), across four surfaces named by D-87/AT-4: role
prompt renders, tool-call arguments (dispatched-variant payloads and lineage
records), campaign-memory entries, and the persisted checkpoint artifact.

A deliberately-leaky control campaign proves the absence assertion is
meaningful: with the Strategist's own `.brief()` monkeypatched to actually
append the canary literal to its output -- simulating exactly the bug this
gate exists to catch -- the SAME capture/assertion machinery used by the
real test DOES find it. Without this control, a gate that scanned the wrong
surface (or never actually invoked the recording model) would pass
identically whether or not the sandbox held.
"""

from __future__ import annotations

import inspect
from typing import AsyncIterator

from llmsec.attacker.config import AttackerConfig
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import (
    StrategistOutput,
    _strategist_brief,
    build_strategist_agent,
)
from llmsec.attacker.runner import run_attacker_campaign
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

from .conftest import CANARY_PRIVILEGED_CONTEXT_LITERAL, flatten_message_batches

_STATIC_CASE_ID = "SANDBOX-001"


class _SandboxModule(BaseModule):
    """A minimal `uses_attacker_llm=True` module -- one static `BLOCKED`
    case, mutated variants score `FULL_COMPROMISE`, mirroring
    `test_tracer_round.py`'s `_StubModule` shape exactly."""

    id = "sandbox_module"
    name = "Sandbox Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    async def generate_cases(self, context: ScanContext) -> AsyncIterator[TestCase]:
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="sandbox parent payload", technique_id=_STATIC_CASE_ID
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        if case.case_id == _STATIC_CASE_ID:
            return EvalResult(
                case_id=case.case_id,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            )
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.9,
            evidence=f"complied: {response.raw_text}",
            detection_layer="regex",
        )


def _make_config(output_dir, *, checkpoint_dir=None) -> ScanConfig:
    """A `ScanConfig` carrying the D-87 canary in the ONE field that models
    a target's privileged system prompt (`known_system_prompt`,
    `ScanContext`'s own docstring) -- the surface `run_attacker_campaign()`
    must never let cross into a role's brief."""
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        output_dir=str(output_dir),
        known_system_prompt=CANARY_PRIVILEGED_CONTEXT_LITERAL,
        attacker=AttackerConfig(
            enabled=True,
            profile="light",
            variants_per_round=1,
            max_rounds=1,
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        ),
    )


def _static_results() -> list[tuple[str, EvalResult]]:
    return [
        (
            "sandbox_module",
            EvalResult(
                case_id=_STATIC_CASE_ID,
                verdict=Verdict.BLOCKED,
                confidence=0.9,
                evidence="refused",
                detection_layer="regex",
            ),
        )
    ]


def _strategist_output() -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
        ordered_case_ids=[_STATIC_CASE_ID],
        escalate=False,
        reason_code=None,
        rationale="Try a direct instruction-override refinement.",
    )


def _mutator_output() -> MutatorOutput:
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload="mutated payload, no privileged content in it",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_CASE_ID,
                rationale="rationale",
            )
        ]
    )


def _patch_roles(monkeypatch, scripted_chat_model, *, strategist_brief_override=None):
    strategist_model = scripted_chat_model([_strategist_output()])
    mutator_model = scripted_chat_model([_mutator_output()])

    monkeypatch.setattr(
        ROLE_REGISTRY["strategist"],
        "build",
        lambda settings, cfg, *, model=None: build_strategist_agent(
            settings, cfg, model=strategist_model
        ),
    )
    if strategist_brief_override is not None:
        monkeypatch.setattr(ROLE_REGISTRY["strategist"], "brief", strategist_brief_override)
    monkeypatch.setattr(
        ROLE_REGISTRY["mutator"],
        "build",
        lambda settings, cfg, *, model=None: build_mutator_agent(settings, cfg, model=mutator_model),
    )
    return strategist_model, mutator_model


async def test_sandbox_canary_absent_across_all_four_surfaces(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    module = _SandboxModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1",
        mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="a normal reply"),
    )

    strategist_model, mutator_model = _patch_roles(monkeypatch, scripted_chat_model)
    patch_analyst_and_recon_roles()

    checkpoint_dir = tmp_path / "checkpoints"
    config = _make_config(tmp_path / "reports", checkpoint_dir=checkpoint_dir)

    result = await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules={"sandbox_module": module},
        static_results=_static_results(),
        scan_id="scan-sandbox-1",
    )

    # Surface 1: role prompt renders -- every message batch every scripted
    # model actually received (i.e. what the role's compiled agent sent to
    # its underlying chat model).
    prompt_text = flatten_message_batches(
        strategist_model.received_message_batches
    ) + flatten_message_batches(mutator_model.received_message_batches)
    assert CANARY_PRIVILEGED_CONTEXT_LITERAL not in prompt_text

    # Surface 2: tool-call arguments -- every dispatched variant's payload
    # and every recorded D-90 lineage field.
    assert result.lineage, "expected at least one dispatched variant to inspect"
    for record in result.lineage.values():
        assert CANARY_PRIVILEGED_CONTEXT_LITERAL not in record["payload"]
        assert CANARY_PRIVILEGED_CONTEXT_LITERAL not in record["parent_case_id"]

    # Surface 3: campaign-memory entries.
    memory = result.final_state.get("bounded_memory") or {}
    assert CANARY_PRIVILEGED_CONTEXT_LITERAL not in str(memory)

    # Surface 4: the persisted checkpoint artifact's raw bytes on disk.
    db_path = checkpoint_dir / "attacker-checkpoints.sqlite"
    assert db_path.exists(), "expected a disk-backed checkpoint file to have been written"
    raw_bytes = db_path.read_bytes()
    assert CANARY_PRIVILEGED_CONTEXT_LITERAL.encode("utf-8") not in raw_bytes


async def test_sandbox_leaky_control_finds_the_canary(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """The deliberately-leaky control (D-94 gate 3's own requirement): the
    Strategist's real `.brief()` is wrapped to actually append the canary,
    simulating the exact bug this gate exists to catch. The SAME Surface-1
    capture/assertion used by the real test above must now find it -- proof
    the absence assertion is not passing merely because nothing was ever
    actually scanned."""

    def _leaky_brief(state):
        return _strategist_brief(state) + f"\nLEAKED PRIVILEGED CONTEXT: {CANARY_PRIVILEGED_CONTEXT_LITERAL}"

    module = _SandboxModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1",
        mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="a normal reply"),
    )

    strategist_model, _mutator_model = _patch_roles(
        monkeypatch, scripted_chat_model, strategist_brief_override=_leaky_brief
    )
    patch_analyst_and_recon_roles()

    config = _make_config(tmp_path / "reports")
    await run_attacker_campaign(
        config=config,
        adapter=adapter,
        modules={"sandbox_module": module},
        static_results=_static_results(),
        scan_id="scan-sandbox-leaky",
    )

    prompt_text = flatten_message_batches(strategist_model.received_message_batches)
    assert CANARY_PRIVILEGED_CONTEXT_LITERAL in prompt_text, (
        "the leaky control must find the canary -- if it does not, Surface 1's "
        "capture mechanism is not actually observing what it claims to"
    )


# --- Structural half: no role factory's own call signature can accept -----
# --- privileged context in the first place ---------------------------------

_FORBIDDEN_BUILD_PARAM_NAMES = frozenset(
    {"config", "scan_config", "known_system_prompt", "system_prompt", "api_key", "credential", "credentials"}
)


def test_no_role_factory_signature_accepts_privileged_context():
    """Every registered role's `.build()` call signature is inspected --
    never hand-listed -- so a sixth role added later cannot silently widen
    the accepted parameter surface to include the raw `ScanConfig`, the
    known system prompt, or a resolved credential value (D-87, made
    testable rather than merely asserted in a prompt)."""
    assert ROLE_REGISTRY, "expected at least one registered attacker role"
    for name, role in ROLE_REGISTRY.items():
        sig = inspect.signature(role.build)
        param_names = set(sig.parameters) - {"self"}
        forbidden_present = param_names & _FORBIDDEN_BUILD_PARAM_NAMES
        assert not forbidden_present, (
            f"role {name!r}'s build() accepts forbidden param(s): {forbidden_present}"
        )
        for param in sig.parameters.values():
            assert param.annotation is not ScanConfig, (
                f"role {name!r}'s build() param {param.name!r} is typed as the raw ScanConfig"
            )
