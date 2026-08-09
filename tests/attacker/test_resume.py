"""05-06-PLAN.md Task 3: `--resume` under the original cap, with fingerprint
refusal and bounded-loss disclosure (D-75, all four numbered sub-decisions).

Drives real checkpoint write-then-restore cycles against a `tmp_path`
checkpoint directory (never a mocked checkpointer), reusing the scripted
role / fake adapter fixtures `test_audit_wiring.py`/
`test_budget_stop_semantics.py`/`test_checkpoint_redaction.py` already
established, plus `ROLE_REGISTRY`-level monkeypatching for the
`run_attacker_campaign()` calls that seed each test's starting checkpoint.

`resume_attacker_campaign()` is called DIRECTLY (not through the CLI) for
every mechanics test below -- CLI-level tests are limited to the flag
validation and `--help` surface, which is genuinely CLI-only behavior;
duplicating every mechanics scenario through `CliRunner` would just add a
second, slower harness around the SAME assertions.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llmsec.attacker.checkpoint import idempotency_key
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.runner import (
    ConfigFingerprintMismatchError,
    UnknownScanIdError,
    resume_attacker_campaign,
    run_attacker_campaign,
)
from llmsec.attacker.state import CampaignState, VariantRecord
from llmsec.cli import app
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

runner = CliRunner()

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"


class _StubModule(BaseModule):
    """One static case; every mutated variant scores `FULL_COMPROMISE`."""

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


def _make_config(*, output_dir, checkpoint_dir, **attacker_overrides) -> ScanConfig:
    defaults: dict[str, object] = dict(
        enabled=True,
        profile="light",
        max_rounds=2,
        variants_per_round=1,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
    )
    defaults.update(attacker_overrides)
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        output_dir=str(output_dir),
        attacker=AttackerConfig(**defaults),
    )


def _static_results() -> list[tuple[str, EvalResult]]:
    return [
        (
            "stub_module",
            EvalResult(
                case_id=_STATIC_CASE_ID, verdict=Verdict.BLOCKED, confidence=0.9, evidence="refused",
                detection_layer="regex",
            ),
        )
    ]


def _strategist_output(rationale: str = "r") -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
        ordered_case_ids=[_STATIC_CASE_ID],
        escalate=False,
        reason_code=None,
        rationale=rationale,
    )


def _mutator_output(rationale: str = "r") -> MutatorOutput:
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload=f"mutated payload {rationale}",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale=rationale,
            )
        ]
    )


def _patch_roles(monkeypatch, scripted_chat_model, *, strategist_outputs, mutator_outputs):
    strategist_model = scripted_chat_model(strategist_outputs)
    mutator_model = scripted_chat_model(mutator_outputs)
    monkeypatch.setattr(
        ROLE_REGISTRY["strategist"],
        "build",
        lambda settings, cfg, *, model=None: build_strategist_agent(settings, cfg, model=strategist_model),
    )
    monkeypatch.setattr(
        ROLE_REGISTRY["mutator"],
        "build",
        lambda settings, cfg, *, model=None: build_mutator_agent(settings, cfg, model=mutator_model),
    )
    return strategist_model, mutator_model


async def _seed_checkpoint_via_mid_campaign_exception(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    config,
    scan_id,
):
    """Runs `run_attacker_campaign()` scripted for exactly ONE round's worth
    of responses against a `max_rounds=2` config -- round 1 completes
    normally (and is genuinely checkpointed to disk), round 2's Strategist
    call then exhausts the scripted model and raises, simulating a
    real process crash mid-campaign with a real, disk-persisted partial
    checkpoint left behind (proven end-to-end during this plan's
    implementation -- see `05-06-SUMMARY.md`)."""
    module = _StubModule()
    modules = {"stub_module": module}
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure")
    )
    _patch_roles(
        monkeypatch, scripted_chat_model,
        strategist_outputs=[_strategist_output("round1")],
        mutator_outputs=[_mutator_output("round1")],
    )
    patch_analyst_and_recon_roles()
    with pytest.raises(AssertionError, match="exhausted its 1-entry script"):
        await run_attacker_campaign(
            config=config, adapter=adapter, modules=modules, static_results=_static_results(), scan_id=scan_id
        )
    return adapter


# --- CLI surface -------------------------------------------------------------


def test_scan_help_lists_resume_flags():
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "--resume" in result.output
    assert "--budget-top-up-usd" in result.output


def test_resume_without_deep_exits_nonzero(tmp_path):
    yaml_path = tmp_path / "llmsec.config.yaml"
    yaml_path.write_text(
        "target:\n  type: http_app\n  method: POST\n  url: http://localhost:8000/chat\n"
        '  body_template: \'{"message": "{{payload}}"}\'\n'
    )
    result = runner.invoke(
        app, ["scan", "--config", str(yaml_path), "--resume", "some-scan-id", "--yes-i-am-authorized"]
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "--resume" in result.output


def test_budget_top_up_without_resume_exits_nonzero(tmp_path):
    yaml_path = tmp_path / "llmsec.config.yaml"
    yaml_path.write_text(
        "target:\n  type: http_app\n  method: POST\n  url: http://localhost:8000/chat\n"
        '  body_template: \'{"message": "{{payload}}"}\'\n'
    )
    result = runner.invoke(
        app,
        [
            "scan", "--config", str(yaml_path), "--deep", "--budget-top-up-usd", "5.0", "--yes-i-am-authorized",
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "--budget-top-up-usd" in result.output


def test_resume_cli_unknown_scan_id_exits_nonzero_no_traceback(tmp_path, monkeypatch):
    """A REAL, end-to-end CLI invocation (no mocking) against a config with
    a configured `checkpoint_dir` that has never had anything checkpointed
    under the requested scan_id -- `_build_adapter()` construction never
    makes a network call, and `resume_attacker_campaign()` raises
    `UnknownScanIdError` before any target request."""
    monkeypatch.setenv("TEST_API_KEY", "fake-key-not-a-real-credential")
    yaml_path = tmp_path / "llmsec.config.yaml"
    checkpoint_dir = tmp_path / "checkpoints"
    yaml_path.write_text(
        "target:\n  type: raw_llm\n  model: openai/gpt-4o-mini\n  api_key_env: TEST_API_KEY\n"
        "attacker:\n  enabled: true\n  profile: light\n"
        f"  checkpoint_dir: {checkpoint_dir}\n"
    )
    result = runner.invoke(
        app,
        ["scan", "--config", str(yaml_path), "--deep", "--resume", "no-such-scan", "--yes-i-am-authorized"],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "no-such-scan" in result.output


# --- `resume_attacker_campaign()` mechanics ---------------------------------


async def test_resume_effective_budget_equals_cap_minus_prior_spent(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    config = _make_config(output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "cp", max_rounds=2)
    await _seed_checkpoint_via_mid_campaign_exception(
        monkeypatch,
        fake_target_adapter,
        mock_target_response,
        scripted_chat_model,
        patch_analyst_and_recon_roles,
        config,
        "scan-budget-1"
    )

    seen: list[tuple[float, float, float]] = []
    modules = {"stub_module": _StubModule()}
    resume_adapter = fake_target_adapter()
    resume_adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure2")
    )
    _patch_roles(
        monkeypatch, scripted_chat_model,
        strategist_outputs=[_strategist_output("round2")],
        mutator_outputs=[_mutator_output("round2")],
    )

    result = await resume_attacker_campaign(
        config=config, adapter=resume_adapter, modules=modules, scan_id="scan-budget-1",
        on_prior_spend=lambda prior, cap, effective: seen.append((prior, cap, effective)),
    )

    assert len(seen) == 1
    prior_spent, original_cap, effective_cap = seen[0]
    assert effective_cap == original_cap  # no top-up requested
    # D-75.1: `remaining = cap - spent`, never a fresh full cap.
    remaining = effective_cap - prior_spent
    assert remaining == pytest.approx(original_cap - prior_spent)
    assert result.final_state.get("termination_reason") is not None


async def test_resume_already_at_cap_terminates_immediately_zero_further_spend(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """`agent_call_ceiling=2` trips after round 1's Strategist+Mutator
    calls -- the checkpoint left behind already shows `BUDGET_CAP_EXCEEDED`
    (this is a genuine, non-crash termination). Resuming it must terminate
    immediately with zero further target dispatch."""
    module = _StubModule()
    modules = {"stub_module": module}
    config = _make_config(
        output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "cp", agent_call_ceiling=2, max_rounds=5
    )
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )
    _patch_roles(
        monkeypatch, scripted_chat_model,
        strategist_outputs=[_strategist_output("r") for _ in range(5)],
        mutator_outputs=[_mutator_output("r") for _ in range(5)],
    )
    patch_analyst_and_recon_roles()
    result = await run_attacker_campaign(
        config=config, adapter=adapter, modules=modules, static_results=_static_results(), scan_id="scan-atcap-1"
    )
    assert result.final_state.get("termination_reason") == "BUDGET_CAP_EXCEEDED"

    resume_adapter = fake_target_adapter()
    resumed = await resume_attacker_campaign(
        config=config, adapter=resume_adapter, modules=modules, scan_id="scan-atcap-1"
    )
    assert resumed.final_state.get("termination_reason") == "BUDGET_CAP_EXCEEDED"
    assert resume_adapter.sent_cases == []


async def test_resume_budget_top_up_prints_prior_spend_before_further_spend(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """D-75.1: the explicit top-up flag is required to raise the cap, and
    the callback (which `cli.py` uses to print prior spend) fires BEFORE
    any further spend -- asserted here by checking the adapter has made
    NO calls yet at the moment the callback runs."""
    config = _make_config(output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "cp", max_rounds=2)
    await _seed_checkpoint_via_mid_campaign_exception(
        monkeypatch,
        fake_target_adapter,
        mock_target_response,
        scripted_chat_model,
        patch_analyst_and_recon_roles,
        config,
        "scan-topup-1"
    )

    modules = {"stub_module": _StubModule()}
    resume_adapter = fake_target_adapter()
    resume_adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure2")
    )
    _patch_roles(
        monkeypatch, scripted_chat_model,
        strategist_outputs=[_strategist_output("round2") for _ in range(5)],
        mutator_outputs=[_mutator_output("round2") for _ in range(5)],
    )

    callback_calls: list[tuple[float, float, float]] = []

    def _on_prior_spend(prior: float, cap: float, effective: float) -> None:
        # Nothing has been dispatched by the RESUME adapter yet.
        assert resume_adapter.sent_cases == []
        callback_calls.append((prior, cap, effective))

    result = await resume_attacker_campaign(
        config=config, adapter=resume_adapter, modules=modules, scan_id="scan-topup-1",
        budget_top_up_usd=50.0, on_prior_spend=_on_prior_spend,
    )

    assert len(callback_calls) == 1
    prior, original_cap, effective_cap = callback_calls[0]
    assert effective_cap == 50.0
    assert effective_cap != original_cap
    assert result.final_state.get("budget_ledger", {}).get("cap_usd") == 50.0


async def test_resume_config_fingerprint_mismatch_refuses_zero_spend(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    config = _make_config(output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "cp")
    await _seed_checkpoint_via_mid_campaign_exception(
        monkeypatch,
        fake_target_adapter,
        mock_target_response,
        scripted_chat_model,
        patch_analyst_and_recon_roles,
        config,
        "scan-fp-1"
    )

    changed_config = config.model_copy(deep=True)
    changed_config.attacker.budget_usd = 999.0

    resume_adapter = fake_target_adapter()
    modules = {"stub_module": _StubModule()}
    with pytest.raises(ConfigFingerprintMismatchError):
        await resume_attacker_campaign(
            config=changed_config, adapter=resume_adapter, modules=modules, scan_id="scan-fp-1"
        )
    assert resume_adapter.sent_cases == []


async def test_resume_unknown_scan_id_raises(tmp_path, fake_target_adapter):
    config = _make_config(output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "cp")
    modules = {"stub_module": _StubModule()}
    with pytest.raises(UnknownScanIdError):
        await resume_attacker_campaign(
            config=config, adapter=fake_target_adapter(), modules=modules, scan_id="never-existed"
        )


async def test_resume_without_checkpoint_dir_raises_unknown_scan_id(tmp_path, fake_target_adapter):
    config = _make_config(output_dir=tmp_path / "out", checkpoint_dir=None)
    modules = {"stub_module": _StubModule()}
    with pytest.raises(UnknownScanIdError):
        await resume_attacker_campaign(
            config=config, adapter=fake_target_adapter(), modules=modules, scan_id="anything"
        )


async def test_resume_does_not_redispatch_already_recorded_triple(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """D-75.3: a `(case, round, variant)` triple already recorded as
    dispatched in the restored state is not dispatched again. Simulated by
    injecting a fake `dispatch_results` entry -- BEFORE resuming -- claiming
    round 2's variant (the one round 2's REAL, scripted Mutator will
    legitimately go on to produce) was already dispatched; the resumed
    campaign's own adapter must then never see it."""
    config = _make_config(
        output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "cp", max_rounds=2, variants_per_round=1
    )
    modules = {"stub_module": _StubModule()}
    await _seed_checkpoint_via_mid_campaign_exception(
        monkeypatch,
        fake_target_adapter,
        mock_target_response,
        scripted_chat_model,
        patch_analyst_and_recon_roles,
        config,
        "scan-idem-1"
    )

    settings = resolve_settings(config.attacker)
    from llmsec.attacker.checkpoint import build_checkpointer

    thread_config = {"configurable": {"thread_id": "scan-idem-1"}}
    async with build_checkpointer(settings) as checkpointer:
        dummy_adapter = fake_target_adapter()
        dummy_strategist, dummy_mutator = scripted_chat_model([]), scripted_chat_model([])
        dummy_compiled = build_campaign_graph(
            roles={
                "strategist": build_strategist_agent(settings, config.attacker, model=dummy_strategist),
                "mutator": build_mutator_agent(settings, config.attacker, model=dummy_mutator),
            },
            adapter=dummy_adapter,
            modules=modules,
            max_concurrency=5,
            role_models={"strategist": "test-model", "mutator": "test-model"},
            checkpointer=checkpointer,
            callbacks=None,
        )
        poisoned_record = VariantRecord(
            payload="poison", technique_family="instruction_override", parent_case_id=_STATIC_CASE_ID,
            parent_technique_id=_STATIC_TECHNIQUE_ID, round=2, contributing_agent="mutator", variant_index=0,
        )
        poisoned_entry = {
            "case_id": f"{_STATIC_CASE_ID}-mut-1-poison", "module_id": "stub_module", "record": poisoned_record,
            "target_response": None,
            "eval_result": EvalResult(
                case_id=f"{_STATIC_CASE_ID}-mut-1-poison", verdict=Verdict.UNCERTAIN, confidence=0.0,
                evidence="poison", detection_layer="regex",
            ),
        }
        await dummy_compiled.aupdate_state(thread_config, {"dispatch_results": [poisoned_entry]})

    resume_adapter = fake_target_adapter()
    resume_adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure2")
    )
    _patch_roles(
        monkeypatch, scripted_chat_model,
        strategist_outputs=[_strategist_output("round2")],
        mutator_outputs=[_mutator_output("round2")],
    )

    await resume_attacker_campaign(
        config=config, adapter=resume_adapter, modules=modules, scan_id="scan-idem-1"
    )

    # The idempotency guard skipped round 2's (already-recorded-as-poisoned)
    # triple -- the resume adapter never saw a dispatch for it.
    assert resume_adapter.sent_cases == []


def test_idempotency_key_precomputed_and_pre_populated_prevents_dispatch(
    monkeypatch, fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
):
    """A focused, deterministic unit-level proof of the SAME mechanism the
    integration test above exercises: `build_campaign_graph
    (resume_dispatched_keys=...)`'s filter, driven directly with a
    precomputed key matching exactly what the scripted campaign's own
    Mutator will produce -- `adapter.sent_cases` must stay empty."""
    import asyncio

    async def _run() -> None:
        config = _make_config(output_dir=tmp_path / "out2", checkpoint_dir=tmp_path / "cp2", max_rounds=1)
        settings = resolve_settings(config.attacker)
        modules = {"stub_module": _StubModule()}
        adapter = fake_target_adapter()

        strategist_model = scripted_chat_model([_strategist_output("only")])
        mutator_model = scripted_chat_model([_mutator_output("only")])
        strategist_agent = build_strategist_agent(settings, config.attacker, model=strategist_model)
        mutator_agent = build_mutator_agent(settings, config.attacker, model=mutator_model)

        # The ONLY variant this deterministic campaign will ever produce:
        # round=1 (first round), variant_index=0 (first/only variant).
        already_dispatched = frozenset({idempotency_key(_STATIC_CASE_ID, 1, 0)})

        compiled = build_campaign_graph(
            roles={"strategist": strategist_agent, "mutator": mutator_agent},
            adapter=adapter,
            modules=modules,
            max_concurrency=5,
            role_models={"strategist": "test-model", "mutator": "test-model"},
            checkpointer=None,
            callbacks=None,
            resume_dispatched_keys=already_dispatched,
        )

        from llmsec.attacker.state import QueuedCase, new_campaign_state

        state = new_campaign_state(
            "scan-idem-unit-1", settings, ["stub_module"],
            [
                QueuedCase(
                    module_id="stub_module", case_id=_STATIC_CASE_ID, technique_id=_STATIC_TECHNIQUE_ID,
                    prompt="stub parent payload", verdict="blocked", turns=None,
                )
            ],
        )
        state["current_module"] = "stub_module"
        final_state = await compiled.ainvoke(
            state, config={"configurable": {"thread_id": "scan-idem-unit-1"}}
        )
        assert final_state.get("termination_reason") is not None
        assert adapter.sent_cases == []

    asyncio.run(_run())


# --- D-75.4: bounded-loss disclosure ----------------------------------------


async def test_bounded_loss_disclosure_present_when_ledger_inconsistent_with_round_count(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    config = _make_config(
        output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "cp", max_rounds=2, variants_per_round=1
    )
    modules = {"stub_module": _StubModule()}
    await _seed_checkpoint_via_mid_campaign_exception(
        monkeypatch,
        fake_target_adapter,
        mock_target_response,
        scripted_chat_model,
        patch_analyst_and_recon_roles,
        config,
        "scan-loss-1"
    )

    settings = resolve_settings(config.attacker)
    from llmsec.attacker.checkpoint import build_checkpointer

    thread_config = {"configurable": {"thread_id": "scan-loss-1"}}
    async with build_checkpointer(settings) as checkpointer:
        dummy_adapter = fake_target_adapter()
        dummy_strategist, dummy_mutator = scripted_chat_model([]), scripted_chat_model([])
        dummy_compiled = build_campaign_graph(
            roles={
                "strategist": build_strategist_agent(settings, config.attacker, model=dummy_strategist),
                "mutator": build_mutator_agent(settings, config.attacker, model=dummy_mutator),
            },
            adapter=dummy_adapter, modules=modules, max_concurrency=5,
            role_models={"strategist": "test-model", "mutator": "test-model"}, checkpointer=checkpointer,
            callbacks=None,
        )
        # Force `round` to 2 while `agent_calls` stays at round 1's own
        # value (2) -- 2 rounds implies >= 4 calls, so this is an
        # inconsistency the disclosure must catch.
        tup = await checkpointer.aget_tuple(thread_config)  # type: ignore[arg-type]
        ledger = dict(tup.checkpoint["channel_values"]["budget_ledger"])
        await dummy_compiled.aupdate_state(thread_config, {"round": 2, "budget_ledger": ledger})

    resume_adapter = fake_target_adapter()
    resume_adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text="sure2")
    )
    _patch_roles(
        monkeypatch, scripted_chat_model,
        strategist_outputs=[_strategist_output("r") for _ in range(3)],
        mutator_outputs=[_mutator_output("r") for _ in range(3)],
    )
    result = await resume_attacker_campaign(
        config=config, adapter=resume_adapter, modules=modules, scan_id="scan-loss-1"
    )
    assert any("unrecorded" in note for note in result.limitations)


async def test_bounded_loss_disclosure_absent_when_ledger_consistent(
    monkeypatch,
    fake_target_adapter,
    mock_target_response,
    scripted_chat_model,
    patch_analyst_and_recon_roles,
    tmp_path,
):
    """The already-at-cap scenario's checkpoint (round=1, agent_calls=4:
    1 Recon + Strategist + Mutator + Analyst, 05-07) is internally
    consistent -- no bounded-loss note."""
    module = _StubModule()
    modules = {"stub_module": module}
    config = _make_config(
        output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "cp", agent_call_ceiling=2, max_rounds=5
    )
    adapter = fake_target_adapter()
    adapter.queue_response(f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1"))
    _patch_roles(
        monkeypatch, scripted_chat_model,
        strategist_outputs=[_strategist_output("r") for _ in range(5)],
        mutator_outputs=[_mutator_output("r") for _ in range(5)],
    )
    patch_analyst_and_recon_roles()
    await run_attacker_campaign(
        config=config, adapter=adapter, modules=modules, static_results=_static_results(), scan_id="scan-noloss-1"
    )

    resume_adapter = fake_target_adapter()
    result = await resume_attacker_campaign(
        config=config, adapter=resume_adapter, modules=modules, scan_id="scan-noloss-1"
    )
    assert not any("unrecorded" in note for note in result.limitations)


# --- WR-02: `_bounded_loss_disclosure()`'s per-round call-count formula ----
#
# Pure unit tests of the private function directly -- `state` is a plain
# `CampaignState` dict (`total=False`, so only the keys each scenario cares
# about need constructing), mirroring `test_deep_summary.py`'s own
# hand-built-fixture convention for a pure reconciliation function.


def test_wr02_refused_round_with_three_calls_never_trips_a_false_positive_disclosure():
    """WR-02 regression: a round that hit the D-95 allowlist refusal makes
    only TWO real attacker calls THAT ROUND (Strategist + Analyst -- the
    Mutator/Crescendo `.ainvoke()` is never called on a refused round),
    not the naive per-round constant of THREE the pre-fix formula
    assumed -- one fewer than the pre-fix formula's own per-round
    constant. A single-round campaign therefore has a true minimum of
    Recon's own one-off call (amortized once per campaign, unconditional
    regardless of refusal) + Strategist + Analyst == 3, NOT the pre-fix
    formula's `3 * 1 + 1 == 4`. `agent_calls == 3` must not be misread as
    spend-loss once the refusal discount is applied."""
    from llmsec.attacker.runner import _bounded_loss_disclosure

    state: CampaignState = {
        "round": 1,
        "budget_ledger": {"agent_calls": 3},  # type: ignore[typeddict-item]
        "constraint_violations": [{"round": 0, "case_id": "STUB-001", "technique": "x"}],
    }
    assert _bounded_loss_disclosure(state) is None


def test_wr02_refused_round_still_flags_a_genuine_undercount():
    """The refusal discount tightens the floor -- it must never widen it
    into blindness. `agent_calls == 2` (one call short of the refused
    round's own true minimum of 3: Recon + Strategist + Analyst) is still
    a genuine bounded-loss scenario and must still be disclosed."""
    from llmsec.attacker.runner import _bounded_loss_disclosure

    state: CampaignState = {
        "round": 1,
        "budget_ledger": {"agent_calls": 2},  # type: ignore[typeddict-item]
        "constraint_violations": [{"round": 0, "case_id": "STUB-001", "technique": "x"}],
    }
    note = _bounded_loss_disclosure(state)
    assert note is not None
    assert "unrecorded" in note


def test_wr02_non_refused_round_keeps_the_original_three_call_floor():
    """No recorded refusal -- the formula falls back to the ORIGINAL
    `3 * round_count + 1` floor unchanged (Recon's own one-off call plus
    the full three-role core for 1 round == 4)."""
    from llmsec.attacker.runner import _bounded_loss_disclosure

    consistent_state: CampaignState = {
        "round": 1,
        "budget_ledger": {"agent_calls": 4},  # type: ignore[typeddict-item]
        "constraint_violations": [],
    }
    assert _bounded_loss_disclosure(consistent_state) is None

    inconsistent_state: CampaignState = {
        "round": 1,
        "budget_ledger": {"agent_calls": 3},  # type: ignore[typeddict-item]
        "constraint_violations": [],
    }
    assert _bounded_loss_disclosure(inconsistent_state) is not None


def test_wr02_multiple_refused_rounds_discount_the_floor_by_one_each():
    """Two refused rounds out of a 3-round campaign: the naive floor
    (`3*3+1 == 10`) discounts by 1 per refusal (`10 - 2 == 8`) -- exactly
    matching 1 normal round (3 calls) + 2 refused rounds (2 calls each) +
    Recon's own 1 call == 8."""
    from llmsec.attacker.runner import _bounded_loss_disclosure

    state: CampaignState = {
        "round": 3,
        "budget_ledger": {"agent_calls": 8},  # type: ignore[typeddict-item]
        "constraint_violations": [
            {"round": 0, "case_id": "STUB-001", "technique": "x"},
            {"round": 1, "case_id": "STUB-001", "technique": "x"},
        ],
    }
    assert _bounded_loss_disclosure(state) is None

    # One call short of that tightened floor is still a genuine undercount.
    short_state: CampaignState = {
        "round": 3,
        "budget_ledger": {"agent_calls": 7},  # type: ignore[typeddict-item]
        "constraint_violations": [
            {"round": 0, "case_id": "STUB-001", "technique": "x"},
            {"round": 1, "case_id": "STUB-001", "technique": "x"},
        ],
    }
    assert _bounded_loss_disclosure(short_state) is not None
