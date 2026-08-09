"""05-06-PLAN.md Task 2: redacting checkpoint serializer with a proven
canary control (D-73 mitigation 2 / D-75, D-94 framework gate #1).

Covers every bullet in `<behavior>`: `RedactingJsonPlusSerializer.dumps_typed()`
masks a planted canary while the unmodified base class's dumps_typed()
serializes it verbatim (the mandatory control, in the SAME test module);
a `dumps_typed()`-then-`loads_typed()` round trip yields a structurally
valid object with no field dropped; `build_checkpointer()` yields a
redacting disk-backed saver when a checkpoint directory is configured and
an in-memory saver otherwise, never a disk-backed saver without the
redacting serde; a FULL scripted campaign's real on-disk SQLite checkpoint
bytes contain zero occurrences of a canary planted in the target's
response, with a same-campaign, non-redacting-serde control that DOES find
it; `config_fingerprint()` is stable/differing as D-75.2 requires; and
`idempotency_key()` is distinct/stable as D-75.3 requires.

Reuses `tests/attacker/conftest.py`'s `CANARY_CREDENTIAL_LITERAL` (already
the codebase's shared attacker-test canary) and the same
scripted-role/fake-adapter fixtures `test_audit_wiring.py` established,
rather than re-deriving a second campaign harness.
"""

from __future__ import annotations

import aiosqlite
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from llmsec.attacker.checkpoint import (
    RedactingJsonPlusSerializer,
    build_checkpointer,
    config_fingerprint,
    idempotency_key,
)
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.state import QueuedCase, new_campaign_state
from llmsec.config import ScanConfig, TargetConfig
from llmsec.models import EvalResult, ScanContext, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

from .conftest import CANARY_CREDENTIAL_LITERAL

#: A genuine (non-canary) SSN literal for the round-trip test below --
#: deliberately NOT `conftest.py`'s `CANARY_PII_LITERAL`, whose area code
#: (912, inside the SSA's never-issued 900-999 range) is EXCLUDED by
#: `pii_patterns._SSN_RE`'s own negative lookahead precisely so it can be
#: used as an echo-detection canary elsewhere without ever being redacted
#: (03-03-SUMMARY.md) -- asserting THAT literal gets redacted here would
#: test a pattern this codebase deliberately built to not match it.
_REAL_SSN_LITERAL = "123-45-6789"

_STATIC_CASE_ID = "STUB-001"
_STATIC_TECHNIQUE_ID = "STUB-001"


class _CanaryStubModule(BaseModule):
    """One static case; `evaluate()` folds the target's raw response text
    into `EvalResult.evidence` verbatim, so a canary planted in the target's
    response flows into checkpointed graph state exactly the way a real
    target-derived secret would."""

    id = "canary_stub_module"
    name = "Canary Stub Module"
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
                payload="mutated payload variant 1",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale="rationale 1",
            )
        ]
    )


async def _run_one_round_campaign_with_checkpointer(
    *, checkpointer, fake_target_adapter, mock_target_response, scripted_chat_model, scan_id, raw_text
):
    """Drives `build_campaign_graph()`/`.ainvoke()` directly (mirroring
    `test_audit_wiring.py`'s own `test_campaign_completes_with_no_audit_
    handler_supplied`) rather than `run_attacker_campaign()`, so a test can
    supply an explicit `checkpointer` -- `run_attacker_campaign()` always
    routes through `build_checkpointer()`'s forced redacting serde, which
    would make a non-redacting CONTROL run impossible to drive through it.
    """
    settings = resolve_settings(
        AttackerConfig(enabled=True, profile="light", max_rounds=1, variants_per_round=1)
    )
    cfg = AttackerConfig(enabled=True, profile="light", max_rounds=1, variants_per_round=1)

    strategist_model = scripted_chat_model([_strategist_output()])
    mutator_model = scripted_chat_model([_mutator_output()])
    strategist_agent = build_strategist_agent(settings, cfg, model=strategist_model)
    mutator_agent = build_mutator_agent(settings, cfg, model=mutator_model)

    module = _CanaryStubModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1", raw_text=raw_text)
    )

    compiled = build_campaign_graph(
        roles={"strategist": strategist_agent, "mutator": mutator_agent},
        adapter=adapter,
        modules={"canary_stub_module": module},
        max_concurrency=5,
        role_models={"strategist": "test-model", "mutator": "test-model"},
        checkpointer=checkpointer,
        callbacks=None,
    )

    state = new_campaign_state(
        scan_id,
        settings,
        ["canary_stub_module"],
        [
            QueuedCase(
                module_id="canary_stub_module",
                case_id=_STATIC_CASE_ID,
                technique_id=_STATIC_TECHNIQUE_ID,
                prompt="stub parent payload",
                verdict="blocked",
                turns=None,
            )
        ],
    )
    state["current_module"] = "canary_stub_module"
    state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(state, config={"configurable": {"thread_id": scan_id}})
    assert len(final_state.get("dispatch_results", [])) == 1
    return final_state


# --- `RedactingJsonPlusSerializer.dumps_typed()` unit-level proof, with the
# --- mandatory control in the SAME test module ------------------------------


def test_dumps_typed_masks_canary_same_type_tag_as_base_control_finds_it():
    obj = {
        "case_id": _STATIC_CASE_ID,
        "evidence": f"complied: {CANARY_CREDENTIAL_LITERAL}",
        "harmless": "no secret here",
    }
    base = JsonPlusSerializer()
    redacting = RedactingJsonPlusSerializer()

    base_type, base_bytes = base.dumps_typed(obj)
    redacted_type, redacted_bytes = redacting.dumps_typed(obj)

    assert redacted_type == base_type
    # Control: the UNMODIFIED base class serializes the canary verbatim --
    # proves the test itself is meaningful, not vacuously passing because
    # the canary never reached a serializer at all.
    assert CANARY_CREDENTIAL_LITERAL.encode() in base_bytes
    # Treatment: the redacting subclass never lets it reach the bytes.
    assert CANARY_CREDENTIAL_LITERAL.encode() not in redacted_bytes


def test_dumps_typed_round_trip_masks_secret_no_field_dropped_or_corrupted():
    obj = {
        "case_id": "DIRECT-001-mut-1",
        "evidence": f"target replied with {CANARY_CREDENTIAL_LITERAL}",
        "confidence": 0.95,
        "verdict": Verdict.FULL_COMPROMISE,
        "eval_result": EvalResult(
            case_id="DIRECT-001-mut-1",
            verdict=Verdict.FULL_COMPROMISE,
            confidence=0.9,
            evidence=f"complied: {CANARY_CREDENTIAL_LITERAL}",
            detection_layer="regex",
        ),
        "target_response": TargetResponse(
            case_id="DIRECT-001-mut-1", raw_text=f"here: {CANARY_CREDENTIAL_LITERAL}", latency_ms=1.0
        ),
        "nested": {"pii": _REAL_SSN_LITERAL, "count": 3},
    }
    redacting = RedactingJsonPlusSerializer()
    type_tag, data = redacting.dumps_typed(obj)
    restored = redacting.loads_typed((type_tag, data))

    # Structurally valid, no field dropped.
    assert restored["case_id"] == "DIRECT-001-mut-1"
    assert restored["confidence"] == 0.95
    assert restored["nested"]["count"] == 3

    # The secret-bearing fields come back masked, never the raw literal.
    assert CANARY_CREDENTIAL_LITERAL not in restored["evidence"]
    assert CANARY_CREDENTIAL_LITERAL not in restored["eval_result"].evidence
    assert CANARY_CREDENTIAL_LITERAL not in restored["target_response"].raw_text
    assert _REAL_SSN_LITERAL not in restored["nested"]["pii"]

    # `Verdict`/`EvalResult`/`TargetResponse` type fidelity is preserved
    # across the round trip (05-06 Task 2's own project-type msgpack
    # allowlist, see checkpoint.py's `_PROJECT_MSGPACK_ALLOWLIST`) -- a
    # `--resume`d campaign's restored `dispatch_results` entries must stay
    # real typed objects, never silently degrade to a raw dict/str.
    assert restored["verdict"] is Verdict.FULL_COMPROMISE
    assert isinstance(restored["eval_result"], EvalResult)
    assert restored["eval_result"].verdict is Verdict.FULL_COMPROMISE
    assert isinstance(restored["target_response"], TargetResponse)


# --- `build_checkpointer()` -------------------------------------------------


async def test_build_checkpointer_yields_memory_saver_when_no_checkpoint_dir():
    settings = resolve_settings(AttackerConfig(enabled=True, profile="light"))
    async with build_checkpointer(settings) as checkpointer:
        assert isinstance(checkpointer, MemorySaver)


async def test_build_checkpointer_yields_disk_backed_saver_with_redacting_serde(tmp_path):
    settings = resolve_settings(
        AttackerConfig(enabled=True, profile="light", checkpoint_dir=str(tmp_path / "checkpoints"))
    )
    async with build_checkpointer(settings) as checkpointer:
        assert isinstance(checkpointer, AsyncSqliteSaver)
        assert isinstance(checkpointer.serde, RedactingJsonPlusSerializer)
    assert (tmp_path / "checkpoints" / "attacker-checkpoints.sqlite").exists()


@pytest.mark.parametrize("checkpoint_dir_suffix", [None, "", "nested/checkpoints"])
async def test_build_checkpointer_never_yields_disk_backed_saver_without_redacting_serde(
    tmp_path, checkpoint_dir_suffix
):
    """Across every argument combination this function exercises: either a
    `MemorySaver` (no checkpoint dir configured), or a disk-backed saver
    that is ALWAYS wired with `RedactingJsonPlusSerializer` -- never a
    disk-backed saver with a different, non-redacting `serde=`."""
    resolved_dir = str(tmp_path / checkpoint_dir_suffix) if checkpoint_dir_suffix else checkpoint_dir_suffix
    settings = resolve_settings(AttackerConfig(enabled=True, profile="light", checkpoint_dir=resolved_dir))
    async with build_checkpointer(settings) as checkpointer:
        if isinstance(checkpointer, AsyncSqliteSaver):
            assert isinstance(checkpointer.serde, RedactingJsonPlusSerializer)
        else:
            assert isinstance(checkpointer, MemorySaver)


# --- Full scripted campaign, real on-disk SQLite bytes, mandatory control --


async def test_full_campaign_checkpoint_bytes_contain_no_canary_on_disk(
    fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
):
    """After a full scripted campaign whose target response contains the
    canary credential, reading EVERY byte of the real on-disk checkpoint
    artifact finds zero occurrences of the literal."""
    canary_raw_text = f"sure, here it is: {CANARY_CREDENTIAL_LITERAL}"
    db_path = tmp_path / "treatment.sqlite"
    async with aiosqlite.connect(db_path) as conn:
        saver = AsyncSqliteSaver(conn, serde=RedactingJsonPlusSerializer())
        await saver.setup()
        await _run_one_round_campaign_with_checkpointer(
            checkpointer=saver,
            fake_target_adapter=fake_target_adapter,
            mock_target_response=mock_target_response,
            scripted_chat_model=scripted_chat_model,
            scan_id="scan-canary-treatment",
            raw_text=canary_raw_text,
        )

    db_bytes = db_path.read_bytes()
    assert CANARY_CREDENTIAL_LITERAL.encode() not in db_bytes


async def test_full_campaign_checkpoint_bytes_contain_canary_with_plain_serde_control(
    fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
):
    """The SAME campaign path, run with a plain (non-redacting) `serde=` --
    the end-to-end control proving the treatment test above is meaningful,
    not vacuously passing because the canary never reached the
    checkpointer at all."""
    canary_raw_text = f"sure, here it is: {CANARY_CREDENTIAL_LITERAL}"
    db_path = tmp_path / "control.sqlite"
    async with aiosqlite.connect(db_path) as conn:
        saver = AsyncSqliteSaver(conn, serde=JsonPlusSerializer())
        await saver.setup()
        await _run_one_round_campaign_with_checkpointer(
            checkpointer=saver,
            fake_target_adapter=fake_target_adapter,
            mock_target_response=mock_target_response,
            scripted_chat_model=scripted_chat_model,
            scan_id="scan-canary-control",
            raw_text=canary_raw_text,
        )

    db_bytes = db_path.read_bytes()
    assert CANARY_CREDENTIAL_LITERAL.encode() in db_bytes


# --- `config_fingerprint()` (D-75.2) -----------------------------------------


def _config(**attacker_overrides) -> ScanConfig:
    defaults = dict(enabled=True, profile="light", budget_usd=1.0, model="openai:gpt-4o-mini")
    defaults.update(attacker_overrides)
    return ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        enabled_modules=["prompt_injection", "pii_exfiltration"],
        attacker=AttackerConfig(**defaults),
    )


def test_config_fingerprint_stable_across_two_calls():
    config = _config()
    settings = resolve_settings(config.attacker)
    assert config_fingerprint(config, settings) == config_fingerprint(config, settings)


def test_config_fingerprint_differs_when_cap_changes():
    base_config = _config()
    changed_config = _config(budget_usd=999.0)
    base_fp = config_fingerprint(base_config, resolve_settings(base_config.attacker))
    changed_fp = config_fingerprint(changed_config, resolve_settings(changed_config.attacker))
    assert base_fp != changed_fp


def test_config_fingerprint_differs_when_model_changes():
    base_config = _config()
    changed_config = _config(model="openai:gpt-4o")
    base_fp = config_fingerprint(base_config, resolve_settings(base_config.attacker))
    changed_fp = config_fingerprint(changed_config, resolve_settings(changed_config.attacker))
    assert base_fp != changed_fp


def test_config_fingerprint_differs_when_profile_changes():
    base_config = _config(profile="light")
    changed_config = _config(profile="thorough")
    base_fp = config_fingerprint(base_config, resolve_settings(base_config.attacker))
    changed_fp = config_fingerprint(changed_config, resolve_settings(changed_config.attacker))
    assert base_fp != changed_fp


def test_config_fingerprint_differs_when_module_list_changes():
    base_config = _config()
    changed_config = _config()
    changed_config.enabled_modules = ["prompt_injection"]
    base_fp = config_fingerprint(base_config, resolve_settings(base_config.attacker))
    changed_fp = config_fingerprint(changed_config, resolve_settings(changed_config.attacker))
    assert base_fp != changed_fp


def test_config_fingerprint_excludes_volatile_fields_via_stable_json_canonicalization():
    """Two configs differing only in a volatile field this function
    deliberately does not consult (mirrors via direct construction rather
    than a `scan_id`/timestamp parameter this function does not accept)
    fingerprint identically -- the digest is over `json.dumps(...,
    sort_keys=True)`, so key ORDER never affects the result either."""
    config = _config()
    settings = resolve_settings(config.attacker)
    fp_a = config_fingerprint(config, settings)
    # Re-fingerprint from an independently reconstructed config/settings
    # pair with fields supplied in a different order -- must still match.
    config_b = ScanConfig(
        target=TargetConfig(type="raw_llm", model="openai/gpt-4o-mini", api_key_env="TEST_API_KEY"),
        attacker=AttackerConfig(
            model="openai:gpt-4o-mini", budget_usd=1.0, profile="light", enabled=True
        ),
        enabled_modules=["prompt_injection", "pii_exfiltration"],
    )
    fp_b = config_fingerprint(config_b, resolve_settings(config_b.attacker))
    assert fp_a == fp_b


# --- `idempotency_key()` (D-75.3) -------------------------------------------


def test_idempotency_key_distinct_for_distinct_triples():
    keys = {
        idempotency_key("DIRECT-001", 1, 0),
        idempotency_key("DIRECT-001", 1, 1),
        idempotency_key("DIRECT-001", 2, 0),
        idempotency_key("DIRECT-002", 1, 0),
    }
    assert len(keys) == 4


def test_idempotency_key_stable_for_repeated_triple():
    assert idempotency_key("DIRECT-001", 1, 0) == idempotency_key("DIRECT-001", 1, 0)


# --- 05-attacker-llm-deep-mode WR-01: budget ledger per_role isolation -----


async def test_wr01_role_call_spend_never_mutates_a_previously_held_per_role_reference(
    fake_target_adapter, mock_target_response, scripted_chat_model
):
    """WR-01 regression: `graph.py`'s `_record_role_call_spend()` did
    `ledger: BudgetLedger = dict(state.get("budget_ledger") or {})` -- a
    SHALLOW copy that left `ledger["per_role"]` pointing at the SAME
    nested dict object `state["budget_ledger"]["per_role"]` already
    references. `record_agent_spend()` (`budget.py`) then does
    `per_role.setdefault(role, ...)`, which -- since `new_campaign_state()`
    already seeds a `calls=0` `RoleSpend` entry for EVERY role name via
    `new_role_spend()` -- returns the EXISTING entry by reference on every
    single call (including the very first), and mutates it in place via
    `+=`.

    This directly exercises the failure mode the fix (per-entry re-copy of
    `per_role`) closes: a reference held onto the PRE-INVOKE `RoleSpend`
    dict (standing in for a checkpoint-history/external-cache reference,
    which is what WR-01 is actually about protecting -- `budget_ledger` is
    a plain `LastValue` channel here, so the initial state's own nested
    objects flow through node invocations by reference, not by copy) must
    see NO mutation once the graph completes -- only the NEWLY RETURNED
    ledger (a genuinely independent object) reflects the increment.
    """
    settings = resolve_settings(
        AttackerConfig(enabled=True, profile="light", max_rounds=1, variants_per_round=1)
    )
    cfg = AttackerConfig(enabled=True, profile="light", max_rounds=1, variants_per_round=1)

    strategist_model = scripted_chat_model([_strategist_output()])
    mutator_model = scripted_chat_model([_mutator_output()])
    strategist_agent = build_strategist_agent(settings, cfg, model=strategist_model)
    mutator_agent = build_mutator_agent(settings, cfg, model=mutator_model)

    module = _CanaryStubModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )

    compiled = build_campaign_graph(
        roles={"strategist": strategist_agent, "mutator": mutator_agent},
        adapter=adapter,
        modules={"canary_stub_module": module},
        max_concurrency=5,
        role_models={"strategist": "test-model", "mutator": "test-model"},
        checkpointer=None,
        callbacks=None,
    )

    state = new_campaign_state(
        "scan-wr01-1",
        settings,
        ["canary_stub_module"],
        [
            QueuedCase(
                module_id="canary_stub_module",
                case_id=_STATIC_CASE_ID,
                technique_id=_STATIC_TECHNIQUE_ID,
                prompt="stub parent payload",
                verdict="blocked",
                turns=None,
            )
        ],
    )
    state["current_module"] = "canary_stub_module"
    state["enabled_techniques"] = ["instruction_override"]

    # Hold a direct reference to the PRE-INVOKE strategist `RoleSpend`
    # dict -- the object `record_agent_spend()`'s `setdefault()` would
    # return (and mutate in place) under the bug.
    pre_invoke_strategist_spend = state["budget_ledger"]["per_role"]["strategist"]
    assert pre_invoke_strategist_spend["calls"] == 0

    final_state = await compiled.ainvoke(
        state, config={"configurable": {"thread_id": "scan-wr01-1"}}
    )

    # The NEW final state's own per_role entry reflects the real call
    # (Strategist is invoked exactly once for a 1-round campaign).
    assert final_state["budget_ledger"]["per_role"]["strategist"]["calls"] == 1
    # The externally-held PRE-INVOKE reference must be untouched -- proof
    # the ledger update never mutated a shared, previously-reachable
    # object rather than returning a genuinely independent one.
    assert pre_invoke_strategist_spend["calls"] == 0


# --- CR-01 order regex acceptance criterion, re-asserted in-suite ----------


def test_source_never_inverts_cr01_redaction_order():
    """05-06-PLAN.md's own acceptance criterion, re-asserted inside the test
    suite (not only as an external `python -c` invocation): the inverted
    call form never appears in this module. `checkpoint.py` reuses
    `audit.py`'s `redact_audit_text()` directly rather than re-composing
    `redact_pii_match`/`redact_credential_match` a second time, so this is
    trivially satisfied -- asserted anyway so a future edit that DID
    re-introduce a direct call pair would be caught here too."""
    import re
    from pathlib import Path

    import llmsec.attacker.checkpoint as checkpoint_module

    src = Path(checkpoint_module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"redact_pii_match\(\s*redact_credential_match\(", src)
