"""D-94 gate 1 / AT-6: valid structured output per role (05-10-PLAN.md
Task 2).

Every registered role -- parametrized from `ROLE_REGISTRY` itself, never a
hand-written list, so a role added in a later phase is covered automatically
-- either produces schema-valid structured output within
`MAX_STRUCTURED_OUTPUT_RETRIES + 1` attempts, or surfaces as a recorded
structural failure. Never an unhandled `pydantic.ValidationError` escaping
`.ainvoke()`, and never a silently skipped round: the campaign always
reaches a `termination_reason`, and the round's real dispatch work still
happens when it was already scheduled.

The fault-injection mechanism (`_AlwaysInvalidAgent`) mirrors
`test_role_analyst.py`'s own established pattern: a role-agent double whose
`.ainvoke()` calls `schema.model_validate({})`, forcing a REAL
`pydantic.ValidationError` -- never a mocked `StructuredOutputFailure`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.structured_output import MultipleStructuredOutputsError
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from llmsec.attacker.audit import AttackerAuditHandler, AttackerAuditWriter
from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.graph import build_campaign_graph
from llmsec.attacker.roles import ROLE_REGISTRY
from llmsec.attacker.roles._structured_retry import MAX_STRUCTURED_OUTPUT_RETRIES, invoke_role_with_retry
from llmsec.attacker.roles.analyst import ObservedDefence, build_analyst_agent
from llmsec.attacker.roles.crescendo import CrescendoOutput, build_crescendo_agent
from llmsec.attacker.roles.mutator import MutatedVariant, MutatorOutput, build_mutator_agent
from llmsec.attacker.roles.strategist import StrategistOutput, build_strategist_agent
from llmsec.attacker.state import QueuedCase, new_campaign_state
from llmsec.models import EvalResult, TargetResponse, TestCase, Verdict
from llmsec.plugins.base import BaseModule

from .conftest import (
    CANARY_CREDENTIAL_LITERAL,
    KNOWN_INCOMPATIBLE_MODEL_NAME,
    ScriptedToolCallChatModel,
    flatten_message_batches,
    malformed_tool_call_missing_fields,
    malformed_tool_call_wrong_types,
)

_STATIC_CASE_ID = "STRUCT-001"
_STATIC_TECHNIQUE_ID = "STRUCT-001"


class _StubModule(BaseModule):
    id = "stub_module"
    name = "Stub Module"
    owasp_ref = "LLM00:2025"
    uses_attacker_llm = True

    async def generate_cases(self, context):  # type: ignore[no-untyped-def]
        yield TestCase(
            case_id=_STATIC_CASE_ID, prompt="stub parent payload", technique_id=_STATIC_TECHNIQUE_ID
        )

    async def evaluate(self, case: TestCase, response: TargetResponse) -> EvalResult:
        return EvalResult(
            case_id=case.case_id,
            verdict=Verdict.BLOCKED,
            confidence=0.9,
            evidence=response.raw_text,
            detection_layer="regex",
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


def _neutral_strategist_output(*, escalate: bool = False) -> StrategistOutput:
    return StrategistOutput(
        technique="instruction_override",
        ordered_case_ids=[_STATIC_CASE_ID],
        escalate=escalate,
        reason_code=None,
        rationale="neutral rationale",
    )


def _neutral_mutator_output() -> MutatorOutput:
    return MutatorOutput(
        variants=[
            MutatedVariant(
                payload="neutral variant payload",
                technique_family="instruction_override",
                parent_technique_id=_STATIC_TECHNIQUE_ID,
                rationale="neutral rationale",
            )
        ]
    )


class _MultipleOutputsThenValidAgent:
    """A role-agent double whose FIRST `.ainvoke()` raises
    `MultipleStructuredOutputsError` -- the exception `ToolStrategy` raises
    when a model calls its structured-output tool more than once in one
    turn (observed live against gpt-4o-mini in the Mutator role), a
    `StructuredOutputError` SIBLING of `StructuredOutputValidationError`,
    never the same class. Its second call succeeds. Proves
    `invoke_role_with_retry()` retries this failure shape too, not only
    the schema-validation one `_AlwaysInvalidAgent` already covers."""

    def __init__(self, valid_output: MutatorOutput) -> None:
        self._valid_output = valid_output
        self.call_count = 0

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        if self.call_count == 1:
            ai_message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "MutatorOutput", "args": {}, "id": "call-1"},
                    {"name": "MutatorOutput", "args": {}, "id": "call-2"},
                ],
            )
            raise MultipleStructuredOutputsError(["MutatorOutput", "MutatorOutput"], ai_message)
        return {"structured_response": self._valid_output}


async def test_multiple_structured_outputs_error_is_retried_not_fatal():
    """Regression test for the bug fixed this session: `_structured_retry.py`
    originally caught only `(StructuredOutputValidationError, ValidationError)`,
    so `MultipleStructuredOutputsError` escaped `invoke_role_with_retry()`
    uncaught and aborted the entire deep-mode campaign on a single
    double-tool-call round instead of retrying it like any other
    structured-output failure. Must now be retried and recovered."""
    valid_output = _neutral_mutator_output()
    agent = _MultipleOutputsThenValidAgent(valid_output)

    result = await invoke_role_with_retry(agent, messages=[], role="mutator")

    assert agent.call_count == 2
    assert result == valid_output


def _neutral_analyst_output() -> ObservedDefence:
    return ObservedDefence(
        refusal_style="unchanged from prior observation",
        apparent_filter="none observed",
        what_moved="no change",
        technique_outcome="inconclusive",
        notes="",
    )


def _neutral_crescendo_output() -> CrescendoOutput:
    return CrescendoOutput(
        turns=["neutral turn one", "neutral turn two"],
        arc_rationale="neutral arc",
        backtrack_from_turn=None,
        abort_recommended=False,
    )


class _AlwaysInvalidAgent:
    """A role-agent double whose `.ainvoke()` always raises a REAL
    `pydantic.ValidationError` (via `schema.model_validate({})`), driving
    `invoke_role_with_retry()`'s genuine bounded-retry loop to real
    exhaustion -- never a mocked `StructuredOutputFailure`. Generalized over
    `schema` (each registered role's own `AgentRole.output_schema`) so ONE
    double drives every role's own fault-injection case, parametrized from
    the registry rather than hand-listed per role (mirrors
    `test_role_analyst.py`'s `_AlwaysInvalidAgent`, generalized)."""

    def __init__(self, schema: type[BaseModel], canary: str | None = None) -> None:
        self.schema = schema
        self.call_count = 0
        # 05-11 Rule 1/2 fix: when given, every field's args value is set
        # to `canary` instead of the args dict being empty. At least one
        # field on every registered role's schema is NOT a plain
        # unconstrained `str` (a `list[str]`/`bool`/`Literal`-typed field),
        # so passing a plain string there still forces a REAL
        # `pydantic.ValidationError` -- one whose OWN message embeds
        # `canary` verbatim via `input_value=<canary>` -- letting the AT-6
        # audit-line redaction test below prove the canary never survives
        # into the persisted audit line, without inventing a second
        # fault-injection mechanism.
        self._canary = canary

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        if self._canary is not None:
            self.schema.model_validate({field: self._canary for field in self.schema.model_fields})
        else:
            self.schema.model_validate({})


def _build_roles(
    role_under_test: str, settings, cfg, scripted_chat_model, *, canary: str | None = None
) -> dict[str, Any]:
    """Build the full `roles` dict `build_campaign_graph()` needs, with
    exactly ONE role substituted for `_AlwaysInvalidAgent` (the role under
    test) and every other role given a working neutral scripted double --
    so a fault-injection case for role X never accidentally also fails
    role Y. `canary`, when given, is forwarded to the role-under-test's own
    `_AlwaysInvalidAgent` (05-11 Rule 1/2 fix)."""
    roles: dict[str, Any] = {}

    if role_under_test == "strategist":
        roles["strategist"] = _AlwaysInvalidAgent(ROLE_REGISTRY["strategist"].output_schema, canary)
    else:
        escalate = role_under_test == "crescendo"
        model = scripted_chat_model([_neutral_strategist_output(escalate=escalate)])
        roles["strategist"] = build_strategist_agent(settings, cfg, model=model)

    if role_under_test == "mutator":
        roles["mutator"] = _AlwaysInvalidAgent(ROLE_REGISTRY["mutator"].output_schema, canary)
    else:
        model = scripted_chat_model([_neutral_mutator_output()])
        roles["mutator"] = build_mutator_agent(settings, cfg, model=model)

    if role_under_test == "crescendo":
        roles["crescendo"] = _AlwaysInvalidAgent(ROLE_REGISTRY["crescendo"].output_schema, canary)
    else:
        model = scripted_chat_model([_neutral_crescendo_output()])
        roles["crescendo"] = build_crescendo_agent(settings, cfg, model=model)

    if role_under_test == "analyst":
        roles["analyst"] = _AlwaysInvalidAgent(ROLE_REGISTRY["analyst"].output_schema, canary)
    else:
        model = scripted_chat_model([_neutral_analyst_output()])
        roles["analyst"] = build_analyst_agent(settings, cfg, model=model)

    if role_under_test == "recon":
        # `recon_node` is reachable only via the START edge and is a pure
        # no-op when `roles.get("recon")` is absent (`graph.py`'s own
        # docstring) -- omitted entirely for every OTHER role's case, since
        # a working double is never needed to isolate a different role's
        # fault.
        roles["recon"] = _AlwaysInvalidAgent(ROLE_REGISTRY["recon"].output_schema, canary)

    return roles


@pytest.mark.parametrize("role_name", sorted(ROLE_REGISTRY))
async def test_each_registered_role_structural_failure_is_recorded_never_unhandled(
    role_name, fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
):
    """Parametrized directly from `ROLE_REGISTRY` -- one test id per
    registered role, so a role added in a later phase is covered
    automatically without editing this file.

    05-11 Rule 1/2 fix (AT-6 expanded rubric, `05-AI-SPEC.md` ~line 530):
    ALSO asserts the audit-trail entry the fix adds (attempt count, schema
    violation, redacted raw output), that `role_structural_failures`
    increments by exactly 1 for this role, and that `constraint_violations`
    (D-95 allowlist refusals only) is untouched -- proving the two fields
    are genuinely separate. `_AlwaysInvalidAgent`'s `canary` param embeds
    `CANARY_CREDENTIAL_LITERAL` into the forced `ValidationError`'s own
    message, so the redaction assertion below is a real negative control,
    not a vacuous one.
    """
    module = _StubModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    roles = _build_roles(
        role_name, settings, cfg, scripted_chat_model, canary=CANARY_CREDENTIAL_LITERAL
    )
    bad_agent = roles[role_name]
    assert isinstance(bad_agent, _AlwaysInvalidAgent)

    writer = AttackerAuditWriter(tmp_path, f"scan-struct-{role_name}")
    handler = AttackerAuditHandler(writer, f"scan-struct-{role_name}")
    compiled = build_campaign_graph(
        roles=roles,
        adapter=adapter,
        modules={"stub_module": module},
        max_concurrency=5,
        callbacks=[handler],
    )

    initial_state = new_campaign_state(
        scan_id=f"scan-struct-{role_name}",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    # Never an unhandled ValidationError: if the retry loop failed to
    # contain it, THIS `await` would raise and the test would error out
    # here, not merely fail an assertion below.
    final_state = await compiled.ainvoke(initial_state)
    writer.close()

    # The bounded retry genuinely exhausted -- exactly
    # MAX_STRUCTURED_OUTPUT_RETRIES + 1 real attempts, never fewer (silently
    # giving up early) and never more (retrying beyond the bound).
    assert bad_agent.call_count == MAX_STRUCTURED_OUTPUT_RETRIES + 1

    # The fault injection genuinely forced at least one real validation
    # failure -- a gate that passed because the double was never actually
    # invoked would be vacuous.
    assert bad_agent.call_count >= 1

    # Never a silently skipped round: the campaign always reaches a
    # recorded termination reason.
    assert final_state.get("termination_reason") is not None

    # `role_structural_failures` incremented by exactly 1 for THIS role;
    # `constraint_violations` (D-95 allowlist refusals only) never does --
    # the two fields are genuinely separate now (05-11 Rule 1/2 fix).
    structural_failures = final_state.get("role_structural_failures", [])
    matching_failures = [f for f in structural_failures if f.get("role") == role_name]
    assert len(matching_failures) == 1
    assert matching_failures[0]["attempt_count"] == MAX_STRUCTURED_OUTPUT_RETRIES + 1
    assert matching_failures[0]["reason"]
    violations = final_state.get("constraint_violations", [])
    assert not any(v.get("role") == role_name for v in violations)

    # An audit-trail entry was written for the failing role, carrying the
    # attempt count and the schema-violation text -- and the canary literal
    # embedded in the forced ValidationError's own message never survives
    # verbatim into the persisted line (D-86, no exemptions).
    written = _read_audit_lines(writer.path)
    structural_lines = [
        line
        for line in written
        if line["event"] == "inter_agent_handoff" and line["agent"] == role_name
    ]
    assert len(structural_lines) == 1
    line_content = structural_lines[0]["content"]
    assert f"attempts={MAX_STRUCTURED_OUTPUT_RETRIES + 1}" in line_content
    assert "schema_violation=" in line_content
    assert CANARY_CREDENTIAL_LITERAL not in line_content
    assert "REDACTED" in line_content


# --- Malformed-fixture set: includes the recorded known-incompatible ------
# --- model's own failure shape ----------------------------------------------


def test_malformed_fixture_set_includes_known_incompatible_model_shape():
    """`KNOWN_INCOMPATIBLE_MODEL_NAME` (STATE.md "Blockers/Concerns":
    `groq/openai/gpt-oss-120b`'s tool-calling fails schema validation on
    most calls) is named explicitly, and its modelled failure shape
    (`malformed_tool_call_wrong_types()`) fails in the SAME recorded way as
    any other malformed response -- never crashing the role, never
    surfacing as an unhandled exception."""
    assert KNOWN_INCOMPATIBLE_MODEL_NAME == "groq/openai/gpt-oss-120b"
    for schema in (ROLE_REGISTRY[name].output_schema for name in ROLE_REGISTRY):
        malformed = malformed_tool_call_wrong_types(schema)
        assert malformed["name"] == schema.__name__
        try:
            schema.model_validate(malformed["args"])
        except Exception:  # noqa: BLE001 -- asserting it DOES fail validation
            continue
        pytest.fail(f"expected {schema.__name__} to reject the wrong-typed fixture args")


async def test_known_incompatible_model_shaped_response_fails_recorded_not_crashed(
    fake_target_adapter, mock_target_response
):
    """Drives the REAL Strategist agent with the known-incompatible model's
    modelled failure shape for every attempt -- a bounded, recorded
    structural failure, never a crash and never a silently skipped round."""
    module = _StubModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    schema = ROLE_REGISTRY["strategist"].output_schema
    script = [malformed_tool_call_wrong_types(schema) for _ in range(MAX_STRUCTURED_OUTPUT_RETRIES + 1)]
    strategist_model = ScriptedToolCallChatModel(script=script)

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(
            settings, cfg, model=ScriptedToolCallChatModel(script=[])
        ),
        "crescendo": build_crescendo_agent(
            settings, cfg, model=ScriptedToolCallChatModel(script=[])
        ),
    }
    compiled = build_campaign_graph(
        roles=roles, adapter=adapter, modules={"stub_module": module}, max_concurrency=5
    )

    initial_state = new_campaign_state(
        scan_id="scan-known-incompatible",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)

    assert strategist_model._call_index == MAX_STRUCTURED_OUTPUT_RETRIES + 1  # noqa: SLF001
    assert final_state.get("termination_reason") is not None
    # The Strategist's own failure is recorded as TECHNIQUES_EXHAUSTED --
    # never the generic ROUND_CAP_REACHED, so a report can distinguish "the
    # model failed" from "the round budget ran out".
    assert final_state.get("termination_reason") == "TECHNIQUES_EXHAUSTED"


# --- Recover-on-second-attempt: succeeds, and every attempt is auditable ---


def _read_audit_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def test_role_recovers_on_second_attempt_and_every_attempt_is_auditable(
    fake_target_adapter, mock_target_response, scripted_chat_model, tmp_path
):
    """A role whose first response is invalid and whose second is valid
    succeeds -- and BOTH attempts are independently observable in the audit
    trail (one `model_start`/`model_end` pair per attempt), proving the
    retry itself left a trace rather than only the final outcome."""
    module = _StubModule()
    adapter = fake_target_adapter()
    adapter.queue_response(
        f"{_STATIC_CASE_ID}-mut-1", mock_target_response(case_id=f"{_STATIC_CASE_ID}-mut-1")
    )

    cfg = AttackerConfig(profile="light", max_rounds=1, variants_per_round=1)
    settings = resolve_settings(cfg)

    valid_output = _neutral_strategist_output()
    script = [
        malformed_tool_call_missing_fields(StrategistOutput),
        {"name": "StrategistOutput", "args": valid_output.model_dump(mode="json")},
    ]
    strategist_model = ScriptedToolCallChatModel(script=script)
    mutator_model = scripted_chat_model([_neutral_mutator_output()])
    analyst_model = scripted_chat_model([_neutral_analyst_output()])
    crescendo_model = scripted_chat_model([_neutral_crescendo_output()])

    roles = {
        "strategist": build_strategist_agent(settings, cfg, model=strategist_model),
        "mutator": build_mutator_agent(settings, cfg, model=mutator_model),
        "analyst": build_analyst_agent(settings, cfg, model=analyst_model),
        "crescendo": build_crescendo_agent(settings, cfg, model=crescendo_model),
    }

    writer = AttackerAuditWriter(tmp_path, "scan-retry-recover")
    handler = AttackerAuditHandler(writer, "scan-retry-recover")
    compiled = build_campaign_graph(
        roles=roles,
        adapter=adapter,
        modules={"stub_module": module},
        max_concurrency=5,
        callbacks=[handler],
    )

    initial_state = new_campaign_state(
        scan_id="scan-retry-recover",
        settings=settings,
        module_order=["stub_module"],
        case_queue=_case_queue(),
    )
    initial_state["current_module"] = "stub_module"
    initial_state["enabled_techniques"] = ["instruction_override"]

    final_state = await compiled.ainvoke(initial_state)
    writer.close()

    # The Strategist recovered on its SECOND attempt: both script entries
    # were consumed, the round completed normally (a variant was
    # dispatched), and the recovered technique is visible in final state.
    assert strategist_model._call_index == 2  # noqa: SLF001 -- test-only introspection
    assert len(final_state.get("dispatch_results", [])) == 1
    assert final_state.get("selected_technique") == "instruction_override"

    # Every attempt is independently auditable: at least two "model_start"
    # lines recorded with role="strategist" -- one per real model call
    # (valid or not), since the audit handler observes the raw LLM call
    # itself, not just the eventual schema-validation outcome.
    written = _read_audit_lines(writer.path)
    strategist_starts = [
        line for line in written if line["role"] == "strategist" and line["event"] == "model_start"
    ]
    assert len(strategist_starts) >= 2

    # The SECOND attempt is not a blind resend of the first attempt's exact
    # brief: it carries the first attempt's own violation text forward, so
    # the model sees what specifically was rejected before trying again.
    # Regression coverage for the fix made this session -- previously
    # `invoke_role_with_retry()` re-sent the identical `messages` list on
    # every attempt with no feedback at all.
    assert len(strategist_model.received_message_batches) >= 2
    second_attempt_text = flatten_message_batches([strategist_model.received_message_batches[1]])
    assert "rejected" in second_attempt_text.lower()
    assert "field required" in second_attempt_text.lower()
