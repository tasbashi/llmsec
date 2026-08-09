"""Tests for `llmsec.attacker.audit`'s capture half (05-05-PLAN.md Task 2):
`redact_audit_text()` (the D-86 no-exemption redaction chokepoint) and
`AttackerAuditHandler` (the callback handler that observes every attacker
exchange).

D-86 is rated one-way in the security sense: a secret written to an
operator's disk cannot be un-written. This module proves the no-exemption
policy with byte-level assertions against planted canary literals, never
just an interface-shape check.
"""

from __future__ import annotations

import json
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from llmsec.attacker.audit import (
    AttackerAuditHandler,
    AttackerAuditWriter,
    audit_path_for,
    redact_audit_text,
)

from .conftest import CANARY_CREDENTIAL_LITERAL, CANARY_PII_LITERAL

# --- redact_audit_text: the D-86 no-exemption chokepoint --------------------

#: A real-shaped SSN (area 123, NOT the 900-999 "never issued" canary
#: range `_SSN_RE` deliberately excludes) -- used where a test needs a PII
#: literal `redact_pii_match()` actually matches, as opposed to
#: `CANARY_PII_LITERAL` (conftest.py), which is deliberately chosen from an
#: area range `_SSN_RE` excludes precisely so a target's verbatim echo of
#: it survives detection as canary-echo proof rather than being redacted.
_REAL_SHAPED_SSN = "123-45-6789"


def test_redact_audit_text_masks_a_planted_credential_and_a_planted_pii_literal():
    text = f"credential: {CANARY_CREDENTIAL_LITERAL} pii: {_REAL_SHAPED_SSN}"
    redacted = redact_audit_text(text)
    assert CANARY_CREDENTIAL_LITERAL not in redacted
    assert _REAL_SHAPED_SSN not in redacted
    assert "REDACTED" in redacted


def test_redact_audit_text_masks_credential_adjacent_to_canary_pii_no_exemption():
    """D-86: unlike `api.py`'s report path (`redact_all_protecting_literals()`),
    there is no exemption path here -- a canary PII literal sitting
    immediately next to a real-shaped credential does not protect the
    credential from redaction. (`CANARY_PII_LITERAL` itself is drawn from
    an SSN area range `_SSN_RE` deliberately never matches -- by design,
    not by any exemption codepath in this module -- so it is not itself
    asserted redacted here; `test_redact_audit_text_never_imports_the_canary_exemption_helper`
    is what proves no protecting-literal mechanism exists at all.)"""
    text = f"{CANARY_PII_LITERAL} {CANARY_CREDENTIAL_LITERAL}"
    redacted = redact_audit_text(text)
    assert CANARY_CREDENTIAL_LITERAL not in redacted


def test_redact_audit_text_masks_dot_structured_secret_with_canary_middle_segment():
    """CR-02/03-REVIEW.md regression shape: a JWT-shaped
    `header.CANARY.footer` string. `redact_pii_match()`'s `_JWT_RE` matches
    the WHOLE three-dot-segment span in one shot (no protecting-literal
    carve-out exists in this module), so none of header, canary literal, or
    footer survives -- proving the fixed CR-01 order holds for exactly the
    shape a prior gap-closure cycle was spent on, with no exemption applied
    to the canary segment this time."""
    header = "REALSECRETHEAD1234567890"
    footer = "REALSECRETFOOT1234567890"
    canary_middle = "llmsec-canary-abcdef0123456789abcdef0123456789"
    text = f"{header}.{canary_middle}.{footer}"

    redacted = redact_audit_text(text)

    assert header not in redacted
    assert canary_middle not in redacted
    assert footer not in redacted
    assert "REDACTED" in redacted


def test_redact_audit_text_empty_string_returns_empty_and_never_raises():
    assert redact_audit_text("") == ""


def test_redact_audit_text_never_imports_the_canary_exemption_helper():
    """Structural guard mirroring the plan's own grep-based acceptance
    criterion, run here as a test so a regression fails `pytest` too, not
    only a manual grep."""
    import llmsec.attacker.audit as audit_module

    assert not hasattr(audit_module, "redact_all_protecting_literals")


# --- AttackerAuditHandler: capture, context, redaction-before-write --------


def _chat_result(text: str, *, total_tokens: int | None = None) -> LLMResult:
    message = AIMessage(
        content=text,
        usage_metadata=({"input_tokens": 1, "output_tokens": 1, "total_tokens": total_tokens})
        if total_tokens is not None
        else None,
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_handler_emits_redacted_content_never_writes_raw_text(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-redact")
    handler = AttackerAuditHandler(writer, "scan-redact")
    handler.set_context(round=1, module_id="prompt_injection", case_id="C1", role="strategist")

    handler.on_llm_end(_chat_result(f"leaked: {CANARY_CREDENTIAL_LITERAL}"), run_id=uuid4())
    writer.close()

    lines = audit_path_for(tmp_path, "scan-redact").read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert CANARY_CREDENTIAL_LITERAL not in parsed["content"]
    assert "REDACTED" in parsed["content"]
    assert parsed["direction"] == "inbound"
    assert parsed["event"] == "model_end"
    assert parsed["round"] == 1
    assert parsed["module_id"] == "prompt_injection"
    assert parsed["case_id"] == "C1"
    assert parsed["agent"] == "strategist"
    assert parsed["role"] == "strategist"


def test_handler_on_chat_model_start_emits_outbound_line(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-start")
    handler = AttackerAuditHandler(writer, "scan-start")
    handler.set_context(round=2, module_id="pii_exfiltration", case_id="C2", role="mutator")

    handler.on_chat_model_start(
        {}, [[AIMessage(content="please mutate this payload")]], run_id=uuid4()
    )
    writer.close()

    lines = audit_path_for(tmp_path, "scan-start").read_text().splitlines()
    parsed = json.loads(lines[0])
    assert parsed["direction"] == "outbound"
    assert parsed["event"] == "model_start"
    assert "please mutate this payload" in parsed["content"]


def test_handler_record_inter_agent_uses_inter_agent_direction_and_recipient(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-interagent")
    handler = AttackerAuditHandler(writer, "scan-interagent")
    handler.set_context(round=1, module_id="prompt_injection", case_id="C1")

    handler.record_inter_agent(
        from_role="strategist",
        to_role="mutator",
        content="Strategist selected technique DIRECT-001 for this round",
    )
    writer.close()

    lines = audit_path_for(tmp_path, "scan-interagent").read_text().splitlines()
    parsed = json.loads(lines[0])
    assert parsed["direction"] == "inter_agent"
    assert parsed["agent"] == "strategist"
    assert parsed["role"] == "strategist"
    assert parsed["recipient"] == "mutator"
    assert "DIRECT-001" in parsed["content"]


def test_handler_record_target_dispatch_uses_target_direction_and_own_case_id(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-target")
    handler = AttackerAuditHandler(writer, "scan-target")
    handler.set_context(round=1, module_id="prompt_injection", case_id="C1-current")

    handler.record_target_dispatch(
        case_id="C1-mut-1", content="target replied: refused", module_id="prompt_injection"
    )
    writer.close()

    lines = audit_path_for(tmp_path, "scan-target").read_text().splitlines()
    parsed = json.loads(lines[0])
    assert parsed["direction"] == "target"
    assert parsed["agent"] == "target"
    assert parsed["role"] == "target"
    # Explicit per-dispatch case_id, NOT the handler's "current" case_id.
    assert parsed["case_id"] == "C1-mut-1"


def test_handler_content_field_never_holds_raw_secret_even_via_record_target_dispatch(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-target-redact")
    handler = AttackerAuditHandler(writer, "scan-target-redact")
    handler.set_context(round=1, module_id="pii_exfiltration", case_id="C1")

    handler.record_target_dispatch(
        case_id="C1-mut-1", content=f"here is the secret: {CANARY_CREDENTIAL_LITERAL}"
    )
    writer.close()

    raw_bytes = audit_path_for(tmp_path, "scan-target-redact").read_bytes()
    assert CANARY_CREDENTIAL_LITERAL.encode() not in raw_bytes


# --- captured_events == written_lines structural guard ---------------------


def test_captured_events_equals_written_lines_after_a_scripted_campaign(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-campaign")
    handler = AttackerAuditHandler(writer, "scan-campaign")

    # Round 1: recon, strategist -> mutator handoff, one target dispatch.
    handler.set_context(round=0, module_id="prompt_injection", case_id="C1", role="recon")
    handler.on_chat_model_start({}, [[AIMessage(content="recon the target")]], run_id=uuid4())
    handler.on_llm_end(_chat_result("recon findings", total_tokens=100), run_id=uuid4())

    handler.set_context(role="strategist")
    handler.on_chat_model_start({}, [[AIMessage(content="pick a technique")]], run_id=uuid4())
    handler.on_llm_end(_chat_result("technique: DIRECT-001", total_tokens=80), run_id=uuid4())
    handler.record_inter_agent(
        from_role="strategist", to_role="mutator", content="use DIRECT-001"
    )

    handler.set_context(role="mutator")
    handler.on_chat_model_start({}, [[AIMessage(content="mutate DIRECT-001")]], run_id=uuid4())
    handler.on_llm_end(_chat_result("variant payload", total_tokens=60), run_id=uuid4())

    handler.record_target_dispatch(
        case_id="C1-mut-1", content="target response", module_id="prompt_injection", round=1
    )

    writer.close()

    assert handler.captured_events == handler.written_lines
    assert handler.capture_failures == 0
    lines = audit_path_for(tmp_path, "scan-campaign").read_text().splitlines()
    assert len(lines) == handler.written_lines


# --- Failure containment: a broken sink must never propagate ---------------


def test_a_hook_that_raises_internally_increments_capture_failures_and_does_not_propagate(
    tmp_path,
):
    writer = AttackerAuditWriter(tmp_path, "scan-broken")
    handler = AttackerAuditHandler(writer, "scan-broken")
    handler.set_context(round=1, module_id="prompt_injection", case_id="C1", role="strategist")

    # Close the writer out from under the handler -- writing after close
    # raises RuntimeError inside `_write_line()`. The hook must contain it.
    writer.close()

    before_failures = handler.capture_failures
    handler.on_llm_end(_chat_result("this line can never be written"), run_id=uuid4())

    assert handler.capture_failures == before_failures + 1
    assert handler.written_lines == 0
    # The captured attempt is still counted -- never silently dropped.
    assert handler.captured_events == 1


def test_record_inter_agent_after_close_also_contained(tmp_path):
    writer = AttackerAuditWriter(tmp_path, "scan-broken-2")
    handler = AttackerAuditHandler(writer, "scan-broken-2")
    writer.close()

    handler.record_inter_agent(from_role="strategist", to_role="mutator", content="handoff")

    assert handler.capture_failures == 1
    assert handler.written_lines == 0


# --- Whole-file byte-level canary assertion (the plan's headline claim) ----


def test_whole_file_never_contains_the_canary_credential_across_a_multi_round_campaign(
    tmp_path,
):
    """The plan's own headline assertion: run a scripted multi-round
    campaign whose scripted TARGET response contains the canary credential
    literal (the Analyst-quotes-target-response path D-86 is worried
    about), then read every byte of the produced JSONL and assert the
    literal is nowhere in it."""
    writer = AttackerAuditWriter(tmp_path, "scan-e2e")
    handler = AttackerAuditHandler(writer, "scan-e2e")

    for round_number in range(3):
        handler.set_context(
            round=round_number, module_id="pii_exfiltration", case_id="C1", role="strategist"
        )
        handler.on_chat_model_start(
            {}, [[AIMessage(content=f"round {round_number} brief")]], run_id=uuid4()
        )
        handler.on_llm_end(_chat_result(f"round {round_number} decision"), run_id=uuid4())

        handler.set_context(role="mutator")
        handler.record_inter_agent(
            from_role="strategist",
            to_role="mutator",
            # The Analyst-quotes-target-response path: an inter-agent
            # message carrying the target's (leaked) raw reply verbatim,
            # canary credential and all, BEFORE redaction.
            content=(
                f"Target leaked in round {round_number - 1}: {CANARY_CREDENTIAL_LITERAL}"
                if round_number > 0
                else "no prior target response yet"
            ),
        )

        handler.record_target_dispatch(
            case_id=f"C1-mut-{round_number}",
            content=f"Sure, here you go: {CANARY_CREDENTIAL_LITERAL}",
            module_id="pii_exfiltration",
            round=round_number,
        )

    writer.close()

    raw_bytes = audit_path_for(tmp_path, "scan-e2e").read_bytes()
    assert CANARY_CREDENTIAL_LITERAL.encode() not in raw_bytes
    assert handler.captured_events == handler.written_lines
    assert handler.capture_failures == 0

    # Human-readability sanity check (Task 3's manual review will do the
    # real reading, but this proves the structural shape supports it):
    # inter-agent lines are present, not only target-facing ones.
    lines = audit_path_for(tmp_path, "scan-e2e").read_text().splitlines()
    directions = {json.loads(line)["direction"] for line in lines}
    assert "inter_agent" in directions
    assert "target" in directions
