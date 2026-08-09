"""Tests for `llmsec.attacker.state` — `CampaignState`, the budget ledger,
and bounded campaign memory (D-71, D-75, D-77, D-80, D-81)."""

import json

import pytest

from llmsec.attacker.config import AttackerConfig, resolve_settings
from llmsec.attacker.state import (
    MEMORY_CAP,
    QUEUE_ELIGIBLE_VERDICTS,
    ROLE_NAMES,
    StrategistReasonCode,
    TerminationReason,
    add_memory_entry,
    is_full,
    new_campaign_memory,
    new_campaign_state,
    new_role_spend,
)
from llmsec.models import Verdict


def _settings():
    return resolve_settings(AttackerConfig(profile="standard"))


# --- new_campaign_state -------------------------------------------------


def test_new_campaign_state_shape():
    state = new_campaign_state(
        scan_id="scan-abc",
        settings=_settings(),
        module_order=["prompt_injection"],
        case_queue=[],
    )
    assert "budget_ledger" in state
    assert state["budget_ledger"]["spent_usd"] == 0.0
    assert state["budget_ledger"]["cap_usd"] == _settings().budget_usd
    assert state["round"] == 0


def test_new_campaign_state_json_round_trip():
    state = new_campaign_state(
        scan_id="scan-abc",
        settings=_settings(),
        module_order=["prompt_injection", "pii_exfiltration"],
        case_queue=[],
    )
    restored = json.loads(json.dumps(state))
    assert restored == state


# --- TerminationReason / StrategistReasonCode ----------------------------


def test_termination_reason_admits_exactly_six_values():
    import typing

    # 05-04/D-82 added WARN_APPROVAL_REFUSED (the operator explicitly
    # declined to approve continued spend past the warn threshold) -- a
    # sixth, distinct reason from BUDGET_CAP_EXCEEDED's own hard-cap trip.
    assert set(typing.get_args(TerminationReason)) == {
        "TECHNIQUES_EXHAUSTED",
        "TARGET_HARDENED",
        "LOW_YIELD",
        "ROUND_CAP_REACHED",
        "BUDGET_CAP_EXCEEDED",
        "WARN_APPROVAL_REFUSED",
    }


def test_strategist_reason_code_admits_exactly_three_values():
    import typing

    assert set(typing.get_args(StrategistReasonCode)) == {
        "TECHNIQUES_EXHAUSTED",
        "TARGET_HARDENED",
        "LOW_YIELD",
    }


# --- CampaignMemory / MEMORY_CAP ------------------------------------------


def test_new_campaign_memory_starts_empty_not_full():
    memory = new_campaign_memory()
    assert memory["refusal_signatures"] == []
    assert memory["dead_techniques"] == []
    assert memory["partial_movement_techniques"] == []
    assert is_full(memory) is False


def test_adding_more_than_cap_keeps_exactly_memory_cap():
    memory = new_campaign_memory()
    for i in range(MEMORY_CAP + 5):
        add_memory_entry(memory, "dead_techniques", f"technique-{i}")
    assert len(memory["dead_techniques"]) == MEMORY_CAP
    assert is_full(memory) is True
    # Oldest-first truncation: the earliest entries are the ones evicted.
    assert memory["dead_techniques"][0] == "technique-5"
    assert memory["dead_techniques"][-1] == f"technique-{MEMORY_CAP + 4}"


# --- QUEUE_ELIGIBLE_VERDICTS (D-77) ---------------------------------------


def test_queue_eligible_verdicts_exact_set():
    assert QUEUE_ELIGIBLE_VERDICTS == {Verdict.BLOCKED, Verdict.PARTIAL_LEAK}
    assert Verdict.UNCERTAIN not in QUEUE_ELIGIBLE_VERDICTS
    assert Verdict.FULL_COMPROMISE not in QUEUE_ELIGIBLE_VERDICTS


# --- new_role_spend / ROLE_NAMES -------------------------------------------


def test_new_role_spend_initializes_all_five_roles():
    spend = new_role_spend()
    assert set(spend.keys()) == set(ROLE_NAMES)
    for role in ROLE_NAMES:
        assert spend[role]["calls"] == 0
        assert spend[role]["usd"] == 0.0


def test_role_names_is_fixed_order_tuple():
    assert isinstance(ROLE_NAMES, tuple)
    assert not isinstance(ROLE_NAMES, set)
    assert ROLE_NAMES == ("recon", "strategist", "mutator", "analyst", "crescendo")


# --- No langchain/langgraph/deepagents import at module scope --------------


def test_state_module_importable_without_deep_extra():
    """Importing llmsec.attacker.state must not require the [deep] extra."""
    import llmsec.attacker.state  # noqa: F401 -- import success is the assertion


@pytest.mark.parametrize("module_name", ["langchain", "langgraph", "deepagents"])
def test_state_module_source_has_no_deep_extra_import(module_name):
    import inspect

    import llmsec.attacker.state as state_mod

    source = inspect.getsource(state_mod)
    assert f"import {module_name}" not in source
