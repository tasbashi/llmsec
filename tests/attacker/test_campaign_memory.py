"""Tests for `llmsec.attacker.memory` -- the D-71 bounded campaign memory
eviction policy: `remember_refusal_signature()`, `mark_technique_dead()`,
`mark_partial_movement()`, `evict_to_cap()`, `render_memory_brief()`.
"""

from __future__ import annotations

import inspect

from llmsec.attacker.memory import (
    MAX_BRIEF_CHARS,
    evict_to_cap,
    mark_partial_movement,
    mark_technique_dead,
    remember_refusal_signature,
    render_memory_brief,
)
from llmsec.attacker.state import MEMORY_CAP, CampaignMemory, new_campaign_memory


def _memory(**overrides) -> CampaignMemory:
    base = new_campaign_memory()
    base.update(overrides)
    return base


# --- remember_refusal_signature -------------------------------------------


def test_remember_refusal_signature_adds_new_entry():
    memory = new_campaign_memory()
    updated = remember_refusal_signature(memory, "polite decline citing policy")
    assert updated["refusal_signatures"] == ["polite decline citing policy"]


def test_remember_refusal_signature_duplicate_does_not_grow_list():
    memory = _memory(refusal_signatures=["polite decline"])
    updated = remember_refusal_signature(memory, "polite decline")
    assert updated["refusal_signatures"] == ["polite decline"]


def test_remember_refusal_signature_falsy_input_is_noop_never_raises():
    memory = new_campaign_memory()
    updated = remember_refusal_signature(memory, "")
    assert updated["refusal_signatures"] == []


def test_remember_refusal_signature_is_pure_does_not_mutate_input():
    memory = new_campaign_memory()
    remember_refusal_signature(memory, "some signature")
    assert memory["refusal_signatures"] == []


# --- mark_technique_dead ---------------------------------------------------


def test_mark_technique_dead_adds_to_dead_list():
    memory = new_campaign_memory()
    updated = mark_technique_dead(memory, "roleplay_jailbreak")
    assert updated["dead_techniques"] == ["roleplay_jailbreak"]


def test_mark_technique_dead_removes_from_partial_movement_if_present():
    memory = _memory(partial_movement_techniques=["roleplay_jailbreak", "other"])
    updated = mark_technique_dead(memory, "roleplay_jailbreak")
    assert "roleplay_jailbreak" not in updated["partial_movement_techniques"]
    assert updated["partial_movement_techniques"] == ["other"]
    assert updated["dead_techniques"] == ["roleplay_jailbreak"]


def test_mark_technique_dead_duplicate_does_not_grow_list():
    memory = _memory(dead_techniques=["roleplay_jailbreak"])
    updated = mark_technique_dead(memory, "roleplay_jailbreak")
    assert updated["dead_techniques"] == ["roleplay_jailbreak"]


def test_mark_technique_dead_falsy_input_is_noop_never_raises():
    memory = new_campaign_memory()
    updated = mark_technique_dead(memory, "")
    assert updated["dead_techniques"] == []


def test_mark_technique_dead_is_pure_does_not_mutate_input():
    memory = _memory(partial_movement_techniques=["roleplay_jailbreak"])
    mark_technique_dead(memory, "roleplay_jailbreak")
    assert memory["partial_movement_techniques"] == ["roleplay_jailbreak"]
    assert memory["dead_techniques"] == []


# --- mark_partial_movement -------------------------------------------------


def test_mark_partial_movement_adds_to_partial_movement_list():
    memory = new_campaign_memory()
    updated = mark_partial_movement(memory, "instruction_override")
    assert updated["partial_movement_techniques"] == ["instruction_override"]


def test_mark_partial_movement_is_noop_for_technique_already_dead():
    memory = _memory(dead_techniques=["instruction_override"])
    updated = mark_partial_movement(memory, "instruction_override")
    assert updated["partial_movement_techniques"] == []
    assert updated["dead_techniques"] == ["instruction_override"]


def test_mark_partial_movement_duplicate_does_not_grow_list():
    memory = _memory(partial_movement_techniques=["instruction_override"])
    updated = mark_partial_movement(memory, "instruction_override")
    assert updated["partial_movement_techniques"] == ["instruction_override"]


def test_mark_partial_movement_falsy_input_is_noop_never_raises():
    memory = new_campaign_memory()
    updated = mark_partial_movement(memory, "")
    assert updated["partial_movement_techniques"] == []


def test_mark_partial_movement_is_pure_does_not_mutate_input():
    memory = new_campaign_memory()
    mark_partial_movement(memory, "instruction_override")
    assert memory["partial_movement_techniques"] == []


# --- Dead / partial-movement are mutually exclusive (round-trip) ----------


def test_technique_never_simultaneously_dead_and_partial_movement():
    memory = new_campaign_memory()
    memory = mark_partial_movement(memory, "leetspeak_obfuscation")
    memory = mark_technique_dead(memory, "leetspeak_obfuscation")
    assert "leetspeak_obfuscation" not in memory["partial_movement_techniques"]
    assert "leetspeak_obfuscation" in memory["dead_techniques"]

    # And the reverse order: once dead, marking partial movement is a no-op.
    memory2 = new_campaign_memory()
    memory2 = mark_technique_dead(memory2, "leetspeak_obfuscation")
    memory2 = mark_partial_movement(memory2, "leetspeak_obfuscation")
    assert "leetspeak_obfuscation" not in memory2["partial_movement_techniques"]
    assert "leetspeak_obfuscation" in memory2["dead_techniques"]


# --- evict_to_cap / MEMORY_CAP ---------------------------------------------


def test_evict_to_cap_trims_each_collection_independently_oldest_first():
    memory = _memory(
        refusal_signatures=[f"sig-{i}" for i in range(MEMORY_CAP + 3)],
        dead_techniques=[f"dead-{i}" for i in range(MEMORY_CAP + 1)],
        partial_movement_techniques=[f"partial-{i}" for i in range(MEMORY_CAP + 2)],
    )
    trimmed = evict_to_cap(memory)
    assert len(trimmed["refusal_signatures"]) == MEMORY_CAP
    assert len(trimmed["dead_techniques"]) == MEMORY_CAP
    assert len(trimmed["partial_movement_techniques"]) == MEMORY_CAP
    # Oldest-first eviction: the newest entries survive.
    assert trimmed["refusal_signatures"][0] == "sig-3"
    assert trimmed["refusal_signatures"][-1] == f"sig-{MEMORY_CAP + 2}"


def test_adding_over_500_entries_across_all_collections_keeps_each_at_cap():
    memory = new_campaign_memory()
    for i in range(200):
        memory = remember_refusal_signature(memory, f"refusal-signature-{i}")
    for i in range(200):
        memory = mark_technique_dead(memory, f"dead-technique-{i}")
    for i in range(150):
        memory = mark_partial_movement(memory, f"partial-technique-{i}")

    total_added = 200 + 200 + 150
    assert total_added > 500

    assert len(memory["refusal_signatures"]) == MEMORY_CAP
    assert len(memory["dead_techniques"]) == MEMORY_CAP
    assert len(memory["partial_movement_techniques"]) == MEMORY_CAP


# --- render_memory_brief ----------------------------------------------------


def test_render_memory_brief_contains_the_typed_fields():
    memory = _memory(
        refusal_signatures=["polite decline citing policy"],
        dead_techniques=["roleplay_jailbreak"],
        partial_movement_techniques=["instruction_override"],
    )
    brief = render_memory_brief(memory)
    assert "polite decline citing policy" in brief
    assert "roleplay_jailbreak" in brief
    assert "instruction_override" in brief


def test_render_memory_brief_bounded_length_flat_regardless_of_campaign_size():
    small_memory = new_campaign_memory()
    small_memory = remember_refusal_signature(small_memory, "one signature")

    big_memory = new_campaign_memory()
    for i in range(200):
        big_memory = remember_refusal_signature(big_memory, f"refusal-signature-number-{i}")
    for i in range(200):
        big_memory = mark_technique_dead(big_memory, f"dead-technique-number-{i}")
    for i in range(150):
        big_memory = mark_partial_movement(big_memory, f"partial-technique-number-{i}")

    assert len(render_memory_brief(small_memory)) <= MAX_BRIEF_CHARS
    assert len(render_memory_brief(big_memory)) <= MAX_BRIEF_CHARS
    # The bound is the SAME declared maximum for both, regardless of how
    # many entries the campaign accumulated -- flat per-call context.
    assert len(render_memory_brief(big_memory)) <= MAX_BRIEF_CHARS
    assert len(render_memory_brief(small_memory)) <= MAX_BRIEF_CHARS


def test_render_memory_brief_truncation_is_logged(caplog):
    import logging

    big_memory = new_campaign_memory()
    for i in range(200):
        big_memory = mark_partial_movement(big_memory, f"partial-technique-number-{i}")
    with caplog.at_level(logging.INFO, logger="llmsec.attacker.memory"):
        render_memory_brief(big_memory, max_chars=50)
    assert any("truncated" in record.message for record in caplog.records)


def test_render_memory_brief_never_exceeds_a_custom_max_chars():
    memory = _memory(partial_movement_techniques=[f"technique-{i}" for i in range(MEMORY_CAP)])
    brief = render_memory_brief(memory, max_chars=80)
    assert len(brief) <= 80


def test_render_memory_brief_prioritizes_partial_movement_when_truncated():
    """D-71: when the brief must be shortened, refusal signatures are cut
    before dead techniques, dead techniques before partial-movement
    techniques -- partial movement survives longest."""
    memory = _memory(
        partial_movement_techniques=["most-actionable-technique"],
        dead_techniques=[f"dead-{i}" for i in range(MEMORY_CAP)],
        refusal_signatures=[f"refusal-{i}" for i in range(MEMORY_CAP)],
    )
    brief = render_memory_brief(memory, max_chars=90)
    assert "most-actionable-technique" in brief
    assert len(brief) <= 90


def test_render_memory_brief_never_raises_on_empty_memory():
    brief = render_memory_brief(new_campaign_memory())
    assert isinstance(brief, str)
    assert len(brief) <= MAX_BRIEF_CHARS


# --- No framework file-backed scratch space usage --------------------------


def test_memory_module_never_uses_framework_scratch_space():
    import llmsec.attacker.memory as memory_mod

    source = inspect.getsource(memory_mod)
    for forbidden in ("read_file", "write_file", "StateBackend", "files["):
        assert forbidden not in source
