"""Bounded campaign memory eviction policy (D-71).

`state.py` (05-02/05-06) owns the `CampaignMemory` SHAPE (three
`MEMORY_CAP`-bounded lists) and the simplest correct cap-enforcing
behavior (oldest-first truncation, via `add_memory_entry()`) so the cap is
never silently violated before this module exists. THIS module owns the
actual eviction POLICY: which entry is least worth keeping when a
collection is over cap, and how a whole campaign's accumulated memory is
compressed into a single bounded string a role's brief can afford to
include.

Bounded structure by design (D-71): the budget model this phase's cost
arithmetic rests on assumes FLAT per-call context as the campaign grows --
`render_memory_brief()`'s hard character ceiling is what keeps that
assumption honest at round 500 of a long campaign exactly as it does at
round 1. A growing raw transcript would invalidate the whole per-round
cost estimate (D-82).

The framework's own file-backed scratch space (DeepAgents' virtual
filesystem tools for reading/writing scratch documents) is DELIBERATELY
not used for campaign memory. That space IS state-backed and therefore
already covered by 05-06's checkpoint redaction chokepoint, but it is
unstructured content with no eviction policy of its own -- exactly what
D-71 rules out for campaign memory (05-RESEARCH.md `### Additional Wave 0
Questions Answered`). Every function below operates on the typed
`CampaignMemory` shape only; this file never references any virtual
filesystem tool or storage backend by name.

Every function here is PURE: it returns a NEW `CampaignMemory` value,
never mutating the one it was given, so a graph node can return the
result directly as part of its own state update (mirroring every other
node-return pattern in `graph.py`) with no hidden side channel. None
raise on empty/falsy input.
"""

from __future__ import annotations

import logging

from llmsec.attacker.state import MEMORY_CAP, CampaignMemory

logger = logging.getLogger(__name__)

#: D-71's documented cross-collection eviction priority: refusal
#: signatures are the least individually valuable entries (a campaign
#: typically converges on one or two distinct refusal shapes, so a
#: duplicate signature costs nothing to lose), dead techniques next, and
#: partial-movement techniques -- the most actionable signal the team
#: holds -- are trimmed last. `evict_to_cap()` processes collections in
#: this order; `render_memory_brief()` renders them in the REVERSE order
#: (most valuable first) so plain trailing truncation implements the same
#: priority when the rendered brief must be shortened to fit its bound.
_EVICTION_PRIORITY: tuple[str, ...] = (
    "refusal_signatures",
    "dead_techniques",
    "partial_movement_techniques",
)

#: Hard ceiling on `render_memory_brief()`'s own output, mirroring
#: `detection/judge.py`'s `MAX_RESPONSE_CHARS` bounded-input discipline --
#: this is the number that keeps a role's brief flat-cost regardless of
#: how long the campaign has run.
MAX_BRIEF_CHARS: int = 2000

_TRUNCATION_MARKER = " …[TRUNCATED]"


def _copy_memory(memory: CampaignMemory) -> CampaignMemory:
    """A fresh `CampaignMemory` with every list independently copied --
    the starting point for every pure operation below, so a caller's own
    `memory` value is never mutated by any function in this module."""
    return CampaignMemory(
        refusal_signatures=list(memory.get("refusal_signatures", [])),
        dead_techniques=list(memory.get("dead_techniques", [])),
        partial_movement_techniques=list(memory.get("partial_movement_techniques", [])),
    )


def evict_to_cap(memory: CampaignMemory) -> CampaignMemory:
    """Trim every `CampaignMemory` collection back to at most `MEMORY_CAP`
    entries, dropping the OLDEST entries within each collection that is
    over cap.

    Processes collections in `_EVICTION_PRIORITY`'s order
    (`refusal_signatures`, then `dead_techniques`, then
    `partial_movement_techniques`) -- documented explicitly per D-71's own
    priority ("refusal signatures are evicted before dead techniques, and
    dead techniques before partial-movement techniques, because partial
    movement is the most actionable signal the team holds"). Each
    collection is independently bounded at `MEMORY_CAP` (never a shared
    cross-collection budget), so this processing order is a discipline
    choice recorded for clarity, not a mechanism that changes any single
    collection's own outcome.

    Pure: returns a new `CampaignMemory`; never mutates `memory`.
    """
    new_memory = _copy_memory(memory)
    for key in _EVICTION_PRIORITY:
        bucket = new_memory[key]  # type: ignore[literal-required]
        if len(bucket) > MEMORY_CAP:
            del bucket[: len(bucket) - MEMORY_CAP]
    return new_memory


def remember_refusal_signature(memory: CampaignMemory, signature: str) -> CampaignMemory:
    """Add `signature` to `memory["refusal_signatures"]`.

    Adding a duplicate signature does not grow the list. Pure (returns an
    unchanged copy of `memory`) and a no-op for a falsy `signature` --
    never raises.
    """
    new_memory = _copy_memory(memory)
    if not signature:
        return new_memory
    if signature not in new_memory["refusal_signatures"]:
        new_memory["refusal_signatures"].append(signature)
    return evict_to_cap(new_memory)


def mark_technique_dead(memory: CampaignMemory, technique: str) -> CampaignMemory:
    """Add `technique` to `memory["dead_techniques"]`, first removing it
    from `memory["partial_movement_techniques"]` if present -- a
    technique is never simultaneously dead and promising.

    Pure and a no-op for a falsy `technique` -- never raises.
    """
    new_memory = _copy_memory(memory)
    if not technique:
        return new_memory
    if technique in new_memory["partial_movement_techniques"]:
        new_memory["partial_movement_techniques"].remove(technique)
    if technique not in new_memory["dead_techniques"]:
        new_memory["dead_techniques"].append(technique)
    return evict_to_cap(new_memory)


def mark_partial_movement(memory: CampaignMemory, technique: str) -> CampaignMemory:
    """Add `technique` to `memory["partial_movement_techniques"]`.

    A no-op for a technique already marked dead (`mark_technique_dead()`'s
    guarantee holds in both directions -- a technique already in
    `dead_techniques` is never also added to `partial_movement_techniques`).
    Pure and a no-op for a falsy `technique` -- never raises.
    """
    new_memory = _copy_memory(memory)
    if not technique:
        return new_memory
    if technique in new_memory["dead_techniques"]:
        return new_memory
    if technique not in new_memory["partial_movement_techniques"]:
        new_memory["partial_movement_techniques"].append(technique)
    return evict_to_cap(new_memory)


def render_memory_brief(memory: CampaignMemory, *, max_chars: int = MAX_BRIEF_CHARS) -> str:
    """Render `memory` into a bounded string for a role's brief.

    Length never exceeds `max_chars` regardless of how many entries were
    ever added to any collection -- the property that keeps per-call
    context flat as the campaign grows (D-71). Contains only the typed
    summary fields (`CampaignMemory`'s own three lists) -- never raw
    target response text; nothing in this function's signature or body
    ever touches a `TargetResponse`/`dispatch_results` value.

    Sections are rendered in `_EVICTION_PRIORITY`'s REVERSE order
    (partial-movement techniques first, dead techniques second, refusal
    signatures last), so that when the full rendering exceeds `max_chars`
    and must be shortened, plain trailing truncation naturally drops the
    least-valuable content first -- refusal signatures are cut before
    dead techniques, dead techniques before partial-movement techniques,
    mirroring `evict_to_cap()`'s own documented priority.

    Logs once, at INFO, whenever truncation actually occurs.
    """
    sections = (
        ("PARTIAL-MOVEMENT TECHNIQUES", memory.get("partial_movement_techniques", [])),
        ("DEAD TECHNIQUES", memory.get("dead_techniques", [])),
        ("KNOWN REFUSAL SIGNATURES", memory.get("refusal_signatures", [])),
    )
    full = "\n".join(
        f"{label} ({len(items)}): {', '.join(items) if items else '(none)'}"
        for label, items in sections
    )
    if len(full) <= max_chars:
        return full
    logger.info(
        "render_memory_brief truncated from %d to %d chars (max_chars=%d)",
        len(full),
        max_chars,
        max_chars,
    )
    return full[: max_chars - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
