"""The redacting checkpoint serializer and disk-backed saver (D-73
mitigation 2 / D-75), plus the two pure helpers `--resume` (Task 3) needs:
`config_fingerprint()` (D-75.2) and `idempotency_key()` (D-75.3).

**Implementation note -- corrects 05-06-PLAN.md's literal `<action>` text
(Rule 1 deviation, see `05-06-SUMMARY.md`).** The plan's `<action>` describes
decoding `super().dumps_typed(obj)`'s returned bytes as text, running the
redaction chokepoint over that text, then re-encoding. That text is
inherited from 05-RESEARCH.md's Wave 0 spike placeholder, which used a
FIXED LITERAL substring replacement ("keep the proof deterministic",
05-RESEARCH.md `### Pattern 3`) -- not the real, installed
`langgraph-checkpoint==4.1.1` serializer. Verified directly against that
installed package's source (`JsonPlusSerializer.dumps_typed()`,
`langgraph/checkpoint/serde/jsonplus.py`): for any non-`None`/`bytes`/
`bytearray` object -- which `CampaignState` always is -- it returns
`("msgpack", ormsgpack_encoded_bytes)`, i.e. length-prefixed MessagePack
binary, never JSON text. Decoding that as UTF-8, doing a substring
replacement, and re-encoding would corrupt msgpack's binary framing: a
string's length prefix would desync from its actual byte length the moment
a substitution changed that string's length, silently corrupting every
sibling value serialized after it in the same buffer -- a correctness bug,
not a style deviation, and not shippable.

**The fix implemented here instead: redact the PYTHON OBJECT TREE, before
`super().dumps_typed()` ever turns it into bytes** (`_redact_object_tree()`
below). This still satisfies every acceptance criterion 05-06-PLAN.md Task 2
names -- byte-level canary absence in the treatment artifact, presence in
the control, `build_checkpointer()` never yielding an unredacted disk-backed
saver, the CR-01 order (inherited unchanged from `redact_audit_text()`,
never re-implemented here), a stable `config_fingerprint()`, and distinct
`idempotency_key()`s -- it only changes the MECHANISM: redaction happens on
structured Python values (dicts/lists/pydantic models/strings), never on an
already-serialized byte buffer, so there is no format to corrupt.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel

# D-73 mitigation 2 / D-75 / AI-SPEC.md Common Pitfalls item 5: this env var
# must be set BEFORE `langgraph.checkpoint.serde._msgpack` is first imported
# in this process -- `STRICT_MSGPACK_ENABLED` there is a MODULE-LEVEL
# constant computed once, at that submodule's own import time
# (`os.getenv("LANGGRAPH_STRICT_MSGPACK", "false")`), not re-read on every
# `dumps_typed()`/`loads_typed()` call. `attacker/checkpoint.py` is the
# first (and, per `run_attacker_campaign()`'s deferred-import discipline,
# only) importer of `langgraph.checkpoint.serde.jsonplus` anywhere in the
# deep-mode import graph, so `setdefault()` here -- strictly before the
# `from langgraph...jsonplus import JsonPlusSerializer` line below --
# is the correct and sufficient place to set it. This restricts
# deserialization to `JsonPlusSerializer`'s own built-in safe-type
# allowlist (`GHSA-wwqv-p2pp-99h5`'s fix; the pinned `langgraph-checkpoint
# ==4.1.1` already exceeds the advisory's `>=3.0` floor regardless -- this
# flag is defense in depth on TOP of the version pin, addressing a
# DIFFERENT threat than redaction: a tampered/untrusted checkpoint file
# on disk, not a leaked secret written to one). `setdefault()` (never a
# blind overwrite) respects an operator or test harness that already set
# this env var to something else before this module was ever imported.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402

from llmsec.attacker.audit import redact_audit_text  # noqa: E402
from llmsec.attacker.config import (  # noqa: E402
    DEFAULT_ENABLED_TECHNIQUES,
    ResolvedAttackerSettings,
)
from llmsec.config import ScanConfig  # noqa: E402
from llmsec.models import EvalResult, TargetResponse, Verdict  # noqa: E402

logger = logging.getLogger(__name__)

# ASVS V6 / PITFALLS P10-D, mirroring `audit.py`'s own `_OUTPUT_DIR_MODE`:
# the checkpoint directory can hold redacted-but-still-sensitive campaign
# state, so it is created (and, if it already existed, TIGHTENED) with
# restrictive owner-only permissions.
_CHECKPOINT_DIR_MODE = 0o700

#: One shared SQLite file per configured `checkpoint_dir` -- distinct
#: campaigns are distinguished by LangGraph's own `thread_id` (set to
#: `scan_id` at every `.ainvoke()` call site), which is already the
#: checkpoints table's own partitioning key
#: (`PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)`), so one file
#: safely holds every campaign's checkpoints rather than needing a
#: per-scan filename.
_CHECKPOINT_DB_FILENAME = "attacker-checkpoints.sqlite"


# --- Redact-before-serialize: the object-tree walk --------------------------


def _redact_object_tree(obj: Any) -> Any:
    """Recursively redact every `str` LEAF reachable from `obj`, via the SAME
    `redact_audit_text()` chokepoint `audit.py` uses (D-86's
    `redact_credential_match(redact_pii_match(text))`, CR-01 order),
    returning a NEW object tree -- `obj` and everything reachable from it is
    left untouched. Checkpoint serialization must never mutate the live
    graph state it is only supposed to be reading a snapshot of.

    Handles every shape `CampaignState` and its Phase 1-4 model values
    (`EvalResult`, `TargetResponse`, `Verdict`, `VariantRecord`/`QueuedCase`
    -- both plain `TypedDict`s, i.e. ordinary `dict`s at runtime) can
    actually contain: `dict`, `list`, `tuple`, pydantic `BaseModel`, plain
    `dataclass`, `Enum` member, and primitive/`None`.

    `Enum` members are checked BEFORE the `str` case and returned
    UNCHANGED. `Verdict` (`class Verdict(str, Enum)`, `models.py`)
    subclasses `str` -- without this ordering, a `Verdict.BLOCKED` member
    reachable from `state["dispatch_results"][*]["eval_result"].verdict`
    would silently degrade to a plain `str` here on its way into
    `super().dumps_typed()`, and `model_copy(update=...)` below (which does
    NOT re-validate) would then hold that degraded plain string in the
    redacted twin. Since `Verdict`'s four members are a closed, fixed
    vocabulary (`blocked`/`partial_leak`/`full_compromise`/`uncertain`),
    never free text a secret could hide inside, there is nothing to gain
    from redacting them and real type-fidelity to lose by doing so.

    Only leaf VALUES are redacted -- dict KEYS are always structural field
    names across this codebase's `CampaignState`/model schemas (e.g.
    `per_role`'s role names, a `Checkpoint` envelope's `channel_values`
    channel names), never arbitrary target-derived text, so they are never
    passed through the chokepoint.
    """
    if isinstance(obj, Enum):
        return obj
    if isinstance(obj, str):
        return redact_audit_text(obj)
    if isinstance(obj, BaseModel):
        updates = {name: _redact_object_tree(getattr(obj, name)) for name in type(obj).model_fields}
        return obj.model_copy(update=updates)
    if isinstance(obj, dict):
        return {key: _redact_object_tree(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_redact_object_tree(value) for value in obj)
    if isinstance(obj, list):
        return [_redact_object_tree(value) for value in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                field.name: _redact_object_tree(getattr(obj, field.name))
                for field in dataclasses.fields(obj)
            },
        )
    return obj


#: The `langgraph.checkpoint.base.Checkpoint` TypedDict's own key set
#: (`v`, `id`, `ts`, `channel_values`, `channel_versions`, `versions_seen`,
#: `updated_channels`) -- used ONLY to detect the full-envelope
#: `dumps_typed()` call shape (see `_is_checkpoint_envelope()`), never
#: imported as the `Checkpoint` type itself (this module stays free of a
#: hard dependency on that exact TypedDict's shape surviving unchanged
#: across a `langgraph-checkpoint` version bump).
_CHECKPOINT_ENVELOPE_KEYS: frozenset[str] = frozenset({"id", "ts", "channel_values"})


def _is_checkpoint_envelope(obj: Any) -> bool:
    """True if `obj` is shaped like a full `Checkpoint` envelope (has AT
    LEAST `id`/`ts`/`channel_values`) rather than a single channel's raw
    value. A plain, duck-typed structural check rather than an `isinstance`
    against `langgraph.checkpoint.base.Checkpoint` -- `Checkpoint` is a
    `TypedDict`, which is not `isinstance`-checkable at runtime."""
    return isinstance(obj, dict) and _CHECKPOINT_ENVELOPE_KEYS.issubset(obj.keys())


#: 05-06 Task 3 (Rule 1 deviation, see `05-06-SUMMARY.md`): `CampaignState`
#: top-level fields that are STRUCTURAL identifiers this codebase itself
#: generates -- never target-derived content -- and whose exact value
#: round-tripping unmangled is load-bearing: `config_fingerprint` (D-75.2)
#: is compared for EXACT equality on every `--resume`, and `scan_id` is
#: `uuid.uuid4().hex` (32 lowercase hex chars). Discovered the hard way,
#: the SAME bug class as `_is_checkpoint_envelope()`'s docstring above:
#: `redact_credential_match()`'s generic `[A-Za-z0-9_-]{32,}` catch-all
#: matches BOTH a 32-char hex `scan_id` and a 64-char hex SHA-256
#: `config_fingerprint` just as readily as it matches a real secret, and
#: masking either would break `--resume`'s own fingerprint-equality check
#: (a redacted fingerprint can never equal a freshly-recomputed one) or
#: silently rename the campaign's own id. Exempted by TOP-LEVEL KEY NAME,
#: never by shape/pattern (a value-shape exemption would risk exempting a
#: REAL secret that happened to look similar) -- scoped to exactly the two
#: fields this module itself can prove are never target-derived.
_STRUCTURAL_STATE_KEYS: frozenset[str] = frozenset({"scan_id", "config_fingerprint"})


def _redact_channel_values(channel_values: Any) -> Any:
    """Redact a `Checkpoint` envelope's `channel_values` dict (i.e. the
    live `CampaignState`) field-by-field, leaving `_STRUCTURAL_STATE_KEYS`
    verbatim and passing every other field through `_redact_object_tree()`
    exactly as before. Falls back to a plain `_redact_object_tree()` call
    (no key-based exemption) if `channel_values` is not a dict -- a
    defensive no-op branch that should not be reachable given
    `Checkpoint.channel_values`'s own `dict[str, Any]` type, kept only so
    this function never raises on an unexpected shape.
    """
    if not isinstance(channel_values, dict):
        return _redact_object_tree(channel_values)
    return {
        key: (value if key in _STRUCTURAL_STATE_KEYS else _redact_object_tree(value))
        for key, value in channel_values.items()
    }


#: 05-06 Task 2 (Rule 1 deviation, see `05-06-SUMMARY.md`): every
#: project-defined type that can reach `CampaignState` and therefore this
#: serializer -- `Verdict` (`state["dispatch_results"][*]["eval_result"]
#: .verdict`), `EvalResult` (`dispatch_results[*]["eval_result"]`),
#: `TargetResponse` (`dispatch_results[*]["target_response"]`). Passed as
#: `allowed_msgpack_modules` at construction (below) so
#: `LANGGRAPH_STRICT_MSGPACK=true` (set at this module's top) does not
#: silently degrade these three types to plain dicts/strings on
#: `loads_typed()` -- verified directly against the installed
#: `langgraph-checkpoint==4.1.1` behavior: strict mode blocks
#: reconstruction of any type NOT in its safe-built-ins list
#: (`langgraph.checkpoint.serde._msgpack.SAFE_MSGPACK_TYPES`, e.g.
#: `datetime`/`uuid`/`langchain_core.messages.*`) UNLESS the caller passes
#: an explicit `allowed_msgpack_modules`, in which case that PLUS the
#: built-ins are honored and everything else stays blocked. Without this
#: allowlist, every `--resume`d campaign's restored `dispatch_results`
#: entries would silently lose their `EvalResult`/`TargetResponse`/
#: `Verdict` typing (degrading to a raw `dict`/`str`), corrupting
#: `CampaignResult.eval_results`' contract with `api.py`/the reporters for
#: any dispatch that happened before the resume point -- a real
#: correctness bug the plan's own `<action>` text does not mention, caught
#: by direct experimentation against the installed package rather than
#: assumed from the CVE advisory summary alone.
_PROJECT_MSGPACK_ALLOWLIST: tuple[type, ...] = (Verdict, EvalResult, TargetResponse)


class RedactingJsonPlusSerializer(JsonPlusSerializer):
    """D-73 mitigation 2 / D-75: the single place persisted graph state is
    redacted -- `dumps_typed()` redacts the PYTHON OBJECT TREE (see module
    docstring for why -- the base class's actual return shape is MessagePack
    binary, not JSON text, so a decode/substitute/re-encode pass over the
    returned bytes would corrupt that binary framing) BEFORE handing the
    redacted twin to `super().dumps_typed()`. This closes the
    unredacted-then-cleaned window a post-processing scrub of an
    already-written file would leave open -- a secret never reaches
    `super().dumps_typed()`, so it never reaches the bytes handed to the
    checkpointer, so it never reaches disk in the first place.

    Calls the SAME `redact_audit_text()` chokepoint `audit.py`'s own D-86
    no-exemption redaction path uses (`redact_credential_match(
    redact_pii_match(text))`, CR-01 order: the PII pass's structurally
    precise patterns run FIRST, the credential pass's generic 32+-char
    catch-all runs SECOND -- never inverted, see `redact_audit_text()`'s
    own docstring for the full rationale) -- reused directly rather than
    re-composing the two primitives a second time, so there remains exactly
    ONE place in this codebase the order is written down.

    `__init__` defaults `allowed_msgpack_modules` to
    `_PROJECT_MSGPACK_ALLOWLIST` (T-05-06-03: the STRICT_MSGPACK
    deserialization-safety threat, distinct from redaction) -- a caller may
    still override it explicitly, but every production call site in this
    module (`build_checkpointer()`) constructs this class with zero
    arguments and relies on this default.

    `loads_typed()` is inherited UNCHANGED from `JsonPlusSerializer` --
    redaction is a write-time-only transform; there is nothing to reverse
    on read, since the redacted bytes ARE the persisted record from the
    moment they were written.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("allowed_msgpack_modules", _PROJECT_MSGPACK_ALLOWLIST)
        super().__init__(**kwargs)

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if _is_checkpoint_envelope(obj):
            # `AsyncSqliteSaver.aput()` calls `dumps_typed()` on the FULL
            # `Checkpoint` envelope (`langgraph.checkpoint.base.Checkpoint`:
            # `v`/`id`/`ts`/`channel_values`/`channel_versions`/
            # `versions_seen`/`updated_channels`) -- redact ONLY
            # `channel_values` (where `CampaignState`, and therefore any
            # target-derived content, actually lives). Discovered the hard
            # way (Rule 1 deviation, see `05-06-SUMMARY.md`): an earlier
            # version of this method redacted the WHOLE envelope, and
            # `checkpoint["id"]` is a UUID6 hex string (36 chars of
            # `[A-Za-z0-9-]`) that `redact_credential_match()`'s generic
            # 32+-char catch-all pattern matches and masks -- corrupting
            # LangGraph's OWN internal checkpoint id on every write, which
            # then failed to hex-parse on the very next `.ainvoke()`
            # (`binascii.Error: Non-hexadecimal digit found` inside
            # LangGraph's `prepare_next_tasks()`). `id`/`ts`/
            # `channel_versions`/`versions_seen`/`v`/`updated_channels` are
            # LangGraph's own structural bookkeeping, never target-derived
            # content -- there is nothing to redact in them, and doing so
            # breaks the checkpointer itself.
            redacted_envelope = dict(obj)
            redacted_envelope["channel_values"] = _redact_channel_values(obj["channel_values"])
            return super().dumps_typed(redacted_envelope)
        # Every OTHER call shape (`AsyncSqliteSaver.aput_writes()` calls
        # `dumps_typed()` once per individual `(channel, value)` pending
        # write, never wrapped in a `Checkpoint` envelope) IS itself a
        # single `CampaignState` field's value -- e.g. `budget_ledger`,
        # `dispatch_results`, `variants` -- so the whole object is
        # redacted directly, exactly as before.
        return super().dumps_typed(_redact_object_tree(obj))


# --- The disk-backed saver, and its in-memory fallback ----------------------


@asynccontextmanager
async def build_checkpointer(
    settings: ResolvedAttackerSettings,
) -> AsyncIterator[BaseCheckpointSaver]:
    """Yield a checkpointer for `run_attacker_campaign()`'s
    `build_campaign_graph(checkpointer=...)` argument: a disk-backed
    `AsyncSqliteSaver` wrapping `settings.checkpoint_dir`, ALWAYS configured
    with `RedactingJsonPlusSerializer()` (no argument combination through
    this function can ever produce a disk-backed saver with a different,
    non-redacting `serde=`), when a checkpoint directory is configured --
    otherwise the framework's own `MemorySaver()` (unredacted, but nothing
    it holds ever reaches disk, so there is nothing to redact).

    An async context manager (not a plain factory function) because a
    disk-backed saver owns a live `aiosqlite.Connection` that must be
    closed on the way out, mirroring `AttackerAuditWriter`'s own
    `finally: writer.close()` discipline in `run_attacker_campaign()` --
    `async with build_checkpointer(settings) as checkpointer:` guarantees
    the connection is closed even if the campaign raises.

    `AsyncSqliteSaver` (not the sync `SqliteSaver`) is REQUIRED here:
    verified directly against the installed `langgraph-checkpoint-sqlite
    ==3.1.1` source that `SqliteSaver.aget_tuple()`/`aput()`/etc. all raise
    `NotImplementedError` ("This async method is not supported by the
    SqliteSaver class... consider using AsyncSqliteSaver") -- and
    `run_attacker_campaign()`'s `compiled.ainvoke(...)` calls exclusively
    exercise the ASYNC checkpointer surface.

    The pinned `langgraph-checkpoint==4.1.1` sits above the fix floor
    (`>=3.0`) `GHSA-wwqv-p2pp-99h5` (a deserialization RCE in
    `JsonPlusSerializer`'s legacy `"json"` mode) requires -- recorded here,
    not because this function's own code depends on that CVE's mechanics,
    but because it is the OTHER half of why a checkpoint file written to
    and read back from a user's disk needs two independent, unrelated
    protections: this pin (plus the module-level
    `LANGGRAPH_STRICT_MSGPACK=true` set above) addresses safe
    DESERIALIZATION of a tampered file (Tampering); `RedactingJsonPlus
    Serializer` addresses the file never holding a secret to begin with
    (Information Disclosure). Neither substitutes for the other.
    """
    if not settings.checkpoint_dir:
        yield MemorySaver()
        return

    # Deferred import: `aiosqlite`/`AsyncSqliteSaver` are `[deep]`-extra-only
    # dependencies -- this function is only ever reached from
    # `run_attacker_campaign()`'s own deferred-import block, after
    # `require_deep_extra()` has already cleared, but importing them at this
    # function's own top keeps that guarantee local and explicit rather than
    # relying on caller discipline alone.
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True, mode=_CHECKPOINT_DIR_MODE)
    os.chmod(checkpoint_dir, _CHECKPOINT_DIR_MODE)
    db_path = checkpoint_dir / _CHECKPOINT_DB_FILENAME

    async with aiosqlite.connect(db_path) as conn:
        saver = AsyncSqliteSaver(conn, serde=RedactingJsonPlusSerializer())
        await saver.setup()
        yield saver


# --- D-75.2: config fingerprint ---------------------------------------------


def config_fingerprint(config: ScanConfig, settings: ResolvedAttackerSettings) -> str:
    """A stable digest over exactly the fields that would make a resumed
    report describe a configuration that never ran (D-75.2): the cap, the
    call ceiling, the round cap, the variants-per-round, the attacker
    model, the enabled technique allowlist, the profile, and the sorted
    enabled-module list.

    Deliberately EXCLUDES anything volatile -- timestamps, `scan_id` --
    so an unchanged configuration always fingerprints identically across
    two separate calls (a fresh campaign start and a later `--resume`).

    The "sorted enabled-module list" is `config.enabled_modules` (D-75.2's
    own wording) -- `ScanConfig`'s own configured allowlist, the same
    field `list_modules_cmd`/`load_allowed()` treat as authoritative --
    rather than a dynamically re-discovered `uses_attacker_llm`-eligible
    subset, since this function's signature is `(config, settings)`, with
    no `modules: dict[str, BaseModule]` to re-discover eligibility from,
    and re-deriving that here would duplicate `run_attacker_campaign()`'s
    OWN D-77/D-78 filter under a different name rather than reusing it.
    """
    attacker_cfg = config.attacker
    if attacker_cfg is None:
        raise RuntimeError("config_fingerprint() called without config.attacker set")

    enabled_techniques = sorted(
        attacker_cfg.enabled_techniques
        if attacker_cfg.enabled_techniques is not None
        else DEFAULT_ENABLED_TECHNIQUES
    )
    fingerprint_fields: dict[str, Any] = {
        "budget_usd": settings.budget_usd,
        "agent_call_ceiling": settings.agent_call_ceiling,
        "max_rounds": settings.max_rounds,
        "variants_per_round": settings.variants_per_round,
        "model": attacker_cfg.model,
        "enabled_techniques": enabled_techniques,
        "profile": attacker_cfg.profile,
        "enabled_modules": sorted(config.enabled_modules),
    }
    canonical = json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- D-75.3: per-dispatch idempotency key -----------------------------------


def idempotency_key(case_id: str, round_index: int, variant_index: int) -> str:
    """The per-`(case, round, variant)` key D-75.3 requires, so `--resume`
    can recognize (and skip) a dispatch already recorded in the restored
    state -- a mid-round crash must never re-dispatch a paid-for payload or
    double-count its finding.

    A plain, stable, human-readable join (never a hash) -- distinctness
    across `(case_id, round_index, variant_index)` triples is exactly what
    the composite string already guarantees, and readability is a genuine
    debugging aid for anyone grepping a checkpoint/audit artifact by hand.
    """
    return f"{case_id}:{round_index}:{variant_index}"
