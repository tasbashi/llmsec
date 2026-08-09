"""The deep-mode attacker audit trail (D-84, D-85, D-86).

Two halves, built and proven in two separate tasks before either is wired
into the campaign graph:

1. **The artifact** (05-05 Task 1): `AuditLine` (the D-85 fixed field set),
   `audit_path_for()`, and `AttackerAuditWriter` -- an ordered, append-only,
   greppable `{scan_id}-attacker-audit.jsonl` mirroring
   `reporting/json_reporter.py`'s one-artifact-per-scan write discipline.
2. **The capture path** (this task): `redact_audit_text()` (the D-86
   no-exemption redaction chokepoint) and `AttackerAuditHandler` (the
   LangChain callback handler that observes every attacker exchange).

**No verbosity flag gates capture, deliberately (D-84).** Capture is
unconditional. The expensive deep-mode run an operator most wants to
inspect afterward is exactly the one where a `--verbose`-style flag was not
set going in -- gating capture behind a flag would make the audit trail
least available exactly when it is most needed. Every attacker message,
inter-agent traffic included, is captured because in a multi-agent team the
inter-agent traffic *is* the attack reasoning (D-84) -- logging only
target-facing payloads would leave a campaign undiagnosable.

**D-86: no redaction exemptions in this artifact, full stop.** `api.py`'s
report path calls a canary-literal exemption helper (`scoring/engine.py`)
so a planted canary PII value can be shown verbatim as proof of an echo,
while a real secret co-located in the same excerpt is still masked. This
module deliberately does NOT import that helper -- grep this file for its
name and find nothing. The Analyst (05-07) quotes the target's raw
response to its peers over the SAME `inter_agent` channel this module
audits, so a canary carve-out here would let a real secret riding alongside
an echoed canary straight through to a file on the operator's disk. A
redacted audit line is a little lossy for reproducing a canary-tier finding
byte-for-byte (the case can always be re-run); a leaked secret cannot be
un-leaked. See the `checkpoint:decision` between 05-05 Task 1 and Task 2 in
`05-05-PLAN.md`, resolved `no-exemptions`.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from llmsec.scoring.engine import redact_credential_match, redact_pii_match

logger = logging.getLogger(__name__)

# ASVS V6 / PITFALLS P10-D, mirroring `reporting/json_reporter.py`'s
# `_OUTPUT_DIR_MODE`: the audit artifact can carry redacted-but-still-
# sensitive campaign transcript, so the output directory is created (and,
# if it already existed, TIGHTENED) with restrictive owner-only
# permissions -- `Path.mkdir(mode=...)` only applies `mode` when the
# directory is actually created, so an unconditional `os.chmod()` follows
# every `mkdir()` call, exactly mirroring `JsonReporter.write()`'s WR-05
# fix for the identical gap.
_OUTPUT_DIR_MODE = 0o700

#: D-85's closed direction vocabulary. `inter_agent` is a first-class
#: citizen -- not an afterthought -- because in a team the inter-agent
#: traffic *is* the attack reasoning (D-84); logging only `target`-facing
#: payloads would leave a campaign undiagnosable.
AuditDirection = Literal["inbound", "outbound", "inter_agent", "target"]


def audit_path_for(output_dir: Path, scan_id: str) -> Path:
    """The single per-scan audit artifact path (D-85):
    `output_dir/{scan_id}-attacker-audit.jsonl`."""
    return output_dir / f"{scan_id}-attacker-audit.jsonl"


class AuditLine(BaseModel):
    """One line of the D-85 audit trail -- one JSON object per physical
    line, never split across lines regardless of `content`'s shape.

    `seq` is assigned (and any caller-supplied value overwritten) by
    `AttackerAuditWriter.write()` itself, which is the only place a
    strictly-increasing-with-no-gaps guarantee can actually be enforced
    across the writer's whole lifetime -- never trust a caller-supplied
    `seq`.
    """

    seq: int = Field(
        ..., description="Monotonically increasing line index, assigned by the writer."
    )
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the captured event.")
    scan_id: str
    #: The specific participant that produced/received this line. For a
    #: standard role this equals `role`; kept as a DISTINCT field (rather
    #: than collapsing into `role`) so a future multi-instance-per-role
    #: registry entry (D-65's "adding one in 5.1 is a prompt plus a
    #: registry entry") can populate a per-instance identifier here without
    #: a schema change.
    agent: str
    #: The `AgentRole` category (D-65/D-69): one of
    #: `attacker.state.ROLE_NAMES`, or `"target"` for a target-dispatch
    #: record.
    role: str
    direction: AuditDirection
    round: int
    module_id: str | None = None
    case_id: str | None = None
    event: str = Field(
        ...,
        description=(
            "e.g. model_start, model_end, model_error, inter_agent_handoff, "
            "target_dispatch."
        ),
    )
    #: Always passed through `redact_audit_text()` BEFORE construction --
    #: this field never holds raw, unredacted text (D-86).
    content: str
    cost_usd: float | None = Field(
        default=None,
        description="None (not 0.0) when no usage/pricing data was available for this call.",
    )
    tokens: int | None = Field(
        default=None,
        description="None (not 0) when no usage data was available for this call.",
    )
    #: Additive (beyond D-85's named field set): populated only for
    #: `direction="inter_agent"` lines, so a human reviewing the transcript
    #: can see who a handoff was addressed to without string-parsing
    #: `content`. Never required by any AT-3-style structural gate.
    recipient: str | None = None


class AttackerAuditWriter:
    """Owns one append-mode file handle for one scan's audit artifact, plus
    the monotonically increasing `seq` counter.

    Mirrors `JsonReporter.write()`'s directory-handling exactly: `mkdir()`
    with the restrictive mode, THEN an unconditional `os.chmod()` -- because
    `mkdir(exist_ok=True)` silently skips tightening a pre-existing
    directory's permissions.
    """

    def __init__(self, output_dir: Path, scan_id: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True, mode=_OUTPUT_DIR_MODE)
        os.chmod(output_dir, _OUTPUT_DIR_MODE)
        self.path: Path = audit_path_for(output_dir, scan_id)
        self._fh = self.path.open("a", encoding="utf-8")
        self._seq = 0
        self._closed = False

    def write(self, line: AuditLine) -> AuditLine:
        """Append exactly one JSON object, followed by exactly one newline.

        Raises `RuntimeError` if the writer has already been closed --
        writing after close is a bug in the caller and must be visible,
        never a silently dropped line.

        Deliberately serializes via `json.dumps(line.model_dump(mode="json"))`
        rather than `AuditLine.model_dump_json()`: pydantic-core's own JSON
        serializer validates UTF-8 encodability and RAISES
        `PydanticSerializationError` on a lone (unpaired) surrogate
        codepoint, which the `<behavior>` contract explicitly forbids
        (`content` must never cause the writer to raise). The stdlib
        `json` module's `ensure_ascii=True` (the default) escapes a lone
        surrogate as a literal `\\udNNN` sequence instead of erroring, and
        that escape round-trips byte-for-byte through `json.loads()` --
        still "the model's own JSON dump" in the sense that matters (built
        from `model_dump()`'s structured representation, never raw string
        concatenation), just routed through the more permissive stdlib
        serializer instead of pydantic-core's stricter one.
        """
        if self._closed:
            raise RuntimeError("AttackerAuditWriter is closed -- cannot write a line")
        stamped = line.model_copy(update={"seq": self._seq})
        self._seq += 1
        self._fh.write(json.dumps(stamped.model_dump(mode="json"), ensure_ascii=True))
        self._fh.write("\n")
        self._fh.flush()
        return stamped

    def close(self) -> None:
        """Idempotent: a second/third call is a harmless no-op."""
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "AttackerAuditWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- D-86 no-exemption redaction chokepoint --------------------------------


def redact_audit_text(text: str) -> str:
    """The D-86 no-exemption redaction chokepoint for the audit artifact.

    `redact_credential_match(redact_pii_match(text))` -- the SAME two
    primitives, in the SAME fixed order, `api.py` already calls at its own
    CR-01 call site: `redact_pii_match()`'s patterns (e.g. `_JWT_RE`) are
    structurally precise (dot-delimited segments, anchored shapes), while
    `redact_credential_match()`'s generic 32+-char catch-all has no `\\b`
    boundary and no dot in its character class. Running the generic pass
    FIRST could partially consume one segment of a larger dot-structured
    secret (a JWT) and insert a `***REDACTED***` marker whose `*`
    characters break the dot-structure the precise pass needs to match the
    remaining segments -- leaving real payload/signature content
    unredacted. Running the structurally precise pass first, then the
    generic catch-all second, ensures the generic pass can only ever mask
    what the precise pass left behind, never destroy its ability to match.
    Never invert this order (mirrors 03-REVIEW.md CR-01).

    Deliberately does NOT call the report path's canary-literal exemption
    helper (`scoring/engine.py`) -- D-86 states there is no exemption path
    for this artifact, even for a planted canary literal, because the
    Analyst quotes the target's raw
    response to its peers over the SAME `inter_agent` channel this module
    audits (a canary-literal carve-out here would let a real secret riding
    alongside an echoed canary straight through to a file on the
    operator's disk). See the module docstring and the `checkpoint:decision`
    in `05-05-PLAN.md`.

    Returns `text` unchanged (never raises) when `text` is falsy, mirroring
    `redact_pii_match()`/`redact_credential_match()`'s own empty-input
    contract.
    """
    if not text:
        return text
    return redact_credential_match(redact_pii_match(text))


# --- The callback handler: every attacker exchange, captured -------------


def _flatten_chat_messages(messages: list[list[Any]]) -> str:
    """Join every message batch `on_chat_model_start` is invoked with into
    one human-readable string -- one line per message, in call order.
    Tolerant of any message-like object exposing a `.content` attribute
    (never assumes a specific `BaseMessage` subclass)."""
    parts: list[str] = []
    for batch in messages:
        for message in batch:
            parts.append(str(getattr(message, "content", "") or ""))
    return "\n".join(parts)


def _extract_llm_result_payload(response: Any) -> tuple[str, int | None, float | None]:
    """Best-effort extraction of the completed call's text content, total
    token count, and dollar cost from an `LLMResult`-shaped `response`.

    Tolerant of a response shape that omits usage metadata entirely --
    returns `None` (never a guessed `0`/`0.0`) for `tokens`/`cost_usd` when
    genuinely absent, so an unpriced call stays distinguishable from a free
    one, mirroring `attacker/budget.py`'s `attacker_call_cost()`
    discipline. Any attribute-access failure surfaces to the caller (this
    function never swallows an exception itself) -- the caller
    (`AttackerAuditHandler`'s hooks, via `_safe_capture()`) is the single
    place that contains and records a capture failure.
    """
    content = ""
    tokens: int | None = None
    cost_usd: float | None = None
    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        generation = generations[0][0]
        message = getattr(generation, "message", None)
        if message is not None:
            content = str(getattr(message, "content", "") or "")
            usage_metadata = getattr(message, "usage_metadata", None)
            if usage_metadata:
                tokens = usage_metadata.get("total_tokens")
            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, dict):
                cost_usd = response_metadata.get("response_cost")
        else:
            content = str(getattr(generation, "text", "") or "")
    return content, tokens, cost_usd


class AttackerAuditHandler(BaseCallbackHandler):
    """The D-84/D-85/D-86 callback handler: one instance attached at the
    top-level graph invocation and explicitly forwarded into every nested
    role invocation's `config` (05-RESEARCH Pitfall 6 -- ambient
    contextvar callback propagation is real but must never be the only
    path an audit callback reaches a nested role invocation), so nothing
    bypasses it even if a future refactor breaks ambient propagation.

    Holds one `AttackerAuditWriter` plus mutable current-context fields
    (`round`, `module_id`, `case_id`, `current_role`) the campaign updates
    via `set_context()` as it advances -- this ONE handler instance is
    shared across every role for the whole campaign, so the model-call
    hooks below always stamp a line with whatever context was most
    recently set, never a value threaded through LangChain's own
    `metadata`/`tags` kwargs.

    Every line this handler writes -- model-start, model-end, model-error,
    inter-agent handoff, and target dispatch alike -- has its `content`
    passed through `redact_audit_text()` (D-86, no exemptions) BEFORE the
    `AuditLine` is constructed; there is no code path in this class that
    writes raw text and cleans it up afterward.

    Every public entry point (the LangChain hooks and the two explicit
    `record_*` methods) routes through `_safe_capture()`, which contains
    any exception raised while building/writing a line, records it via the
    `capture_failures` counter and a logged exception, and returns --
    never letting a broken audit sink propagate into (and cancel) the
    graph run (mirrors `attacker/budget.py`'s "never a bare callback
    handler" instinct, but for a DIFFERENT failure mode: THIS handler's
    OWN body must never raise, not the abort-semantics `budget.py` is
    talking about).

    `captured_events` and `written_lines` are incremented together, in the
    same successful path, so they stay equal by construction whenever no
    capture failure occurs -- 05-10's AT-3 structural gate re-derives this
    same equality independently from the persisted file, and a future
    refactor that broke ambient config propagation (05-RESEARCH Pitfall 6)
    would show up there as a handler that was never invoked at all for a
    role's calls, not as a mismatch between these two counters.
    """

    def __init__(self, writer: AttackerAuditWriter, scan_id: str) -> None:
        super().__init__()
        self._writer = writer
        self.scan_id = scan_id

        #: Mutable current-context fields the graph updates (via
        #: `set_context()`) as the campaign advances. `current_role` is
        #: additive beyond D-85's named `AuditLine` field set: it is what
        #: lets ONE shared handler instance correctly stamp `agent`/`role`
        #: on a model-start/model-end/model-error line without threading
        #: anything through LangChain's own `metadata` kwarg.
        self.round: int = 0
        self.module_id: str | None = None
        self.case_id: str | None = None
        self.current_role: str = "unknown"

        #: 05-10 AT-3's structural guard: `captured_events == written_lines`
        #: after a healthy campaign (see class docstring).
        self.captured_events: int = 0
        self.written_lines: int = 0
        #: Incremented, never silently dropped, whenever a hook body raises
        #: internally -- the visible signal a broken sink leaves behind.
        self.capture_failures: int = 0

    def set_context(
        self,
        *,
        round: int | None = None,
        module_id: str | None = None,
        case_id: str | None = None,
        role: str | None = None,
    ) -> None:
        """Update whichever current-context fields are given (a `None`
        argument leaves that field unchanged) -- the graph calls this
        immediately before invoking a role's model so every line captured
        during that invocation carries the right `round`/`module_id`/
        `case_id`/`agent`/`role`."""
        if round is not None:
            self.round = round
        if module_id is not None:
            self.module_id = module_id
        if case_id is not None:
            self.case_id = case_id
        if role is not None:
            self.current_role = role

    def _safe_capture(self, do: Callable[[], None]) -> None:
        """The one place every public entry point routes through: counts
        the attempt in `captured_events` FIRST (so an event is counted as
        "observed" even if building/writing its line then fails), runs
        `do()`, and contains any exception it raises -- recording it via
        `capture_failures` and a logged exception, never re-raising."""
        self.captured_events += 1
        try:
            do()
        except Exception:  # noqa: BLE001 -- a broken audit sink must never cancel a campaign
            self.capture_failures += 1
            logger.exception("AttackerAuditHandler: failed to capture an audit event")

    def _write_line(
        self,
        *,
        agent: str,
        role: str,
        direction: AuditDirection,
        event: str,
        content: str,
        recipient: str | None = None,
        cost_usd: float | None = None,
        tokens: int | None = None,
        round: int | None = None,
        module_id: str | None = None,
        case_id: str | None = None,
    ) -> None:
        """Build one `AuditLine` -- `content` passed through
        `redact_audit_text()` BEFORE construction, per-call `round`/
        `module_id`/`case_id` overrides falling back to the current-context
        fields when not given (a target dispatch touches a different
        `case_id` per variant within the SAME round, so it always passes
        its own explicit `case_id`) -- write it, and bump `written_lines`.
        Never wrapped in its own try/except: the caller (always a
        `_safe_capture()`-wrapped closure) is what contains a failure.
        """
        line = AuditLine(
            seq=0,  # overwritten by AttackerAuditWriter.write()
            timestamp=_now_iso(),
            scan_id=self.scan_id,
            agent=agent,
            role=role,
            direction=direction,
            round=round if round is not None else self.round,
            module_id=module_id if module_id is not None else self.module_id,
            case_id=case_id if case_id is not None else self.case_id,
            event=event,
            content=redact_audit_text(content),
            cost_usd=cost_usd,
            tokens=tokens,
            recipient=recipient,
        )
        self._writer.write(line)
        self.written_lines += 1

    # --- LangChain callback hooks: model-start / model-end / model-error --
    #
    # Every role in this package is built exclusively on a `BaseChatModel`
    # (never the legacy string-prompt `BaseLLM`), so `on_chat_model_start`
    # is the hook that actually fires for a role's model call -- `on_llm_start`
    # is deliberately left at `BaseCallbackHandler`'s own default (which
    # raises `NotImplementedError`, the documented signal LangChain's
    # dispatcher uses to fall back -- never exercised here since
    # `on_chat_model_start` always succeeds or is itself contained).

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        def _do() -> None:
            role = self.current_role
            content = _flatten_chat_messages(messages)
            self._write_line(
                agent=role, role=role, direction="outbound", event="model_start", content=content
            )

        self._safe_capture(_do)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        def _do() -> None:
            role = self.current_role
            content, tokens, cost_usd = _extract_llm_result_payload(response)
            self._write_line(
                agent=role,
                role=role,
                direction="inbound",
                event="model_end",
                content=content,
                cost_usd=cost_usd,
                tokens=tokens,
            )

        self._safe_capture(_do)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        def _do() -> None:
            role = self.current_role
            self._write_line(
                agent=role, role=role, direction="inbound", event="model_error", content=str(error)
            )

        self._safe_capture(_do)

    # --- Explicit role-boundary / target-dispatch recording ---------------
    #
    # Neither event above is a LangChain model-call hook: a role-to-role
    # handoff and a target dispatch are graph-topology events, not model
    # invocations, so the graph calls these directly rather than relying
    # on any callback dispatch to observe them.

    def record_inter_agent(
        self,
        *,
        from_role: str,
        to_role: str,
        content: str,
        round: int | None = None,
        module_id: str | None = None,
        case_id: str | None = None,
    ) -> None:
        """Record one role-to-role handoff -- `direction="inter_agent"`
        (D-84: in a team, inter-agent traffic *is* the attack reasoning).
        `recipient` carries `to_role` so a human reading the transcript can
        see who a handoff was addressed to without string-parsing
        `content`."""

        def _do() -> None:
            self._write_line(
                agent=from_role,
                role=from_role,
                direction="inter_agent",
                event="inter_agent_handoff",
                content=content,
                recipient=to_role,
                round=round,
                module_id=module_id,
                case_id=case_id,
            )

        self._safe_capture(_do)

    def record_campaign_start(self, *, module_order: list[str]) -> None:
        """05-06 Task 1 (Rule 2 deviation -- see `05-06-SUMMARY.md`): record
        D-68's deterministic cross-module iteration order as ONE dedicated
        audit line at campaign start, so two runs of the identical
        configuration are provably comparable and the cross-module ordering
        effect never silently varies between runs.

        `direction="inter_agent"` is the closest fit in D-85's CLOSED
        vocabulary (`inbound`/`outbound`/`inter_agent`/`target`) for a
        campaign-level, non-single-model, non-target event; `agent=role=
        "system"` marks it as distinct from any real role's own traffic.
        Routes through `_safe_capture()` like every other public entry
        point, so this line is counted in `captured_events`/`written_lines`
        exactly like any other -- the whole-file line-count equality
        (05-10 AT-3) holds for this line too, not just role/target lines.
        """

        def _do() -> None:
            self._write_line(
                agent="system",
                role="system",
                direction="inter_agent",
                event="campaign_start",
                content=json.dumps({"module_order": list(module_order)}),
                round=0,
            )

        self._safe_capture(_do)

    def record_target_dispatch(
        self,
        *,
        case_id: str,
        content: str,
        module_id: str | None = None,
        round: int | None = None,
    ) -> None:
        """Record one target dispatch -- `direction="target"`. Always takes
        an explicit `case_id` (never falls back to `self.case_id` alone):
        `dispatch_variants_node` dispatches many variants with distinct
        `case_id`s within the same round, so relying on the single
        "current" context field would misattribute every line but one."""

        def _do() -> None:
            self._write_line(
                agent="target",
                role="target",
                direction="target",
                event="target_dispatch",
                content=content,
                round=round,
                module_id=module_id,
                case_id=case_id,
            )

        self._safe_capture(_do)
