"""Bounded manual structured-output retry (D-94 AT-6; 05-RESEARCH Pitfall 5).

LangGraph's built-in node-level retry mechanism does not fire on
`pydantic.ValidationError` (05-RESEARCH.md Pitfall 5) -- and, verified
empirically this session, DeepAgents' own default handling for a bare
`response_format=<Schema>` class retries internally on a validation
failure, bounded only by LangGraph's (much larger) graph recursion limit
rather than the 2-bounded-retry contract D-94's "valid structured output
per role" gate requires.

Every role factory in this package therefore passes
`response_format=ToolStrategy(<Schema>, handle_errors=False)`, not the
bare schema class. With `handle_errors=False`, a schema-invalid tool call
raises `langchain.agents.structured_output.StructuredOutputValidationError`
immediately after exactly one model call -- handing retry control entirely
to `invoke_role_with_retry()` below, never to LangGraph's built-in node
retry mechanism and never to DeepAgents' own unbounded internal
reflection loop.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from langchain.agents.structured_output import StructuredOutputValidationError
from pydantic import BaseModel, ValidationError

#: Bound to the specific role output schema at each call site (e.g.
#: `StrategistOutput`/`MutatorOutput`) via the caller's own annotated
#: assignment -- this function itself is schema-agnostic.
_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

logger = logging.getLogger(__name__)

#: D-94 gate: at most this many manual re-asks after the first attempt (3
#: attempts total) before a role's structured-output failure is surfaced as
#: a recorded structural failure -- never a silently skipped round.
MAX_STRUCTURED_OUTPUT_RETRIES = 2


class StructuredOutputFailure(RuntimeError):
    """Raised when a role's compiled agent fails to produce schema-valid
    structured output within `MAX_STRUCTURED_OUTPUT_RETRIES` manual
    retries.

    A recorded structural failure, never a silently skipped round (D-94
    AT-6) -- a silently-skipped round would look identical to a clean one
    under D-77's `UNCERTAIN`-exclusion logic. Callers must handle this
    explicitly, never let it propagate unhandled.
    """


async def invoke_role_with_retry(
    agent: Any,
    messages: list[Any],
    *,
    role: str,
    config: dict[str, Any] | None = None,
    on_attempt: Callable[[int, str, Any], None] | None = None,
    on_success: Callable[[dict[str, Any]], None] | None = None,
) -> _SchemaT:
    """Invoke `agent.ainvoke({"messages": messages}, config=config)` up to
    `MAX_STRUCTURED_OUTPUT_RETRIES + 1` times, returning the first
    schema-valid `structured_response`.

    `config` is forwarded EXPLICITLY into every attempt's `.ainvoke()` call
    (05-RESEARCH Pitfall 6 -- ambient contextvar callback propagation is
    real but must never be the only path an audit callback reaches a
    nested role invocation; explicit forwarding is defense in depth with
    no downside).

    `on_success`, when given, is called with the raw (still-`dict`-shaped)
    successful `result` -- 05-04's budget ledger reads `result["messages"]`'s
    final `AIMessage.usage_metadata` off of it to price the call, without
    this function's own return type ever changing shape.

    On a missing OR schema-invalid `structured_response`, calls
    `on_attempt(attempt_index, violation_text, raw_output)` (if given) and
    retries. After the bounded attempts, raises `StructuredOutputFailure`
    -- never returns `None` and never silently drops the round.
    """
    last_violation = "structured_response missing from role output"
    last_raw: Any = None
    for attempt in range(MAX_STRUCTURED_OUTPUT_RETRIES + 1):
        try:
            result = await agent.ainvoke({"messages": messages}, config=config)
        except (StructuredOutputValidationError, ValidationError) as exc:
            last_violation = str(exc)
            last_raw = None
            logger.warning(
                "Role %r structured-output attempt %d/%d failed schema validation: %s",
                role,
                attempt + 1,
                MAX_STRUCTURED_OUTPUT_RETRIES + 1,
                exc,
            )
            if on_attempt is not None:
                on_attempt(attempt, last_violation, last_raw)
            continue

        structured = result.get("structured_response") if isinstance(result, dict) else None
        if structured is not None:
            if on_success is not None:
                on_success(result)
            return structured

        last_violation = "structured_response missing from role output"
        last_raw = result
        logger.warning(
            "Role %r structured-output attempt %d/%d produced no structured_response",
            role,
            attempt + 1,
            MAX_STRUCTURED_OUTPUT_RETRIES + 1,
        )
        if on_attempt is not None:
            on_attempt(attempt, last_violation, last_raw)

    raise StructuredOutputFailure(
        f"Role {role!r} failed to produce valid structured output after "
        f"{MAX_STRUCTURED_OUTPUT_RETRIES + 1} attempts: {last_violation}"
    )
