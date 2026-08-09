"""`HttpAppAdapter` — talks to an arbitrary, operator-configured HTTP app.

D-09: request bodies are built from `body_template` via plain string
`.replace()` substitution, never a template engine (Jinja2/etc.) — this is a
deliberate anti-SSTI choice for a field that accepts arbitrary user-supplied
template strings from config (T-01-07). The same discipline extends to the
`{{session_id}}` token below (T-02-07).

Multi-turn session round-trip (D-12/D-15) is operator-configured, opt-in,
and deliberately not auto-detected: `supports_multi_turn` only becomes True
when the operator has configured both a session-id extraction path
(`session_id_path`) and a re-injection point (`session_id_header` or a
`{{session_id}}` token in `body_template`). A silent upgrade would let a
degraded run be reported as a genuine crescendo, which is the precise
honesty failure D-15 exists to prevent.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable

import httpx

from llmsec.adapters.base import TargetAdapter
from llmsec.models import TestCase, TargetResponse

logger = logging.getLogger(__name__)

_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")

# WR-02: matches both template tokens so `_send_request()` can substitute
# them in a single pass over the original `body_template` string, rather
# than via two sequential `.replace()` calls that could re-scan (and
# corrupt) a literal token already substituted in from attack-payload text.
_TOKEN_PATTERN = re.compile(r"\{\{payload\}\}|\{\{session_id\}\}")


def extract_path(data: dict, path: str) -> str:
    """Minimal dotted/bracket path resolver — NOT full JSONPath.

    Covers `"response"`, `"choices[0].message.content"`, etc. Escape hatch:
    swap in `jsonpath-ng` if a real target needs wildcards/filters.

    IN-05: `_PATH_TOKEN.finditer()` only matches well-formed key/bracket
    tokens — a malformed path (e.g. a leading `.` typo like `.response`
    instead of `response`, or a stray double `..`) leaves a gap that
    `finditer()` silently skips over rather than erroring on, so a typo in
    an operator-supplied `response_path`/`session_id_path` would otherwise
    degrade silently to "seems to work" instead of a clear config error.
    This walks the match positions and verifies each token starts exactly
    where the previous one ended (allowing exactly one `.` separator
    between tokens), raising `ValueError` on any unaccounted-for gap.
    """
    current = data
    pos = 0
    for match in _PATH_TOKEN.finditer(path):
        gap = path[pos : match.start()]
        # A leading gap (before the first token) must be empty — a leading
        # "." (e.g. ".response") is a typo, not a valid separator, since
        # there is no preceding token for it to separate. A gap between
        # two tokens may be empty (bracket immediately follows a key, e.g.
        # "choices[0]") or exactly one "." separator.
        allowed_gaps = ("",) if pos == 0 else ("", ".")
        if gap not in allowed_gaps:
            raise ValueError(
                f"Malformed path {path!r}: unexpected {gap!r} before position {match.start()}"
            )
        pos = match.end()
        key, idx = match.groups()
        current = current[key] if key is not None else current[int(idx)]
    trailing = path[pos:]
    if trailing:
        raise ValueError(f"Malformed path {path!r}: unexpected trailing {trailing!r}")
    if pos == 0:
        raise ValueError(f"Malformed path {path!r}: no valid tokens found")
    return str(current)


class HttpAppAdapter(TargetAdapter):
    """Sends `TestCase`s to an arbitrary HTTP app via a templated request."""

    def __init__(
        self,
        method: str,
        url: str,
        headers: dict,
        body_template: str,
        response_path: str,
        timeout: float = 30.0,
        session_id_path: str | None = None,
        session_id_header: str | None = None,
    ):
        self.method, self.url = method, url
        self.headers, self.body_template, self.response_path = headers, body_template, response_path
        self.session_id_path = session_id_path
        self.session_id_header = session_id_header
        # Multi-turn support activates only when an extraction path AND a
        # re-injection point are both configured — an extractable id with
        # nowhere to go is not multi-turn support (D-12/D-15/T-02-15).
        self.supports_multi_turn = bool(session_id_path) and bool(
            session_id_header or "{{session_id}}" in body_template
        )
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def _send_request(
        self, prompt: str, session_id: str | None
    ) -> tuple[str, str | None, int]:
        """Build and send one HTTP request.

        Returns `(response_text, next_session_id, status_code)`. Body
        building keeps the D-09 discipline exactly: plain string
        substitution, never a template engine, with both the payload and
        the session id escaped via the `json.dumps(value)[1:-1]` idiom
        before substitution. When no id is available yet, `{{session_id}}`
        is substituted with an empty string.

        WR-02: both tokens are substituted in a single pass over the
        original `body_template` (via `_TOKEN_PATTERN.sub()`), never via
        two sequential `.replace()` calls against the same growing string.
        Two sequential calls would let the second substitution re-scan (and
        potentially corrupt) literal token text that the first substitution
        already inserted from the attack payload — e.g. a payload that
        itself contains the literal substring `{{session_id}}`.
        """
        substitutions = {
            "{{payload}}": json.dumps(prompt)[1:-1],
            "{{session_id}}": json.dumps(session_id)[1:-1] if session_id else "",
        }
        body_str = _TOKEN_PATTERN.sub(lambda m: substitutions[m.group(0)], self.body_template)

        # A per-request copy — self.headers is shared across
        # concurrently-dispatched cases and must never be mutated.
        headers = dict(self.headers)
        if self.session_id_header and session_id:
            headers[self.session_id_header] = session_id

        resp = await self._client.request(self.method, self.url, headers=headers, content=body_str)

        # WR-01: a 429 (rate-limited) or 5xx response is a transient,
        # retryable failure shape exactly like a transport-level timeout —
        # `RETRYABLE_EXCEPTIONS` in `orchestrator.py` documents this
        # contract, but httpx never raises on a non-2xx status by itself.
        # Raise explicitly here so the orchestrator's exponential-backoff
        # retry policy actually engages instead of silently evaluating the
        # error body as an ordinary response.
        if resp.status_code == 429 or resp.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"Retryable status {resp.status_code} from target",
                request=resp.request,
                response=resp,
            )

        next_session_id = session_id
        # WR-01: `resp.json()` parsing and the `response_path` lookup are
        # kept in their own try so a `response_path` extraction failure on
        # an otherwise-valid JSON body cannot force `data` back to `None`
        # and silently suppress an independent, still-resolvable
        # `session_id_path` lookup below.
        try:
            data = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            # `httpx.Response.json()` runs `json.loads()` over the raw
            # response bytes, which raises `UnicodeDecodeError` (a
            # `ValueError` subclass, not `json.JSONDecodeError`) when those
            # bytes are not valid UTF-8 — e.g. a JSON-shaped body with an
            # embedded invalid byte sequence. Catching it here too lets a
            # non-UTF-8-but-otherwise-meaningful body fall through to the
            # `resp.text` fallback below instead of propagating uncaught
            # past `_send_request()` (WR-01 follow-up, 02-REVIEW.md).
            data = None

        if data is not None:
            try:
                text = extract_path(data, self.response_path) if self.response_path else json.dumps(data)
            except (KeyError, IndexError, TypeError):
                text = resp.text
        else:
            text = resp.text

        if self.session_id_path and data is not None:
            try:
                next_session_id = extract_path(data, self.session_id_path)
            except (KeyError, IndexError, TypeError):
                # A malformed/missing session id leaves the id unchanged
                # rather than raising, so the sequence continues (T-02-17).
                pass

        return text, next_session_id, resp.status_code

    async def send(self, case: TestCase) -> TargetResponse:
        start = time.monotonic()
        text, _next_session_id, status_code = await self._send_request(case.prompt, None)
        latency_ms = (time.monotonic() - start) * 1000
        return TargetResponse(
            case_id=case.case_id,
            raw_text=text,
            status_code=status_code,
            latency_ms=latency_ms,
        )

    async def send_conversation(
        self, case: TestCase, stop_when: Callable[[str], bool] | None = None
    ) -> TargetResponse:
        """Session/conversation-id round-trip across turns.

        Only runs when `supports_multi_turn` is True (both an extraction
        path and a re-injection point are configured); otherwise delegates
        to the ABC's honestly-labelled `multi_turn_concatenated` default —
        never fabricating a `multi_turn_real` label for an unconfigured
        adapter (D-15).
        """
        if not self.supports_multi_turn:
            return await super().send_conversation(case, stop_when)

        turns = case.turns or [case.prompt]
        turn_replies: list[str] = []
        transcript_parts: list[str] = []
        session_id: str | None = None
        status_code: int | None = None

        start = time.monotonic()
        for turn_number, turn_text in enumerate(turns, start=1):
            text, session_id, status_code = await self._send_request(turn_text, session_id)
            turn_replies.append(text)
            transcript_parts.append(f"Turn {turn_number} USER: {turn_text}\nTurn {turn_number} ASSISTANT: {text}")
            if stop_when is not None and stop_when(text):
                break
        latency_ms = (time.monotonic() - start) * 1000

        return TargetResponse(
            case_id=case.case_id,
            raw_text="\n\n".join(transcript_parts),
            status_code=status_code,
            latency_ms=latency_ms,
            transport_mode="multi_turn_real",
            turn_replies=turn_replies,
        )

    async def health_check(self) -> bool:
        try:
            r = await self._client.request(self.method, self.url, headers=self.headers, content="{}")
            return r.status_code < 500
        except Exception as exc:
            # IN-02 (02-REVIEW.md): this method's `-> bool` contract is "never
            # raise" -- a header-encoding error or misconfigured `method`
            # string is just as much a "target is unreachable/misconfigured"
            # signal as an `httpx.HTTPError`, so it must degrade to `False`
            # too rather than propagate past the caller's health-check gate.
            logger.debug("health_check() request failed: %s", exc)
            return False

    async def close(self) -> None:
        await self._client.aclose()
