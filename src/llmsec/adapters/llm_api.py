"""`LLMApiAdapter` — talks to a raw LLM API (OpenAI/Anthropic/etc.) via litellm.

Note the module reference `litellm.acompletion(...)` (not `from litellm import
acompletion`) is deliberate: `tests/conftest.py`'s `mock_litellm_acompletion`
fixture monkeypatches the `litellm` module's `acompletion` attribute, which
only intercepts calls resolved through the module object at call time.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import litellm

from llmsec.adapters.base import TargetAdapter
from llmsec.models import TestCase, TargetResponse


class LLMApiAdapter(TargetAdapter):
    """Sends `TestCase`s directly to a raw LLM API via litellm's unified interface."""

    supports_multi_turn = True
    supports_system_prompt_override = True

    def __init__(self, model: str, api_key_env: str, system_prompt: str | None = None):
        self.model = model
        # D-08: resolve the *literal* key from the environment here, at
        # construction time, and never persist it on any config/model field —
        # only the env-var name (`api_key_env`) is ever stored on ScanConfig.
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"Environment variable {api_key_env!r} is not set "
                "(referenced by config.target.api_key_env)"
            )
        self.system_prompt = system_prompt

    async def send(self, case: TestCase) -> TargetResponse:
        messages = []
        sp = case.system_prompt_override or self.system_prompt
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": case.prompt})

        start = time.monotonic()
        resp = await litellm.acompletion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            temperature=0.0,
        )
        latency_ms = (time.monotonic() - start) * 1000

        return TargetResponse(
            case_id=case.case_id,
            raw_text=resp.choices[0].message.content or "",
            latency_ms=latency_ms,
            tokens_used=getattr(resp.usage, "total_tokens", None),
        )

    async def send_conversation(
        self, case: TestCase, stop_when: Callable[[str], bool] | None = None
    ) -> TargetResponse:
        """Genuine multi-turn dialogue: each turn's assistant reply is
        appended to `messages` before the next user turn is sent, so turn N
        sees turns 1..N-1 (D-12).

        `stop_when`, when given, is evaluated against each turn's reply; a
        truthy result aborts the sequence early (D-14) — the decision of
        when to abort belongs to the caller, never to the adapter.
        """
        messages = []
        sp = case.system_prompt_override or self.system_prompt
        if sp:
            messages.append({"role": "system", "content": sp})

        turns = case.turns or [case.prompt]
        turn_replies: list[str] = []
        transcript_parts: list[str] = []
        total_tokens = 0
        start = time.monotonic()

        for turn_number, turn_text in enumerate(turns, start=1):
            messages.append({"role": "user", "content": turn_text})
            resp = await litellm.acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                temperature=0.0,
            )
            reply = resp.choices[0].message.content or ""
            # Appending the reply back as an assistant message is what
            # carries dialogue state forward — skipping this loses all
            # prior context and silently reduces every crescendo to a
            # series of independent single shots.
            messages.append({"role": "assistant", "content": reply})
            turn_replies.append(reply)
            transcript_parts.append(f"Turn {turn_number} USER: {turn_text}\nTurn {turn_number} ASSISTANT: {reply}")
            total_tokens += getattr(resp.usage, "total_tokens", None) or 0

            if stop_when is not None and stop_when(reply):
                break

        latency_ms = (time.monotonic() - start) * 1000

        return TargetResponse(
            case_id=case.case_id,
            raw_text="\n\n".join(transcript_parts),
            latency_ms=latency_ms,
            tokens_used=total_tokens or None,
            transport_mode="multi_turn_real",
            turn_replies=turn_replies,
        )

    async def health_check(self) -> bool:
        try:
            await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                api_key=self.api_key,
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass  # litellm manages its own client lifecycle; nothing to release here
