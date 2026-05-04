"""Text I/O adapter — drives the dispatcher with plain HTTP turns instead of
a Pipecat audio pipeline.

One TextSession per /api/chat/session call. Reuses the dispatcher core
(Session, FlowState, routing.plan/resolve, assigns, capabilities, prompt_builder)
and the framework-agnostic `apply_tool_call` from dispatcher.processor —
nothing changes between voice and text below the cognitive layer.

Mode-specific wiring this file owns:
  - Construct `GoogleLLMService` with a BYOK AI Studio key (NOT Vertex).
  - Construct `LLMContext` standalone (no aggregator, no pipeline).
  - Per turn: call `_run_inference()` (one generate_content), parse text +
    function_calls from response parts, invoke `apply_tool_call` for any tool
    fired, run a SECOND inference inline if the decision was an interrupt or
    return-to-caller (Gemini quirk — see `apply_tool_call` docstring).
  - Append assistant + tool messages to LLMContext to keep history honest.
  - Buffer events for `drain()` so the HTTP response can return them inline.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from google.genai.types import GenerateContentConfig
from loguru import logger

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.vertex.llm import GoogleVertexLLMService

from uxflows_runner.config import Config
from uxflows_runner.dispatcher.capabilities import CapabilityDispatcher
from uxflows_runner.dispatcher.processor import (
    add_capability_result_listener,
    apply_tool_call,
    plan_for_active_flow,
)
from uxflows_runner.dispatcher.prompt_builder import build_system_prompt
from uxflows_runner.dispatcher.session import Session
from uxflows_runner.events.emitter import BufferingEventEmitter
from uxflows_runner.events.schema import Event, SessionEnded
from uxflows_runner.spec.loader import LoadedSpec


DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class TextSession:
    session: Session
    llm: GoogleLLMService
    events: BufferingEventEmitter
    last_active_at: float = field(default_factory=time.monotonic)

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def ended(self) -> bool:
        return self.session.ended

    @classmethod
    async def start(
        cls,
        spec: LoadedSpec,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
        execution_endpoints: dict[str, str] | None = None,
        config: Config | None = None,
    ) -> tuple["TextSession", str]:
        """Construct + run session_started + flow_entered + opening turn (if
        chatbot_initiates). Returns (self, opening_agent_text).

        If `api_key` is provided, uses GoogleLLMService against the AI Studio
        API (BYOK). If omitted, falls back to GoogleVertexLLMService against
        the env service account — same auth voice mode uses, no extra key
        needed for local dev.
        """
        lang = language or (
            spec.agent.meta.languages[0] if spec.agent.meta.languages else "en-US"
        )
        entry_flow = spec.entry_flow
        initial_prompt = build_system_prompt(spec.agent, entry_flow, lang)
        context = LLMContext(messages=[{"role": "system", "content": initial_prompt}])

        if api_key:
            llm = GoogleLLMService(
                api_key=api_key,
                settings=GoogleLLMService.Settings(model=model or DEFAULT_MODEL),
            )
        else:
            cfg = config or Config.from_env()
            llm = GoogleVertexLLMService(
                credentials_path=cfg.google_credentials_path,
                project_id=cfg.google_project_id,
                location=cfg.google_location,
                settings=GoogleVertexLLMService.Settings(model=model or cfg.llm_model),
            )

        events = BufferingEventEmitter()
        capabilities = CapabilityDispatcher(spec=spec, endpoints=execution_endpoints or {})
        session = Session.start(
            spec=spec,
            llm_context=context,
            events=events,
            capabilities=capabilities,
            language=lang,
        )
        add_capability_result_listener(session)

        ts = cls(session=session, llm=llm, events=events)

        session.emit_session_started()
        session.emit_flow_entered(entry_flow.id, via="entry")

        opening = ""
        if spec.agent.chatbot_initiates:
            # Mirror the voice path's "(begin)" kick — gives the LLM a turn
            # to respond from. The synthetic user message stays in context;
            # this matches voice for parity.
            context.add_message({"role": "user", "content": "(begin)"})
            opening = await ts._run_one_turn()

        return ts, opening

    async def turn(self, user_text: str) -> str:
        """Send a user turn; return the agent's reply text. Mutates session
        state (flow transitions, variables, etc.) and buffers events for
        `drain_events()`."""
        self.last_active_at = time.monotonic()
        if self.session.ended:
            raise SessionAlreadyEnded(self.session_id)

        self.session.llm_context.add_message({"role": "user", "content": user_text})
        return await self._run_one_turn()

    def drain_events(self) -> list[Event]:
        return self.events.drain()

    async def end(self) -> None:
        """Idempotent. Emit session_ended if not already ended; close clients."""
        if not self.session.ended:
            self.session.events.emit(
                SessionEnded(session_id=self.session_id, reason="user_stop")
            )
            self.session.state.end()
            self.session.ended = True
        await self.session.capabilities.aclose()

    # ----- internals -----

    async def _run_one_turn(self) -> str:
        """One inference + tool dispatch + optional follow-up inference.

        Mirrors the voice path's per-turn shape: PreLLMPlanner equivalent
        (plan_for_active_flow), then inference, then any tool handler runs
        synchronously via apply_tool_call, then optional follow-up for
        interrupts/returns.
        """
        s = self.session
        if s.ended:
            return ""

        plan_for_active_flow(s)
        s.tool_handler_fired_this_turn = False

        text, tool_calls = await self._run_inference()
        self._append_assistant(text, tool_calls)

        if not tool_calls:
            return text

        # In v0 the dispatcher emits at most one tool call per turn (one of
        # take_exit_path / trigger_interrupt). Even if Gemini ever fires more,
        # processing the first one wins — matches the precedence in
        # routing.resolve.
        first_call = tool_calls[0]
        decision = await apply_tool_call(s, first_call["name"], first_call["args"])

        # Append a tool-result message so follow-up inferences see a closed
        # tool turn. Empty payload — the dispatcher's "result" is the state
        # mutation it already applied.
        self._append_tool_result(first_call["id"], first_call["name"])

        if decision is not None and decision.kind in ("trigger_interrupt", "return_to_caller"):
            # Voice path pushes LLMRunFrame; text mode just calls inference
            # again. plan_for_active_flow already ran inside apply_tool_call,
            # so the new flow's prompt + tools are loaded.
            s.tool_handler_fired_this_turn = False
            followup_text, followup_calls = await self._run_inference()
            self._append_assistant(followup_text, followup_calls)
            # Follow-up tool calls (rare — would mean the new flow immediately
            # routed off itself) get applied too. Don't recurse further.
            if followup_calls:
                fc = followup_calls[0]
                await apply_tool_call(s, fc["name"], fc["args"])
                self._append_tool_result(fc["id"], fc["name"])
            return followup_text

        # Silent-take_exit follow-up: Gemini sometimes emits a take_exit_path
        # tool call with no text part — the routing decision became the entire
        # response. The state mutation is correct; the user just gets nothing
        # to read. Issue ONE follow-up inference with tools DROPPED so the
        # model can only produce text (no risk of chained transitions or
        # premature cancel/confirm calls — see RUNNER-PLAN §"Live-test follow-up").
        # Capped at one follow-up per turn structurally — we don't recurse.
        if (
            decision is not None
            and decision.kind == "take_exit"
            and not text
            and not s.ended
        ):
            followup_text, _ = await self._run_inference(include_tools=False)
            self._append_assistant(followup_text, [])
            return followup_text

        return text

    async def _run_inference(
        self, include_tools: bool = True
    ) -> tuple[str, list[dict[str, Any]]]:
        """One generate_content call. Returns (text, tool_calls) where
        tool_calls is a list of {"id", "name", "args"} dicts in part order.

        `include_tools=False` forces text-only output — used by the silent
        take_exit follow-up to guarantee the model produces words and can't
        chain another routing decision.
        """
        adapter = GeminiLLMAdapter()
        params = adapter.get_llm_invocation_params(self.session.llm_context)

        tools = params["tools"] if (include_tools and params["tools"]) else None
        gen_params = self.llm._build_generation_params(  # noqa: SLF001
            system_instruction=params["system_instruction"],
            tools=tools,
        )
        # Match voice path: disable thinking on 2.5 Flash for low TTFT.
        gen_params["thinking_config"] = {"thinking_budget": 0}

        response = await self.llm._client.aio.models.generate_content(  # noqa: SLF001
            model=self.llm._settings.model,  # noqa: SLF001
            contents=params["messages"],
            config=GenerateContentConfig(**gen_params),
        )

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts or []:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        {
                            "id": fc.id or uuid.uuid4().hex,
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }
                    )

        return "".join(text_parts).strip(), tool_calls

    def _append_assistant(self, text: str, tool_calls: list[dict[str, Any]]) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in tool_calls
            ]
        self.session.llm_context.add_message(msg)

    def _append_tool_result(self, tool_call_id: str, tool_name: str) -> None:
        # Empty {} payload — the dispatcher consumed the call, no return value
        # to surface to the LLM. Including the message keeps the conversation
        # history grammatical for any follow-up inference.
        self.session.llm_context.add_message(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({}),
            }
        )
        logger.debug("text turn applied tool {} (id={})", tool_name, tool_call_id)


class SessionAlreadyEnded(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"session {session_id} has already ended")
        self.session_id = session_id
