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

import time
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
from uxflows_runner.dispatcher import routing_protocol
from uxflows_runner.dispatcher.processor import (
    apply_planned_shortcut,
    apply_route,
    plan_for_active_flow,
)
from uxflows_runner.dispatcher.prompt_builder import build_system_prompt
from uxflows_runner.dispatcher.session import Session
from uxflows_runner.events.emitter import (
    BufferingEventEmitter,
    JsonlEventEmitter,
    MultiEventEmitter,
)
from uxflows_runner.events.schema import Event, SessionEnded, TurnCompleted
from uxflows_runner.spec.loader import LoadedSpec


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
        context_vars: dict[str, Any] | None = None,
        mock_returns: dict[str, dict[str, Any]] | None = None,
    ) -> tuple["TextSession", str]:
        """Construct + run session_started + flow_entered + opening turn (if
        chatbot_initiates). Returns (self, opening_agent_text).

        If `api_key` is provided, uses GoogleLLMService against the AI Studio
        API (BYOK). If omitted, falls back to GoogleVertexLLMService against
        the env service account — same auth voice mode uses, no extra key
        needed for local dev.

        `context_vars` seeds the dispatcher's variable bag at session start
        — used both for `{KEY}` placeholder substitution in the composed
        system prompt and as initial values readable by routing conditions /
        capability inputs. Not emitted as `variable_set` events (those are
        for exit-path-fired assigns; these are session-start seeds).

        `mock_returns` is a per-session simulation fixture keyed by capability
        NAME (not id): when a dispatched capability has a matching entry, the
        dispatcher returns that dict instead of hitting an HTTP endpoint. Lets
        the editor's SimulatePanel probe happy / sad paths without standing up
        a mock server. Production deployments leave it unset.
        """
        # No fallback: None means "all languages" — the prompt builder emits
        # every script bucket. Clients that want single-language behavior
        # pass `language` explicitly.
        lang = language
        entry_flow = spec.entry_flow

        cfg = config or Config.from_env()
        chosen_model = model or cfg.llm_model

        if api_key:
            llm = GoogleLLMService(
                api_key=api_key,
                settings=GoogleLLMService.Settings(model=chosen_model),
            )
        else:
            llm = GoogleVertexLLMService(
                credentials_path=cfg.google_credentials_path,
                project_id=cfg.google_project_id,
                location=cfg.google_location,
                settings=GoogleVertexLLMService.Settings(model=chosen_model),
            )

        events = BufferingEventEmitter()
        capabilities = CapabilityDispatcher(
            spec=spec,
            endpoints=execution_endpoints or {},
            mock_returns=mock_returns,
        )
        # Build LLMContext with a placeholder system message; we'll replace it
        # below after seeding context_vars so the initial prompt has them
        # substituted in.
        context = LLMContext(messages=[{"role": "system", "content": ""}])
        session = Session.start(
            spec=spec,
            llm_context=context,
            events=events,
            capabilities=capabilities,
            language=lang,
        )
        # Tee events to disk if configured. `events` (BufferingEventEmitter)
        # is kept on `self` for `drain_events()`; the dispatcher reads via
        # `session.events`, which we wrap with the JSONL sink.
        if config is not None and config.event_log_dir is not None:
            jsonl_path = config.event_log_dir / f"{session.session_id}.jsonl"
            session.events = MultiEventEmitter([events, JsonlEventEmitter(jsonl_path)])
        if context_vars:
            session.state.variables.update(context_vars)

        # Now compose the real initial prompt with seeded variables in scope.
        initial_prompt = build_system_prompt(
            spec, entry_flow, lang, variables=session.state.variables
        )
        context.messages[0]["content"] = initial_prompt

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
        self.session.events.emit(
            TurnCompleted(
                session_id=self.session.session_id, role="user", text=user_text
            )
        )
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
        """One inference + optional in-text routing + optional follow-up.

        Shape:
          1. plan_for_active_flow (builds system prompt incl. routing protocol)
          2. _run_inference (returns full reply text — no tools)
          3. find_tag → strip → return cleaned reply to caller
          4. apply_route on the parsed tag (or apply_planned_shortcut if the
             flow is a terminator and the LLM emitted no tag)
          5. follow-up inference rules:
             - interrupt or return-into-caller: ALWAYS follow up. The
               destination owns the substantive response to this turn.
             - take_exit with empty reply text: follow up. The source flow
               judged it had nothing to say (the user's message was
               topic-shifted); let the destination's prompt speak instead
               of forcing the source to improvise.
             - take_exit with non-empty reply: no follow-up. The source
               flow already responded; the destination's opener fires on
               the next user turn.
             - end / stay: no follow-up.
        """
        s = self.session
        if s.ended:
            return ""

        plan_for_active_flow(s)
        s.tool_handler_fired_this_turn = False

        raw = await self._run_inference()
        cleaned, tag = routing_protocol.find_tag(raw)
        self._append_assistant(cleaned)

        if tag is None:
            # No route tag — fire the planned shortcut if this is a
            # single-unconditional-exit terminator flow. Otherwise stay.
            await apply_planned_shortcut(s)
            return cleaned

        decision = await apply_route(s, tag)
        if decision is None or s.ended:
            return cleaned

        needs_followup = (
            decision.kind in ("trigger_interrupt", "return")
            or (decision.kind == "take_exit" and not cleaned.strip())
        )
        if not needs_followup:
            return cleaned

        # New flow is now active — give its opener a chance to speak.
        # plan_for_active_flow already ran inside apply_route; the new
        # flow's prompt (with its own routing protocol) is loaded.
        s.tool_handler_fired_this_turn = False
        followup_raw = await self._run_inference()
        followup_cleaned, followup_tag = routing_protocol.find_tag(followup_raw)
        self._append_assistant(followup_cleaned)
        # Rare: the new flow itself routes off on its first turn. Honor
        # it but don't recurse further.
        if followup_tag is not None:
            await apply_route(s, followup_tag)
        return followup_cleaned

    async def _run_inference(self) -> str:
        """One generate_content call. Returns the model's full text output
        (with any trailing `<route .../>` tag still present — caller strips).

        No tools — routing is in-text via the route protocol described in
        the system prompt.
        """
        adapter = GeminiLLMAdapter()
        params = adapter.get_llm_invocation_params(self.session.llm_context)

        gen_params = self.llm._build_generation_params(  # noqa: SLF001
            system_instruction=params["system_instruction"],
            tools=None,
        )
        # Match voice path: disable thinking on 2.5 Flash for low TTFT.
        gen_params["thinking_config"] = {"thinking_budget": 0}

        response = await self.llm._client.aio.models.generate_content(  # noqa: SLF001
            model=self.llm._settings.model,  # noqa: SLF001
            contents=params["messages"],
            config=GenerateContentConfig(**gen_params),
        )

        text_parts: list[str] = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts or []:
                if part.text:
                    text_parts.append(part.text)

        return "".join(text_parts).strip()

    def _append_assistant(self, text: str) -> None:
        """Record the cleaned reply text (route tag already stripped) on the
        conversation history AND emit a TurnCompleted event."""
        self.session.llm_context.add_message(
            {"role": "assistant", "content": text}
        )
        if text:
            self.session.events.emit(
                TurnCompleted(
                    session_id=self.session.session_id, role="agent", text=text
                )
            )


class SessionAlreadyEnded(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"session {session_id} has already ended")
        self.session_id = session_id
