"""Per-connection session state.

Single owner — the FastAPI handler creates one on `/run`, hands it to the
pipeline runner, and drops the reference on disconnect. Tool handlers and
processors all close over the same instance.

This module imports nothing from Pipecat — it owns plain Python state plus a
reference to Pipecat's `LLMContext` (an opaque handle for our purposes).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from uxflows_runner.events import EventEmitter, NullEventEmitter
from uxflows_runner.spec.loader import LoadedSpec

from .capabilities import CapabilityDispatcher
from .flow_state import FlowState

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext

    from .routing import RoutingPlan


@dataclass
class Session:
    spec: LoadedSpec
    state: FlowState
    capabilities: CapabilityDispatcher
    events: EventEmitter
    llm_context: "LLMContext"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    current_plan: "RoutingPlan | None" = None
    tool_handler_fired_this_turn: bool = False
    ended: bool = False

    @classmethod
    def start(
        cls,
        spec: LoadedSpec,
        llm_context: "LLMContext",
        events: EventEmitter | None = None,
        capabilities: CapabilityDispatcher | None = None,
        language: str | None = None,
    ) -> "Session":
        lang = language or (spec.agent.meta.languages[0] if spec.agent.meta.languages else "en-US")
        state = FlowState.start(spec.agent.entry_flow_id, language=lang)
        return cls(
            spec=spec,
            state=state,
            capabilities=capabilities or CapabilityDispatcher(spec=spec, endpoints={}),
            events=events or NullEventEmitter(),
            llm_context=llm_context,
        )

    def emit_session_started(self) -> None:
        from uxflows_runner.events.schema import SessionStarted

        self.events.emit(
            SessionStarted(
                session_id=self.session_id,
                agent_id=self.spec.agent.id,
                lang=self.state.language,
                spec_hash=self.spec.spec_hash,
            )
        )

    def emit_flow_entered(
        self, flow_id: str, via: str, caller_flow_id: str | None = None
    ) -> None:
        from uxflows_runner.events.schema import FlowEntered

        self.events.emit(
            FlowEntered(
                session_id=self.session_id,
                flow_id=flow_id,
                via=via,  # type: ignore[arg-type]
                caller_flow_id=caller_flow_id,
            )
        )
