"""Event schema — the runner-side contract.

Per RUNNER-PLAN §"Event schema" (lines ~282-296). Each event is a small,
flat record. Phase 1 emits a subset matching the v0 dispatcher feature
scope. Phase 2 (canvas integration) consumes the same stream.

Mirrored manually in `lib/runtime/eventTypes.ts` on the editor side once
that's wired up.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ts: str = Field(default_factory=_now)


class SessionStarted(_Base):
    type: Literal["session_started"] = "session_started"
    agent_id: str
    lang: str
    spec_hash: str


class SessionEnded(_Base):
    type: Literal["session_ended"] = "session_ended"
    reason: Literal["user_stop", "agent_terminal", "error", "idle"]


class FlowEntered(_Base):
    type: Literal["flow_entered"] = "flow_entered"
    flow_id: str
    via: Literal["transition", "interrupt", "return_to_caller", "entry"]
    caller_flow_id: str | None = None


class FlowExited(_Base):
    type: Literal["flow_exited"] = "flow_exited"
    flow_id: str
    exit_path_id: str | None = None
    reason: Literal["transition", "terminal", "interrupted", "returned_to_caller"]


class ExitPathTaken(_Base):
    type: Literal["exit_path_taken"] = "exit_path_taken"
    from_flow_id: str
    exit_path_id: str
    to_flow_id: str | None = None
    method: Literal["llm", "calculation", "direct"]


class InterruptTriggered(_Base):
    type: Literal["interrupt_triggered"] = "interrupt_triggered"
    from_flow_id: str
    interrupt_flow_id: str
    method: Literal["llm", "calculation", "direct"]


class TurnStarted(_Base):
    type: Literal["turn_started"] = "turn_started"
    role: Literal["agent", "user"]


class TurnCompleted(_Base):
    type: Literal["turn_completed"] = "turn_completed"
    role: Literal["agent", "user"]
    text: str


class VariableSet(_Base):
    type: Literal["variable_set"] = "variable_set"
    variable_name: str
    value: Any
    method: Literal["llm", "calculation", "direct"]
    source_flow_id: str
    source_exit_path_id: str


class CapabilityInvoked(_Base):
    type: Literal["capability_invoked"] = "capability_invoked"
    capability_name: str
    args: dict[str, Any]


class CapabilityReturned(_Base):
    type: Literal["capability_returned"] = "capability_returned"
    capability_name: str
    result: Any | None = None
    error: str | None = None


class Error(_Base):
    type: Literal["error"] = "error"
    code: str
    message: str
    recoverable: bool


Event = Union[
    SessionStarted,
    SessionEnded,
    FlowEntered,
    FlowExited,
    ExitPathTaken,
    InterruptTriggered,
    TurnStarted,
    TurnCompleted,
    VariableSet,
    CapabilityInvoked,
    CapabilityReturned,
    Error,
]


class EventEnvelope(BaseModel):
    """For SSE wire format — `event` + `data` JSON. Phase 2 wiring."""

    event: str
    data: dict[str, Any]

    @classmethod
    def from_event(cls, ev: Event) -> "EventEnvelope":
        return cls(event=ev.type, data=ev.model_dump(mode="json"))
