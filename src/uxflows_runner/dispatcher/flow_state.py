"""Stack-based flow state.

Top of stack = active flow. Entering a callable flow (one with a `goto:
"RETURN"` exit, including any interrupt) pushes a new frame; taking a
`goto: "RETURN"` exit pops back to the caller. Variables live in a single
flat bag (agent and flow scopes share). The turn counter is per-frame:
turns inside a pushed frame do NOT increment the caller's counter — load-
bearing for a future `_turn_count` reserved variable (see SCHEMA.md Open
Questions).

This module is pure data — no Pipecat, no IO. Tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowFrame:
    flow_id: str
    turn_count: int = 0
    caller_flow_id: str | None = None  # set for pushed frames; None for the root


@dataclass
class FlowState:
    """Live session state. Mutated in place by the dispatcher."""

    stack: list[FlowFrame]
    variables: dict[str, Any] = field(default_factory=dict)
    # None means "all languages" — the system prompt emits every script bucket.
    # A concrete code (e.g. "en-US") filters to just that bucket.
    language: str | None = None

    @classmethod
    def start(cls, entry_flow_id: str, language: str | None = None) -> "FlowState":
        return cls(stack=[FlowFrame(flow_id=entry_flow_id)], language=language)

    @property
    def active(self) -> FlowFrame:
        if not self.stack:
            raise RuntimeError("flow stack is empty — session has ended")
        return self.stack[-1]

    @property
    def active_flow_id(self) -> str:
        return self.active.flow_id

    @property
    def has_caller(self) -> bool:
        """True iff there's a caller frame to return to — i.e., we're inside
        a callable flow (interrupt, utility subroutine, anything entered via
        push_call)."""
        return len(self.stack) > 1

    def push_call(self, target_flow_id: str) -> FlowFrame:
        """Enter a callable flow: push a new frame whose caller is the current
        active flow. Used for interrupt pivots and any other transition into
        a flow that has a `goto: "RETURN"` exit."""
        caller_id = self.active_flow_id
        frame = FlowFrame(flow_id=target_flow_id, caller_flow_id=caller_id)
        self.stack.append(frame)
        return frame

    def pop_to_caller(self) -> tuple[FlowFrame, FlowFrame]:
        """`goto: "RETURN"`: pop the active frame, return (popped, caller).
        Raises if there's no caller frame to pop."""
        if not self.has_caller:
            raise RuntimeError("RETURN used outside a callable frame")
        popped = self.stack.pop()
        return popped, self.active

    def transition(self, target_flow_id: str) -> FlowFrame:
        """Normal (non-callable) transition: replace the active frame with a
        fresh one on `target_flow_id`. Preserves caller chain — if we're inside
        a callable frame and its exit_path takes a normal flow goto (rare but
        legal), the new frame stays scoped under the same caller."""
        old = self.stack.pop()
        new = FlowFrame(flow_id=target_flow_id, caller_flow_id=old.caller_flow_id)
        self.stack.append(new)
        return new

    def end(self) -> None:
        """Terminal exit (`goto: "END"`, or `goto: "RETURN"` with no caller)."""
        self.stack.clear()

    def increment_turn(self) -> int:
        self.active.turn_count += 1
        return self.active.turn_count
