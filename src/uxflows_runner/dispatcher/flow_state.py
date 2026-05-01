"""Stack-based flow state.

Top of stack = active flow. Interrupts push a new frame; `return_to_caller` pops.
Variables live in a single flat bag (v0 — agent and flow scopes share). The
turn counter is per-frame: interrupted turns do NOT increment the caller's
counter. `max_turns` checks the active frame's counter only.

This module is pure data — no Pipecat, no IO. Tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowFrame:
    flow_id: str
    turn_count: int = 0
    caller_flow_id: str | None = None  # set for interrupt frames; None for the root


@dataclass
class FlowState:
    """Live session state. Mutated in place by the dispatcher."""

    stack: list[FlowFrame]
    variables: dict[str, Any] = field(default_factory=dict)
    language: str = "en-US"

    @classmethod
    def start(cls, entry_flow_id: str, language: str = "en-US") -> "FlowState":
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
    def is_in_interrupt(self) -> bool:
        return len(self.stack) > 1

    def push_interrupt(self, interrupt_flow_id: str) -> FlowFrame:
        """Trigger an interrupt: push a new frame whose caller is the current active flow."""
        caller_id = self.active_flow_id
        frame = FlowFrame(flow_id=interrupt_flow_id, caller_flow_id=caller_id)
        self.stack.append(frame)
        return frame

    def pop_to_caller(self) -> tuple[FlowFrame, FlowFrame]:
        """`return_to_caller`: pop the active interrupt frame, return (popped, caller).
        Raises if not currently inside an interrupt."""
        if not self.is_in_interrupt:
            raise RuntimeError("return_to_caller used outside an interrupt frame")
        popped = self.stack.pop()
        return popped, self.active

    def transition(self, next_flow_id: str) -> FlowFrame:
        """Normal exit-path transition: replace the active frame with a fresh one
        on `next_flow_id`. Preserves caller chain — if we're inside an interrupt
        and the interrupt's exit_path takes a normal `next_flow_id` (rare but
        legal), the new frame stays scoped under the same caller."""
        old = self.stack.pop()
        new = FlowFrame(flow_id=next_flow_id, caller_flow_id=old.caller_flow_id)
        self.stack.append(new)
        return new

    def end(self) -> None:
        """Terminal exit (`type: exit` or `next_flow_id: null` on a non-return path)."""
        self.stack.clear()

    def increment_turn(self) -> int:
        self.active.turn_count += 1
        return self.active.turn_count
