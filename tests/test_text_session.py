"""End-to-end tests for the text-mode adapter (Phase 1.5).

Mocks `TextSession._run_inference` to avoid touching the network. Drives the
real dispatcher (Session, FlowState, routing.plan/resolve, assigns,
capabilities, prompt_builder, processor.apply_tool_call) against examples/
coffee.json to confirm:

  - chatbot_initiates fires the opening turn.
  - take_exit_path mutates state and emits events.
  - variable_set fires for llm-method assigns.
  - Multiple turns walk through real flow transitions.
  - End-of-conversation terminal exit emits session_ended.

The LLM responses are scripted: each call to _run_inference returns the next
(text, tool_calls) tuple from a queue.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from uxflows_runner.events.schema import (
    ExitPathTaken,
    FlowEntered,
    SessionEnded,
    SessionStarted,
    VariableSet,
)
from uxflows_runner.server.text_session import TextSession
from uxflows_runner.spec.loader import load_spec


COFFEE_SPEC = Path(__file__).resolve().parents[1] / "examples" / "coffee.json"


def _scripted_inference(turns: list[tuple[str, list[dict]]]):
    """Build an async _run_inference mock that pops scripted responses.

    Accepts `include_tools` kwarg (used by the silent-take_exit follow-up)
    but doesn't branch on it — the scripted responses control behavior.
    """
    queue = list(turns)

    async def _mock(self, include_tools: bool = True) -> tuple[str, list]:
        if not queue:
            raise AssertionError("Test ran out of scripted LLM turns")
        return queue.pop(0)

    return _mock


def _tc(name: str, args: dict, tc_id: str = "call-1") -> dict:
    return {"id": tc_id, "name": name, "args": args}


@pytest.fixture
def coffee_spec():
    return load_spec(str(COFFEE_SPEC))


@pytest.mark.asyncio
async def test_start_emits_session_and_flow_events_and_runs_opening_turn(
    coffee_spec, monkeypatch
):
    """chatbot_initiates: true → opening turn fires; events buffered correctly."""
    monkeypatch.setattr(
        TextSession,
        "_run_inference",
        _scripted_inference([("Hi! What can I get started for you?", [])]),
    )

    ts, opening = await TextSession.start(spec=coffee_spec, api_key="dummy")

    assert opening == "Hi! What can I get started for you?"
    assert not ts.ended

    events = ts.drain_events()
    types = [type(e).__name__ for e in events]
    assert "SessionStarted" in types
    assert "FlowEntered" in types
    fe = next(e for e in events if isinstance(e, FlowEntered))
    assert fe.flow_id == "flow_greet"
    assert fe.via == "entry"


@pytest.mark.asyncio
async def test_turn_with_take_exit_path_transitions_flow_and_fires_assigns(
    coffee_spec, monkeypatch
):
    """User asks for a latte → LLM emits text + take_exit_path tool with
    drink_type assign → state transitions to flow_coffee_order, variable_set
    + exit_path_taken + flow_entered events fire."""
    script = [
        # Opening turn (chatbot_initiates)
        ("Hi! What can I get started for you?", []),
        # User: "I'd love a latte" — text + tool call together (Gemini's normal mode)
        (
            "Got it, a latte coming up. What size?",
            [_tc("take_exit_path", {"exit_path_id": "xp_greet_to_coffee", "drink_type": "coffee"})],
        ),
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()  # discard start events

    reply = await ts.turn("I'd love a latte")
    assert reply == "Got it, a latte coming up. What size?"

    events = ts.drain_events()
    types = [type(e).__name__ for e in events]
    assert "VariableSet" in types
    assert "ExitPathTaken" in types
    assert "FlowEntered" in types

    var_set = next(e for e in events if isinstance(e, VariableSet))
    assert var_set.variable_name == "drink_type"
    assert var_set.value == "coffee"
    # coffee.json's xp_greet_to_coffee.assigns.drink_type uses method=direct
    # (literal "coffee") even though the *condition* is llm. The dispatcher
    # honors the assign method, not the condition method.
    assert var_set.method == "direct"

    xpt = next(e for e in events if isinstance(e, ExitPathTaken))
    assert xpt.from_flow_id == "flow_greet"
    assert xpt.exit_path_id == "xp_greet_to_coffee"
    assert xpt.to_flow_id == "flow_coffee_order"

    fe = next(e for e in events if isinstance(e, FlowEntered))
    assert fe.flow_id == "flow_coffee_order"
    assert fe.via == "transition"

    # Variable bag updated on the actual session
    assert ts.session.state.variables["drink_type"] == "coffee"

    # Active flow advanced
    assert ts.session.state.active_flow_id == "flow_coffee_order"


@pytest.mark.asyncio
async def test_turn_without_tool_call_just_returns_text(coffee_spec, monkeypatch):
    """Plain reply, no tool — should return text and stay in the active flow."""
    script = [
        ("Hi! What can I get started?", []),
        ("Sure, take your time.", []),
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    reply = await ts.turn("hmm let me think")
    assert reply == "Sure, take your time."
    assert ts.session.state.active_flow_id == "flow_greet"


@pytest.mark.asyncio
async def test_terminal_exit_ends_session(coffee_spec, monkeypatch):
    """Walkaway exit from greet flow ends the session."""
    script = [
        ("Welcome, what'll it be?", []),
        # User says "nevermind" → walkaway exit (terminal, next_flow_id null)
        (
            "No worries, come back anytime!",
            [_tc("take_exit_path", {"exit_path_id": "xp_greet_walkaway"})],
        ),
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    await ts.turn("nevermind, I'll come back later")

    assert ts.ended
    events = ts.drain_events()
    assert any(isinstance(e, SessionEnded) for e in events)


@pytest.mark.asyncio
async def test_end_is_idempotent_and_emits_session_ended_once(coffee_spec, monkeypatch):
    """end() called on a live session emits session_ended; second call is a no-op."""
    monkeypatch.setattr(
        TextSession, "_run_inference", _scripted_inference([("Hi!", [])])
    )

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    await ts.end()
    events_after_first = ts.drain_events()
    assert any(isinstance(e, SessionEnded) for e in events_after_first)
    assert ts.ended

    await ts.end()
    assert ts.drain_events() == []  # no new events

    # Subsequent turn raises
    from uxflows_runner.server.text_session import SessionAlreadyEnded
    with pytest.raises(SessionAlreadyEnded):
        await ts.turn("anything")


@pytest.mark.asyncio
async def test_silent_take_exit_triggers_no_tools_followup(coffee_spec, monkeypatch):
    """Gemini sometimes returns a take_exit_path tool call with NO text part —
    state mutates correctly but the user gets nothing to read. The follow-up
    inference should run text-only (no tools) and its text becomes the reply.

    Also asserts include_tools=False is passed on the follow-up call so the
    model can't chain another transition.
    """
    calls: list[bool] = []  # include_tools value per inference call

    script = [
        # Opening turn
        ("Welcome! What can I get you?", []),
        # User: "i'd like a latte" — Gemini routes silently (no text)
        (
            "",
            [_tc("take_exit_path", {"exit_path_id": "xp_greet_to_coffee", "drink_type": "coffee"})],
        ),
        # Follow-up: must produce text now, called with include_tools=False
        ("Coming right up — what size would you like?", []),
    ]
    queue = list(script)

    async def _mock(self, include_tools: bool = True) -> tuple[str, list]:
        calls.append(include_tools)
        return queue.pop(0)

    monkeypatch.setattr(TextSession, "_run_inference", _mock)

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    reply = await ts.turn("i'd like a latte")
    assert reply == "Coming right up — what size would you like?"

    # 3 inference calls total: opening, silent take_exit, no-tools follow-up
    assert calls == [True, True, False]

    # State still transitioned cleanly — the follow-up didn't undo anything
    assert ts.session.state.active_flow_id == "flow_coffee_order"
    assert ts.session.state.variables["drink_type"] == "coffee"


@pytest.mark.asyncio
async def test_silent_terminal_take_exit_does_not_trigger_followup(
    coffee_spec, monkeypatch
):
    """If the silent take_exit was a terminal exit, the session ended; we
    should NOT do a follow-up — there's no flow left to speak from."""
    calls: list[bool] = []

    script = [
        ("Welcome!", []),
        # Walkaway exit — terminal, ends the session
        ("", [_tc("take_exit_path", {"exit_path_id": "xp_greet_walkaway"})]),
        # No third call should happen
    ]
    queue = list(script)

    async def _mock(self, include_tools: bool = True) -> tuple[str, list]:
        calls.append(include_tools)
        return queue.pop(0)

    monkeypatch.setattr(TextSession, "_run_inference", _mock)

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    reply = await ts.turn("nevermind")
    assert reply == ""  # silent terminal — accepted, no follow-up
    assert calls == [True, True]  # only opening + the silent take_exit
    assert ts.ended


@pytest.mark.asyncio
async def test_assistant_message_with_tool_call_appended_to_context(
    coffee_spec, monkeypatch
):
    """Tool-call turns should append both the assistant message (with tool_calls)
    and a tool-result message — keeps the LLMContext history valid for any
    follow-up inference."""
    script = [
        ("Welcome!", []),
        (
            "Coffee, got it. What size?",
            [_tc("take_exit_path", {"exit_path_id": "xp_greet_to_coffee", "drink_type": "coffee"})],
        ),
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    await ts.turn("a coffee please")

    msgs = ts.session.llm_context.messages
    # Should contain: system, "(begin)" user, opening assistant, real user,
    # assistant with tool_calls, tool result.
    assistant_with_tools = [
        m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_with_tools) == 1
    tc = assistant_with_tools[0]["tool_calls"][0]
    assert tc["function"]["name"] == "take_exit_path"
    args = json.loads(tc["function"]["arguments"])
    assert args["exit_path_id"] == "xp_greet_to_coffee"

    tool_results = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_results) == 1
