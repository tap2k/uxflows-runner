"""End-to-end tests for the text-mode adapter (Phase 1.5).

Mocks `TextSession._run_inference` to avoid touching the network. Drives the
real dispatcher (Session, FlowState, routing.plan/resolve, assigns,
capabilities, prompt_builder, processor.apply_route) against examples/
coffee.json to confirm:

  - chatbot_initiates fires the opening turn.
  - In-text route tags transition flows and fire assigns.
  - variable_set fires for llm-method assigns parsed off the tag.
  - Multiple turns walk through real flow transitions.
  - End-of-conversation terminal exit emits session_ended.

The LLM responses are scripted: each call to _run_inference returns the next
string from a queue. The string may include a trailing `<route ... />` tag
which the runner parses and dispatches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uxflows_runner.config import Config
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


def _scripted_inference(turns: list[str]):
    """Build an async _run_inference mock that pops scripted reply strings.

    Each entry is the full model output, optionally containing a trailing
    `<route exit="..." />` or `<route interrupt="..." />` tag. The runner
    strips the tag and dispatches via apply_route.
    """
    queue = list(turns)

    async def _mock(self) -> str:
        if not queue:
            raise AssertionError("Test ran out of scripted LLM turns")
        return queue.pop(0)

    return _mock


def _route_exit(exit_id: str, **captures: str) -> str:
    """Render an in-text route tag for an exit, with optional captures."""
    attrs = " ".join(f'{k}="{v}"' for k, v in captures.items())
    body = f'exit="{exit_id}"' + (f" {attrs}" if attrs else "")
    return f"<route {body} />"


def _route_interrupt(interrupt_id: str) -> str:
    return f'<route interrupt="{interrupt_id}" />'


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
        _scripted_inference(["Hi! What can I get started for you?"]),
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
async def test_turn_with_route_tag_transitions_flow_and_fires_assigns(
    coffee_spec, monkeypatch
):
    """User asks for a latte → LLM emits reply text + trailing route tag with
    drink_type capture → state transitions to flow_coffee_order, variable_set
    + exit_path_taken + flow_entered events fire."""
    script = [
        # Opening turn (chatbot_initiates)
        "Hi! What can I get started for you?",
        # User: "I'd love a latte" — reply + trailing route tag
        "Got it, a latte coming up. What size? "
        + _route_exit("xp_greet_to_coffee", drink_type="coffee"),
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
async def test_turn_without_route_tag_just_returns_text(coffee_spec, monkeypatch):
    """Plain reply, no route tag — should return text and stay in the active flow."""
    script = [
        "Hi! What can I get started?",
        "Sure, take your time.",
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    reply = await ts.turn("hmm let me think")
    assert reply == "Sure, take your time."
    assert ts.session.state.active_flow_id == "flow_greet"


@pytest.mark.asyncio
async def test_terminal_exit_ends_session(coffee_spec, monkeypatch):
    """Walkaway exit from greet flow ends the session — closing line streams
    before the tag, then SessionEnded fires immediately on tag parse."""
    script = [
        "Welcome, what'll it be?",
        # Closing line + walkaway tag (terminal)
        "No worries, come back anytime! " + _route_exit("xp_greet_walkaway"),
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    reply = await ts.turn("nevermind, I'll come back later")
    assert reply == "No worries, come back anytime!"
    assert ts.ended
    events = ts.drain_events()
    assert any(isinstance(e, SessionEnded) for e in events)


@pytest.mark.asyncio
async def test_end_is_idempotent_and_emits_session_ended_once(coffee_spec, monkeypatch):
    """end() called on a live session emits session_ended; second call is a no-op."""
    monkeypatch.setattr(
        TextSession, "_run_inference", _scripted_inference(["Hi!"])
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
async def test_route_tag_with_captures_passes_captures_to_assigns(
    coffee_spec, monkeypatch
):
    """Attributes on the route tag beyond `exit` are routed to llm-method
    assigns. (xp_greet_to_coffee's drink_type assign happens to be method=direct
    in coffee.json, so the captured value is overwritten by the direct value —
    that's expected. This test asserts the capture survived parsing.)"""
    captured_args: list[dict] = []
    real_apply_route = None

    from uxflows_runner.dispatcher import processor

    async def _wrap(session, tag):
        captured_args.append({"exit": tag.exit, "captures": tag.captures})
        return await real_apply_route(session, tag)

    real_apply_route = processor.apply_route
    monkeypatch.setattr("uxflows_runner.server.text_session.apply_route", _wrap)

    script = [
        "Welcome!",
        "Got it. " + _route_exit("xp_greet_to_coffee", drink_type="coffee"),
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()
    await ts.turn("a coffee please")

    # The captured tag had both exit and captures
    assert captured_args[-1]["exit"] == "xp_greet_to_coffee"
    assert captured_args[-1]["captures"] == {"drink_type": "coffee"}


@pytest.mark.asyncio
async def test_terminal_route_tag_speaks_closing_line_in_same_response(
    coffee_spec, monkeypatch
):
    """In-text routing folds the "speak closing line, then end" into a single
    response: the reply streams to TTS / the caller, then the trailing tag
    fires SessionEnded. No follow-up inference needed."""
    inference_count = [0]

    async def _mock(self) -> str:
        inference_count[0] += 1
        if inference_count[0] == 1:
            return "Welcome!"
        return "Bye, come back anytime! " + _route_exit("xp_greet_walkaway")

    monkeypatch.setattr(TextSession, "_run_inference", _mock)

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    reply = await ts.turn("nevermind")
    assert reply == "Bye, come back anytime!"
    # Two inferences total — opening + closing. No follow-up.
    assert inference_count[0] == 2
    assert ts.ended


@pytest.mark.asyncio
async def test_context_vars_seed_variable_bag_without_emitting_events(
    coffee_spec, monkeypatch
):
    """context_vars passed to TextSession.start get merged into the variable
    bag at session start, but do NOT emit `variable_set` events (those are
    semantically reserved for exit-path-fired assigns)."""
    monkeypatch.setattr(
        TextSession, "_run_inference", _scripted_inference(["Hi Maria!"])
    )

    ts, _ = await TextSession.start(
        spec=coffee_spec,
        api_key="dummy",
        context_vars={"customer_name": "Maria", "loyalty_tier": "gold"},
    )

    # Bag seeded
    assert ts.session.state.variables["customer_name"] == "Maria"
    assert ts.session.state.variables["loyalty_tier"] == "gold"

    # No variable_set events fired — only session start + flow_entered
    events = ts.drain_events()
    assert all(type(e).__name__ != "VariableSet" for e in events)


@pytest.mark.asyncio
async def test_context_vars_substituted_into_system_prompt_seen_by_llm(
    coffee_spec, monkeypatch
):
    """End-to-end: seeded context_vars should appear substituted in the
    system message of the LLMContext that's about to be sent to the model.
    Verifies both the seeding AND the prompt-builder substitution wire up."""
    captured_system: list[str] = []

    async def _capture(self) -> str:
        # Snapshot the system message at the moment of inference
        for msg in self.session.llm_context.messages:
            if msg.get("role") == "system":
                captured_system.append(msg["content"])
                break
        return "ok"

    monkeypatch.setattr(TextSession, "_run_inference", _capture)

    # coffee.json's agent.system_prompt doesn't have placeholders by default;
    # patch it to add one for this test.
    coffee_spec.agent.system_prompt = (
        coffee_spec.agent.system_prompt + " You are speaking with {customer_name}."
    )

    await TextSession.start(
        spec=coffee_spec,
        api_key="dummy",
        context_vars={"customer_name": "Maria"},
    )

    assert len(captured_system) == 1
    assert "speaking with Maria" in captured_system[0]
    assert "{customer_name}" not in captured_system[0]


@pytest.mark.asyncio
async def test_assistant_message_appended_as_clean_text(
    coffee_spec, monkeypatch
):
    """Assistant turns record the cleaned reply text (route tag stripped) on
    the conversation history. No tool_calls / tool-result messages — those
    were a tool-mode artifact."""
    script = [
        "Welcome!",
        "Coffee, got it. What size? " + _route_exit(
            "xp_greet_to_coffee", drink_type="coffee"
        ),
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    await ts.turn("a coffee please")

    msgs = ts.session.llm_context.messages
    # No tool_calls or tool-result rows — routing is in-text.
    assert all("tool_calls" not in m for m in msgs)
    assert not any(m.get("role") == "tool" for m in msgs)
    # The latest assistant message is the cleaned reply (no `<route` literal).
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    last = assistants[-1]
    assert "<route" not in last["content"]
    assert "Coffee, got it." in last["content"]


@pytest.mark.asyncio
async def test_event_log_dir_writes_session_jsonl(coffee_spec, monkeypatch, tmp_path):
    """When config.event_log_dir is set, events for a session are appended to
    {dir}/{session_id}.jsonl. Buffer-side `drain_events()` keeps working."""
    script = [
        "Welcome!",
        "Coffee, got it. " + _route_exit(
            "xp_greet_to_coffee", drink_type="coffee"
        ),
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    log_dir = tmp_path / "sessions"
    cfg = Config(
        google_credentials_path="",
        google_project_id="",
        google_location="us-east4",
        llm_model="gemini-2.5-flash",
        tts_voice="",
        host="127.0.0.1",
        port=8000,
        spec_path="",
        execution_config_path=None,
        event_log_dir=log_dir,
    )

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy", config=cfg)
    await ts.turn("a coffee please")
    await ts.end()

    expected = log_dir / f"{ts.session_id}.jsonl"
    assert expected.is_file()
    lines = expected.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    types = [p["type"] for p in parsed]
    # Must include the full arc — session_started ... session_ended
    assert types[0] == "session_started"
    assert types[-1] == "session_ended"
    assert "exit_path_taken" in types
    assert "variable_set" in types
    # Every record carries the same session_id
    assert all(p["session_id"] == ts.session_id for p in parsed)


@pytest.mark.asyncio
async def test_interrupt_then_return_via_route_tag(coffee_spec, monkeypatch):
    """Regression: trigger int_menu, take a follow-up turn inside it, then
    return via a route tag picking xp_int_menu_return.

    Each interrupt / return causes one follow-up inference (so the new flow's
    opener gets to speak this turn). That follow-up is just another scripted
    response in the queue.
    """
    script = [
        # Opening
        "Welcome! What can I get you?",
        # User asks "what do you have?" → reply with interrupt tag.
        "We've got drip, americano, latte for coffee, plus tea — all sizes. "
        + _route_interrupt("int_menu"),
        # Follow-up inside int_menu (apply_route triggered the follow-up).
        "What sounds good?",
        # User asks a follow-up inside the interrupt → plain text, no tag.
        "Latte's espresso with steamed milk — popular pick. Want one?",
        # User says "I'll have coffee" → return tag.
        "Coffee it is — coming up. " + _route_exit("xp_int_menu_return"),
        # Follow-up after the return (new active flow's opener).
        "What size would you like?",
    ]
    monkeypatch.setattr(TextSession, "_run_inference", _scripted_inference(script))

    ts, _ = await TextSession.start(spec=coffee_spec, api_key="dummy")
    ts.drain_events()

    # Trigger the interrupt — reply is the follow-up text (new flow's opener).
    reply = await ts.turn("what do you have?")
    assert reply == "What sounds good?"
    assert ts.session.state.active_flow_id == "int_menu"
    assert ts.session.state.has_caller
    ts.drain_events()

    # Multi-turn inside the interrupt — plain text, stays in int_menu.
    reply = await ts.turn("what's a latte?")
    assert reply == "Latte's espresso with steamed milk — popular pick. Want one?"
    assert ts.session.state.active_flow_id == "int_menu"
    assert ts.session.state.has_caller
    ts.drain_events()

    # Return tag → pop back to flow_greet, follow-up inference fires.
    reply = await ts.turn("I'll have coffee")
    assert reply == "What size would you like?"
    assert ts.session.state.active_flow_id == "flow_greet"
    assert not ts.session.state.has_caller

    events = ts.drain_events()
    fes = [e for e in events if isinstance(e, FlowEntered)]
    assert any(e.flow_id == "flow_greet" and e.via == "return" for e in fes)
