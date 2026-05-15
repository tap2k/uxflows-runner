"""Streaming stripper tests for RouteTagFrameProcessor.

The stripper sits between the LLM and TTS. It must:

  1. Forward all conversational text to TTS in stream order.
  2. Swallow `<route .../>` tag bytes entirely (TTS must never speak them).
  3. Capture the parsed tag and dispatch via apply_route on LLMFullResponseEndFrame.

These tests exercise the state machine directly by feeding it streaming
LLMTextFrame chunks of various shapes — tag at end, tag mid-stream, tag
straddling chunk boundaries, stray `<`, malformed tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from uxflows_runner.dispatcher.processor import RouteTagFrameProcessor
from uxflows_runner.dispatcher.session import Session
from uxflows_runner.events.emitter import BufferingEventEmitter
from uxflows_runner.events.schema import Event, TurnCompleted
from uxflows_runner.spec.loader import load_spec


COFFEE_SPEC = Path(__file__).resolve().parents[1] / "examples" / "coffee.json"


class _StubLLMContext:
    """Minimal LLMContext stand-in for sessions in unit tests."""

    def __init__(self) -> None:
        self.messages: list[dict] = [{"role": "system", "content": "stub"}]

    def set_tools(self, _tools: Any) -> None:
        pass


@dataclass
class _Capture:
    """Records what the stripper forwarded downstream + which route tag it
    captured. Replaces the `push_frame` plumbing for testing."""

    forwarded_text: list[str]
    other_frames: list[Frame]
    applied_routes: list[Any]


@pytest.fixture
def coffee_session() -> Session:
    spec = load_spec(str(COFFEE_SPEC))
    return Session.start(spec, _StubLLMContext(), events=BufferingEventEmitter())


async def _drive(stripper: RouteTagFrameProcessor, chunks: list[str]) -> _Capture:
    """Feed text chunks + a final response-end through the stripper, capturing
    what would have been forwarded downstream."""
    cap = _Capture(forwarded_text=[], other_frames=[], applied_routes=[])

    async def _push(frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMTextFrame):
            cap.forwarded_text.append(frame.text)
        else:
            cap.other_frames.append(frame)

    # Monkey-patch push_frame so we capture rather than queue.
    stripper.push_frame = _push  # type: ignore[method-assign]

    # Also intercept apply_route so we don't actually mutate session state in
    # these unit tests (the dispatcher's own tests cover routing).
    from uxflows_runner.dispatcher import processor as proc_mod

    async def _capture_route(_session, tag) -> None:
        cap.applied_routes.append(tag)
        return None

    proc_mod_apply_route_orig = proc_mod.apply_route
    proc_mod.apply_route = _capture_route  # type: ignore[assignment]
    try:
        for chunk in chunks:
            await stripper.process_frame(LLMTextFrame(text=chunk), FrameDirection.DOWNSTREAM)
        await stripper.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    finally:
        proc_mod.apply_route = proc_mod_apply_route_orig  # type: ignore[assignment]
    return cap


# --------------------------------------------------------------------------
# Happy path: tag at end of response
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_at_end_is_stripped_and_captured(coffee_session):
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(
        stripper,
        ['Hello, what can I get you? <route exit="xp_greet_to_coffee" />'],
    )
    spoken = "".join(cap.forwarded_text)
    assert "Hello, what can I get you?" in spoken
    assert "<route" not in spoken
    assert len(cap.applied_routes) == 1
    assert cap.applied_routes[0].exit == "xp_greet_to_coffee"


@pytest.mark.asyncio
async def test_tag_split_across_chunk_boundary_is_stripped(coffee_session):
    """Tag bytes are split across LLMTextFrame boundaries — the stripper
    must buffer until it has the whole tag."""
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(
        stripper,
        ["Got it. ", '<route exit=', '"xp_greet_to_coffee"', " />"],
    )
    spoken = "".join(cap.forwarded_text)
    assert spoken.strip() == "Got it."
    assert len(cap.applied_routes) == 1
    assert cap.applied_routes[0].exit == "xp_greet_to_coffee"


@pytest.mark.asyncio
async def test_no_tag_passes_through_untouched(coffee_session):
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(stripper, ["Sure, take your time."])
    assert "".join(cap.forwarded_text) == "Sure, take your time."
    assert cap.applied_routes == []


@pytest.mark.asyncio
async def test_stray_less_than_is_eventually_flushed(coffee_session):
    """A `<` that turns out not to be a route tag should still reach TTS —
    just slightly delayed. Critical: no bytes are lost."""
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(stripper, ["You owe me <$5> tomorrow."])
    spoken = "".join(cap.forwarded_text)
    assert spoken == "You owe me <$5> tomorrow."
    assert cap.applied_routes == []


@pytest.mark.asyncio
async def test_html_tag_in_reply_is_not_swallowed(coffee_session):
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(stripper, ["I said <strong>yes</strong>."])
    spoken = "".join(cap.forwarded_text)
    assert spoken == "I said <strong>yes</strong>."
    assert cap.applied_routes == []


# --------------------------------------------------------------------------
# Malformed tags fail closed (no route applied, bytes survive)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_tag_does_not_dispatch(coffee_session):
    """A `<route ... />` with neither exit nor interrupt is invalid — no
    route should fire. The tag bytes still don't reach TTS (the stripper
    swallows them as well-formed-but-empty)."""
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(stripper, ["Reply text. <route />"])
    spoken = "".join(cap.forwarded_text)
    assert "Reply text." in spoken
    assert "<route" not in spoken
    assert cap.applied_routes == []


# --------------------------------------------------------------------------
# TurnCompleted event on end-of-response
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_completed_emitted_with_clean_text(coffee_session):
    """The agent's TurnCompleted event records the spoken text — the tag
    must NOT appear in it."""
    stripper = RouteTagFrameProcessor(coffee_session)
    await _drive(
        stripper,
        ['All set. <route exit="xp_greet_to_coffee" />'],
    )
    events = coffee_session.events.drain()
    turn_completed = [e for e in events if isinstance(e, TurnCompleted)]
    assert len(turn_completed) == 1
    assert turn_completed[0].text == "All set."
    assert "<route" not in turn_completed[0].text


# --------------------------------------------------------------------------
# Multiple tags in one response — first wins
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_tags_only_first_dispatches(coffee_session):
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(
        stripper,
        ['<route exit="xp_a" /> middle <route exit="xp_b" />'],
    )
    assert len(cap.applied_routes) == 1
    assert cap.applied_routes[0].exit == "xp_a"


# --------------------------------------------------------------------------
# `<think>...</think>` reserved sentinel
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_think_block_is_swallowed_entirely(coffee_session):
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(
        stripper,
        ["Before <think>reasoning text</think> after."],
    )
    spoken = "".join(cap.forwarded_text)
    assert "<think" not in spoken
    assert "reasoning text" not in spoken
    assert "Before" in spoken
    assert "after." in spoken


@pytest.mark.asyncio
async def test_think_block_split_across_chunks(coffee_session):
    """The `<think>` open, contents, and `</think>` close arrive in
    separate streamed chunks. None of them reach TTS."""
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(
        stripper,
        ["Hi. <think>", "step ", "one", "</think>", " There."],
    )
    spoken = "".join(cap.forwarded_text)
    assert "<think" not in spoken
    assert "step" not in spoken
    assert "Hi." in spoken
    assert "There." in spoken


@pytest.mark.asyncio
async def test_think_then_route_tag_both_work(coffee_session):
    """A response that uses both the `<think>` sentinel for reasoning and
    a trailing `<route>` tag for dispatch. Both are stripped, the route
    fires."""
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(
        stripper,
        [
            "Sure. <think>route to coffee</think> Coming up. "
            '<route exit="xp_greet_to_coffee" />'
        ],
    )
    spoken = "".join(cap.forwarded_text)
    assert "<think" not in spoken
    assert "route to coffee" not in spoken
    assert "<route" not in spoken
    assert "Sure." in spoken
    assert "Coming up." in spoken
    assert len(cap.applied_routes) == 1
    assert cap.applied_routes[0].exit == "xp_greet_to_coffee"


@pytest.mark.asyncio
async def test_other_tags_pass_through(coffee_session):
    """Only `<route>` and `<think>` are reserved. Any other tag (here a
    `<VERIFICATION>` block, mimicking the Tala collection spec's prompt
    pattern) flows through to TTS unchanged. The spec author is responsible
    for not using such tags when the voice channel will speak them aloud."""
    stripper = RouteTagFrameProcessor(coffee_session)
    cap = await _drive(
        stripper,
        ["<VERIFICATION>steps</VERIFICATION> ok"],
    )
    spoken = "".join(cap.forwarded_text)
    assert "<VERIFICATION>" in spoken
    assert "</VERIFICATION>" in spoken
    assert "ok" in spoken
