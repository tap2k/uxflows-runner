"""Direct tests for processor.apply_route + apply_planned_shortcut.

These cover the seam between voice and text modes — the place where parsed
route tags become state mutations and events. Voice (RouteTagFrameProcessor)
and text (TextSession) both funnel through apply_route, so testing it
directly avoids needing the full Pipecat pipeline.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from uxflows_runner.dispatcher import capabilities as caps
from uxflows_runner.dispatcher.processor import (
    PostLLMResolver,
    apply_planned_shortcut,
    apply_route,
)
from uxflows_runner.dispatcher.routing_protocol import RouteTag
from uxflows_runner.dispatcher.session import Session
from uxflows_runner.events.emitter import BufferingEventEmitter
from uxflows_runner.events.schema import (
    CapabilityInvoked,
    CapabilityReturned,
    ExitPathTaken,
    FlowEntered,
    FlowExited,
    SessionEnded,
    VariableSet,
)
from uxflows_runner.spec.loader import _index
from uxflows_runner.spec.types import (
    Action,
    Agent,
    AgentMeta,
    Assign,
    Capability,
    ExitPath,
    Flow,
    Spec,
)


class _StubLLMContext:
    def __init__(self) -> None:
        self.messages: list[dict] = [{"role": "system", "content": "stub"}]

    def set_tools(self, _tools: Any) -> None:
        pass


def _session(spec) -> Session:
    return Session.start(spec, _StubLLMContext(), events=BufferingEventEmitter())


@pytest.mark.asyncio
async def test_take_exit_emits_events_in_order():
    """apply_route on a take_exit tag emits VariableSet → ExitPathTaken →
    FlowExited → FlowEntered, in that order. Downstream consumers (the editor
    canvas) rely on this ordering for live highlighting."""
    a = Flow(
        id="a",
        type="happy",
        exit_paths=[
            ExitPath(
                id="x_done",
                goto="b",
                condition=None,
                assigns={"status": Assign(method="direct", value="ok")},
            )
        ],
    )
    b = Flow(id="b", type="happy")
    spec = _index(
        Spec(agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="a"), flows=[a, b]),
        raw="{}",
    )
    s = _session(spec)
    # Plan must be set — apply_route guards against parse-before-plan.
    from uxflows_runner.dispatcher import routing as routing_mod
    s.current_plan = routing_mod.plan(spec, "a", {}, has_caller=False)

    await apply_route(s, RouteTag(exit="x_done"))

    types = [type(e).__name__ for e in s.events.drain()]
    # Skip the SessionStarted / FlowEntered(entry) from Session.start
    transition = types[types.index("VariableSet"):]
    assert transition == ["VariableSet", "ExitPathTaken", "FlowExited", "FlowEntered"]
    assert s.state.active_flow_id == "b"


@pytest.mark.asyncio
async def test_capture_from_route_tag_flows_into_llm_assign():
    """Captures on a route tag become llm-method assign values — this is the
    end-to-end wire from `<route exit=... drink_style="latte" />` through
    routing_protocol.to_llm_results to assigns.fire. Coffee spec uses a
    direct-method assign so this can't be verified there; we need a synthetic
    spec with an llm-method assign."""
    a = Flow(
        id="a",
        type="happy",
        exit_paths=[
            ExitPath(
                id="x_done",
                goto="b",
                condition=None,
                assigns={"drink_style": Assign(method="llm", value=None)},
            )
        ],
    )
    b = Flow(id="b", type="happy")
    spec = _index(
        Spec(agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="a"), flows=[a, b]),
        raw="{}",
    )
    s = _session(spec)
    from uxflows_runner.dispatcher import routing as routing_mod
    s.current_plan = routing_mod.plan(spec, "a", {}, has_caller=False)

    await apply_route(s, RouteTag(exit="x_done", captures={"drink_style": "latte"}))

    assert s.state.variables["drink_style"] == "latte"
    var_set = next(e for e in s.events.drain() if isinstance(e, VariableSet))
    assert var_set.variable_name == "drink_style"
    assert var_set.value == "latte"
    assert var_set.method == "llm"


@pytest.mark.asyncio
async def test_terminator_shortcut_fires_session_ended():
    """A terminator flow (single unconditional exit to END) gets its shortcut
    pre-resolved at plan time. apply_planned_shortcut fires it — without this,
    terminator flows loop forever (the Tala wrong-number repeat bug)."""
    closer = Flow(
        id="closer",
        type="sad",
        exit_paths=[ExitPath(id="x_end", goto="END", condition=None)],
    )
    spec = _index(
        Spec(agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="closer"), flows=[closer]),
        raw="{}",
    )
    s = _session(spec)
    from uxflows_runner.dispatcher import routing as routing_mod
    s.current_plan = routing_mod.plan(spec, "closer", {}, has_caller=False)

    assert await apply_planned_shortcut(s) is True
    assert s.ended

    types = [type(e).__name__ for e in s.events.drain()]
    assert "ExitPathTaken" in types
    assert "FlowExited" in types
    assert "SessionEnded" in types
    # FlowExited must precede SessionEnded so consumers see the source flow
    # close before the session record terminates.
    assert types.index("FlowExited") < types.index("SessionEnded")


@pytest.mark.asyncio
async def test_apply_route_ignored_when_no_plan():
    """apply_route is a no-op if the LLM emitted a tag before PreLLMPlanner
    set the plan — defensive guard, otherwise resolve() blows up."""
    a = Flow(id="a", type="happy")
    spec = _index(
        Spec(agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="a"), flows=[a]),
        raw="{}",
    )
    s = _session(spec)
    # No s.current_plan set
    result = await apply_route(s, RouteTag(exit="x_anything"))
    assert result is None
    assert not s.ended


@pytest.mark.asyncio
async def test_capability_outputs_land_before_transition():
    """Capability dispatch is synchronous and runs BEFORE the FlowExited/
    FlowEntered transition events fire. Rationale (RUNNER-PLAN §"Capability
    outputs bind to variable scope"): declared outputs must be in the variable
    bag by the time the next flow's prompt is built, otherwise downstream
    calculation conditions see stale state."""
    cap = Capability(
        id="cap_log",
        name="log_event",
        kind="function",
        inputs=[],
        outputs=["receipt_id"],
    )
    a = Flow(
        id="a",
        type="happy",
        exit_paths=[
            ExitPath(
                id="x_done",
                goto="b",
                condition=None,
                actions=[Action(capability_id="cap_log")],
            )
        ],
    )
    b = Flow(id="b", type="happy")
    spec = _index(
        Spec(
            agent=Agent(
                id="ag",
                meta=AgentMeta(modes=["voice"]),
                entry_flow_id="a",
                capabilities=[cap],
            ),
            flows=[a, b],
        ),
        raw="{}",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"receipt_id": "R-42"})

    transport = httpx.MockTransport(lambda req: handler(req))
    client = httpx.AsyncClient(transport=transport)
    dispatcher = caps.CapabilityDispatcher(
        spec=spec,
        endpoints={"log_event": caps.CapabilityEndpoint(url="https://example.test/log")},
        client=client,
    )

    s = Session.start(spec, _StubLLMContext(), events=BufferingEventEmitter(), capabilities=dispatcher)
    from uxflows_runner.dispatcher import routing as routing_mod
    s.current_plan = routing_mod.plan(spec, "a", {}, has_caller=False)

    await apply_route(s, RouteTag(exit="x_done"))
    await dispatcher.aclose()

    # Output landed in the variable bag
    assert s.state.variables["receipt_id"] == "R-42"

    types = [type(e).__name__ for e in s.events.drain()]
    # Capability must be invoked + returned BEFORE the transition closes the flow.
    inv = types.index("CapabilityInvoked")
    ret = types.index("CapabilityReturned")
    xpt = types.index("ExitPathTaken")
    fx = types.index("FlowExited")
    assert inv < ret < xpt < fx
    # Capability's output is recorded as a VariableSet too (method="capability").
    assert "VariableSet" in types


# --------------------------------------------------------------------------
# PostLLMResolver — the no-tag-fired backstop
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_llm_resolver_short_circuits_when_route_already_fired():
    """If apply_route already ran this turn (tool_handler_fired_this_turn=True),
    PostLLMResolver must NOT double-fire the shortcut or double-increment the
    turn counter. This gate is what prevents the silent double-routing bug
    that motivated the in-text protocol."""
    from pipecat.frames.frames import LLMFullResponseEndFrame
    from pipecat.processors.frame_processor import FrameDirection

    closer = Flow(
        id="closer",
        type="sad",
        exit_paths=[ExitPath(id="x_end", goto="END", condition=None)],
    )
    spec = _index(
        Spec(agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="closer"), flows=[closer]),
        raw="{}",
    )
    s = _session(spec)
    from uxflows_runner.dispatcher import routing as routing_mod
    s.current_plan = routing_mod.plan(spec, "closer", {}, has_caller=False)
    s.tool_handler_fired_this_turn = True  # apply_route already ran

    resolver = PostLLMResolver(s)
    pushed: list = []
    resolver.push_frame = lambda frame, direction: pushed.append(frame) or _async_noop()  # type: ignore[method-assign,assignment]

    await resolver.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    # No shortcut fired (session not ended), turn counter not incremented.
    assert not s.ended
    assert s.state.active.turn_count == 0


@pytest.mark.asyncio
async def test_post_llm_resolver_fires_shortcut_when_no_route_fired():
    """Plain reply with no route tag in a terminator flow — PostLLMResolver
    is the backstop that ends the session via the planned shortcut."""
    from pipecat.frames.frames import LLMFullResponseEndFrame
    from pipecat.processors.frame_processor import FrameDirection

    closer = Flow(
        id="closer",
        type="sad",
        exit_paths=[ExitPath(id="x_end", goto="END", condition=None)],
    )
    spec = _index(
        Spec(agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="closer"), flows=[closer]),
        raw="{}",
    )
    s = _session(spec)
    from uxflows_runner.dispatcher import routing as routing_mod
    s.current_plan = routing_mod.plan(spec, "closer", {}, has_caller=False)
    s.tool_handler_fired_this_turn = False

    resolver = PostLLMResolver(s)
    resolver.push_frame = lambda frame, direction: _async_noop()  # type: ignore[method-assign,assignment]

    await resolver.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    assert s.ended
    assert any(isinstance(e, SessionEnded) for e in s.events.drain())


async def _async_noop() -> None:
    return None


# --------------------------------------------------------------------------
# PreLLMPlanner — per-turn replan + turn-flag reset
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_llm_planner_replans_and_resets_turn_flag():
    """On each LLMContextFrame downstream, PreLLMPlanner must (a) rebuild
    current_plan against the active flow and (b) reset
    tool_handler_fired_this_turn so PostLLMResolver's gate works on the
    next turn. Both are load-bearing — if the flag stays True across
    turns, terminator shortcuts never fire from the backstop."""
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.frame_processor import FrameDirection

    from uxflows_runner.dispatcher.processor import PreLLMPlanner

    a = Flow(
        id="a",
        type="happy",
        exit_paths=[ExitPath(id="x", goto="b", condition=None)],
    )
    b = Flow(id="b", type="happy")
    spec = _index(
        Spec(agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="a"), flows=[a, b]),
        raw="{}",
    )
    s = _session(spec)
    s.tool_handler_fired_this_turn = True  # stale flag from previous turn
    assert s.current_plan is None

    planner = PreLLMPlanner(s)
    planner.push_frame = lambda frame, direction: _async_noop()  # type: ignore[method-assign,assignment]

    await planner.process_frame(
        LLMContextFrame(context=s.llm_context), FrameDirection.DOWNSTREAM
    )

    assert s.current_plan is not None
    assert s.current_plan.active_flow.id == "a"
    assert s.tool_handler_fired_this_turn is False
