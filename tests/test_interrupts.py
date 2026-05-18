"""End-to-end interrupt + per-frame turn-counter coverage.

Drives the dispatcher through synthetic turns by:
  1. Building a small spec with interrupts.
  2. Constructing a Session against a stub LLMContext + NullEventEmitter.
  3. Simulating per-turn flow: PreLLMPlanner-equivalent, then the tool handler
     by calling _apply_decision (the same code path real handlers run).

Bypasses Pipecat — these tests prove the dispatcher's stack semantics, not
the frame plumbing. Live behavior is covered by the manual browser tests
documented in RUNNER-PLAN §"Live-test follow-up".
"""

from __future__ import annotations

import pytest

from uxflows_runner.dispatcher import routing
from uxflows_runner.dispatcher.processor import _apply_decision
from uxflows_runner.dispatcher.session import Session
from uxflows_runner.events.emitter import NullEventEmitter
from uxflows_runner.spec.loader import _index
from uxflows_runner.spec.types import (
    Agent,
    AgentMeta,
    Condition,
    ExitPath,
    Flow,
    Spec,
)


class _StubLLMContext:
    """Minimal LLMContext stand-in. The dispatcher only mutates `messages`
    and calls `set_tools(...)`; we don't care what either does in tests."""

    def __init__(self):
        self.messages = [{"role": "system", "content": "init"}]
        self.tools = None

    def set_tools(self, tools):
        self.tools = tools


def _build_spec(
    main_flows: list[Flow],
    interrupts: list[Flow],
    entry: str,
) -> "routing.LoadedSpec":
    spec = Spec(
        agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id=entry),
        flows=[*main_flows, *interrupts],
    )
    return _index(spec, raw="{}")


def _session_for(spec) -> Session:
    return Session.start(
        spec=spec,
        llm_context=_StubLLMContext(),
        events=NullEventEmitter(),
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def globally_scoped_spec():
    """flow_a (entry) → flow_b. int_global is an interrupt (implicitly global)."""
    main_a = Flow(
        id="flow_a",
        type="happy",
        exit_paths=[
            ExitPath(
                id="ax",
                goto="flow_b",
                condition=Condition(method="llm", expression="user is ready"),
            )
        ],
    )
    main_b = Flow(id="flow_b", type="happy")
    interrupt = Flow(
        id="int_global",
        type="interrupt",
        entry_condition=Condition(method="llm", expression="user asked a side q"),
        exit_paths=[ExitPath(id="x_back", goto="RETURN")],
    )
    return _build_spec([main_a, main_b], [interrupt], entry="flow_a")


# --------------------------------------------------------------------------
# Global-scope interrupt round-trip
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_global_interrupt_push_and_return(globally_scoped_spec):
    s = _session_for(globally_scoped_spec)
    assert s.state.active_flow_id == "flow_a"

    # Plan + LLM picks the interrupt
    plan = routing.plan(globally_scoped_spec, "flow_a", {}, has_caller=False)
    s.current_plan = plan
    decision = routing.resolve(
        plan,
        globally_scoped_spec,
        llm_results={"trigger_interrupt": {"interrupt_flow_id": "int_global"}},
    )
    await _apply_decision(decision, s)
    assert s.state.active_flow_id == "int_global"
    assert s.state.has_caller
    assert s.state.active.caller_flow_id == "flow_a"

    # Inside the interrupt: only exit is RETURN, surfaced as an
    # LLM-driven take_exit_path candidate. LLM picks it when ready to return.
    plan2 = routing.plan(globally_scoped_spec, "int_global", {}, has_caller=True)
    s.current_plan = plan2
    assert plan2.shortcut is None
    assert {ep.id for ep in plan2.llm_exit_paths} == {"x_back"}
    decision2 = routing.resolve(
        plan2,
        globally_scoped_spec,
        llm_results={"take_exit_path": {"exit_path_id": "x_back"}},
    )
    assert decision2.kind == "return"
    await _apply_decision(decision2, s)
    assert s.state.active_flow_id == "flow_a"
    assert not s.state.has_caller


# --------------------------------------------------------------------------
# Interrupts are implicitly global; visibility doesn't depend on caller id
# --------------------------------------------------------------------------


def test_interrupt_visible_from_any_flow(globally_scoped_spec):
    plan_a = routing.plan(globally_scoped_spec, "flow_a", {}, has_caller=False)
    assert "int_global" in {f.id for f in plan_a.llm_interrupts}
    plan_b = routing.plan(globally_scoped_spec, "flow_b", {}, has_caller=False)
    assert "int_global" in {f.id for f in plan_b.llm_interrupts}


def test_no_interrupts_offered_inside_an_interrupt(globally_scoped_spec):
    """Inside an interrupt, further interrupts aren't offered (we'd nest)."""
    plan = routing.plan(globally_scoped_spec, "int_global", {}, has_caller=True)
    assert plan.llm_interrupts == []


# --------------------------------------------------------------------------
# Entry condition methods: calc/direct short-circuit, llm becomes a candidate
# --------------------------------------------------------------------------


def test_calc_entry_condition_fires_without_llm():
    """An interrupt with calculation-method entry_condition fires when its
    expression evaluates True against the variable bag — no LLM needed."""
    main = Flow(id="flow_a", type="happy")
    interrupt = Flow(
        id="int_alarm",
        type="interrupt",
        entry_condition=Condition(
            method="calculation", expression="alarm == True"
        ),
        exit_paths=[ExitPath(id="b", goto="RETURN")],
    )
    spec = _build_spec([main], [interrupt], entry="flow_a")

    plan = routing.plan(spec, "flow_a", {"alarm": True}, has_caller=False)
    assert plan.shortcut is not None
    assert plan.shortcut.kind == "trigger_interrupt"
    assert plan.shortcut.target_flow_id == "int_alarm"

    plan_off = routing.plan(spec, "flow_a", {"alarm": False}, has_caller=False)
    # Calc-false: still shows as a candidate via the LLM bucket? No — calc
    # branches don't fall back to llm. It's just absent.
    assert plan_off.shortcut is None
    assert plan_off.llm_interrupts == []


# --------------------------------------------------------------------------
# Decision precedence — interrupts beat shortcuts (live-test rule)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_interrupt_beats_calc_shortcut_in_resolve():
    """Live rule from RUNNER-PLAN: a triggered interrupt wins even when a
    calc shortcut was already chosen by plan(). The shortcut applies on the
    next turn after the RETURN."""
    main = Flow(
        id="flow_a",
        type="happy",
        exit_paths=[
            ExitPath(
                id="ready_exit",
                goto="flow_b",
                condition=Condition(method="calculation", expression="ready == True"),
            )
        ],
    )
    flow_b = Flow(id="flow_b", type="happy")
    interrupt = Flow(
        id="int_pause",
        type="interrupt",
        entry_condition=Condition(method="llm", expression="user pauses"),
        exit_paths=[ExitPath(id="b", goto="RETURN")],
    )
    spec = _build_spec([main, flow_b], [interrupt], entry="flow_a")
    s = _session_for(spec)

    plan = routing.plan(spec, "flow_a", {"ready": True}, has_caller=False)
    s.current_plan = plan
    assert plan.shortcut is not None and plan.shortcut.kind == "take_exit"

    # LLM emits the interrupt — should win
    decision = routing.resolve(
        plan, spec, {"trigger_interrupt": {"interrupt_flow_id": "int_pause"}}
    )
    assert decision.kind == "trigger_interrupt"
    await _apply_decision(decision, s)
    assert s.state.active_flow_id == "int_pause"


# --------------------------------------------------------------------------
# Callable destination: any flow with a RETURN exit pushes a frame on entry
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callable_utility_flow_pushes_call_frame():
    """A non-interrupt flow with a `goto: RETURN` exit is callable: entering
    it pushes a frame; taking its RETURN exit pops back to the caller."""
    main = Flow(
        id="flow_a",
        type="happy",
        exit_paths=[
            ExitPath(
                id="call",
                goto="util",
                condition=Condition(method="llm", expression="time to call"),
            ),
        ],
    )
    util = Flow(
        id="util",
        type="utility",
        exit_paths=[ExitPath(id="done", goto="RETURN")],
    )
    spec = _build_spec([main, util], [], entry="flow_a")
    s = _session_for(spec)

    plan = routing.plan(spec, "flow_a", {}, has_caller=False)
    s.current_plan = plan
    decision = routing.resolve(
        plan, spec, {"take_exit_path": {"exit_path_id": "call"}}
    )
    assert decision.kind == "take_exit"
    await _apply_decision(decision, s, llm_results={"take_exit_path": {"exit_path_id": "call"}})
    # Frame pushed (caller is flow_a)
    assert s.state.active_flow_id == "util"
    assert s.state.has_caller  # = "has a caller"

    # Inside util: RETURN exit pops back to flow_a
    plan2 = routing.plan(spec, "util", {}, has_caller=True)
    s.current_plan = plan2
    decision2 = routing.resolve(
        plan2, spec, {"take_exit_path": {"exit_path_id": "done"}}
    )
    assert decision2.kind == "return"
    await _apply_decision(decision2, s)
    assert s.state.active_flow_id == "flow_a"
    assert not s.state.has_caller


# --------------------------------------------------------------------------
# RETURN with no caller frame collapses to END
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_return_at_top_level_collapses_to_end():
    """Per schema: RETURN from a top-level frame (nothing to pop) behaves as
    END. In-text routing tears down immediately (the closing line streamed
    before the route tag, so no defer is needed)."""
    main = Flow(
        id="flow_a",
        type="happy",
        exit_paths=[ExitPath(id="bye", goto="RETURN")],
    )
    spec = _build_spec([main], [], entry="flow_a")
    s = _session_for(spec)

    plan = routing.plan(spec, "flow_a", {}, has_caller=False)
    s.current_plan = plan
    decision = routing._build_take_exit(main, main.exit_paths[0])
    assert decision.kind == "return"
    await _apply_decision(decision, s)
    assert s.ended
    assert s.state.stack == []
