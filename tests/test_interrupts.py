"""End-to-end interrupt + max_turns coverage.

Drives the dispatcher through synthetic turns by:
  1. Building a small spec with both global and scoped interrupts.
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
    Routing,
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
    """flow_a (entry) → flow_b. int_global is global-scope, llm-triggered."""
    main_a = Flow(
        id="flow_a",
        type="happy",
        routing=Routing(
            exit_paths=[
                ExitPath(
                    id="ax",
                    type="happy",
                    condition=Condition(method="llm", expression="user is ready"),
                    next_flow_id="flow_b",
                )
            ]
        ),
    )
    main_b = Flow(id="flow_b", type="happy")
    interrupt = Flow(
        id="int_global",
        type="interrupt",
        scope=["global"],
        routing=Routing(
            entry_condition=Condition(method="llm", expression="user asked a side q"),
            exit_paths=[ExitPath(id="x_back", type="return_to_caller", next_flow_id=None)],
        ),
    )
    return _build_spec([main_a, main_b], [interrupt], entry="flow_a")


@pytest.fixture
def scoped_spec():
    """int_only_a is scoped to flow_a only — should NOT fire from flow_b."""
    main_a = Flow(
        id="flow_a",
        type="happy",
        routing=Routing(
            exit_paths=[
                ExitPath(
                    id="ax",
                    type="happy",
                    condition=Condition(method="llm", expression="user is ready"),
                    next_flow_id="flow_b",
                )
            ]
        ),
    )
    main_b = Flow(id="flow_b", type="happy")
    interrupt = Flow(
        id="int_only_a",
        type="interrupt",
        scope=["flow_a"],
        routing=Routing(
            entry_condition=Condition(method="llm", expression="trigger phrase"),
            exit_paths=[ExitPath(id="x_back", type="return_to_caller", next_flow_id=None)],
        ),
    )
    return _build_spec([main_a, main_b], [interrupt], entry="flow_a")


# --------------------------------------------------------------------------
# Global-scope interrupt round-trip
# --------------------------------------------------------------------------


async def test_global_interrupt_push_and_return(globally_scoped_spec):
    s = _session_for(globally_scoped_spec)
    assert s.state.active_flow_id == "flow_a"

    # Plan + LLM picks the interrupt
    plan = routing.plan(globally_scoped_spec, "flow_a", {}, in_interrupt=False)
    s.current_plan = plan
    decision = routing.resolve(
        plan,
        globally_scoped_spec,
        llm_results={"trigger_interrupt": {"interrupt_flow_id": "int_global"}},
    )
    await _apply_decision(decision, s)
    assert s.state.active_flow_id == "int_global"
    assert s.state.is_in_interrupt
    assert s.state.active.caller_flow_id == "flow_a"

    # Inside the interrupt: only exit is return_to_caller, plan short-circuits
    plan2 = routing.plan(globally_scoped_spec, "int_global", {}, in_interrupt=True)
    s.current_plan = plan2
    assert plan2.shortcut is not None
    assert plan2.shortcut.kind == "return_to_caller"
    await _apply_decision(plan2.shortcut, s)
    assert s.state.active_flow_id == "flow_a"
    assert not s.state.is_in_interrupt


# --------------------------------------------------------------------------
# Scoped interrupts: only fire on matching caller
# --------------------------------------------------------------------------


def test_scoped_interrupt_visible_in_a(scoped_spec):
    plan = routing.plan(scoped_spec, "flow_a", {}, in_interrupt=False)
    assert "int_only_a" in {f.id for f in plan.llm_interrupts}


def test_scoped_interrupt_invisible_in_b(scoped_spec):
    plan = routing.plan(scoped_spec, "flow_b", {}, in_interrupt=False)
    assert plan.llm_interrupts == []


def test_no_interrupts_offered_inside_an_interrupt(globally_scoped_spec):
    """Per RUNNER-PLAN: scope matches against top-of-stack only. Once we're
    inside int_global, no further interrupts are offered (they'd nest)."""
    plan = routing.plan(globally_scoped_spec, "int_global", {}, in_interrupt=True)
    assert plan.llm_interrupts == []


# --------------------------------------------------------------------------
# Nested interrupts: stack model survives push/push/pop/pop
# --------------------------------------------------------------------------


async def test_nested_interrupts_via_stack():
    """Simulate two interrupts pushed in sequence — even though normal scope
    rules omit interrupts when inside one, the stack itself tolerates nesting
    if we forced it (defensive against future spec patterns)."""
    a = Flow(id="flow_a", type="happy")
    int1 = Flow(
        id="int1",
        type="interrupt",
        scope=["global"],
        routing=Routing(
            entry_condition=Condition(method="llm", expression="t1"),
            exit_paths=[ExitPath(id="b1", type="return_to_caller", next_flow_id=None)],
        ),
    )
    int2 = Flow(
        id="int2",
        type="interrupt",
        scope=["global"],
        routing=Routing(
            entry_condition=Condition(method="llm", expression="t2"),
            exit_paths=[ExitPath(id="b2", type="return_to_caller", next_flow_id=None)],
        ),
    )
    spec = _build_spec([a], [int1, int2], entry="flow_a")
    s = _session_for(spec)

    # Push int1 from flow_a
    plan = routing.plan(spec, "flow_a", {}, in_interrupt=False)
    s.current_plan = plan
    await _apply_decision(
        routing.resolve(
            plan, spec, {"trigger_interrupt": {"interrupt_flow_id": "int1"}}
        ),
        s,
    )
    assert [f.flow_id for f in s.state.stack] == ["flow_a", "int1"]

    # Force-push int2 from int1 (skipping the in_interrupt=True guard, mimicking
    # a hypothetical future where nested interrupts are allowed)
    s.state.push_interrupt("int2")
    assert [f.flow_id for f in s.state.stack] == ["flow_a", "int1", "int2"]

    # Pop both
    s.state.pop_to_caller()
    assert s.state.active_flow_id == "int1"
    s.state.pop_to_caller()
    assert s.state.active_flow_id == "flow_a"
    assert not s.state.is_in_interrupt


# --------------------------------------------------------------------------
# Interrupted turns DO NOT count toward caller's max_turns
# --------------------------------------------------------------------------


async def test_interrupted_turns_dont_count_toward_max_turns(globally_scoped_spec):
    s = _session_for(globally_scoped_spec)

    # Two normal turns in flow_a — counter ticks
    s.state.increment_turn()
    s.state.increment_turn()
    assert s.state.active.turn_count == 2

    # Interrupt fires — caller's counter must not advance during interrupt turns
    plan = routing.plan(globally_scoped_spec, "flow_a", {}, in_interrupt=False)
    s.current_plan = plan
    await _apply_decision(
        routing.resolve(
            plan, globally_scoped_spec,
            {"trigger_interrupt": {"interrupt_flow_id": "int_global"}},
        ),
        s,
    )
    # Inside interrupt; tick a turn here — only the interrupt frame's counter moves
    s.state.increment_turn()
    s.state.increment_turn()
    interrupt_turns = s.state.active.turn_count
    assert interrupt_turns == 2

    # Pop back; caller still at 2
    s.state.pop_to_caller()
    assert s.state.active_flow_id == "flow_a"
    assert s.state.active.turn_count == 2


# --------------------------------------------------------------------------
# max_turns with unconditional sad fallback fires on exhaustion
# --------------------------------------------------------------------------


def test_max_turns_force_fallback_picks_unconditional_sad():
    flow = Flow(
        id="flow_a",
        type="happy",
        max_turns=3,
        routing=Routing(
            exit_paths=[
                # llm-conditional happy exit (does NOT qualify as fallback)
                ExitPath(
                    id="happy",
                    type="happy",
                    condition=Condition(method="llm", expression="user is done"),
                    next_flow_id=None,
                ),
                # unconditional sad — the convention: condition is None
                ExitPath(id="sad_fallback", type="sad", condition=None, next_flow_id=None),
            ]
        ),
    )
    decision = routing.force_max_turns_fallback(flow)
    assert decision.kind in ("take_exit", "end")
    assert decision.exit_path.id == "sad_fallback"


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
        scope=["global"],
        routing=Routing(
            entry_condition=Condition(
                method="calculation", expression="alarm == True"
            ),
            exit_paths=[ExitPath(id="b", type="return_to_caller", next_flow_id=None)],
        ),
    )
    spec = _build_spec([main], [interrupt], entry="flow_a")

    plan = routing.plan(spec, "flow_a", {"alarm": True}, in_interrupt=False)
    assert plan.shortcut is not None
    assert plan.shortcut.kind == "trigger_interrupt"
    assert plan.shortcut.target_flow_id == "int_alarm"

    plan_off = routing.plan(spec, "flow_a", {"alarm": False}, in_interrupt=False)
    # Calc-false: still shows as a candidate via the LLM bucket? No — calc
    # branches don't fall back to llm. It's just absent.
    assert plan_off.shortcut is None
    assert plan_off.llm_interrupts == []


# --------------------------------------------------------------------------
# Decision precedence — interrupts beat shortcuts (live-test rule)
# --------------------------------------------------------------------------


async def test_trigger_interrupt_beats_calc_shortcut_in_resolve():
    """Live rule from RUNNER-PLAN: a triggered interrupt wins even when a
    calc shortcut was already chosen by plan(). The shortcut applies on the
    next turn after pop_to_caller."""
    main = Flow(
        id="flow_a",
        type="happy",
        routing=Routing(
            exit_paths=[
                ExitPath(
                    id="ready_exit",
                    type="happy",
                    condition=Condition(method="calculation", expression="ready == True"),
                    next_flow_id="flow_b",
                )
            ]
        ),
    )
    flow_b = Flow(id="flow_b", type="happy")
    interrupt = Flow(
        id="int_pause",
        type="interrupt",
        scope=["global"],
        routing=Routing(
            entry_condition=Condition(method="llm", expression="user pauses"),
            exit_paths=[ExitPath(id="b", type="return_to_caller", next_flow_id=None)],
        ),
    )
    spec = _build_spec([main, flow_b], [interrupt], entry="flow_a")
    s = _session_for(spec)

    plan = routing.plan(spec, "flow_a", {"ready": True}, in_interrupt=False)
    s.current_plan = plan
    assert plan.shortcut is not None and plan.shortcut.kind == "take_exit"

    # LLM emits the interrupt — should win
    decision = routing.resolve(
        plan, spec, {"trigger_interrupt": {"interrupt_flow_id": "int_pause"}}
    )
    assert decision.kind == "trigger_interrupt"
    await _apply_decision(decision, s)
    assert s.state.active_flow_id == "int_pause"
