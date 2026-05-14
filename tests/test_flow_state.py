"""FlowState stack semantics — the load-bearing model for interrupts."""

from __future__ import annotations

import pytest

from uxflows_runner.dispatcher.flow_state import FlowState


def test_starts_with_entry_flow():
    s = FlowState.start("flow_greet")
    assert s.active_flow_id == "flow_greet"
    assert s.active.turn_count == 0
    assert not s.has_caller


def test_increment_turn_only_on_active_frame():
    s = FlowState.start("flow_greet")
    s.increment_turn()
    s.increment_turn()
    assert s.active.turn_count == 2

    s.push_call("int_menu")
    assert s.active.turn_count == 0  # fresh frame
    s.increment_turn()
    assert s.active.turn_count == 1

    # Caller's counter untouched (the rule that lets interrupted turns not
    # increment the caller's per-frame turn count).
    s.pop_to_caller()
    assert s.active_flow_id == "flow_greet"
    assert s.active.turn_count == 2


def test_push_call_records_caller():
    s = FlowState.start("flow_coffee_order")
    frame = s.push_call("int_menu")
    assert frame.flow_id == "int_menu"
    assert frame.caller_flow_id == "flow_coffee_order"
    assert s.has_caller


def test_pop_to_caller_returns_popped_and_caller():
    s = FlowState.start("flow_coffee_order")
    s.push_call("int_menu")
    popped, caller = s.pop_to_caller()
    assert popped.flow_id == "int_menu"
    assert caller.flow_id == "flow_coffee_order"
    assert not s.has_caller


def test_pop_outside_interrupt_raises():
    s = FlowState.start("flow_greet")
    with pytest.raises(RuntimeError):
        s.pop_to_caller()


def test_transition_replaces_active():
    s = FlowState.start("flow_greet")
    s.increment_turn()
    s.transition("flow_coffee_order")
    assert s.active_flow_id == "flow_coffee_order"
    assert s.active.turn_count == 0  # fresh counter on new flow
    assert len(s.stack) == 1


def test_nested_interrupts():
    s = FlowState.start("flow_a")
    s.push_call("int_x")
    s.push_call("int_y")  # legal: nested interrupt
    assert [f.flow_id for f in s.stack] == ["flow_a", "int_x", "int_y"]
    s.pop_to_caller()
    assert s.active_flow_id == "int_x"
    s.pop_to_caller()
    assert s.active_flow_id == "flow_a"
    assert not s.has_caller


def test_end_clears_stack():
    s = FlowState.start("flow_greet")
    s.end()
    assert s.stack == []
    with pytest.raises(RuntimeError):
        _ = s.active
