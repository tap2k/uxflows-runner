"""Smoke-validate the v0 type/loader machinery against the coffee fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from uxflows_runner.spec.loader import applicable_interrupts, load_spec
from uxflows_runner.spec.types import is_return_goto

REPO_ROOT = Path(__file__).resolve().parent.parent
COFFEE = REPO_ROOT / "examples" / "coffee.json"


@pytest.fixture(scope="module")
def coffee_spec():
    return load_spec(COFFEE)


def test_loads_coffee(coffee_spec):
    assert coffee_spec.agent.id == "agent_bluebird_coffee"
    assert coffee_spec.agent.entry_flow_id == "flow_greet"
    assert coffee_spec.entry_flow.id == "flow_greet"


def test_flow_index(coffee_spec):
    expected = {"flow_greet", "flow_coffee_order", "flow_tea_order", "flow_confirm", "int_menu"}
    assert set(coffee_spec.flows_by_id.keys()) == expected


def test_capability_index(coffee_spec):
    assert set(coffee_spec.capabilities_by_name.keys()) == {"place_order", "log_walkaway"}
    assert set(coffee_spec.capabilities_by_id.keys()) == {"cap_place_order", "cap_log_walkaway"}


def test_global_interrupt_indexed(coffee_spec):
    globals_ = coffee_spec.global_interrupts
    assert [f.id for f in globals_] == ["int_menu"]
    # Interrupts are implicitly globally callable.
    assert "int_menu" in {f.id for f in applicable_interrupts(coffee_spec)}


def test_return_exit_carries_condition_expression(coffee_spec):
    """Spec authors can put a `condition.expression` on a RETURN exit to tell
    the LLM when to return — same idiom as forward exits. The runner surfaces
    it in the take_exit_path tool description."""
    int_menu = coffee_spec.flows_by_id["int_menu"]
    [exit_path] = int_menu.exit_paths
    assert is_return_goto(exit_path.goto)
    assert exit_path.condition is not None
    assert exit_path.condition.method == "llm"
    assert "named a drink" in exit_path.condition.expression


def test_greet_routing_methods(coffee_spec):
    flow_greet = coffee_spec.flows_by_id["flow_greet"]
    methods = [ep.condition.method for ep in flow_greet.exit_paths if ep.condition]
    # All three exits are llm-method now: coffee/tea routing is decided by
    # patron intent (semantic), not by a pre-set variable; walkaway too.
    assert methods == ["llm", "llm", "llm"]


def test_action_capability_id_resolves(coffee_spec):
    flow_confirm = coffee_spec.flows_by_id["flow_confirm"]
    placed = next(ep for ep in flow_confirm.exit_paths if ep.id == "xp_cf_placed")
    assert placed.actions[0].capability_id == "cap_place_order"
    cap = coffee_spec.capabilities_by_id[placed.actions[0].capability_id]
    assert cap.name == "place_order"


def test_assigns_direct(coffee_spec):
    flow_confirm = coffee_spec.flows_by_id["flow_confirm"]
    placed = next(ep for ep in flow_confirm.exit_paths if ep.id == "xp_cf_placed")
    assert placed.assigns["order_status"].method == "direct"
    assert placed.assigns["order_status"].value == "confirmed"
