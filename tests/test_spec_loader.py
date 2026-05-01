"""Smoke-validate the v0 type/loader machinery against the coffee fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from uxflows_runner.spec.loader import GLOBAL_SCOPE_KEY, applicable_interrupts, load_spec

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
    globals_ = coffee_spec.interrupts_by_scope[GLOBAL_SCOPE_KEY]
    assert [f.id for f in globals_] == ["int_menu"]
    # any active flow should see int_menu as applicable
    for caller in ("flow_greet", "flow_coffee_order", "flow_confirm"):
        assert "int_menu" in {f.id for f in applicable_interrupts(coffee_spec, caller)}


def test_return_to_caller_has_no_condition(coffee_spec):
    int_menu = coffee_spec.flows_by_id["int_menu"]
    [exit_path] = int_menu.routing.exit_paths
    assert exit_path.type == "return_to_caller"
    assert exit_path.condition is None


def test_greet_routing_methods(coffee_spec):
    flow_greet = coffee_spec.flows_by_id["flow_greet"]
    methods = [ep.condition.method for ep in flow_greet.routing.exit_paths if ep.condition]
    # All three exits are llm-method now: coffee/tea routing is decided by
    # patron intent (semantic), not by a pre-set variable; walkaway too.
    assert methods == ["llm", "llm", "llm"]


def test_action_capability_id_resolves(coffee_spec):
    flow_confirm = coffee_spec.flows_by_id["flow_confirm"]
    placed = next(ep for ep in flow_confirm.routing.exit_paths if ep.id == "xp_cf_placed")
    assert placed.actions[0].capability_id == "cap_place_order"
    cap = coffee_spec.capabilities_by_id[placed.actions[0].capability_id]
    assert cap.name == "place_order"


def test_assigns_direct(coffee_spec):
    flow_confirm = coffee_spec.flows_by_id["flow_confirm"]
    placed = next(ep for ep in flow_confirm.routing.exit_paths if ep.id == "xp_cf_placed")
    assert placed.assigns["order_status"].method == "direct"
    assert placed.assigns["order_status"].value == "confirmed"
