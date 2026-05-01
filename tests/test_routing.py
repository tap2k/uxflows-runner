"""Routing — plan + resolve, the heart of the dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from uxflows_runner.dispatcher import routing
from uxflows_runner.spec.loader import load_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
COFFEE = REPO_ROOT / "examples" / "coffee.json"


@pytest.fixture(scope="module")
def coffee():
    return load_spec(COFFEE)


def test_plan_collects_all_llm_exits_on_greet(coffee):
    """flow_greet has three llm-method exits — coffee, tea, walkaway —
    decided by patron intent. All three are LLM candidates each turn."""
    plan = routing.plan(coffee, "flow_greet", {}, in_interrupt=False)
    assert plan.shortcut is None
    assert {ep.id for ep in plan.llm_exit_paths} == {
        "xp_greet_to_coffee",
        "xp_greet_to_tea",
        "xp_greet_walkaway",
    }


def test_plan_calc_shortcut_in_confirm_when_var_present():
    """Synthesize a flow with a calc exit: short-circuit fires."""
    from uxflows_runner.spec.types import (
        Agent, AgentMeta, Condition, ExitPath, Flow, Routing, Spec,
    )
    from uxflows_runner.spec.loader import _index

    flow = Flow(
        id="f",
        type="happy",
        routing=Routing(
            exit_paths=[
                ExitPath(
                    id="x_done",
                    type="happy",
                    condition=Condition(method="calculation", expression='status == "ok"'),
                    next_flow_id="f2",
                ),
                ExitPath(id="f2_marker", type="happy", condition=None, next_flow_id=None),
            ]
        ),
    )
    flow2 = Flow(id="f2", type="happy")
    spec = Spec(
        agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="f"),
        flows=[flow, flow2],
    )
    loaded = _index(spec, raw='{}')
    plan = routing.plan(loaded, "f", {"status": "ok"}, in_interrupt=False)
    assert plan.shortcut is not None
    assert plan.shortcut.exit_path.id == "x_done"


def test_plan_includes_global_interrupts(coffee):
    """int_menu is global; should appear as a candidate from any flow."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, in_interrupt=False)
    assert "int_menu" in {f.id for f in plan.llm_interrupts}


def test_plan_omits_interrupts_when_already_in_one(coffee):
    """No nested interrupts in v0 — once we're inside int_menu, don't offer
    int_menu again. (The plan's stack model evaluates scope against
    top-of-stack only; interrupts have empty scope effectively.)"""
    plan = routing.plan(coffee, "int_menu", {}, in_interrupt=True)
    assert plan.llm_interrupts == []


def test_plan_in_interrupt_offers_return_to_caller(coffee):
    """Inside int_menu, the only exit is return_to_caller — it should
    short-circuit immediately, no LLM call needed for routing."""
    plan = routing.plan(coffee, "int_menu", {}, in_interrupt=True)
    assert plan.shortcut is not None
    assert plan.shortcut.kind == "return_to_caller"


def test_resolve_uses_llm_take_exit(coffee):
    """LLM picks xp_greet_to_tea; resolve honors it."""
    plan = routing.plan(coffee, "flow_greet", {}, in_interrupt=False)
    decision = routing.resolve(
        plan,
        coffee,
        llm_results={"take_exit_path": {"exit_path_id": "xp_greet_to_tea"}},
    )
    assert decision.kind == "take_exit"
    assert decision.exit_path.id == "xp_greet_to_tea"


def test_resolve_uses_llm_pick_when_no_shortcut(coffee):
    """No calc shortcut — LLM picks the walkaway. resolve() should honor it."""
    plan = routing.plan(coffee, "flow_greet", {}, in_interrupt=False)
    decision = routing.resolve(
        plan, coffee, llm_results={"take_exit_path": {"exit_path_id": "xp_greet_walkaway"}}
    )
    assert decision.kind == "end"  # walkaway has next_flow_id=null, type=sad
    assert decision.exit_path.id == "xp_greet_walkaway"


def test_resolve_llm_trigger_interrupt(coffee):
    plan = routing.plan(coffee, "flow_coffee_order", {}, in_interrupt=False)
    decision = routing.resolve(
        plan, coffee, llm_results={"trigger_interrupt": {"interrupt_flow_id": "int_menu"}}
    )
    assert decision.kind == "trigger_interrupt"
    assert decision.target_flow_id == "int_menu"


def test_resolve_falls_through_to_stay(coffee):
    """LLM emitted neither tool call — dispatcher stays in the active flow."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, in_interrupt=False)
    decision = routing.resolve(plan, coffee, llm_results={})
    assert decision.kind == "stay"


def test_resolve_ignores_unknown_exit_id(coffee):
    """Hallucinated exit_path_id from the LLM should not crash; fall through."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, in_interrupt=False)
    decision = routing.resolve(
        plan, coffee, llm_results={"take_exit_path": {"exit_path_id": "made_up"}}
    )
    assert decision.kind == "stay"


def test_resolve_trigger_interrupt_beats_take_exit(coffee):
    """If the LLM emits both tool calls, the interrupt wins. Rationale: the
    user actually asked something off-path; the routing decision will get
    re-evaluated when the interrupt returns."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, in_interrupt=False)
    decision = routing.resolve(
        plan,
        coffee,
        llm_results={
            "take_exit_path": {"exit_path_id": "xp_co_to_confirm"},
            "trigger_interrupt": {"interrupt_flow_id": "int_menu"},
        },
    )
    assert decision.kind == "trigger_interrupt"
    assert decision.target_flow_id == "int_menu"


def test_resolve_interrupt_beats_shortcut():
    """A triggered interrupt is a topical detour; honor it even when the
    plan has a calc shortcut. The shortcut still applies on return-to-caller
    (re-evaluated on the next turn after pop)."""
    from uxflows_runner.spec.types import (
        Agent, AgentMeta, Condition, ExitPath, Flow, Routing, Spec,
    )
    from uxflows_runner.spec.loader import _index

    interrupt = Flow(
        id="int_x",
        type="interrupt",
        scope=["global"],
        routing=Routing(
            entry_condition=Condition(method="llm", expression="patron asks something"),
            exit_paths=[ExitPath(id="x_back", type="return_to_caller", next_flow_id=None)],
        ),
    )
    main_flow = Flow(
        id="f",
        type="happy",
        routing=Routing(
            exit_paths=[
                ExitPath(
                    id="x_done",
                    type="happy",
                    condition=Condition(method="calculation", expression='ready == True'),
                    next_flow_id="f2",
                ),
            ]
        ),
    )
    f2 = Flow(id="f2", type="happy")
    spec = Spec(
        agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="f"),
        flows=[main_flow, f2, interrupt],
    )
    loaded = _index(spec, raw='{}')

    plan = routing.plan(loaded, "f", {"ready": True}, in_interrupt=False)
    assert plan.shortcut is not None
    decision = routing.resolve(
        plan,
        loaded,
        llm_results={"trigger_interrupt": {"interrupt_flow_id": "int_x"}},
    )
    assert decision.kind == "trigger_interrupt"
    assert decision.target_flow_id == "int_x"


def test_force_max_turns_fallback_picks_unconditional_sad(coffee):
    """flow_coffee_order has xp_co_cancel (sad, llm-conditional). It does NOT
    qualify as 'unconditional sad' — so the helper should raise. This forces
    spec authors to provide a real fallback when they set max_turns."""
    flow_coffee = coffee.flows_by_id["flow_coffee_order"]
    with pytest.raises(RuntimeError):
        routing.force_max_turns_fallback(flow_coffee)
