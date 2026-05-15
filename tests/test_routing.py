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
    plan = routing.plan(coffee, "flow_greet", {}, has_caller=False)
    assert plan.shortcut is None
    assert {ep.id for ep in plan.llm_exit_paths} == {
        "xp_greet_to_coffee",
        "xp_greet_to_tea",
        "xp_greet_walkaway",
    }


def test_plan_calc_shortcut_in_confirm_when_var_present():
    """Synthesize a flow with a calc exit: short-circuit fires."""
    from uxflows_runner.spec.loader import _index
    from uxflows_runner.spec.types import (
        Agent,
        AgentMeta,
        Condition,
        ExitPath,
        Flow,
        Spec,
    )

    flow = Flow(
        id="f",
        type="happy",
        exit_paths=[
            ExitPath(
                id="x_done",
                goto="f2",
                condition=Condition(method="calculation", expression='status == "ok"'),
            ),
            ExitPath(id="f2_marker", goto="END", condition=None),
        ],
    )
    flow2 = Flow(id="f2", type="happy")
    spec = Spec(
        agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="f"),
        flows=[flow, flow2],
    )
    loaded = _index(spec, raw='{}')
    plan = routing.plan(loaded, "f", {"status": "ok"}, has_caller=False)
    assert plan.shortcut is not None
    assert plan.shortcut.exit_path.id == "x_done"


def test_plan_terminator_flow_fires_unconditional_exit_as_shortcut():
    """A flow whose only exits are unconditional is a pure terminator: speak
    the script, then leave. The planner pre-resolves the shortcut so the
    runner can fire END after the LLM produces its closing line, instead of
    looping the flow forever. Regression for the Tala wrong-number repeat bug."""
    from uxflows_runner.spec.loader import _index
    from uxflows_runner.spec.types import (
        Agent,
        AgentMeta,
        ExitPath,
        Flow,
        Spec,
    )

    flow = Flow(
        id="closer",
        type="sad",
        exit_paths=[ExitPath(id="x_end", goto="END", condition=None)],
    )
    spec = Spec(
        agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="closer"),
        flows=[flow],
    )
    loaded = _index(spec, raw="{}")
    plan = routing.plan(loaded, "closer", {}, has_caller=False)
    assert plan.shortcut is not None
    assert plan.shortcut.kind == "end"
    assert plan.shortcut.exit_path.id == "x_end"
    assert plan.llm_exit_paths == []


def test_plan_includes_global_interrupts(coffee):
    """int_menu is global; should appear as a candidate from any flow."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, has_caller=False)
    assert "int_menu" in {f.id for f in plan.llm_interrupts}


def test_plan_omits_interrupts_when_already_in_one(coffee):
    """No nested interrupts in v0 — once we're inside int_menu, don't offer
    int_menu again."""
    plan = routing.plan(coffee, "int_menu", {}, has_caller=True)
    assert plan.llm_interrupts == []


def test_plan_with_caller_offers_return_path(coffee):
    """Inside int_menu, the only exit is a `goto: RETURN` — surface it as an
    LLM-driven take_exit_path candidate so the LLM picks when to return."""
    plan = routing.plan(coffee, "int_menu", {}, has_caller=True)
    assert plan.shortcut is None
    assert {ep.id for ep in plan.llm_exit_paths} == {"xp_int_menu_return"}


def test_resolve_return_via_take_exit_path(coffee):
    """LLM picks the return path via take_exit_path; resolve builds a
    return Decision (not a take_exit)."""
    plan = routing.plan(coffee, "int_menu", {}, has_caller=True)
    decision = routing.resolve(
        plan,
        coffee,
        llm_results={"take_exit_path": {"exit_path_id": "xp_int_menu_return"}},
    )
    assert decision.kind == "return"
    assert decision.exit_path.id == "xp_int_menu_return"


def test_resolve_stays_inside_callable_when_llm_does_not_call_tool(coffee):
    """No tool call inside the interrupt → stay (multi-turn side conversation
    OK; LLM decides when to fire the return)."""
    plan = routing.plan(coffee, "int_menu", {}, has_caller=True)
    decision = routing.resolve(plan, coffee, llm_results={})
    assert decision.kind == "stay"


def test_resolve_uses_llm_take_exit(coffee):
    """LLM picks xp_greet_to_tea; resolve honors it."""
    plan = routing.plan(coffee, "flow_greet", {}, has_caller=False)
    decision = routing.resolve(
        plan,
        coffee,
        llm_results={"take_exit_path": {"exit_path_id": "xp_greet_to_tea"}},
    )
    assert decision.kind == "take_exit"
    assert decision.exit_path.id == "xp_greet_to_tea"


def test_resolve_uses_llm_pick_when_no_shortcut(coffee):
    """No calc shortcut — LLM picks the walkaway. resolve() should honor it."""
    plan = routing.plan(coffee, "flow_greet", {}, has_caller=False)
    decision = routing.resolve(
        plan, coffee, llm_results={"take_exit_path": {"exit_path_id": "xp_greet_walkaway"}}
    )
    assert decision.kind == "end"  # walkaway has goto=END
    assert decision.exit_path.id == "xp_greet_walkaway"


def test_resolve_llm_trigger_interrupt(coffee):
    plan = routing.plan(coffee, "flow_coffee_order", {}, has_caller=False)
    decision = routing.resolve(
        plan, coffee, llm_results={"trigger_interrupt": {"interrupt_flow_id": "int_menu"}}
    )
    assert decision.kind == "trigger_interrupt"
    assert decision.target_flow_id == "int_menu"


def test_resolve_falls_through_to_stay(coffee):
    """LLM emitted neither tool call — dispatcher stays in the active flow."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, has_caller=False)
    decision = routing.resolve(plan, coffee, llm_results={})
    assert decision.kind == "stay"


def test_resolve_ignores_unknown_exit_id(coffee):
    """Hallucinated exit_path_id from the LLM should not crash; fall through."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, has_caller=False)
    decision = routing.resolve(
        plan, coffee, llm_results={"take_exit_path": {"exit_path_id": "made_up"}}
    )
    assert decision.kind == "stay"


def test_resolve_trigger_interrupt_beats_take_exit(coffee):
    """If the LLM emits both tool calls, the interrupt wins. Rationale: the
    user actually asked something off-path; the routing decision will get
    re-evaluated when the interrupt returns."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, has_caller=False)
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
    plan has a calc shortcut. The shortcut still applies on RETURN
    (re-evaluated on the next turn after pop)."""
    from uxflows_runner.spec.loader import _index
    from uxflows_runner.spec.types import (
        Agent,
        AgentMeta,
        Condition,
        ExitPath,
        Flow,
        Spec,
    )

    interrupt = Flow(
        id="int_x",
        type="interrupt",
        entry_condition=Condition(method="llm", expression="patron asks something"),
        exit_paths=[ExitPath(id="x_back", goto="RETURN")],
    )
    main_flow = Flow(
        id="f",
        type="happy",
        exit_paths=[
            ExitPath(
                id="x_done",
                goto="f2",
                condition=Condition(method="calculation", expression='ready == True'),
            ),
        ],
    )
    f2 = Flow(id="f2", type="happy")
    spec = Spec(
        agent=Agent(id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="f"),
        flows=[main_flow, f2, interrupt],
    )
    loaded = _index(spec, raw='{}')

    plan = routing.plan(loaded, "f", {"ready": True}, has_caller=False)
    assert plan.shortcut is not None
    decision = routing.resolve(
        plan,
        loaded,
        llm_results={"trigger_interrupt": {"interrupt_flow_id": "int_x"}},
    )
    assert decision.kind == "trigger_interrupt"
    assert decision.target_flow_id == "int_x"


