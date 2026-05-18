"""Prompt + per-turn tool schema composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from uxflows_runner.dispatcher import prompt_builder, routing
from uxflows_runner.dispatcher.prompt_builder import substitute_variables
from uxflows_runner.spec.loader import load_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
COFFEE = REPO_ROOT / "examples" / "coffee.json"


@pytest.fixture(scope="module")
def coffee():
    return load_spec(COFFEE)


def test_routing_protocol_lists_exits_and_interrupts_on_greet(coffee):
    """flow_greet has three llm-method exits plus a global interrupt
    (int_menu). All four should appear in the rendered routing protocol
    section, identified by their ids."""
    flow = coffee.flows_by_id["flow_greet"]
    plan = routing.plan(coffee, "flow_greet", {}, has_caller=False)
    prompt = prompt_builder.build_system_prompt(coffee, flow, lang="en-US", plan=plan)
    assert "emit a route tag" in prompt
    assert "xp_greet_to_coffee" in prompt
    assert "xp_greet_to_tea" in prompt
    assert "xp_greet_walkaway" in prompt
    assert "int_menu" in prompt


def test_routing_protocol_omitted_when_no_plan(coffee):
    """At session start (no plan yet) the routing section is skipped — the
    LLM just talks until the first turn produces a plan."""
    flow = coffee.flows_by_id["flow_greet"]
    prompt = prompt_builder.build_system_prompt(coffee, flow, lang="en-US")
    assert "emit a route tag" not in prompt


def test_routing_protocol_inside_interrupt_offers_return(coffee):
    """Inside int_menu the only exit is a RETURN — render it as a route
    candidate. Editor-canonical coffee.json has no condition on the return
    exit, so the protocol falls back to the default return intent prose."""
    flow = coffee.flows_by_id["int_menu"]
    plan = routing.plan(coffee, "int_menu", {}, has_caller=True)
    prompt = prompt_builder.build_system_prompt(coffee, flow, lang="en-US", plan=plan)
    assert "xp_int_menu_return" in prompt
    # Default fallback intent for an unconditional RETURN exit, from
    # prompt_builder._exit_path_intent.
    assert "side conversation is naturally complete" in prompt


def test_routing_protocol_render_includes_destinations(coffee):
    """Each exit in the routing section names the destination flow so the
    LLM can reason about where each path leads."""
    flow = coffee.flows_by_id["flow_greet"]
    plan = routing.plan(coffee, "flow_greet", {}, has_caller=False)
    prompt = prompt_builder.build_system_prompt(coffee, flow, lang="en-US", plan=plan)
    # Destination is rendered as "→ <flow.name>" (or id fallback).
    coffee_order = coffee.flows_by_id["flow_coffee_order"]
    assert f"→ {coffee_order.name or 'flow_coffee_order'}" in prompt


def test_per_language_faq_used():
    from uxflows_runner.spec.loader import _index
    from uxflows_runner.spec.types import (
        Agent,
        AgentKnowledge,
        AgentMeta,
        FAQEntry,
        Flow,
        Spec,
    )

    agent = Agent(
        id="ag",
        meta=AgentMeta(languages=["en-US", "es-ES"], modes=["voice"]),
        knowledge=AgentKnowledge(
            faq=[
                FAQEntry(
                    id="faq_hours",
                    question="hours?",
                    answer={"en-US": "open 9-5", "es-ES": "abierto de 9 a 17"},
                )
            ]
        ),
        entry_flow_id="f",
    )
    flow = Flow(id="f", type="happy")
    spec = _index(Spec(agent=agent, flows=[flow]), raw="{}")
    es = prompt_builder.build_system_prompt(spec, flow, lang="es-ES")
    en = prompt_builder.build_system_prompt(spec, flow, lang="en-US")
    assert "abierto de 9 a 17" in es
    assert "open 9-5" in en
    assert "abierto" not in en


# --------------------------------------------------------------------------
# Variable substitution (whatsupp2-parity {KEY} placeholder semantics)
# --------------------------------------------------------------------------


def test_substitute_variables_replaces_filled_keys():
    out = substitute_variables(
        "Hi {customer_name}, you have ${balance}.",
        {"customer_name": "Maria", "balance": 5000},
    )
    assert out == "Hi Maria, you have $5000."


def test_substitute_variables_leaves_unfilled_placeholders_literal():
    """Missing-key, None-valued, and empty-string-valued placeholders all
    stay as `{KEY}` literal — better to surface unfilled state than silently
    substitute something else (matches whatsupp2's resolvePromptVariables)."""
    template = "Hello {customer_name}, your account is {account} and tier is {tier}."
    out = substitute_variables(
        template, {"customer_name": "Maria", "account": "", "tier": None}
    )
    assert out == "Hello Maria, your account is {account} and tier is {tier}."


def test_substitute_variables_is_case_insensitive():
    out = substitute_variables(
        "Hi {Customer_Name} aka {CUSTOMER_NAME} aka {customer_name}.",
        {"customer_name": "Maria"},
    )
    assert out == "Hi Maria aka Maria aka Maria."


def test_substitute_variables_handles_none_and_empty_inputs():
    assert substitute_variables("hi {name}", None) == "hi {name}"
    assert substitute_variables("hi {name}", {}) == "hi {name}"
    assert substitute_variables("", {"name": "x"}) == ""


# --------------------------------------------------------------------------
# Variable substitution in rendered condition expressions
# --------------------------------------------------------------------------


def test_routing_protocol_substitutes_session_variables():
    """Session variables substitute inside condition expressions when rendered
    into the routing protocol section — same {placeholder} mechanism that
    runs over scripts and instructions. Lets a spec author write a date /
    amount-aware routing gate like 'on or before {extended_loan_due_date}'
    and have the LLM see the resolved value at decision time."""
    from uxflows_runner.spec.loader import _index
    from uxflows_runner.spec.types import (
        Agent,
        AgentMeta,
        Condition,
        ExitPath,
        Flow,
        Spec,
    )

    main_flow = Flow(
        id="f",
        type="happy",
        exit_paths=[
            ExitPath(
                id="xp_in_grace",
                goto="END",
                condition=Condition(
                    method="llm",
                    expression="customer's date is on or before {extended_loan_due_date}",
                ),
            )
        ],
    )
    spec = _index(
        Spec(
            agent=Agent(
                id="ag", meta=AgentMeta(modes=["voice"]), entry_flow_id="f"
            ),
            flows=[main_flow],
        ),
        raw="{}",
    )
    plan = routing.plan(spec, "f", {}, has_caller=False)
    prompt = prompt_builder.build_system_prompt(
        spec,
        main_flow,
        lang="en-US",
        variables={"extended_loan_due_date": "2026-06-12"},
        plan=plan,
    )
    assert "on or before 2026-06-12" in prompt
    assert "{extended_loan_due_date}" not in prompt


def test_build_system_prompt_substitutes_across_multiple_sections(coffee):
    """Variables seeded into the variable bag get substituted in the
    synthesized role line (meta.purpose / meta.tone) AND flow.scripts AND any
    other composed section. Verifies substitution runs as a final pass over
    the joined prompt, not per-section."""
    from uxflows_runner.spec.loader import _index
    from uxflows_runner.spec.types import (
        Agent,
        AgentMeta,
        Flow,
        Script,
        Spec,
    )

    agent = Agent(
        id="agent_x",
        version="0.1.0",
        meta=AgentMeta(name="X", purpose="Help {customer_name} with their order.", client="c"),
        chatbot_initiates=True,
        entry_flow_id="f",
    )
    flow = Flow(
        id="f",
        type="happy",
        instructions="Greet {customer_name} warmly.",
        scripts=[Script(id="s_welcome", text="Welcome, {customer_name}!")],
    )
    spec = _index(Spec(agent=agent, flows=[flow]), raw="{}")

    prompt = prompt_builder.build_system_prompt(
        spec, flow, lang="en-US", variables={"customer_name": "Maria"}
    )
    assert "Help Maria with their order" in prompt
    assert "Greet Maria warmly" in prompt
    assert "Welcome, Maria!" in prompt
    assert "{customer_name}" not in prompt
