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


def test_system_prompt_contains_agent_and_flow_pieces(coffee):
    flow = coffee.flows_by_id["flow_greet"]
    prompt = prompt_builder.build_system_prompt(coffee.agent, flow, lang="en-US")

    # Agent system prompt
    assert "barista at Bluebird Coffee" in prompt
    # Agent guardrails
    assert "Operating principles:" in prompt
    assert "Patrons want their coffee" in prompt
    # FAQ
    assert "Frequently asked:" in prompt
    assert "Do you have decaf?" in prompt
    # Glossary
    assert "Terminology:" in prompt
    assert "Drip:" in prompt
    # Flow section
    assert "This flow (flow_greet):" in prompt
    assert "Open warmly" in prompt
    # Sample lines
    assert "Sample lines" in prompt
    assert "Welcome to Bluebird" in prompt or "welcome to Bluebird" in prompt


def test_tools_includes_take_exit_and_trigger_interrupt_on_greet(coffee):
    """flow_greet now has all-llm exit conditions, plus a global interrupt.
    Both tools should appear in the per-turn schema."""
    plan = routing.plan(coffee, "flow_greet", {}, in_interrupt=False)
    tools = prompt_builder.build_tools(plan)
    assert tools is not None
    names = {t.name for t in tools.standard_tools}
    assert names == {"take_exit_path", "trigger_interrupt"}


def test_tools_includes_take_exit_path(coffee):
    plan = routing.plan(coffee, "flow_greet", {}, in_interrupt=False)
    tools = prompt_builder.build_tools(plan)
    assert tools is not None
    names = {t.name for t in tools.standard_tools}
    assert "take_exit_path" in names


def test_tools_includes_trigger_interrupt_when_applicable(coffee):
    plan = routing.plan(coffee, "flow_coffee_order", {}, in_interrupt=False)
    tools = prompt_builder.build_tools(plan)
    assert tools is not None
    names = {t.name for t in tools.standard_tools}
    assert "trigger_interrupt" in names
    interrupt_tool = next(t for t in tools.standard_tools if t.name == "trigger_interrupt")
    enum = interrupt_tool.properties["interrupt_flow_id"]["enum"]
    assert "int_menu" in enum


def test_take_exit_path_enum_includes_only_llm_paths(coffee):
    """flow_coffee_order has two llm exits (xp_co_to_confirm, xp_co_cancel)
    and no calc-shortcut on an empty bag. Both should be in the enum."""
    plan = routing.plan(coffee, "flow_coffee_order", {}, in_interrupt=False)
    tools = prompt_builder.build_tools(plan)
    take_tool = next(t for t in tools.standard_tools if t.name == "take_exit_path")
    enum = take_tool.properties["exit_path_id"]["enum"]
    assert set(enum) == {"xp_co_to_confirm", "xp_co_cancel"}


def test_no_tools_inside_interrupt(coffee):
    """Inside int_menu, the only exit is return_to_caller (unconditional, no
    LLM work). build_tools should return None."""
    plan = routing.plan(coffee, "int_menu", {}, in_interrupt=True)
    tools = prompt_builder.build_tools(plan)
    assert tools is None


def test_per_language_faq_used():
    from uxflows_runner.spec.types import (
        Agent, AgentKnowledge, AgentMeta, FAQEntry, Flow, Routing,
    )

    agent = Agent(
        id="ag",
        meta=AgentMeta(languages=["en-US", "es-ES"], modes=["voice"]),
        system_prompt="x",
        knowledge=AgentKnowledge(
            faq=[
                FAQEntry(
                    question="hours?",
                    answer="open 9-5",
                    scripts={"es-ES": "abierto de 9 a 17"},
                )
            ]
        ),
        entry_flow_id="f",
    )
    flow = Flow(id="f", type="happy", routing=Routing())
    es = prompt_builder.build_system_prompt(agent, flow, lang="es-ES")
    en = prompt_builder.build_system_prompt(agent, flow, lang="en-US")
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


def test_build_system_prompt_substitutes_across_multiple_sections(coffee):
    """Variables seeded into the variable bag get substituted in agent.system_prompt
    AND flow.scripts AND any other composed section. Verifies the substitution
    runs as a final pass over the joined prompt, not per-section."""
    from uxflows_runner.spec.types import (
        Agent,
        AgentMeta,
        Flow,
        Routing,
        Script,
    )

    agent = Agent(
        id="agent_x",
        version="0.1.0",
        meta=AgentMeta(name="X", purpose="p", client="c"),
        system_prompt="You are helping {customer_name}.",
        chatbot_initiates=True,
        entry_flow_id="f",
    )
    flow = Flow(
        id="f",
        type="happy",
        routing=Routing(),
        instructions="Greet {customer_name} warmly.",
        scripts={"en-US": [Script(id="s_welcome", text="Welcome, {customer_name}!")]},
    )

    prompt = prompt_builder.build_system_prompt(
        agent, flow, lang="en-US", variables={"customer_name": "Maria"}
    )
    assert "helping Maria" in prompt
    assert "Greet Maria warmly" in prompt
    assert "Welcome, Maria!" in prompt
    assert "{customer_name}" not in prompt
