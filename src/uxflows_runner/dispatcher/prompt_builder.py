"""Compose the per-flow system prompt and the per-turn tool schema.

System prompt (RUNNER-PLAN §"v0 dispatcher feature scope"):
  agent.system_prompt
  + agent.guardrails (Operating principles)
  + agent.knowledge.faq (Frequently asked, with per-language scripts when present)
  + agent.knowledge.glossary (Terminology)
  + flow.instructions (This flow)
  + flow.guardrails
  + flow.knowledge.faq
  + flow.scripts[lang] (sample lines)

Tool schema (RUNNER-PLAN §"Gemini tool-call shape"):
  - One `take_exit_path` FunctionSchema with an `exit_path_id` parameter
    enumerated over the LLM-method exit paths in the routing plan, plus
    one parameter per llm-method assign on those exit paths (string-typed;
    typed-parameter v0.5).
  - One `trigger_interrupt` FunctionSchema (only when interrupts are
    applicable on this turn), with `interrupt_flow_id` enum.

Both are emitted via Pipecat's provider-neutral `FunctionSchema` /
`ToolsSchema`; provider adapters in pipecat-ai translate to Gemini's format.
"""

from __future__ import annotations

import re
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from uxflows_runner.spec.types import Agent, ExitPath, Flow

from .routing import RoutingPlan


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def substitute_variables(text: str, variables: dict[str, Any] | None) -> str:
    """Replace {KEY} placeholders against `variables` (case-insensitive).

    Mirrors whatsupp2's resolvePromptVariables semantics: unfilled placeholders
    (missing key OR empty/null value) stay as `{KEY}` literal in the output.
    Better to see the unfilled placeholder in the agent's prompt than to
    silently substitute something else — designers debugging "why did the
    agent ignore the customer name" can see `{customer_name}` is unfilled
    at a glance.
    """
    if not variables or not text:
        return text
    lower = {k.lower(): v for k, v in variables.items()}

    def _replace(match: re.Match) -> str:
        v = lower.get(match.group(1).lower())
        if v is None or v == "":
            return match.group(0)
        return str(v)

    return _PLACEHOLDER.sub(_replace, text)


def build_system_prompt(
    agent: Agent,
    flow: Flow,
    lang: str,
    variables: dict[str, Any] | None = None,
) -> str:
    sections: list[str] = []
    if agent.system_prompt:
        sections.append(agent.system_prompt.strip())

    if agent.guardrails:
        sections.append(
            "Operating principles:\n"
            + "\n".join(f"- {g.statement}" for g in agent.guardrails)
        )

    if agent.knowledge.faq:
        sections.append("Frequently asked:\n" + _format_faq(agent.knowledge.faq, lang))

    if agent.knowledge.glossary:
        sections.append(
            "Terminology:\n"
            + "\n".join(f"- {g.term}: {g.definition}" for g in agent.knowledge.glossary)
        )

    sections.append(f"This flow ({flow.id}):")
    if flow.instructions:
        sections.append(flow.instructions.strip())

    if flow.guardrails:
        sections.append(
            "For this flow specifically:\n"
            + "\n".join(f"- {g.statement}" for g in flow.guardrails)
        )

    if flow.knowledge and flow.knowledge.faq:
        sections.append("Flow-specific FAQ:\n" + _format_faq(flow.knowledge.faq, lang))

    scripts = flow.scripts.get(lang) or []
    if scripts:
        lines: list[str] = []
        for s in scripts:
            lines.append(f"- {s.text}")
            for v in s.variations:
                if v:
                    lines.append(f"  | {v}")
        sections.append(
            f"Sample lines you might use ({lang}) — paraphrase, don't recite verbatim:\n"
            + "\n".join(lines)
        )

    # Two short tool-use reminders. Both compensate for Gemini behaviors we
    # observed during live testing: (1) AUTO mode sometimes emits a tool call
    # alone, leaving the patron in silence; (2) when tools are present, the
    # model can over-index on "I should be calling one" and stall on
    # off-path questions that have no matching tool. Resist adding a third —
    # that's the signal the prompt structure needs refactoring.
    sections.append(
        "Always speak naturally to the patron when calling a tool — they hear "
        "your reply, not the tool call. You can also answer without calling "
        "any tool; call tools only when their description fits."
    )

    return substitute_variables("\n\n".join(sections), variables)


def build_tools(
    plan: RoutingPlan,
    variables: dict[str, Any] | None = None,
) -> ToolsSchema | None:
    """Return the per-turn tool schema, or None when there's no LLM routing
    work to do this turn (the caller can skip `tools` entirely on the LLM
    call to save tokens). When None, the LLM still produces text — it just
    has no routing decisions to emit.

    `variables` are substituted into `{key}` placeholders inside each rendered
    condition expression before it's inlined into a tool description — so a
    spec author can write a routing gate like
      "customer's committed date is on or before {extended_loan_due_date}"
    and the LLM sees the resolved value at decision time.
    """
    declarations: list[FunctionSchema] = []

    if plan.llm_exit_paths:
        declarations.append(_take_exit_path_schema(plan, variables))

    if plan.llm_interrupts:
        declarations.append(_trigger_interrupt_schema(plan, variables))

    if not declarations:
        return None
    return ToolsSchema(standard_tools=declarations)


def _take_exit_path_schema(
    plan: RoutingPlan,
    variables: dict[str, Any] | None,
) -> FunctionSchema:
    exit_ids = [ep.id for ep in plan.llm_exit_paths]
    descriptions = "\n".join(
        f"- {ep.id}: {substitute_variables(_exit_path_intent(ep), variables)}"
        for ep in plan.llm_exit_paths
    )

    properties: dict[str, dict] = {
        "exit_path_id": {
            "type": "string",
            "enum": exit_ids,
            "description": (
                "Pick the exit path that matches the patron's current state. "
                "Choose only when the conversation has clearly reached one of "
                "these conditions; otherwise keep talking and don't call this "
                "tool.\n\nAvailable exit paths:\n" + descriptions
            ),
        }
    }

    # Per-exit-path llm-method assigns become parameters. Multiple exit paths
    # may assign to the same variable; we union their declarations into one
    # parameter (LLM picks one exit_path_id and supplies the relevant ones).
    seen_assign_vars: dict[str, str] = {}  # var -> description
    for ep in plan.llm_exit_paths:
        for var, assign in ep.assigns.items():
            if assign.method != "llm":
                continue
            if var not in seen_assign_vars:
                seen_assign_vars[var] = (
                    f"Captured value for `{var}` when taking exit_path_id={ep.id}. "
                    "Omit if not relevant to the chosen exit."
                )

    for var, desc in seen_assign_vars.items():
        properties[var] = {"type": "string", "description": desc}

    return FunctionSchema(
        name="take_exit_path",
        description=(
            f"Route out of the current flow ({plan.active_flow.id}) onto one of "
            "its declared exit paths. Call this on the same turn as your spoken "
            "reply, when the conversation has reached the exit condition."
        ),
        properties=properties,
        required=["exit_path_id"],
    )


def _exit_path_intent(ep: ExitPath) -> str:
    """One-line intent text used in the take_exit_path enum description so
    the LLM can disambiguate between exit_path_id choices.

    `condition.expression` wins when present — same idiom forward exits use,
    now also available to return paths so spec authors can express "return
    when X" without writing it as plumbing in flow.instructions.
    """
    if ep.condition is not None:
        return ep.condition.expression
    if ep.type == "return_to_caller":
        return (
            "Return to the previous flow when this side conversation is "
            "naturally complete and the patron is ready to move on."
        )
    return ep.id


def _interrupt_intent(f: Flow) -> str:
    """One-line intent text for an interrupt's trigger_interrupt entry, with
    the same precedence as exits: entry_condition.expression wins; fall back
    to flow.description, then flow.id."""
    if f.routing.entry_condition is not None:
        return f.routing.entry_condition.expression
    return f.description or f.id


def _trigger_interrupt_schema(
    plan: RoutingPlan,
    variables: dict[str, Any] | None,
) -> FunctionSchema:
    interrupt_ids = [f.id for f in plan.llm_interrupts]
    descriptions = "\n".join(
        f"- {f.id}: {substitute_variables(_interrupt_intent(f), variables)}"
        for f in plan.llm_interrupts
    )
    return FunctionSchema(
        name="trigger_interrupt",
        description=(
            "Trigger an interrupt flow because the patron asked something off "
            "the current routing path. Speak the answer naturally in the same "
            "response. Available interrupts:\n" + descriptions
        ),
        properties={
            "interrupt_flow_id": {
                "type": "string",
                "enum": interrupt_ids,
                "description": "Which interrupt to fire.",
            }
        },
        required=["interrupt_flow_id"],
    )


def _format_faq(entries, lang: str) -> str:
    lines = []
    for entry in entries:
        # Per-language script overrides the base answer if available.
        answer = entry.scripts.get(lang) if entry.scripts else None
        lines.append(f"- Q: {entry.question}\n  A: {answer or entry.answer}")
    return "\n".join(lines)
