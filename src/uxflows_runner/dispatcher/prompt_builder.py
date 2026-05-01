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

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from uxflows_runner.spec.types import Agent, Flow

from .routing import RoutingPlan


def build_system_prompt(agent: Agent, flow: Flow, lang: str) -> str:
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
        sections.append(
            f"Sample lines you might use ({lang}) — paraphrase, don't recite verbatim:\n"
            + "\n".join(f"- {s.text}" for s in scripts)
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

    return "\n\n".join(sections)


def build_tools(plan: RoutingPlan) -> ToolsSchema | None:
    """Return the per-turn tool schema, or None when there's no LLM routing
    work to do this turn (the caller can skip `tools` entirely on the LLM
    call to save tokens). When None, the LLM still produces text — it just
    has no routing decisions to emit."""
    declarations: list[FunctionSchema] = []

    if plan.llm_exit_paths:
        declarations.append(_take_exit_path_schema(plan))

    if plan.llm_interrupts:
        declarations.append(_trigger_interrupt_schema(plan))

    if not declarations:
        return None
    return ToolsSchema(standard_tools=declarations)


def _take_exit_path_schema(plan: RoutingPlan) -> FunctionSchema:
    exit_ids = [ep.id for ep in plan.llm_exit_paths]

    properties: dict[str, dict] = {
        "exit_path_id": {
            "type": "string",
            "enum": exit_ids,
            "description": (
                "Pick the exit path that matches the patron's current state. "
                "Choose only when the conversation has clearly reached one of "
                "these conditions; otherwise keep talking and don't call this tool."
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


def _trigger_interrupt_schema(plan: RoutingPlan) -> FunctionSchema:
    interrupt_ids = [f.id for f in plan.llm_interrupts]
    descriptions = "\n".join(
        f"- {f.id}: {(f.routing.entry_condition.expression if f.routing.entry_condition else f.description) or f.id}"
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
