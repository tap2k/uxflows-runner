"""Compose the per-flow system prompt — agent + flow sections plus the
in-text routing protocol that lists this turn's available exits and
interrupts.

Layout:
  agent.system_prompt
  + agent.guardrails (Operating principles)
  + agent.knowledge.faq (Frequently asked)
  + agent.knowledge.glossary (Terminology)
  + flow.instructions (This flow)
  + flow.guardrails
  + flow.knowledge.faq
  + flow.scripts (sample lines for the active language)
  + Routing protocol (exits + interrupts available this turn, format,
    one-shot example) — only when a RoutingPlan is supplied.

Translatable fields (`system_prompt`, `guardrail.statement`, `faq.answer`,
`glossary.definition`, `flow.instructions`, `script.text`) are all
LocalizedString — resolved through `resolve_localized` against the
session's active language with fallback to the agent's default language.

Routing protocol (instead of tool calls): the LLM emits a self-closing
`<route ... />` tag at the end of its reply. The streaming stripper
(voice) / response splitter (text) parses it post-emission and feeds it
to the dispatcher via `apply_route`. See `routing_protocol.py` for the
wire format.
"""

from __future__ import annotations

import re
from typing import Any

from uxflows_runner.spec.loader import LoadedSpec
from uxflows_runner.spec.types import (
    ExitPath,
    Flow,
    LocalizedString,
    default_language,
    is_end_goto,
    is_flow_goto,
    is_return_goto,
    resolve_localized,
)

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


def _resolve(value: LocalizedString | None, lang: str | None, default_lang: str) -> str:
    return resolve_localized(value, lang, default_lang)


def build_system_prompt(
    spec: LoadedSpec,
    flow: Flow,
    lang: str | None,
    variables: dict[str, Any] | None = None,
    plan: "RoutingPlan | None" = None,
) -> str:
    """`lang=None` means "use the agent's default language" — translatable
    fields resolve to the default-language string.

    `plan` is the routing plan for the current turn — used to render the
    in-text routing protocol section so the LLM sees the exits + interrupts
    available *this turn* (after calc shortcuts evaluate, RETURN visibility,
    nested-interrupt filtering). Omit at session start; the protocol section
    is skipped and the LLM just talks.
    """
    agent = spec.agent
    default_lang = default_language(agent.meta.languages)
    sections: list[str] = []

    # In-text routing protocol goes FIRST — above the agent's system_prompt.
    # The model treats top-of-prompt as load-bearing structure; if the
    # protocol lands at the bottom of a 20k-character prompt it gets read
    # as a footnote and the model fluently improvises off-path. We saw this
    # against the Tala spec: long agent.system_prompt + the protocol at the
    # tail produced zero route tags across a 15-turn session.
    if plan is not None:
        routing_section = _render_routing_protocol(spec, plan, variables)
        if routing_section:
            sections.append(routing_section)

    sp = _resolve(agent.system_prompt, lang, default_lang)
    if sp:
        sections.append(sp.strip())

    if agent.guardrails:
        sections.append(
            "Operating principles:\n"
            + "\n".join(
                f"- {_resolve(g.statement, lang, default_lang)}"
                for g in agent.guardrails
            )
        )

    if agent.knowledge.faq:
        sections.append(
            "Frequently asked:\n" + _format_faq(agent.knowledge.faq, lang, default_lang)
        )

    if agent.knowledge.glossary:
        sections.append(
            "Terminology:\n"
            + "\n".join(
                f"- {g.term}: {_resolve(g.definition, lang, default_lang)}"
                for g in agent.knowledge.glossary
            )
        )

    sections.append(f"This flow ({flow.id}):")
    instr = _resolve(flow.instructions, lang, default_lang)
    if instr:
        sections.append(instr.strip())

    if flow.guardrails:
        sections.append(
            "For this flow specifically:\n"
            + "\n".join(
                f"- {_resolve(g.statement, lang, default_lang)}"
                for g in flow.guardrails
            )
        )

    if flow.knowledge and flow.knowledge.faq:
        sections.append(
            "Flow-specific FAQ:\n" + _format_faq(flow.knowledge.faq, lang, default_lang)
        )

    # Per-flow scripts. Each line has a LocalizedString text and optional
    # per-language variations. Resolve against the active language.
    bucket_lines: list[str] = []
    for s in flow.scripts:
        text = _resolve(s.text, lang, default_lang)
        if not text:
            continue
        bucket_lines.append(f"- [{s.id}] {text}")
        variations_for_lang: list[str] = []
        if s.variations:
            effective = lang or default_lang
            variations_for_lang = (
                s.variations.get(effective)
                or s.variations.get(default_lang)
                or []
            )
        for v in variations_for_lang:
            if v:
                bucket_lines.append(f"  | {v}")
    if bucket_lines:
        sections.append(
            "Scripted lines — when instructions reference a script by [id], "
            "say that line verbatim (pick any listed variation if present):\n"
            + "\n".join(bucket_lines)
        )

    # Trailing reminder so the protocol stays salient even after a long
    # agent.system_prompt + guardrails + flow.instructions wall. Models
    # weight the tail of the prompt heavily.
    if plan is not None:
        reminder = _render_routing_reminder(plan)
        if reminder:
            sections.append(reminder)

    return substitute_variables("\n\n".join(sections), variables)


def _render_routing_protocol(
    spec: LoadedSpec,
    plan: "RoutingPlan",
    variables: dict[str, Any] | None,
) -> str:
    """Render the routing-protocol section: exits + interrupts available
    this turn, format, and a one-shot example. Empty string when there's
    nothing to route on (the LLM should just talk)."""
    exit_lines: list[str] = []
    for ep in plan.llm_exit_paths:
        intent = substitute_variables(_exit_path_intent(ep), variables)
        dest = _exit_destination(ep, spec.flows_by_id)
        exit_lines.append(f"- {ep.id} (→ {dest}): {intent}")

    interrupt_lines: list[str] = []
    for f in plan.llm_interrupts:
        trigger = substitute_variables(_interrupt_intent(f), variables)
        interrupt_lines.append(f"- {f.id}: {trigger}")

    if not exit_lines and not interrupt_lines:
        return ""

    parts = [
        "# ROUTING PROTOCOL (load-bearing — read this first)\n\n"
        "Every response you produce is BOTH a conversational reply AND a "
        "routing decision. You are running inside a flow graph; the "
        "conversation cannot advance to the next flow without an explicit "
        "route tag from you.\n\n"
        "RULES:\n"
        "1. When the user's latest message satisfies any exit condition "
        "below, you MUST end your reply with the matching `<route .../>` "
        "tag. No tag = stuck in this flow forever = broken UX.\n"
        "2. This applies even when you've just spoken what feels like a "
        "closing line (\"have a great day\", \"contact support if you have "
        "issues\", etc.). The closing line is YOUR text; the tag is what "
        "tells the runtime the flow is over. Without the tag, the runtime "
        "asks you to keep talking. Always emit the tag.\n"
        "3. When NO exit condition matches, omit the tag and keep talking. "
        "The tag must be the LAST thing in your response — never put text "
        "after it."
    ]
    if exit_lines:
        parts.append(
            f"Exits from `{plan.active_flow.id}` (active flow):\n"
            + "\n".join(exit_lines)
        )
    if interrupt_lines:
        parts.append(
            "Interrupts (fire when the trigger matches the user's latest "
            "message, regardless of which flow is active):\n"
            + "\n".join(interrupt_lines)
        )
    parts.append(
        "Tag format (self-closing, double-quoted attribute values):\n"
        '  <route exit="EXIT_ID" />                 — take a flow exit\n'
        '  <route exit="EXIT_ID" varname="value" /> — take with a capture\n'
        '  <route interrupt="INTERRUPT_ID" />       — fire an interrupt\n\n'
        "Examples:\n"
        '  Got it, I will send that now. <route exit="xp_send_confirmation" />\n'
        "  Thanks for letting me know. Please contact support if you "
        'believe this is an error. Have a great day. <route exit="xp_wrong_caller" />\n'
        '  Sorry — let me check that for you. <route interrupt="int_lookup" />'
    )
    return "\n\n".join(parts)


def _render_routing_reminder(plan: "RoutingPlan") -> str:
    """Short reminder appended at the end of the system prompt so the model
    sees the routing requirement most recently — the long body of agent +
    flow instructions can otherwise drown out the protocol section's
    authority by the time the model is ready to respond."""
    exit_ids = [ep.id for ep in plan.llm_exit_paths]
    interrupt_ids = [f.id for f in plan.llm_interrupts]
    if not exit_ids and not interrupt_ids:
        return ""
    available = exit_ids + interrupt_ids
    return (
        "REMINDER — your reply must end with a route tag if any of these "
        f"conditions has been reached: {', '.join(available)}. See the "
        "ROUTING PROTOCOL section at the top of this prompt for tag "
        "format. If no condition matches, omit the tag and keep talking."
    )


def _exit_destination(ep: ExitPath, flows_by_id: dict[str, Flow] | None) -> str:
    """Render the exit's destination as a short phrase for the tool enum.

    Forward-flow gotos resolve to the target flow's `name` (falling back to
    `id` if name is unset). END / RETURN render as fixed strings.
    """
    if is_end_goto(ep.goto):
        return "end the conversation"
    if is_return_goto(ep.goto):
        return "return to the calling flow"
    assert is_flow_goto(ep.goto)
    if flows_by_id is not None:
        target = flows_by_id.get(ep.goto)
        if target is not None:
            return target.name or target.id
    return ep.goto


def _exit_path_intent(ep: ExitPath) -> str:
    """One-line intent text used in the take_exit_path enum description so
    the LLM can disambiguate between exit_path_id choices.

    `condition.expression` wins when present — same idiom forward exits use,
    now also available to return paths so spec authors can express "return
    when X" without writing it as plumbing in flow.instructions.
    """
    if ep.condition is not None:
        return ep.condition.expression
    if is_return_goto(ep.goto):
        return (
            "Return to the previous flow when this side conversation is "
            "naturally complete and the patron is ready to move on."
        )
    return ep.id


def _interrupt_intent(f: Flow) -> str:
    """One-line intent text for an interrupt's trigger_interrupt entry, with
    the same precedence as exits: entry_condition.expression wins; fall back
    to flow.name, then flow.id."""
    if f.entry_condition is not None:
        return f.entry_condition.expression
    return f.name or f.id


def _format_faq(entries, lang: str | None, default_lang: str) -> str:
    """FAQ entries are { id, question, answer: LocalizedString }. Resolve
    answer against the active language."""
    lines = []
    for entry in entries:
        answer = _resolve(entry.answer, lang, default_lang)
        lines.append(f"- Q: {entry.question}\n  A: {answer}")
    return "\n".join(lines)
