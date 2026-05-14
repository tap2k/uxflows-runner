"""Routing — what happens after the user's turn.

Two phases per turn:
  1. plan() — before the LLM call. Walk the active flow's exit_paths in order;
     short-circuit calculation/direct evaluations against the current variable
     bag. Collect remaining `llm`-method exit paths and applicable interrupts
     into a `RoutingPlan` that the prompt builder turns into the per-turn tool
     schema.
  2. resolve() — after the LLM call returns. Given the LLM's tool-call payload
     (`llm_results`), pick the final Decision. Calculation/direct shortcuts
     from the plan win over LLM picks (they evaluated locally with full
     certainty); among LLM picks, exactly one — `take_exit_path` OR
     `trigger_interrupt` — fires.

`goto: "RETURN"` exits are exposed as LLM-method `take_exit_path`
candidates — the LLM picks when to return based on conversational signals
("patron is satisfied", "patron clearly moved on"). Auto-firing every turn
would pop the interrupt before the patron could ask follow-ups inside it
(e.g. "what's on the menu?" → "tell me about the latte" → return).

Interrupts (`flow.type == "interrupt"`) are implicitly globally callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from uxflows_runner.spec.loader import LoadedSpec, applicable_interrupts
from uxflows_runner.spec.types import (
    ExitPath,
    Flow,
    is_end_goto,
    is_flow_goto,
    is_return_goto,
)

from . import methods


@dataclass
class Decision:
    """What the dispatcher tells the processor to do after this turn.

    Exactly one of the action fields is meaningful per Decision instance.
    """

    kind: str  # "stay" | "take_exit" | "trigger_interrupt" | "return" | "end"
    exit_path: ExitPath | None = None
    target_flow_id: str | None = None  # flow id for take_exit; interrupt id for trigger
    source_flow: Flow | None = None  # the flow that owned the exit_path (for assigns/actions)
    interrupt_flow: Flow | None = None  # populated on trigger_interrupt


@dataclass
class RoutingPlan:
    """Plan for the upcoming LLM call. Populated by plan().

    `shortcut` is set when a calc/direct exit matches before the LLM call —
    the LLM still runs (we want a natural-language response) but its routing
    output is ignored. `llm_exit_paths` and `llm_interrupts` are passed
    through to the prompt_builder to materialize tool-call variants.
    """

    active_flow: Flow
    shortcut: Decision | None = None
    llm_exit_paths: list[ExitPath] = field(default_factory=list)
    llm_interrupts: list[Flow] = field(default_factory=list)


def plan(
    spec: LoadedSpec,
    active_flow_id: str,
    variables: dict[str, Any],
    *,
    has_caller: bool,
) -> RoutingPlan:
    """Pre-LLM phase: short-circuit on calc/direct, collect LLM candidates."""
    active = spec.flows_by_id[active_flow_id]
    plan = RoutingPlan(active_flow=active)

    for ep in active.exit_paths:
        # A RETURN exit is only meaningful when we have a frame to pop.
        # Surface as an LLM-driven `take_exit_path` candidate — the LLM picks
        # when to return based on conversational signals.
        if is_return_goto(ep.goto):
            if has_caller:
                plan.llm_exit_paths.append(ep)
            continue

        condition = ep.condition
        if condition is None:
            # Unconditional exit. Treated as a fallback — not auto-fired by
            # the planner; reserved for future explicit-fallback semantics.
            continue

        if condition.method == "llm":
            plan.llm_exit_paths.append(ep)
            continue

        # calculation or direct — evaluate now.
        if methods.evaluate_condition(condition, variables, {}, llm_key=ep.id):
            if plan.shortcut is None:
                plan.shortcut = _build_take_exit(active, ep)
            # A calc shortcut wins, period. Stop walking.
            break

    # Collect applicable interrupts (only when not already inside one — nested
    # interrupts are legal but we don't auto-offer them in the same turn).
    if not has_caller:
        for interrupt in applicable_interrupts(spec):
            ec = interrupt.entry_condition
            if ec is None:
                continue  # interrupt with no entry_condition can't fire
            if ec.method == "llm":
                plan.llm_interrupts.append(interrupt)
            else:
                if methods.evaluate_condition(ec, variables, {}, llm_key=interrupt.id):
                    if plan.shortcut is None:
                        plan.shortcut = Decision(
                            kind="trigger_interrupt",
                            target_flow_id=interrupt.id,
                            source_flow=active,
                            interrupt_flow=interrupt,
                        )

    return plan


def resolve(
    plan: RoutingPlan,
    spec: LoadedSpec,
    llm_results: dict[str, Any],
) -> Decision:
    """Post-LLM phase: combine the plan's shortcut (if any) with the LLM's
    tool-call output to pick a final Decision.

    `llm_results` shape:
      {
        "take_exit_path": {"exit_path_id": "...", "<assign_var>": ...},
        "trigger_interrupt": {"interrupt_flow_id": "..."},
      }
    Either / both / neither key may be present.

    Resolution order:
      1. trigger_interrupt — an interrupt is a topical detour; the routing
         decision (shortcut or LLM-picked) still holds and applies once the
         interrupt returns. Honoring the interrupt first means a user can ask
         a side question on the same turn that would otherwise route them.
      2. shortcut (calc/direct exit) — deterministic, beats LLM exit picks.
      3. LLM take_exit_path — chosen exit from the candidates.
      4. stay.
    """
    trigger = llm_results.get("trigger_interrupt") or {}
    interrupt_id = trigger.get("interrupt_flow_id")
    if interrupt_id:
        for interrupt in plan.llm_interrupts:
            if interrupt.id == interrupt_id:
                return Decision(
                    kind="trigger_interrupt",
                    target_flow_id=interrupt.id,
                    source_flow=plan.active_flow,
                    interrupt_flow=interrupt,
                )

    if plan.shortcut is not None:
        return plan.shortcut

    take = llm_results.get("take_exit_path") or {}
    chosen_exit_id = take.get("exit_path_id")
    if chosen_exit_id:
        for ep in plan.llm_exit_paths:
            if ep.id == chosen_exit_id:
                return _build_take_exit(plan.active_flow, ep)
        # LLM hallucinated an unknown exit_path_id — fall through to stay.

    return Decision(kind="stay", source_flow=plan.active_flow)


def _build_take_exit(active_flow: Flow, ep: ExitPath) -> Decision:
    if is_return_goto(ep.goto):
        return Decision(
            kind="return",
            exit_path=ep,
            source_flow=active_flow,
        )
    if is_end_goto(ep.goto):
        return Decision(kind="end", exit_path=ep, source_flow=active_flow)
    # Otherwise: flow-id goto.
    assert is_flow_goto(ep.goto)
    return Decision(
        kind="take_exit",
        exit_path=ep,
        target_flow_id=ep.goto,
        source_flow=active_flow,
    )
