"""Pipecat seam — the only module in the dispatcher that imports Pipecat.

Two FrameProcessors and two LLM tool handlers, all closing over a shared
`Session`:

  PreLLMPlanner — sits between context_aggregator.user() and the LLM.
    Per turn:
      1. routing.plan(...) against active flow + variables
      2. mutate session.llm_context.tools to the per-turn tool schema
      3. mutate session.llm_context messages so the system prompt is the
         active flow's composed prompt (in case the previous turn ended a
         transition that hasn't pushed the new prompt yet)
      4. clear session.tool_handler_fired_this_turn

  Tool handlers (take_exit_path, trigger_interrupt) — registered on the LLM
  service via register_function. THESE ARE THE RESOLVER. Each handler:
      1. routing.resolve(plan, spec, llm_results) -> Decision
      2. assigns.fire(...) on take_exit (mutates variable bag, emits events)
      3. dispatch capability actions (fire-and-forget)
      4. mutate FlowState (transition / push / pop / end)
      5. push LLMMessagesUpdateFrame for the new flow's system prompt
      6. mutate session.llm_context.tools for the next turn
      7. emit events (flow_exited, exit_path_taken, flow_entered, etc.)
      8. set session.tool_handler_fired_this_turn = True
      9. result_callback(run_llm=False)  ← suppress automatic re-inference

  PostLLMResolver — sits after the LLM (and TTS, before transport.output).
    On LLMFullResponseEndFrame:
      - if tool_handler_fired_this_turn: no-op
      - else: increment_turn(); check max_turns; on exhaustion, run the
        unconditional sad fallback synthetically (re-uses the take_exit
        code path).
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallResultProperties,
    LLMService,
)

from uxflows_runner.events.schema import (
    CapabilityInvoked,
    CapabilityReturned,
    ExitPathTaken,
    FlowEntered,
    FlowExited,
    InterruptTriggered,
    SessionEnded,
    VariableSet,
)
from uxflows_runner.spec.types import Flow

from . import assigns as assigns_mod
from . import routing as routing_mod
from .capabilities import CapabilityResult
from .prompt_builder import build_system_prompt, build_tools
from .session import Session


# --------------------------------------------------------------------------
# PreLLMPlanner
# --------------------------------------------------------------------------


class PreLLMPlanner(FrameProcessor):
    """Per-turn: plan routing + load tools into context."""

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            s = self._session
            if not s.ended and s.state.stack:
                plan_for_active_flow(s)
                s.tool_handler_fired_this_turn = False
        await self.push_frame(frame, direction)


def plan_for_active_flow(s: Session) -> None:
    """Build the routing plan for the active flow and sync it into the
    LLMContext: per-turn tool schema + active-flow system prompt. Stashes
    the plan on the session so tool handlers can read it during resolve.

    Idempotent — safe to call multiple times per turn.
    """
    plan = routing_mod.plan(
        s.spec,
        s.state.active_flow_id,
        s.state.variables,
        in_interrupt=s.state.is_in_interrupt,
    )
    s.current_plan = plan
    tools = build_tools(plan)
    # LLMContext requires ToolsSchema or NOT_GIVEN — never None.
    s.llm_context.set_tools(tools if tools is not None else NOT_GIVEN)
    active = s.spec.flows_by_id[s.state.active_flow_id]
    new_prompt = build_system_prompt(s.spec.agent, active, s.state.language)
    _replace_system_message(s.llm_context, new_prompt)


def _replace_system_message(context: Any, new_prompt: str) -> None:
    """Find the first system message in the context and replace its content.
    LLMContext.messages is a list of dicts; first system entry wins."""
    for msg in context.messages:
        if msg.get("role") == "system":
            msg["content"] = new_prompt
            return
    # No system message yet — prepend one.
    context.messages.insert(0, {"role": "system", "content": new_prompt})


# --------------------------------------------------------------------------
# Tool handler factory
# --------------------------------------------------------------------------


def register_dispatcher_tools(llm: LLMService, session: Session) -> None:
    """Register `take_exit_path` and `trigger_interrupt` on the LLM service.

    Pipecat-mode glue: thin wrappers around `apply_tool_call` (the framework-
    agnostic core) plus the Pipecat-specific result_callback + LLMRunFrame
    follow-up. Text mode (server/text_session.py) calls `apply_tool_call`
    directly with its own follow-up mechanism.
    """

    async def _take_exit_path(params: FunctionCallParams) -> None:
        decision = await apply_tool_call(session, "take_exit_path", dict(params.arguments))
        await params.result_callback(
            {}, properties=FunctionCallResultProperties(run_llm=False)
        )
        if decision is not None and decision.kind in ("trigger_interrupt", "return_to_caller"):
            await params.llm.push_frame(LLMRunFrame())

    async def _trigger_interrupt(params: FunctionCallParams) -> None:
        decision = await apply_tool_call(session, "trigger_interrupt", dict(params.arguments))
        await params.result_callback(
            {}, properties=FunctionCallResultProperties(run_llm=False)
        )
        if decision is not None and decision.kind in ("trigger_interrupt", "return_to_caller"):
            await params.llm.push_frame(LLMRunFrame())

    llm.register_function("take_exit_path", _take_exit_path)
    llm.register_function("trigger_interrupt", _trigger_interrupt)


async def apply_tool_call(
    session: Session, tool_name: str, args: dict[str, Any]
) -> routing_mod.Decision | None:
    """Framework-agnostic core of dispatch: resolve LLM tool args into a
    Decision, mutate session state (assigns, capabilities, flow stack, events),
    and re-plan the active flow if a transition happened.

    Returns the Decision so the caller can decide whether a follow-up LLM
    inference is needed. `trigger_interrupt` and `return_to_caller` need one
    (Gemini AUTO mode reliably emits text + tool together on `take_exit_path`,
    but tends to emit ONLY the function call on interrupts — silence to the
    patron. We tried prompt + tool-description mandates; they don't reliably
    override the model's instinct that an interrupt-style off-path question
    is *itself* the answer. The 2-call-per-interrupt cost is acceptable given
    interrupts are rare. take_exit transitions don't need a follow-up: the
    LLM answered this turn, and the new flow speaks on the NEXT user turn
    per RUNNER-PLAN's "routing is next-turn" rule.)

    Voice mode pushes an LLMRunFrame for the follow-up; text mode does a
    second generate_content call inline.

    Returns None if the tool call arrived without a current_plan (guard
    against PreLLMPlanner not having run).
    """
    s = session
    s.tool_handler_fired_this_turn = True

    if s.current_plan is None:
        logger.warning("tool handler fired without a current_plan; ignoring")
        return None

    llm_results = {tool_name: args}
    decision = routing_mod.resolve(s.current_plan, s.spec, llm_results)
    await _apply_decision(decision, s, llm_results=llm_results)

    if decision.kind in ("trigger_interrupt", "return_to_caller"):
        # Re-plan for the new active flow before its first turn fires.
        plan_for_active_flow(s)

    return decision


async def _apply_decision(
    decision: routing_mod.Decision,
    s: Session,
    llm_results: dict[str, Any] | None = None,
) -> None:
    """Branch on Decision.kind and execute. `llm_results` is None when called
    synthetically from the PostLLMResolver backstop (max_turns fallback) —
    in that path llm-method assigns are unresolvable and get skipped."""
    llm_results = llm_results or {}

    if decision.kind == "stay":
        return

    if decision.kind == "trigger_interrupt":
        await _do_trigger_interrupt(s, decision)
        return

    if decision.kind == "return_to_caller":
        await _do_return_to_caller(s, decision)
        return

    if decision.kind in ("take_exit", "end"):
        await _do_take_exit(s, decision, llm_results)
        return


async def _do_take_exit(
    s: Session, decision: routing_mod.Decision, llm_results: dict[str, Any]
) -> None:
    exit_path = decision.exit_path
    source_flow = decision.source_flow
    assert exit_path is not None and source_flow is not None

    # Method tag for events: pull from the condition (None on unconditional
    # exits — call those "direct" since they fire unconditionally).
    method = exit_path.condition.method if exit_path.condition else "direct"

    # 1) Fire assigns first so capability inputs see the new values.
    fired = assigns_mod.fire(exit_path, s.state.variables, llm_results)
    for r in fired:
        if r.skipped:
            continue
        s.events.emit(
            VariableSet(
                session_id=s.session_id,
                variable_name=r.variable,
                value=r.value,
                method=r.method,  # type: ignore[arg-type]
                source_flow_id=source_flow.id,
                source_exit_path_id=exit_path.id,
            )
        )

    # 2) Capabilities — fire-and-forget; results arrive later via callback.
    for action in exit_path.actions:
        invocation = s.capabilities.invoke(action.capability_id, s.state.variables)
        s.events.emit(
            CapabilityInvoked(
                session_id=s.session_id,
                capability_name=invocation.capability_name,
                args=invocation.args,
            )
        )

    # 3) State + events
    s.events.emit(
        ExitPathTaken(
            session_id=s.session_id,
            from_flow_id=source_flow.id,
            exit_path_id=exit_path.id,
            to_flow_id=decision.target_flow_id,
            method=method,  # type: ignore[arg-type]
        )
    )

    if decision.kind == "end":
        s.events.emit(
            FlowExited(
                session_id=s.session_id,
                flow_id=source_flow.id,
                exit_path_id=exit_path.id,
                reason="terminal",
            )
        )
        s.events.emit(
            SessionEnded(session_id=s.session_id, reason="agent_terminal")
        )
        s.state.end()
        s.ended = True
        return

    # take_exit (transition or return_to_caller routed via direct take_exit)
    s.events.emit(
        FlowExited(
            session_id=s.session_id,
            flow_id=source_flow.id,
            exit_path_id=exit_path.id,
            reason="transition",
        )
    )
    new_frame = s.state.transition(decision.target_flow_id)  # type: ignore[arg-type]
    new_flow = s.spec.flows_by_id[new_frame.flow_id]
    s.events.emit(
        FlowEntered(
            session_id=s.session_id,
            flow_id=new_flow.id,
            via="transition",
            caller_flow_id=new_frame.caller_flow_id,
        )
    )
    await _push_new_flow_prompt(s, new_flow)


async def _do_trigger_interrupt(s: Session, decision: routing_mod.Decision) -> None:
    interrupt = decision.interrupt_flow
    source = decision.source_flow
    assert interrupt is not None and source is not None

    # entry_condition method for events
    ec = interrupt.routing.entry_condition
    method = ec.method if ec is not None else "direct"

    s.events.emit(
        InterruptTriggered(
            session_id=s.session_id,
            from_flow_id=source.id,
            interrupt_flow_id=interrupt.id,
            method=method,  # type: ignore[arg-type]
        )
    )
    new_frame = s.state.push_interrupt(interrupt.id)
    s.events.emit(
        FlowEntered(
            session_id=s.session_id,
            flow_id=interrupt.id,
            via="interrupt",
            caller_flow_id=new_frame.caller_flow_id,
        )
    )
    await _push_new_flow_prompt(s, interrupt)


async def _do_return_to_caller(s: Session, decision: routing_mod.Decision) -> None:
    exit_path = decision.exit_path
    assert exit_path is not None
    popped, caller = s.state.pop_to_caller()
    s.events.emit(
        FlowExited(
            session_id=s.session_id,
            flow_id=popped.flow_id,
            exit_path_id=exit_path.id,
            reason="returned_to_caller",
        )
    )
    caller_flow = s.spec.flows_by_id[caller.flow_id]
    s.events.emit(
        FlowEntered(
            session_id=s.session_id,
            flow_id=caller_flow.id,
            via="return_to_caller",
            caller_flow_id=caller.caller_flow_id,
        )
    )
    await _push_new_flow_prompt(s, caller_flow)


async def _push_new_flow_prompt(s: Session, flow: Flow) -> None:
    """After a transition: replace the system message AND clear tools so the
    next PreLLMPlanner pass sets them fresh. We don't push LLMMessagesUpdateFrame
    here because we already own the LLMContext and can mutate it directly.
    The change takes effect on the next user turn."""
    new_prompt = build_system_prompt(s.spec.agent, flow, s.state.language)
    _replace_system_message(s.llm_context, new_prompt)
    # Tools will be recomputed on the next PreLLMPlanner pass for this new flow.


# --------------------------------------------------------------------------
# PostLLMResolver — backstop
# --------------------------------------------------------------------------


class PostLLMResolver(FrameProcessor):
    """Listens for LLMFullResponseEndFrame and handles the case where no
    tool handler ran this turn (the LLM just produced a plain reply)."""

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._on_turn_end()
        await self.push_frame(frame, direction)

    async def _on_turn_end(self) -> None:
        s = self._session
        if s.ended:
            return
        if s.tool_handler_fired_this_turn:
            return  # handler already advanced state

        # No tool fired — the LLM produced text only. We "stay" by default,
        # but tick the turn counter and check max_turns.
        s.state.increment_turn()
        active = s.spec.flows_by_id[s.state.active_flow_id]
        if active.max_turns is not None and s.state.active.turn_count >= active.max_turns:
            try:
                fallback = routing_mod.force_max_turns_fallback(active)
            except RuntimeError:
                logger.warning(
                    "{} hit max_turns but no unconditional sad exit available; staying",
                    active.id,
                )
                return
            await _apply_decision(fallback, s, llm_results=None)


def add_capability_result_listener(session: Session) -> None:
    """Wire the CapabilityDispatcher's on_result callback to emit events.
    Call once per session after constructing capabilities."""

    def _on_result(result: CapabilityResult) -> None:
        session.events.emit(
            CapabilityReturned(
                session_id=session.session_id,
                capability_name=result.capability_name,
                result=result.result,
                error=result.error,
            )
        )

    session.capabilities._on_result = _on_result  # type: ignore[attr-defined]
