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
      - else: increment_turn() and stay. (Per-frame turn counter remains
        for observability and a future reserved-variable primitive; see
        SCHEMA.md Open Questions for the "turn budgets" design.)
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
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
    TurnCompleted,
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
    """Per-turn: plan routing + load tools into context.

    Honors `session.skip_next_planning` for the silent-take_exit follow-up:
    when set, this pass leaves tools/prompt as-is (the take_exit handler
    already cleared tools and pushed the new flow's prompt) so the follow-up
    inference runs text-only and can't chain another routing decision.
    """

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            s = self._session
            if not s.ended and s.state.stack:
                if s.skip_next_planning:
                    # Consume the flag — only this single inference is exempt.
                    s.skip_next_planning = False
                else:
                    plan_for_active_flow(s)
                s.tool_handler_fired_this_turn = False
                s.text_emitted_this_turn = False
        await self.push_frame(frame, direction)


class TextFrameWatcher(FrameProcessor):
    """Sits between the LLM service and TTS. Two responsibilities:

    1. Flip `session.text_emitted_this_turn = True` whenever the LLM pushes a
       non-empty `LLMTextFrame` — used by the silent-take_exit follow-up to
       detect responses where the model emitted ONLY a tool call (silent UX).
    2. Accumulate the streamed text parts of this response and emit a
       `TurnCompleted` event when the response ends — so JSONL traces show
       what the agent actually said, not just which exits fired.

    Pipecat's LLM service streams text parts via `push_frame()` synchronously
    during response generation; tool handlers run after streaming ends.
    By the time the take_exit handler reads this flag, all text frames from
    THIS response have already propagated past the watcher.
    """

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session
        self._buf: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame) and frame.text:
                self._session.text_emitted_this_turn = True
                self._buf.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                if self._buf:
                    self._session.events.emit(
                        TurnCompleted(
                            session_id=self._session.session_id,
                            role="agent",
                            text="".join(self._buf),
                        )
                    )
                self._buf.clear()
        await self.push_frame(frame, direction)


class UserTranscriptWatcher(FrameProcessor):
    """Emits a `TurnCompleted` event for each finalized STT transcript so
    JSONL traces show what the runner actually heard from the user.

    Sits between `stt` and `context_aggregator.user()`. Interim partials are
    ignored; only `finalized=True` frames (or finalized==unset, treated as
    final by default for STT services that don't set the flag) are emitted.
    """

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text
        ):
            self._session.events.emit(
                TurnCompleted(
                    session_id=self._session.session_id,
                    role="user",
                    text=frame.text,
                )
            )
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
        has_caller=s.state.has_caller,
    )
    s.current_plan = plan
    tools = build_tools(plan, s.state.variables)
    # LLMContext requires ToolsSchema or NOT_GIVEN — never None.
    s.llm_context.set_tools(tools if tools is not None else NOT_GIVEN)
    active = s.spec.flows_by_id[s.state.active_flow_id]
    new_prompt = build_system_prompt(
        s.spec.agent, active, s.state.language, variables=s.state.variables
    )
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
        if decision is None:
            return
        if decision.kind in ("trigger_interrupt", "return") and not session.pending_end:
            await params.llm.push_frame(LLMRunFrame())
            return
        # Silent take_exit / end follow-up: Gemini sometimes emits the tool
        # call with no text part — state mutates correctly but the user gets
        # dead air. Push ONE follow-up inference with tools cleared so the
        # model produces words and can't chain another transition. The source
        # flow's prompt is still loaded (take_exit only swaps AFTER this
        # frame fires; end defers tear-down via s.pending_end), so the
        # closing line is spoken in the source flow's voice. For pending
        # ends, finalize_pending_end() runs on the resulting turn-end.
        # `return` decisions that collapsed to a pending_end (RETURN with no
        # caller frame, per schema) take this path too — same lifecycle as
        # an explicit END.
        if (
            (decision.kind in ("take_exit", "end") or session.pending_end)
            and not session.text_emitted_this_turn
            and not session.ended
        ):
            session.llm_context.set_tools(NOT_GIVEN)
            session.skip_next_planning = True
            await params.llm.push_frame(LLMRunFrame())

    async def _trigger_interrupt(params: FunctionCallParams) -> None:
        decision = await apply_tool_call(session, "trigger_interrupt", dict(params.arguments))
        await params.result_callback(
            {}, properties=FunctionCallResultProperties(run_llm=False)
        )
        if decision is not None and decision.kind in ("trigger_interrupt", "return"):
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
    inference is needed. Two follow-up shapes exist:

    - **New-flow follow-up** (`trigger_interrupt`, `return` into a caller):
      the new flow's prompt is loaded and the LLM should speak its opener.
      Gemini AUTO mode tends to emit ONLY the function call on these (silence
      to the user); a second LLMRunFrame against the new prompt is needed.

    - **Source-flow silent follow-up** (`take_exit`, `end`, or `return`
      collapsed to a pending end): the source flow's prompt is still loaded
      and the LLM should speak the flow's closing line before we transition
      or tear down. Gemini reliably co-emits text with `take_exit_path`, but
      not always — when it doesn't, we fire one tools-cleared follow-up so
      the model can only produce text (no chained transitions). For `end`
      decisions, tear-down is deferred via `s.pending_end` so the prompt
      stays loaded; `finalize_pending_end` completes the lifecycle at
      turn-end.

    Voice mode pushes an LLMRunFrame for the follow-up; text mode does a
    second generate_content call inline.

    Returns None if the tool call arrived without a current_plan (guard
    against PreLLMPlanner not having run).

    SMELL: callers currently classify which follow-up shape applies by
    checking `decision.kind` plus `session.pending_end` directly. That
    classification is duplicated in voice (`_take_exit_path`) and text
    (`_run_one_turn`). If a third call-site appears or the rules grow
    another case, extract a `follow_up_for(decision, session)` helper that
    returns a pipeline-action enum. Two call-sites isn't yet worth it.
    """
    s = session
    s.tool_handler_fired_this_turn = True

    if s.current_plan is None:
        logger.warning("tool handler fired without a current_plan; ignoring")
        return None

    llm_results = {tool_name: args}
    decision = routing_mod.resolve(s.current_plan, s.spec, llm_results)
    await _apply_decision(decision, s, llm_results=llm_results)

    if decision.kind in ("trigger_interrupt", "return"):
        # Re-plan for the new active flow before its first turn fires.
        plan_for_active_flow(s)

    return decision


def finalize_pending_end(s: Session) -> bool:
    """Complete a deferred session end. Called at turn-end after any silent
    follow-up has produced the closing utterance. Emits SessionEnded and
    tears down. Returns True if a tear-down occurred.

    The deferral pattern: when an exit-path's goto is END (or RETURN with no
    caller), `_do_take_exit` / `_do_return` set `s.pending_end = True` and
    return without tearing down. The source flow's prompt stays loaded so a
    silent-take_exit follow-up can speak the flow's closing line in the
    source flow's voice. Once that turn completes, this finalizer fires.
    """
    if not s.pending_end or s.ended:
        return False
    from uxflows_runner.events.schema import SessionEnded

    s.events.emit(SessionEnded(session_id=s.session_id, reason="agent_terminal"))
    s.state.end()
    s.ended = True
    s.pending_end = False
    return True


async def apply_planned_shortcut(s: Session) -> bool:
    """If the active flow's plan has a pre-resolved shortcut (a terminator
    flow with only unconditional exits), apply it and return True. Used by
    both voice (PostLLMResolver) and text (TextSession) after the LLM turn
    when no tool call fired. Without this, terminator flows loop forever —
    they emit their script every turn but never transition out.
    """
    if s.ended:
        return False
    if s.current_plan is None or s.current_plan.shortcut is None:
        return False
    await _apply_decision(s.current_plan.shortcut, s)
    return True


async def _apply_decision(
    decision: routing_mod.Decision,
    s: Session,
    llm_results: dict[str, Any] | None = None,
) -> None:
    """Branch on Decision.kind and execute. `llm_results` is None when called
    synthetically (no LLM turn produced the decision); in that path llm-method
    assigns are unresolvable and get skipped."""
    llm_results = llm_results or {}

    if decision.kind == "stay":
        return

    if decision.kind == "trigger_interrupt":
        await _do_trigger_interrupt(s, decision)
        return

    if decision.kind == "return":
        await _do_return(s, decision)
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
        # Decoupled lifecycle: emit FlowExited and mark the session as
        # pending_end, but do NOT tear down yet. The source flow's prompt
        # stays loaded so the silent-follow-up can produce a closing line.
        # finalize_pending_end() runs at turn-end and emits SessionEnded.
        s.events.emit(
            FlowExited(
                session_id=s.session_id,
                flow_id=source_flow.id,
                exit_path_id=exit_path.id,
                reason="terminal",
            )
        )
        s.pending_end = True
        return

    # take_exit — transition into a flow.
    s.events.emit(
        FlowExited(
            session_id=s.session_id,
            flow_id=source_flow.id,
            exit_path_id=exit_path.id,
            reason="transition",
        )
    )
    target_id = decision.target_flow_id
    assert target_id is not None
    target_flow = s.spec.flows_by_id[target_id]
    # Callable destinations push a frame; non-callable transitions replace
    # the active frame in place. Schema derives callability from structure:
    # a flow is callable iff it has at least one `goto: "RETURN"` exit.
    if target_flow.is_callable:
        new_frame = s.state.push_call(target_id)
    else:
        new_frame = s.state.transition(target_id)
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
    ec = interrupt.entry_condition
    method = ec.method if ec is not None else "direct"

    s.events.emit(
        InterruptTriggered(
            session_id=s.session_id,
            from_flow_id=source.id,
            interrupt_flow_id=interrupt.id,
            method=method,  # type: ignore[arg-type]
        )
    )
    new_frame = s.state.push_call(interrupt.id)
    s.events.emit(
        FlowEntered(
            session_id=s.session_id,
            flow_id=interrupt.id,
            via="interrupt",
            caller_flow_id=new_frame.caller_flow_id,
        )
    )
    await _push_new_flow_prompt(s, interrupt)


async def _do_return(s: Session, decision: routing_mod.Decision) -> None:
    exit_path = decision.exit_path
    source_flow = decision.source_flow
    assert exit_path is not None and source_flow is not None

    # RETURN with no caller frame collapses to END (per schema). Defer the
    # tear-down — same lifecycle as `_do_take_exit`'s end branch.
    if not s.state.has_caller:
        s.events.emit(
            FlowExited(
                session_id=s.session_id,
                flow_id=source_flow.id,
                exit_path_id=exit_path.id,
                reason="terminal",
            )
        )
        s.pending_end = True
        return

    popped, caller = s.state.pop_to_caller()
    s.events.emit(
        FlowExited(
            session_id=s.session_id,
            flow_id=popped.flow_id,
            exit_path_id=exit_path.id,
            reason="returned",
        )
    )
    caller_flow = s.spec.flows_by_id[caller.flow_id]
    s.events.emit(
        FlowEntered(
            session_id=s.session_id,
            flow_id=caller_flow.id,
            via="return",
            caller_flow_id=caller.caller_flow_id,
        )
    )
    await _push_new_flow_prompt(s, caller_flow)


async def _push_new_flow_prompt(s: Session, flow: Flow) -> None:
    """After a transition: replace the system message AND clear tools so the
    next PreLLMPlanner pass sets them fresh. We don't push LLMMessagesUpdateFrame
    here because we already own the LLMContext and can mutate it directly.
    The change takes effect on the next user turn."""
    new_prompt = build_system_prompt(
        s.spec.agent, flow, s.state.language, variables=s.state.variables
    )
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

        # If a previous turn deferred a session end so its silent follow-up
        # could speak, this is the turn after that follow-up — finalize now.
        if finalize_pending_end(s):
            return

        if s.tool_handler_fired_this_turn:
            return  # handler already advanced state

        # No tool fired — the LLM produced text only. If this is a terminator
        # flow, apply its planned shortcut so it ends after speaking its line.
        # (The shortcut itself may set pending_end; finalize runs next turn.)
        if await apply_planned_shortcut(s):
            return
        s.state.increment_turn()


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
