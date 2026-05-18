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

import re
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
from . import routing_protocol
from .prompt_builder import build_system_prompt
from .routing_protocol import RouteTag
from .session import Session


# --------------------------------------------------------------------------
# PreLLMPlanner
# --------------------------------------------------------------------------


class PreLLMPlanner(FrameProcessor):
    """Per-turn: re-plan routing for the active flow so the system prompt
    rendered for this inference includes the right `<route>` candidates.
    The plan is also stashed on the session for `apply_route` to consume
    when the LLM emits a tag.
    """

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


class RouteTagFrameProcessor(FrameProcessor):
    """Sits between the LLM service and TTS. Four responsibilities:

    1. Strip `<route .../>` tag bytes out of the text stream before they
       reach TTS — the user must never hear "less than route exit equals...".
    2. Strip `<think>...</think>` blocks (reserved reasoning sentinel; see
       routing_protocol module docstring). These are reasoning scaffolding
       the LLM is encouraged to emit but the user shouldn't hear.
    3. Accumulate the streamed conversational text and emit a TurnCompleted
       event for the agent's turn (with both sentinel forms stripped).
    4. On end-of-response, dispatch the parsed route tag (if any) via
       apply_route so the dispatcher mutates flow state and re-plans.

    Other tag-shaped text (`<strong>`, `<VERIFICATION>`, etc.) is NOT
    stripped — those are domain concepts an author might legitimately use.
    Add a new reserved sentinel here only when a concrete need arises.

    The state machine handles streaming chunks of any size — bytes arrive
    in arbitrary boundaries, and a `<` may straddle a chunk boundary.
    """

    # State
    _FORWARDING = "forwarding"
    _BUFFERING = "buffering"  # saw a `<`, haven't yet determined what it is
    _IN_THINK = "in_think"  # inside a `<think>...</think>` block; discard

    # Once the buffer exceeds this length without resolving, give up and
    # flush as plain text. Generous — accommodates `<route ...attrs.../>`
    # of reasonable length without blocking forever on a malformed prefix.
    _MAX_BUFFER = 256

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session
        # Conversational text that will reach TTS and be recorded in
        # TurnCompleted (sentinel bytes are NOT included).
        self._spoken: list[str] = []
        self._state = self._FORWARDING
        self._buf = ""
        # Parsed tag captured during this response (apply_route fires once,
        # on LLMFullResponseEndFrame).
        self._captured_tag: RouteTag | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame) and frame.text:
                emit = self._consume_text(frame.text)
                if emit:
                    await self.push_frame(LLMTextFrame(text=emit), direction)
                return  # don't push the original frame
            elif isinstance(frame, LLMFullResponseEndFrame):
                # Flush any pending buffer that never resolved — it's plain
                # text we held back; speak it now. Exception: if we were
                # inside a `<think>` block that never closed, discard it.
                if self._state == self._BUFFERING and self._buf:
                    await self.push_frame(LLMTextFrame(text=self._buf), direction)
                    self._spoken.append(self._buf)
                self._buf = ""
                self._state = self._FORWARDING
                # Emit the TurnCompleted event for the agent's reply.
                if self._spoken:
                    self._session.events.emit(
                        TurnCompleted(
                            session_id=self._session.session_id,
                            role="agent",
                            text="".join(self._spoken).strip(),
                        )
                    )
                # Dispatch the parsed tag (if any). apply_route fires the
                # routing decision and mutates state. Done AFTER text frames
                # have streamed past so TTS gets the closing line before
                # any session tear-down.
                tag = self._captured_tag
                spoken_was_empty = not "".join(self._spoken).strip()
                self._spoken.clear()
                self._captured_tag = None
                if tag is not None:
                    decision = await apply_route(self._session, tag)
                    # Follow-up inference rules (mirror text mode):
                    # - interrupt / return-into-caller → ALWAYS follow up.
                    # - silent take_exit (no reply text) → follow up so the
                    #   destination speaks instead of leaving dead air.
                    # - take_exit with reply / end / stay → no follow-up.
                    if decision is not None and not self._session.ended:
                        needs_followup = (
                            decision.kind in ("trigger_interrupt", "return")
                            or (decision.kind == "take_exit" and spoken_was_empty)
                        )
                        if needs_followup:
                            # Push UPSTREAM so the LLM service (which sits
                            # above this processor in the pipeline) sees the
                            # frame and runs another inference against the
                            # newly-loaded destination prompt.
                            await self.push_frame(
                                LLMRunFrame(), FrameDirection.UPSTREAM
                            )
        await self.push_frame(frame, direction)

    def _consume_text(self, chunk: str) -> str:
        """Process one streamed text chunk through the state machine.
        Returns the substring that should be forwarded to TTS (may be empty).
        """
        out_parts: list[str] = []
        i = 0
        while i < len(chunk):
            if self._state == self._FORWARDING:
                lt = chunk.find("<", i)
                if lt < 0:
                    out_parts.append(chunk[i:])
                    break
                if lt > i:
                    out_parts.append(chunk[i:lt])
                self._state = self._BUFFERING
                self._buf = "<"
                i = lt + 1
            elif self._state == self._IN_THINK:
                # Scan for `</think>` close. Until we find it, every byte is
                # discarded. The close-tag spelling is loose (`</think>`,
                # `</ think >`, case-insensitive) to match the regex used in
                # routing_protocol.strip_think_blocks.
                end = _find_think_close(chunk, i)
                if end < 0:
                    i = len(chunk)
                else:
                    self._state = self._FORWARDING
                    i = end
            else:  # BUFFERING
                gt = chunk.find(">", i)
                if gt < 0:
                    self._buf += chunk[i:]
                    i = len(chunk)
                else:
                    self._buf += chunk[i : gt + 1]
                    i = gt + 1
                resolved = self._resolve_buffer()
                if resolved is _ENTERED_THINK:
                    self._state = self._IN_THINK
                    self._buf = ""
                elif resolved is not None:
                    out_parts.append(resolved)
                    self._state = self._FORWARDING
                    self._buf = ""
                elif len(self._buf) > self._MAX_BUFFER:
                    out_parts.append(self._buf)
                    self._state = self._FORWARDING
                    self._buf = ""
        spoken = "".join(out_parts)
        if spoken:
            self._spoken.append(spoken)
        return spoken

    def _resolve_buffer(self) -> "str | object | None":
        """Inspect the current buffer. Outcomes:

        - `<route .../>` self-closing: capture, return "" (swallow bytes).
        - `<think>` opening tag: signal IN_THINK via the _ENTERED_THINK
          sentinel so the caller switches state to discard mode.
        - `<...>` complete non-reserved tag: flush verbatim.
        - Incomplete prefix that COULD still grow into a reserved sentinel:
          return None (keep buffering).
        - Incomplete prefix that can't be reserved: flush early.
        """
        if ">" not in self._buf:
            if not _could_be_reserved_prefix(self._buf):
                return self._buf
            return None
        # Buffer is `<...>`. Check `<think>` open first (block-form, not
        # self-closing). The regex `<\s*think\b[^>]*>` matches the open tag.
        if _THINK_OPEN_RE.match(self._buf):
            return _ENTERED_THINK
        # Then try the route tag.
        cleaned, tag = routing_protocol.find_tag(self._buf)
        if tag is None:
            return self._buf  # plain `<...>` text — flush
        if tag.is_valid:
            if self._captured_tag is None:
                self._captured_tag = tag
            else:
                logger.warning(
                    "multiple route tags in one response; ignoring all but the first"
                )
        return cleaned


# Sentinel returned by _resolve_buffer when the buffer opens a <think> block.
# Distinct from None / strings so the caller can tell "switch to IN_THINK"
# apart from "flush this text" or "keep buffering".
#
# SMELL: the `str | object | None` return shape works for two sentinels but
# won't generalize. If a third reserved sentinel is added (see
# routing_protocol module docstring), refactor `_resolve_buffer` to return
# a typed ResolveResult dataclass with an explicit action enum + payload.
# Likewise `_could_be_reserved_prefix` / `_could_grow_into` should be
# driven by a single registered list of prefixes rather than hand-rolled.
_ENTERED_THINK = object()

_THINK_OPEN_RE = re.compile(r"<\s*think\b[^>]*>", re.IGNORECASE)


def _could_be_reserved_prefix(buf: str) -> bool:
    """True if `buf` could still grow into a reserved sentinel
    (`<route ...` or `<think>...`). False once we're sure it can't — so the
    buffering processor can flush early."""
    return _could_grow_into(buf, "<route") or _could_grow_into(buf, "<think")


def _could_grow_into(buf: str, target: str) -> bool:
    """True if `buf` is a prefix of `target` or vice-versa with a valid
    boundary (whitespace / `/` / `>`)."""
    if len(buf) <= len(target):
        return target.startswith(buf.lower())
    head = buf[: len(target)].lower()
    if head != target:
        return False
    nxt = buf[len(target)]
    return nxt in (" ", "\t", "\n", "/", ">")


def _find_think_close(chunk: str, start: int) -> int:
    """Return the index just AFTER `</think>` in chunk starting at start, or
    -1 if not present. Mirrors the loose close-tag spelling of the regex."""
    match = re.search(r"<\s*/\s*think\s*>", chunk[start:], re.IGNORECASE)
    if match is None:
        return -1
    return start + match.end()




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
    # No tools — routing is in-text via <route .../> tags. Clear any prior
    # tool schema so providers don't switch into tool-mode (causes silent
    # function-call-only responses we used to patch around).
    s.llm_context.set_tools(NOT_GIVEN)
    active = s.spec.flows_by_id[s.state.active_flow_id]
    new_prompt = build_system_prompt(
        s.spec, active, s.state.language,
        variables=s.state.variables,
        plan=plan,
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


async def apply_route(
    session: Session, tag: RouteTag
) -> routing_mod.Decision | None:
    """Framework-agnostic core of in-text routing: resolve a parsed route
    tag into a Decision, mutate session state, and re-plan if a transition
    happened.

    The tag is whatever was inside the LLM's `<route .../>` emission. By
    the time we get here:
      - the LLM's conversational reply has already streamed to TTS / been
        returned to the text caller (the tag is the trailing token); and
      - the streaming stripper (voice) / response splitter (text) has
        already removed the tag bytes from the channel that would reach
        the user.

    Returns the Decision (None if no current_plan — guard against the
    parse firing before PreLLMPlanner). Unlike the legacy tool-call path,
    callers do NOT need to issue a follow-up inference: the closing line
    streamed before the tag did, so END can tear down immediately and
    flow transitions don't need a "silent take_exit" workaround.
    """
    s = session
    s.tool_handler_fired_this_turn = True

    if s.current_plan is None:
        logger.warning("route tag arrived without a current_plan; ignoring")
        return None
    if not tag.is_valid:
        logger.warning("route tag invalid (neither/both exit and interrupt); ignoring")
        return None

    llm_results = routing_protocol.to_llm_results(tag)
    decision = routing_mod.resolve(s.current_plan, s.spec, llm_results)
    await _apply_decision(decision, s, llm_results=llm_results)

    if decision.kind in ("take_exit", "trigger_interrupt", "return"):
        # Re-plan for the new active flow. The next inference (whether it's
        # an immediate silent-take_exit follow-up or the next user turn)
        # needs the destination flow's prompt + routing protocol loaded.
        plan_for_active_flow(s)

    return decision


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
    """Fire an exit: assigns → capability actions (sync await, bind outputs) →
    state transition.

    Order is deliberate: assigns first so capability inputs see the new values;
    then each action runs sequentially, with declared `outputs` writing into
    variable scope as they return. If multiple actions on this exit declare the
    same output name, last-write-wins — actions execute in the order they
    appear in `exit_path.actions`. Spec authors avoid collisions by namespacing
    output names per-capability (e.g., `policy_active`, not `active`).
    """
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

    # 2) Capabilities — synchronous dispatch so declared outputs land before
    #    the transition. See RUNNER-PLAN §"Capability outputs bind to variable
    #    scope" for the rationale.
    for action in exit_path.actions:
        cap = s.spec.capabilities_by_id.get(action.capability_id)
        invocation, result = await s.capabilities.invoke(
            action.capability_id, s.state.variables
        )
        s.events.emit(
            CapabilityInvoked(
                session_id=s.session_id,
                capability_name=invocation.capability_name,
                args=invocation.args,
            )
        )
        s.events.emit(
            CapabilityReturned(
                session_id=s.session_id,
                capability_name=result.capability_name,
                result=result.result,
                error=result.error,
            )
        )
        # Bind declared outputs into variable scope. Failure → outputs simply
        # don't land; downstream calculation conditions can branch on
        # `var != True` / `var == None` to cover both False and undefined.
        if cap and result.result and isinstance(result.result, dict):
            for output_name in cap.outputs:
                if output_name not in result.result:
                    continue
                value = result.result[output_name]
                s.state.variables[output_name] = value
                s.events.emit(
                    VariableSet(
                        session_id=s.session_id,
                        variable_name=output_name,
                        value=value,
                        method="capability",
                        source_flow_id=source_flow.id,
                        source_exit_path_id=exit_path.id,
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
        # In-text routing: the closing line streamed BEFORE this tag was
        # parsed, so tear down immediately. No need for the pending_end /
        # silent-follow-up dance the tool-call path used to require.
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

    # RETURN with no caller frame collapses to END (per schema). The reply
    # already streamed before the route tag, so tear down immediately.
    if not s.state.has_caller:
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
        s.spec, flow, s.state.language, variables=s.state.variables
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
        if s.tool_handler_fired_this_turn:
            return  # apply_route already advanced state earlier in this turn

        # No route tag fired — the LLM produced plain text. If this is a
        # terminator flow (single unconditional exit), fire its pre-resolved
        # shortcut so it ends after speaking its line instead of looping.
        if await apply_planned_shortcut(s):
            return
        s.state.increment_turn()
