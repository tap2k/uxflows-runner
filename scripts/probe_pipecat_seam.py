"""Probe two open questions about the Pipecat seam:

  Q1. Does pushing LLMSetToolsFrame mid-pipeline cause the *next* LLM
      inference to use the new tool list?
  Q2. With run_llm=False on the function-call result_callback, what's the
      ordering between the registered handler completing and the LLM service
      pushing LLMFullResponseEndFrame? Critical for deciding whether the
      resolver can listen on the end-frame.

Approach: build a minimal pipeline with a fake user-context source feeding the
real GoogleVertexLLMService (Gemini 2.5 Flash, same as production). Register
a `take_exit_path` tool. Push two consecutive LLMContextFrames; between them,
push an LLMSetToolsFrame with a *different* tool. Capture all outgoing frames
in a sink processor and dump the timeline.

Run: uv run python scripts/probe_pipecat_seam.py
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    StartFrame,
    TextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.vertex.llm import GoogleVertexLLMService
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultProperties

load_dotenv()


@dataclass
class TimelineEvent:
    t: float
    label: str


timeline: list[TimelineEvent] = []
START = time.monotonic()


def log(label: str) -> None:
    timeline.append(TimelineEvent(t=time.monotonic() - START, label=label))


class FrameSpy(FrameProcessor):
    """Sink that records every frame name + extra metadata for ordering."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (StartFrame, EndFrame)):
            return
        name = type(frame).__name__
        extra = ""
        if isinstance(frame, TextFrame):
            extra = f" text={frame.text[:60]!r}"
        if isinstance(frame, FunctionCallInProgressFrame):
            extra = f" name={frame.function_name}"
        if isinstance(frame, FunctionCallResultFrame):
            extra = f" name={frame.function_name} run_llm={getattr(frame, 'run_llm', None)}"
        log(f"[spy] frame={name}{extra}")
        await self.push_frame(frame, direction)


# ---- Tool definitions for Q1 ----
TOOLS_A = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="take_exit_path",
            description="Pick an exit path. Use after each user turn.",
            properties={
                "exit_path_id": {
                    "type": "string",
                    "enum": ["xp_to_coffee", "xp_to_tea"],
                    "description": "Which path.",
                }
            },
            required=["exit_path_id"],
        )
    ]
)

TOOLS_B = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="take_exit_path",
            description="Pick an exit path. Use after each user turn.",
            properties={
                "exit_path_id": {
                    "type": "string",
                    "enum": ["xp_to_confirm", "xp_to_cancel"],  # different enum!
                    "description": "Which path.",
                }
            },
            required=["exit_path_id"],
        )
    ]
)


async def take_exit_path_handler(params: FunctionCallParams) -> None:
    log(f"[handler] start args={dict(params.arguments)}")
    # Simulate ~50ms of work to make ordering visible.
    await asyncio.sleep(0.05)
    log("[handler] before result_callback")
    await params.result_callback(
        {"ok": True}, properties=FunctionCallResultProperties(run_llm=False)
    )
    log("[handler] after result_callback")


async def main() -> None:
    creds = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4")
    model = os.environ.get("UXFLOWS_LLM_MODEL", "gemini-2.5-flash")

    llm = GoogleVertexLLMService(
        credentials_path=creds,
        project_id=project,
        location=location,
        settings=GoogleVertexLLMService.Settings(model=model),
        run_in_parallel=True,  # the production default — we want to see what this gives us
    )

    llm.register_function("take_exit_path", take_exit_path_handler)

    spy = FrameSpy()

    # Build a tiny pipeline: llm -> spy. We feed LLMContextFrame directly to
    # the LLM service; no transport, no STT/TTS, no aggregator.
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    pipeline = Pipeline([llm, spy])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False, enable_metrics=False))

    async def driver() -> None:
        # Wait for StartFrame to settle.
        await asyncio.sleep(0.5)

        # ----- Turn 1: tools = TOOLS_A; user asks coffee or tea -----
        log("[driver] turn1: setting tools=A")
        ctx = LLMContext(messages=[
            {"role": "system", "content": "You are a barista. After each user turn, call take_exit_path with the right exit_path_id and reply briefly."},
            {"role": "user", "content": "I'd love a latte"},
        ])
        ctx.set_tools(TOOLS_A)
        log("[driver] turn1: queueing LLMContextFrame")
        await task.queue_frames([LLMContextFrame(context=ctx)])

        # Wait for turn1 to complete (LLMFullResponseEndFrame).
        await asyncio.sleep(8)
        log("[driver] turn1 done; pushing LLMSetToolsFrame to swap to B")
        await task.queue_frames([LLMSetToolsFrame(tools=TOOLS_B)])
        await asyncio.sleep(0.2)

        # ----- Turn 2: simulate that we've moved to a confirm flow -----
        # Build a fresh context that references TOOLS_B's enum (confirm/cancel).
        ctx2 = LLMContext(messages=[
            {"role": "system", "content": "You're confirming an order. After the user turn, call take_exit_path with xp_to_confirm or xp_to_cancel and reply briefly."},
            {"role": "user", "content": "yes that's right"},
        ])
        ctx2.set_tools(TOOLS_B)
        log("[driver] turn2: queueing LLMContextFrame")
        await task.queue_frames([LLMContextFrame(context=ctx2)])

        await asyncio.sleep(8)
        log("[driver] sending EndFrame")
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)
    await asyncio.gather(runner.run(task), driver())


def dump_timeline() -> None:
    print("\n=== TIMELINE ===")
    for ev in timeline:
        print(f"  {ev.t:7.3f}s  {ev.label}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        dump_timeline()
