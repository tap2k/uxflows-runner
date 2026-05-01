"""Pipecat pipeline construction.

Phase 1: spec-driven dispatcher slotted between context_aggregator.user() and
the LLM. The PreLLMPlanner mutates LLMContext per turn (system prompt + tools)
based on the active flow; tool handlers do the routing + state mutation; the
PostLLMResolver handles plain-text turns where no tool fired.
"""

from __future__ import annotations

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.google.vertex.llm import GoogleVertexLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from uxflows_runner.config import Config
from uxflows_runner.dispatcher.capabilities import CapabilityDispatcher, load_execution_config
from uxflows_runner.dispatcher.processor import (
    PostLLMResolver,
    PreLLMPlanner,
    add_capability_result_listener,
    register_dispatcher_tools,
)
from uxflows_runner.dispatcher.prompt_builder import build_system_prompt
from uxflows_runner.dispatcher.session import Session
from uxflows_runner.events.emitter import LoggingEventEmitter
from uxflows_runner.spec.loader import LoadedSpec


async def run_session(
    connection: SmallWebRTCConnection,
    config: Config,
    spec: LoadedSpec,
    execution_config_path: str | None = None,
) -> None:
    """Run a single voice session bound to one WebRTC peer connection."""
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = GoogleSTTService(credentials_path=config.google_credentials_path)
    tts = GoogleTTSService(
        credentials_path=config.google_credentials_path,
        settings=GoogleTTSService.Settings(voice=config.tts_voice),
    )
    llm = GoogleVertexLLMService(
        credentials_path=config.google_credentials_path,
        project_id=config.google_project_id,
        location=config.google_location,
        settings=GoogleVertexLLMService.Settings(model=config.llm_model),
    )

    # Compose the entry flow's prompt up front; PreLLMPlanner re-syncs each
    # turn so this is just a sane starting state.
    entry_flow = spec.entry_flow
    lang = spec.agent.meta.languages[0] if spec.agent.meta.languages else "en-US"
    initial_prompt = build_system_prompt(spec.agent, entry_flow, lang)
    context = LLMContext(messages=[{"role": "system", "content": initial_prompt}])
    context_aggregator = LLMContextAggregatorPair(context)

    # Capability dispatch — sibling execution.json keyed by capability name.
    endpoints = (
        load_execution_config(execution_config_path) if execution_config_path else {}
    )
    capabilities = CapabilityDispatcher(spec=spec, endpoints=endpoints)

    events = LoggingEventEmitter()
    session = Session.start(
        spec=spec,
        llm_context=context,
        events=events,
        capabilities=capabilities,
        language=lang,
    )
    add_capability_result_listener(session)

    register_dispatcher_tools(llm, session)

    pre = PreLLMPlanner(session)
    post = PostLLMResolver(session)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            pre,
            llm,
            tts,
            transport.output(),
            post,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info(
            "client connected — entry_flow={} agent={}",
            entry_flow.id,
            spec.agent.id,
        )
        session.emit_session_started()
        session.emit_flow_entered(entry_flow.id, via="entry")
        if spec.agent.chatbot_initiates:
            # Kick the agent off — it should produce the entry flow's opening
            # naturally from the system prompt + scripts.
            context.add_message({"role": "user", "content": "(begin)"})
            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("client disconnected — cancelling pipeline")
        await capabilities.aclose()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
