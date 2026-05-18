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
    RouteTagFrameProcessor,
    UserTranscriptWatcher,
)
from uxflows_runner.dispatcher.prompt_builder import build_system_prompt
from uxflows_runner.dispatcher.session import Session
from uxflows_runner.events.emitter import (
    JsonlEventEmitter,
    LoggingEventEmitter,
    MultiEventEmitter,
)
from uxflows_runner.spec.loader import LoadedSpec


async def run_session(
    connection: SmallWebRTCConnection,
    config: Config,
    spec: LoadedSpec,
    execution_config_path: str | None = None,
    context_vars: dict | None = None,
    language: str | None = None,
    mock_returns: dict[str, dict] | None = None,
) -> None:
    """Run a single voice session bound to one WebRTC peer connection.

    `context_vars` seeds the dispatcher's variable bag at session start —
    used both for `{KEY}` placeholder substitution in the composed system
    prompt and as initial values readable by routing conditions / capability
    inputs. Mirrors text mode (server/text_session.py). Not emitted as
    `variable_set` events (those are reserved for exit-path-fired assigns).
    """
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

    entry_flow = spec.entry_flow
    # No fallback: None means "all languages" — the prompt builder emits every
    # script bucket. Clients that want single-language behavior pass `language`
    # explicitly via the /api/offer body.
    lang = language
    # Build the context with a placeholder system message; we'll fill it in
    # below once context_vars (if any) have been seeded into the variable
    # bag, so {KEY} substitution sees them.
    context = LLMContext(messages=[{"role": "system", "content": ""}])
    context_aggregator = LLMContextAggregatorPair(context)

    # Capability dispatch — sibling execution.json keyed by capability name.
    endpoints = (
        load_execution_config(execution_config_path) if execution_config_path else {}
    )
    capabilities = CapabilityDispatcher(
        spec=spec,
        endpoints=endpoints,
        mock_returns=mock_returns,
    )

    events = LoggingEventEmitter()
    session = Session.start(
        spec=spec,
        llm_context=context,
        events=events,
        capabilities=capabilities,
        language=lang,
    )
    if config.event_log_dir is not None:
        jsonl_path = config.event_log_dir / f"{session.session_id}.jsonl"
        session.events = MultiEventEmitter([events, JsonlEventEmitter(jsonl_path)])
    if context_vars:
        session.state.variables.update(context_vars)

    initial_prompt = build_system_prompt(
        spec, entry_flow, lang, variables=session.state.variables
    )
    context.messages[0]["content"] = initial_prompt

    pre = PreLLMPlanner(session)
    user_transcript_watch = UserTranscriptWatcher(session)
    route_tag_processor = RouteTagFrameProcessor(session)
    post = PostLLMResolver(session)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_transcript_watch,
            context_aggregator.user(),
            pre,
            llm,
            route_tag_processor,
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
        # Emit a terminal event so traces always end with a SessionEnded —
        # otherwise voice sessions that end via the user closing the tab
        # just truncate, indistinguishable from a runner crash.
        if not session.ended:
            from uxflows_runner.events.schema import SessionEnded as _SessionEnded

            session.events.emit(
                _SessionEnded(session_id=session.session_id, reason="user_stop")
            )
            session.ended = True
        await capabilities.aclose()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
