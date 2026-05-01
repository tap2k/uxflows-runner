"""Bare audio pipeline — no spec, no dispatcher.

Useful as a debug surface when iterating on voices, STT, VAD, etc., where
spec interpretation would just be in the way. Mirrors the Phase 0 pipeline
that shipped in commit a40f7b3 before the dispatcher landed.
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

RAW_SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Greet the user warmly when the "
    "conversation starts, then have a short, natural conversation. "
    "Keep replies under two sentences."
)


async def run_raw_session(connection: SmallWebRTCConnection, config: Config) -> None:
    """Bare audio loop — Pipecat default behavior with a hardcoded prompt."""
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

    context = LLMContext(messages=[{"role": "system", "content": RAW_SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, enable_metrics=False),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("[raw] client connected — kicking off greeting")
        context.add_message({"role": "user", "content": "Say hello and ask how you can help."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("[raw] client disconnected — cancelling pipeline")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
