"""Pipecat pipeline construction.

Phase 0: hardcoded system prompt, Google STT/LLM/TTS, SmallWebRTC transport
(plain WebRTC peer connection — browser uses RTCPeerConnection + getUserMedia,
no protobuf framing required). The dispatcher (spec interpreter) lands in
Phase 1 and slots in between context_aggregator.user() and the LLM.
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

PHASE_0_SYSTEM_PROMPT = (
    "You are a friendly voice assistant in a Phase 0 demo. "
    "Greet the user warmly when the conversation starts, then have a short, "
    "natural conversation. Keep replies under two sentences."
)


async def run_session(connection: SmallWebRTCConnection, config: Config) -> None:
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

    context = LLMContext(messages=[{"role": "system", "content": PHASE_0_SYSTEM_PROMPT}])
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
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("client connected — kicking off greeting")
        context.add_message({"role": "user", "content": "Say hello and ask how you can help."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("client disconnected — cancelling pipeline")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
