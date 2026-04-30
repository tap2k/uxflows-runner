"""FastAPI app — SmallWebRTC offer endpoint + static test page.

Phase 0 surface:
  GET  /                  — static test page (web/index.html)
  GET  /health            — liveness probe
  POST /api/offer         — WebRTC SDP exchange; spawns a pipeline session
  POST /api/offer/patch   — ICE candidate patch
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from uxflows_runner.config import Config
from uxflows_runner.server.pipeline import run_session


WEB_DIR = Path(__file__).resolve().parents[3] / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = Config.from_env()
    app.state.webrtc = SmallWebRTCRequestHandler()
    app.state.tasks = set()
    logger.info(
        "uxflows-runner ready (model={}, voice={}, project={})",
        app.state.config.llm_model,
        app.state.config.tts_voice,
        app.state.config.google_project_id,
    )
    yield
    await app.state.webrtc.close()


app = FastAPI(lifespan=lifespan)

# Wide-open CORS for local dev. Tighten when this leaves localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/api/offer")
async def webrtc_offer(request: Request):
    body = await request.json()
    req = SmallWebRTCRequest.from_dict(body)

    async def on_connection(connection: SmallWebRTCConnection):
        # Run the pipeline in a background task — handle_web_request awaits the
        # callback before returning the SDP answer, so we can't block here.
        task = asyncio.create_task(run_session(connection, app.state.config))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)

    return await app.state.webrtc.handle_web_request(req, on_connection)


@app.post("/api/offer/patch")
async def webrtc_patch(request: Request):
    body = await request.json()
    req = SmallWebRTCPatchRequest(**body)
    await app.state.webrtc.handle_patch_request(req)
    return {"ok": True}


# Static test page — mount last so it doesn't shadow API routes.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:
    logger.warning("web/ not found at {} — static page disabled", WEB_DIR)
