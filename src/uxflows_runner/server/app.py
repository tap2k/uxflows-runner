"""FastAPI app — SmallWebRTC offer endpoint + static test page.

Surface:
  GET  /                  — static test page (web/index.html)
  GET  /audio-test.html   — bare audio debug page (no spec, hardcoded prompt)
  GET  /health            — liveness probe
  POST /api/offer         — WebRTC SDP exchange; body MAY include `spec` (a
                            full v0 spec JSON object) to drive THIS session.
                            If absent, falls back to UXFLOWS_SPEC_PATH from env.
  POST /api/offer/raw     — same SDP exchange, but runs a bare audio pipeline
                            with a hardcoded system prompt — no spec, no
                            dispatcher. Debug surface for voices / STT / VAD.
  POST /api/offer/patch   — ICE candidate patch (shared between both offer paths)
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import ValidationError

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from uxflows_runner.config import Config
from uxflows_runner.server.pipeline import run_session
from uxflows_runner.server.pipeline_raw import run_raw_session
from uxflows_runner.spec.loader import LoadedSpec, load_spec, parse_spec


# Cap the offer body to keep a misuse case (someone POSTing a megabyte of
# nothing) from eating memory. Realistic spec sizes are 20-200KB; 1MB leaves
# 5-50x headroom for SDP + spec.
MAX_OFFER_BODY_BYTES = 1_000_000


WEB_DIR = Path(__file__).resolve().parents[3] / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = Config.from_env()
    # Eagerly load the env-default spec so a missing/broken file fails at
    # boot, not on the first connect. Per-session overrides arrive via the
    # offer body and are parsed there.
    app.state.default_spec = load_spec(app.state.config.spec_path)
    app.state.webrtc = SmallWebRTCRequestHandler()
    app.state.tasks = set()
    logger.info(
        "uxflows-runner ready (default_spec={} agent={} model={} voice={} project={})",
        app.state.config.spec_path,
        app.state.default_spec.agent.id,
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
    raw = await request.body()
    if len(raw) > MAX_OFFER_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"offer body exceeds {MAX_OFFER_BODY_BYTES} bytes",
        )
    body = json.loads(raw)
    # Strip our extension fields before handing the rest to Pipecat's SDP
    # request parser — it errors on unknown keys.
    raw_spec = body.pop("spec", None)
    req = SmallWebRTCRequest.from_dict(body)

    spec = _resolve_spec(raw_spec)

    async def on_connection(connection: SmallWebRTCConnection):
        # Run the pipeline in a background task — handle_web_request awaits the
        # callback before returning the SDP answer, so we can't block here.
        task = asyncio.create_task(
            run_session(
                connection,
                app.state.config,
                spec,
                execution_config_path=app.state.config.execution_config_path,
            )
        )
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)

    return await app.state.webrtc.handle_web_request(req, on_connection)


def _resolve_spec(raw_spec: dict | None) -> LoadedSpec:
    """Pick the spec for THIS session: prefer the uploaded spec; fall back to
    the env-default loaded at startup."""
    if raw_spec is None:
        return app.state.default_spec
    try:
        return parse_spec(json.dumps(raw_spec))
    except ValidationError as exc:
        # Surface the first error so the browser can show something useful.
        raise HTTPException(
            status_code=400,
            detail=f"invalid spec: {exc.errors()[:3]}",
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid spec: {exc}")


@app.post("/api/offer/raw")
async def webrtc_offer_raw(request: Request):
    """Bare audio loop — no spec, hardcoded prompt. Voice / STT / VAD debug."""
    body = await request.json()
    req = SmallWebRTCRequest.from_dict(body)

    async def on_connection(connection: SmallWebRTCConnection):
        task = asyncio.create_task(run_raw_session(connection, app.state.config))
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
