"""FastAPI app — SmallWebRTC offer endpoint + text chat endpoints + static test page.

Surface:
  GET  /                  — static test page (web/index.html)
  GET  /audio-test.html   — bare audio debug page (no spec, hardcoded prompt)
  GET  /text.html         — text chat debug page (no audio, BYOK key)
  GET  /health            — liveness probe
  POST /api/offer         — WebRTC SDP exchange; body MAY include `spec` (a
                            full v0 spec JSON object) to drive THIS session.
                            If absent, falls back to UXFLOWS_SPEC_PATH from env.
  POST /api/offer/raw     — same SDP exchange, but runs a bare audio pipeline
                            with a hardcoded system prompt — no spec, no
                            dispatcher. Debug surface for voices / STT / VAD.
  POST /api/offer/patch   — ICE candidate patch (shared between both offer paths)
  POST /api/chat/session  — start a text session (BYOK Google AI Studio key);
                            returns session_id + opening turn + events
  POST /api/chat/turn     — send a user turn, get agent reply + events
  POST /api/chat/end      — tear down a text session early
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from uxflows_runner.config import Config
from uxflows_runner.dispatcher.capabilities import load_execution_config
from uxflows_runner.server.pipeline import run_session
from uxflows_runner.server.pipeline_raw import run_raw_session
from uxflows_runner.server.text_registry import TextSessionRegistry
from uxflows_runner.server.text_session import (
    DEFAULT_MODEL,
    SessionAlreadyEnded,
    TextSession,
)
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
    app.state.text_sessions = TextSessionRegistry()
    app.state.text_sessions.start_sweeper()
    logger.info(
        "uxflows-runner ready (default_spec={} agent={} model={} voice={} project={})",
        app.state.config.spec_path,
        app.state.default_spec.agent.id,
        app.state.config.llm_model,
        app.state.config.tts_voice,
        app.state.config.google_project_id,
    )
    yield
    await app.state.text_sessions.stop_sweeper()
    await app.state.text_sessions.drop_all()
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


# --------------------------------------------------------------------------
# Text chat endpoints (Phase 1.5 — see SIMULATE-PLAN.md §"Wire protocol")
# --------------------------------------------------------------------------


class StartSessionRequest(BaseModel):
    spec: dict[str, Any] | None = None
    api_key: str | None = None
    model: str | None = None
    language: str | None = None


class StartSessionResponse(BaseModel):
    session_id: str
    agent_text: str
    events: list[dict[str, Any]]
    ended: bool


class TurnRequest(BaseModel):
    session_id: str
    user_text: str = Field(..., min_length=1)


class TurnResponse(BaseModel):
    agent_text: str
    events: list[dict[str, Any]]
    ended: bool


class EndRequest(BaseModel):
    session_id: str


def _events_to_payload(events: list) -> list[dict[str, Any]]:
    return [e.model_dump(mode="json") for e in events]


@app.post("/api/chat/session", response_model=StartSessionResponse)
async def chat_start_session(req: StartSessionRequest) -> StartSessionResponse:
    spec = _resolve_spec(req.spec)
    endpoints = (
        load_execution_config(app.state.config.execution_config_path)
        if app.state.config.execution_config_path
        else {}
    )
    try:
        ts, opening = await TextSession.start(
            spec=spec,
            api_key=req.api_key,
            model=req.model,
            language=req.language,
            execution_endpoints=endpoints,
            config=app.state.config,
        )
    except Exception as exc:  # noqa: BLE001
        # Most likely: invalid API key, network error, or LLM auth failure
        # during the chatbot_initiates opening turn. Surface the message;
        # spec-shape errors are caught earlier by _resolve_spec.
        logger.warning("text session start failed: {}", exc)
        raise HTTPException(status_code=400, detail=f"session start failed: {exc}")

    app.state.text_sessions.register(ts)
    events = _events_to_payload(ts.drain_events())
    return StartSessionResponse(
        session_id=ts.session_id,
        agent_text=opening,
        events=events,
        ended=ts.ended,
    )


@app.post("/api/chat/turn", response_model=TurnResponse)
async def chat_turn(req: TurnRequest) -> TurnResponse:
    ts = app.state.text_sessions.get(req.session_id)
    if ts is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    try:
        agent_text = await ts.turn(req.user_text)
    except SessionAlreadyEnded as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("text session turn failed (session={}): {}", req.session_id, exc)
        raise HTTPException(status_code=500, detail=f"turn failed: {exc}")

    events = _events_to_payload(ts.drain_events())
    return TurnResponse(agent_text=agent_text, events=events, ended=ts.ended)


@app.post("/api/chat/end")
async def chat_end(req: EndRequest) -> dict[str, bool]:
    await app.state.text_sessions.drop(req.session_id)
    return {"ok": True}


# Static test page — mount last so it doesn't shadow API routes.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:
    logger.warning("web/ not found at {} — static page disabled", WEB_DIR)
