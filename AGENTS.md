# This is NOT the Pipecat you know
This codebase pins **Pipecat 1.1.0**. APIs, module paths, and frame names have churned between versions; class signatures in your training data may be stale. Before writing code that touches Pipecat, read the relevant module under `.venv/lib/python3.12/site-packages/pipecat/` (or `uv pip show pipecat-ai` for the path). Heed deprecation notices on `params=` / `settings=` constructors — many services moved to `settings=` recently.

# Do not keep agent memory for this project

Do not write to the agent memory system for this project. If prior memories exist, ignore them. Persistent guidance, principles, and project context belong in this file (and the related docs listed below), not in per-conversation memory files. When the user tells you something worth remembering across conversations, propose adding it here instead.

# uxflows-runner

Python runner for [UX4 v0 specs](../uxflows/SCHEMA.md). Interprets a v0 JSON spec live, drives a real-time voice or text conversation, and emits a UX4-id-keyed event stream that downstream consumers (the editor's canvas, eventually whatsupp2's simulator) read.

## Product Context

uxflows-runner is one of three sibling repos that compose UX4. Read these before making non-obvious architectural decisions:

- [`../uxflows/AGENTS.md`](../uxflows/AGENTS.md) — editor architecture, mission, schema rationale.
- [`../uxflows/SCHEMA.md`](../uxflows/SCHEMA.md) — v0 + v1 spec schema (the contract).
- [`../uxflows/STRATEGY.md`](../uxflows/STRATEGY.md) — cross-repo product strategy and roadmap.
- [`../whatsupp2/AGENT-TESTING.md`](../whatsupp2/AGENT-TESTING.md) — simulation/evaluation product design.
- [RUNNER-PLAN.md](./RUNNER-PLAN.md) — operational plan for this runner: phases, decisions, risks, open questions.

The schema is the contract across all three repos. The runner imports nothing from uxflows; it consumes a v0 JSON file and writes events.

## Mission

Interpret v0 specs live. **Drive voice (Pipecat pipeline) or text (WebSocket) I/O depending on `agent.meta.modes`**, evaluate flows / captures / routing through the three-method substrate, dispatch to capabilities, and emit a single event stream that multiple consumers read.

Two consumers by design:

1. **Standalone test page** at [web/](./web/) — runner-served debug surface (vanilla `RTCPeerConnection` + `getUserMedia`, no Pipecat browser SDK). Self-contained: one process serves the page and the WebRTC offer endpoint.
2. **Editor canvas integration** — eventual second consumer of the same event stream; lives in [`../uxflows/`](../uxflows/), not here.

The runner has dual identity: prototyping component when invoked from the editor, simulation substrate when invoked by whatsupp2. One executor, two roles.

## Tech Stack

- **Python 3.12+**, managed by **uv** (`uv sync`, `uv add`, `uv run`).
- **FastAPI + uvicorn** — single process serves both `/api/offer` (WebRTC) and the static `web/` page.
- **Pipecat 1.1.0** — pipeline framework; provides `SmallWebRTCTransport`, `SileroVADAnalyzer`, `GoogleSTTService`, `GoogleVertexLLMService`, `GoogleTTSService`, context aggregators.
- **aiortc** — Pipecat's WebRTC backend (transitive dep of `pipecat-ai[webrtc]`).
- **Google Cloud SDK** — STT, TTS, Vertex AI Gemini. Single service-account JSON authenticates all three.

For v0, the provider stack is **Google all-three** (Gemini 2.5 Flash via Vertex, Cloud STT, Cloud TTS Chirp 3 HD). Provider abstraction lives in `config.py`; swapping individual services (Deepgram for STT, Cartesia for TTS) is a one-line change. See [RUNNER-PLAN.md](./RUNNER-PLAN.md#provider-stack-default) for the rationale.

## Module Boundaries

The dispatcher (spec interpreter) **must stay framework-agnostic**. This is the deliberate hedge that lets us swap to Patter for telephony, or reuse the dispatcher inside whatsupp2's text simulator, without rework.

- **Pipecat-specific code is confined to** `src/uxflows_runner/server/pipeline.py` and (when it lands) `src/uxflows_runner/dispatcher/processor.py` (the `FrameProcessor` wrapper).
- **The rest of the dispatcher** (`methods.py`, `expressions.py`, `captures.py`, `routing.py`, `capabilities.py`, `prompt_builder.py`) imports nothing from Pipecat.
- **Dispatcher interface**: `dispatch(user_text, flow_state, variables) -> (assistant_text, transitions, events)`.

The full layout (current + planned) is in [RUNNER-PLAN.md](./RUNNER-PLAN.md#repository-layout).

## Credentials & Secrets

- Service-account JSON lives at `data/credentials.json`. The whole `data/` directory is gitignored. **Never** commit credentials. **Never** copy them into `../uxflows/` — the editor must not see runtime keys. Specs themselves are flat files under `examples/` (e.g. `examples/coffee.json`); credentials are infrastructure, not per-spec content.
- Env config in `.env` (gitignored). Use `.env.example` as the template.
- The runner *dispatches to* customer-owned capability backends (HTTP/MCP) and knowledge backends (retrieval); it doesn't *build* them. Authoring those is out of scope.

## Style

- Only add comments when the *why* is non-obvious. Never docstring-style multi-paragraph comments.
- Prefer editing existing files over creating new ones.
- Don't add backwards-compat shims. It's early — break freely.
- Match conventions in [`../uxflows/`](../uxflows/) where reasonable; same product, different language.
- The standalone `web/` test page is a debug surface, not a product surface. Keep it minimal — vanilla HTML/JS, no bundler, no React. Real UX lives in the editor (`../uxflows/`).

## Running

```sh
uv sync
cp .env.example .env  # fill in GOOGLE_CLOUD_PROJECT
# drop service-account JSON at data/credentials.json
uv run uxflows-runner serve
```

Then open <http://127.0.0.1:8000>.
