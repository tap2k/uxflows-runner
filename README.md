# uxflows-runner

Voice/text runner for [UX4 v0 specs](../uxflows/SCHEMA.md). Pipecat-based dispatcher with self-contained browser test pages for both modes.

Plan and rationale: [RUNNER-PLAN.md](./RUNNER-PLAN.md). Strategy: [../uxflows/STRATEGY.md](../uxflows/STRATEGY.md). Editor integration is wired through the `/api/chat/*` endpoints below.

## Status

- **Phase 0** ✅ — bare audio loop end-to-end (Pipecat WebRTC + Google STT/LLM/TTS).
- **Phase 1** ✅ — v0 dispatcher: spec-driven flow interpretation, three-method routing (`direct` / `calculation` / `llm`), interrupts with `return_to_caller`, post-exit assigns + capability dispatch, event emission.
- **Phase 1.5** ✅ — text I/O adapter: `/api/chat/{session,turn,end}` endpoints, BYOK or env-fallback auth, `context_vars` for placeholder substitution + initial variable seeding.
- **Phase 2 (text path)** ✅ — editor canvas highlighting from event stream. `FlowNode` rings the active flow; edges pulse on `exit_path_taken`; variables stream into the simulate panel. Wired through HTTP (Phase 1.5's `/api/chat/*`), not SSE. Voice-mode → editor integration (SSE broker, `/api/offer` event subscription) deferred until a voice consumer needs it.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) if not already installed.
2. Sync dependencies:
   ```sh
   uv sync
   ```
3. **Auth** — pick one (or both):
   - **Vertex (default, used by voice mode and text-mode env-fallback)**: drop your GCP service-account JSON at `data/credentials.json`. The `data/` directory is gitignored. Service account needs *Vertex AI User* (Gemini), *Cloud Speech Client* (STT), *Cloud Text-to-Speech User* (TTS).
   - **AI Studio (BYOK, text mode only)**: get a key from <https://aistudio.google.com/app/apikey>. No setup file — passed per-session via the API.
4. Copy the env template and fill in your project ID:
   ```sh
   cp .env.example .env
   $EDITOR .env
   ```

## Run

```sh
uv run uxflows-runner serve
```

Then pick the mode you want:

| Mode | URL | What it does |
|---|---|---|
| **Voice** | <http://localhost:8000/> | Real-time voice via WebRTC. Mic in, TTS out. Full dispatcher loop. |
| **Text** | <http://localhost:8000/text.html> | Text chat against the dispatcher. No audio, no STT/TTS. Same routing/state/events as voice. |
| **Bare audio** | <http://localhost:8000/audio-test.html> | Audio loop with a hardcoded prompt — no spec, no dispatcher. Use to debug voice/STT/VAD when something feels off. |

`curl localhost:8000/health` returns `{"ok": true}` once the server has loaded.

### Quick sanity test (text, headless)

```sh
SPEC=$(cat examples/coffee.json)
curl -s -X POST http://localhost:8000/api/chat/session \
  -H "Content-Type: application/json" \
  -d "{\"spec\": $SPEC}" | python3 -m json.tool
```

You should get back a `session_id`, an opening agent turn, and `session_started` + `flow_entered` events. Send a turn:

```sh
curl -s -X POST http://localhost:8000/api/chat/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<paste-id>","user_text":"i'"'"'d like a latte"}' | python3 -m json.tool
```

## API surface

Three endpoints power the text mode (canonical implementations in [`server/app.py`](src/uxflows_runner/server/app.py)):

- **`POST /api/chat/session`** — start a session. Body: `{spec, api_key?, model?, language?, context_vars?}`. Returns `{session_id, agent_text, events, ended}`. `api_key` falls back to env Vertex creds when omitted; `context_vars` seeds the variable bag and substitutes `{KEY}` placeholders in the system prompt (case-insensitive; unfilled placeholders stay as `{KEY}` literal).
- **`POST /api/chat/turn`** — send a user turn. Body: `{session_id, user_text}`. Returns `{agent_text, events, ended}`.
- **`POST /api/chat/end`** — explicit cleanup. Idle sessions get GC'd after 30 min regardless.

Voice uses WebRTC SDP exchange via `/api/offer` (Pipecat's `SmallWebRTCRequestHandler`).

## Layout

```
src/uxflows_runner/
  cli.py                         # `uxflows-runner serve`
  config.py                      # env-driven config
  spec/
    types.py, loader.py          # pydantic v0 spec types + per-spec lookup tables
  dispatcher/
    flow_state.py                # stack-based FlowState (interrupts push/pop) + variable bag
    expressions.py               # calculation engine (simpleeval-based)
    methods.py                   # three-method evaluator (direct / calculation / llm)
    routing.py                   # plan() + resolve() — exit-path eval, interrupt collection
    assigns.py                   # exit-fired variable assignment
    capabilities.py              # HTTP fire-and-forget + retrieval stub
    prompt_builder.py            # per-flow system prompt + per-turn tool schema + {KEY} substitution
    processor.py                 # Pipecat seam — PreLLMPlanner, tool handlers, PostLLMResolver, apply_tool_call
    session.py                   # per-connection Session (FlowState + LLMContext + emitter)
  events/
    schema.py                    # pydantic event types — runner-side contract
    emitter.py                   # LoggingEmitter / QueueEmitter / BufferingEmitter
  server/
    app.py                       # FastAPI: /api/offer (voice), /api/chat/* (text), static /web mount
    pipeline.py                  # Pipecat voice pipeline
    pipeline_raw.py              # Bare audio pipeline (no spec/dispatcher)
    text_session.py              # Text adapter — drives dispatcher per-turn without Pipecat audio
    text_registry.py             # In-memory TextSession registry + idle GC
web/
  index.html, client.js          # voice debug page (vanilla RTCPeerConnection)
  text.html, text.js, text.css   # text debug page (vanilla fetch)
  audio-test.html, audio-test.js # bare audio debug page
  style.css
examples/
  coffee.json                    # self-contained order-bot spec
data/
  credentials.json               # GCP service-account JSON (gitignored)
tests/                           # 95 passing — dispatcher core + text-mode e2e
```

## Why three debug pages

Each page isolates a layer:

- **voice page (`/`)** — full stack, the canonical voice surface.
- **text page (`/text.html`)** — same dispatcher as voice, no audio. Use when you want to iterate on a spec's logic without burning STT/TTS minutes, when you're on a flaky mic, or when reviewing flow transitions visually beats listening. Real designers use this through the editor's Simulate panel ([`../uxflows/components/runtime/SimulatePanel.tsx`](../uxflows/components/runtime/SimulatePanel.tsx)); the standalone page is for runner-side dev work.
- **bare audio page (`/audio-test.html`)** — no spec, no dispatcher, hardcoded prompt. First-pass triage for "is the audio path or the dispatcher misbehaving?" If the issue reproduces here, it's audio.

All three stay around as debug surfaces forever. They don't disappear when the editor integration lands.

## Known rough edges

- **Walkaway gap** — Gemini sometimes responds to a graceful goodbye with text only, no `take_exit_path`. Session stays "live" until idle GC. Documented in [RUNNER-PLAN §"Live-test follow-up"](./RUNNER-PLAN.md#live-test-follow-up-2026-04-30-evening). Click Reset (or close the tab) to recover.
- **Silent take_exit (voice mode only)** — fixed in text mode (no-tools follow-up inference). Voice mode TODO; sketch in same section.
- **Single user, single host** — v0 deployment model. Concurrent sessions across one process should work but are untested at scale.
