# uxflows runner plan

Operational plan for the v0 voice runner — a local Python process that interprets a uxflows v0 spec, drives a real-time voice conversation in the browser, and streams UX4-id-keyed events back to the canvas for live highlighting.

Schema contract in [SCHEMA.md](../uxflows/SCHEMA.md). Codegen reference in [TRANSLATIONS.md](../uxflows/TRANSLATIONS.md). Overall strategy in [STRATEGY.md](../uxflows/STRATEGY.md).

## Why a runner at all

Three options were on the table for connecting the spec to a runtime:

1. **Codegen for each target** (Pipecat, LiveKit, LangGraph, OpenAI Agents SDK) — generate framework-native code. 4× maintenance; schema features that don't survive every target get dropped at the boundary; sim/prod drift becomes structural.
2. **Native runner** — interpret the spec directly. Stable IDs, multilingual scripts, eval metadata, guardrails flow through to runtime without round-tripping. One executor for prototyping and simulation. Schema evolution costs scale linearly, not by target count.
3. **Hybrid** — own the runner; ship codegen as demand-driven delivery formats later.

Choosing option 3 with **runner-first as canonical**. Codegen deferred until paid for. Rationale lives in [STRATEGY.md](../uxflows/STRATEGY.md) (decomposition as substrate, spec-as-contract, leverage-maximizing split).

The structural decision underneath: the runner does **not** own audio infrastructure. It sits inside a Pipecat pipeline as a custom `FrameProcessor` between context aggregation and LLM dispatch. Pipecat handles transport, VAD, STT, TTS, frame plumbing, barge-in, smart endpointing — everything below the cognitive layer. The dispatcher owns flow state, three-method evaluation, captures, routing, capability dispatch, event emission. This boundary keeps audio losses near zero while preserving full control over spec interpretation.

A side benefit worth naming: owning the runner contracts v1 schema scope. Several planned v1 fields existed only because v1 was being scoped to make codegen viable — `pipecat` hints, `pre_actions`, `context_strategy`, `respond_immediately`, aggressive typing pressure. With a native runner, these become runner config rather than spec fields, or relax in priority.

## Why Pipecat (vs. Patter, LiveKit, OpenAI Realtime, native audio)

Evaluated alternatives:

- **Patter** — telephony-first SDK with Twilio/Telnyx parity, AMD, voicemail drop, transfer, recording out of the box. Compelling for production phone deployments. Less natural fit for our dispatcher (we'd squat in the `onMessage` "custom LLM" hook rather than slot into a pipeline). Smaller community, less battle-tested. **Conclusion: revisit as a production telephony adapter when a client deployment materializes; not the v0 substrate.**
- **LiveKit Agents** — strong WebRTC story, but instructions-and-tool architecture flattens UX4 flow boundaries into prose. Wrong structural fit for graph-native interpretation.
- **OpenAI Realtime API** — speech-to-speech model bypasses text-level intercept. Our dispatcher requires text dispatch between user and assistant turns; Realtime closes off that seam.
- **Native audio (own VAD/AEC/transports)** — multi-engineer-year effort with zero differentiation. Rejected.

Pipecat wins on architectural fit (frame pipeline matches the dispatcher-as-processor model), browser/local dev story (`LocalAudioTransport` + `WebsocketTransport` + reference browser client), provider breadth (Deepgram/Cartesia/ElevenLabs/OpenAI/Anthropic all first-class), community depth, and frame-level event granularity that maximizes future canvas fidelity.

## Why browser-mic for v0

The first demo target is **browser-microphone voice with live canvas highlighting** — not phone telephony. Reasons:

- The visual demo (canvas pulsing live during a real voice conversation) is the strongest differentiator. Browser delivery makes it shareable in a meeting without anyone dialing a phone.
- Skips Twilio/Telnyx account setup, AMD edge cases, telephony codec issues. Phase 0 compresses meaningfully.
- Pipecat's `WebsocketTransport` + reference browser client is mature; mic capture and TTS playback in a web page is roughly 50 lines of glue.
- Telephony comes later, demand-driven, via a Patter integration that reuses the same dispatcher unchanged.

## The runner's role in the stack

For the full layer model and where each concept lives, see [STRATEGY.md](../uxflows/STRATEGY.md). For the runner specifically:

- It interprets the spec live, drives **voice** (Pipecat pipeline) or **text** (WebSocket) I/O depending on `agent.meta.modes` at session start, and emits a single UX4-id-keyed event stream.
- One dispatcher, two I/O adapters — voice and text share the cognitive layer. Mode is selected per session; the same spec can run either way without re-authoring.
- That event stream has multiple subscribers in time order: the editor consumes it today (canvas highlighting); whatsupp2's simulator consumes it next (Phase E in STRATEGY.md); production observability could consume it eventually.
- The runner has **dual identity by design** — prototyping component when invoked from the editor, simulation substrate when invoked by whatsupp2. One executor, two roles, two channels.

Boundaries the runner respects:

- API keys, endpoints, voice IDs → execution config (sibling to spec, never inside).
- Capability backends (HTTP/MCP), knowledge backends (retrieval) → customer-owned. The runner dispatches *to* them; UX4 doesn't build them.
- Scenario asserts, personas, evaluator logic → simulation layer (whatsupp2). Not the runner's concern.

## Goal

Ship a local voice runner that interprets a v0 spec end-to-end, talks via browser mic, and lights up the editor canvas live as the conversation progresses.

**Definition of done** (this is the final v0 — earlier phases land in stages):

- Designer clicks **Run** in the editor toolbar; a status indicator shows the runner connecting.
- Browser asks for mic permission; on grant, audio streams to the runner.
- The first agent turn fires (TTS heard in the browser) using the agent's `entry_flow_id` flow's composed system prompt.
- User speaks; ASR transcribes; the runner evaluates captures, routes via `routing.exit_paths`, transitions flows, and emits events.
- The canvas pulses on the active flow node; edges flash on `exit_path_taken`; recent flows show a fading trail.
- A live variables panel shows captures as they fire; a transcript panel shows agent/user turns.
- Stopping the run cleanly disconnects audio, stops the runner, and clears the runtime overlay.
- The full loop runs against the existing `public/example.json` spec without code changes.
- The standalone runner-served test page (`uxflows-runner/web/`) still works in parallel and is the canonical debug surface — the editor is a *second* consumer of the same event stream, never a replacement for it.

## Architecture

### High-level shape

```
                    Editor (Next.js)                       Runner (Python)
   +-------------------------+------------------+   +-------------------------+
   |  Canvas + inspectors    |  zustand store   |   |  FastAPI server         |
   |  (existing)             |  + runtime slice |   |  - GET  /events  (SSE)  |
   |  ↑ runtimeState         |  ↑ events        |---|  - POST /run            |
   |  ↑ edge pulse           |  ↑ transcript    |   |  - POST /stop           |
   |  ↑ variables panel      |  ↑ variables     |   |  - POST /api/offer      |
   |  ↑ transcript panel     |                  |   |  - POST /api/offer/patch|
   +-------------------------+------------------+   |                         |
                                                    |  Pipecat pipeline       |
   Browser audio glue (~30 LOC, vanilla):           |  ┌─────────────────────┐|
   - getUserMedia mic capture                       |  │ webrtc_in → vad     ││
   - RTCPeerConnection ↔ /api/offer                 |  │ → stt → context_agg ││
   - <audio> element for TTS playback               |  │ → DISPATCHER        ││
                                                    |  │ → llm → tts         ││
                                                    |  │ → webrtc_out        ││
                                                    |  └─────────────────────┘|
                                                    |                         |
                                                    |  Spec interpreter:      |
                                                    |  - flow_state           |
                                                    |  - three-method eval    |
                                                    |  - captures             |
                                                    |  - routing              |
                                                    |  - capabilities         |
                                                    |  - event emitter        |
                                                    +-------------------------+
```

The dispatcher is a `FrameProcessor` inside the Pipecat pipeline. Each user-turn-completed frame triggers it: the dispatcher composes the per-flow system prompt and tool set, runs the LLM call, evaluates captures and exit_paths, transitions flow state if needed, and emits events. The next agent-turn-completed frame runs TTS and gets sent back to the browser.

### Repository layout

The runner is a separate Python repo at `../uxflows-runner/` (sibling to `uxflows/` and `whatsupp2/`). The boundary is clean: the runner imports nothing from uxflows; it consumes a v0 JSON file and writes events.

```
uxflows-runner/
  pyproject.toml                 # uv-managed
  src/uxflows_runner/
    __init__.py
    cli.py                       # `uxflows-runner serve`
    config.py                    # env vars, provider stack selection
    spec/
      loader.py                  # parse v0 JSON, resolve references
      types.py                   # pydantic types mirroring SCHEMA.md v0
    dispatcher/
      processor.py               # Pipecat FrameProcessor — the core
      flow_state.py              # current flow id, history, transitions
      methods.py                 # three-method evaluator: llm / calc / direct
      expressions.py             # calculation expression engine
      captures.py                # extract values from user turns
      routing.py                 # exit_path evaluation, including LLM router tools
      capabilities.py            # HTTP function dispatch; retrieval stub
      prompt_builder.py          # compose per-flow system prompt
    events/
      schema.py                  # pydantic event types — the runner-side contract
      emitter.py                 # SSE broker, fan-out to subscribers
    server/
      app.py                     # FastAPI: /run, /stop, /events, /audio, /health, static /web
      pipeline.py                # Pipecat pipeline construction
  web/                           # standalone test page served by the runner
    index.html                   # spec picker, Connect/Disconnect, transcript
    client.js                    # mic + Pipecat browser client (esm.sh CDN)
    style.css
  examples/
    hello/credentials.json       # GCP service-account JSON (gitignored)
  tests/
    test_methods.py
    test_routing.py
    test_captures.py
```

**Two consumers of the runner, by design:**

1. **Standalone test page** (`web/` above) — runner-served debug surface. Self-contained: drop creds, run the runner, open `http://localhost:8000`, talk to it. No editor needed. Stays as a debug tool forever; doesn't disappear when the editor integration lands.
2. **Editor canvas integration** (Phase 2+) — the editor consumes the *same* event stream the test page does, and renders runtime state on the canvas (active flow ring, edge pulse on routing). Lives in [uxflows](./), under fixed module roots:
   - `lib/runtime/sseClient.ts` — open SSE connection, decode events into the store
   - `lib/runtime/eventTypes.ts` — TypeScript mirror of `events/schema.py`
   - `lib/store/runtime.ts` — zustand slice
   - `pages/run.tsx` *or* a runtime overlay mounted at the editor shell — single mount point per the discipline below
   - Canvas reads runtime state via node/edge `data` props; no direct store imports from `FlowNode` / edges

## Design decisions

### The dispatcher is canonical, audio infra is rented

The dispatcher implements the three-method substrate described in [AGENTS.md](../uxflows/AGENTS.md). Each routing decision, capture, and condition is dispatched through one of:

- **`llm`** — structured-output LLM call (function-tool for routing decisions; JSON-mode extraction for slot-filled captures).
- **`calculation`** — deterministic expression evaluation (variable refs, equality, arithmetic, regex via pattern-matching subtype).
- **`direct`** — literal value or unconditional transition.

The dispatcher composes these per-flow at runtime. Pipecat does *not* orchestrate flow transitions; FlowManager is not used. We use Pipecat's pipeline, services, transport, VAD, aggregators, and frame model — nothing else.

### Spec is read at run start, immutable per session

For v0, the runner loads the spec on `POST /run` and treats it as immutable for the session's duration. Hot-reload-on-edit is a Phase 4 nicety, not v0. This avoids the entire class of "mid-conversation spec mutation" race conditions.

### Events, not RPC, as the editor contract

The runner pushes events via SSE; the editor never queries the runner for state. The editor's runtime store is built up from the event stream alone. This keeps the contract one-directional and replayable — record the event log, replay it later, get an identical canvas illumination experience. Bidirectional control (pause, step, inject) is deferred to a future websocket channel.

### Dispatcher must stay framework-agnostic

The dispatcher's interface is `dispatch(user_text, flow_state, variables) -> (assistant_text, transitions, events)`. Pipecat-specific code is confined to `dispatcher/processor.py` (the `FrameProcessor` wrapper) and `server/pipeline.py`. The core dispatcher logic in `methods.py`, `expressions.py`, `captures.py`, `routing.py`, `capabilities.py`, `prompt_builder.py` knows nothing about Pipecat. This is the deliberate hedge: if we ever want to swap to Patter for production telephony, or reuse the dispatcher inside whatsupp2's text simulator, the swap is bounded to the integration glue.

### Editor-side module boundaries

The runner UI is embedded in the uxflows editor but kept as an independent component code-wise. The discipline:

- **One-way dependency.** Runtime code can read authoring state (it needs the spec), but authoring code never imports from `lib/runtime/*` or `components/runtime/*`. The arrow points one direction.
- **Module homes.** All runtime code lives under fixed roots: `components/runtime/` (RunControl, TranscriptPanel, VariablesPanel, AudioBridge), `lib/store/runtime.ts` (separate zustand slice or store), `lib/runtime/sseClient.ts`, `lib/runtime/audioBridge.ts`, `lib/runtime/eventTypes.ts`. Nothing runtime-related leaks into `lib/schema/`, `lib/store/spec.ts`, `components/inspector/`, or `components/sheets/`.
- **Single mount point.** A `<RuntimeOverlay />` (or similar) renders once at the editor shell when a session is active. Runtime UI is not woven into inspector forms or sheet components.
- **The narrow seam.** The canvas reads runtime state to render highlighting — the only place authoring-side code touches runtime data. It does so through node/edge `data` props (`runtimeState`, `pulse`) computed in `buildGraph` from a single runtime selector. `FlowNode` and edges read those props; they don't import the runtime store directly.
- **Snapshot-on-run.** Runtime code does not mutate the spec or freeze the editor's spec store. The runner takes a spec snapshot at session start and works against the snapshot for the session's duration; the editor's spec stays editable.
- **Provider keys / execution config** belong to the runtime module's settings surface, not the editor's general settings.

What this buys: clean test isolation (authoring tests don't need to mock the runtime), future extractability (the runtime module has a defined boundary if we ever want a separate viewer/replay app), and the kind of mental clarity that prevents authoring code from gaining "Run-mode" branches over time.

What violates it (watch for):
- Inspector forms gaining a "while running" branch.
- Authoring code reading `runtime.connected` to disable controls.
- Runtime code calling editor mutators to "freeze the spec."
- Provider keys ending up in a generic editor `Settings` modal.

### v0 dispatcher feature scope

In v0, the dispatcher implements:

- Flow entry: compose system prompt from `instructions` + flow-level `scripts` (current language) + agent-level guardrails + agent `system_prompt`.
- Per-turn LLM dispatch with provider-specific function-tool schema.
- Captures: `direct` (literal assign), `calculation` (expression eval against current variable bag), `llm` (structured extraction).
- Routing: evaluate `routing.exit_paths` in order; `calculation` short-circuits, fall through to `llm` evaluation as a function-tool decision, `direct` is unconditional.
- Variables: in-memory dict keyed by name; agent-level and flow-level both flat in the same bag for v0.
- Capabilities: `kind: function` → HTTP POST to a configured endpoint with the inputs as JSON body; response merged into variable bag. `kind: retrieval` → stub returning empty context (real retrieval lands later).
- Interrupt flows: out of scope for v0. Re-evaluation pass after each user turn checks `entry_condition` on interrupt-typed flows; deferred unless trivially expressible.
- Guardrails: spec carries them; runtime evaluation is **not** in v0. They flow through to events as metadata only.

### Browser audio glue

A small TS module in the editor establishes an `RTCPeerConnection` and exchanges SDP via `POST /api/offer` (mirroring the standalone `web/` client). Mic capture via `getUserMedia`, TTS playback via an `<audio>` element bound to the inbound track. No frontend audio processing of our own — Pipecat's pipeline handles VAD, format conversion, etc.

### Provider stack default

For v0 development, **Google all-three** — single GCP service-account credential covers the whole stack:

- **STT**: Google Cloud Speech-to-Text (streaming).
- **LLM**: Gemini 2.5 Flash (fast TTFT, ~25× cheaper than gpt-4o for dev iteration; Pro available per-call when reasoning depth matters).
- **TTS**: Google Cloud TTS (Chirp 3 HD voices for low first-byte latency).

Single auth surface (service account JSON) for all three. Pipecat ships `GoogleSTTService` / `GoogleLLMService` / `GoogleTTSService` as first-class. Function-tool schema differs from OpenAI's; Pipecat translates but budget half a day for tool-format quirks (see Risks).

Credentials held in `examples/<spec>/credentials.json` (gitignored, runner repo only — never inside `uxflows/`). Provider abstraction lives in `config.py`; swapping individual services (e.g., Deepgram for STT, Cartesia for TTS) is a one-line change.

## Event schema

The contract between runner and editor. Every event includes `session_id` and `ts` (ISO 8601, runner-local clock). Defined in `events/schema.py` (Python pydantic) and mirrored in `lib/runtime/eventTypes.ts`.

```
session_started     { session_id, agent_id, lang, spec_hash }
session_ended       { session_id, reason: "user_stop" | "agent_terminal" | "error" }
flow_entered        { flow_id }
flow_exited         { flow_id, exit_path_id?, reason: "transition" | "terminal" | "interrupted" }
exit_path_taken     { from_flow_id, exit_path_id, to_flow_id?, method: "llm" | "calculation" | "direct" }
turn_started        { role: "agent" | "user" }
turn_completed      { role, text }
capture_set         { variable_name, value, method, source_flow_id }
capability_invoked  { capability_id, args }
capability_returned { capability_id, result?, error? }
guardrail_metadata  { guardrail_id, statement }     # spec-level only in v0
error               { code, message, recoverable }
```

Events are intentionally flat and small. Anything heavier (full transcripts, large capture payloads) gets truncated server-side at emission; full payloads are inspectable via a separate `/session/:id/log` endpoint (deferred).

## Work chunks

Ordered. Each chunk leaves something demo-able if you stop after it.

### Phase 0 — hello-world bot in browser ✅ shipped 2026-04-30

Stand up the audio loop end-to-end with a hardcoded prompt. No spec involvement yet, no editor changes — runner-served standalone page only.

- Created `uxflows-runner` repo (uv-managed, `src/` layout), FastAPI app.
- Pipecat pipeline: `SmallWebRTCTransport` + SileroVAD → `GoogleSTTService` → context aggregator → `GoogleVertexLLMService` (Gemini 2.5 Flash) → `GoogleTTSService` (Chirp 3 HD).
- `web/index.html` + `web/client.js` — vanilla `RTCPeerConnection` + `getUserMedia`, no Pipecat browser SDK, no protobuf. FastAPI mounts the directory at `/`; one process serves both API and page.
- `.env.example` with `GOOGLE_APPLICATION_CREDENTIALS` + `GOOGLE_CLOUD_PROJECT`; README with quickstart.
- **Acceptance:** ✅ drop creds → `uv run uxflows-runner serve` → open `http://localhost:8000` → Connect → agent greets → free-form conversation works.

**Pivot from the original plan worth recording**: the plan called for `FastAPIWebsocketTransport` + `ProtobufFrameSerializer` + a Pipecat browser client via esm.sh. Switched to `SmallWebRTCTransport` mid-build because vanilla WebRTC in the browser is dramatically simpler — encoding, jitter buffering, AEC, and bidirectional audio all come "for free" from `RTCPeerConnection`. The browser client is ~30 LOC of vanilla JS with no npm/CDN deps. Server-side, `SmallWebRTCRequestHandler` exchanges SDP over a single `POST /api/offer` endpoint. The WebSocket path is still available in Pipecat if telephony or constrained environments demand it later.

### Phase 1 — v0 dispatcher (3–5 days)

Replace the hardcoded prompt with a spec-driven flow interpreter.

- `spec/types.py` — pydantic models mirroring SCHEMA.md v0. Validate against `../uxflows/public/example.json` as the test fixture.
- `spec/loader.py` — load + validate JSON. Resolve `entry_flow_id`. Build a flow lookup table.
- `dispatcher/flow_state.py` — `FlowState` class: current flow id, variable bag, history list, language.
- `dispatcher/prompt_builder.py` — `build_system_prompt(flow, agent, lang) -> str`. Compose `agent.system_prompt + agent guardrails + flow.instructions + flow.scripts[lang] (joined)`.
- `dispatcher/methods.py` — `evaluate(method_spec, vars, llm) -> value`. Branches by method.
- `dispatcher/expressions.py` — minimal expression engine: variable refs (`${var}` or bare), equality, AND/OR, regex match. Lean on `simpleeval` or similar; do not write a parser.
- `dispatcher/captures.py` — given a user turn, run all captures defined on the current flow. Update variable bag. Emit `capture_set` events.
- `dispatcher/routing.py` — given current flow + variables, evaluate `exit_paths` in order. Returns `(next_flow_id | None, exit_path_id, method)`. For `llm` exit paths, batch all exit paths into a single LLM call as a function-tool decision (one tool per exit path).
- `dispatcher/capabilities.py` — `invoke(capability_id, args, agent) -> result`. v0: HTTP POST only.
- `dispatcher/processor.py` — Pipecat `FrameProcessor` subclass. On `LLMFullResponseEndFrame`, run captures + routing; on transition, build new prompt, push it as a `LLMMessagesUpdateFrame`. Emit events to the SSE broker.
- **Acceptance:** Run against `example.json`. The agent's first turn matches the entry flow's `instructions`. Captures fire. Routing transitions to the right next flow. CLI logs show events. Canvas not yet wired.

### Phase 2 — editor wiring + canvas highlight (2–3 days)

The visual payoff phase. By now the runner emits a clean event stream that the standalone test page already consumes. This phase adds the editor as a *second* consumer of that stream and renders runtime state on the canvas.

- `events/schema.py` + `lib/runtime/eventTypes.ts` — define the event types in both languages, kept in sync manually for now (codegen later).
- `events/emitter.py` — SSE broker class. `subscribe(session_id) -> AsyncIterator[Event]`. Fan-out to multiple subscribers — the standalone page, the editor, eventually replay tools all read the same stream.
- `server/app.py` — `GET /events?session_id=...` SSE endpoint; `POST /run` returns `{session_id}` and starts the pipeline; `POST /stop` ends the session. Standalone page already used `/api/offer` for WebRTC; that path persists for both consumers.
- `lib/runtime/sseClient.ts` — `EventSource` wrapper. Decode JSON lines, dispatch to store actions. Reconnect with backoff on disconnect.
- `lib/store/runtime.ts` — zustand slice. Reducer-style: each event type updates the slice. Key fields: `connected`, `sessionId`, `currentFlowId`, `recentFlowIds[]` (last 5 with timestamps for fade), `variables`, `transcript[]`, `error`.
- `components/runtime/RunControl.tsx` — button reading from `runtime.connected`; click → fetch `/run`, then open SSE; second click → fetch `/stop`. Mounted via a single `<RuntimeOverlay />` at the editor shell.
- [components/canvas/FlowNode.tsx](../uxflows/components/canvas/FlowNode.tsx) — read `data.runtimeState`, render border ring (active = pulsing emerald, recent = fading emerald, none = current behavior). React Flow re-renders nodes on `data` change.
- [components/canvas/Canvas.tsx](../uxflows/components/canvas/Canvas.tsx) — on `exit_path_taken` event, set the matching edge's `data.pulse = true` for ~600ms then unset.
- Live variables / transcript appear in a runtime panel (single mount point, not woven into inspector forms).
- **Acceptance:** click Run in editor → canvas lights up the entry flow → talk to the agent → watch nodes change as flows transition. Edges pulse on routing decisions. Variables stream in. Standalone test page still works unchanged.

### Phase 3 — polish (2–4 days)

- Reconnect logic on transient SSE drop; surface persistent failure as a banner.
- Error overlays for runner-side failures (`error` event with `code`/`message`).
- Language picker in the run-control toolbar — defaults to `agent.meta.languages[0]`, sets `lang` on `/run`.
- Variables panel: show method tag (llm / calculation / direct) on each capture so the audit story is visible.
- Transcript panel: timestamps, speaker color-coded.
- Idle timeout: if no audio for 60s, runner ends session with `reason: "idle"`.
- Run a real spec end-to-end (`example.json` or a scratch one); document anything that surprised us in this file.
- **Acceptance:** non-trivial demo recordable; one external person watches and can describe what's happening without help.

## Risks

- **Provider quirks on function-tool schema.** OpenAI vs. Anthropic disagree on tool-call message format; Pipecat's LLM service classes paper over some but not all of this. Budget half a day for fighting it.
- **Browser audio fidelity.** WebSocket-streamed audio in browsers is more fragile than `getUserMedia` straight to a desktop pipeline. Echo, packet loss, sample-rate mismatches. Most issues are solved upstream in Pipecat; some won't be.
- **LLM-routing latency.** When `routing.exit_paths` includes multiple `llm`-method paths on a single flow, they batch into one LLM call. If a flow has many such paths plus `calculation` short-circuits, evaluation order matters for latency. Order them so common cases fast-path.
- **Visual noise on rapid event streams.** A spec with many quick transitions could strobe the canvas. Add a 200ms minimum hold per `runtimeState: "active"` paint.
- **`expressions.py` scope creep.** The calculation expression engine is the easiest place to over-engineer. Stay tiny: variable references, equality, AND/OR, regex. Anything more requires explicit case-by-case justification.
- **Spec validation drift.** Pydantic types mirror SCHEMA.md by hand; they will drift. Mitigate by running editor-exported JSON through the runner's loader as a CI test once the runner has CI.
- **Multi-process state on Run.** v0 assumes one runner process per editor session, single-user. Don't add session-isolation primitives until needed; they're all reversible.

## Decision dependencies

Three things to nail before starting Phase 0:

1. **Provider stack confirmed.** Default proposal: Deepgram + OpenAI + Cartesia. Open to substitutions but pick now and move; switching mid-build is a half-day each.
2. **Runner repo location.** Proposal: `../uxflows-runner/` sibling. Alternative: nested under `whatsupp2/runner/` to share its Python environment and get closer to the simulator. Decision drives import paths.
3. **Local-only vs. tunneled.** v0 is `localhost`-only. No Cloudflare/ngrok tunnel. Demo target is the laptop running both editor and runner. This collapses one whole class of network/permissions issues; revisit if a buyer demo across networks emerges.

## Out of scope for v0

**v0.5 — immediately after v0 ships:**
- **Text chat testing UI.** Sibling to `web/` — a vanilla page that talks to the runner over a WebSocket text channel instead of WebRTC audio. Useful when you want to iterate on a spec's logic without burning STT/TTS minutes, when you're on a flaky mic, or when reviewing flow transitions visually beats listening. The dispatcher is already mode-agnostic (`agent.meta.modes` toggles voice vs. text per session), so this is a new I/O adapter + a new static page, not new cognitive code. Stays as a debug surface alongside the voice page; both consume the same event stream.
- **Runtime guardrail enforcement.** Currently spec-level metadata only. Adding a post-LLM filter that flags or blocks responses violating guardrails is the runtime evidence behind the "decomposition + auditability" pitch.
- **Session event log persistence.** Append the event stream to JSONL on disk. Cheap; unlocks replay and offline analysis at near-zero cost. (The replay UI itself is editor-side and follows when there's demand.)

**Post-v0:**
- **Interrupt flows.** v0 does not re-evaluate `entry_condition` on interrupt-typed flows after each user turn. Voice is interruptive by nature, so this lands soon after v0 — explicit phase, not "when a use case demands it."
- **Phone telephony / Patter integration.** Browser is the v0 demo target. Patter (or Pipecat's Daily transport) swaps in as the audio adapter when a deployment requires PSTN. Dispatcher unchanged.
- **Bidirectional control plane.** Pause / step / inject user input via websocket. Debugging UX, not v0.
- **Multi-session / multi-user runner.** v0 is single-user, single-session, localhost. SaaS shape is a bigger decision than the runner.
- **Reusing the dispatcher in whatsupp2's text simulator.** Strategic — sim/prod parity through one executor — but a separate effort. The framework-agnostic dispatcher boundary makes it possible without rework.

## Open questions

- Should the runner emit `capture_set` events with full values or hashes? Privacy: a real user might mention a SSN. Default: emit values for v0 (single-user local dev), add redaction when telephony lands.
- Where does `execution.json` live? Proposal: alongside the spec, in the runner's working directory. The editor never sees it.
- How does the editor know *which* runner to talk to? v0 assumes `localhost:8000`. Configurable via a setting later.
- Do we need a session id at all in v0? Single-user, single-runner — probably no. Defer multi-session until it matters.
