# uxflows runner plan

Operational plan for the v0 voice runner — a local Python process that interprets a uxflows v0 spec, drives a real-time voice conversation in the browser, and streams UX4-id-keyed events back to the canvas for live highlighting.

Schema contract in [SCHEMA.md](../uxflows/SCHEMA.md). Codegen reference in [TRANSLATIONS.md](../uxflows/TRANSLATIONS.md). Overall strategy in [STRATEGY.md](../uxflows/STRATEGY.md).

## Why a runner at all

Three options were on the table for connecting the spec to a runtime:

1. **Codegen for each target** (Pipecat, LiveKit, LangGraph, OpenAI Agents SDK) — generate framework-native code. 4× maintenance; schema features that don't survive every target get dropped at the boundary; sim/prod drift becomes structural.
2. **Native runner** — interpret the spec directly. Stable IDs, multilingual scripts, eval metadata, guardrails flow through to runtime without round-tripping. One executor for prototyping and simulation. Schema evolution costs scale linearly, not by target count.
3. **Hybrid** — own the runner; ship codegen as demand-driven delivery formats later.

Choosing option 3 with **runner-first as canonical**. Codegen deferred until paid for. Rationale lives in [STRATEGY.md](../uxflows/STRATEGY.md) (decomposition as substrate, spec-as-contract, leverage-maximizing split).

The structural decision underneath: the runner does **not** own audio infrastructure. It sits inside a Pipecat pipeline as a custom `FrameProcessor` between context aggregation and LLM dispatch. Pipecat handles transport, VAD, STT, TTS, frame plumbing, barge-in, smart endpointing — everything below the cognitive layer. The dispatcher owns flow state, three-method evaluation, routing, exit-path assigns, capability dispatch, event emission. This boundary keeps audio losses near zero while preserving full control over spec interpretation.

A side benefit worth naming: owning the runner contracts v1 schema scope. Several planned v1 fields existed only because v1 was being scoped to make codegen viable — `pipecat` hints, `pre_actions`, `context_strategy`, `respond_immediately`, aggressive typing pressure. With a native runner, these become runner config rather than spec fields, or relax in priority.

## Why Pipecat (vs. Patter, LiveKit, OpenAI Realtime, native audio)

Evaluated alternatives:

- **Patter** — telephony-first SDK with Twilio/Telnyx parity, AMD, voicemail drop, transfer, recording out of the box. Compelling for production phone deployments. Less natural fit for our dispatcher (we'd squat in the `onMessage` "custom LLM" hook rather than slot into a pipeline). Smaller community, less battle-tested. **Conclusion: revisit as a production telephony adapter when a client deployment materializes; not the v0 substrate.**
- **LiveKit Agents** — strong WebRTC story, but instructions-and-tool architecture flattens UX4 flow boundaries into prose. Wrong structural fit for graph-native interpretation.
- **OpenAI Realtime API** — speech-to-speech model bypasses text-level intercept. Our dispatcher requires text dispatch between user and assistant turns; Realtime closes off that seam.
- **Native audio (own VAD/AEC/transports)** — multi-engineer-year effort with zero differentiation. Rejected.

Pipecat wins on architectural fit (frame pipeline matches the dispatcher-as-processor model), browser/local dev story (`SmallWebRTCTransport` exchanges SDP over a single HTTP endpoint, browser side is vanilla `RTCPeerConnection`), provider breadth (Deepgram/Cartesia/ElevenLabs/OpenAI/Anthropic all first-class), community depth, and frame-level event granularity that maximizes future canvas fidelity.

## Why browser-mic for v0

The first demo target is **browser-microphone voice with live canvas highlighting** — not phone telephony. Reasons:

- The visual demo (canvas pulsing live during a real voice conversation) is the strongest differentiator. Browser delivery makes it shareable in a meeting without anyone dialing a phone.
- Skips Twilio/Telnyx account setup, AMD edge cases, telephony codec issues. Phase 0 compresses meaningfully.
- Pipecat's `SmallWebRTCTransport` plus vanilla `RTCPeerConnection` in the browser keeps the glue tiny — encoding, jitter buffering, AEC, and bidirectional audio all come from the WebRTC stack. Browser client is ~30 LOC of plain JS, no npm/CDN deps.
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
- User speaks; ASR transcribes; the runner evaluates `routing.exit_paths`, runs the chosen path's `assigns`, transitions flows, and emits events.
- The canvas pulses on the active flow node; edges flash on `exit_path_taken`; recent flows show a fading trail.
- A live variables panel shows assignments as they fire; a transcript panel shows agent/user turns.
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
                                                    |  - routing              |
                                                    |  - assigns              |
                                                    |  - capabilities         |
                                                    |  - event emitter        |
                                                    +-------------------------+
```

The dispatcher is a `FrameProcessor` inside the Pipecat pipeline. Each user-turn-completed frame triggers it: the dispatcher composes the per-flow system prompt and tool set, runs the LLM call, evaluates exit_paths, fires the chosen path's assigns and actions, transitions flow state if needed, and emits events. The next agent-turn-completed frame runs TTS and gets sent back to the browser.

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
      flow_state.py              # current flow id, turn count, history, transitions
      methods.py                 # three-method evaluator: llm / calc / direct
      expressions.py             # calculation expression engine
      assigns.py                 # evaluate exit_path.assigns when an exit fires
      routing.py                 # exit_path evaluation, including LLM router tools
      capabilities.py            # HTTP function dispatch; retrieval stub
      prompt_builder.py          # compose per-flow system prompt + knowledge
    events/
      schema.py                  # pydantic event types — the runner-side contract
      emitter.py                 # SSE broker, fan-out to subscribers
    server/
      app.py                     # FastAPI: /api/offer, /api/offer/patch, /run, /stop, /events, /health, static /web
      pipeline.py                # Pipecat pipeline construction
  web/                           # standalone test page served by the runner
    index.html                   # spec picker, Connect/Disconnect, transcript
    client.js                    # vanilla RTCPeerConnection — no SDK, no CDN
    style.css
  examples/
    coffee.json                  # self-contained order-bot spec (Phase 1 fixture)
  data/                          # gitignored — credentials and other runtime data
    credentials.json             # GCP service-account JSON
  tests/
    test_methods.py
    test_routing.py
    test_assigns.py
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

The dispatcher implements the three-method substrate described in [AGENTS.md](../uxflows/AGENTS.md). Each routing decision, exit-path assign, and condition is dispatched through one of:

- **`llm`** — structured-output LLM call (function-tool for routing decisions; structured parameters for slot-filled assigns).
- **`calculation`** — deterministic expression evaluation (variable refs, equality, arithmetic, regex via pattern-matching subtype).
- **`direct`** — literal value or unconditional transition.

The dispatcher composes these per-flow at runtime. Pipecat does *not* orchestrate flow transitions; FlowManager is not used. We use Pipecat's pipeline, services, transport, VAD, aggregators, and frame model — nothing else.

### Spec is read at run start, immutable per session

For v0, the runner loads the spec on `POST /run` and treats it as immutable for the session's duration. Hot-reload-on-edit is a Phase 4 nicety, not v0. This avoids the entire class of "mid-conversation spec mutation" race conditions.

### Events, not RPC, as the editor contract

The runner pushes events via SSE; the editor never queries the runner for state. The editor's runtime store is built up from the event stream alone. This keeps the contract one-directional and replayable — record the event log, replay it later, get an identical canvas illumination experience. Bidirectional control (pause, step, inject) is deferred to a future websocket channel.

### Dispatcher must stay framework-agnostic

The dispatcher's interface is `dispatch(user_text, flow_state, variables) -> (assistant_text, transitions, events)`. Pipecat-specific code is confined to `dispatcher/processor.py` (the `FrameProcessor` wrapper) and `server/pipeline.py`. The core dispatcher logic in `methods.py`, `expressions.py`, `assigns.py`, `routing.py`, `capabilities.py`, `prompt_builder.py` knows nothing about Pipecat. This is the deliberate hedge: if we ever want to swap to Patter for production telephony, or reuse the dispatcher inside whatsupp2's text simulator, the swap is bounded to the integration glue.

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

### One LLM call per turn

The dispatcher makes **at most one LLM call per user turn**. Response generation, `llm`-method exit-path decisions, and the `llm`-method assigns scoped to each candidate exit path are bundled into a single call via function-tools — the LLM emits assistant text plus structured side-data (routing decision, assigned values) in one forward pass.

Why: ~3× TTFT improvement and ~3× cost reduction over sequential calls; the response, routing, and assigns all reason from the same context window, which keeps them coherent.

Constraints this rule imposes on the dispatcher:

- **Pre-call short-circuiting.** `direct` and `calculation` methods evaluate locally before the LLM call. Many turns have no remaining `llm`-method work and collapse to a plain text generation.
- **Routing is next-turn.** A routing decision made on turn N takes effect for turn N+1's prompt; turn N's response is always composed from the *current* flow. This matches natural conversation — "Got it, let me check..." → next turn enters the new flow.
- **One tool schema per flow.** `prompt_builder` composes the system prompt *and* the flow-specific tool schema (`take_exit_path` variants, each carrying its own assigns parameters) together. The processor reads tool calls + text from a single `LLMFullResponseEndFrame`.

The plan accommodates an escape hatch: if a flow genuinely needs sequential reasoning (e.g., capability invocation whose result must influence the same turn's response), it can opt into a two-call turn explicitly. This is *not* the default and is not in v0.

### Schema coverage

What v0 schema fields the runner honors, with explicit punts. Keeps Phase 1 from quietly dropping behavior.

**Honored in v0:**

- `agent.system_prompt`, `agent.guardrails`, `flow.guardrails`, `flow.instructions`, `flow.scripts[lang]` → composed into the per-flow system prompt.
- `agent.knowledge.faq` + `flow.knowledge.faq` → embedded into the system prompt as a "Frequently asked" section. Per-language `scripts` selected by current session language.
- `agent.knowledge.glossary` → embedded as a "Terminology" section.
- `agent.chatbot_initiates` → if true, the runner pushes the entry flow's prompt and elicits an immediate agent turn at session start; otherwise waits for the user.
- `agent.entry_flow_id` → first flow at session start.
- `agent.meta.languages[0]` → default session language; overridable via `/run` payload.
- `agent.meta.modes` → selects voice (Pipecat WebRTC) or text (WebSocket) I/O adapter at session start.
- `agent.capabilities[]` → catalog. Dispatch resolves by `name` (snake_case, stable), not `id`. `kind: function` → HTTP POST. `kind: retrieval` → stub returning empty context.
- `flow.routing.exit_paths[].condition` → evaluated in declaration order; `calculation`/`direct` short-circuit; `llm` paths batched into the per-turn LLM call.
- `flow.routing.exit_paths[].assigns` → evaluated when the exit fires, *not* per-turn. `direct`/`calculation` evaluated locally; `llm` assigns bundled into the per-turn call as function-tool parameters scoped to the chosen exit path. Emits `variable_set` events.
- **Variables are assigned only on exit-path firing** (v0 schema rule — per-turn captures are v1). Inside a flow, the LLM has the full conversation history and can compose responses using mid-flow user statements, but those values do not enter the runner's variable bag — and are not available to subsequent flows or capability dispatch — until an exit path that names them in `assigns` actually fires.
- `flow.routing.exit_paths[].actions` → fired post-exit. Inputs resolved implicitly: runtime reads `capabilities[name].inputs` and pulls those variables from scope at fire-time. No explicit input binding syntax. No output binding (fire-and-forget).
- `flow.routing.exit_paths[].type: "exit"` and/or `next_flow_id: null` → session-terminal. Emits `flow_exited{reason: "terminal"}` then `session_ended{reason: "agent_terminal"}`.
- `flow.max_turns` → per-flow turn counter on the active stack frame. On exhaustion, the runtime auto-routes to the flow's unconditional sad exit path (no paired `exit_path_id` field; convention). Interrupted turns do not count.
- `variables` declarations (agent + flow level) → loaded into a typed registry. v0 uses string defaults for `llm`-method extraction tools; typed parameters land in v0.5.
- **Interrupt flows.** `flow.type: "interrupt"`, `flow.scope` (`["global"]` or list of caller flow ids), `routing.entry_condition` (the trigger), and exit type `return_to_caller`. Implemented via a stack-based `FlowState`: triggering an interrupt pushes the active flow, `return_to_caller` pops. Interrupt entry-conditions are precomputed per turn and bundled into the per-turn LLM call as `trigger_interrupt` tool variants alongside `take_exit_path` (still one call). Nested interrupts fall out of the stack model. **Scope matches against the top-of-stack flow only**, not anywhere in the stack — an interrupt scoped to flow X fires only when X is the active flow, not when X is buried under another interrupt. `["global"]` always fires regardless of stack state. Runner ignores `entry_condition` on non-interrupt flows (schema notes it's "rarely needed" there).

**Deferred (not in v0):**

- **Knowledge tables.** `agent.knowledge.tables` is structured lookup data that benefits from real retrieval; embedding all rows blows context. Punted to v0.5 alongside the retrieval upgrade.
- **Typed-parameter LLM extraction.** `variables[name].type` driving function-tool parameter types is a quality lever for `llm` assigns; v0 uses string defaults. Revisit in Phase 3 / v0.5 if extraction reliability is poor.
- **Runtime guardrail enforcement.** `agent.guardrails` + `flow.guardrails` flow into the prompt and `guardrail_metadata` events as spec-level metadata only. Post-LLM filtering lands in v0.5.
- **`flow.example`.** Annotation-only per schema; runtime ignores it.

### v0 dispatcher feature scope

In v0, the dispatcher implements:

- Flow entry: compose system prompt from `agent.system_prompt` + agent guardrails + agent FAQ + agent glossary + flow `instructions` + flow guardrails + flow FAQ + flow `scripts[lang]` (joined).
- Per-turn LLM dispatch with provider-specific function-tool schema, single call (see "One LLM call per turn" above).
- Stack-based `FlowState`: top of stack is the active flow; interrupts push, `return_to_caller` pops.
- Assigns: `direct` (literal), `calculation` (expression eval against variable bag), `llm` (structured extraction). Evaluated when an exit fires, scoped to the chosen exit path. Emits `variable_set` events. Interrupt triggering does *not* fire the interrupted flow's assigns or actions.
- Routing: evaluate `routing.exit_paths` in order; `calculation` short-circuits, fall through to `llm` evaluation as a function-tool decision (bundled into the per-turn call), `direct` is unconditional. Terminal on `type: "exit"` or `next_flow_id: null`.
- Interrupts: per turn, collect interrupt-typed flows whose `scope` matches the active flow (or is `["global"]`). Evaluate `entry_condition`: `calculation`/`direct` short-circuit; `llm` triggers bundled into the per-turn call as `trigger_interrupt` tool variants. LLM picks at most one of `take_exit_path | trigger_interrupt`.
- `chatbot_initiates`: drives whether the agent or user opens the first turn.
- `max_turns`: per-flow turn counter on the active stack frame; interrupted turns do not count; exhaustion auto-routes to the unconditional sad exit path.
- Variables: in-memory dict keyed by name; agent-level and flow-level both flat in the same bag for v0.
- Capabilities: dispatched by `name` (not `id`). `kind: function` → HTTP POST with implicit input resolution from `capabilities[].inputs`. `kind: retrieval` → stub returning empty context.

### Browser audio glue

A small TS module in the editor establishes an `RTCPeerConnection` and exchanges SDP via `POST /api/offer` (mirroring the standalone `web/` client). Mic capture via `getUserMedia`, TTS playback via an `<audio>` element bound to the inbound track. No frontend audio processing of our own — Pipecat's pipeline handles VAD, format conversion, etc.

### Provider stack default

For v0 development, **Google all-three** — single GCP service-account credential covers the whole stack:

- **STT**: Google Cloud Speech-to-Text (streaming).
- **LLM**: Gemini 2.5 Flash (fast TTFT, ~25× cheaper than gpt-4o for dev iteration; Pro available per-call when reasoning depth matters).
- **TTS**: Google Cloud TTS (Chirp 3 HD voices for low first-byte latency).

Single auth surface (service account JSON) for all three. Pipecat ships `GoogleSTTService` / `GoogleLLMService` / `GoogleTTSService` as first-class. Function-tool schema differs from OpenAI's; Pipecat translates but budget half a day for tool-format quirks (see Risks).

Credentials held in `data/credentials.json` (the whole `data/` dir is gitignored, runner repo only — never inside `uxflows/`). Provider abstraction lives in `config.py`; swapping individual services (e.g., Deepgram for STT, Cartesia for TTS) is a one-line change.

## Event schema

The contract between runner and editor. Every event includes `session_id` and `ts` (ISO 8601, runner-local clock). Defined in `events/schema.py` (Python pydantic) and mirrored in `lib/runtime/eventTypes.ts`.

```
session_started     { session_id, agent_id, lang, spec_hash }
session_ended       { session_id, reason: "user_stop" | "agent_terminal" | "error" }
flow_entered        { flow_id, via: "transition" | "interrupt" | "return_to_caller", caller_flow_id? }
flow_exited         { flow_id, exit_path_id?, reason: "transition" | "terminal" | "interrupted" | "returned_to_caller" }
exit_path_taken     { from_flow_id, exit_path_id, to_flow_id?, method: "llm" | "calculation" | "direct" }
interrupt_triggered { from_flow_id, interrupt_flow_id, method: "llm" | "calculation" | "direct" }
turn_started        { role: "agent" | "user" }
turn_completed      { role, text }
variable_set        { variable_name, value, method, source_flow_id, source_exit_path_id }
capability_invoked  { capability_name, args }
capability_returned { capability_name, result?, error? }
guardrail_metadata  { guardrail_id, statement }     # spec-level only in v0
error               { code, message, recoverable }
```

Events are intentionally flat and small. Anything heavier (full transcripts, large variable payloads) gets truncated server-side at emission; full payloads are inspectable via a separate `/session/:id/log` endpoint (deferred).

## Work chunks

Ordered. Each chunk leaves something demo-able if you stop after it.

### Phase 0 — hello-world bot in browser ✅ shipped 2026-04-30

Stand up the audio loop end-to-end with a hardcoded prompt. No spec involvement yet, no editor changes — runner-served standalone page only.

- Created `uxflows-runner` repo (uv-managed, `src/` layout), FastAPI app.
- Pipecat pipeline: `SmallWebRTCTransport` + SileroVAD → `GoogleSTTService` → context aggregator → `GoogleVertexLLMService` (Gemini 2.5 Flash) → `GoogleTTSService` (Chirp 3 HD).
- `web/index.html` + `web/client.js` — vanilla `RTCPeerConnection` + `getUserMedia`, no Pipecat browser SDK, no protobuf. FastAPI mounts the directory at `/`; one process serves both API and page.
- `.env.example` with `GOOGLE_APPLICATION_CREDENTIALS` + `GOOGLE_CLOUD_PROJECT`; README with quickstart.
- **Acceptance:** ✅ drop creds → `uv run uxflows-runner serve` → open `http://localhost:8000` → Connect → agent greets → free-form conversation works.

**Audio transport**: `SmallWebRTCTransport` server-side, vanilla `RTCPeerConnection` in the browser. SDP exchange happens over a single `POST /api/offer` endpoint (plus `POST /api/offer/patch` for ICE). The browser stack handles encoding, jitter buffering, AEC, and bidirectional audio for free; no Pipecat browser SDK, no protobuf serializer, no npm/CDN deps. Pipecat's WebSocket transport remains available if telephony or constrained environments need it later.

### Phase 1 — v0 dispatcher (4–6 days)

Replace the hardcoded prompt with a spec-driven flow interpreter.

- `spec/types.py` — pydantic models mirroring SCHEMA.md v0. Validate against `../uxflows/public/example.json` as the test fixture.
- `spec/loader.py` — load + validate JSON. Resolve `entry_flow_id`. Build flow + capability-by-name lookup tables. Build interrupt-flows-by-scope index for fast per-turn lookup.
- `dispatcher/flow_state.py` — `FlowState` class with a flow stack (each frame: flow id, variable bag inheritance, turn counter, language). Top of stack is active. Helpers for push (interrupt trigger), pop (`return_to_caller`), peek.
- `dispatcher/prompt_builder.py` — `build_system_prompt(flow, agent, lang) -> str`. Compose `agent.system_prompt` + agent guardrails + agent FAQ (per-language scripts when present) + agent glossary + flow `instructions` + flow guardrails + flow FAQ + flow `scripts[lang]` (joined). Also computes the per-turn tool schema: `take_exit_path` variants + `trigger_interrupt` variants for applicable interrupts (matched by `scope`).
- `dispatcher/methods.py` — `evaluate(method_spec, vars, llm) -> value`. Branches by method.
- `dispatcher/expressions.py` — minimal expression engine: variable refs (bare names), equality, comparison, `and`/`or`/`not`, regex (via the `pattern` field on `calculation`-method conditions). Lean on `simpleeval` or similar; do not write a parser.
- `dispatcher/assigns.py` — when an exit path fires, evaluate its `assigns` block: `direct`/`calculation` evaluated locally, `llm` reads pre-bundled values from the per-turn LLM call's tool result. Update variable bag. Emit `variable_set` events.
- `dispatcher/routing.py` — given active flow + variables, evaluate `exit_paths` in order. Also evaluates applicable interrupts' `entry_condition`s; `calculation`/`direct` short-circuit, `llm` becomes `trigger_interrupt` tool variants in the per-turn LLM call. Returns one of: `(stay)`, `(take_exit, exit_path_id)`, `(trigger_interrupt, interrupt_flow_id, caller_flow_id)`, or `(return_to_caller, popped_flow_id)`. Terminal on `type: "exit"` or `next_flow_id: null`.
- `dispatcher/capabilities.py` — `invoke(capability_name, agent) -> result`. Resolves by `name` from the catalog. Implicit input resolution: reads `capabilities[name].inputs` and pulls values from the variable bag at fire-time. v0: HTTP POST only; `kind: retrieval` returns empty context.
- `dispatcher/processor.py` — Pipecat `FrameProcessor` subclass. At session start, honor `agent.chatbot_initiates` (push entry flow prompt + elicit immediate turn, or wait for user). On `LLMFullResponseEndFrame`:
  - **Stay** → no transition, increment active-frame turn counter
  - **Take exit** → fire chosen exit path's assigns and actions, transition (push new flow or pop on terminal)
  - **Trigger interrupt** → push interrupt flow, do *not* fire interrupted flow's assigns/actions, do *not* increment caller's turn counter
  - **Return to caller** → pop, rebuild prompt from caller's flow

  On transition, push `LLMMessagesUpdateFrame`. On caller-flow `max_turns` exhaustion (counter only ticks for non-interrupted turns), force the unconditional sad exit.
- `tests/test_interrupts.py` — global-scope, scoped, nested, return-to-caller, max_turns counter behavior.
- **Acceptance:** Run against `example.json`. The agent's first turn matches the entry flow's `instructions` (or the user opens, per `chatbot_initiates`). Routing transitions to the right next flow. Exit-path assigns fire and emit `variable_set` events. Post-exit actions fire. Terminal exits end the session cleanly. `max_turns` exhaustion auto-routes (interrupted turns don't count). Saying an interrupt trigger phrase pushes the interrupt flow, returns to the caller on `return_to_caller`. CLI logs show events. Canvas not yet wired.

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
- Variables panel: show method tag (llm / calculation / direct) on each assignment so the audit story is visible.
- Transcript panel: timestamps, speaker color-coded.
- Idle timeout: if no audio for 60s, runner ends session with `reason: "idle"`.
- Run a real spec end-to-end (`example.json` or a scratch one); document anything that surprised us in this file.
- **Acceptance:** non-trivial demo recordable; one external person watches and can describe what's happening without help.

## Risks

- **Provider quirks on function-tool schema.** OpenAI vs. Anthropic disagree on tool-call message format; Pipecat's LLM service classes paper over some but not all of this. Budget half a day for fighting it.
- **Browser audio fidelity.** WebRTC handles encoding, jitter buffering, and AEC for us, but real-world variance in mic hardware, OS audio routing, and network conditions still bites. Most issues are solved upstream in Pipecat / the browser stack; some won't be.
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
- **Phone telephony / Patter integration.** Browser is the v0 demo target. Patter (or Pipecat's Daily transport) swaps in as the audio adapter when a deployment requires PSTN. Dispatcher unchanged.
- **Bidirectional control plane.** Pause / step / inject user input via websocket. Debugging UX, not v0.
- **Multi-session / multi-user runner.** v0 is single-user, single-session, localhost. SaaS shape is a bigger decision than the runner.
- **Reusing the dispatcher in whatsupp2's text simulator.** Strategic — sim/prod parity through one executor — but a separate effort. The framework-agnostic dispatcher boundary makes it possible without rework.

## Open questions

- Should the runner emit `variable_set` events with full values or hashes? Privacy: a real user might mention a SSN. Default: emit values for v0 (single-user local dev), add redaction when telephony lands.
- Where does `execution.json` live? Proposal: alongside the spec, in the runner's working directory. The editor never sees it.
- How does the editor know *which* runner to talk to? v0 assumes `localhost:8000`. Configurable via a setting later.
- Do we need a session id at all in v0? Single-user, single-runner — probably no. Defer multi-session until it matters.
