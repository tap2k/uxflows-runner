# uxflows-runner

Voice/text runner for [UX4 v0 specs](../uxflows/SCHEMA.md). Pipecat-based
dispatcher with self-contained browser test page.

Plan and rationale: [RUNNER-PLAN.md](./RUNNER-PLAN.md).
Strategy: [../uxflows/STRATEGY.md](../uxflows/STRATEGY.md).

## Phase 0 status

Hello-world voice loop — hardcoded system prompt, no spec interpretation yet.
Single process serves both the WebRTC peer endpoint and a vanilla HTML test
page. Phase 1 will swap the prompt for the v0 dispatcher.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) if not already installed.
2. Sync dependencies:
   ```sh
   uv sync
   ```
3. Drop your GCP service-account JSON at `data/credentials.json`.
   The `data/` directory is gitignored. The service account needs:
   - **Vertex AI User** (for Gemini 2.5 Flash)
   - **Cloud Speech Client** (for STT)
   - **Cloud Text-to-Speech User** (for TTS)
4. Copy the env template and fill in your project ID:
   ```sh
   cp .env.example .env
   $EDITOR .env
   ```

## Run

```sh
uv run uxflows-runner serve
```

Then open <http://127.0.0.1:8000> in a Chromium-based browser, click
**Connect**, grant microphone permission, and start talking. The agent
greets you first; ASR/LLM/TTS happen on the runner; audio flows over
WebRTC.

`curl localhost:8000/health` returns `{"ok": true}` once the server has
loaded credentials.

## Layout

```
src/uxflows_runner/
  cli.py                 # `uxflows-runner serve`
  config.py              # env-driven config
  server/
    app.py               # FastAPI: /api/offer (WebRTC), static /web mount
    pipeline.py          # Pipecat pipeline (Silero VAD → Google STT → Gemini → Google TTS)
web/
  index.html             # standalone test page
  client.js              # vanilla RTCPeerConnection + getUserMedia
  style.css
examples/
  coffee.json            # self-contained order-bot spec (Phase 1 fixture)
data/
  credentials.json       # GCP service-account JSON (gitignored)
```

Future phases extend this layout (`spec/`, `dispatcher/`, `events/`) per the
[runner plan](./RUNNER-PLAN.md#repository-layout). The `web/` test page sticks
around as a debug surface even after the editor gains canvas integration.
