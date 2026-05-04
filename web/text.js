// Text chat debug page — talk to the runner via /api/chat/* with no audio.
// Vanilla fetch, no bundler, no deps. Sibling to client.js (the voice page).

const $ = (id) => document.getElementById(id);

const els = {
  status: $("status"),
  statusLabel: document.querySelector("#status .label"),
  pickSpec: $("pick-spec"),
  specFile: $("spec-file"),
  specSummary: $("spec-summary"),
  clearSpec: $("clear-spec"),
  apiKey: $("api-key"),
  start: $("start"),
  reset: $("reset"),
  errors: $("errors"),
  transcript: $("transcript"),
  composer: $("composer"),
  send: $("send"),
};

// ---------- state ----------

const state = {
  spec: null,         // parsed JSON spec object
  specName: null,     // display label
  sessionId: null,
  ended: false,
  inflight: false,
};

// ---------- spec picker ----------

els.pickSpec.addEventListener("click", () => els.specFile.click());

els.specFile.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    setSpec(parsed, file.name);
  } catch (err) {
    showError(`Failed to parse ${file.name}: ${err.message}`);
  }
  els.specFile.value = "";
});

els.clearSpec.addEventListener("click", () => {
  setSpec(null, null);
});

function setSpec(spec, name) {
  state.spec = spec;
  state.specName = name;
  if (spec) {
    const flowCount = (spec.flows || []).length;
    const agentId = spec.agent?.id || "(no id)";
    els.specSummary.textContent = `${name} — ${agentId}, ${flowCount} flows`;
    els.specSummary.classList.add("loaded");
    els.clearSpec.hidden = false;
  } else {
    els.specSummary.textContent = "no spec loaded";
    els.specSummary.classList.remove("loaded");
    els.clearSpec.hidden = true;
  }
  refreshButtons();
}

// ---------- start / reset / end ----------

els.start.addEventListener("click", startSession);
els.reset.addEventListener("click", async () => {
  await endSession();
  await startSession();
});

async function startSession() {
  if (!state.spec || state.inflight) return;
  setStatus("starting", "starting…");
  els.errors.hidden = true;
  els.transcript.innerHTML = "";
  state.sessionId = null;
  state.ended = false;
  state.inflight = true;
  refreshButtons();

  const body = {
    spec: state.spec,
    api_key: els.apiKey.value.trim() || undefined,
  };

  try {
    const res = await fetch("/api/chat/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await safeJson(res);
      throw new Error(detail?.detail || `start failed: ${res.status}`);
    }
    const data = await res.json();
    state.sessionId = data.session_id;
    state.ended = data.ended;
    if (data.events?.length) renderEvents(data.events);
    if (data.agent_text) renderBubble("agent", data.agent_text);
    setStatus(state.ended ? "ended" : "ready", state.ended ? "ended" : "ready");
  } catch (err) {
    showError(err.message);
    setStatus("idle", "idle");
  } finally {
    state.inflight = false;
    refreshButtons();
    els.composer.focus();
  }
}

async function endSession() {
  if (!state.sessionId) return;
  const sid = state.sessionId;
  state.sessionId = null;
  state.ended = true;
  setStatus("idle", "idle");
  refreshButtons();
  try {
    await fetch("/api/chat/end", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    });
  } catch {
    /* swallow — best-effort cleanup */
  }
}

// ---------- composer ----------

els.composer.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    sendTurn();
  }
});

els.send.addEventListener("click", sendTurn);

async function sendTurn() {
  const text = els.composer.value.trim();
  if (!text || !state.sessionId || state.ended || state.inflight) return;

  renderBubble("user", text);
  els.composer.value = "";
  state.inflight = true;
  setStatus("thinking", "thinking…");
  refreshButtons();

  try {
    const res = await fetch("/api/chat/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        user_text: text,
      }),
    });
    if (!res.ok) {
      const detail = await safeJson(res);
      throw new Error(detail?.detail || `turn failed: ${res.status}`);
    }
    const data = await res.json();
    if (data.events?.length) renderEvents(data.events);
    if (data.agent_text) renderBubble("agent", data.agent_text);
    state.ended = data.ended;
    setStatus(state.ended ? "ended" : "ready", state.ended ? "ended" : "ready");
  } catch (err) {
    showError(err.message);
    setStatus("ready", "ready");
  } finally {
    state.inflight = false;
    refreshButtons();
    els.composer.focus();
  }
}

// ---------- rendering ----------

function renderBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  els.transcript.appendChild(div);
  scrollToEnd();
}

function renderEvents(events) {
  const wrap = document.createElement("div");
  wrap.className = "events";
  for (const ev of events) {
    const row = document.createElement("div");
    row.className = "event-row";
    row.title = JSON.stringify(ev, null, 2);
    const detail = formatEventDetail(ev);
    row.innerHTML =
      `<span class="event-arrow">→</span>` +
      `<span class="event-name">${escapeHtml(ev.type)}</span>` +
      (detail ? `<span class="event-detail">${escapeHtml(detail)}</span>` : "");
    wrap.appendChild(row);
  }
  els.transcript.appendChild(wrap);
  scrollToEnd();
}

function formatEventDetail(ev) {
  switch (ev.type) {
    case "flow_entered":
      return `${ev.flow_id} (via ${ev.via})`;
    case "flow_exited":
      return `${ev.flow_id} (${ev.reason})`;
    case "exit_path_taken":
      return `${ev.from_flow_id} → ${ev.to_flow_id ?? "end"} via ${ev.exit_path_id} [${ev.method}]`;
    case "interrupt_triggered":
      return `${ev.from_flow_id} → ${ev.interrupt_flow_id} [${ev.method}]`;
    case "variable_set":
      return `${ev.variable_name} = ${truncate(JSON.stringify(ev.value))} [${ev.method}]`;
    case "capability_invoked":
      return `${ev.capability_name}(${truncate(JSON.stringify(ev.args))})`;
    case "capability_returned":
      return `${ev.capability_name} ${ev.error ? "ERROR " + ev.error : "ok"}`;
    case "session_started":
      return `${ev.agent_id} (${ev.lang})`;
    case "session_ended":
      return ev.reason;
    default:
      return "";
  }
}

function scrollToEnd() {
  els.transcript.parentElement.scrollTop = els.transcript.parentElement.scrollHeight;
}

// ---------- helpers ----------

function refreshButtons() {
  const hasSession = !!state.sessionId;
  els.start.disabled = !state.spec || hasSession || state.inflight;
  els.reset.disabled = !state.spec || state.inflight;
  els.composer.disabled = !hasSession || state.ended || state.inflight;
  els.send.disabled = els.composer.disabled;
}

function setStatus(cls, label) {
  els.status.className = `status ${cls}`;
  els.statusLabel.textContent = label;
}

function showError(message) {
  els.errors.textContent = message;
  els.errors.hidden = false;
}

function truncate(s, n = 60) {
  if (typeof s !== "string") s = String(s);
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

async function safeJson(res) {
  try { return await res.json(); } catch { return null; }
}

refreshButtons();
